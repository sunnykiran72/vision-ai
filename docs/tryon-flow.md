# Try-on Flow

## Purpose

This document describes the current `/v1/tryon` implementation in the Python AI service.

It covers:
- request contract
- JWT/user identity flow
- garment reference processing
- AI-Toolkit runtime execution
- Azure upload
- request cleanup

This is the implemented flow today, not a speculative future design.

## Entry point

Route:
- `app/routes/tryon.py`

Endpoint:
- `POST /v1/tryon`

Request body:
- `user_image`
- `products[]`
  - `image_url`
  - `type`
  - `prompt`
- optional:
  - `seed`
  - `steps`
  - `guidance_scale`

Response envelope:
- `status`
- `message`
- `data.url`
- `data.metadata`

## Authentication and user identity

JWT verification happens in middleware:
- `app/middleware/auth.py`

The middleware:
1. reads `Authorization: Bearer <token>`
2. verifies the JWT using `JWT_ACCESS_SECRET`
3. validates the decoded payload against `AuthPayload`
4. stores the parsed payload on `request.state.auth_payload`

Protected routes do not parse JWT themselves. They read the already-validated payload through:
- `app/dependencies/auth.py`

For `/v1/tryon`, the route extracts:
- `current_user.user_id`

This `user_id` is then passed into the service explicitly.

### Multi-user isolation rule

The try-on request body does not contain a user identifier.

The only user identity used by the try-on service is:
- the `user_id` derived from the verified JWT middleware payload

That `user_id` is used in the storage path:
- `wardrobe_output/tryon/<user_id>/<job_id>/output.jpg`

This prevents one request from choosing another user's output path through request data.

## Route flow

File:
- `app/routes/tryon.py`

The route is intentionally thin.

It does three things:
1. accepts the validated `TryonRequest`
2. receives the validated auth payload from the dependency
3. dispatches the blocking workflow into `run_in_threadpool(...)`

The route does not:
- build collages
- load models
- upload files
- manipulate storage paths

## Service flow

File:
- `app/services/tryon.py`

Main entrypoint:
- `run_tryon_request(...)`

This is the workflow owner for try-on.

### Step 1: resolve request defaults

If the request does not include:
- `seed`
- `steps`
- `guidance_scale`

the service uses try-on defaults from `app/config.py`.

### Step 2: download the person image

Shared helper:
- `app/utils/media_utils.py`

The service downloads `user_image`, opens it with PIL, and converts it to RGB.

### Step 3: download garment images

For each product in `products[]`, the service:
1. downloads the image
2. opens it with PIL
3. converts it to RGB
4. wraps it into a `ProductReferenceInput`

These inputs are then passed into the garment-reference builder.

## Garment reference processing

File:
- `app/utils/tryon_collage.py`

Main entrypoint:
- `build_product_reference(products)`

The model never receives a raw list of garments.

Instead, the garment list is collapsed into one garment reference image.

### Case 1: single garment

If `len(products) == 1`:
- the single garment image is used directly
- no collage is created

Mode:
- `single_product`

### Case 2: exactly one top and one bottom

If the selected garment types are exactly:
- `top`
- `bottom`

the service builds a special vertical collage.

Mode:
- `top_bottom_vertical_collage`

This logic was adapted from the earlier pilot collage flow and is kept isolated in `tryon_collage.py`.

### Case 3: other multi-garment combinations

For other multi-garment selections:
- garments are resized to a common height
- they are arranged horizontally
- one combined garment reference image is produced

Mode:
- `multi_product_horizontal_collage`

## Prompt construction

Prompt construction lives in:
- `app/services/tryon.py`

The prompt is built from explicit constant sections:
- `TRYON_SINGLE_REFERENCE_PROMPT`
- `TRYON_MULTI_REFERENCE_PROMPT`
- `TRYON_TOP_SECTION_TEMPLATE`
- `TRYON_BOTTOM_SECTION_TEMPLATE`
- `TRYON_DRESS_SECTION_TEMPLATE`
- `TRYON_OUTER_SECTION_TEMPLATE`
- `TRYON_GENERIC_SECTION_TEMPLATE`
- `TRYON_IDENTITY_CLAUSE`

The service does not accept a free-form user prompt today.

Instead, it builds the final prompt from the structured product list.

### Ordering rules

Products are ordered by type priority:
1. top / outer
2. dress
3. bottom

Within the same priority, original request order is preserved.

### Prompt examples

Single garment:
- `Apply the reference garment from image 2 to the person in image 1. Top: red structured jacket. Preserve the person's face, identity, body proportions, pose, and background.`

Multiple garments:
- `Apply the reference garments from image 2 to the person in image 1. Top: red structured jacket. Bottom: black straight trousers. Preserve the person's face, identity, body proportions, pose, and background.`

## Request-local workspace

Shared helper:
- `app/utils/media_utils.py`

Try-on uses a dedicated request-local workspace builder:
- `build_tryon_job_media_paths(...)`

Per request it creates:
- `job_id`
- `job_dir`
- `person.jpg`
- `garment_reference.jpg`
- `output.jpg`

These files are isolated per request to prevent path collisions across concurrent users.

## Model input contract

Before model execution, the service writes:
- `person.jpg`
- `garment_reference.jpg`

The AI-Toolkit runtime consumes those two files separately.

This is the key try-on input contract:
- `ctrl_img_1 = person`
- `ctrl_img_2 = garment_reference`

The user image and garment reference are not combined into one board before runtime execution.

Only the garment side uses collage logic when needed.

## Runtime layer

Files:
- `app/runtime/tryon_runtime.py`
- `app/clients/qwen_tryon_aitk.py`

The route and service do not talk to AI-Toolkit directly.

They go through:
1. a bounded execution coordinator
2. a resident try-on runner

### Coordinator responsibility

The coordinator:
- bounds queue size
- bounds queue wait time
- serializes access to the resident try-on runtime

This is the concurrency control boundary for the in-process try-on implementation.

### Runner responsibility

The runner:
- loads the AI-Toolkit environment
- loads the Qwen Image Edit runtime
- loads the try-on LoRA checkpoint
- executes generation

## AI-Toolkit implementation

File:
- `app/clients/qwen_tryon_aitk.py`

The try-on client follows the AI-Toolkit reference path closely.

It does the following:
1. changes cwd into `AI_TOOLKIT_ROOT`
2. adds `AI_TOOLKIT_ROOT` to `sys.path`
3. loads the Qwen model with `arch="qwen_image_edit_plus"`
4. creates `LoRASpecialNetwork`
5. loads the checkpoint from `TRYON_LORA_PATH`
6. builds `GenerateImageConfig`
7. calls `generate_images(..., sampler="flowmatch")`

### Runtime control-image order

The runtime uses:
- `ctrl_img_1 = person`
- `ctrl_img_2 = garment_reference`

This order is intentional and matches the parity-oriented try-on path.

## Azure upload

File:
- `app/clients/storage.py`

After generation succeeds, the service uploads `output.jpg` to Azure.

The storage object name is server-generated and includes:
- try-on prefix
- authenticated `user_id`
- request `job_id`

Current shape:
- `wardrobe_output/tryon/<user_id>/<job_id>/output.jpg`

The request body does not control this path.

## Cleanup

The request-local directory is always removed in `finally`.

That means:
- person image is deleted
- garment reference image is deleted
- local output file is deleted

Only the Azure-uploaded result remains after completion.

## Error handling

The service maps errors into the standard API envelope.

Current important cases:
- invalid/missing auth: `401`
- queue full: `503`
- queue wait timeout: `504`
- download failure: `400`
- invalid image content: `400`
- no generated image: `400`
- runtime initialization/execution failure: `500`
- storage missing: `500`

## Startup behavior

Files:
- `app/config.py`
- `app/main.py`
- `app/runtime/warmup.py`

Startup validation runs before serving requests.

For try-on, required runtime configuration is validated up front, including:
- `AI_TOOLKIT_ROOT`
- `QWEN_IMAGE_EDIT_MODEL_PATH`
- `TRYON_LORA_PATH`

If warmup is enabled, the try-on runtime also loads the checkpoint during startup.

## Current limitations

This is still an in-process resident runner.

That means:
- requests are isolated by temp directory and storage path
- the API can safely handle many incoming requests
- actual model execution is still bounded and coordinated internally

The current implementation does not try to preempt a running generation mid-flight.

## Summary

The current try-on implementation is:
- JWT-authenticated
- user-id scoped through middleware-derived identity
- request-isolated on disk
- AI-Toolkit-based for model execution
- garment-collage aware
- Azure-uploading
- cleanup-enforced

The most important behavior to remember is:
- the person image is passed separately
- the garments are collapsed into one garment reference image
- only the JWT-derived `user_id` is used for output ownership

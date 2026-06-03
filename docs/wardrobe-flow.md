# Wardrobe API Flow

## Purpose

This document describes the implemented `/v1/wardrobe` flow in the Python AI service.

It covers:
- JSON request and response contract
- authentication and user isolation
- image validation and preprocessing
- garment verification
- AI-Toolkit extraction runtime
- Marqo category classification
- Azure storage
- background Glamify progress sync
- all major success and failure cases

This is the current implementation, not the old multipart `/analyze` flow.

## Entry Point

Route:
- `app/routes/wardrobe.py`

Endpoint:
- `POST /v1/wardrobe`

Request body:

```json
{
  "image": "<raw-base64-or-data-url>",
  "type": "top"
}
```

Allowed `type` values:
- `top`
- `bottom`
- `dress`

Success response:

```json
{
  "status": 200,
  "message": "",
  "data": {
    "id": "89bd1674-0296-4652-8505-092d677b93a5",
    "type": "bottom",
    "image": "<raw-jpeg-base64>",
    "category": "track_pants",
    "categoryLabel": "Track pants"
  }
}
```

Error response shape:

```json
{
  "status": 400,
  "message": "No garment was detected in the image.",
  "data": null
}
```

The API is JSON-only. It does not accept multipart `file`, multipart `image`, `debug`, or other legacy `/analyze` fields.

## Authentication And User Identity

JWT verification happens before the route handler in:
- `app/middleware/auth.py`

The middleware:
1. reads `Authorization: Bearer <token>`
2. verifies the token with `JWT_ACCESS_SECRET`
3. validates the decoded payload as `AuthPayload`
4. stores the auth payload on `request.state.auth_payload`

The route receives the already-validated auth payload through:
- `app/dependencies/auth.py`

The route passes two auth-related values into the service:
- `user_id` from the verified token
- the original `Authorization` header for background Glamify progress sync

The request body cannot choose a user id.

## Route Flow

The route is intentionally thin.

It does:
1. validates JSON into `WardrobeAnalyzeRequest`
2. resolves `CurrentAuth`
3. calls `run_wardrobe_request(...)` in a threadpool
4. sets the HTTP status code to match `response.status`

The route does not:
- decode base64
- resize images
- run model inference
- upload to Azure
- call Glamify backend

## Constants

Wardrobe constants live in:
- `app/constants/wardrobe.py`

Current values:

| Concern | Value |
| --- | --- |
| Input formats | PNG, JPEG |
| Minimum accepted width and height | `350px` |
| Preprocess max edge | `1024px` |
| Garment detector | `yainage90/fashion-object-detection` |
| Detector threshold | `0.25` |
| Marqo classifier | `Marqo/marqo-fashionSigLIP` |
| Marqo confidence threshold | `0.25` |
| Marqo top-k metadata | `5` |
| LoRA rank | `16` |
| LoRA alpha | `16` |
| Queue wait timeout | `30s` |
| Glamify progress timeout | `20s` |
| Output size | `832x1248` |
| Seed | `43` |
| Steps | `15` |
| Guidance scale | `1.0` |
| Guidance rescale | `0.0` |
| Network multiplier | `1.0` |
| Sampler | `flowmatch` |
| CFG norm | `false` |

Static prompts:

| Type | Prompt |
| --- | --- |
| `top` | `GlamTopExt. Extract top wear as a standalone product.` |
| `bottom` | `GlamBtmExt. Extract bottom wear as a standalone product.` |
| `dress` | `GlamDressExt. Extract dress as a standalone product.` |

These values are code constants, not environment variables. Environment-specific endpoints and file paths live in runtime configuration.

## Runtime Configuration

Deployment-specific paths and queue settings live in:
- `app/config.py`

Important environment variables:

| Variable | Purpose |
| --- | --- |
| `AI_TOOLKIT_ROOT` | AI-Toolkit checkout path |
| `QWEN_IMAGE_EDIT_MODEL_PATH` | Qwen Image Edit model path |
| `WARDROBE_LORA_TOP_PATH` | Top extraction LoRA checkpoint |
| `WARDROBE_LORA_BOTTOM_PATH` | Bottom extraction LoRA checkpoint |
| `WARDROBE_LORA_DRESS_PATH` | Dress extraction LoRA checkpoint |
| `WARDROBE_QUEUE_MAX_SIZE` | GPU queue size, default `8` |
| `WARDROBE_WORK_ROOT` | Request-local temp directory root |
| `WARDROBE_STORAGE_PREFIX` | Azure blob prefix |
| `GLAMIFY_API_BASE_URL` | Environment-specific Glamify backend base URL |

Startup validation requires the Qwen model path, AI-Toolkit root, Azure settings, wardrobe LoRA paths, and `GLAMIFY_API_BASE_URL`.

## Service Flow

Main service:
- `app/services/wardrobe.py`

Entrypoint:
- `run_wardrobe_request(...)`

### Step 1: Resolve Type And Prompt

The request `type` is a Pydantic enum.

The service maps it to one static prompt:
- `top` -> top extraction prompt
- `bottom` -> bottom extraction prompt
- `dress` -> dress extraction prompt

No free-form prompt is accepted from the user.

### Step 2: Decode Base64 Image

The service accepts either:
- raw base64
- data URL base64, such as `data:image/png;base64,...`

Validation rules:
- input must decode as base64
- decoded bytes must open as an image through Pillow
- image format must be PNG or JPEG
- image is converted to RGB

If this fails, response is `422`.

### Step 3: Validate Dimensions

The service checks the decoded image dimensions.

The image is accepted only when both dimensions are `>= 350px`.

Accepted examples:
- `350x350`
- `512x512`

Rejected examples:
- `100x120`
- `100x350`
- `350x100`
- `349x512`

If either dimension is below `350px`, response is `422`.

### Step 4: Preprocess With LANCZOS

The service resizes large images before detection and extraction.

Rules:
- if longest edge is `<= 1024`, keep the image size
- if longest edge is `> 1024`, scale the image down so longest edge is `1024`
- preserve aspect ratio
- use Pillow `Image.Resampling.LANCZOS`

Examples:

| Input | Preprocessed |
| --- | --- |
| `512x768` | `512x768` |
| `1600x1200` | `1024x768` |
| `1200x1600` | `768x1024` |

The preprocessed image is saved as request-local `input.jpg`.

## Garment Verification

Client:
- `app/clients/fashion_detection.py`

Model:
- `yainage90/fashion-object-detection`

The detector is used only as a fast garment-presence gate.

Important behavior:
- it does not choose the final extraction type
- it does not need to match the requested `type`
- any detection above threshold is enough to continue

Example:
- request type is `dress`
- detector finds a `top`
- request still continues, because detector is only checking that the input looks like a garment image

If no garment is detected, response is:

```json
{
  "status": 400,
  "message": "No garment was detected in the image.",
  "data": null
}
```

## Request-Local Workspace

Every request gets a generated UUID job id.

Workspace shape:

```text
<WARDROBE_WORK_ROOT>/<job_id>/
  input.jpg
  output.jpg
```

The request body never controls:
- job id
- directory name
- storage path
- output filename

The service deletes the request-local directory in `finally`, for both success and failure after the directory exists.

## Runtime Execution

Runtime files:
- `app/runtime/wardrobe_runtime.py`
- `app/clients/qwen_wardrobe_aitk.py`
- `app/runtime/coordinator.py`

The wardrobe runtime uses AI-Toolkit directly.

Model architecture:
- `qwen_image_edit`

This is intentional. Wardrobe extraction has one control image:
- `ctrl_img = garment_input`

It does not use `qwen_image_edit_plus`, which is reserved for two-control flows such as try-on.

Generation config:

```python
GenerateImageConfig(
    prompt=<static category prompt>,
    width=832,
    height=1248,
    negative_prompt="",
    seed=43,
    guidance_scale=1.0,
    guidance_rescale=0.0,
    num_inference_steps=15,
    network_multiplier=1.0,
    output_path=<job_dir>/output.jpg,
    output_ext="jpg",
    ctrl_img=<job_dir>/input.jpg,
    do_cfg_norm=False,
)
```

Generation runs through:

```python
pipeline.generate_images([config], sampler="flowmatch")
```

## LoRA Loading And Switching

The runtime loads Qwen once and keeps it resident.

It then prepares one LoRA network and caches three state dicts:
- `top`
- `bottom`
- `dress`

On request:
1. resolve requested type
2. load the cached state dict into the active network
3. call `_update_torch_multiplier()`
4. run generation

This avoids loading the full Qwen model or reading checkpoint files on every request.

If the requested LoRA checkpoint is missing or the AI-Toolkit runtime cannot load, response is `500`.

## Queue And Concurrency

Wardrobe uses the shared bounded execution coordinator:
- `app/runtime/coordinator.py`

Current behavior:
- queue size is controlled by `WARDROBE_QUEUE_MAX_SIZE`
- queue wait timeout is the constant `QUEUE_WAIT_TIMEOUT_SECONDS`
- detector, Qwen generation, and Marqo classification run behind the same coordinator

This prevents GPU-heavy wardrobe work from overlapping in one process and prevents multiple wardrobe requests from mutating the active LoRA network at the same time.

Current error mapping:
- queue full -> `503`
- queue wait timeout -> `503`

The public response still follows the required envelope with `data: null`.

## Output Handling

After generation:
1. runtime returns a PIL image and metadata
2. service converts the image to RGB
3. service saves `output.jpg` in the job directory
4. service later returns the image as raw JPEG base64

Returned `data.image` is raw base64, not a data URL.

Consumers should treat it as JPEG bytes:

```text
base64_decode(data.image) -> image/jpeg
```

## Marqo Category Classification

Client:
- `app/clients/marqo_fashion.py`

Model:
- `Marqo/marqo-fashionSigLIP`

Marqo runs after generation, not before generation.

Purpose:
- classify the generated standalone garment into a wardrobe category key
- verify the output is confident enough to save

The candidate list depends on requested `type`.

Top candidates include:
- `tops`: `t_shirts`, `long_sleeve_t_shirts`, `polo_shirts`, `crop_tops`, `blouses`, `shirts`, `sweatshirts`, `hoodies`, `sweaters`, `sweater_vests`, `bodysuits`, `knitwear`, `corsets`, `tunics`, `bustiers`, `sleeveless_tops`, `tank_tops_and_camis`
- `outerwear`: `coats`, `trench_coats`, `blazers`, `jackets`, `varsity_jackets`, `biker_jackets`, `cardigans`, `parkas`, `down_jackets`, `puffer_jackets`, `capes`, `ponchos`, `leather_jackets`, `bomber_jackets`, `denim_jackets`, `windbreakers`
- `layering_pieces`: `vests`, `shawls`, `shrugs`, `boleros`
- `office_wear_formal`: `suits`, `knit_tops`, `work_dresses`, `structured_dresses`

Bottom candidates include:
- `bottoms`: `pants`, `trousers`, `dress_pants`, `track_pants`, `leggings`, `sweatpants`, `shorts`, `jeans`, `palazzos`, `jeggings`, `skorts`, `cargo_pants`, `wide_leg_pants`
- `skirts`: `mini_skirts`, `midi_skirts`, `maxi_skirts`, `a_line_skirts`, `pencil_skirts`, `pleated_skirts`, `wrap_skirts`, `denim_skirts`
- `activewear_sportswear`: `sports_tops`, `sports_jackets`, `gym_leggings`, `bike_shorts`, `tracksuits`, `tennis_skirts`

Dress candidates include:
- `dresses`: `day_dresses`, `t_shirt_dresses`, `shirt_dresses`, `sweater_dresses`, `jacket_dresses`, `party_dresses`, `mini_dresses`, `maxi_dresses`, `slip_dresses`, `bodycon_dresses`, `casual_dresses`, `evening_dresses`, `midi_dresses`, `strapless_dresses`, `off_shoulder_dresses`, `wrap_dresses`
- `loungewear`: `lounge_sets`, `lounge_pants`, `lounge_tops`, `oversized_hoodies`, `soft_knit_sets`
- `sets_one_pieces`: `jumpsuits`, `rompers`, `playsuits`, `two_piece_sets`, `matching_sets`, `co_ords`

These candidate groups come from the live backend category lookup and the older analyze Marqo lane defaults:
- old analyze `top`: `tops`, `layering_pieces`
- old analyze `outer`: `outerwear`, `office_wear_formal`; folded into wardrobe `top`
- old analyze `bottom`: `bottoms`, `skirts`, `activewear_sportswear`
- old analyze `dress`: `dresses`, `loungewear`, `sets_one_pieces`

Marqo scores labels with image/text similarity and softmax probability.

The best category is accepted only if:
- score is `>= 0.25`
- category key is non-empty

If Marqo is below threshold, response is:

```json
{
  "status": 400,
  "message": "Generated garment category is uncertain.",
  "data": null
}
```

There is no fallback to the requested type. The returned `category` is always a Marqo category key.

## Azure Storage

Client:
- `app/clients/storage.py`

The API response returns base64. Azure upload still happens because the Glamify backend progress endpoint expects URLs, but upload is now background-only.

Storage path pattern:

```text
<WARDROBE_STORAGE_PREFIX>/<user_id>/<job_id>/input.jpg
<WARDROBE_STORAGE_PREFIX>/<user_id>/<job_id>/output.jpg
```

Default prefix:

```text
wardrobe_output/wardrobe
```

Content type:

```text
image/jpeg
```

If Azure storage is missing or upload fails, the error is logged from the background worker. The already completed API response is not changed.

## Background Glamify Progress Sync

Client:
- `app/clients/glamify_progress.py`

The sync target is the `GLAMIFY_API_BASE_URL` setting in `app/config.py`.

Endpoint:

```text
POST <GLAMIFY_API_BASE_URL>/wardrobe/progress
```

Headers:

```text
Authorization: <original incoming Authorization header>
Content-Type: application/json
```

Payload:

```json
{
  "id": "<job_id>",
  "inputImage": "<azure_input_url>",
  "outputImage": "<azure_output_url>",
  "metadata": {
    "classification": {
      "primary_category": "bottoms",
      "category": "track_pants",
      "category_label": "Track pants",
      "score": 0.88
    },
    "marqo": {
      "model": "Marqo/marqo-fashionSigLIP",
      "threshold": 0.25,
      "top_matches": []
    },
    "prompt": "GlamBtmExt. Extract bottom wear as a standalone product.",
    "requested_type": "bottom"
  }
}
```

This sync is intentionally background-only.

Important behavior:
- input upload starts in the background after preprocessing and before GPU inference
- output upload starts in the background after generation and Marqo classification
- successful `/v1/wardrobe` response does not wait for Azure or the Glamify backend
- sync failure is logged
- sync failure does not change the already-returned API response

## Success Case

Happy path:

1. Auth middleware validates JWT.
2. Route validates JSON body.
3. Service resolves `user_id` from JWT.
4. Base64 image is decoded.
5. PNG/JPEG format is verified.
6. Minimum dimensions are verified.
7. Image is resized to max edge `1024` if needed.
8. Wardrobe coordinator grants GPU execution slots for model-backed work.
9. Fashion detector finds at least one garment.
10. Job directory is created.
11. Preprocessed input is saved as `input.jpg`.
12. Input upload to Azure is queued in the background.
13. Qwen wardrobe runner switches to the requested LoRA.
14. AI-Toolkit generates fixed-size `832x1248` `output.jpg`.
15. Marqo classifies generated output.
16. Marqo best category clears threshold.
17. Output upload and Glamify progress sync are queued in the background.
18. Job directory is deleted.
19. API returns `200` with job id, JPEG base64, and Marqo category key.

## Failure Cases

### Missing Or Invalid Auth

Handled by middleware before route execution.

Response:
- HTTP `401`
- JSON `status: 401`
- `data: null`

### Invalid JSON Shape

Examples:
- missing `image`
- missing `type`
- invalid `type`
- wrong JSON types

Handled by FastAPI request validation.

Response:
- HTTP `422`
- JSON `status: 422`
- `message: "Invalid request."`
- `data: null`

### Invalid Base64

Examples:
- `image: "not base64"`
- malformed data URL

Handled by service validation.

Response:
- HTTP `422`
- JSON `status: 422`
- `data: null`

### Unsupported Image Type

Examples:
- GIF
- WebP
- SVG
- PDF
- text file encoded as base64

Response:
- HTTP `422`
- JSON `status: 422`
- `data: null`

### Image Too Small

If either width or height is below `350px`, response is `422`.

The rule is "both edges must be at least 350px."

### No Garment Detected

If `yainage90/fashion-object-detection` returns no detections above `0.25`, response is `400`.

This means the image decoded correctly, but it did not pass the garment gate.

### Runtime Queue Failure

If the wardrobe GPU queue is full or times out, response is `503`.

No generated image is returned.

### Qwen Runtime Failure

Examples:
- missing AI-Toolkit root
- missing Qwen model path
- missing LoRA checkpoint
- AI-Toolkit import failure
- generation produces no output file

Response:
- HTTP `500`
- JSON `data: null`

### Marqo Runtime Failure

Examples:
- missing `open_clip`
- model load failure
- inference failure

Response:
- HTTP `500`
- JSON `data: null`

### Marqo Low Confidence

If generation succeeds but Marqo best score is below `0.25`, response is `400`.

No fallback category is used.

### Azure Upload Failure

Azure upload runs in the background.

If Azure storage is missing or upload fails after the main flow completes:
- the error is logged
- the user still keeps the `200` response
- Glamify progress sync cannot send valid URLs for that job

### Background Sync Failure

If `/wardrobe/progress` fails after the API response:
- the error is logged
- the user still keeps the `200` response
- no retry mechanism currently exists in this service

## Response Semantics

`data` is present only when the whole main flow completes:
- validation
- detection
- generation
- Marqo category classification

All errors return:

```json
{
  "data": null
}
```

The response `category` is:
- Marqo category key, for example `track_pants`

The response `categoryLabel` is:
- display text with the first letter capitalized, for example `Track pants`

The response `type` is:
- the original requested input type: `top`, `bottom`, or `dress`

It is not:
- the requested type
- the detector label
- a fallback value

## Health And Warmup

Warmup:
- `app/runtime/warmup.py`

Startup always runs wardrobe runtime warmup before serving requests:
1. loads Qwen Image Edit
2. creates the LoRA network
3. loads and caches top/bottom/dress LoRA state dicts

Health:
- `app/services/health.py`

Health includes wardrobe runtime status:
- runner loaded
- backend name
- queue active jobs
- queue waiting jobs
- queue max size

## Operational Notes

Use one API process per GPU-backed runtime set unless deployment has been explicitly designed for multi-process GPU loading.

Do not run many Uvicorn workers against the same GPU without planning VRAM use. Each worker can load its own Qwen model and LoRA cache.

The service assumes the pod has enough VRAM to keep the wardrobe Qwen runtime resident. This matches the intended H100-style deployment where latency is more important than minimizing residency.

## Minimal Curl Shape

```bash
curl -X POST "$BASE_URL/v1/wardrobe" \
  -H "Authorization: Bearer $JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'"$BASE64_IMAGE"'",
    "type": "bottom"
  }'
```

The returned `data.image` is raw JPEG base64. To inspect it locally:

```bash
jq -r '.data.image' response.json | base64 --decode > output.jpg
```

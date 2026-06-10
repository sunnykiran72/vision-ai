# Wardrobe API Flow

## Purpose

This document describes the implemented `/v1/wardrobe` flow in the Python AI service.

It covers:
- multipart request and JSON response contract
- authentication and user isolation
- image validation and preprocessing
- garment verification
- MiniCPM garment description
- diffusers extraction runtime
- Marqo category classification
- Azure storage
- background Glamify progress sync
- all major success and failure cases

## Entry Point

Route:
- `app/routes/wardrobe.py`

Endpoint:
- `POST /v1/wardrobe`

Request: `multipart/form-data`

| Field | Type | Description |
| --- | --- | --- |
| `image` | file | PNG or JPEG garment image |
| `type` | form text | `top`, `bottom`, or `dress` |

The request is multipart only. It does not accept JSON, base64, or a free-form `prompt` override.

Success response (JSON):

```json
{
  "status": 200,
  "message": "",
  "data": {
    "id": "89bd1674-0296-4652-8505-092d677b93a5",
    "type": "bottom",
    "image": "https://<account>.blob.core.windows.net/wardrobe-outputs/<user_id>/<job_id>/output.jpg",
    "category": "track_pants",
    "categoryLabel": "Track pants"
  }
}
```

`data.image` is a **public URL** to the uploaded extracted garment (JPEG), not base64.

Error response shape:

```json
{
  "status": 400,
  "message": "No garment was detected in the image.",
  "data": null
}
```

## Authentication And User Identity

JWT verification happens before the route handler in:
- `app/middleware/auth.py`

Auth is verified once, globally, in the HTTP middleware (`app.middleware("http")(...)`) — not in
each handler. The middleware:
1. allows public path prefixes (`/health`, `/docs`, `/redoc`, `/openapi.json`, `/tools`)
2. reads `Authorization: Bearer <token>`
3. verifies the token with `JWT_ACCESS_SECRET`
4. validates the decoded payload as `AuthPayload`
5. stores the payload on `request.state.auth_payload` **and** the raw token on
   `request.state.access_token`

Handlers read identity directly via dependencies in `app/dependencies/auth.py`:
- `CurrentAuth` -> `AuthPayload` (so `current_user.user_id`)
- `CurrentAccessToken` -> the raw verified JWT (passed downstream to the Glamify backend)

### Concurrency safety (important)

`request.state` lives on the per-request Starlette `Request` object — there is one `Request` per
HTTP request, never a shared module-level/global. So simultaneous requests cannot overwrite each
other's `user_id` or token, even while a long-running inference is in flight.

The handler reads `user_id` and `access_token` from state and passes them as **function
arguments** into `run_wardrobe_request(...)` (executed in a threadpool). Those values are bound to
that call's stack frame for its entire lifetime, so the response always uses the identity of the
request that started it. The request body cannot choose a user id.

## Route Flow

The route is intentionally thin.

It does:
1. accepts the multipart `image` file and `type` form field
2. resolves `CurrentAuth`
3. reads the image bytes and calls `run_wardrobe_request(...)` in a threadpool
4. sets the HTTP status code to match `response.status`

The route does not:
- validate or resize images
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
| Marqo confidence threshold | `0.20` |
| Marqo top-k metadata | `5` |
| LoRA rank | `16` |
| LoRA alpha | `16` |
| Glamify progress timeout | `20s` |
| Azure upload join timeout | `60s` |
| Output size | `832x1248` |
| Seed | `7777` (`GENERATION_SEED`) |
| Steps | `12` (`GENERATION_STEPS`) |
| LoRA scale | `1.0` (`GENERATION_NETWORK_MULTIPLIER`) |
| true_cfg_scale | `1.0` (`GENERATION_TRUE_CFG_SCALE`) |

MiniCPM-V (in-process vLLM) config, also in `app/constants/wardrobe.py`:

| Concern | Value |
| --- | --- |
| GPU memory utilization | `0.10` AWQ profile (`MINICPM_GPU_MEMORY_UTILIZATION`) |
| Max tokens | `100` (`MINICPM_MAX_TOKENS`) |
| Max slice nums | `6` (`MINICPM_MAX_SLICE_NUMS`) |
| Max model len | `2048` (`MINICPM_MAX_MODEL_LEN`) |
| Temperature | `0.0` (`MINICPM_TEMPERATURE`) |
| weight dtype | `auto` for AWQ (`MINICPM_DTYPE`) |
| KV cache dtype | `auto` (`MINICPM_KV_CACHE_DTYPE`) |
| KV scale calculation | `false` (`MINICPM_CALCULATE_KV_SCALES`) |
| CUDA graph mode | `MINICPM_ENFORCE_EAGER=false` |

The diffusers backend uses `seed`, `steps`, LoRA scale, and `true_cfg_scale`. The legacy
`GENERATION_GUIDANCE_RESCALE` / `GENERATION_SAMPLER` / `GENERATION_DO_CFG_NORM` constants remain in
the file for historical compatibility and are not used by wardrobe.

Qwen diffusers load dtype is controlled by `QWEN_IMAGE_EDIT_DTYPE`. Use `bfloat16` for the
production baseline. Do not set this to `float8_e4m3fn`: diffusers cannot load Qwen Image Edit that
way because fp8 needs quantized weights plus scales. The working fp8 path is
`QWEN_IMAGE_EDIT_DTYPE=bfloat16` with `QWEN_FP8=1`, which quantizes the transformer with torchao
after loading.

Qwen transformer-block compilation is controlled by `QWEN_COMPILE`. The production default is
`false`. When enabled, the wardrobe runtime compiles the Qwen transformer blocks with
`torch.compile(dynamic=False)` after all three wardrobe LoRAs are loaded, then warms top, bottom,
and dress once at startup. This is benchmark-only for now: varied MiniCPM caption lengths and input
image shapes can trigger new graph specializations, causing slow first requests even though repeated
identical prompts can be faster.

Static prompts (all in `app/constants/wardrobe.py`):

- `MINICPM_PROMPT_BY_TYPE[top|bottom|dress]` — the MiniCPM garment-description prompt per type.
- `QWEN_EXTRACT_PROMPT_TEMPLATE_BY_TYPE[top|bottom|dress]` — the Qwen extraction template with a
  `{caption}` slot filled by the MiniCPM caption. Only the leading trigger sentence differs:
  `GlamTopExt. Extract top wear as a standalone product.` /
  `GlamBtmExt. Extract bottom wear as a standalone product.` /
  `GlamDressExt. Extract dress as a standalone product.`
- `PROMPT_BY_TYPE[...]` — the short trigger-only prompts, used by the diffusers parity test
  endpoint and engine warmup, not by the live flow.

These values are code constants, not environment variables. Environment-specific endpoints and file paths live in runtime configuration.

## Runtime Configuration

Deployment-specific paths and queue settings live in:
- `app/config.py`

Important environment variables:

| Variable | Purpose |
| --- | --- |
| `QWEN_IMAGE_EDIT_MODEL_PATH` | Qwen Image Edit model path (e.g. `/mnt/models/qwen-image-edit-2511`) |
| `WARDROBE_LORA_TOP_PATH` | Top extraction LoRA checkpoint (top 23k) |
| `WARDROBE_LORA_BOTTOM_PATH` | Bottom extraction LoRA checkpoint (bottom 30k) |
| `WARDROBE_LORA_DRESS_PATH` | Dress extraction LoRA checkpoint (dress 27k) |
| `MINICPM_MODEL_PATH` | MiniCPM-V model id or local path, loaded in-process via vLLM |
| `AZURE_WARDROBE_INPUT_CONTAINER` | Private container for input images (default `wardrobe-inputs`) |
| `AZURE_WARDROBE_OUTPUT_CONTAINER` | Container for output images (default `wardrobe-outputs`) |
| `SYSTEM_QUEUE_MAX_SIZE` | System-wide GPU queue size, default `8` |
| `SYSTEM_QUEUE_WAIT_TIMEOUT_SECONDS` | System-wide queue wait timeout, default `30` |
| `GLAMIFY_API_BASE_URL` | Environment-specific Glamify backend base URL |

Wardrobe and try-on both use the diffusers Qwen backend, so `AI_TOOLKIT_ROOT` is not required for
either production runtime. When the `wardrobe` runtime is resident, startup validation requires the
Qwen model path, `MINICPM_MODEL_PATH`, the input/output containers, Azure settings, the three
wardrobe LoRA paths, and `GLAMIFY_API_BASE_URL`.

## Service Flow

Main service:
- `app/services/wardrobe.py`

Entrypoint:
- `run_wardrobe_request(...)`

### Step 1: Resolve Type

The form `type` is a Pydantic enum (`top`, `bottom`, `dress`). It selects the MiniCPM prompt, the
Qwen extraction template, and the extraction LoRA. No free-form prompt is accepted from the user;
the extraction prompt is built from the MiniCPM caption (Step 5).

### Step 2: Load The Uploaded Image

The service reads the multipart `image` file bytes and:
- opens them as an image through Pillow
- requires the format to be PNG or JPEG
- converts to RGB

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

Rules (`resize_input_for_model`, shared with the diffusers engine):
- if longest edge is `<= 1024`, keep the image size unchanged (no rounding)
- if longest edge is `> 1024`, scale the longest edge down to `1024` and round the other edge to
  the nearest multiple of 16
- preserve aspect ratio
- use Pillow `Image.Resampling.LANCZOS`

Examples:

| Input | Preprocessed |
| --- | --- |
| `512x768` | `512x768` |
| `1600x1200` | `1024x768` |
| `1200x1600` | `768x1024` |
| `2000x1000` | `1024x512` |

The preprocessed image stays **in memory** (no local file). The same object is fed to the fashion
detector, MiniCPM, and the diffusers engine, and is the exact image uploaded to Azure as the
wardrobe input (see Input Image Upload).

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

## Input Image Upload

Only after the detector confirms a garment, the service generates the `job_id` (UUID) and starts
uploading the **preprocessed image** (the in-memory LANCZOS-resized JPEG, not the raw upload) to
the private `wardrobe-inputs` container:

```text
<AZURE_WARDROBE_INPUT_CONTAINER>/<user_id>/<job_id>/input.jpg
```

Key behavior:
- it runs **strictly in the background** (a `Future`) and never blocks the MiniCPM -> Qwen -> Marqo
  pipeline; the request does not wait on it
- the returned URL is held in a temporary `Future` and is read only later, inside the background
  Glamify progress sync, where it is sent as `inputImage`
- this is the image stored as the user's wardrobe **input** (paired with the extracted output)
- if it fails, the error is logged in the background and the API response is unaffected

It is started here (post-detection) rather than earlier so a rejected, garment-less image never
produces a stored input.

## MiniCPM Garment Description

Client:
- `app/clients/minicpm_vllm.py` (`openbmb/MiniCPM-V-4_5`), shared singleton `get_minicpm_client()`.

MiniCPM-V is loaded **in-process via vLLM inside this service** — there is no external model
server or separate port. It is a faithful port of the validated reference engine: vLLM loads the
model in this process and coexists on one GPU beside the resident Qwen model, capped via
`MINICPM_GPU_MEMORY_UTILIZATION` and `max_num_seqs=1`. `MINICPM_ENFORCE_EAGER=false` allows vLLM
CUDA graphs for lower caption latency when VRAM headroom permits. The model pointer, dtype, KV cache
dtype, memory cap, max slices, and eager/CUDA-graph mode are environment-specific.

After the detector gate, the preprocessed image and the per-type `MINICPM_PROMPT_BY_TYPE[type]`
prompt are passed to MiniCPM, which returns one factual caption describing only the requested
garment.

The caption is used two ways:
1. it fills the `{caption}` slot of `QWEN_EXTRACT_PROMPT_TEMPLATE_BY_TYPE[type]` to build the Qwen
   extraction prompt;
2. it is sent to the Glamify backend as `promptDescription`.

MiniCPM runs through the same system GPU coordinator as detection and extraction, and is
warm-loaded at startup (see Health And Warmup). If MiniCPM fails, response is `500`.

## No Local Workspace

Every request gets a generated UUID `job_id`. This id is the **final wardrobe id** and the storage
path key, but the request does **not** write to local disk. The preprocessed input image (a PIL
object) is passed directly to MiniCPM and Qwen in memory, and the JPEG bytes are uploaded to Azure
directly. The request body never controls the job id or storage path.

## Runtime Execution

Runtime files:
- `app/runtime/wardrobe_runtime.py`
- `app/clients/qwen_diffusers_engine.py`
- `app/runtime/coordinator.py`

The wardrobe runtime uses **diffusers** directly, not AI-Toolkit. It is a faithful port of the
validated standalone diffusers tester. See `docs/diffusers-inference-parity.md` for the full
parity spec and the reference source.

Pipeline:
- `diffusers.QwenImageEditPlusPipeline`, bf16, resident on `cuda`.

Wardrobe extraction has one control image: the garment input.

Generation core:

```python
generator = torch.Generator(device="cuda").manual_seed(43)
pipe.set_adapters([garment_type], [1.0])
with torch.inference_mode():
    pipe(
        image=<preprocessed garment>,   # single control image
        prompt=<static category prompt>,
        true_cfg_scale=1.0,
        num_inference_steps=15,
        height=1248,
        width=832,
        generator=generator,
    ).images[0]
```

Steps, seed, LoRA scale, `true_cfg_scale`, and output size are constants in
`app/constants/wardrobe.py` (`GENERATION_STEPS`, `GENERATION_SEED`,
`GENERATION_NETWORK_MULTIPLIER`, `GENERATION_TRUE_CFG_SCALE`, `OUTPUT_WIDTH`, `OUTPUT_HEIGHT`).

## LoRA Loading And Switching

The engine loads the Qwen base pipeline once and keeps it resident.

Unlike the lazy-loading tester (which carries ~18 LoRAs), wardrobe carries only three, so all
three are loaded **eagerly during warmup, before the API serves requests**:
- `top`
- `bottom`
- `dress`

Each LoRA is loaded via `load_lora_weights(remapped_state_dict, adapter_name=category)`, where the
AI-Toolkit `diffusion_model.*` keys are remapped onto the diffusers `transformer.*` namespace.

On request:
1. resolve requested type
2. `set_adapters([type], [lora_scale])`
3. run generation

If the requested LoRA checkpoint is missing or the diffusers runtime cannot load, response is `500`.

## Queue And Concurrency

GPU work runs behind **one system-wide coordinator** (`app/runtime/system_coordinator.py`,
built on `app/runtime/coordinator.py`), not a per-feature queue. Wardrobe and try-on run the same
heavy Qwen weights on one GPU, so a single queue is correct.

Current behavior:
- queue size is controlled by `SYSTEM_QUEUE_MAX_SIZE`
- queue wait timeout by `SYSTEM_QUEUE_WAIT_TIMEOUT_SECONDS`
- detector, MiniCPM caption, Qwen generation, and Marqo classification all run behind it

This serializes GPU-heavy work in one process and prevents concurrent requests from mutating the
active LoRA at the same time.

Current error mapping:
- queue full -> `503`
- queue wait timeout -> `503`

The public response still follows the required envelope with `data: null`.

## Output Handling

After generation:
1. the engine returns a PIL image directly (no local file)
2. the service encodes it to JPEG bytes in memory
3. the output upload to the `wardrobe-outputs` container is **started immediately** (background),
   so it overlaps Marqo classification (saving ~1s)
4. after Marqo, the service **joins** the output upload to obtain the URL and returns it

Returned `data.image` is the Azure blob URL of the extracted garment JPEG. The input upload (to
`wardrobe-inputs`) stays fully in the background; only the output upload is joined because its URL
is part of the response.

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

Category resolution (`_resolve_wardrobe_category`):
- if Marqo returns a category key + label, it is used; `source` is `marqo` when the score clears
  `0.25`, otherwise `marqo_low_confidence` (the ranked match is still returned)
- if Marqo returns no usable category, the first candidate for the requested type is used with
  `source` `default_candidate_fallback`

Low Marqo confidence does **not** fail the request; the returned `category` is always populated.

## Azure Storage

Client:
- `app/clients/storage.py`

Input and output images are stored in **two separate private containers**. If Azure is not
configured, the request fails with `500` before generation.

The storage client keeps a process-wide cached `BlobServiceClient`, so repeated uploads reuse the
SDK client and its underlying connection pool instead of rebuilding the Azure client for every
input/output image. The API still waits for the output upload to complete before returning
`data.image`; this optimization only removes avoidable connection setup overhead.

Storage layout:

```text
<AZURE_WARDROBE_INPUT_CONTAINER>/<user_id>/<job_id>/input.jpg     # default container: wardrobe-inputs
<AZURE_WARDROBE_OUTPUT_CONTAINER>/<user_id>/<job_id>/output.jpg   # default container: wardrobe-outputs
```

`<job_id>` is the UUID returned as `data.id`. Content type is `image/jpeg`.

If the input upload or progress sync fails, the error is logged from the background worker and the
already completed API response is not changed.

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
Authorization: Bearer <verified access token from request.state>
Content-Type: application/json
```

Payload:

```json
{
  "id": "<job_id>",
  "inputImage": "<azure_input_url>",
  "outputImage": "<azure_output_url>",
  "promptDescription": "<minicpm garment caption>",
  "metadata": {
    "classification": {
      "primary_category": "bottoms",
      "category": "track_pants",
      "category_label": "Track pants",
      "score": 0.88,
      "source": "marqo"
    },
    "marqo": {
      "model": "Marqo/marqo-fashionSigLIP",
      "threshold": 0.25,
      "applied": true,
      "reason": "applied",
      "top_matches": []
    },
    "prompt": "GlamBtmExt. Extract bottom wear as a standalone product. Target regenerate garment is <caption> ; ...",
    "requested_type": "bottom"
  }
}
```

`promptDescription` is the MiniCPM garment caption (not the Qwen prompt). `metadata.prompt` is the
full Qwen extraction prompt with the caption embedded.

This sync is intentionally background-only.

Important behavior:
- input upload (to `wardrobe-inputs`) starts in the background after the detector gate
- output upload (to `wardrobe-outputs`) starts in the background right after generation, overlapping
  Marqo; the service joins it only to read the URL for the response
- the Glamify progress POST itself is fully background; the `/v1/wardrobe` response does not wait
  for it
- sync failure is logged and does not change the already-returned API response

## Success Case

Happy path:

1. Auth middleware validates JWT.
2. Route reads the multipart `image` file and `type` form field.
3. Service resolves `user_id` from JWT.
4. Image bytes are opened and PNG/JPEG format is verified.
5. Minimum dimensions are verified.
6. Image is resized to max edge `1024` (LANCZOS) if needed.
7. Azure storage is verified as configured.
8. Fashion detector finds at least one garment.
9. `job_id` (UUID) is generated; the input upload to `wardrobe-inputs` is queued in the background.
10. MiniCPM produces the garment caption from `MINICPM_PROMPT_BY_TYPE[type]`.
11. The caption fills `QWEN_EXTRACT_PROMPT_TEMPLATE_BY_TYPE[type]` to build the extraction prompt.
12. Diffusers runner activates the requested LoRA and generates the `832x1248` image (in memory).
13. The output upload to `wardrobe-outputs` is started in the background.
14. Marqo classifies the generated output (overlapping the upload) and the category is resolved.
15. The output upload is joined to obtain its URL.
16. Glamify progress sync (with `promptDescription` = caption) is queued in the background.
17. API returns `200` with `job_id`, the output image URL, and the Marqo category key.

## Error Messages

All errors return the envelope with `data: null` and a user-facing `message`:

| Instance | Status | `message` |
| --- | --- | --- |
| Missing/invalid auth | `401` | `Invalid token` |
| Missing `image`/`type` (FastAPI validation) | `422` | `Invalid request.` |
| Empty image file | `422` | `No image file was provided.` |
| Not a decodable image | `422` | `The uploaded file is not a valid image. Please upload a PNG or JPEG.` |
| Unsupported format (GIF/WebP/etc.) | `422` | `Unsupported image format. Only PNG and JPEG images are supported.` |
| Below min dimensions | `422` | `Image is too small. Width and height must both be at least 350px.` |
| No garment detected | `400` | `No garment was detected in the image. Please upload a clear photo of a single garment.` |
| Azure not configured | `500` | `Azure storage is required for wardrobe output.` |
| Queue full | `503` | `The system is busy. Please try again shortly.` |
| Queue wait timeout | `503` | `Timed out while waiting for an execution slot. Please try again.` |
| MiniCPM failure | `500` | `Garment description failed. Please try again.` |
| Detector/Marqo runtime failure | `500` | `Wardrobe validation runtime failed.` |
| Qwen runtime/generation failure | `500` | `Wardrobe runtime failed to initialize or execute.` |
| Output upload failure | `500` | `Failed to upload the extracted garment image.` |
| Any other error | `500` | `Wardrobe request failed.` |

## Failure Cases

### Missing Or Invalid Auth

Handled by middleware before route execution.

Response:
- HTTP `401`
- JSON `status: 401`
- `data: null`

### Invalid Multipart Shape

Examples:
- missing `image` file
- missing `type`
- invalid `type`

Handled by FastAPI request validation.

Response:
- HTTP `422`
- JSON `status: 422`
- `message: "Invalid request."`
- `data: null`

### Invalid Or Unsupported Image

Examples:
- `image` bytes are not a valid image
- GIF / WebP / SVG / PDF / non-image content

Handled by service validation.

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
- missing Qwen model path
- missing LoRA checkpoint
- diffusers/torch import failure
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

### MiniCPM Runtime Failure

If MiniCPM fails to load or describe the garment, response is `500`.

### Marqo Low Confidence

A low Marqo score does not fail the request. The category is still resolved (with `source`
`marqo_low_confidence` or a default candidate fallback) and a `200` is returned.

### Azure Not Configured

If Azure storage is not configured, the request fails with `500` before generation (the output URL
cannot be produced).

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

The response `image` is:
- the Azure blob URL of the extracted garment JPEG

The response `category` is:
- a Marqo category key (or the resolved fallback), for example `track_pants`

The response `categoryLabel` is:
- display text with the first letter capitalized, for example `Track pants`

The response `type` is:
- the original requested input type: `top`, `bottom`, or `dress`

## Health And Warmup

Warmup:
- `app/runtime/warmup.py`

Startup warm-loads **every** model the wardrobe flow depends on, before serving requests, and
nothing is unloaded afterwards:
1. loads the `QwenImageEditPlusPipeline` diffusers base (bf16, cuda) + the top/bottom/dress
   extraction LoRAs as adapters, then runs one warm pass
2. loads MiniCPM-V in-process via vLLM (after Qwen, so vLLM sizes its allocation around it)
3. loads the fashion detector
4. loads the Marqo classifier

The Qwen base is loaded before MiniCPM so vLLM's capped `gpu_memory_utilization` accounts for the
already resident Qwen weights.

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
  -F image=@garment.jpg \
  -F type=bottom
```

The returned `data.image` is a public Azure blob URL:

```bash
jq -r '.data.image' response.json   # https://<account>.blob.core.windows.net/.../output.jpg
```

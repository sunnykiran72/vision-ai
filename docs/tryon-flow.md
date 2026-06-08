# Try-on Flow

## Purpose

This document describes the current `/v1/tryon` implementation in the Python AI service.

Try-on now shares the resident **diffusers Qwen Image Edit Plus** runtime used by wardrobe. It no
longer depends on AI-Toolkit at runtime and does not load a second Qwen base model.

## Entry Point

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
- optional `seed`, `steps`, `guidance_scale`

Response envelope:
- `status`
- `message`
- `data.url`
- `data.metadata`

The API contract is unchanged by the diffusers migration.

## Authentication

JWT verification happens in `app/middleware/auth.py`. The route reads the already-validated
`current_user.user_id` through `app/dependencies/auth.py` and passes it into the service. The request
body never chooses the user id.

The Azure object path is server-generated:

```text
wardrobe_output/tryon/<user_id>/<job_id>/output.jpg
```

## Service Flow

Owner:
- `app/services/tryon.py`

Main entrypoint:
- `run_tryon_request(...)`

Flow:
1. Resolve `seed`, `steps`, and `guidance_scale` from the request or `Settings`.
2. Download the person image URL and every product image URL in parallel, then convert each image
   to RGB.
4. Collapse all products into one garment reference image through `build_product_reference(...)`.
5. Resize the garment reference so its longest side is at most `768px`.
6. Route to the try-on LoRA (`top`, `bottom`, `dress`, or `multi`).
7. Build the prompt.
8. Run the shared diffusers Qwen runner through the **system GPU coordinator** using the exact
   original person image dimensions as the requested output size.
9. Resize the generated image back to the original person image dimensions only if the backend
   returns a different size.
10. Encode JPEG bytes in memory and upload them to Azure.
11. Return the uploaded URL and metadata.

The service no longer writes `person.jpg`, `garment_reference.jpg`, or `output.jpg` to local disk for
model execution.

## Garment Reference

Product reference construction lives in:
- `app/utils/tryon_collage.py`

Modes:
- `single_product`
- `top_bottom_vertical_collage`
- `multi_product_horizontal_collage`

The Qwen pipeline receives two control images:

```text
image_1 = person
image_2 = garment_reference
```

The person image and garment reference image remain separate; only the product side is collaged.
The garment reference image is capped by `tryon.GARMENT_REFERENCE_MAX_EDGE_PX = 768` before it is
passed to Qwen, for lower latency and consistent input size.

## Constants

Try-on constants live in:
- `app/constants/tryon.py`

Important constants:
- `DEFAULT_STEPS = 12`
- `DEFAULT_SEED = 43`
- `DEFAULT_GUIDANCE_SCALE = 1.0`
- `GARMENT_REFERENCE_MAX_EDGE_PX = 768`
- `JPEG_QUALITY = 95`
- prompt triggers and static prompt section templates

## Runtime

Try-on runtime files:
- `app/runtime/tryon_runtime.py`
- `app/runtime/wardrobe_runtime.py`
- `app/clients/qwen_diffusers_engine.py`

`get_tryon_runner(...)` returns the same resident `QwenDiffusersWardrobeEngine` instance used by
wardrobe. Try-on adds extra LoRA adapters onto that same Qwen pipeline:

```text
wardrobe_top
wardrobe_bottom
wardrobe_dress
tryon_top
tryon_bottom
tryon_dress
tryon_multi
```

Adapter names are namespaced so wardrobe and try-on LoRAs cannot collide.

Try-on generation calls `QwenImageEditPlusPipeline` with:

```python
pipe(
    image=[person_image, garment_reference_image],
    prompt=prompt,
    true_cfg_scale=guidance_scale,
    num_inference_steps=steps,
    height=output_height,
    width=output_width,
    generator=generator,
)
```

Default steps are now:

```text
TRYON_DEFAULT_STEPS=12
```

The requested generation dimensions are always the source person image dimensions. There are no
fixed try-on output dimensions and no `/64` bucketing in the current diffusers path.

## Coordination

Try-on uses the process-wide system coordinator:
- `app/runtime/system_coordinator.py`

This is the same GPU queue used by wardrobe. It prevents simultaneous Qwen generations from
competing for the same resident model/GPU memory.

The old try-on-only coordinator is no longer used.

## Startup Configuration

For try-on, startup validation requires:
- `QWEN_IMAGE_EDIT_MODEL_PATH`
- enabled `TRYON_LORA_<KEY>_PATH` values

It does **not** require:
- `AI_TOOLKIT_ROOT`

Recommended RunPod specialist layout:

```text
TRYON_ENABLED_SPECIALISTS=top,bottom,dress,multi
TRYON_LORA_TOP_PATH=/workspace/loras/tryon/top.safetensors
TRYON_LORA_BOTTOM_PATH=/workspace/loras/tryon/bottom.safetensors
TRYON_LORA_DRESS_PATH=/workspace/loras/tryon/dress.safetensors
TRYON_LORA_MULTI_PATH=/workspace/loras/tryon/multi.safetensors
TRYON_DEFAULT_STEPS=12
```

## Output Upload

Output upload uses:
- `app/clients/storage.py`

The service uploads in-memory JPEG bytes with content type `image/jpeg`. The Azure storage client is
cached process-wide, so repeated uploads reuse the SDK client and its underlying connection pool.

The response is returned only after the output upload completes.

## Error Handling

Important mappings:
- invalid/missing auth: `401`
- queue full: `503`
- queue wait timeout: `504`
- user or garment download/open failure: `422` with `data: null`
- no generated image: `400`
- runtime initialization/execution failure: `500`
- storage missing: `500`

Invalid image response shape:

```json
{
  "status": 422,
  "message": "Garment image is invalid or could not be downloaded.",
  "data": null
}
```

## Timings

The response metadata includes per-section timing values under `data.metadata.timings`, including:
- `download_seconds`
- `reference_build_seconds`
- `prompt_route_seconds`
- `qwen_generation_seconds`
- `qwen_generation_queued_wall_seconds`
- `output_resize_seconds`
- `output_jpeg_encode_seconds`
- `output_upload_seconds`
- `total_wall_seconds`

The same major sections are also logged through the `glamify-ai` logger.

## Summary

The current try-on implementation is:
- JWT-authenticated
- user-id scoped through middleware-derived identity
- diffusers-based, not AI-Toolkit-based
- sharing one resident Qwen base with wardrobe
- specialist-LoRA capable
- garment-collage aware
- in-memory for model inputs and output encoding
- Azure-uploading before response

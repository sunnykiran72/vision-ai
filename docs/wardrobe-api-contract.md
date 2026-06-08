# Wardrobe API Contract

This document describes the `/v1/wardrobe` request and response shape used by the Glamify AI
service.

## Endpoint

```text
POST /v1/wardrobe
Content-Type: multipart/form-data
Authorization: Bearer <JWT>
```

The route is authenticated by the shared auth middleware before the request reaches the wardrobe
handler. The JWT must be signed with `JWT_ACCESS_SECRET` and must contain a UUID `userId`.

## Input

The request is `multipart/form-data`.

| Field | Type | Required | Values | Notes |
|---|---:|---:|---|---|
| `image` | file | yes | PNG or JPEG | Garment/source image. |
| `type` | string | yes | `top`, `bottom`, `dress` | Requested garment extraction type. |

Validation rules:
- `image` must be a valid PNG or JPEG.
- width and height must both be at least the configured minimum image edge.
- `type` must be one of `top`, `bottom`, or `dress`.

Example:

```bash
curl -X POST "$BASE_URL/v1/wardrobe" \
  -H "Authorization: Bearer $JWT" \
  -F image=@garment.jpg \
  -F type=top
```

## Success Response

HTTP status: `200`

```json
{
  "status": 200,
  "message": "",
  "data": {
    "id": "7f4b8f8e-cd48-41fb-9f2d-91b9ff2f02c9",
    "type": "top",
    "image": "https://<account>.blob.core.windows.net/wardrobe-outputs/<user_id>/<id>/output.jpg",
    "category": "shirts",
    "categoryLabel": "Shirts",
    "metadata": {
      "feature": "wardrobe",
      "id": "7f4b8f8e-cd48-41fb-9f2d-91b9ff2f02c9",
      "requested_type": "top",
      "promptDescription": "A concise MiniCPM garment description.",
      "prompt": "Final Qwen extraction prompt.",
      "minicpm_prompt": "MiniCPM prompt used for the requested type.",
      "classification": {},
      "marqo": {},
      "runtime": {},
      "uploads": {},
      "timings": {},
      "progress": {},
      "detections": [],
      "sizes": {}
    }
  }
}
```

Top-level `data` fields:

| Field | Type | Meaning |
|---|---:|---|
| `id` | string | Job id. Also used in storage paths and Glamify progress payloads. |
| `type` | string | Original requested type: `top`, `bottom`, or `dress`. |
| `image` | string | Public Azure URL of the extracted garment JPEG. |
| `category` | string | Final resolved category key from Marqo or fallback logic. |
| `categoryLabel` | string | Display label for the resolved category. |
| `metadata` | object | Debug/trace metadata for prompts, timings, uploads, model runtime, and progress sync. |

## Metadata

Important `metadata` fields:

| Field | Meaning |
|---|---|
| `promptDescription` | MiniCPM caption/description generated from the uploaded garment image. This is also sent to Glamify as `promptDescription`. |
| `prompt` | Final Qwen image extraction prompt built from the static template plus `promptDescription`. |
| `minicpm_prompt` | Static MiniCPM prompt selected by `type`. |
| `classification` | Final category decision used in the response. |
| `marqo` | Marqo model id, threshold, reason, and top matches. |
| `runtime` | Qwen runtime metadata such as backend, dtype, LoRA, steps, seed, and output size. |
| `uploads` | Input and output Azure upload metadata. Output upload is joined before returning. Input upload can be pending/failed without failing the response. |
| `timings` | Per-stage backend timings in seconds. |
| `progress.payload` | Payload prepared for the Glamify backend progress update. |
| `detections` | Fashion detector results from the preprocessed input. |
| `sizes` | Original input, preprocessed input, and output image dimensions. |

Common `metadata.timings` keys:

| Key | Meaning |
|---|---|
| `preprocess_seconds` | Decode, validation, and resize/preprocess time. |
| `fashion_detection_seconds` | Detector runtime. |
| `input_jpeg_encode_seconds` | Time to encode the preprocessed input JPEG for storage. |
| `minicpm_caption_seconds` | MiniCPM caption runtime. |
| `qwen_generation_seconds` | Qwen model generation time reported by the runtime. |
| `qwen_generation_queued_wall_seconds` | Wall time around the queued Qwen execution, including any queue wait. |
| `output_jpeg_encode_seconds` | Time to encode output image as JPEG. |
| `marqo_classification_seconds` | Marqo classification runtime. |
| `output_upload_wait_seconds` | Time spent waiting for output upload completion. |
| `input_upload_wait_seconds` | Time spent checking/waiting for input upload metadata after output is ready. |
| `total_wall_seconds` | Full backend wardrobe request wall time. |

## Storage

The service uploads:

```text
<AZURE_WARDROBE_INPUT_CONTAINER>/<user_id>/<job_id>/input.jpg
<AZURE_WARDROBE_OUTPUT_CONTAINER>/<user_id>/<job_id>/output.jpg
```

The response `data.image` points to the output image URL.

The input upload runs in the background. The output upload is required for the response because
`data.image` must be live when the API returns.

## Error Response

All errors use the shared response envelope:

```json
{
  "status": 422,
  "message": "The uploaded file is not a valid image. Please upload a PNG or JPEG.",
  "data": null
}
```

Common statuses:

| HTTP status | Example message | Cause |
|---:|---|---|
| `401` | `Invalid token` | Missing, invalid, expired, or malformed JWT. |
| `422` | `Invalid request.` | Request shape validation failed, such as missing `image` or invalid `type`. |
| `422` | `Unsupported image format. Only PNG and JPEG images are supported.` | Uploaded file is not supported. |
| `422` | `Image is too small...` | Image dimensions are below the configured minimum. |
| `400` | `No garment was detected in the image...` | Fashion detector did not find a garment. |
| `503` | `The system is busy. Please try again shortly.` | GPU queue is full. |
| `503` | `Timed out while waiting for an execution slot. Please try again.` | Request waited too long for GPU execution. |
| `500` | `Garment description failed. Please try again.` | MiniCPM captioning failed. |
| `500` | `Wardrobe validation runtime failed.` | Detector or Marqo runtime failed. |
| `500` | `Wardrobe runtime failed to initialize or execute.` | Qwen runtime/generation failed. |
| `500` | `Failed to upload the extracted garment image.` | Output upload failed. |

## Tester Notes

The local tester page is:

```text
http://localhost:8765/
```

Use **Generate Access Token** first, then choose an image and `type`, then send the wardrobe
request. The tester shows:
- total latency: browser/proxy wall time
- feature latency: backend-reported wardrobe timing
- prompt details: `promptDescription`, final Qwen prompt, and MiniCPM prompt
- timings: backend per-stage timings
- runtime/classification/upload metadata

# Upscale Reference Design

## Purpose

`/v1/upscale` is the first reference endpoint for the new GPU API implementation pattern.

It demonstrates:

- typed request and response contracts
- thin route / orchestration service / model client split
- request-local file isolation
- resident model runner usage
- optional Azure upload

This endpoint should be treated as the implementation template for the other core APIs.

## Public request contract

Request fields:

- `image_url`
- `metric`

Notes:

- `image_url` is the public input image URL
- `metric` is the product-facing target size option

Security rule:

- server-generated job ids and ownership context must drive actual artifact storage names

No runtime tuning fields are exposed publicly.

## Public response contract

Response envelope:

```json
{
  "status": 200,
  "message": "Image upscale completed successfully.",
  "data": {
    "url": "https://...",
    "metadata": {}
  }
}
```

Core response fields:

- `status`
- `message`
- `data.url`
- `data.metadata`

## Size mapping

The endpoint keeps the same quality-facing metric behavior as the previous system:

- `2k` -> `2048`
- `4k` -> `4096`

This preserves the target sizing logic users already liked.

## Image format decision

The default generated artifact format is `.jpg`.

This applies to:

- normalized request-local input artifacts
- generated output artifacts
- Azure upload content type

This is the current service-level default unless a later model integration forces a different output format.

This default is specific to photo-like upscale output.

For the other APIs, format policy should be explicit per endpoint:

- JPEG for photo outputs
- PNG or WebP for alpha-sensitive outputs
- do not force JPEG where transparency or masks are required

## Skip-inference branch

If the input image already meets the requested target long edge:

- do not run SeedVR2
- normalize and save the output artifact directly
- return the output in the same response shape

Why:

- avoids unnecessary GPU work
- reduces latency
- preserves throughput

This is an optimization, not a different public API mode.

## Resident SeedVR2 runner

The endpoint uses a resident in-process SeedVR2 runner.

Current design:

- import the SeedVR2 runtime module in-process
- keep runtime/module cache resident
- reuse that runner across requests
- avoid subprocess-per-request execution

This keeps the implementation closer to a high-performance GPU service model.

Concurrency must still be controlled around that resident runner. Do not assume unrestricted parallel access to a shared in-process runtime.

## Input safety

`image_url` requires production-grade download controls.

Required controls:

- block private IP ranges and link-local ranges
- block cloud metadata endpoints
- restrict redirect count
- enforce max download size
- validate content type and actual image decodability
- enforce image dimension limits
- use request timeout limits
- protect against decompression-bomb style payloads

## Azure upload behavior

If Azure storage is configured:

- upload the generated output artifact
- return the blob URL in `data.url`

If Azure storage is not configured:

- do not fail the request for that reason alone
- return success with `data.url = null`
- keep the local artifact metadata in the response

Production rule:

- if storage is required in a given environment, the service should fail startup rather than silently run misconfigured
- local filesystem paths should not be returned to external clients in production responses

Storage key rule:

- actual blob keys should be server-generated, for example:

```text
{tenant_id}/{job_id}/output.jpg
```

## Failure and status mapping

Expected status mapping:

- input download failure -> `400`
- downloaded content is not a valid image -> `400`
- unauthenticated request -> `401`
- unauthorized artifact access -> `403`
- file too large -> `413`
- unsupported media -> `415`
- semantically invalid request -> `422`
- queue saturated -> `429` or `503`
- queue or runtime timeout -> `504`
- model/runtime failure -> `500`
- successful upscale or skip-path normalization -> `200`

The response body `status` should match the real HTTP response code.

Overload and timeout conditions should not be collapsed into generic `500` responses.

## What should be copied to the other APIs

These parts of the design should be reused for the other core APIs:

- thin route pattern
- typed Pydantic models
- response envelope structure
- request-local file isolation
- shared media helper usage
- storage client usage
- resident runner boundary in `clients/`
- service-owned orchestration

Also copy:

- input safety rules
- storage key ownership rules
- error/status discipline
- request-local cleanup and retention policies

Only the endpoint-specific request contract and model runtime behavior should change.

## LLM review checklist

- Is the public contract minimal and stable?
- Is the route/service/client split clean?
- Is the resident runner strategy appropriate?
- Are runtime details hidden from the public contract?
- Is the skip-inference branch correct and well-scoped?
- Are SSRF and storage-key safety rules explicit enough?
- Are failure mappings production-grade?
- Which parts of this endpoint design are directly reusable for the other three APIs?

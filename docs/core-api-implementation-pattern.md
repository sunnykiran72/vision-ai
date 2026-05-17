# Core API Implementation Pattern

## Purpose

This service is one GPU-focused container that owns four core APIs:

- `/v1/wardrobe`
- `/v1/user_validation`
- `/v1/tryon`
- `/v1/upscale`

The goal is not to build a generic AI platform. The goal is to provide one clean, reusable implementation pattern for the GPU-backed APIs that live in this service.

## Standard internal layering

Each core API should follow the same structure:

- `app/routes/`
  - thin HTTP handlers only
  - parse request body
  - call service
  - set response status

- `app/models/`
  - Pydantic request models
  - Pydantic response models
  - shared response envelope models

- `app/services/`
  - orchestration layer
  - owns business flow and error mapping
  - calls model/storage/media clients

- `app/clients/`
  - model runtime integrations
  - storage integrations
  - no route-aware logic

- `app/utils/`
  - reusable helpers for media, logging, and lightweight file operations
  - no GPU runtime orchestration

## Shared API envelope

The shared response envelope is:

```json
{
  "status": 200,
  "message": "Image upscale completed successfully.",
  "data": {}
}
```

Rules:

- use real HTTP status codes at the transport layer
- keep the JSON `status` field aligned with the actual HTTP response code
- keep `message` human-readable
- keep `data` typed per endpoint

## Security and tenant isolation

Every core API must operate with explicit request ownership.

Required rules:

- every authenticated request must carry resolved server-side identity context
  - `user_id`
  - `tenant_id` when multi-tenant
- every job must use a server-generated `job_id`
- every artifact must be owned by exactly one `(tenant_id, user_id, job_id)` tuple
- all artifact lookup or retrieval paths must re-check authorization against that tuple
- user-controlled fields must never be used as trusted storage or filesystem identifiers
- blob/object names must be server-generated and non-guessable

Recommended storage key pattern:

```text
{tenant_id}/{job_id}/output.jpg
```

`output_file_name` may exist as a user-facing display hint, but it must not become the actual storage key.

## Validation standard

Pydantic is the standard validation layer for this service.

Why:

- native fit for FastAPI
- request and response validation
- schema generation
- shared settings loading through `pydantic-settings`

Use Pydantic types wherever possible:

- `HttpUrl` for URLs
- enums for stable option sets
- typed nested models for response payloads

## Reusable helpers and clients

The reusable media and storage concerns should stay centralized:

- `media_utils.py`
  - download media from URL
  - create job directories
  - normalize output filenames
  - build request-local input/output paths

- `storage.py`
  - Azure Blob upload
  - content-type handling
  - blob URL construction

These helpers should be reused by all core APIs where applicable.

## Resident runner model

GPU-heavy model execution should follow this pattern:

- load the runtime once
- keep the model/runtime resident in the API process
- reuse internal model cache across requests
- avoid subprocess-per-request execution
- avoid offload-first design on large GPU nodes

This keeps the hot path cleaner and reduces repeated startup overhead.

## Runtime execution contract

The default production concurrency contract must be explicit.

Required default:

- one execution lock per resident runner unless model thread safety is proven
- bounded in-memory queue in front of each runner
- configured queue size limit
- configured queue wait timeout
- configured request execution timeout
- explicit cancellation policy for disconnected clients

Backpressure behavior:

- queue full -> `429` or `503`
- queue wait timeout -> `503` or `504`
- runtime execution timeout -> `504`

The route/service layer must not implement concurrency policy directly. It should delegate to a coordinator.

## Deployment topology

Resident GPU runners must be documented against the FastAPI/Uvicorn process model.

Rules:

- do not assume `uvicorn --workers N` is safe for GPU-backed resident models
- each worker process may load its own model copy and exhaust VRAM
- default topology for single-GPU deployment is one API process per GPU-backed runtime set
- multi-GPU scaling should be explicit:
  - one process per GPU
  - or one orchestrator plus dedicated worker processes pinned to GPUs

Startup warmup must respect this topology so models are loaded once per intended GPU runtime process.

## Startup warmup

The preferred loading strategy is startup warmup.

Meaning:

- load required runtimes when the API process starts
- fail fast if required model/runtime paths are invalid for that deployment
- avoid first-request cold start penalties

Future model warmup should use the same mechanism for:

- SeedVR2
- try-on runtime
- wardrobe runtime
- user validation runtime

## What should stay out of public API contracts

The following are internal runtime concerns and should not be public request fields:

- offload flags
- subprocess mode
- GPU residency toggles
- batch size tuning
- timeout tuning
- internal cache behavior

These belong in config and runner internals, not in public endpoint contracts.

## Operational behavior

Every core API implementation should define:

- startup validation of required model/runtime paths
- startup validation of production-required storage settings
- health and readiness behavior
- request id / job id correlation
- metrics for:
  - queue depth
  - active jobs
  - GPU memory usage
  - latency
  - error rate

## Reuse across the other core APIs

`/v1/upscale` is the first reference implementation of this pattern.

The same structure should be applied to:

- `tryon`
- `wardrobe`
- `user_validation`

The public request contract will differ by endpoint, but the route/service/client/utils split should remain consistent.

## LLM review checklist

- Is the route/service/client split clean and reusable?
- Are request and response contracts minimal and stable?
- Are runtime details correctly hidden from public APIs?
- Is Pydantic used in the right places?
- Are media and storage helpers separated cleanly?
- Are tenant isolation and artifact ownership rules explicit enough?
- Is the runtime execution contract concrete enough for production?
- Is this pattern reusable for the other three core APIs?

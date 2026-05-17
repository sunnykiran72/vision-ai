# Architecture

## Goals

- Rebuild the four core APIs in one deployable container
- Keep the runtime structure easy to scan
- Separate route logic from workflow logic without over-layering
- Keep GPU-heavy model execution behind resident runners and service-owned orchestration

## Proposed layers

- `app/routes/`: HTTP routes
- `app/models/`: request and response models
- `app/services/`: workflow orchestration
- `app/clients/`: model and external service clients
- `app/utils/`: small helpers
- `app/config.py`: app settings

## Current API surface

- `/health`
- `/v1/wardrobe`
- `/v1/user_validation`
- `/v1/tryon`
- `/v1/upscale`

## Runtime direction

- GPU-backed runtimes should be loaded as resident runners rather than spawned per request
- Startup warmup is the preferred model-loading strategy so first user request is predictable
- Public API contracts should stay minimal and product-facing
- Runtime tuning details such as offload mode, subprocess execution, and model residency should stay internal

## Reference implementation

- `/v1/upscale` is the first reference implementation of the new pattern
- The same route/service/client/utils structure should be reused for:
  - `wardrobe`
  - `user_validation`
  - `tryon`

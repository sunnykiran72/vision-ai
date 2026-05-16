# Architecture

## Goals

- Rebuild the four core APIs in one deployable container
- Keep the runtime structure easy to scan
- Separate route logic from workflow logic without over-layering

## Proposed layers

- `app/routes/`: HTTP routes
- `app/models/`: request and response models
- `app/services/`: workflow orchestration
- `app/clients/`: model and external service clients
- `app/utils/`: small helpers
- `app/config.py`: app settings

## Endpoint migration order

1. `/v1/user-image/prepare`
2. `/analyze`
3. `/v1/flux2/tryon`
4. `/v1/user-image/upscale`

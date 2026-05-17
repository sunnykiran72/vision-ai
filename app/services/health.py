from __future__ import annotations

import platform

from app.config import Settings
from app.models.health import (
    HealthMetadata,
    HealthResponse,
    RuntimeMetadata,
    RuntimeQueueMetadata,
    RuntimeRunnerMetadata,
)
from app.runtime.upscale_runtime import get_upscale_runtime_status


def build_health_response(settings: Settings) -> HealthResponse:
    configured_domains: list[str] = []
    if settings.jwt_access_secret:
        configured_domains.append("security")
    if settings.azure_storage_connection_string and settings.azure_storage_container:
        configured_domains.append("storage")
    if settings.tryon_model_path or settings.tryon_lora_path:
        configured_domains.append("tryon")
    if settings.upscale_model_path:
        configured_domains.append("upscale")

    upscale_runtime = get_upscale_runtime_status(settings)
    metadata = HealthMetadata(
        environment=settings.app_env,
        version="0.1.0",
        python_version=platform.python_version(),
        available_api_groups=[
            "health",
            "wardrobe",
            "user_validation",
            "tryon",
            "upscale",
        ],
        configured_domains=configured_domains,
        runtimes={
            "upscale": RuntimeMetadata(
                runner=RuntimeRunnerMetadata(
                    loaded=upscale_runtime.runner.loaded,
                    backend=upscale_runtime.runner.backend,
                ),
                queue=RuntimeQueueMetadata(
                    active_jobs=upscale_runtime.coordinator.active_jobs,
                    waiting_jobs=upscale_runtime.coordinator.waiting_jobs,
                    max_queue_size=upscale_runtime.coordinator.max_queue_size,
                ),
            ),
        },
    )
    return HealthResponse(
        status="ok",
        service="glamify-vision-ai",
        metadata=metadata,
    )

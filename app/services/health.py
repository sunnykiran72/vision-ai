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
from app.runtime.tryon_runtime import get_tryon_runtime_status
from app.runtime.upscale_runtime import get_upscale_runtime_status
from app.runtime.wardrobe_runtime import get_wardrobe_runtime_status


def build_health_response(settings: Settings) -> HealthResponse:
    configured_domains: list[str] = []
    if settings.jwt_access_secret:
        configured_domains.append("security")
    if settings.azure_storage_connection_string and settings.azure_storage_container:
        configured_domains.append("storage")
    if (
        settings.qwen_image_edit_model_path
        or settings.wardrobe_lora_top_path
        or settings.wardrobe_lora_bottom_path
        or settings.wardrobe_lora_dress_path
    ):
        configured_domains.append("wardrobe")
    if (
        settings.qwen_image_edit_model_path
        or settings.tryon_lora_top_path
        or settings.tryon_lora_bottom_path
        or settings.tryon_lora_dress_path
        or settings.tryon_lora_multi_path
    ):
        configured_domains.append("tryon")
    if settings.upscale_model_path:
        configured_domains.append("upscale")

    wardrobe_runtime = get_wardrobe_runtime_status(settings)
    tryon_runtime = get_tryon_runtime_status(settings)
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
            "wardrobe": RuntimeMetadata(
                runner=RuntimeRunnerMetadata(
                    loaded=wardrobe_runtime.runner.loaded,
                    backend=wardrobe_runtime.runner.backend,
                ),
                queue=RuntimeQueueMetadata(
                    active_jobs=wardrobe_runtime.coordinator.active_jobs,
                    waiting_jobs=wardrobe_runtime.coordinator.waiting_jobs,
                    max_queue_size=wardrobe_runtime.coordinator.max_queue_size,
                ),
            ),
            "tryon": RuntimeMetadata(
                runner=RuntimeRunnerMetadata(
                    loaded=tryon_runtime.runner.loaded,
                    backend=tryon_runtime.runner.backend,
                ),
                queue=RuntimeQueueMetadata(
                    active_jobs=tryon_runtime.coordinator.active_jobs,
                    waiting_jobs=tryon_runtime.coordinator.waiting_jobs,
                    max_queue_size=tryon_runtime.coordinator.max_queue_size,
                ),
            ),
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
        service="glamify-image-ai",
        metadata=metadata,
    )

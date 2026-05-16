from __future__ import annotations

import platform

from app.config import Settings
from app.models.health import HealthMetadata, HealthResponse


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
    )
    return HealthResponse(
        status="ok",
        service="glamify-vision-ai",
        metadata=metadata,
    )

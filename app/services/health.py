from __future__ import annotations

import platform

from app.config import Settings
from app.models.health import HealthMetadata, HealthResponse


def build_health_response(settings: Settings) -> HealthResponse:
    metadata = HealthMetadata(
        environment=settings.app_env,
        version=settings.app_version,
        python_version=platform.python_version(),
        host=settings.host,
        port=settings.port,
        configured_services=[
            "minicpm" if settings.minicpm_service_url else "minicpm_unconfigured",
            (
                "azure_storage"
                if settings.azure_storage_connection_string
                else "azure_storage_unconfigured"
            ),
        ],
        available_api_groups=[
            "health",
            "wardrobe",
            "user_validation",
            "tryon",
            "upscale",
        ],
    )
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        metadata=metadata,
    )

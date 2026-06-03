from __future__ import annotations

from dataclasses import dataclass

from app.config import (
    Settings,
    get_enabled_resident_runtimes,
    get_settings,
    validate_startup_settings,
)
from app.runtime.tryon_runtime import warmup_tryon_runtime
from app.runtime.upscale_runtime import warmup_upscale_runtime
from app.runtime.wardrobe_runtime import warmup_wardrobe_runtime


@dataclass(frozen=True)
class WarmupResult:
    runtime_name: str
    warmed: bool


def validate_required_service_config(settings: Settings | None = None) -> None:
    resolved_settings = settings or get_settings()
    validate_startup_settings(resolved_settings)


def warmup_resident_runtimes(settings: Settings | None = None) -> list[WarmupResult]:
    resolved_settings = settings or get_settings()
    validate_required_service_config(resolved_settings)
    results: list[WarmupResult] = []

    enabled_runtimes = get_enabled_resident_runtimes(resolved_settings)

    if "wardrobe" in enabled_runtimes:
        warmup_wardrobe_runtime(resolved_settings)
        results.append(WarmupResult(runtime_name="wardrobe", warmed=True))

    if "tryon" in enabled_runtimes:
        warmup_tryon_runtime(resolved_settings)
        results.append(WarmupResult(runtime_name="tryon", warmed=True))

    if "upscale" in enabled_runtimes:
        warmup_upscale_runtime(resolved_settings)
        results.append(WarmupResult(runtime_name="upscale", warmed=True))

    return results

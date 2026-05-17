from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings
from app.runtime.upscale_runtime import warmup_upscale_runtime


@dataclass(frozen=True)
class WarmupResult:
    runtime_name: str
    warmed: bool


def warmup_resident_runtimes(settings: Settings | None = None) -> list[WarmupResult]:
    resolved_settings = settings or get_settings()
    results: list[WarmupResult] = []

    warmup_upscale_runtime(resolved_settings)
    results.append(WarmupResult(runtime_name="upscale", warmed=True))

    return results

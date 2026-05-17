from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.clients.seedvr2 import (
    SeedVR2Client,
    SeedVR2RunResult,
    SeedVR2RuntimeStatus,
    get_seedvr2_client,
)
from app.config import Settings, get_settings
from app.runtime.coordinator import BoundedExecutionCoordinator, CoordinatorSnapshot


@dataclass(frozen=True)
class UpscaleRuntimeStatus:
    runner: SeedVR2RuntimeStatus
    coordinator: CoordinatorSnapshot


def get_upscale_runner(settings: Settings | None = None) -> SeedVR2Client:
    resolved_settings = settings or get_settings()
    return _get_upscale_runner_cached(
        resolved_settings.upscale_model_path,
        resolved_settings.upscale_model_variant,
        resolved_settings.upscale_cli_path,
    )


@lru_cache(maxsize=8)
def _get_upscale_runner_cached(
    upscale_model_path: str,
    upscale_model_variant: str,
    upscale_cli_path: str,
) -> SeedVR2Client:
    return get_seedvr2_client(
        upscale_model_path,
        upscale_model_variant,
        upscale_cli_path,
    )


def get_upscale_execution_coordinator(
    settings: Settings | None = None,
) -> BoundedExecutionCoordinator[SeedVR2RunResult]:
    resolved_settings = settings or get_settings()
    return _get_upscale_execution_coordinator_cached(
        resolved_settings.upscale_queue_max_size,
        resolved_settings.upscale_queue_wait_timeout_seconds,
    )


@lru_cache(maxsize=8)
def _get_upscale_execution_coordinator_cached(
    max_queue_size: int,
    queue_wait_timeout_seconds: int,
) -> BoundedExecutionCoordinator[SeedVR2RunResult]:
    return BoundedExecutionCoordinator(
        max_queue_size=max_queue_size,
        queue_wait_timeout_seconds=queue_wait_timeout_seconds,
    )


def warmup_upscale_runtime(settings: Settings | None = None) -> None:
    resolved_settings = settings or get_settings()
    get_upscale_runner(resolved_settings).warmup()


def get_upscale_runtime_status(settings: Settings | None = None) -> UpscaleRuntimeStatus:
    resolved_settings = settings or get_settings()
    return UpscaleRuntimeStatus(
        runner=get_upscale_runner(resolved_settings).status(),
        coordinator=get_upscale_execution_coordinator(resolved_settings).snapshot(),
    )

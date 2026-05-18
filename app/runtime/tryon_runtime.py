from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.clients.qwen_image_edit import (
    QwenImageEditClient,
    QwenImageEditRunResult,
    QwenImageEditRuntimeStatus,
)
from app.config import Settings, get_settings
from app.runtime.coordinator import BoundedExecutionCoordinator, CoordinatorSnapshot


@dataclass(frozen=True)
class TryonRuntimeStatus:
    runner: QwenImageEditRuntimeStatus
    coordinator: CoordinatorSnapshot


def get_tryon_runner(settings: Settings | None = None) -> QwenImageEditClient:
    resolved_settings = settings or get_settings()
    return _get_tryon_runner_cached(
        resolved_settings.qwen_image_edit_model_path,
        resolved_settings.tryon_lora_path,
        resolved_settings.tryon_lora_weight_name,
        resolved_settings.tryon_lora_scale,
    )


@lru_cache(maxsize=8)
def _get_tryon_runner_cached(
    qwen_image_edit_model_path: str,
    tryon_lora_path: str,
    tryon_lora_weight_name: str,
    tryon_lora_scale: float,
) -> QwenImageEditClient:
    settings = Settings(QWEN_IMAGE_EDIT_MODEL_PATH=qwen_image_edit_model_path)
    return QwenImageEditClient(
        settings,
        lora_path=tryon_lora_path,
        lora_weight_name=tryon_lora_weight_name,
        lora_scale=tryon_lora_scale,
        adapter_name="tryon",
    )


def get_tryon_execution_coordinator(
    settings: Settings | None = None,
) -> BoundedExecutionCoordinator[QwenImageEditRunResult]:
    resolved_settings = settings or get_settings()
    return _get_tryon_execution_coordinator_cached(
        resolved_settings.tryon_queue_max_size,
        resolved_settings.tryon_queue_wait_timeout_seconds,
    )


@lru_cache(maxsize=8)
def _get_tryon_execution_coordinator_cached(
    max_queue_size: int,
    queue_wait_timeout_seconds: int,
) -> BoundedExecutionCoordinator[QwenImageEditRunResult]:
    return BoundedExecutionCoordinator(
        max_queue_size=max_queue_size,
        queue_wait_timeout_seconds=queue_wait_timeout_seconds,
    )


def warmup_tryon_runtime(settings: Settings | None = None) -> None:
    resolved_settings = settings or get_settings()
    get_tryon_runner(resolved_settings).warmup()


def get_tryon_runtime_status(settings: Settings | None = None) -> TryonRuntimeStatus:
    resolved_settings = settings or get_settings()
    return TryonRuntimeStatus(
        runner=get_tryon_runner(resolved_settings).status(),
        coordinator=get_tryon_execution_coordinator(resolved_settings).snapshot(),
    )

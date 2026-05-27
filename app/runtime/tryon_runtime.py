from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.clients.qwen_tryon_aitk import QwenTryonAitkClient
from app.config import Settings, get_settings
from app.runtime.coordinator import BoundedExecutionCoordinator, CoordinatorSnapshot
from app.runtime.tryon_types import TryonRunner, TryonRunResult, TryonRuntimeStatus


@dataclass(frozen=True)
class TryonRuntimeSnapshot:
    runner: TryonRuntimeStatus
    coordinator: CoordinatorSnapshot


def get_tryon_runner(settings: Settings | None = None) -> TryonRunner:
    resolved_settings = settings or get_settings()
    return _get_tryon_runner_cached(
        resolved_settings.ai_toolkit_root,
        resolved_settings.qwen_image_edit_model_path,
        resolved_settings.tryon_lora_path,
        resolved_settings.tryon_lora_rank,
        resolved_settings.tryon_lora_alpha,
        resolved_settings.tryon_lora_scale,
        resolved_settings.tryon_sampler,
        resolved_settings.tryon_guidance_rescale,
        resolved_settings.tryon_do_cfg_norm,
        resolved_settings.tryon_output_width,
        resolved_settings.tryon_output_height,
    )


@lru_cache(maxsize=8)
def _get_tryon_runner_cached(
    ai_toolkit_root: str,
    qwen_image_edit_model_path: str,
    tryon_lora_path: str,
    tryon_lora_rank: int,
    tryon_lora_alpha: int,
    tryon_lora_scale: float,
    tryon_sampler: str,
    tryon_guidance_rescale: float,
    tryon_do_cfg_norm: bool,
    tryon_output_width: int,
    tryon_output_height: int,
) -> TryonRunner:
    settings = Settings(
        AI_TOOLKIT_ROOT=ai_toolkit_root,
        QWEN_IMAGE_EDIT_MODEL_PATH=qwen_image_edit_model_path,
        TRYON_LORA_PATH=tryon_lora_path,
        TRYON_LORA_RANK=tryon_lora_rank,
        TRYON_LORA_ALPHA=tryon_lora_alpha,
        TRYON_LORA_SCALE=tryon_lora_scale,
        TRYON_SAMPLER=tryon_sampler,
        TRYON_GUIDANCE_RESCALE=tryon_guidance_rescale,
        TRYON_DO_CFG_NORM=tryon_do_cfg_norm,
        TRYON_OUTPUT_WIDTH=tryon_output_width,
        TRYON_OUTPUT_HEIGHT=tryon_output_height,
    )
    return QwenTryonAitkClient(settings)


def get_tryon_execution_coordinator(
    settings: Settings | None = None,
) -> BoundedExecutionCoordinator[TryonRunResult]:
    resolved_settings = settings or get_settings()
    return _get_tryon_execution_coordinator_cached(
        resolved_settings.tryon_queue_max_size,
        resolved_settings.tryon_queue_wait_timeout_seconds,
    )


@lru_cache(maxsize=8)
def _get_tryon_execution_coordinator_cached(
    max_queue_size: int,
    queue_wait_timeout_seconds: int,
) -> BoundedExecutionCoordinator[TryonRunResult]:
    return BoundedExecutionCoordinator(
        max_queue_size=max_queue_size,
        queue_wait_timeout_seconds=queue_wait_timeout_seconds,
    )


def warmup_tryon_runtime(settings: Settings | None = None) -> None:
    resolved_settings = settings or get_settings()
    get_tryon_runner(resolved_settings).warmup()


def get_tryon_runtime_status(settings: Settings | None = None) -> TryonRuntimeSnapshot:
    resolved_settings = settings or get_settings()
    return TryonRuntimeSnapshot(
        runner=get_tryon_runner(resolved_settings).status(),
        coordinator=get_tryon_execution_coordinator(resolved_settings).snapshot(),
    )

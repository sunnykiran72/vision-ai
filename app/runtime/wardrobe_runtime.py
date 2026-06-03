from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.clients.qwen_wardrobe_aitk import QwenWardrobeAitkClient
from app.config import Settings, get_settings
from app.constants import wardrobe as wardrobe_constants
from app.runtime.coordinator import BoundedExecutionCoordinator, CoordinatorSnapshot
from app.runtime.wardrobe_types import (
    WardrobeRunner,
    WardrobeRuntimeStatus,
)


@dataclass(frozen=True)
class WardrobeRuntimeSnapshot:
    runner: WardrobeRuntimeStatus
    coordinator: CoordinatorSnapshot


def get_wardrobe_runner(settings: Settings | None = None) -> WardrobeRunner:
    resolved_settings = settings or get_settings()
    return _get_wardrobe_runner_cached(
        resolved_settings.ai_toolkit_root,
        resolved_settings.qwen_image_edit_model_path,
        resolved_settings.wardrobe_lora_top_path,
        resolved_settings.wardrobe_lora_bottom_path,
        resolved_settings.wardrobe_lora_dress_path,
    )


@lru_cache(maxsize=8)
def _get_wardrobe_runner_cached(
    ai_toolkit_root: str,
    qwen_image_edit_model_path: str,
    wardrobe_lora_top_path: str,
    wardrobe_lora_bottom_path: str,
    wardrobe_lora_dress_path: str,
) -> WardrobeRunner:
    settings = Settings(
        AI_TOOLKIT_ROOT=ai_toolkit_root,
        QWEN_IMAGE_EDIT_MODEL_PATH=qwen_image_edit_model_path,
        WARDROBE_LORA_TOP_PATH=wardrobe_lora_top_path,
        WARDROBE_LORA_BOTTOM_PATH=wardrobe_lora_bottom_path,
        WARDROBE_LORA_DRESS_PATH=wardrobe_lora_dress_path,
    )
    return QwenWardrobeAitkClient(settings)


def get_wardrobe_execution_coordinator(
    settings: Settings | None = None,
) -> BoundedExecutionCoordinator[Any]:
    resolved_settings = settings or get_settings()
    return _get_wardrobe_execution_coordinator_cached(
        resolved_settings.wardrobe_queue_max_size,
    )


@lru_cache(maxsize=8)
def _get_wardrobe_execution_coordinator_cached(
    max_queue_size: int,
) -> BoundedExecutionCoordinator[Any]:
    return BoundedExecutionCoordinator(
        max_queue_size=max_queue_size,
        queue_wait_timeout_seconds=wardrobe_constants.QUEUE_WAIT_TIMEOUT_SECONDS,
    )


def warmup_wardrobe_runtime(settings: Settings | None = None) -> None:
    resolved_settings = settings or get_settings()
    get_wardrobe_runner(resolved_settings).warmup()


def get_wardrobe_runtime_status(settings: Settings | None = None) -> WardrobeRuntimeSnapshot:
    resolved_settings = settings or get_settings()
    return WardrobeRuntimeSnapshot(
        runner=get_wardrobe_runner(resolved_settings).status(),
        coordinator=get_wardrobe_execution_coordinator(resolved_settings).snapshot(),
    )

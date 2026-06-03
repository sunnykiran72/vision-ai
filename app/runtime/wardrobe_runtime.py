from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from app.config import Settings, get_settings
from app.constants import wardrobe as wardrobe_constants
from app.runtime.coordinator import BoundedExecutionCoordinator, CoordinatorSnapshot
from app.runtime.qwen_shared_runtime import get_shared_qwen_runner
from app.runtime.wardrobe_types import (
    WardrobeRunner,
    WardrobeRuntimeStatus,
)


@dataclass(frozen=True)
class WardrobeRuntimeSnapshot:
    runner: WardrobeRuntimeStatus
    coordinator: CoordinatorSnapshot


class SharedWardrobeRunnerAdapter:
    def __init__(self, shared_runner: Any) -> None:
        self._shared_runner = shared_runner

    def warmup(self) -> None:
        self._shared_runner.warmup()

    def status(self) -> WardrobeRuntimeStatus:
        return cast(WardrobeRuntimeStatus, self._shared_runner.wardrobe_status())

    def run_extract(
        self,
        *,
        input_image_path: str,
        prompt: str,
        garment_type: str,
        output_path: str,
    ) -> Any:
        return self._shared_runner.run_extract(
            input_image_path=input_image_path,
            prompt=prompt,
            garment_type=garment_type,
            output_path=output_path,
        )


def get_wardrobe_runner(settings: Settings | None = None) -> WardrobeRunner:
    resolved_settings = settings or get_settings()
    return SharedWardrobeRunnerAdapter(get_shared_qwen_runner(resolved_settings))


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
    return SharedWardrobeRunnerAdapter(get_shared_qwen_runner(settings))


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
    runner = get_wardrobe_runner(resolved_settings)
    return WardrobeRuntimeSnapshot(
        runner=runner.status(),
        coordinator=get_wardrobe_execution_coordinator(resolved_settings).snapshot(),
    )

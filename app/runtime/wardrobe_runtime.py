from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from app.clients.fashion_detection import get_fashion_detection_client
from app.clients.marqo_fashion import get_marqo_fashion_client
from app.clients.minicpm_vllm import get_minicpm_client
from app.clients.qwen_diffusers_engine import QwenDiffusersWardrobeEngine
from app.config import Settings, get_settings
from app.runtime.coordinator import CoordinatorSnapshot
from app.runtime.system_coordinator import get_system_execution_coordinator
from app.runtime.wardrobe_types import WardrobeRuntimeStatus

logger = logging.getLogger("glamify-ai")


@dataclass(frozen=True)
class WardrobeRuntimeSnapshot:
    runner: WardrobeRuntimeStatus
    coordinator: CoordinatorSnapshot


def get_wardrobe_runner(settings: Settings | None = None) -> QwenDiffusersWardrobeEngine:
    resolved_settings = settings or get_settings()
    return _get_wardrobe_runner_cached(
        resolved_settings.qwen_image_edit_model_path,
        resolved_settings.qwen_image_edit_dtype,
        resolved_settings.qwen_compile,
        resolved_settings.qwen_fp8,
        resolved_settings.wardrobe_lora_top_path,
        resolved_settings.wardrobe_lora_bottom_path,
        resolved_settings.wardrobe_lora_dress_path,
        resolved_settings.tryon_enabled_specialists,
        resolved_settings.tryon_lora_top_path,
        resolved_settings.tryon_lora_bottom_path,
        resolved_settings.tryon_lora_dress_path,
        resolved_settings.tryon_lora_multi_path,
        resolved_settings.tryon_lora_rank,
        resolved_settings.tryon_lora_alpha,
        resolved_settings.tryon_lora_scale,
    )


@lru_cache(maxsize=8)
def _get_wardrobe_runner_cached(
    qwen_image_edit_model_path: str,
    qwen_image_edit_dtype: str,
    qwen_compile: bool,
    qwen_fp8: bool,
    wardrobe_lora_top_path: str,
    wardrobe_lora_bottom_path: str,
    wardrobe_lora_dress_path: str,
    tryon_enabled_specialists: str,
    tryon_lora_top_path: str,
    tryon_lora_bottom_path: str,
    tryon_lora_dress_path: str,
    tryon_lora_multi_path: str,
    tryon_lora_rank: int,
    tryon_lora_alpha: int,
    tryon_lora_scale: float,
) -> QwenDiffusersWardrobeEngine:
    settings = Settings(
        QWEN_IMAGE_EDIT_MODEL_PATH=qwen_image_edit_model_path,
        QWEN_IMAGE_EDIT_DTYPE=qwen_image_edit_dtype,
        QWEN_COMPILE=qwen_compile,
        QWEN_FP8=qwen_fp8,
        WARDROBE_LORA_TOP_PATH=wardrobe_lora_top_path,
        WARDROBE_LORA_BOTTOM_PATH=wardrobe_lora_bottom_path,
        WARDROBE_LORA_DRESS_PATH=wardrobe_lora_dress_path,
        TRYON_ENABLED_SPECIALISTS=tryon_enabled_specialists,
        TRYON_LORA_TOP_PATH=tryon_lora_top_path,
        TRYON_LORA_BOTTOM_PATH=tryon_lora_bottom_path,
        TRYON_LORA_DRESS_PATH=tryon_lora_dress_path,
        TRYON_LORA_MULTI_PATH=tryon_lora_multi_path,
        TRYON_LORA_RANK=tryon_lora_rank,
        TRYON_LORA_ALPHA=tryon_lora_alpha,
        TRYON_LORA_SCALE=tryon_lora_scale,
    )
    return QwenDiffusersWardrobeEngine(settings)


def warmup_wardrobe_runtime(settings: Settings | None = None) -> None:
    """Eagerly load every model the wardrobe flow depends on, before serving requests.

    Nothing is unloaded afterwards: the Qwen base + the 3 extraction LoRAs, the fashion detector,
    and the Marqo classifier stay resident; MiniCPM connectivity (a separate vLLM server) is
    verified.
    """
    resolved_settings = settings or get_settings()
    # Load the Qwen base first so vLLM (MiniCPM) sizes its GPU allocation around it.
    get_wardrobe_runner(resolved_settings).warmup()
    get_minicpm_client().warmup()
    get_fashion_detection_client().ensure_ready()
    get_marqo_fashion_client().ensure_ready()


def get_wardrobe_runtime_status(settings: Settings | None = None) -> WardrobeRuntimeSnapshot:
    resolved_settings = settings or get_settings()
    runner = get_wardrobe_runner(resolved_settings)
    return WardrobeRuntimeSnapshot(
        runner=runner.status(),
        coordinator=get_system_execution_coordinator(resolved_settings).snapshot(),
    )

from __future__ import annotations

from functools import lru_cache

from app.clients.qwen_shared_aitk import QwenSharedAitkClient
from app.config import Settings


def get_shared_qwen_runner(settings: Settings) -> QwenSharedAitkClient:
    return _get_shared_qwen_runner_cached(
        settings.ai_toolkit_root,
        settings.qwen_image_edit_model_path,
        settings.wardrobe_lora_top_path,
        settings.wardrobe_lora_bottom_path,
        settings.wardrobe_lora_dress_path,
        settings.tryon_use_specialists,
        settings.tryon_enabled_specialists,
        settings.tryon_lora_path,
        settings.tryon_lora_top_path,
        settings.tryon_lora_bottom_path,
        settings.tryon_lora_dress_path,
        settings.tryon_lora_multi_path,
        settings.tryon_lora_rank,
        settings.tryon_lora_alpha,
        settings.tryon_lora_scale,
        settings.tryon_sampler,
        settings.tryon_guidance_rescale,
        settings.tryon_do_cfg_norm,
    )


@lru_cache(maxsize=8)
def _get_shared_qwen_runner_cached(
    ai_toolkit_root: str,
    qwen_image_edit_model_path: str,
    wardrobe_lora_top_path: str,
    wardrobe_lora_bottom_path: str,
    wardrobe_lora_dress_path: str,
    tryon_use_specialists: bool,
    tryon_enabled_specialists: str,
    tryon_lora_path: str,
    tryon_lora_top_path: str,
    tryon_lora_bottom_path: str,
    tryon_lora_dress_path: str,
    tryon_lora_multi_path: str,
    tryon_lora_rank: int,
    tryon_lora_alpha: int,
    tryon_lora_scale: float,
    tryon_sampler: str,
    tryon_guidance_rescale: float,
    tryon_do_cfg_norm: bool,
) -> QwenSharedAitkClient:
    settings = Settings(
        AI_TOOLKIT_ROOT=ai_toolkit_root,
        QWEN_IMAGE_EDIT_MODEL_PATH=qwen_image_edit_model_path,
        WARDROBE_LORA_TOP_PATH=wardrobe_lora_top_path,
        WARDROBE_LORA_BOTTOM_PATH=wardrobe_lora_bottom_path,
        WARDROBE_LORA_DRESS_PATH=wardrobe_lora_dress_path,
        TRYON_USE_SPECIALISTS=tryon_use_specialists,
        TRYON_ENABLED_SPECIALISTS=tryon_enabled_specialists,
        TRYON_LORA_PATH=tryon_lora_path,
        TRYON_LORA_TOP_PATH=tryon_lora_top_path,
        TRYON_LORA_BOTTOM_PATH=tryon_lora_bottom_path,
        TRYON_LORA_DRESS_PATH=tryon_lora_dress_path,
        TRYON_LORA_MULTI_PATH=tryon_lora_multi_path,
        TRYON_LORA_RANK=tryon_lora_rank,
        TRYON_LORA_ALPHA=tryon_lora_alpha,
        TRYON_LORA_SCALE=tryon_lora_scale,
        TRYON_SAMPLER=tryon_sampler,
        TRYON_GUIDANCE_RESCALE=tryon_guidance_rescale,
        TRYON_DO_CFG_NORM=tryon_do_cfg_norm,
    )
    return QwenSharedAitkClient(settings)

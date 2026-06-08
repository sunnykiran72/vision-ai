from __future__ import annotations

from app.clients.qwen_diffusers_engine import QwenDiffusersWardrobeEngine
from app.runtime import system_coordinator, wardrobe_runtime


def test_wardrobe_status_does_not_force_runtime_load() -> None:
    client = QwenDiffusersWardrobeEngine.__new__(QwenDiffusersWardrobeEngine)
    client._pipeline = None
    client._loaded_loras = set()

    status = client.status()

    assert status.loaded is False
    assert status.backend is None
    assert status.loras_loaded is False


def _runner_args(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "qwen_image_edit_model_path": "/model",
        "wardrobe_lora_top_path": "/top.safetensors",
        "wardrobe_lora_bottom_path": "/bottom.safetensors",
        "wardrobe_lora_dress_path": "/dress.safetensors",
        "qwen_image_edit_dtype": "bfloat16",
        "qwen_compile": False,
        "tryon_enabled_specialists": "top,bottom,dress,multi",
        "tryon_lora_top_path": "/tryon-top.safetensors",
        "tryon_lora_bottom_path": "/tryon-bottom.safetensors",
        "tryon_lora_dress_path": "/tryon-dress.safetensors",
        "tryon_lora_multi_path": "/tryon-multi.safetensors",
        "tryon_lora_rank": 64,
        "tryon_lora_alpha": 64,
        "tryon_lora_scale": 1.0,
    }
    base.update(overrides)
    return base


def test_wardrobe_runner_cache_changes_with_runtime_fields() -> None:
    wardrobe_runtime._get_wardrobe_runner_cached.cache_clear()

    runner_a = wardrobe_runtime._get_wardrobe_runner_cached(**_runner_args())
    runner_b = wardrobe_runtime._get_wardrobe_runner_cached(
        **_runner_args(wardrobe_lora_top_path="/top-v2.safetensors"),
    )
    runner_c = wardrobe_runtime._get_wardrobe_runner_cached(
        **_runner_args(wardrobe_lora_bottom_path="/bottom-v2.safetensors"),
    )

    assert runner_a is not runner_b
    assert runner_a is not runner_c


def test_system_coordinator_is_shared_singleton() -> None:
    system_coordinator._get_system_execution_coordinator_cached.cache_clear()

    coordinator_a = system_coordinator._get_system_execution_coordinator_cached(8, 30)
    coordinator_b = system_coordinator._get_system_execution_coordinator_cached(8, 30)
    coordinator_c = system_coordinator._get_system_execution_coordinator_cached(4, 30)

    assert coordinator_a is coordinator_b
    assert coordinator_a is not coordinator_c

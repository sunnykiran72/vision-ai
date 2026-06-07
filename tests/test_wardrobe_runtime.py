from __future__ import annotations

from app.clients.qwen_wardrobe_aitk import QwenWardrobeAitkClient
from app.runtime import system_coordinator, wardrobe_runtime


def test_wardrobe_status_does_not_force_runtime_load() -> None:
    client = QwenWardrobeAitkClient.__new__(QwenWardrobeAitkClient)
    client._pipeline = None
    client._network = None
    client._specialist_state_dicts = {}

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

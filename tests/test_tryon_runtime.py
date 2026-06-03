from __future__ import annotations

from pytest import MonkeyPatch

from app.clients.qwen_tryon_aitk import QwenTryonAitkClient
from app.runtime import tryon_runtime


def test_tryon_status_does_not_force_runtime_load(monkeypatch: MonkeyPatch) -> None:
    client = QwenTryonAitkClient.__new__(QwenTryonAitkClient)
    client._pipeline = None
    client._network = None
    client._loaded_checkpoint = None
    client._use_specialists = False
    client._specialist_state_dicts = {}

    def fail() -> None:
        raise AssertionError("status() should not call _ensure_ready()")

    monkeypatch.setattr(client, "_ensure_ready", fail)

    status = client.status()

    assert status.loaded is False
    assert status.backend is None
    assert status.lora_loaded is False


def _runner_args(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "ai_toolkit_root": "/toolkit",
        "qwen_image_edit_model_path": "/model",
        "tryon_use_specialists": False,
        "tryon_enabled_specialists": "top,bottom,dress,multi",
        "tryon_lora_path": "/checkpoint-a.safetensors",
        "tryon_lora_top_path": "",
        "tryon_lora_bottom_path": "",
        "tryon_lora_dress_path": "",
        "tryon_lora_multi_path": "",
        "tryon_lora_rank": 16,
        "tryon_lora_alpha": 16,
        "tryon_lora_scale": 1.0,
        "tryon_sampler": "flowmatch",
        "tryon_guidance_rescale": 0.0,
        "tryon_do_cfg_norm": False,
    }
    base.update(overrides)
    return base


def test_tryon_runner_cache_changes_with_runtime_fields() -> None:
    tryon_runtime._get_tryon_runner_cached.cache_clear()

    runner_a = tryon_runtime._get_tryon_runner_cached(**_runner_args())
    runner_b = tryon_runtime._get_tryon_runner_cached(
        **_runner_args(tryon_lora_path="/checkpoint-b.safetensors"),
    )
    runner_c = tryon_runtime._get_tryon_runner_cached(
        **_runner_args(
            tryon_use_specialists=True,
            tryon_lora_top_path="/top.safetensors",
            tryon_lora_bottom_path="/bottom.safetensors",
            tryon_lora_dress_path="/dress.safetensors",
            tryon_lora_multi_path="/multi.safetensors",
        ),
    )
    runner_d = tryon_runtime._get_tryon_runner_cached(
        **_runner_args(
            tryon_use_specialists=True,
            tryon_lora_top_path="/top-v2.safetensors",
            tryon_lora_bottom_path="/bottom.safetensors",
            tryon_lora_dress_path="/dress.safetensors",
            tryon_lora_multi_path="/multi.safetensors",
        ),
    )

    assert runner_a is not runner_b
    assert runner_a is not runner_c
    assert runner_c is not runner_d


def test_tryon_coordinator_cache_depends_only_on_queue_settings() -> None:
    tryon_runtime._get_tryon_execution_coordinator_cached.cache_clear()

    coordinator_a = tryon_runtime._get_tryon_execution_coordinator_cached(8, 30)
    coordinator_b = tryon_runtime._get_tryon_execution_coordinator_cached(8, 30)
    coordinator_c = tryon_runtime._get_tryon_execution_coordinator_cached(4, 30)

    assert coordinator_a is coordinator_b
    assert coordinator_a is not coordinator_c

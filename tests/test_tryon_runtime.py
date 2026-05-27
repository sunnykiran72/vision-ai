from __future__ import annotations

from pytest import MonkeyPatch

from app.clients.qwen_tryon_aitk import QwenTryonAitkClient
from app.runtime import tryon_runtime


def test_tryon_status_does_not_force_runtime_load(monkeypatch: MonkeyPatch) -> None:
    client = QwenTryonAitkClient.__new__(QwenTryonAitkClient)
    client._pipeline = None
    client._network = None
    client._loaded_checkpoint = None

    def fail() -> None:
        raise AssertionError("status() should not call _ensure_ready()")

    monkeypatch.setattr(client, "_ensure_ready", fail)

    status = client.status()

    assert status.loaded is False
    assert status.backend is None
    assert status.lora_loaded is False


def test_tryon_runner_cache_changes_with_runtime_fields() -> None:
    tryon_runtime._get_tryon_runner_cached.cache_clear()

    runner_a = tryon_runtime._get_tryon_runner_cached(
        "/toolkit",
        "/model",
        "/checkpoint-a.safetensors",
        64,
        64,
        1.0,
        "flowmatch",
        0.0,
        False,
        1024,
        1536,
    )
    runner_b = tryon_runtime._get_tryon_runner_cached(
        "/toolkit",
        "/model",
        "/checkpoint-b.safetensors",
        64,
        64,
        1.0,
        "flowmatch",
        0.0,
        False,
        1024,
        1536,
    )
    runner_c = tryon_runtime._get_tryon_runner_cached(
        "/toolkit",
        "/model",
        "/checkpoint-a.safetensors",
        64,
        64,
        1.0,
        "flowmatch",
        0.0,
        False,
        896,
        1344,
    )

    assert runner_a is not runner_b
    assert runner_a is not runner_c


def test_tryon_coordinator_cache_depends_only_on_queue_settings() -> None:
    tryon_runtime._get_tryon_execution_coordinator_cached.cache_clear()

    coordinator_a = tryon_runtime._get_tryon_execution_coordinator_cached(8, 30)
    coordinator_b = tryon_runtime._get_tryon_execution_coordinator_cached(8, 30)
    coordinator_c = tryon_runtime._get_tryon_execution_coordinator_cached(4, 30)

    assert coordinator_a is coordinator_b
    assert coordinator_a is not coordinator_c

from __future__ import annotations

from pytest import MonkeyPatch

from app.clients.qwen_diffusers_engine import QwenDiffusersWardrobeEngine
from app.config import Settings
from app.runtime import system_coordinator, tryon_runtime
from app.runtime.tryon_types import TryonRuntimeStatus


def test_tryon_status_does_not_force_runtime_load() -> None:
    client = QwenDiffusersWardrobeEngine.__new__(QwenDiffusersWardrobeEngine)
    client._pipeline = None
    client._loaded_loras = set()

    status = client.tryon_status()

    assert status.loaded is False
    assert status.backend is None
    assert status.lora_loaded is False


def test_tryon_runner_uses_shared_diffusers_runner(monkeypatch: MonkeyPatch) -> None:
    expected = object()

    monkeypatch.setattr(
        tryon_runtime,
        "get_wardrobe_runner",
        lambda _settings: expected,
    )

    assert tryon_runtime.get_tryon_runner(Settings()) is expected


def test_tryon_runtime_status_uses_system_coordinator(monkeypatch: MonkeyPatch) -> None:
    class FakeRunner:
        def tryon_status(self):
            return TryonRuntimeStatus(loaded=False, backend=None, lora_loaded=False)

    monkeypatch.setattr(tryon_runtime, "get_wardrobe_runner", lambda _settings: FakeRunner())
    system_coordinator._get_system_execution_coordinator_cached.cache_clear()

    snapshot = tryon_runtime.get_tryon_runtime_status(Settings())

    assert snapshot.coordinator.max_queue_size == 8

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings
from app.runtime.coordinator import CoordinatorSnapshot
from app.runtime.system_coordinator import get_system_execution_coordinator
from app.runtime.tryon_types import TryonRuntimeStatus
from app.runtime.wardrobe_runtime import get_wardrobe_runner


@dataclass(frozen=True)
class TryonRuntimeSnapshot:
    runner: TryonRuntimeStatus
    coordinator: CoordinatorSnapshot


def get_tryon_runner(settings: Settings | None = None):
    resolved_settings = settings or get_settings()
    return get_wardrobe_runner(resolved_settings)


def warmup_tryon_runtime(settings: Settings | None = None) -> None:
    resolved_settings = settings or get_settings()
    get_tryon_runner(resolved_settings).warmup_tryon()


def get_tryon_runtime_status(settings: Settings | None = None) -> TryonRuntimeSnapshot:
    resolved_settings = settings or get_settings()
    runner = get_tryon_runner(resolved_settings)
    return TryonRuntimeSnapshot(
        runner=runner.tryon_status(),
        coordinator=get_system_execution_coordinator(resolved_settings).snapshot(),
    )

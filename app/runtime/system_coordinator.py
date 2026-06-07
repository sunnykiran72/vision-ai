from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.config import Settings, get_settings
from app.runtime.coordinator import BoundedExecutionCoordinator


def get_system_execution_coordinator(
    settings: Settings | None = None,
) -> BoundedExecutionCoordinator[Any]:
    """Single process-wide GPU execution queue.

    Wardrobe and try-on run the same heavy Qwen weights on one GPU, so all GPU-backed work is
    serialized through one coordinator rather than per-feature queues.
    """
    resolved_settings = settings or get_settings()
    return _get_system_execution_coordinator_cached(
        resolved_settings.system_queue_max_size,
        resolved_settings.system_queue_wait_timeout_seconds,
    )


@lru_cache(maxsize=1)
def _get_system_execution_coordinator_cached(
    max_queue_size: int,
    queue_wait_timeout_seconds: int,
) -> BoundedExecutionCoordinator[Any]:
    return BoundedExecutionCoordinator(
        max_queue_size=max_queue_size,
        queue_wait_timeout_seconds=queue_wait_timeout_seconds,
    )

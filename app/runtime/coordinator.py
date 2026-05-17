from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


class QueueFullError(RuntimeError):
    pass


class QueueTimeoutError(RuntimeError):
    pass


@dataclass(frozen=True)
class CoordinatorSnapshot:
    active_jobs: int
    waiting_jobs: int
    max_queue_size: int


class BoundedExecutionCoordinator(Generic[T]):
    def __init__(self, *, max_queue_size: int, queue_wait_timeout_seconds: int):
        self._max_queue_size = max(1, int(max_queue_size))
        self._queue_wait_timeout_seconds = max(1, int(queue_wait_timeout_seconds))
        self._execution_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._waiting_jobs = 0
        self._active_jobs = 0

    def run(self, fn: Callable[[], T]) -> T:
        deadline = time.monotonic() + float(self._queue_wait_timeout_seconds)
        with self._state_lock:
            if self._waiting_jobs >= self._max_queue_size:
                raise QueueFullError("GPU execution queue is full.")
            self._waiting_jobs += 1

        acquired = False
        try:
            remaining = max(0.0, deadline - time.monotonic())
            acquired = self._execution_lock.acquire(timeout=remaining)
            if not acquired:
                raise QueueTimeoutError("Timed out while waiting for GPU execution slot.")

            with self._state_lock:
                self._waiting_jobs -= 1
                self._active_jobs += 1

            return fn()
        finally:
            if acquired:
                with self._state_lock:
                    self._active_jobs -= 1
                self._execution_lock.release()
            else:
                with self._state_lock:
                    if self._waiting_jobs > 0:
                        self._waiting_jobs -= 1

    def snapshot(self) -> CoordinatorSnapshot:
        with self._state_lock:
            return CoordinatorSnapshot(
                active_jobs=self._active_jobs,
                waiting_jobs=self._waiting_jobs,
                max_queue_size=self._max_queue_size,
            )

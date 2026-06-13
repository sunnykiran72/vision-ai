from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

logger = logging.getLogger("glamify-ai")

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
    degraded: bool = False


class BoundedExecutionCoordinator(Generic[T]):
    """Serializes GPU work through one lock (one GPU = one op at a time) with a bounded wait queue.

    Fault isolation: exceptions release the lock in ``finally`` (a failed request returns cleanly,
    the server keeps serving). A separate watchdog handles the rarer case of a *hung* op (CUDA hang /
    deadlock) that never returns and would otherwise hold the lock forever and silently wedge the pod:
    if the active op exceeds ``max_execution_seconds`` the pod is marked ``degraded`` so ``/ready``
    drops it from the load-balancer pool (the rest of the fleet absorbs traffic; recycle the pod).
    The flag auto-clears when an op completes, so a merely-slow-but-finishing op causes no permanent
    false positive. We deliberately do NOT kill the process here.
    """

    def __init__(
        self,
        *,
        max_queue_size: int,
        queue_wait_timeout_seconds: int,
        max_execution_seconds: int = 0,
    ):
        self._max_queue_size = max(1, int(max_queue_size))
        self._queue_wait_timeout_seconds = max(1, int(queue_wait_timeout_seconds))
        # 0 disables the hang watchdog. Must exceed the longest LEGITIMATE op (real tryon+upscale
        # is ~15s; cold compile up to ~300s shouldn't reach here once readiness-gated).
        self._max_execution_seconds = max(0, int(max_execution_seconds))
        self._execution_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._waiting_jobs = 0
        self._active_jobs = 0
        self._active_since: float | None = None
        self._degraded = False
        if self._max_execution_seconds > 0:
            threading.Thread(
                target=self._watchdog_loop,
                name="gpu-exec-watchdog",
                daemon=True,
            ).start()

    def _watchdog_loop(self) -> None:
        interval = max(5, self._max_execution_seconds // 4)
        while True:
            time.sleep(interval)
            with self._state_lock:
                started = self._active_since
                already = self._degraded
            if started is None or already:
                continue
            elapsed = time.monotonic() - started
            if elapsed > self._max_execution_seconds:
                with self._state_lock:
                    # Re-check it's still the same in-flight op before flagging.
                    if self._active_since is not None and not self._degraded:
                        self._degraded = True
                        flagged = True
                    else:
                        flagged = False
                if flagged:
                    logger.critical(
                        "GPU execution watchdog: op running %.0fs > %ds limit; marking pod "
                        "DEGRADED (dropped from /ready). GPU is likely wedged; recycle this pod.",
                        elapsed,
                        self._max_execution_seconds,
                    )

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
                self._active_since = time.monotonic()

            return fn()
        finally:
            if acquired:
                with self._state_lock:
                    self._active_jobs -= 1
                    self._active_since = None
                    # Op finished -> not wedged. Clear any transient degraded flag.
                    self._degraded = False
                self._execution_lock.release()
            else:
                with self._state_lock:
                    if self._waiting_jobs > 0:
                        self._waiting_jobs -= 1

    def is_degraded(self) -> bool:
        with self._state_lock:
            return self._degraded

    def snapshot(self) -> CoordinatorSnapshot:
        with self._state_lock:
            return CoordinatorSnapshot(
                active_jobs=self._active_jobs,
                waiting_jobs=self._waiting_jobs,
                max_queue_size=self._max_queue_size,
                degraded=self._degraded,
            )

from app.runtime.coordinator import (
    BoundedExecutionCoordinator,
    QueueFullError,
)


def test_queue_full_when_waiting_limit_reached() -> None:
    coordinator: BoundedExecutionCoordinator[str] = BoundedExecutionCoordinator(
        max_queue_size=1,
        queue_wait_timeout_seconds=1,
    )

    coordinator._waiting_jobs = 1
    try:
        coordinator.run(lambda: "ok")
    except QueueFullError:
        pass
    else:
        raise AssertionError("Expected QueueFullError when waiting queue is full")


def test_snapshot_reports_limits() -> None:
    coordinator: BoundedExecutionCoordinator[str] = BoundedExecutionCoordinator(
        max_queue_size=4,
        queue_wait_timeout_seconds=15,
    )

    snapshot = coordinator.snapshot()

    assert snapshot.max_queue_size == 4
    assert snapshot.active_jobs == 0
    assert snapshot.waiting_jobs == 0

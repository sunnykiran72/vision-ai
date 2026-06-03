from __future__ import annotations

import base64
from collections.abc import Callable
from concurrent.futures import Future
from io import BytesIO
from pathlib import Path
from typing import TypeVar

from PIL import Image
from pytest import MonkeyPatch

from app.clients.marqo_fashion import MarqoClassificationResult
from app.config import Settings
from app.constants import wardrobe as wardrobe_constants
from app.models.wardrobe import WardrobeAnalyzeRequest
from app.runtime.coordinator import QueueFullError, QueueTimeoutError
from app.runtime.wardrobe_types import WardrobeRunResult
from app.services import wardrobe as wardrobe_service

T = TypeVar("T")


def test_run_wardrobe_request_returns_success_and_cleans_workspace(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = WardrobeAnalyzeRequest.model_validate(
        {"image": _build_base64_image(1600, 1200), "type": "bottom"},
    )
    calls: dict[str, object] = {}

    class FakeDetector:
        def detect(self, image: Image.Image) -> list[dict[str, object]]:
            calls["detector_size"] = image.size
            return [{"label": "pants", "score": 0.9, "bbox": [0, 0, image.width, image.height]}]

    monkeypatch.setattr(wardrobe_service, "get_fashion_detection_client", lambda: FakeDetector())

    class FakeRunner:
        def run_extract(
            self,
            *,
            input_image_path: str,
            prompt: str,
            garment_type: str,
            output_path: str,
        ) -> WardrobeRunResult:
            calls["input_exists"] = Path(input_image_path).exists()
            calls["prompt"] = prompt
            calls["garment_type"] = garment_type
            calls["output_path"] = output_path
            output = Image.new(
                "RGB",
                (wardrobe_constants.OUTPUT_WIDTH, wardrobe_constants.OUTPUT_HEIGHT),
                (20, 30, 40),
            )
            output.save(output_path, format="JPEG", quality=95)
            return WardrobeRunResult(
                image=output,
                metadata={"backend": "ai_toolkit_exact"},
                wall_seconds=1.2,
            )

    monkeypatch.setattr(wardrobe_service, "get_wardrobe_runner", lambda *_args: FakeRunner())

    class ImmediateCoordinator:
        def run(self, fn: Callable[[], T]) -> T:
            return fn()

    monkeypatch.setattr(
        wardrobe_service,
        "get_wardrobe_execution_coordinator",
        lambda *_args: ImmediateCoordinator(),
    )

    class FakeMarqo:
        def classify(self, *, image: Image.Image, garment_type: str):
            calls["marqo_size"] = image.size
            calls["marqo_type"] = garment_type
            return MarqoClassificationResult(
                applied=True,
                category_key="trousers",
                category_label="trousers",
                score=0.88,
                min_confidence=0.25,
                top_matches=[],
                reason="applied",
            )

    monkeypatch.setattr(wardrobe_service, "get_marqo_fashion_client", lambda: FakeMarqo())

    sync_calls: list[dict[str, object]] = []

    class FakeProgressClient:
        def __init__(self, _settings: Settings):
            pass

        def upload_input_background(self, **kwargs: object) -> Future[str]:
            sync_calls.append({"stage": "input", **kwargs})
            future: Future[str] = Future()
            future.set_result("https://example.com/input.jpg")
            return future

        def submit_output_and_progress_background(self, **kwargs: object) -> None:
            sync_calls.append(kwargs)

    monkeypatch.setattr(wardrobe_service, "GlamifyProgressClient", FakeProgressClient)

    settings = Settings(
        WARDROBE_WORK_ROOT=str(tmp_path),
        WARDROBE_STORAGE_PREFIX="wardrobe_output/wardrobe",
    )
    response = wardrobe_service.run_wardrobe_request(
        payload,
        settings=settings,
        user_id="user-123",
        bearer_token="Bearer token",
    )

    assert response.status == 200
    assert response.data is not None
    assert response.data.type == "bottom"
    assert response.data.category == "trousers"
    assert response.data.category_label == "Trousers"
    assert Image.open(BytesIO(base64.b64decode(response.data.image))).format == "JPEG"
    assert calls["detector_size"] == (1024, 768)
    assert calls["prompt"] == wardrobe_constants.PROMPT_BY_TYPE["bottom"]
    assert calls["garment_type"] == "bottom"
    assert calls["marqo_size"] == (
        wardrobe_constants.OUTPUT_WIDTH,
        wardrobe_constants.OUTPUT_HEIGHT,
    )
    assert sync_calls[0]["stage"] == "input"
    assert sync_calls[0]["object_name"] == (
        f"wardrobe_output/wardrobe/user-123/{response.data.id}/input.jpg"
    )
    assert sync_calls[1]["bearer_token"] == "Bearer token"
    assert sync_calls[1]["progress_id"] == response.data.id
    assert sync_calls[1]["output_object_name"] == (
        f"wardrobe_output/wardrobe/user-123/{response.data.id}/output.jpg"
    )
    assert sync_calls[1]["classification"] == {
        "primary_category": "bottoms",
        "category": "trousers",
        "category_label": "Trousers",
        "score": 0.88,
    }
    assert not any(tmp_path.iterdir())


def test_wardrobe_detector_no_hit_returns_400(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    payload = WardrobeAnalyzeRequest.model_validate(
        {"image": _build_base64_image(512, 512), "type": "top"},
    )

    class FakeDetector:
        def detect(self, image: Image.Image) -> list[dict[str, object]]:
            return []

    monkeypatch.setattr(wardrobe_service, "get_fashion_detection_client", lambda: FakeDetector())

    response = wardrobe_service.run_wardrobe_request(
        payload,
        settings=Settings(WARDROBE_WORK_ROOT=str(tmp_path)),
        user_id="user-123",
        bearer_token="Bearer token",
    )

    assert response.status == 400
    assert response.data is None


def test_wardrobe_marqo_low_confidence_returns_400(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = WardrobeAnalyzeRequest.model_validate(
        {"image": _build_base64_image(512, 512), "type": "dress"},
    )

    class FakeDetector:
        def detect(self, image: Image.Image) -> list[dict[str, object]]:
            return [{"label": "dress", "score": 0.9}]

    monkeypatch.setattr(wardrobe_service, "get_fashion_detection_client", lambda: FakeDetector())

    class FakeRunner:
        def run_extract(self, **kwargs: object) -> WardrobeRunResult:
            output_path = Path(str(kwargs["output_path"]))
            output = Image.new("RGB", (832, 1248), (20, 30, 40))
            output.save(output_path, format="JPEG")
            return WardrobeRunResult(image=output, metadata={}, wall_seconds=0.1)

    monkeypatch.setattr(wardrobe_service, "get_wardrobe_runner", lambda *_args: FakeRunner())

    class ImmediateCoordinator:
        def run(self, fn: Callable[[], T]) -> T:
            return fn()

    monkeypatch.setattr(
        wardrobe_service,
        "get_wardrobe_execution_coordinator",
        lambda *_args: ImmediateCoordinator(),
    )

    class FakeMarqo:
        def classify(self, *, image: Image.Image, garment_type: str):
            return MarqoClassificationResult(
                applied=False,
                category_key="",
                category_label="",
                score=0.1,
                min_confidence=0.25,
                top_matches=[],
                reason="below_threshold",
            )

    monkeypatch.setattr(wardrobe_service, "get_marqo_fashion_client", lambda: FakeMarqo())

    response = wardrobe_service.run_wardrobe_request(
        payload,
        settings=Settings(WARDROBE_WORK_ROOT=str(tmp_path)),
        user_id="user-123",
        bearer_token="Bearer token",
    )

    assert response.status == 400
    assert response.data is None
    assert not any(tmp_path.iterdir())


def test_wardrobe_rejects_invalid_base64() -> None:
    payload = WardrobeAnalyzeRequest.model_validate({"image": "not base64", "type": "top"})
    response = wardrobe_service.run_wardrobe_request(
        payload,
        settings=Settings(),
        user_id="user-123",
        bearer_token="Bearer token",
    )

    assert response.status == 422
    assert response.data is None


def test_wardrobe_rejects_tiny_image() -> None:
    payload = WardrobeAnalyzeRequest.model_validate(
        {"image": _build_base64_image(100, 350), "type": "top"},
    )
    response = wardrobe_service.run_wardrobe_request(
        payload,
        settings=Settings(),
        user_id="user-123",
        bearer_token="Bearer token",
    )

    assert response.status == 422
    assert response.data is None


def test_wardrobe_queue_full_returns_503(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    payload = WardrobeAnalyzeRequest.model_validate(
        {"image": _build_base64_image(512, 512), "type": "top"},
    )

    class FakeDetector:
        def detect(self, image: Image.Image) -> list[dict[str, object]]:
            return [{"label": "shirt", "score": 0.9}]

    class FullCoordinator:
        def run(self, fn: Callable[[], T]) -> T:
            raise QueueFullError("GPU execution queue is full.")

    monkeypatch.setattr(wardrobe_service, "get_fashion_detection_client", lambda: FakeDetector())
    monkeypatch.setattr(
        wardrobe_service,
        "get_wardrobe_execution_coordinator",
        lambda *_args: FullCoordinator(),
    )

    response = wardrobe_service.run_wardrobe_request(
        payload,
        settings=Settings(WARDROBE_WORK_ROOT=str(tmp_path)),
        user_id="user-123",
        bearer_token="Bearer token",
    )

    assert response.status == 503
    assert response.data is None


def test_wardrobe_queue_timeout_returns_503(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    payload = WardrobeAnalyzeRequest.model_validate(
        {"image": _build_base64_image(512, 512), "type": "top"},
    )

    class FakeDetector:
        def detect(self, image: Image.Image) -> list[dict[str, object]]:
            return [{"label": "shirt", "score": 0.9}]

    class TimeoutCoordinator:
        def run(self, fn: Callable[[], T]) -> T:
            raise QueueTimeoutError("Timed out while waiting for GPU execution slot.")

    monkeypatch.setattr(wardrobe_service, "get_fashion_detection_client", lambda: FakeDetector())
    monkeypatch.setattr(
        wardrobe_service,
        "get_wardrobe_execution_coordinator",
        lambda *_args: TimeoutCoordinator(),
    )

    response = wardrobe_service.run_wardrobe_request(
        payload,
        settings=Settings(WARDROBE_WORK_ROOT=str(tmp_path)),
        user_id="user-123",
        bearer_token="Bearer token",
    )

    assert response.status == 503
    assert response.data is None


def _build_base64_image(width: int, height: int, *, fmt: str = "PNG") -> str:
    buffer = BytesIO()
    Image.new("RGB", (width, height), (220, 110, 60)).save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")

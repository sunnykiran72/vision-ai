from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from io import BytesIO
from types import SimpleNamespace
from typing import TypeVar

from PIL import Image
from pytest import MonkeyPatch

from app.clients.glamify_progress import TimedUploadResult
from app.clients.marqo_fashion import MarqoClassificationResult
from app.config import Settings
from app.constants import wardrobe as wardrobe_constants
from app.models.wardrobe import WardrobeGarmentType
from app.runtime.coordinator import QueueFullError, QueueTimeoutError
from app.runtime.wardrobe_types import WardrobeRunResult
from app.services import wardrobe as wardrobe_service

T = TypeVar("T")

CAPTION = "a structured navy tailored garment with concrete factual details."


class ImmediateCoordinator:
    def run(self, fn: Callable[[], T]) -> T:
        return fn()


class FakeStorage:
    def __init__(self, _settings: Settings, *, configured: bool = True) -> None:
        self._configured = configured

    @property
    def is_configured(self) -> bool:
        return self._configured


class FakeProgressClient:
    def __init__(self, _settings: Settings) -> None:
        self.uploads: list[dict[str, object]] = []
        self.progress: list[dict[str, object]] = []

    @property
    def is_configured(self) -> bool:
        return True

    def upload_background(
        self,
        *,
        content: bytes,
        object_name: str,
        container: str,
        content_type: str,
    ) -> Future[str]:
        self.uploads.append({"object_name": object_name, "container": container})
        future: Future[str] = Future()
        future.set_result(f"https://blob.example.com/{container}/{object_name}")
        return future

    def upload_background_timed(
        self,
        *,
        content: bytes,
        object_name: str,
        container: str,
        content_type: str,
    ) -> Future[TimedUploadResult]:
        self.uploads.append({"object_name": object_name, "container": container})
        future: Future[TimedUploadResult] = Future()
        future.set_result(
            TimedUploadResult(
                url=f"https://blob.example.com/{container}/{object_name}",
                wall_seconds=0.123,
                container=container,
                object_name=object_name,
                bytes=len(content),
            ),
        )
        return future

    def submit_progress_background(self, **kwargs: object) -> None:
        self.progress.append(kwargs)


def _fake_minicpm() -> object:
    class FakeMiniCPM:
        def describe_garment(self, *, image: Image.Image, prompt: str) -> object:
            return SimpleNamespace(text=CAPTION)

    return FakeMiniCPM()


def _patch_common(
    monkeypatch: MonkeyPatch,
    *,
    storage_configured: bool = True,
) -> FakeProgressClient:
    progress = FakeProgressClient(Settings())
    monkeypatch.setattr(wardrobe_service, "get_minicpm_client", _fake_minicpm)
    monkeypatch.setattr(
        wardrobe_service,
        "get_system_execution_coordinator",
        lambda *_args: ImmediateCoordinator(),
    )
    monkeypatch.setattr(
        wardrobe_service,
        "AzureStorageClient",
        lambda settings: FakeStorage(settings, configured=storage_configured),
    )
    monkeypatch.setattr(wardrobe_service, "GlamifyProgressClient", lambda _settings: progress)
    return progress


def _output_runner(calls: dict[str, object] | None = None) -> object:
    class FakeRunner:
        def run_extract(
            self,
            *,
            input_image: Image.Image,
            prompt: str,
            garment_type: str,
        ) -> WardrobeRunResult:
            if calls is not None:
                calls["input_size"] = input_image.size
                calls["prompt"] = prompt
                calls["garment_type"] = garment_type
            output = Image.new(
                "RGB",
                (wardrobe_constants.OUTPUT_WIDTH, wardrobe_constants.OUTPUT_HEIGHT),
                (20, 30, 40),
            )
            return WardrobeRunResult(
                image=output,
                metadata={"dtype": "bfloat16", "seed": wardrobe_constants.GENERATION_SEED},
                wall_seconds=1.2,
            )

    return FakeRunner()


def test_run_wardrobe_request_success(monkeypatch: MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeDetector:
        def detect(self, image: Image.Image) -> list[dict[str, object]]:
            calls["detector_size"] = image.size
            return [{"label": "pants", "score": 0.9}]

    monkeypatch.setattr(wardrobe_service, "get_fashion_detection_client", lambda: FakeDetector())
    monkeypatch.setattr(wardrobe_service, "get_wardrobe_runner", lambda *_a: _output_runner(calls))

    class FakeMarqo:
        def classify(self, *, image: Image.Image, garment_type: str) -> MarqoClassificationResult:
            calls["marqo_size"] = image.size
            return MarqoClassificationResult(
                applied=True,
                category_key="trousers",
                category_label="trousers",
                score=0.88,
                min_confidence=0.20,
                top_matches=[],
                reason="applied",
            )

    monkeypatch.setattr(wardrobe_service, "get_marqo_fashion_client", lambda: FakeMarqo())
    progress = _patch_common(monkeypatch)

    response = wardrobe_service.run_wardrobe_request(
        _image_bytes(1600, 1200),
        garment_type=WardrobeGarmentType.BOTTOM,
        settings=Settings(
            AZURE_WARDROBE_INPUT_CONTAINER="wardrobe-inputs",
            AZURE_WARDROBE_OUTPUT_CONTAINER="wardrobe-outputs",
        ),
        user_id="user-123",
        access_token="jwt-token",
    )

    assert response.status == 200
    assert response.data is not None
    job_id = response.data.id
    output_object = f"user-123/{job_id}/output.jpg"
    expected_prompt = wardrobe_constants.QWEN_EXTRACT_PROMPT_TEMPLATE_BY_TYPE["bottom"].format(
        caption=CAPTION,
    )
    assert response.data.type == "bottom"
    assert response.data.category == "trousers"
    assert response.data.image == f"https://blob.example.com/wardrobe-outputs/{output_object}"
    assert response.data.metadata["promptDescription"] == CAPTION
    assert response.data.metadata["prompt"] == expected_prompt
    assert response.data.metadata["runtime"]["dtype"] == "bfloat16"
    assert response.data.metadata["timings"]["qwen_generation_seconds"] == 1.2
    assert response.data.metadata["uploads"]["input"]["wall_seconds"] == 0.123
    assert response.data.metadata["uploads"]["output"]["container"] == "wardrobe-outputs"
    assert response.data.metadata["progress"]["payload"]["promptDescription"] == CAPTION
    assert (
        response.data.metadata["progress"]["payload"]["metadata"]["classification"]["category"]
        == "trousers"
    )
    assert calls["detector_size"] == (1024, 768)
    assert calls["prompt"] == expected_prompt
    assert calls["marqo_size"] == (
        wardrobe_constants.OUTPUT_WIDTH,
        wardrobe_constants.OUTPUT_HEIGHT,
    )
    # input -> wardrobe-inputs (background), output -> wardrobe-outputs
    assert progress.uploads[0] == {
        "object_name": f"user-123/{job_id}/input.jpg",
        "container": "wardrobe-inputs",
    }
    assert progress.uploads[1] == {
        "object_name": output_object,
        "container": "wardrobe-outputs",
    }
    sync = progress.progress[0]
    assert sync["access_token"] == "jwt-token"
    assert sync["progress_id"] == job_id
    assert sync["prompt_description"] == CAPTION
    assert sync["output_url"] == f"https://blob.example.com/wardrobe-outputs/{output_object}"
    assert sync["classification"]["category"] == "trousers"


def test_wardrobe_detector_no_hit_returns_400(monkeypatch: MonkeyPatch) -> None:
    class FakeDetector:
        def detect(self, image: Image.Image) -> list[dict[str, object]]:
            return []

    monkeypatch.setattr(wardrobe_service, "get_fashion_detection_client", lambda: FakeDetector())
    _patch_common(monkeypatch)

    response = wardrobe_service.run_wardrobe_request(
        _image_bytes(512, 512),
        garment_type=WardrobeGarmentType.TOP,
        settings=Settings(),
        user_id="user-123",
        access_token="jwt-token",
    )

    assert response.status == 400
    assert response.data is None
    assert "No garment was detected" in response.message


def test_wardrobe_requires_configured_storage(monkeypatch: MonkeyPatch) -> None:
    _patch_common(monkeypatch, storage_configured=False)

    response = wardrobe_service.run_wardrobe_request(
        _image_bytes(512, 512),
        garment_type=WardrobeGarmentType.TOP,
        settings=Settings(),
        user_id="user-123",
        access_token="jwt-token",
    )

    assert response.status == 500
    assert response.data is None


def test_wardrobe_marqo_low_confidence_uses_default_category(monkeypatch: MonkeyPatch) -> None:
    class FakeDetector:
        def detect(self, image: Image.Image) -> list[dict[str, object]]:
            return [{"label": "dress", "score": 0.9}]

    monkeypatch.setattr(wardrobe_service, "get_fashion_detection_client", lambda: FakeDetector())
    monkeypatch.setattr(wardrobe_service, "get_wardrobe_runner", lambda *_a: _output_runner())

    class FakeMarqo:
        def classify(self, *, image: Image.Image, garment_type: str) -> MarqoClassificationResult:
            return MarqoClassificationResult(
                applied=False,
                category_key="",
                category_label="",
                score=0.1,
                min_confidence=0.20,
                top_matches=[],
                reason="below_threshold",
            )

    monkeypatch.setattr(wardrobe_service, "get_marqo_fashion_client", lambda: FakeMarqo())
    _patch_common(monkeypatch)

    response = wardrobe_service.run_wardrobe_request(
        _image_bytes(512, 512),
        garment_type=WardrobeGarmentType.DRESS,
        settings=Settings(),
        user_id="user-123",
        access_token="jwt-token",
    )

    assert response.status == 200
    assert response.data is not None
    assert response.data.category == "day_dresses"
    assert response.data.category_label == "Day Dresses"


def test_wardrobe_marqo_low_confidence_ranked_match_is_returned(monkeypatch: MonkeyPatch) -> None:
    class FakeDetector:
        def detect(self, image: Image.Image) -> list[dict[str, object]]:
            return [{"label": "pants", "score": 0.9}]

    monkeypatch.setattr(wardrobe_service, "get_fashion_detection_client", lambda: FakeDetector())
    monkeypatch.setattr(wardrobe_service, "get_wardrobe_runner", lambda *_a: _output_runner())

    class FakeMarqo:
        def classify(self, *, image: Image.Image, garment_type: str) -> MarqoClassificationResult:
            return MarqoClassificationResult(
                applied=False,
                category_key="trousers",
                category_label="Trousers",
                score=0.12,
                min_confidence=0.20,
                top_matches=[{"category_key": "trousers", "score": 0.12}],
                reason="below_threshold",
            )

    monkeypatch.setattr(wardrobe_service, "get_marqo_fashion_client", lambda: FakeMarqo())
    progress = _patch_common(monkeypatch)

    response = wardrobe_service.run_wardrobe_request(
        _image_bytes(512, 512),
        garment_type=WardrobeGarmentType.BOTTOM,
        settings=Settings(),
        user_id="user-123",
        access_token="jwt-token",
    )

    assert response.status == 200
    assert response.data is not None
    assert response.data.category == "trousers"
    assert progress.progress[0]["classification"]["source"] == "marqo_low_confidence"


def test_wardrobe_rejects_invalid_image(monkeypatch: MonkeyPatch) -> None:
    _patch_common(monkeypatch)
    response = wardrobe_service.run_wardrobe_request(
        b"not an image",
        garment_type=WardrobeGarmentType.TOP,
        settings=Settings(),
        user_id="user-123",
        access_token="jwt-token",
    )

    assert response.status == 422
    assert response.data is None
    assert "not a valid image" in response.message


def test_wardrobe_rejects_tiny_image(monkeypatch: MonkeyPatch) -> None:
    _patch_common(monkeypatch)
    response = wardrobe_service.run_wardrobe_request(
        _image_bytes(100, 350),
        garment_type=WardrobeGarmentType.TOP,
        settings=Settings(),
        user_id="user-123",
        access_token="jwt-token",
    )

    assert response.status == 422
    assert response.data is None
    assert "too small" in response.message


def test_wardrobe_queue_full_returns_503(monkeypatch: MonkeyPatch) -> None:
    class FakeDetector:
        def detect(self, image: Image.Image) -> list[dict[str, object]]:
            return [{"label": "shirt", "score": 0.9}]

    class FullCoordinator:
        def run(self, fn: Callable[[], T]) -> T:
            raise QueueFullError("GPU execution queue is full.")

    monkeypatch.setattr(wardrobe_service, "get_fashion_detection_client", lambda: FakeDetector())
    monkeypatch.setattr(wardrobe_service, "AzureStorageClient", lambda s: FakeStorage(s))
    monkeypatch.setattr(
        wardrobe_service,
        "get_system_execution_coordinator",
        lambda *_a: FullCoordinator(),
    )

    response = wardrobe_service.run_wardrobe_request(
        _image_bytes(512, 512),
        garment_type=WardrobeGarmentType.TOP,
        settings=Settings(),
        user_id="user-123",
        access_token="jwt-token",
    )

    assert response.status == 503
    assert response.data is None


def test_wardrobe_queue_timeout_returns_503(monkeypatch: MonkeyPatch) -> None:
    class FakeDetector:
        def detect(self, image: Image.Image) -> list[dict[str, object]]:
            return [{"label": "shirt", "score": 0.9}]

    class TimeoutCoordinator:
        def run(self, fn: Callable[[], T]) -> T:
            raise QueueTimeoutError("Timed out while waiting for GPU execution slot.")

    monkeypatch.setattr(wardrobe_service, "get_fashion_detection_client", lambda: FakeDetector())
    monkeypatch.setattr(wardrobe_service, "AzureStorageClient", lambda s: FakeStorage(s))
    monkeypatch.setattr(
        wardrobe_service,
        "get_system_execution_coordinator",
        lambda *_a: TimeoutCoordinator(),
    )

    response = wardrobe_service.run_wardrobe_request(
        _image_bytes(512, 512),
        garment_type=WardrobeGarmentType.TOP,
        settings=Settings(),
        user_id="user-123",
        access_token="jwt-token",
    )

    assert response.status == 503
    assert response.data is None


def _image_bytes(width: int, height: int, *, fmt: str = "PNG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), (220, 110, 60)).save(buffer, format=fmt)
    return buffer.getvalue()

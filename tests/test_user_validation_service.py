from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from typing import TypeVar

from PIL import Image
from pytest import MonkeyPatch

from app.config import Settings
from app.constants import user_validation as constants
from app.services import user_validation as service

T = TypeVar("T")


class ImmediateCoordinator:
    def run(self, fn: Callable[[], T]) -> T:
        return fn()


class FakePersonDetector:
    def __init__(self, detections: list[dict[str, object]]) -> None:
        self.detections = detections
        self.calls: list[dict[str, object]] = []
        self.model_id = constants.PERSON_DETECTION_MODEL_ID
        self.device = "cpu"
        self.dtype = "float32"

    def detect(self, image: Image.Image) -> list[dict[str, object]]:
        self.calls.append({"size": image.size})
        return self.detections


class FakeStorage:
    def __init__(self, _settings: Settings, *, configured: bool = True) -> None:
        self._configured = configured
        self.uploads: list[dict[str, object]] = []

    @property
    def is_configured(self) -> bool:
        return self._configured

    def upload_bytes(
        self,
        content: bytes,
        *,
        object_name: str,
        content_type: str | None = None,
        container: str | None = None,
    ) -> str:
        self.uploads.append(
            {
                "bytes": len(content),
                "object_name": object_name,
                "content_type": content_type,
                "container": container,
            },
        )
        return f"https://blob.example.com/{container}/{object_name}"


def test_user_validation_success_resizes_validates_and_uploads(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_detector = FakePersonDetector([_person_detection()])
    fake_storage = FakeStorage(Settings())
    _patch_common(monkeypatch, fake_detector=fake_detector, fake_storage=fake_storage)

    response = service.run_user_validation_request(
        _image_bytes(900, 1200),
        filename="person.jpg",
        content_type="image/jpeg",
        user_id="user-123",
        settings=Settings(AZURE_USER_IMAGE_CONTAINER="user-images"),
    )

    assert response.status == 200
    assert response.data is not None
    assert response.data.image.startswith("https://blob.example.com/user-images/inputs/")
    assert fake_detector.calls[0]["size"] == (936, 1248)
    assert fake_storage.uploads[0]["content_type"] == "image/jpeg"
    assert fake_storage.uploads[0]["container"] == "user-images"
    assert fake_storage.uploads[0]["object_name"].startswith("inputs/user-123/")
    metadata = response.data.metadata
    assert metadata["validation"]["accepted"] is True
    assert metadata["validation"]["thresholds"]["height_ratio"] == 0.35
    assert metadata["person_detection"]["model"] == constants.PERSON_DETECTION_MODEL_ID
    assert metadata["blur"]["score"] == 100.0
    assert metadata["sizes"]["input"] == {"width": 900, "height": 1200}
    assert metadata["sizes"]["normalized"] == {"width": 936, "height": 1248}
    assert metadata["timings"]["upload_seconds"] >= 0


def test_user_validation_rejects_detection_failure(monkeypatch: MonkeyPatch) -> None:
    fake_detector = FakePersonDetector([_person_detection(height_ratio=0.2, bottom_ratio=0.3)])
    fake_storage = FakeStorage(Settings())
    _patch_common(monkeypatch, fake_detector=fake_detector, fake_storage=fake_storage)

    response = service.run_user_validation_request(
        _image_bytes(900, 1200),
        filename="person.jpg",
        content_type="image/jpeg",
        user_id="user-123",
        settings=Settings(),
    )

    assert response.status == 422
    assert response.data is None
    assert response.message == "The person is too small vertically in the image."
    assert fake_storage.uploads == []


def test_user_validation_rejects_extreme_aspect_ratio(monkeypatch: MonkeyPatch) -> None:
    fake_detector = FakePersonDetector([_person_detection()])
    fake_storage = FakeStorage(Settings())
    _patch_common(monkeypatch, fake_detector=fake_detector, fake_storage=fake_storage)

    response = service.run_user_validation_request(
        _image_bytes(2400, 400),
        filename="wide.jpg",
        content_type="image/jpeg",
        user_id="user-123",
        settings=Settings(),
    )

    assert response.status == 422
    assert response.data is None
    assert "aspect ratio" in str(response.message)
    assert fake_detector.calls == []
    assert fake_storage.uploads == []


def test_user_validation_rejects_invalid_image(monkeypatch: MonkeyPatch) -> None:
    fake_detector = FakePersonDetector([_person_detection()])
    fake_storage = FakeStorage(Settings())
    _patch_common(monkeypatch, fake_detector=fake_detector, fake_storage=fake_storage)

    response = service.run_user_validation_request(
        b"not an image",
        filename="bad.txt",
        content_type="text/plain",
        user_id="user-123",
        settings=Settings(),
    )

    assert response.status == 422
    assert response.data is None
    assert "Unsupported image format" in str(response.message)


def test_user_validation_requires_configured_storage(monkeypatch: MonkeyPatch) -> None:
    fake_detector = FakePersonDetector([_person_detection()])
    fake_storage = FakeStorage(Settings(), configured=False)
    _patch_common(monkeypatch, fake_detector=fake_detector, fake_storage=fake_storage)

    response = service.run_user_validation_request(
        _image_bytes(900, 1200),
        filename="person.jpg",
        content_type="image/jpeg",
        user_id="user-123",
        settings=Settings(),
    )

    assert response.status == 500
    assert response.data is None
    assert "Azure storage" in str(response.message)


def _patch_common(
    monkeypatch: MonkeyPatch,
    *,
    fake_detector: FakePersonDetector,
    fake_storage: FakeStorage,
) -> None:
    monkeypatch.setattr(service, "get_person_detection_client", lambda: fake_detector)
    monkeypatch.setattr(
        service,
        "_compute_blur_metadata",
        lambda _image: {
            "method": "opencv_laplacian_variance",
            "score": 100.0,
            "threshold": constants.BLUR_SCORE_THRESHOLD,
        },
    )
    monkeypatch.setattr(
        service,
        "get_system_execution_coordinator",
        lambda *_args: ImmediateCoordinator(),
    )
    monkeypatch.setattr(service, "AzureStorageClient", lambda _settings: fake_storage)


def _image_bytes(width: int, height: int, *, fmt: str = "JPEG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), (80, 120, 180)).save(buffer, format=fmt)
    return buffer.getvalue()


def _person_detection(
    *,
    score: float = 0.96,
    height_ratio: float = 0.8,
    area_ratio: float = 0.4,
    bottom_ratio: float = 0.9,
) -> dict[str, object]:
    return {
        "label": "person",
        "score": score,
        "box": {"x1": 10, "y1": 20, "x2": 500, "y2": 1000},
        "metrics": {
            "width_ratio": 0.5,
            "height_ratio": height_ratio,
            "area_ratio": area_ratio,
            "top_ratio": 0.1,
            "bottom_ratio": bottom_ratio,
        },
    }

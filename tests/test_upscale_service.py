from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import TypeVar

from PIL import Image
from pytest import MonkeyPatch

from app.clients.seedvr2 import SeedVR2RunResult
from app.config import Settings
from app.models.upscale import UpscaleRequest
from app.services import upscale as upscale_service
from app.utils.media_utils import DownloadedMedia

T = TypeVar("T")


def test_run_upscale_request_returns_success_without_storage(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_bytes = _build_png_bytes(512, 512)

    monkeypatch.setattr(
        upscale_service,
        "download_media_from_url",
        lambda _url: DownloadedMedia(
            content=image_bytes,
            content_type="image/png",
            source_url="https://example.com/image.png",
            filename="image.png",
        ),
    )

    class FakeSeedVR2Client:
        def run(
            self,
            *,
            input_path: Path,
            output_path: Path,
            log_path: Path,
            target_long_edge: int,
        ) -> SeedVR2RunResult:
            with Image.open(input_path) as image:
                image.resize((2048, 2048)).save(output_path, format="JPEG", quality=95)
            log_path.write_text("ok", encoding="utf-8")
            return SeedVR2RunResult(
                output_path=output_path,
                output_width=2048,
                output_height=2048,
                wall_seconds=1.25,
                target_long_edge=target_long_edge,
                derived_short_edge=2048,
                model_variant="seedvr2-test.safetensors",
                log_path=log_path,
                runner_backend="cuda",
            )

    monkeypatch.setattr(
        upscale_service,
        "get_upscale_runner",
        lambda *_args: FakeSeedVR2Client(),
    )

    class ImmediateCoordinator:
        def run(self, fn: Callable[[], T]) -> T:
            return fn()

    monkeypatch.setattr(
        upscale_service,
        "get_upscale_execution_coordinator",
        lambda *_args: ImmediateCoordinator(),
    )

    settings = Settings(
        UPSCALE_WORK_ROOT=str(tmp_path),
        UPSCALE_MODEL_VARIANT="seedvr2-test.safetensors",
    )
    payload = UpscaleRequest.model_validate(
        {
            "image_url": "https://example.com/image.png",
            "metric": "2k",
            "output_file_name": "result.jpg",
        },
    )

    response = upscale_service.run_upscale_request(payload, settings=settings)

    assert response.status == 200
    assert response.data.url is None
    assert response.data.metadata["feature"] == "upscale"
    assert response.data.metadata["action"] == "upscaled"
    assert response.data.metadata["output"]["width"] == 2048


def _build_png_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

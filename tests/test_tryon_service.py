from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import TypeVar

from PIL import Image
from pytest import MonkeyPatch

from app.clients.qwen_image_edit import QwenImageEditRunResult
from app.config import Settings
from app.models.tryon import TryonRequest
from app.services import tryon as tryon_service
from app.utils.media_utils import DownloadedMedia

T = TypeVar("T")


def test_run_tryon_request_returns_success_and_cleans_workspace(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_bytes = _build_png_bytes(512, 768)

    monkeypatch.setattr(
        tryon_service,
        "download_media_from_url",
        lambda url: DownloadedMedia(
            content=image_bytes,
            content_type="image/png",
            source_url=url,
            filename="image.png",
        ),
    )

    class FakeQwenClient:
        def run_tryon(
            self,
            *,
            garment_reference_image: Image.Image,
            user_image: Image.Image,
            prompt: str,
            steps: int,
            guidance_scale: float,
            seed: int,
            output_width: int | None = None,
            output_height: int | None = None,
        ) -> QwenImageEditRunResult:
            del prompt, steps, guidance_scale, seed, output_width, output_height, user_image
            output = garment_reference_image.resize((1024, 1536))
            return QwenImageEditRunResult(
                image=output,
                metadata={
                    "backend": "cuda",
                    "lora_loaded": True,
                    "input_mode": "separate_garment_and_user_images",
                },
                wall_seconds=1.4,
            )

    monkeypatch.setattr(tryon_service, "get_tryon_runner", lambda *_args: FakeQwenClient())

    class ImmediateCoordinator:
        def run(self, fn: Callable[[], T]) -> T:
            return fn()

    monkeypatch.setattr(
        tryon_service,
        "get_tryon_execution_coordinator",
        lambda *_args: ImmediateCoordinator(),
    )

    class FakeStorageClient:
        def __init__(self, _settings: Settings):
            self.is_configured = True

        def upload_file(
            self,
            file_path: Path,
            *,
            object_name: str,
            content_type: str | None = None,
        ) -> str:
            assert file_path.exists()
            assert object_name.endswith("/output.jpg")
            assert content_type == "image/jpeg"
            return f"https://example.com/{object_name}"

    monkeypatch.setattr(tryon_service, "AzureStorageClient", FakeStorageClient)

    settings = Settings(
        TRYON_WORK_ROOT=str(tmp_path),
        TRYON_LORA_PATH="/workspace/models/lora/tryon",
        TRYON_LORA_WEIGHT_NAME="tryon.safetensors",
    )
    payload = TryonRequest.model_validate(
        {
            "user_image": "https://example.com/user.png",
            "products": [
                {
                    "image_url": "https://example.com/top.png",
                    "type": "top",
                    "prompt": "red structured jacket",
                }
            ],
        },
    )

    response = tryon_service.run_tryon_request(payload, settings=settings, user_id="user-123")

    assert response.status == 200
    assert response.data.url is not None
    assert response.data.metadata["feature"] == "tryon"
    assert response.data.metadata["reference"]["product_reference_mode"] == "single_product"
    assert response.data.metadata["reference"]["input_mode"] == "separate_garment_and_user_images"
    assert response.data.metadata["resolved_settings"]["seed"] == 44
    assert response.data.metadata["output"]["width"] == 1024
    assert not any(tmp_path.iterdir())


def _build_png_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

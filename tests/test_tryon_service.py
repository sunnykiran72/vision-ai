from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import TypeVar

from PIL import Image
from pytest import MonkeyPatch

from app.clients.seedvr2 import SeedVR2RunResult
from app.config import Settings
from app.models.tryon import TryonRequest
from app.runtime.tryon_types import TryonRunResult
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
            person_image: Image.Image,
            garment_reference_image: Image.Image,
            prompt: str,
            steps: int,
            guidance_scale: float,
            seed: int,
            output_width: int,
            output_height: int,
            lora_key: str | None = None,
        ) -> TryonRunResult:
            del prompt, steps, guidance_scale, seed, lora_key
            assert person_image.size == (512, 768)
            output = garment_reference_image.convert("RGB").resize(
                (output_width, output_height),
            )
            return TryonRunResult(
                image=output,
                metadata={
                    "backend": "diffusers_qwen_image_edit_plus",
                    "lora_loaded": True,
                    "control_order": {
                        "image_1": "person",
                        "image_2": "garment_reference",
                    },
                },
                wall_seconds=1.4,
            )

    monkeypatch.setattr(tryon_service, "get_tryon_runner", lambda *_args: FakeQwenClient())

    class ImmediateCoordinator:
        def run(self, fn: Callable[[], T]) -> T:
            return fn()

    monkeypatch.setattr(
        tryon_service,
        "get_system_execution_coordinator",
        lambda *_args: ImmediateCoordinator(),
    )

    class FakeStorageClient:
        def __init__(self, _settings: Settings):
            self.is_configured = True

        def upload_bytes(
            self,
            content: bytes,
            *,
            object_name: str,
            content_type: str | None = None,
        ) -> str:
            assert content
            assert "/user-123/" in object_name
            assert object_name.endswith("/output.jpg")
            assert content_type == "image/jpeg"
            return f"https://example.com/{object_name}"

    monkeypatch.setattr(tryon_service, "AzureStorageClient", FakeStorageClient)

    settings = Settings(
        QWEN_IMAGE_EDIT_MODEL_PATH="/workspace/models/qwen-image-edit-2511",
        TRYON_UPSCALE_AFTER_QWEN=False,
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
    assert response.data.metadata["reference"]["control_order"]["image_1"] == "person"
    assert response.data.metadata["reference"]["control_order"]["image_2"] == "garment_reference"
    assert response.data.metadata["resolved_settings"]["seed"] == 7777
    assert response.data.metadata["resolved_settings"]["steps"] == 12
    assert response.data.metadata["output"]["width"] == 512
    assert response.data.metadata["output"]["height"] == 768
    assert response.data.metadata["output"]["inference_width"] == 512
    assert response.data.metadata["output"]["inference_height"] == 768
    assert response.data.metadata["routing"]["lora_key"] == "top"
    assert response.data.metadata["reference"]["garment_reference_max_edge"] == 768
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_tryon_uses_person_dimensions_and_caps_garment_reference(
    monkeypatch: MonkeyPatch,
) -> None:
    user_bytes = _build_png_bytes(1024, 1536)
    garment_bytes = _build_png_bytes(1600, 1200)

    def fake_download(url: str) -> DownloadedMedia:
        content = user_bytes if "user" in url else garment_bytes
        return DownloadedMedia(
            content=content,
            content_type="image/png",
            source_url=url,
            filename="image.png",
        )

    monkeypatch.setattr(tryon_service, "download_media_from_url", fake_download)

    class FakeQwenClient:
        def run_tryon(
            self,
            *,
            person_image: Image.Image,
            garment_reference_image: Image.Image,
            prompt: str,
            steps: int,
            guidance_scale: float,
            seed: int,
            output_width: int,
            output_height: int,
            lora_key: str | None = None,
        ) -> TryonRunResult:
            del prompt, steps, guidance_scale, seed, lora_key
            assert person_image.size == (1024, 1536)
            assert max(garment_reference_image.size) == 768
            assert (output_width, output_height) == (1024, 1536)
            return TryonRunResult(
                image=Image.new("RGB", (output_width, output_height), "white"),
                metadata={},
                wall_seconds=0.2,
            )

    monkeypatch.setattr(tryon_service, "get_tryon_runner", lambda *_args: FakeQwenClient())

    class ImmediateCoordinator:
        def run(self, fn: Callable[[], T]) -> T:
            return fn()

    monkeypatch.setattr(
        tryon_service,
        "get_system_execution_coordinator",
        lambda *_args: ImmediateCoordinator(),
    )

    class FakeStorageClient:
        def __init__(self, _settings: Settings):
            self.is_configured = True

        def upload_bytes(
            self,
            content: bytes,
            *,
            object_name: str,
            content_type: str | None = None,
        ) -> str:
            del content, content_type
            return f"https://example.com/{object_name}"

    monkeypatch.setattr(tryon_service, "AzureStorageClient", FakeStorageClient)

    payload = TryonRequest.model_validate(
        {
            "user_image": "https://example.com/user.png",
            "products": [
                {
                    "image_url": "https://example.com/garment.png",
                    "type": "top",
                    "prompt": "red jacket",
                }
            ],
        },
    )

    response = tryon_service.run_tryon_request(
        payload,
        settings=Settings(TRYON_UPSCALE_AFTER_QWEN=False),
        user_id="user-123",
    )

    assert response.status == 200
    assert response.data is not None
    assert response.data.metadata["output"]["inference_width"] == 1024
    assert response.data.metadata["output"]["inference_height"] == 1536
    assert response.data.metadata["reference"]["garment_reference_size"] == {
        "width": 768,
        "height": 576,
    }


def test_tryon_inline_upscale_outputs_2048_long_edge(
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
            person_image: Image.Image,
            garment_reference_image: Image.Image,
            prompt: str,
            steps: int,
            guidance_scale: float,
            seed: int,
            output_width: int,
            output_height: int,
            lora_key: str | None = None,
        ) -> TryonRunResult:
            del person_image, garment_reference_image, prompt, steps, guidance_scale, seed, lora_key
            return TryonRunResult(
                image=Image.new("RGB", (output_width, output_height), "white"),
                metadata={"backend": "diffusers_qwen_image_edit_plus"},
                wall_seconds=1.2,
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
            assert target_long_edge == 3072
            with Image.open(input_path) as image:
                assert image.size == (512, 768)
            Image.new("RGB", (2048, 3072), "white").save(output_path, format="PNG")
            log_path.write_text("ok", encoding="utf-8")
            return SeedVR2RunResult(
                output_path=output_path,
                output_width=2048,
                output_height=3072,
                wall_seconds=3.5,
                target_long_edge=3072,
                derived_short_edge=2048,
                model_variant="seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                log_path=log_path,
                runner_backend="cuda",
            )

    monkeypatch.setattr(tryon_service, "get_tryon_runner", lambda *_args: FakeQwenClient())
    monkeypatch.setattr(tryon_service, "get_upscale_runner", lambda *_args: FakeSeedVR2Client())

    class ImmediateCoordinator:
        def run(self, fn: Callable[[], T]) -> T:
            return fn()

    monkeypatch.setattr(
        tryon_service,
        "get_system_execution_coordinator",
        lambda *_args: ImmediateCoordinator(),
    )
    monkeypatch.setattr(
        tryon_service,
        "get_upscale_execution_coordinator",
        lambda *_args: ImmediateCoordinator(),
    )

    uploaded_dimensions: list[tuple[int, int]] = []

    class FakeStorageClient:
        def __init__(self, _settings: Settings):
            self.is_configured = True

        def upload_bytes(
            self,
            content: bytes,
            *,
            object_name: str,
            content_type: str | None = None,
        ) -> str:
            del object_name, content_type
            with Image.open(BytesIO(content)) as image:
                uploaded_dimensions.append(image.size)
            return "https://example.com/output.jpg"

    monkeypatch.setattr(tryon_service, "AzureStorageClient", FakeStorageClient)

    payload = TryonRequest.model_validate(
        {
            "user_image": "https://example.com/user.png",
            "products": [
                {
                    "image_url": "https://example.com/top.png",
                    "type": "top",
                    "prompt": "red jacket",
                }
            ],
        },
    )
    settings = Settings(
        UPSCALE_WORK_ROOT=str(tmp_path),
        TRYON_UPSCALE_AFTER_QWEN=True,
        TRYON_UPSCALE_TARGET_LONG_EDGE=3072,
        TRYON_FINAL_OUTPUT_LONG_EDGE=2048,
    )

    response = tryon_service.run_tryon_request(payload, settings=settings, user_id="user-123")

    assert response.status == 200
    assert uploaded_dimensions == [(1365, 2048)]
    assert response.data.metadata["output"]["qwen_width"] == 512
    assert response.data.metadata["output"]["qwen_height"] == 768
    assert response.data.metadata["output"]["width"] == 1365
    assert response.data.metadata["output"]["height"] == 2048
    assert response.data.metadata["upscale"]["enabled"] is True
    assert response.data.metadata["upscale"]["model_variant"] == (
        "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    )
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_tryon_returns_422_null_data_for_invalid_garment_image(
    monkeypatch: MonkeyPatch,
) -> None:
    user_bytes = _build_png_bytes(512, 768)

    def fake_download(url: str) -> DownloadedMedia:
        content = user_bytes if "user" in url else b"not an image"
        return DownloadedMedia(
            content=content,
            content_type="image/png",
            source_url=url,
            filename="image.png",
        )

    monkeypatch.setattr(tryon_service, "download_media_from_url", fake_download)

    payload = TryonRequest.model_validate(
        {
            "user_image": "https://example.com/user.png",
            "products": [
                {
                    "image_url": "https://example.com/garment.png",
                    "type": "top",
                    "prompt": "red jacket",
                }
            ],
        },
    )

    response = tryon_service.run_tryon_request(payload, settings=Settings(), user_id="user-123")

    assert response.status == 422
    assert response.message == "Garment image is invalid or could not be downloaded."
    assert response.data is None


def test_tryon_returns_422_null_data_for_invalid_user_image(
    monkeypatch: MonkeyPatch,
) -> None:
    garment_bytes = _build_png_bytes(512, 768)

    def fake_download(url: str) -> DownloadedMedia:
        content = b"not an image" if "user" in url else garment_bytes
        return DownloadedMedia(
            content=content,
            content_type="image/png",
            source_url=url,
            filename="image.png",
        )

    monkeypatch.setattr(tryon_service, "download_media_from_url", fake_download)

    payload = TryonRequest.model_validate(
        {
            "user_image": "https://example.com/user.png",
            "products": [
                {
                    "image_url": "https://example.com/garment.png",
                    "type": "top",
                    "prompt": "red jacket",
                }
            ],
        },
    )

    response = tryon_service.run_tryon_request(payload, settings=Settings(), user_id="user-123")

    assert response.status == 422
    assert response.message == "User image is invalid or could not be downloaded."
    assert response.data is None


def test_build_tryon_prompt_preserves_duplicate_types_in_priority_order() -> None:
    payload = TryonRequest.model_validate(
        {
            "user_image": "https://example.com/user.png",
            "products": [
                {
                    "image_url": "https://example.com/bottom.png",
                    "type": "bottom",
                    "prompt": "grey checked cropped pants",
                },
                {
                    "image_url": "https://example.com/top-1.png",
                    "type": "top",
                    "prompt": "charcoal camisole top.",
                },
                {
                    "image_url": "https://example.com/dress.png",
                    "type": "dress",
                    "prompt": "navy pleated evening dress",
                },
                {
                    "image_url": "https://example.com/top-2.png",
                    "type": "top",
                    "prompt": "white asymmetrical short sleeve top!",
                },
            ],
        },
    )

    prompt = tryon_service._build_tryon_prompt(payload)

    assert prompt == (
        "Apply the reference garments from image 2 to the person in image 1. "
        "Top: charcoal camisole top. "
        "Top: white asymmetrical short sleeve top. "
        "Dress: navy pleated evening dress. "
        "Bottom: grey checked cropped pants. "
        "Preserve the person's face, identity, body proportions, pose, and background."
    )


def test_specialist_prompt_uses_trigger_caption_and_product_detail() -> None:
    payload = TryonRequest.model_validate(
        {
            "user_image": "https://example.com/user.png",
            "products": [
                {
                    "image_url": "https://example.com/top.png",
                    "type": "top",
                    "prompt": "red structured jacket with notched lapels",
                },
            ],
        },
    )
    settings = Settings()
    from app.services.tryon_routing import resolve_tryon_route

    routing = resolve_tryon_route(payload.products, settings)
    prompt = tryon_service._build_specialist_prompt(payload.products, routing, settings)

    assert prompt == (
        "Apply GlamifyTopTryon on this person. Replace the entire top garment on the person in "
        "image 1 with the red structured jacket with notched lapels from image 2. Remove any "
        "outer layer or jacket completely if present. Strictly preserve the person's face, "
        "identity, hair, skin tone, body shape, body size, body proportions, hands, pose and the "
        "background exactly; change only the top garment, fitting it naturally to the body with "
        "realistic drape. "
        "Render the entire image in sharp focus with crisp, high-resolution detail and absolutely "
        "no blur, softness, or smudging anywhere. Keep the person's face perfectly sharp and "
        "identical to image 1, preserving the exact facial features, expression and natural skin "
        "texture; keep the hair, skin, garment fabric, and background equally sharp and clearly "
        "detailed."
    )


def test_specialist_routing_outer_maps_to_top() -> None:
    from app.services.tryon_routing import resolve_tryon_route

    payload = TryonRequest.model_validate(
        {
            "user_image": "https://example.com/user.png",
            "products": [
                {
                    "image_url": "https://example.com/outer.png",
                    "type": "outer",
                    "prompt": "navy bomber jacket",
                },
            ],
        },
    )
    settings = Settings()
    routing = resolve_tryon_route(payload.products, settings)

    assert routing.lora_key == "top"
    assert routing.trigger_caption == "Apply GlamifyTopTryon on this person"


def test_specialist_routing_multi_for_two_or_more_products() -> None:
    from app.services.tryon_routing import resolve_tryon_route

    payload = TryonRequest.model_validate(
        {
            "user_image": "https://example.com/user.png",
            "products": [
                {
                    "image_url": "https://example.com/top.png",
                    "type": "top",
                    "prompt": "white tee",
                },
                {
                    "image_url": "https://example.com/bottom.png",
                    "type": "bottom",
                    "prompt": "blue jeans",
                },
            ],
        },
    )
    settings = Settings()
    routing = resolve_tryon_route(payload.products, settings)

    assert routing.lora_key == "multi"
    assert routing.trigger_caption == "Apply GlamifyMultiTryon on this person"


def test_specialist_routing_rejects_disabled_specialist() -> None:
    from app.services.tryon_routing import resolve_tryon_route

    payload = TryonRequest.model_validate(
        {
            "user_image": "https://example.com/user.png",
            "products": [
                {
                    "image_url": "https://example.com/dress.png",
                    "type": "dress",
                    "prompt": "black midi dress",
                },
            ],
        },
    )
    settings = Settings(
        TRYON_ENABLED_SPECIALISTS="top,bottom",
    )

    try:
        resolve_tryon_route(payload.products, settings)
    except ValueError as exc:
        assert "dress" in str(exc)
    else:
        raise AssertionError("Expected disabled specialist routing to fail")


def test_specialist_prompt_multi_lists_each_category() -> None:
    from app.services.tryon_routing import resolve_tryon_route

    payload = TryonRequest.model_validate(
        {
            "user_image": "https://example.com/user.png",
            "products": [
                {
                    "image_url": "https://example.com/bottom.png",
                    "type": "bottom",
                    "prompt": "black straight trousers",
                },
                {
                    "image_url": "https://example.com/top.png",
                    "type": "top",
                    "prompt": "red structured jacket",
                },
            ],
        },
    )
    settings = Settings()
    routing = resolve_tryon_route(payload.products, settings)
    prompt = tryon_service._build_specialist_prompt(payload.products, routing, settings)

    assert prompt == (
        "Apply GlamifyMultiTryon on this person. Replace the person's outfit in image 1 with the "
        "Top: red structured jacket and Bottom: black straight trousers from image 2. Remove any "
        "outer layer or jacket completely if present. Strictly preserve the person's face, "
        "identity, hair, skin tone, body shape, body size, body proportions, hands, pose and the "
        "background exactly; change only the specified garments, fitting them naturally to the "
        "body with realistic drape. "
        "Render the entire image in sharp focus with crisp, high-resolution detail and absolutely "
        "no blur, softness, or smudging anywhere. Keep the person's face perfectly sharp and "
        "identical to image 1, preserving the exact facial features, expression and natural skin "
        "texture; keep the hair, skin, garment fabric, and background equally sharp and clearly "
        "detailed."
    )


def test_build_tryon_prompt_orders_outer_with_tops_before_dress_and_bottom() -> None:
    payload = TryonRequest.model_validate(
        {
            "user_image": "https://example.com/user.png",
            "products": [
                {
                    "image_url": "https://example.com/bottom.png",
                    "type": "bottom",
                    "prompt": "black straight trousers",
                },
                {
                    "image_url": "https://example.com/dress.png",
                    "type": "dress",
                    "prompt": "ivory slip dress",
                },
                {
                    "image_url": "https://example.com/outer.png",
                    "type": "outer",
                    "prompt": "cropped denim jacket",
                },
            ],
        },
    )

    prompt = tryon_service._build_tryon_prompt(payload)

    assert prompt == (
        "Apply the reference garments from image 2 to the person in image 1. "
        "Outer: cropped denim jacket. "
        "Dress: ivory slip dress. "
        "Bottom: black straight trousers. "
        "Preserve the person's face, identity, body proportions, pose, and background."
    )


def _build_png_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

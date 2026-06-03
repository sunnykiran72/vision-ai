from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import TypeVar

from PIL import Image
from pytest import MonkeyPatch

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
            person_image_path: str,
            garment_reference_path: str,
            prompt: str,
            steps: int,
            guidance_scale: float,
            seed: int,
            output_path: str,
            output_width: int,
            output_height: int,
            lora_key: str | None = None,
        ) -> TryonRunResult:
            del prompt, steps, guidance_scale, seed, lora_key
            assert Path(person_image_path).exists()
            assert Path(garment_reference_path).exists()
            output = Image.open(garment_reference_path).convert("RGB").resize(
                (output_width, output_height),
            )
            output.save(output_path, format="JPEG", quality=95)
            return TryonRunResult(
                image=output,
                metadata={
                    "backend": "ai_toolkit_exact",
                    "lora_loaded": True,
                    "control_order": {
                        "ctrl_img_1": "person",
                        "ctrl_img_2": "garment_reference",
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
            assert "/user-123/" in object_name
            assert object_name.endswith("/output.jpg")
            assert content_type == "image/jpeg"
            return f"https://example.com/{object_name}"

    monkeypatch.setattr(tryon_service, "AzureStorageClient", FakeStorageClient)

    settings = Settings(
        TRYON_WORK_ROOT=str(tmp_path),
        AI_TOOLKIT_ROOT="/workspace/ai-toolkit",
        QWEN_IMAGE_EDIT_MODEL_PATH="/workspace/models/qwen-image-edit-2511",
        TRYON_LORA_PATH="/workspace/models/lora/tryon",
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
    assert response.data.metadata["reference"]["control_order"]["ctrl_img_1"] == "person"
    assert response.data.metadata["reference"]["control_order"]["ctrl_img_2"] == "garment_reference"
    assert response.data.metadata["resolved_settings"]["seed"] == 43
    assert response.data.metadata["output"]["width"] == 512
    assert response.data.metadata["output"]["height"] == 768
    assert response.data.metadata["output"]["inference_width"] == 512
    assert response.data.metadata["output"]["inference_height"] == 768
    assert response.data.metadata["routing"]["use_specialists"] is False
    assert not any(tmp_path.iterdir())


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
    settings = Settings(TRYON_USE_SPECIALISTS=True)
    from app.services.tryon_routing import resolve_tryon_route

    routing = resolve_tryon_route(payload.products, settings)
    prompt = tryon_service._build_specialist_prompt(payload.products, routing, settings)

    assert prompt == (
        "Apply GlamifyTopTryon on this person. "
        "Top: red structured jacket with notched lapels. "
        "Preserve the person's face, identity, body proportions, pose, and background."
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
    settings = Settings(TRYON_USE_SPECIALISTS=True)
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
    settings = Settings(TRYON_USE_SPECIALISTS=True)
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
        TRYON_USE_SPECIALISTS=True,
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
    settings = Settings(TRYON_USE_SPECIALISTS=True)
    routing = resolve_tryon_route(payload.products, settings)
    prompt = tryon_service._build_specialist_prompt(payload.products, routing, settings)

    assert prompt == (
        "Apply GlamifyMultiTryon on this person. "
        "Top: red structured jacket. "
        "Bottom: black straight trousers. "
        "Preserve the person's face, identity, body proportions, pose, and background."
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

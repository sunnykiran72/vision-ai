from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.main import app
from app.models.minicpm import MiniCPMGarmentResponse, MiniCPMGarmentResult
from app.models.tryon import TryonResponse, TryonResponseData
from app.models.upscale import UpscaleResponse, UpscaleResponseData
from app.models.user_validation import UserValidationResponse, UserValidationResult
from app.models.wardrobe import WardrobeAnalyzeResponse, WardrobeAnalyzeResult
from app.routes import minicpm as minicpm_route
from app.routes import tryon as tryon_route
from app.routes import upscale as upscale_route
from app.routes import user_validation as user_validation_route
from app.routes import wardrobe as wardrobe_route

client = TestClient(app)


def test_wardrobe_route_exists(auth_header: dict[str, str]) -> None:
    response = client.post("/v1/wardrobe", headers=auth_header)
    assert response.status_code == 422
    assert response.json()["data"] is None


def test_wardrobe_route_uses_multipart_request(
    monkeypatch: MonkeyPatch,
    auth_header: dict[str, str],
    auth_user_id: str,
) -> None:
    def fake_run_wardrobe_request(
        image_bytes: bytes,
        *,
        garment_type,
        user_id: str,
        access_token: str,
    ) -> WardrobeAnalyzeResponse:
        assert image_bytes == b"image-bytes"
        assert garment_type == "top"
        assert user_id == auth_user_id
        assert access_token and not access_token.startswith("Bearer ")
        return WardrobeAnalyzeResponse(
            status=200,
            message="",
            data=WardrobeAnalyzeResult(
                id="a9178f00-2d78-47c3-928d-80a28f6e082e",
                type="top",
                image="https://blob.example.com/output.jpg",
                category="t_shirts",
                categoryLabel="T-shirt",
            ),
        )

    monkeypatch.setattr(wardrobe_route, "run_wardrobe_request", fake_run_wardrobe_request)

    response = client.post(
        "/v1/wardrobe",
        headers=auth_header,
        files={"image": ("garment.png", b"image-bytes", "image/png")},
        data={"type": "top"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["category"] == "t_shirts"
    assert response.json()["data"]["categoryLabel"] == "T-shirt"
    assert response.json()["data"]["image"] == "https://blob.example.com/output.jpg"
    assert response.json()["data"]["type"] == "top"


def test_user_validation_route_requires_multipart_image(auth_header: dict[str, str]) -> None:
    response = client.post("/v1/user_validation", headers=auth_header)
    assert response.status_code == 422
    assert response.json()["data"] is None


def test_user_validation_route_uses_multipart_request(
    monkeypatch: MonkeyPatch,
    auth_header: dict[str, str],
    auth_user_id: str,
) -> None:
    def fake_run_user_validation_request(
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str | None,
        user_id: str,
    ) -> UserValidationResponse:
        assert image_bytes == b"user-image-bytes"
        assert filename == "person.jpg"
        assert content_type == "image/jpeg"
        assert user_id == auth_user_id
        return UserValidationResponse(
            status=200,
            message="",
            data=UserValidationResult(
                image="https://blob.example.com/user.jpg",
                metadata={"feature": "user_validation"},
            ),
        )

    monkeypatch.setattr(
        user_validation_route,
        "run_user_validation_request",
        fake_run_user_validation_request,
    )

    response = client.post(
        "/v1/user_validation",
        headers=auth_header,
        files={"image": ("person.jpg", b"user-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["data"]["image"] == "https://blob.example.com/user.jpg"
    assert response.json()["data"]["metadata"]["feature"] == "user_validation"


def test_minicpm_route_uses_structured_request(
    monkeypatch: MonkeyPatch,
    auth_header: dict[str, str],
) -> None:
    def fake_run_minicpm_garment_request(payload) -> MiniCPMGarmentResponse:
        assert payload.type == "top"
        return MiniCPMGarmentResponse(
            status=200,
            message="",
            data=MiniCPMGarmentResult(
                type="top",
                description="plain garment construction sentence.",
                prompt="prompt text",
                model="openbmb/MiniCPM-V-4.6",
                metadata={"latency_ms": 10},
            ),
        )

    monkeypatch.setattr(
        minicpm_route,
        "run_minicpm_garment_request",
        fake_run_minicpm_garment_request,
    )

    response = client.post(
        "/dev/minicpm/garment",
        headers=auth_header,
        json={"image": "base64", "type": "top"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["description"] == "plain garment construction sentence."
    assert response.json()["data"]["metadata"]["latency_ms"] == 10


def test_tryon_route_exists(
    monkeypatch: MonkeyPatch,
    auth_header: dict[str, str],
    auth_user_id: str,
) -> None:
    monkeypatch.setattr(
        tryon_route,
        "run_tryon_request",
        lambda _payload, user_id: TryonResponse(
            status=200,
            message="mocked",
            data=TryonResponseData(
                url=None,
                metadata={"feature": "tryon", "user_id": user_id},
            ),
        ),
    )
    response = client.post(
        "/v1/tryon",
        headers=auth_header,
        json={
            "user_image": "https://example.com/user.jpg",
            "products": [
                {
                    "image_url": "https://example.com/product.jpg",
                    "type": "top",
                    "prompt": "red structured jacket",
                }
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["data"]["metadata"]["feature"] == "tryon"
    assert response.json()["data"]["metadata"]["user_id"] == auth_user_id


def test_tryon_route_uses_structured_request(
    monkeypatch: MonkeyPatch,
    auth_header: dict[str, str],
    auth_user_id: str,
) -> None:
    monkeypatch.setattr(
        tryon_route,
        "run_tryon_request",
        lambda _payload, user_id: TryonResponse(
            status=200,
            message="mocked",
            data=TryonResponseData(
                url=None,
                metadata={"feature": "tryon", "user_id": user_id},
            ),
        ),
    )
    response = client.post(
        "/v1/tryon",
        headers=auth_header,
        json={
            "user_image": "https://example.com/user.jpg",
            "products": [
                {
                    "image_url": "https://example.com/product.jpg",
                    "type": "top",
                    "prompt": "red structured jacket",
                }
            ],
            "seed": 44,
            "steps": 8,
            "guidance_scale": 2.5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == 200
    assert payload["data"]["metadata"]["feature"] == "tryon"
    assert payload["data"]["metadata"]["user_id"] == auth_user_id


def test_upscale_route_exists(
    monkeypatch: MonkeyPatch,
    auth_header: dict[str, str],
) -> None:
    monkeypatch.setattr(
        upscale_route,
        "run_upscale_request",
        lambda _payload, user_id=None: UpscaleResponse(
            status=200,
            message="mocked",
            data=UpscaleResponseData(
                url="https://example.com/upscaled.png",
                metadata={"feature": "upscale", "user_id": user_id},
            ),
        ),
    )
    response = client.post(
        "/v1/upscale",
        headers=auth_header,
        json={
            "image_url": "https://example.com/image.png",
            "metric": "2k",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == 200
    assert payload["data"]["url"] == "https://example.com/upscaled.png"
    assert payload["data"]["metadata"]["feature"] == "upscale"

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.main import app
from app.models.tryon import TryonResponse, TryonResponseData
from app.models.upscale import UpscaleResponse, UpscaleResponseData
from app.models.wardrobe import WardrobeAnalyzeResponse, WardrobeAnalyzeResult
from app.routes import tryon as tryon_route
from app.routes import upscale as upscale_route
from app.routes import wardrobe as wardrobe_route

client = TestClient(app)


def test_wardrobe_route_exists(auth_header: dict[str, str]) -> None:
    response = client.post("/v1/wardrobe", headers=auth_header)
    assert response.status_code == 422
    assert response.json()["data"] is None


def test_wardrobe_route_uses_structured_request(
    monkeypatch: MonkeyPatch,
    auth_header: dict[str, str],
    auth_user_id: str,
) -> None:
    def fake_run_wardrobe_request(
        payload,
        *,
        user_id: str,
        bearer_token: str,
    ) -> WardrobeAnalyzeResponse:
        assert payload.type == "top"
        assert user_id == auth_user_id
        assert bearer_token.startswith("Bearer ")
        return WardrobeAnalyzeResponse(
            status=200,
            message="",
            data=WardrobeAnalyzeResult(
                id="a9178f00-2d78-47c3-928d-80a28f6e082e",
                type="top",
                image="jpeg-base64",
                category="t_shirts",
                categoryLabel="T-shirt",
            ),
        )

    monkeypatch.setattr(wardrobe_route, "run_wardrobe_request", fake_run_wardrobe_request)

    response = client.post(
        "/v1/wardrobe",
        headers=auth_header,
        json={"image": "base64", "type": "top"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["category"] == "t_shirts"
    assert response.json()["data"]["categoryLabel"] == "T-shirt"
    assert response.json()["data"]["type"] == "top"


def test_user_validation_route_exists(auth_header: dict[str, str]) -> None:
    response = client.post("/v1/user_validation", headers=auth_header)
    assert response.status_code == 200
    assert response.json()["feature"] == "user_validation"


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

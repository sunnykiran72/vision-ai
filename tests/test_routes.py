from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.main import app
from app.models.tryon import TryonResponse, TryonResponseData
from app.models.upscale import UpscaleResponse, UpscaleResponseData
from app.routes import tryon as tryon_route
from app.routes import upscale as upscale_route

client = TestClient(app)


def test_wardrobe_route_exists(auth_header: dict[str, str]) -> None:
    response = client.post("/v1/wardrobe", headers=auth_header)
    assert response.status_code == 200
    assert response.json()["feature"] == "wardrobe"


def test_user_validation_route_exists(auth_header: dict[str, str]) -> None:
    response = client.post("/v1/user_validation", headers=auth_header)
    assert response.status_code == 200
    assert response.json()["feature"] == "user_validation"


def test_tryon_route_exists(
    monkeypatch: MonkeyPatch,
    auth_header: dict[str, str],
) -> None:
    monkeypatch.setattr(
        tryon_route,
        "run_tryon_request",
        lambda _payload, user_id=None: TryonResponse(
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


def test_tryon_route_uses_structured_request(
    monkeypatch: MonkeyPatch,
    auth_header: dict[str, str],
) -> None:
    monkeypatch.setattr(
        tryon_route,
        "run_tryon_request",
        lambda _payload, user_id=None: TryonResponse(
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

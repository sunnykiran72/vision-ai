from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.main import app
from app.models.upscale import UpscaleResponse, UpscaleResponseData
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


def test_tryon_route_exists(auth_header: dict[str, str]) -> None:
    response = client.post("/v1/tryon", headers=auth_header)
    assert response.status_code == 200
    assert response.json()["feature"] == "tryon"


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
            "output_file_name": "result.jpg",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == 200
    assert payload["data"]["url"] == "https://example.com/upscaled.png"
    assert payload["data"]["metadata"]["feature"] == "upscale"

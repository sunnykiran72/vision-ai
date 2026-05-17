from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.main import app
from app.models.upscale import UpscaleResponse, UpscaleResponseData
from app.routes import upscale as upscale_route

client = TestClient(app)


def test_health_route_is_public() -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_protected_route_requires_bearer_token() -> None:
    response = client.post(
        "/v1/upscale",
        json={
            "image_url": "https://example.com/image.png",
            "metric": "2k",
            "output_file_name": "result.jpg",
        },
    )
    assert response.status_code == 401
    assert response.json()["status"] == 401


def test_invalid_token_is_rejected() -> None:
    response = client.post(
        "/v1/upscale",
        headers={"Authorization": "Bearer invalid-token"},
        json={
            "image_url": "https://example.com/image.png",
            "metric": "2k",
            "output_file_name": "result.jpg",
        },
    )
    assert response.status_code == 401
    assert response.json()["status"] == 401


def test_valid_token_allows_access(
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
                url="https://example.com/upscaled.jpg",
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
    assert response.json()["data"]["metadata"]["feature"] == "upscale"

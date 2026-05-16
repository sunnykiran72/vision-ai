from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_wardrobe_route_exists() -> None:
    response = client.post("/v1/wardrobe")
    assert response.status_code == 200
    assert response.json()["feature"] == "wardrobe"


def test_user_validation_route_exists() -> None:
    response = client.post("/v1/user_validation")
    assert response.status_code == 200
    assert response.json()["feature"] == "user_validation"


def test_tryon_route_exists() -> None:
    response = client.post("/v1/tryon")
    assert response.status_code == 200
    assert response.json()["feature"] == "tryon"


def test_upscale_route_exists() -> None:
    response = client.post(
        "/v1/upscale",
        json={"image_url": "https://example.com/image.png", "target_long_edge": 2048},
    )
    assert response.status_code == 200
    assert response.json()["feature"] == "upscale"

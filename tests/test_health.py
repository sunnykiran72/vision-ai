from fastapi.testclient import TestClient

from app.main import app


def test_healthcheck() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "Glamify AI"
    assert payload["metadata"]["environment"] == "local"
    assert "health" in payload["metadata"]["available_api_groups"]

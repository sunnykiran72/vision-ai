from __future__ import annotations

import time
from collections.abc import Generator
from uuid import uuid4

import jwt
import pytest

from app.config import get_settings

TEST_JWT_SECRET = "test-jwt-secret-1234567890-abcdef"
TEST_JWT_ALGORITHM = "HS256"


@pytest.fixture(autouse=True)
def auth_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setenv("JWT_ACCESS_SECRET", TEST_JWT_SECRET)
    monkeypatch.setenv("JWT_ALGORITHM", TEST_JWT_ALGORITHM)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def auth_header() -> dict[str, str]:
    payload = {
        "userId": str(uuid4()),
        "authType": "EMAIL",
        "token_id": str(uuid4()),
        "exp": int(time.time()) + 3600,
    }
    token = jwt.encode(payload, TEST_JWT_SECRET, algorithm=TEST_JWT_ALGORITHM)
    return {"Authorization": f"Bearer {token}"}

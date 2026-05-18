from __future__ import annotations

from collections.abc import Awaitable, Callable

import jwt
from fastapi import Request
from fastapi.responses import JSONResponse, Response
from jwt import InvalidTokenError

from app.config import get_settings
from app.constants import http_status
from app.models.auth import AuthPayload

PUBLIC_PATH_PREFIXES = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
)


async def authorization_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    path = request.url.path
    is_public_path = any(path.startswith(prefix) for prefix in PUBLIC_PATH_PREFIXES)
    auth_header = request.headers.get("Authorization", "")

    if not is_public_path and not auth_header.startswith("Bearer "):
        return _json_unauthorized()

    token = auth_header.split(" ", 1)[1].strip() if auth_header.startswith("Bearer ") else ""
    if is_public_path and not token:
        request.state.auth_payload = None
        return await call_next(request)
    if not token:
        return _json_unauthorized()

    settings = get_settings()
    try:
        decoded = jwt.decode(
            token,
            settings.jwt_access_secret,
            algorithms=[settings.jwt_algorithm],
        )
        request.state.auth_payload = AuthPayload.model_validate(decoded)
    except (InvalidTokenError, ValueError):
        return _json_unauthorized()

    return await call_next(request)


def _json_unauthorized() -> JSONResponse:
    return JSONResponse(
        status_code=http_status.UNAUTHORIZED,
        content={
            "status": http_status.UNAUTHORIZED,
            "message": "Invalid token",
            "data": None,
        },
    )

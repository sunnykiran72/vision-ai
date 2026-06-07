from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request

from app.constants import http_status
from app.models.auth import AuthPayload


def get_current_auth_payload(request: Request) -> AuthPayload:
    payload = getattr(request.state, "auth_payload", None)
    if payload is None:
        raise HTTPException(
            status_code=http_status.UNAUTHORIZED,
            detail="Invalid token",
        )
    return cast(AuthPayload, payload)


def get_current_access_token(request: Request) -> str:
    """Raw verified JWT, captured per-request by the auth middleware.

    Read from ``request.state`` (one Request per HTTP request), so concurrent requests keep their
    own token. Handlers pass this value as a function argument into the worker, never via shared
    module state.
    """
    token = getattr(request.state, "access_token", "") or ""
    if not token:
        raise HTTPException(
            status_code=http_status.UNAUTHORIZED,
            detail="Invalid token",
        )
    return str(token)


CurrentAuth = Annotated[AuthPayload, Depends(get_current_auth_payload)]
CurrentAccessToken = Annotated[str, Depends(get_current_access_token)]

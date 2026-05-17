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


CurrentAuth = Annotated[AuthPayload, Depends(get_current_auth_payload)]

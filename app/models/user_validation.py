from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.models.common import ApiResponse


class UserValidationResult(BaseModel):
    image: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserValidationResponse(ApiResponse[UserValidationResult]):
    data: UserValidationResult | None = None

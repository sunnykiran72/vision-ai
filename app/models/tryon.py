from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator

from app.models.common import ApiResponse


class TryonProductType(StrEnum):
    TOP = "top"
    BOTTOM = "bottom"
    DRESS = "dress"
    OUTER = "outer"


class TryonProduct(BaseModel):
    image_url: HttpUrl = Field(..., description="Public HTTP(S) URL for the garment image")
    type: TryonProductType = Field(..., description="Garment category")
    prompt: str = Field(..., description="Prompt describing the garment for try-on")

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        cleaned = " ".join(str(value or "").split()).strip()
        if not cleaned:
            raise ValueError("prompt must not be empty")
        return cleaned


class TryonRequest(BaseModel):
    user_image: HttpUrl = Field(..., description="Public HTTP(S) URL for the user image")
    products: list[TryonProduct] = Field(
        ...,
        min_length=1,
        description="Products to apply to the user image",
    )
    seed: int | None = Field(default=None, ge=0, le=2147483647, description="Optional random seed")
    steps: int | None = Field(default=None, ge=1, le=60, description="Optional generation steps")
    guidance_scale: float | None = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="Optional guidance scale",
    )


class TryonResponseData(BaseModel):
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TryonResponse(ApiResponse[TryonResponseData]):
    data: TryonResponseData

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from app.models.common import ApiResponse


class WardrobeGarmentType(StrEnum):
    TOP = "top"
    BOTTOM = "bottom"
    DRESS = "dress"


class WardrobeAnalyzeRequest(BaseModel):
    image: str = Field(..., min_length=1, description="Raw base64 or image data URL")
    type: WardrobeGarmentType = Field(..., description="Requested garment extraction type")
    prompt: str | None = Field(
        default=None,
        description="Optional Qwen prompt override for prompt testing",
    )


class WardrobeAnalyzeResult(BaseModel):
    id: str
    type: WardrobeGarmentType
    image: str
    category: str
    category_label: str = Field(..., alias="categoryLabel")


class WardrobeAnalyzeResponse(ApiResponse[WardrobeAnalyzeResult]):
    data: WardrobeAnalyzeResult | None = None

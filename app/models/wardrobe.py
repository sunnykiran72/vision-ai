from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.models.common import ApiResponse


class WardrobeGarmentType(StrEnum):
    TOP = "top"
    BOTTOM = "bottom"
    DRESS = "dress"


class WardrobeAnalyzeResult(BaseModel):
    id: str
    type: WardrobeGarmentType
    image: str = Field(..., description="Public URL of the extracted garment image")
    category: str
    category_label: str = Field(..., alias="categoryLabel")
    metadata: dict[str, Any] = Field(default_factory=dict)


class WardrobeAnalyzeResponse(ApiResponse[WardrobeAnalyzeResult]):
    data: WardrobeAnalyzeResult | None = None

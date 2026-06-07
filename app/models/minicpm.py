from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.common import ApiResponse
from app.models.wardrobe import WardrobeGarmentType


class MiniCPMGarmentRequest(BaseModel):
    image: str = Field(..., min_length=1, description="Raw base64 or image data URL")
    type: WardrobeGarmentType = Field(..., description="Garment type to describe")
    prompt: str | None = Field(
        default=None,
        description="Optional prompt override for prompt A/B testing",
    )


class MiniCPMGarmentResult(BaseModel):
    type: WardrobeGarmentType
    description: str
    prompt: str
    model: str
    metadata: dict[str, object] = Field(default_factory=dict)


class MiniCPMGarmentResponse(ApiResponse[MiniCPMGarmentResult]):
    data: MiniCPMGarmentResult | None = None

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl

from app.models.common import ApiResponse


class UpscaleMetric(StrEnum):
    TWO_K = "2k"
    FOUR_K = "4k"


class UpscaleRequest(BaseModel):
    image_url: HttpUrl = Field(..., description="Public HTTP(S) URL for the input image")
    metric: UpscaleMetric = Field(default=UpscaleMetric.TWO_K, description="Target output preset")
    output_file_name: str | None = Field(
        default=None,
        description="Optional output filename to use for the generated file",
    )


class UpscaleResponseData(BaseModel):
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpscaleResponse(ApiResponse[UpscaleResponseData]):
    data: UpscaleResponseData

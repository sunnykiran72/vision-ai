from pydantic import BaseModel, Field


class UpscaleRequest(BaseModel):
    image_url: str = Field(..., description="Public image URL")
    target_long_edge: int = Field(default=2048, ge=512, le=4096)


class UpscaleResponse(BaseModel):
    status: str
    message: str
    feature: str


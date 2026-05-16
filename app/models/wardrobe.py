from pydantic import BaseModel, Field


class WardrobeAnalyzeRequest(BaseModel):
    garment_type: str | None = Field(
        default=None,
        description="Optional type hint such as top, bottom, dress, outer",
    )
    debug: bool = Field(default=False)


class WardrobeAnalyzeResponse(BaseModel):
    status: str
    message: str
    feature: str

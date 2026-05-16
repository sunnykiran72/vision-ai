from pydantic import BaseModel, Field


class UserValidationRequest(BaseModel):
    resize_method: str | None = Field(default=None)
    output_max_edge: int | None = Field(default=None, ge=512, le=4096)


class UserValidationResponse(BaseModel):
    status: str
    message: str
    feature: str


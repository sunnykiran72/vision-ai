from pydantic import BaseModel


class TryonResponse(BaseModel):
    status: str
    message: str
    feature: str


from fastapi import APIRouter

from app.models.tryon import TryonResponse
from app.services.tryon import build_tryon_placeholder

router = APIRouter()

@router.post("/v1/tryon", response_model=TryonResponse)
async def tryon() -> TryonResponse:
    return build_tryon_placeholder()

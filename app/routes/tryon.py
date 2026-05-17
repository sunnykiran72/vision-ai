from fastapi import APIRouter

from app.dependencies.auth import CurrentAuth
from app.models.tryon import TryonResponse
from app.services.tryon import build_tryon_placeholder

router = APIRouter()

@router.post("/v1/tryon", response_model=TryonResponse)
async def tryon(current_user: CurrentAuth) -> TryonResponse:
    del current_user
    return build_tryon_placeholder()

from fastapi import APIRouter

from app.models.upscale import UpscaleRequest, UpscaleResponse
from app.services.upscale import build_upscale_placeholder

router = APIRouter()


@router.post("/v1/upscale", response_model=UpscaleResponse)
async def upscale(_payload: UpscaleRequest) -> UpscaleResponse:
    return build_upscale_placeholder()

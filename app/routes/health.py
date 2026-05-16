from fastapi import APIRouter

from app.config import get_settings
from app.models.health import HealthResponse
from app.services.health import build_health_response

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    return build_health_response(get_settings())

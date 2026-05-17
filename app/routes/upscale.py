from fastapi import APIRouter, Response
from fastapi.concurrency import run_in_threadpool

from app.constants import http_status
from app.dependencies.auth import CurrentAuth
from app.models.upscale import UpscaleRequest, UpscaleResponse
from app.services.upscale import run_upscale_request

router = APIRouter()


@router.post("/v1/upscale", response_model=UpscaleResponse, status_code=http_status.OK)
async def upscale(
    payload: UpscaleRequest,
    response: Response,
    current_user: CurrentAuth,
) -> UpscaleResponse:
    result = await run_in_threadpool(
        run_upscale_request,
        payload,
        user_id=str(current_user.user_id),
    )
    response.status_code = result.status
    return result

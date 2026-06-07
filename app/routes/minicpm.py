from fastapi import APIRouter, Response
from fastapi.concurrency import run_in_threadpool

from app.constants import http_status
from app.dependencies.auth import CurrentAuth
from app.models.minicpm import MiniCPMGarmentRequest, MiniCPMGarmentResponse
from app.services.minicpm import run_minicpm_garment_request

router = APIRouter()


@router.post(
    "/dev/minicpm/garment",
    response_model=MiniCPMGarmentResponse,
    status_code=http_status.OK,
)
async def minicpm_garment(
    payload: MiniCPMGarmentRequest,
    response: Response,
    _current_user: CurrentAuth,
) -> MiniCPMGarmentResponse:
    result = await run_in_threadpool(run_minicpm_garment_request, payload)
    response.status_code = result.status
    return result

from fastapi import APIRouter, Query, Response
from fastapi.concurrency import run_in_threadpool

from app.constants import http_status
from app.dependencies.auth import CurrentAuth
from app.models.tryon import TryonRequest, TryonResponse
from app.services.tryon import run_tryon_request

router = APIRouter()


@router.post("/v1/tryon", response_model=TryonResponse, status_code=http_status.OK)
async def tryon(
    payload: TryonRequest,
    response: Response,
    current_user: CurrentAuth,
    upscale: bool | None = Query(
        default=None,
        description=(
            "Override the inline SeedVR2 upscale. true=force the ~2496px upscale, "
            "false=skip it and return the raw Qwen output (faster), "
            "omitted=use the server default (TRYON_UPSCALE_AFTER_QWEN)."
        ),
    ),
) -> TryonResponse:
    result = await run_in_threadpool(
        run_tryon_request,
        payload,
        user_id=str(current_user.user_id),
        upscale_override=upscale,
    )
    response.status_code = result.status
    return result

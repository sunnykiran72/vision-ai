from fastapi import APIRouter, Request, Response
from fastapi.concurrency import run_in_threadpool

from app.constants import http_status
from app.dependencies.auth import CurrentAuth
from app.models.wardrobe import WardrobeAnalyzeRequest, WardrobeAnalyzeResponse
from app.services.wardrobe import run_wardrobe_request

router = APIRouter()


@router.post("/v1/wardrobe", response_model=WardrobeAnalyzeResponse, status_code=http_status.OK)
async def wardrobe_analyze(
    payload: WardrobeAnalyzeRequest,
    request: Request,
    response: Response,
    current_user: CurrentAuth,
) -> WardrobeAnalyzeResponse:
    result = await run_in_threadpool(
        run_wardrobe_request,
        payload,
        user_id=str(current_user.user_id),
        bearer_token=request.headers.get("Authorization", ""),
    )
    response.status_code = result.status
    return result

from typing import Annotated

from fastapi import APIRouter, File, Form, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.constants import http_status
from app.dependencies.auth import CurrentAccessToken, CurrentAuth
from app.models.wardrobe import WardrobeAnalyzeResponse, WardrobeGarmentType
from app.services.wardrobe import run_wardrobe_request

router = APIRouter()


@router.post("/v1/wardrobe", response_model=WardrobeAnalyzeResponse, status_code=http_status.OK)
async def wardrobe_analyze(
    response: Response,
    current_user: CurrentAuth,
    access_token: CurrentAccessToken,
    image: Annotated[UploadFile, File()],
    type: Annotated[WardrobeGarmentType, Form()],
) -> WardrobeAnalyzeResponse:
    image_bytes = await image.read()
    result = await run_in_threadpool(
        run_wardrobe_request,
        image_bytes,
        garment_type=type,
        user_id=str(current_user.user_id),
        access_token=access_token,
    )
    response.status_code = result.status
    return result

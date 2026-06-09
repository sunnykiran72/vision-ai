from typing import Annotated

from fastapi import APIRouter, File, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.dependencies.auth import CurrentAuth
from app.models.user_validation import UserValidationResponse
from app.services.user_validation import run_user_validation_request

router = APIRouter()


@router.post("/v1/user_validation", response_model=UserValidationResponse)
async def user_validation(
    response: Response,
    current_user: CurrentAuth,
    image: Annotated[UploadFile, File()],
) -> UserValidationResponse:
    image_bytes = await image.read()
    result = await run_in_threadpool(
        run_user_validation_request,
        image_bytes,
        filename=image.filename or "image",
        content_type=image.content_type,
        user_id=str(current_user.user_id),
    )
    response.status_code = result.status
    return result

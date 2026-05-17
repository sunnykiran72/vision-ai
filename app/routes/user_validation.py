from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from app.dependencies.auth import CurrentAuth
from app.models.user_validation import UserValidationResponse
from app.services.user_validation import build_user_validation_placeholder

router = APIRouter()


@router.post("/v1/user_validation", response_model=UserValidationResponse)
async def user_validation(
    current_user: CurrentAuth,
    file: Annotated[UploadFile | None, File()] = None,
    image: Annotated[UploadFile | None, File()] = None,
    resize_method: Annotated[str | None, Form()] = None,
    resizeMethod: Annotated[str | None, Form()] = None,
    output_max_edge: Annotated[int | None, Form()] = None,
    outputMaxEdge: Annotated[int | None, Form()] = None,
) -> UserValidationResponse:
    del current_user, file, image, resize_method, resizeMethod, output_max_edge, outputMaxEdge
    return build_user_validation_placeholder()

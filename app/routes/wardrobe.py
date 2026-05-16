from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from app.models.wardrobe import WardrobeAnalyzeResponse
from app.services.wardrobe import build_wardrobe_placeholder

router = APIRouter()


@router.post("/v1/wardrobe", response_model=WardrobeAnalyzeResponse)
async def wardrobe_analyze(
    file: Annotated[UploadFile | None, File()] = None,
    image: Annotated[UploadFile | None, File()] = None,
    garment_type: Annotated[str | None, Form(alias="type")] = None,
    debug: Annotated[bool, Form()] = False,
) -> WardrobeAnalyzeResponse:
    del file, image, garment_type, debug
    return build_wardrobe_placeholder()

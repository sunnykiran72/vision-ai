"""Unauthenticated diffusers extraction test endpoints for parity testing.

These mirror the standalone diffusers tester's `/api/extract` and `/api/loras` so the
wardrobe extraction running in this service can be A/B compared against the reference
app under identical multipart inputs. They share the resident wardrobe engine and the
same GPU execution coordinator as `/v1/wardrobe`.
"""
from __future__ import annotations

from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError

from app.clients.qwen_diffusers_engine import (
    WARDROBE_CATEGORIES,
    WardrobeDiffusersGenerationError,
    WardrobeDiffusersRuntimeError,
)
from app.config import get_settings
from app.constants import http_status
from app.constants import wardrobe as wardrobe_constants
from app.runtime.system_coordinator import get_system_execution_coordinator
from app.runtime.wardrobe_runtime import get_wardrobe_runner

router = APIRouter()


@router.get("/tools/diffusers/loras")
async def diffusers_loras() -> JSONResponse:
    return JSONResponse(
        [{"key": category, "category": category} for category in WARDROBE_CATEGORIES],
    )


@router.post("/tools/diffusers/extract")
async def diffusers_extract(
    source: Annotated[UploadFile, File()],
    lora_key: Annotated[str, Form()],
    steps: Annotated[int, Form()] = wardrobe_constants.GENERATION_STEPS,
    seed: Annotated[int, Form()] = wardrobe_constants.GENERATION_SEED,
    width: Annotated[int, Form()] = wardrobe_constants.OUTPUT_WIDTH,
    height: Annotated[int, Form()] = wardrobe_constants.OUTPUT_HEIGHT,
    lora_scale: Annotated[float, Form()] = wardrobe_constants.GENERATION_NETWORK_MULTIPLIER,
    prompt: Annotated[str, Form()] = "",
) -> StreamingResponse:
    category = str(lora_key).strip().lower()
    if category not in WARDROBE_CATEGORIES:
        raise HTTPException(http_status.BAD_REQUEST, f"unknown lora_key {lora_key}")

    try:
        source_image = Image.open(BytesIO(await source.read())).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(http_status.BAD_REQUEST, "source is not a valid image.") from exc

    resolved_prompt = prompt.strip() or wardrobe_constants.QWEN_EXTRACT_PROMPT_TEMPLATE_BY_TYPE[
        category
    ].format(caption="the garment")
    settings = get_settings()
    engine = get_wardrobe_runner(settings)
    coordinator = get_system_execution_coordinator(settings)

    try:
        image, seconds = await run_in_threadpool(
            lambda: coordinator.run(
                lambda: engine.generate_preview(
                    garment_type=category,
                    image=source_image,
                    prompt=resolved_prompt,
                    steps=int(steps),
                    seed=int(seed),
                    width=int(width),
                    height=int(height),
                    lora_scale=float(lora_scale),
                ),
            ),
        )
    except (WardrobeDiffusersGenerationError, WardrobeDiffusersRuntimeError) as exc:
        raise HTTPException(http_status.INTERNAL_SERVER_ERROR, f"extraction failed: {exc}") from exc

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    headers = {
        "X-Extract-Seconds": str(seconds),
        "X-Extract-Lora": category,
        "X-Extract-Steps": str(int(steps)),
        "X-Extract-Seed": str(int(seed)),
        "X-Extract-Prompt": resolved_prompt,
        "X-Extract-Output-Width": str(int(image.width)),
        "X-Extract-Output-Height": str(int(image.height)),
    }
    return StreamingResponse(buffer, media_type="image/jpeg", headers=headers)

"""Unauthenticated try-on lab endpoints, served under the public ``/tools`` prefix.

Mirrors the upscale lab: a page plus a ``/run`` endpoint that executes the real
try-on engine on uploaded images and returns the output + full metadata/timings.
Test-pod only; never used by the production ``/v1/tryon`` path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from app.constants import http_status
from app.constants import tryon as tryon_constants
from app.services.tryon_lab import run_lab_tryon

router = APIRouter()

_LAB_PAGE = Path(__file__).resolve().parents[2] / "tools" / "tryon_lab.html"


@router.get("/tools/tryon-lab", include_in_schema=False)
async def tryon_lab_page() -> FileResponse:
    return FileResponse(_LAB_PAGE)


@router.post("/tools/tryon-lab/run")
async def tryon_lab_run(
    person: Annotated[UploadFile, File()],
    garments: Annotated[list[UploadFile], File()],
    types: Annotated[list[str], Form()],
    prompts: Annotated[list[str], Form()],
    steps: Annotated[int, Form()] = tryon_constants.DEFAULT_STEPS,
    seed: Annotated[int, Form()] = tryon_constants.DEFAULT_SEED,
    guidance_scale: Annotated[float, Form()] = tryon_constants.DEFAULT_GUIDANCE_SCALE,
    upscale: Annotated[bool, Form()] = True,
) -> JSONResponse:
    person_bytes = await person.read()
    if not person_bytes:
        return JSONResponse(
            {"ok": False, "error": "Empty person image upload."},
            status_code=http_status.BAD_REQUEST,
        )

    garment_specs: list[dict[str, object]] = []
    for index, garment_file in enumerate(garments):
        content = await garment_file.read()
        if not content:
            continue
        garment_specs.append(
            {
                "bytes": content,
                "type": types[index] if index < len(types) else "top",
                "prompt": prompts[index] if index < len(prompts) else "",
            },
        )
    if not garment_specs:
        return JSONResponse(
            {"ok": False, "error": "At least one garment image is required."},
            status_code=http_status.BAD_REQUEST,
        )

    try:
        result = await run_in_threadpool(
            run_lab_tryon,
            person_bytes=person_bytes,
            garments=garment_specs,
            steps=int(steps),
            seed=int(seed),
            guidance_scale=float(guidance_scale),
            upscale=bool(upscale),
        )
    except ValueError as exc:
        return JSONResponse(
            {"ok": False, "error": str(exc)},
            status_code=http_status.BAD_REQUEST,
        )
    except Exception as exc:  # noqa: BLE001 - surface any runtime error to the lab UI
        return JSONResponse(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            status_code=http_status.INTERNAL_SERVER_ERROR,
        )

    return JSONResponse(result)

"""Unauthenticated SeedVR2 A/B latency lab endpoints.

Served under the public ``/tools`` prefix (see ``PUBLIC_PATH_PREFIXES``), so these
require no auth and are meant for a separate test pod only. They never call the
production ``/v1/upscale`` service or its resident runner config; variant loading
goes through ``get_seedvr2_client`` keyed by the explicitly chosen variant.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from app.constants import http_status
from app.services.upscale_lab import list_variants, run_lab_upscale

router = APIRouter()

_LAB_PAGE = Path(__file__).resolve().parents[2] / "tools" / "upscale_lab.html"


@router.get("/tools/upscale-lab", include_in_schema=False)
async def upscale_lab_page() -> FileResponse:
    return FileResponse(_LAB_PAGE)


@router.get("/tools/upscale-lab/variants")
async def upscale_lab_variants() -> JSONResponse:
    variants = [asdict(info) for info in list_variants()]
    return JSONResponse({"variants": variants})


@router.post("/tools/upscale-lab/run")
async def upscale_lab_run(
    image: Annotated[UploadFile, File()],
    model_variant: Annotated[str, Form()],
    pre_resize_max_edge: Annotated[int, Form()] = 1024,
    output_max_edge: Annotated[int, Form()] = 1536,
) -> JSONResponse:
    image_bytes = await image.read()
    if not image_bytes:
        return JSONResponse(
            {"ok": False, "error": "Empty image upload."},
            status_code=http_status.BAD_REQUEST,
        )

    try:
        result = await run_in_threadpool(
            run_lab_upscale,
            image_bytes=image_bytes,
            model_variant=model_variant,
            pre_resize_max_edge=int(pre_resize_max_edge),
            output_max_edge=int(output_max_edge),
        )
    except FileNotFoundError as exc:
        return JSONResponse(
            {"ok": False, "error": str(exc)},
            status_code=http_status.BAD_REQUEST,
        )
    except ValueError as exc:
        return JSONResponse(
            {"ok": False, "error": str(exc)},
            status_code=http_status.BAD_REQUEST,
        )
    except SystemExit as exc:
        # The SeedVR2 CLI uses argparse, which calls sys.exit() (raising SystemExit, a
        # BaseException) when given a --dit_model name outside its hardcoded choices.
        # Catch it here so an unsupported variant returns a clean 400 instead of
        # tearing down the worker process.
        return JSONResponse(
            {
                "ok": False,
                "error": (
                    f"SeedVR2 CLI rejected the arguments (exit {exc.code}). "
                    "The selected variant is likely unsupported by this CLI build."
                ),
            },
            status_code=http_status.BAD_REQUEST,
        )
    except Exception as exc:  # noqa: BLE001 - surface any runtime error to the tester UI
        return JSONResponse(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            status_code=http_status.INTERNAL_SERVER_ERROR,
        )

    return JSONResponse(result)

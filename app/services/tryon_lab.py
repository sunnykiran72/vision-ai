"""Unauthenticated try-on lab.

Runs the REAL try-on path (garment reference build -> prompt assembly -> fp8 Qwen
generation -> optional inline SeedVR2 upscale) from uploaded images, and returns the
output image plus the full prompt/metadata/timings as JSON. Served under the public
``/tools`` prefix (no auth, no storage) for inspection/iteration on a test pod.

It reuses the production resident runners and helpers, so what you see here matches
what ``/v1/tryon`` produces (minus URL download + Azure upload).
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from app.config import Settings, get_settings
from app.constants import tryon as tryon_constants
from app.models.tryon import TryonProduct
from app.runtime.system_coordinator import get_system_execution_coordinator
from app.runtime.tryon_runtime import get_tryon_runner
from app.runtime.upscale_runtime import (
    get_upscale_execution_coordinator,
    get_upscale_runner,
)
from app.services.tryon import (
    _build_specialist_prompt,
    _resize_longest_side,
    _resize_to_long_edge,
)
from app.services.tryon_routing import resolve_tryon_route
from app.utils.media_utils import build_job_media_paths, cleanup_directory
from app.utils.tryon_collage import ProductReferenceInput, build_product_reference

# Routing/prompt only read product.type and product.prompt; the URL is never fetched here.
_PLACEHOLDER_URL = "https://example.com/garment.png"


def _encode_png(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def run_lab_tryon(
    *,
    person_bytes: bytes,
    garments: list[dict[str, Any]],
    steps: int,
    seed: int,
    guidance_scale: float,
    upscale: bool,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """``garments`` is a list of ``{"bytes": ..., "type": ..., "prompt": ...}``."""
    resolved = settings or get_settings()
    if not garments:
        raise ValueError("At least one garment is required.")

    timings: dict[str, float] = {}
    total_started = perf_counter()

    person_image = Image.open(BytesIO(person_bytes)).convert("RGB")
    user_width, user_height = person_image.size

    products: list[TryonProduct] = []
    reference_inputs: list[ProductReferenceInput] = []
    for garment in garments:
        gtype = str(garment["type"]).strip().lower()
        gprompt = " ".join(str(garment.get("prompt", "")).split()).strip() or "garment"
        products.append(TryonProduct(image_url=_PLACEHOLDER_URL, type=gtype, prompt=gprompt))
        gimage = Image.open(BytesIO(garment["bytes"])).convert("RGB")
        reference_inputs.append(ProductReferenceInput(image=gimage, type=gtype))

    reference_started = perf_counter()
    product_reference = build_product_reference(reference_inputs)
    garment_reference_image = _resize_longest_side(
        product_reference.image,
        max_edge=tryon_constants.GARMENT_REFERENCE_MAX_EDGE_PX,
    )
    timings["reference_build_seconds"] = round(perf_counter() - reference_started, 4)

    routing = resolve_tryon_route(products, resolved)
    prompt_text = _build_specialist_prompt(products, routing, resolved)

    qwen_started = perf_counter()
    run_result = get_system_execution_coordinator(resolved).run(
        lambda: get_tryon_runner(resolved).run_tryon(
            person_image=person_image,
            garment_reference_image=garment_reference_image,
            prompt=prompt_text,
            steps=int(steps),
            guidance_scale=float(guidance_scale),
            seed=int(seed),
            output_width=user_width,
            output_height=user_height,
            lora_key=routing.lora_key,
        ),
    )
    timings["qwen_generation_seconds"] = float(run_result.wall_seconds)
    timings["qwen_generation_queued_wall_seconds"] = round(perf_counter() - qwen_started, 4)

    output_image = run_result.image.convert("RGB")
    if output_image.size != (user_width, user_height):
        output_image = output_image.resize((user_width, user_height), Image.Resampling.LANCZOS)
    qwen_output_size = {"width": output_image.width, "height": output_image.height}

    upscale_meta: dict[str, Any] = {"enabled": False}
    if upscale:
        upscale_started = perf_counter()
        job = build_job_media_paths(
            Path(resolved.upscale_work_root) / "tryon-lab",
            input_extension=".png",
            output_extension=".png",
        )
        try:
            output_image.save(job.input_path, format="PNG")
            up_result = get_upscale_execution_coordinator(resolved).run(
                lambda: get_upscale_runner(resolved).run(
                    input_path=job.input_path,
                    output_path=job.output_path,
                    log_path=job.job_dir / "seedvr2.log",
                    target_long_edge=int(resolved.tryon_upscale_target_long_edge),
                ),
            )
            with Image.open(up_result.output_path) as up_img:
                output_image = up_img.convert("RGB")
            before_downscale = {"width": output_image.width, "height": output_image.height}
            output_image = _resize_to_long_edge(
                output_image,
                target_long_edge=int(resolved.tryon_final_output_long_edge),
            )
            timings["seedvr2_upscale_seconds"] = float(up_result.wall_seconds)
            timings["seedvr2_upscale_wall_seconds"] = round(perf_counter() - upscale_started, 4)
            upscale_meta = {
                "enabled": True,
                "model_variant": up_result.model_variant,
                "target_long_edge": int(up_result.target_long_edge),
                "derived_short_edge": int(up_result.derived_short_edge),
                "upscaled_size_before_downscale": before_downscale,
                "final_long_edge": int(resolved.tryon_final_output_long_edge),
            }
        finally:
            cleanup_directory(job.job_dir)

    encode_started = perf_counter()
    output_data_uri = _encode_png(output_image)
    timings["encode_seconds"] = round(perf_counter() - encode_started, 4)
    timings["total_seconds"] = round(perf_counter() - total_started, 4)

    return {
        "ok": True,
        "lora_key": routing.lora_key,
        "prompt": prompt_text,
        "products": [{"type": p.type.value, "prompt": p.prompt} for p in products],
        "input": {"width": user_width, "height": user_height},
        "garment_reference": {
            "width": garment_reference_image.width,
            "height": garment_reference_image.height,
            "mode": product_reference.mode,
        },
        "qwen_output_size": qwen_output_size,
        "output": {
            "width": output_image.width,
            "height": output_image.height,
            "image": output_data_uri,
        },
        "settings": {
            "steps": int(steps),
            "seed": int(seed),
            "guidance_scale": float(guidance_scale),
            "lora_scale": float(resolved.tryon_lora_scale),
            "lora_rank": int(resolved.tryon_lora_rank),
            "lora_alpha": int(resolved.tryon_lora_alpha),
        },
        "upscale": upscale_meta,
        "timings": timings,
    }

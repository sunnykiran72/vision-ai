"""Isolated SeedVR2 A/B latency lab.

Standalone, UNAUTHENTICATED tester (served under the public ``/tools`` prefix)
for comparing SeedVR2 variants and resize/upscale latency WITHOUT touching the
production ``/v1/upscale`` path. Intended to run on a separate test pod that
mounts the same network volume (so all downloaded variants are visible).

Flow per run:
  original image
    -> LANCZOS downscale longest edge to ``pre_resize_max_edge`` (timed)
    -> SeedVR2 upscale with ``target_long_edge = output_max_edge`` (timed)
    -> return both intermediate + final images (base64) and a latency breakdown.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from app.clients.seedvr2 import get_seedvr2_client
from app.config import Settings, get_settings
from app.constants.upscale import (
    KNOWN_SEEDVR2_VARIANT_FILENAMES,
    KNOWN_SEEDVR2_VARIANTS,
)
from app.utils.media_utils import build_job_media_paths, cleanup_directory


@dataclass(frozen=True)
class VariantInfo:
    filename: str
    label: str
    model: str
    precision: str
    approx_size: str
    present: bool
    size_bytes: int | None


def list_variants(settings: Settings | None = None) -> list[VariantInfo]:
    """Return the known SeedVR2 variants annotated with on-disk availability."""
    resolved = settings or get_settings()
    model_dir = Path(resolved.upscale_model_path)
    infos: list[VariantInfo] = []
    for variant in KNOWN_SEEDVR2_VARIANTS:
        path = model_dir / variant["filename"]
        present = path.exists()
        size_bytes = path.stat().st_size if present else None
        infos.append(
            VariantInfo(
                filename=variant["filename"],
                label=variant["label"],
                model=variant["model"],
                precision=variant["precision"],
                approx_size=variant["approx_size"],
                present=present,
                size_bytes=size_bytes,
            )
        )
    return infos


def _resize_longest_edge(image: Image.Image, max_edge: int) -> tuple[Image.Image, bool]:
    width, height = image.size
    longest = max(width, height)
    if max_edge <= 0 or longest <= max_edge:
        return image, False
    scale = float(max_edge) / float(longest)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.LANCZOS), True


def _encode_png(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def run_lab_upscale(
    *,
    image_bytes: bytes,
    model_variant: str,
    pre_resize_max_edge: int,
    output_max_edge: int,
    settings: Settings | None = None,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    model_dir = Path(resolved.upscale_model_path)

    variant = model_variant.strip()
    if variant not in KNOWN_SEEDVR2_VARIANT_FILENAMES:
        raise ValueError(f"Unknown model_variant: {variant!r}")
    variant_path = model_dir / variant
    if not variant_path.exists():
        raise FileNotFoundError(
            f"Variant not downloaded on this pod: {variant_path}"
        )

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    original_width, original_height = image.size

    resize_started = time.perf_counter()
    preprocessed, was_resized = _resize_longest_edge(image, int(pre_resize_max_edge))
    resize_seconds = round(time.perf_counter() - resize_started, 4)
    pre_width, pre_height = preprocessed.size

    job = build_job_media_paths(
        Path(resolved.upscale_work_root),
        input_extension=".png",
        output_extension=".png",
    )
    try:
        preprocessed.save(job.input_path, format="PNG")
        log_path = job.job_dir / "seedvr2.log"

        client = get_seedvr2_client(
            resolved.upscale_model_path,
            variant,
            resolved.upscale_cli_path,
        )

        upscale_started = time.perf_counter()
        run_result = client.run(
            input_path=job.input_path,
            output_path=job.output_path,
            log_path=log_path,
            target_long_edge=int(output_max_edge),
        )
        upscale_seconds = round(time.perf_counter() - upscale_started, 4)

        with Image.open(job.output_path) as raw_output:
            output_image = raw_output.convert("RGB")
            output_width, output_height = output_image.size
            output_data_uri = _encode_png(output_image)
        preprocessed_data_uri = _encode_png(preprocessed)

        return {
            "ok": True,
            "model_variant": variant,
            "backend": run_result.runner_backend,
            "input": {"width": original_width, "height": original_height},
            "preprocessed": {
                "width": pre_width,
                "height": pre_height,
                "was_resized": was_resized,
                "image": preprocessed_data_uri,
            },
            "output": {
                "width": output_width,
                "height": output_height,
                "image": output_data_uri,
            },
            "target": {
                "pre_resize_max_edge": int(pre_resize_max_edge),
                "output_max_edge": int(output_max_edge),
                "derived_short_edge": run_result.derived_short_edge,
            },
            "timings": {
                "resize_seconds": resize_seconds,
                "upscale_seconds": upscale_seconds,
                "runner_wall_seconds": run_result.wall_seconds,
                "total_seconds": round(resize_seconds + upscale_seconds, 4),
            },
        }
    finally:
        cleanup_directory(job.job_dir)

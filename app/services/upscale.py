from __future__ import annotations

from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, UnidentifiedImageError

from app.clients.storage import AzureStorageClient
from app.config import Settings, get_settings
from app.constants import http_status
from app.models.upscale import (
    UpscaleMetric,
    UpscaleRequest,
    UpscaleResponse,
    UpscaleResponseData,
)
from app.runtime.coordinator import QueueFullError, QueueTimeoutError
from app.runtime.upscale_runtime import (
    get_upscale_execution_coordinator,
    get_upscale_runner,
)
from app.utils.media_utils import (
    build_job_media_paths,
    build_storage_object_name,
    cleanup_directory,
    download_media_from_url,
)


def run_upscale_request(
    payload: UpscaleRequest,
    *,
    settings: Settings | None = None,
    user_id: str | None = None,
) -> UpscaleResponse:
    resolved_settings = settings or get_settings()
    job_paths = None
    try:
        downloaded_media = download_media_from_url(str(payload.image_url))
        job_paths = build_job_media_paths(
            Path(resolved_settings.upscale_work_root),
            input_extension=".png",
            output_extension=".png",
        )
        image = Image.open(BytesIO(downloaded_media.content)).convert("RGB")
        input_width, input_height = image.size

        target_long_edge = _resolve_target_long_edge(payload.metric)
        input_long_edge = max(int(input_width), int(input_height))
        input_short_edge = min(int(input_width), int(input_height))
        if input_long_edge <= 0 or input_short_edge <= 0:
            return _error_response(
                http_status.BAD_REQUEST,
                "Invalid input image dimensions.",
                {
                    "feature": "upscale",
                    "image_url": str(payload.image_url),
                },
            )

        action = "upscaled"
        runner_metadata: dict[str, object]
        if input_long_edge >= target_long_edge:
            output_image = image
            runner_metadata = {
                "mode": "skipped",
                "reason": "input_already_meets_target",
                "model_variant": resolved_settings.upscale_model_variant,
                "target_long_edge": target_long_edge,
                "derived_short_edge": input_short_edge,
            }
            output_width, output_height = image.size
            action = "skipped_existing_resolution"
            elapsed_seconds = 0.0
        else:
            # In-memory upscale: tensor in -> tensor out, no PNG/disk round-trip. Verified
            # pixel-identical to the file path at both 2730 (compiled) and 4096 (eager).
            run_result = get_upscale_execution_coordinator(resolved_settings).run(
                lambda: get_upscale_runner(resolved_settings).run_tensor(
                    image=image,
                    target_long_edge=target_long_edge,
                ),
            )
            output_image = run_result.image
            runner_metadata = {
                "mode": "resident_runner_tensor",
                "backend": run_result.runner_backend,
                "model_variant": run_result.model_variant,
                "target_long_edge": run_result.target_long_edge,
                "derived_short_edge": run_result.derived_short_edge,
            }
            output_width = run_result.output_width
            output_height = run_result.output_height
            elapsed_seconds = run_result.wall_seconds

        output_url: str | None = None
        storage_client = AzureStorageClient(resolved_settings)
        if not storage_client.is_configured:
            return _error_response(
                http_status.INTERNAL_SERVER_ERROR,
                "Azure storage is required for upscale output.",
                {
                    "feature": "upscale",
                    "image_url": str(payload.image_url),
                },
            )

        storage_prefix = (
            f"{resolved_settings.upscale_storage_prefix}/"
            f"{user_id or 'anonymous'}/{job_paths.job_id}"
        )
        object_name = build_storage_object_name(
            output_filename=None,
            prefix=storage_prefix,
            default_name="output",
            extension=".jpg",
        )
        # Deliver a JPEG (consistent with wardrobe/try-on, far smaller than a 4k PNG). Encoded once
        # from the in-memory result — no intermediate PNG decode.
        output_buffer = BytesIO()
        output_image.convert("RGB").save(
            output_buffer,
            format="JPEG",
            quality=95,
            subsampling=0,
        )
        output_url = storage_client.upload_bytes(
            output_buffer.getvalue(),
            object_name=object_name,
            content_type="image/jpeg",
        )

        return UpscaleResponse(
            status=http_status.OK,
            message="Image upscale completed successfully.",
            data=UpscaleResponseData(
                url=output_url,
                metadata={
                    "feature": "upscale",
                    "action": action,
                    "input": {
                        "image_url": str(payload.image_url),
                        "width": int(input_width),
                        "height": int(input_height),
                    },
                    "output": {
                        "width": int(output_width),
                        "height": int(output_height),
                    },
                    "settings": {
                        "metric": payload.metric.value,
                        "target_long_edge": int(target_long_edge),
                    },
                    "runner": runner_metadata,
                    "storage": {
                        "uploaded": True,
                        "url": output_url,
                    },
                    "job": {
                        "job_id": job_paths.job_id,
                    },
                    "timings": {
                        "wall_seconds": float(elapsed_seconds),
                    },
                },
            ),
        )
    except QueueFullError as exc:
        return _error_response(
            http_status.SERVICE_UNAVAILABLE,
            "Upscale queue is full.",
            {
                "feature": "upscale",
                "image_url": str(payload.image_url),
                "error": str(exc),
            },
        )
    except QueueTimeoutError as exc:
        return _error_response(
            http_status.GATEWAY_TIMEOUT,
            "Timed out while waiting for upscale execution.",
            {
                "feature": "upscale",
                "image_url": str(payload.image_url),
                "error": str(exc),
            },
        )
    except httpx.HTTPError as exc:
        return _error_response(
            http_status.BAD_REQUEST,
            "Unable to download input image.",
            {
                "feature": "upscale",
                "image_url": str(payload.image_url),
                "error": str(exc),
            },
        )
    except UnidentifiedImageError:
        return _error_response(
            http_status.BAD_REQUEST,
            "Downloaded content is not a valid image.",
            {
                "feature": "upscale",
                "image_url": str(payload.image_url),
            },
        )
    except RuntimeError as exc:
        return _error_response(
            http_status.BAD_REQUEST,
            "No image was generated by the upscale runtime.",
            {
                "feature": "upscale",
                "image_url": str(payload.image_url),
                "error": str(exc),
            },
        )
    except Exception as exc:
        return _error_response(
            http_status.INTERNAL_SERVER_ERROR,
            "Upscale request failed.",
            {
                "feature": "upscale",
                "image_url": str(payload.image_url),
                "error": str(exc),
            },
        )
    finally:
        if job_paths is not None:
            cleanup_directory(job_paths.job_dir)


def _resolve_target_long_edge(metric: UpscaleMetric) -> int:
    if metric == UpscaleMetric.FOUR_K:
        return 4096
    return 2048


def _error_response(status_code: int, message: str, metadata: dict[str, object]) -> UpscaleResponse:
    return UpscaleResponse(
        status=status_code,
        message=message,
        data=UpscaleResponseData(
            url=None,
            metadata=metadata,
        ),
    )

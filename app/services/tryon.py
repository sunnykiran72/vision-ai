from __future__ import annotations

from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, UnidentifiedImageError

from app.clients.storage import AzureStorageClient
from app.config import Settings, get_settings
from app.constants import http_status
from app.models.tryon import TryonRequest, TryonResponse, TryonResponseData
from app.runtime.coordinator import QueueFullError, QueueTimeoutError
from app.runtime.tryon_runtime import (
    get_tryon_execution_coordinator,
    get_tryon_runner,
)
from app.utils.media_utils import (
    build_job_media_paths,
    build_storage_object_name,
    cleanup_directory,
    download_media_from_url,
)
from app.utils.tryon_collage import (
    ProductReferenceInput,
    build_product_reference,
)


def run_tryon_request(
    payload: TryonRequest,
    *,
    settings: Settings | None = None,
    user_id: str | None = None,
) -> TryonResponse:
    resolved_settings = settings or get_settings()
    job_paths = None
    try:
        resolved_seed = (
            int(payload.seed)
            if payload.seed is not None
            else int(resolved_settings.tryon_default_seed)
        )
        resolved_steps = (
            int(payload.steps)
            if payload.steps is not None
            else int(resolved_settings.tryon_default_steps)
        )
        resolved_guidance_scale = (
            float(payload.guidance_scale)
            if payload.guidance_scale is not None
            else float(resolved_settings.tryon_default_guidance_scale)
        )

        user_download = download_media_from_url(str(payload.user_image))
        user_image = Image.open(BytesIO(user_download.content)).convert("RGB")

        product_inputs: list[ProductReferenceInput] = []
        downloaded_products: list[dict[str, str]] = []
        for product in payload.products:
            downloaded = download_media_from_url(str(product.image_url))
            product_image = Image.open(BytesIO(downloaded.content)).convert("RGB")
            product_inputs.append(
                ProductReferenceInput(image=product_image, type=product.type.value),
            )
            downloaded_products.append(
                {
                    "image_url": str(product.image_url),
                    "type": product.type.value,
                    "prompt": product.prompt,
                },
            )

        product_reference = build_product_reference(product_inputs)

        job_paths = build_job_media_paths(Path(resolved_settings.tryon_work_root))
        product_reference.image.save(job_paths.input_path, format="JPEG", quality=95)

        prompt_text = _build_tryon_prompt(payload)
        run_result = get_tryon_execution_coordinator(resolved_settings).run(
            lambda: get_tryon_runner(resolved_settings).run_tryon(
                garment_reference_image=product_reference.image,
                user_image=user_image,
                prompt=prompt_text,
                steps=resolved_steps,
                guidance_scale=resolved_guidance_scale,
                seed=resolved_seed,
            ),
        )

        output_image = run_result.image.convert("RGB")
        output_image.save(job_paths.output_path, format="JPEG", quality=95)

        storage_client = AzureStorageClient(resolved_settings)
        if not storage_client.is_configured:
            return _error_response(
                http_status.INTERNAL_SERVER_ERROR,
                "Azure storage is required for try-on output.",
                {
                    "feature": "tryon",
                    "user_image": str(payload.user_image),
                },
            )

        storage_prefix = (
            f"{resolved_settings.tryon_storage_prefix}/"
            f"{user_id or 'anonymous'}/{job_paths.job_id}"
        )
        object_name = build_storage_object_name(
            output_filename=None,
            prefix=storage_prefix,
            default_name="output",
        )
        output_url = storage_client.upload_file(
            job_paths.output_path,
            object_name=object_name,
            content_type="image/jpeg",
        )

        return TryonResponse(
            status=http_status.OK,
            message="Try-on completed successfully.",
            data=TryonResponseData(
                url=output_url,
                metadata={
                    "feature": "tryon",
                    "request": {
                        "user_image": str(payload.user_image),
                        "product_count": len(payload.products),
                        "products": downloaded_products,
                    },
                    "resolved_settings": {
                        "seed": resolved_seed,
                        "steps": resolved_steps,
                        "guidance_scale": resolved_guidance_scale,
                    },
                    "reference": {
                        "product_reference_mode": product_reference.mode,
                        "input_mode": "separate_garment_and_user_images",
                    },
                    "runner": {
                        **run_result.metadata,
                        "wall_seconds": float(run_result.wall_seconds),
                    },
                    "storage": {
                        "uploaded": True,
                        "url": output_url,
                    },
                    "job": {
                        "job_id": job_paths.job_id,
                    },
                    "output": {
                        "width": int(output_image.width),
                        "height": int(output_image.height),
                    },
                },
            ),
        )
    except QueueFullError as exc:
        return _error_response(
            http_status.SERVICE_UNAVAILABLE,
            "Try-on queue is full.",
            {
                "feature": "tryon",
                "user_image": str(payload.user_image),
                "error": str(exc),
            },
        )
    except QueueTimeoutError as exc:
        return _error_response(
            http_status.GATEWAY_TIMEOUT,
            "Timed out while waiting for try-on execution.",
            {
                "feature": "tryon",
                "user_image": str(payload.user_image),
                "error": str(exc),
            },
        )
    except httpx.HTTPError as exc:
        return _error_response(
            http_status.BAD_REQUEST,
            "Unable to download one or more try-on images.",
            {
                "feature": "tryon",
                "user_image": str(payload.user_image),
                "error": str(exc),
            },
        )
    except UnidentifiedImageError:
        return _error_response(
            http_status.BAD_REQUEST,
            "Downloaded content is not a valid image.",
            {
                "feature": "tryon",
                "user_image": str(payload.user_image),
            },
        )
    except RuntimeError as exc:
        return _error_response(
            http_status.BAD_REQUEST,
            "No image was generated by the try-on runtime.",
            {
                "feature": "tryon",
                "user_image": str(payload.user_image),
                "error": str(exc),
            },
        )
    except Exception as exc:
        return _error_response(
            http_status.INTERNAL_SERVER_ERROR,
            "Try-on request failed.",
            {
                "feature": "tryon",
                "user_image": str(payload.user_image),
                "error": str(exc),
            },
        )
    finally:
        if job_paths is not None:
            cleanup_directory(job_paths.job_dir)


def _build_tryon_prompt(payload: TryonRequest) -> str:
    product_lines = [
        f"{product.type.value}: {product.prompt}"
        for product in payload.products
    ]
    joined_products = "; ".join(product_lines)
    return (
        "Put the provided garments onto the person. "
        "Use the garment reference image and the user image as separate inputs. "
        "Keep the same face, pose, body proportions, and camera framing. "
        f"Products: {joined_products}."
    ).strip()


def _error_response(status_code: int, message: str, metadata: dict[str, object]) -> TryonResponse:
    return TryonResponse(
        status=status_code,
        message=message,
        data=TryonResponseData(
            url=None,
            metadata=metadata,
        ),
    )

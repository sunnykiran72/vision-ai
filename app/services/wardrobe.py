from __future__ import annotations

import base64
import binascii
import time
from io import BytesIO
from typing import Any
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from app.clients.fashion_detection import (
    FashionDetectionRuntimeError,
    get_fashion_detection_client,
)
from app.clients.glamify_progress import GlamifyProgressClient, TimedUploadResult
from app.clients.marqo_fashion import (
    MarqoClassificationRuntimeError,
    get_marqo_fashion_client,
)
from app.clients.minicpm_vllm import MiniCPMRuntimeError, get_minicpm_client
from app.clients.qwen_diffusers_engine import (
    WardrobeDiffusersGenerationError,
    WardrobeDiffusersRuntimeError,
    resize_input_for_model,
)
from app.clients.storage import AzureStorageClient
from app.config import Settings, get_settings
from app.constants import http_status
from app.constants import wardrobe as wardrobe_constants
from app.models.wardrobe import (
    WardrobeAnalyzeResponse,
    WardrobeAnalyzeResult,
    WardrobeGarmentType,
)
from app.runtime.coordinator import QueueFullError, QueueTimeoutError
from app.runtime.system_coordinator import get_system_execution_coordinator
from app.runtime.wardrobe_runtime import get_wardrobe_runner

JPEG_CONTENT_TYPE = "image/jpeg"


class WardrobeValidationError(ValueError):
    pass


def run_wardrobe_request(
    image_bytes: bytes,
    *,
    garment_type: WardrobeGarmentType,
    settings: Settings | None = None,
    user_id: str,
    access_token: str,
) -> WardrobeAnalyzeResponse:
    resolved_settings = settings or get_settings()
    request_started_at = time.perf_counter()
    timings: dict[str, float] = {}
    try:
        garment_type_value = garment_type.value
        stage_started_at = time.perf_counter()
        decoded_image = _load_image_from_bytes(image_bytes)
        _validate_min_dimensions(decoded_image)
        preprocessed = resize_input_for_model(decoded_image)
        timings["preprocess_seconds"] = _elapsed_seconds(stage_started_at)

        storage_client = AzureStorageClient(resolved_settings)
        if not storage_client.is_configured:
            return _error_response(
                http_status.INTERNAL_SERVER_ERROR,
                "Azure storage is required for wardrobe output.",
            )

        coordinator = get_system_execution_coordinator(resolved_settings)

        stage_started_at = time.perf_counter()
        detections = coordinator.run(
            lambda: get_fashion_detection_client().detect(preprocessed),
        )
        timings["fashion_detection_seconds"] = _elapsed_seconds(stage_started_at)
        if not detections:
            return _error_response(
                http_status.BAD_REQUEST,
                "No garment was detected in the image. "
                "Please upload a clear photo of a single garment.",
            )

        # The job id doubles as the final wardrobe id and the storage path key.
        job_id = str(uuid4())
        stage_started_at = time.perf_counter()
        input_jpeg = _image_to_jpeg_bytes(preprocessed)
        timings["input_jpeg_encode_seconds"] = _elapsed_seconds(stage_started_at)
        progress_client = GlamifyProgressClient(resolved_settings)

        # Input upload to the private wardrobe-inputs container runs strictly in the background.
        input_url_future = progress_client.upload_background_timed(
            content=input_jpeg,
            object_name=f"{user_id}/{job_id}/input.jpg",
            container=resolved_settings.azure_wardrobe_input_container,
            content_type=JPEG_CONTENT_TYPE,
        )

        # MiniCPM caption drives the Qwen prompt and is sent to Glamify as promptDescription.
        minicpm_prompt = wardrobe_constants.MINICPM_PROMPT_BY_TYPE[garment_type_value]
        stage_started_at = time.perf_counter()
        caption = coordinator.run(
            lambda: get_minicpm_client().describe_garment(
                image=preprocessed,
                prompt=minicpm_prompt,
            ),
        ).text
        timings["minicpm_caption_seconds"] = _elapsed_seconds(stage_started_at)
        extraction_prompt = wardrobe_constants.QWEN_EXTRACT_PROMPT_TEMPLATE_BY_TYPE[
            garment_type_value
        ].format(caption=caption)

        stage_started_at = time.perf_counter()
        run_result = coordinator.run(
            lambda: get_wardrobe_runner(resolved_settings).run_extract(
                input_image=preprocessed,
                prompt=extraction_prompt,
                garment_type=garment_type_value,
            ),
        )
        timings["qwen_generation_seconds"] = float(run_result.wall_seconds)
        timings["qwen_generation_queued_wall_seconds"] = _elapsed_seconds(stage_started_at)
        output_image = run_result.image.convert("RGB")
        stage_started_at = time.perf_counter()
        output_jpeg = _image_to_jpeg_bytes(output_image)
        timings["output_jpeg_encode_seconds"] = _elapsed_seconds(stage_started_at)

        # Start the output upload immediately so it overlaps Marqo classification.
        output_url_future = progress_client.upload_background_timed(
            content=output_jpeg,
            object_name=f"{user_id}/{job_id}/output.jpg",
            container=resolved_settings.azure_wardrobe_output_container,
            content_type=JPEG_CONTENT_TYPE,
        )

        stage_started_at = time.perf_counter()
        marqo_result = coordinator.run(
            lambda: get_marqo_fashion_client().classify(
                image=output_image,
                garment_type=garment_type_value,
            ),
        )
        timings["marqo_classification_seconds"] = _elapsed_seconds(stage_started_at)
        category_key, category_label, category_score, category_source = (
            _resolve_wardrobe_category(
                garment_type=garment_type_value,
                category_key=marqo_result.category_key,
                category_label=marqo_result.category_label,
                score=marqo_result.score,
                applied=marqo_result.applied,
            )
        )

        # Join the output upload (its URL is the response payload).
        try:
            output_wait_started_at = time.perf_counter()
            output_upload = output_url_future.result(
                timeout=wardrobe_constants.AZURE_UPLOAD_TIMEOUT_SECONDS,
            )
            timings["output_upload_wait_seconds"] = _elapsed_seconds(output_wait_started_at)
        except Exception:
            return _error_response(
                http_status.INTERNAL_SERVER_ERROR,
                "Failed to upload the extracted garment image.",
            )
        output_upload_metadata = _timed_upload_metadata(output_upload)
        output_url = output_upload_metadata["url"]

        input_upload_metadata = _pending_upload_metadata()
        try:
            input_wait_started_at = time.perf_counter()
            input_upload = input_url_future.result(
                timeout=wardrobe_constants.AZURE_UPLOAD_TIMEOUT_SECONDS,
            )
            timings["input_upload_wait_seconds"] = _elapsed_seconds(input_wait_started_at)
            input_upload_metadata = _timed_upload_metadata(input_upload)
        except Exception as exc:
            timings["input_upload_wait_seconds"] = _elapsed_seconds(input_wait_started_at)
            input_upload_metadata = {
                "status": "failed",
                "error": str(exc),
                "container": resolved_settings.azure_wardrobe_input_container,
                "object_name": f"{user_id}/{job_id}/input.jpg",
            }

        generation_metadata = {
            "prompt": extraction_prompt,
            "requested_type": garment_type_value,
            "minicpm_prompt": minicpm_prompt,
            "prompt_description": caption,
            "runtime": run_result.metadata,
        }
        classification = {
            "primary_category": _primary_category_for_marqo_result(
                garment_type_value,
                category_key,
            ),
            "category": category_key,
            "category_label": _display_category_label(category_label),
            "score": category_score,
            "source": category_source,
        }
        marqo_metadata = {
            "model": wardrobe_constants.MARQO_MODEL_ID,
            "threshold": marqo_result.min_confidence,
            "applied": marqo_result.applied,
            "reason": marqo_result.reason,
            "top_matches": marqo_result.top_matches,
        }
        uploads_metadata = {
            "input": input_upload_metadata,
            "output": output_upload_metadata,
        }
        progress_metadata = {
            "classification": classification,
            "marqo": marqo_metadata,
            **generation_metadata,
            "uploads": uploads_metadata,
            "timings": timings,
        }
        progress_payload = {
            "id": job_id,
            "inputImage": input_upload_metadata.get("url"),
            "outputImage": output_url,
            "promptDescription": caption,
            "metadata": progress_metadata,
        }
        timings["total_wall_seconds"] = _elapsed_seconds(request_started_at)
        response_metadata = {
            "feature": "wardrobe",
            "id": job_id,
            "requested_type": garment_type_value,
            "promptDescription": caption,
            "prompt": extraction_prompt,
            "minicpm_prompt": minicpm_prompt,
            "classification": classification,
            "marqo": marqo_metadata,
            "runtime": run_result.metadata,
            "uploads": uploads_metadata,
            "timings": timings,
            "progress": {
                "configured": progress_client.is_configured,
                "queued": progress_client.is_configured,
                "payload": progress_payload,
            },
            "detections": detections,
            "sizes": {
                "input": {"width": decoded_image.width, "height": decoded_image.height},
                "preprocessed": {"width": preprocessed.width, "height": preprocessed.height},
                "output": {"width": output_image.width, "height": output_image.height},
            },
        }
        # Glamify progress sync runs in the background so it never blocks the response.
        progress_client.submit_progress_background(
            access_token=access_token,
            progress_id=job_id,
            input_url_future=input_url_future,
            output_url=output_url,
            prompt_description=caption,
            classification=classification,
            marqo=marqo_metadata,
            metadata=generation_metadata | {"uploads": uploads_metadata, "timings": timings},
        )

        return WardrobeAnalyzeResponse(
            status=http_status.OK,
            message="",
            data=WardrobeAnalyzeResult(
                id=job_id,
                type=garment_type,
                image=output_url,
                category=category_key,
                categoryLabel=_display_category_label(category_label),
                metadata=response_metadata,
            ),
        )
    except WardrobeValidationError as exc:
        return _error_response(http_status.UNPROCESSABLE_CONTENT, str(exc))
    except QueueFullError:
        return _error_response(
            http_status.SERVICE_UNAVAILABLE,
            "The system is busy. Please try again shortly.",
        )
    except QueueTimeoutError:
        return _error_response(
            http_status.SERVICE_UNAVAILABLE,
            "Timed out while waiting for an execution slot. Please try again.",
        )
    except MiniCPMRuntimeError:
        return _error_response(
            http_status.INTERNAL_SERVER_ERROR,
            "Garment description failed. Please try again.",
        )
    except (FashionDetectionRuntimeError, MarqoClassificationRuntimeError):
        return _error_response(
            http_status.INTERNAL_SERVER_ERROR,
            "Wardrobe validation runtime failed.",
        )
    except (WardrobeDiffusersGenerationError, WardrobeDiffusersRuntimeError):
        return _error_response(
            http_status.INTERNAL_SERVER_ERROR,
            "Wardrobe runtime failed to initialize or execute.",
        )
    except Exception:
        return _error_response(http_status.INTERNAL_SERVER_ERROR, "Wardrobe request failed.")


def _decode_base64_image(raw_image: str) -> Image.Image:
    raw = str(raw_image or "").strip()
    if not raw:
        raise WardrobeValidationError("image is required.")
    if raw.startswith("data:"):
        header, separator, data = raw.partition(",")
        if not separator:
            raise WardrobeValidationError("Invalid image data URL.")
        mime_type = header.split(";", 1)[0].removeprefix("data:").strip().lower()
        if mime_type not in wardrobe_constants.ALLOWED_IMAGE_MIME_TYPES:
            raise WardrobeValidationError("Only PNG and JPEG images are supported.")
        raw = data
    try:
        content = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise WardrobeValidationError("image must be valid base64.") from exc
    return _load_image_from_bytes(content)


def _load_image_from_bytes(content: bytes) -> Image.Image:
    if not content:
        raise WardrobeValidationError("No image file was provided.")
    try:
        image = Image.open(BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise WardrobeValidationError(
            "The uploaded file is not a valid image. Please upload a PNG or JPEG.",
        ) from exc
    if str(image.format or "").upper() not in wardrobe_constants.ALLOWED_IMAGE_FORMATS:
        raise WardrobeValidationError(
            "Unsupported image format. Only PNG and JPEG images are supported.",
        )
    return image.convert("RGB")


def _validate_min_dimensions(image: Image.Image) -> None:
    if (
        image.width < wardrobe_constants.MIN_IMAGE_EDGE_PX
        or image.height < wardrobe_constants.MIN_IMAGE_EDGE_PX
    ):
        raise WardrobeValidationError(
            "Image is too small. Width and height must both be at least "
            f"{wardrobe_constants.MIN_IMAGE_EDGE_PX}px.",
        )


def _image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _elapsed_seconds(started_at: float) -> float:
    return float(round(time.perf_counter() - started_at, 3))


def _timed_upload_metadata(upload: TimedUploadResult | str) -> dict[str, Any]:
    if isinstance(upload, TimedUploadResult):
        return {
            "status": "completed",
            "url": upload.url,
            "wall_seconds": upload.wall_seconds,
            "container": upload.container,
            "object_name": upload.object_name,
            "bytes": upload.bytes,
        }
    return {
        "status": "completed",
        "url": str(upload),
    }


def _pending_upload_metadata() -> dict[str, Any]:
    return {"status": "pending"}


def _primary_category_for_marqo_result(garment_type: str, category_key: str) -> str:
    for candidate in wardrobe_constants.MARQO_CANDIDATES_BY_TYPE.get(garment_type, ()):
        if candidate.key == category_key:
            return candidate.parent_key
    return _primary_category_key_for_type(garment_type)


def _primary_category_key_for_type(garment_type: str) -> str:
    if garment_type == "bottom":
        return "bottoms"
    if garment_type == "dress":
        return "dresses"
    return "tops"


def _resolve_wardrobe_category(
    *,
    garment_type: str,
    category_key: str,
    category_label: str,
    score: float,
    applied: bool,
) -> tuple[str, str, float, str]:
    if category_key and category_label:
        source = "marqo" if applied else "marqo_low_confidence"
        return category_key, category_label, float(score), source

    candidates = wardrobe_constants.MARQO_CANDIDATES_BY_TYPE.get(str(garment_type), ())
    if candidates:
        candidate = candidates[0]
        return candidate.key, candidate.label, 0.0, "default_candidate_fallback"

    fallback_label = _display_category_label(garment_type)
    return str(garment_type), fallback_label, 0.0, "requested_type_fallback"


def _display_category_label(label: str) -> str:
    normalized = " ".join(str(label or "").replace("_", " ").split()).strip()
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]


def _error_response(status_code: int, message: str) -> WardrobeAnalyzeResponse:
    return WardrobeAnalyzeResponse(status=status_code, message=message, data=None)

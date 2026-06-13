from __future__ import annotations

import time
from io import BytesIO
from typing import Any
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from app.clients.person_detection import PersonDetectionRuntimeError, get_person_detection_client
from app.clients.storage import AzureStorageClient
from app.config import Settings, get_settings
from app.constants import http_status
from app.constants import user_validation as constants
from app.models.user_validation import UserValidationResponse, UserValidationResult
from app.runtime.coordinator import QueueFullError, QueueTimeoutError
from app.runtime.system_coordinator import get_system_execution_coordinator

JPEG_CONTENT_TYPE = "image/jpeg"


class UserImageValidationError(ValueError):
    pass


def run_user_validation_request(
    image_bytes: bytes,
    *,
    filename: str,
    content_type: str | None,
    user_id: str,
    settings: Settings | None = None,
) -> UserValidationResponse:
    resolved_settings = settings or get_settings()
    request_started_at = time.perf_counter()
    timings: dict[str, float] = {}
    try:
        stage_started_at = time.perf_counter()
        decoded = _load_image_from_bytes(image_bytes, content_type=content_type)
        normalized = _normalize_user_image(decoded)
        timings["preprocess_seconds"] = _elapsed_seconds(stage_started_at)

        stage_started_at = time.perf_counter()
        blur = _compute_blur_metadata(normalized)
        timings["blur_seconds"] = _elapsed_seconds(stage_started_at)

        coordinator = get_system_execution_coordinator(resolved_settings)
        stage_started_at = time.perf_counter()
        detector = get_person_detection_client()
        detections = coordinator.run(lambda: detector.detect(normalized))
        timings["person_detection_seconds"] = _elapsed_seconds(stage_started_at)
        validation = _validate_person_image(detections=detections, blur=blur)
        if not validation["accepted"]:
            timings["total_wall_seconds"] = _elapsed_seconds(request_started_at)
            return _error_response(
                http_status.UNPROCESSABLE_CONTENT,
                str(validation["message"]),
                metadata={
                    "feature": "user_validation",
                    "filename": filename,
                    "validation": validation,
                    "detections": detections,
                    "blur": blur,
                    "sizes": _sizes_metadata(decoded, normalized),
                    "timings": timings,
                },
            )

        storage_client = AzureStorageClient(resolved_settings)
        if not storage_client.is_configured:
            return _error_response(
                http_status.INTERNAL_SERVER_ERROR,
                "Azure storage is required for user image upload.",
            )

        job_id = str(uuid4())
        stage_started_at = time.perf_counter()
        output_jpeg = _image_to_jpeg_bytes(normalized)
        timings["jpeg_encode_seconds"] = _elapsed_seconds(stage_started_at)

        object_name = f"{constants.STORAGE_PREFIX}/{user_id}/{job_id}/input.jpg"
        stage_started_at = time.perf_counter()
        image_url = storage_client.upload_bytes(
            output_jpeg,
            object_name=object_name,
            container=resolved_settings.azure_user_image_container,
            content_type=JPEG_CONTENT_TYPE,
        )
        timings["upload_seconds"] = _elapsed_seconds(stage_started_at)
        timings["total_wall_seconds"] = _elapsed_seconds(request_started_at)

        return UserValidationResponse(
            status=http_status.OK,
            message="",
            data=UserValidationResult(
                image=image_url,
                metadata={
                    "feature": "user_validation",
                    "id": job_id,
                    "filename": filename,
                    "validation": validation,
                    "person_detection": {
                        "model": detector.model_id,
                        "device": detector.device,
                        "dtype": detector.dtype,
                        "detections": detections,
                    },
                    "blur": blur,
                    "sizes": _sizes_metadata(decoded, normalized),
                    "upload": {
                        "container": resolved_settings.azure_user_image_container,
                        "object_name": object_name,
                        "url": image_url,
                        "bytes": len(output_jpeg),
                    },
                    "timings": timings,
                },
            ),
        )
    except UserImageValidationError as exc:
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
    except PersonDetectionRuntimeError:
        return _error_response(
            http_status.INTERNAL_SERVER_ERROR,
            "User image validation failed. Please try again.",
        )
    except Exception:
        return _error_response(http_status.INTERNAL_SERVER_ERROR, "User image upload failed.")


def _load_image_from_bytes(content: bytes, *, content_type: str | None) -> Image.Image:
    if not content:
        raise UserImageValidationError("No image file was provided.")
    normalized_content_type = (content_type or "").split(";", 1)[0].strip().lower()
    if (
        normalized_content_type
        and normalized_content_type not in constants.ALLOWED_IMAGE_MIME_TYPES
    ):
        raise UserImageValidationError(
            "Unsupported image format. Only PNG, JPEG, and WebP images are supported.",
        )
    try:
        image = Image.open(BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise UserImageValidationError(
            "The uploaded file is not a valid image. Please upload a PNG, JPEG, or WebP.",
        ) from exc
    if str(image.format or "").upper() not in constants.ALLOWED_IMAGE_FORMATS:
        raise UserImageValidationError(
            "Unsupported image format. Only PNG, JPEG, and WebP images are supported.",
        )
    return image.convert("RGB")


def _normalize_user_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise UserImageValidationError("Invalid image dimensions.")
    long_edge = max(width, height)
    scale = constants.NORMALIZED_LONG_EDGE_PX / long_edge
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    if min(new_width, new_height) < constants.MIN_NORMALIZED_EDGE_PX:
        raise UserImageValidationError(
            "Image aspect ratio is too extreme. "
            "Please upload a clearer portrait or full-body photo.",
        )
    # Enforce the EXACT try-on training dimension (832x1248, portrait 2:3): center-crop to the 2:3
    # aspect, then resize. Inputs already arrive as 2:3 from the frontend, so for real traffic this
    # is a no-op (a 2:3 image normalizes to exactly 832x1248 either way). It guarantees every try-on
    # input is exactly 832x1248 -> the inline SeedVR2 upscale output is always the prewarmed
    # 1820x2730 shape (never a cold recompile). Validation-only; the try-on path is untouched.
    return _center_crop_to_aspect_and_resize(
        image.convert("RGB"),
        constants.NORMALIZED_TARGET_WIDTH,
        constants.NORMALIZED_TARGET_HEIGHT,
    )


def _center_crop_to_aspect_and_resize(
    image: Image.Image, target_width: int, target_height: int
) -> Image.Image:
    """Center-crop ``image`` to the target aspect ratio, then resize to (target_width,
    target_height). A true-aspect input (within rounding tolerance) skips the crop entirely."""
    width, height = image.size
    target_ratio = target_width / target_height
    current_ratio = width / height
    if abs(current_ratio - target_ratio) > 1e-3:
        if current_ratio > target_ratio:  # too wide -> trim width
            crop_width = max(1, round(height * target_ratio))
            crop_height = height
        else:  # too tall -> trim height
            crop_width = width
            crop_height = max(1, round(width / target_ratio))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        image = image.crop((left, top, left + crop_width, top + crop_height))
    if image.size != (target_width, target_height):
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return image


def _compute_blur_metadata(image: Image.Image) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise PersonDetectionRuntimeError(
            f"Unable to import blur detection dependencies: {exc}",
        ) from exc

    gray = np.asarray(image.convert("L"))
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "method": "opencv_laplacian_variance",
        "score": round(score, 3),
        "threshold": constants.BLUR_SCORE_THRESHOLD,
    }


def _validate_person_image(
    *,
    detections: list[dict[str, object]],
    blur: dict[str, Any],
) -> dict[str, Any]:
    person_boxes = [_normalize_detection(item) for item in detections if _is_person(item)]
    primary = person_boxes[0] if person_boxes else None
    secondary_large_people = [
        item
        for item in person_boxes[1:]
        if item["score"] >= constants.SECONDARY_LARGE_PERSON_SCORE_THRESHOLD
        and item["metrics"]["area_ratio"] >= constants.SECONDARY_LARGE_PERSON_AREA_RATIO
    ]
    blur_score = blur.get("score")
    checks: dict[str, bool] = {
        "person_present": primary is not None,
        "primary_person_score": False,
        "height_ratio": False,
        "area_ratio": False,
        "bottom_ratio": False,
        "not_blurry": False,
        "no_secondary_large_person": not secondary_large_people,
    }
    reasons: list[str] = []

    if primary is None:
        reasons.append("No person was detected in the image.")
    else:
        checks["primary_person_score"] = (
            primary["score"] >= constants.PRIMARY_PERSON_SCORE_THRESHOLD
        )
        checks["height_ratio"] = (
            primary["metrics"]["height_ratio"] >= constants.PRIMARY_PERSON_MIN_HEIGHT_RATIO
        )
        checks["area_ratio"] = (
            primary["metrics"]["area_ratio"] >= constants.PRIMARY_PERSON_MIN_AREA_RATIO
        )
        checks["bottom_ratio"] = (
            primary["metrics"]["bottom_ratio"] >= constants.PRIMARY_PERSON_MIN_BOTTOM_RATIO
        )
        if not checks["primary_person_score"]:
            reasons.append("Person detection confidence is too low.")
        if not checks["height_ratio"]:
            reasons.append("The person is too small vertically in the image.")
        if not checks["area_ratio"]:
            reasons.append("The person is too small in the frame.")
        if not checks["bottom_ratio"]:
            reasons.append("The visible body is cropped too high for try-on.")

    if blur_score is not None:
        checks["not_blurry"] = float(blur_score) >= constants.BLUR_SCORE_THRESHOLD
    if not checks["not_blurry"]:
        reasons.append("The image is too blurry for try-on.")

    if secondary_large_people:
        reasons.append("More than one large person is visible in the image.")

    accepted = all(checks.values())
    message = (
        "User image is suitable for try-on."
        if accepted
        else reasons[0] if reasons else "User image is invalid."
    )
    return {
        "accepted": accepted,
        "message": message,
        "reasons": reasons,
        "checks": checks,
        "thresholds": {
            "primary_person_score": constants.PRIMARY_PERSON_SCORE_THRESHOLD,
            "height_ratio": constants.PRIMARY_PERSON_MIN_HEIGHT_RATIO,
            "area_ratio": constants.PRIMARY_PERSON_MIN_AREA_RATIO,
            "bottom_ratio": constants.PRIMARY_PERSON_MIN_BOTTOM_RATIO,
            "blur_score": constants.BLUR_SCORE_THRESHOLD,
            "secondary_large_person_score": constants.SECONDARY_LARGE_PERSON_SCORE_THRESHOLD,
            "secondary_large_person_area_ratio": constants.SECONDARY_LARGE_PERSON_AREA_RATIO,
        },
        "primary_person": primary,
        "person_count": len(person_boxes),
        "secondary_large_person_count": len(secondary_large_people),
    }


def _is_person(detection: dict[str, object]) -> bool:
    return str(detection.get("label") or "").strip().lower() in {"person", "human"}


def _normalize_detection(detection: dict[str, object]) -> dict[str, Any]:
    metrics = detection.get("metrics") if isinstance(detection.get("metrics"), dict) else {}
    return {
        "label": str(detection.get("label") or ""),
        "score": float(detection.get("score") or 0.0),
        "box": detection.get("box") or {},
        "metrics": {
            "width_ratio": float(metrics.get("width_ratio") or 0.0),
            "height_ratio": float(metrics.get("height_ratio") or 0.0),
            "area_ratio": float(metrics.get("area_ratio") or 0.0),
            "top_ratio": float(metrics.get("top_ratio") or 0.0),
            "bottom_ratio": float(metrics.get("bottom_ratio") or 0.0),
        },
    }


def _image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=constants.JPEG_QUALITY)
    return buffer.getvalue()


def _sizes_metadata(original: Image.Image, normalized: Image.Image) -> dict[str, dict[str, int]]:
    return {
        "input": {"width": int(original.width), "height": int(original.height)},
        "normalized": {"width": int(normalized.width), "height": int(normalized.height)},
    }


def _elapsed_seconds(started_at: float) -> float:
    return round(time.perf_counter() - started_at, 3)


def _error_response(
    status_code: int,
    message: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> UserValidationResponse:
    del metadata
    return UserValidationResponse(
        status=status_code,
        message=message,
        data=None,
    )

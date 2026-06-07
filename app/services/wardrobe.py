from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from app.clients.fashion_detection import FashionDetectionClient, FashionDetectionRuntimeError
from app.clients.glamify_progress import GlamifyProgressClient
from app.clients.marqo_fashion import MarqoClassificationRuntimeError, MarqoFashionSiglipClient
from app.clients.qwen_wardrobe_aitk import WardrobeGenerationError, WardrobeRuntimeError
from app.config import Settings, get_settings
from app.constants import http_status
from app.constants import wardrobe as wardrobe_constants
from app.models.wardrobe import (
    WardrobeAnalyzeRequest,
    WardrobeAnalyzeResponse,
    WardrobeAnalyzeResult,
)
from app.runtime.coordinator import QueueFullError, QueueTimeoutError
from app.runtime.wardrobe_runtime import (
    get_wardrobe_execution_coordinator,
    get_wardrobe_runner,
)
from app.utils.media_utils import build_storage_object_name, cleanup_directory, ensure_directory

JPEG_CONTENT_TYPE = "image/jpeg"


class WardrobeValidationError(ValueError):
    pass


@dataclass(frozen=True)
class WardrobeJobPaths:
    job_id: str
    job_dir: Path
    input_path: Path
    output_path: Path


def run_wardrobe_request(
    payload: WardrobeAnalyzeRequest,
    *,
    settings: Settings | None = None,
    user_id: str,
    bearer_token: str,
) -> WardrobeAnalyzeResponse:
    resolved_settings = settings or get_settings()
    job_paths: WardrobeJobPaths | None = None
    try:
        garment_type = payload.type.value
        prompt = _resolve_generation_prompt(payload.prompt, garment_type)
        decoded_image = _decode_base64_image(payload.image)
        _validate_min_dimensions(decoded_image)
        preprocessed = _resize_to_max_edge(decoded_image, wardrobe_constants.PREPROCESS_MAX_EDGE_PX)
        coordinator = get_wardrobe_execution_coordinator(resolved_settings)

        detector = get_fashion_detection_client()
        detections = coordinator.run(lambda: detector.detect(preprocessed))
        if not detections:
            return _error_response(http_status.BAD_REQUEST, "No garment was detected in the image.")

        job_paths = _build_wardrobe_job_paths(Path(resolved_settings.wardrobe_work_root))
        input_jpeg = _image_to_jpeg_bytes(preprocessed)
        job_paths.input_path.write_bytes(input_jpeg)
        storage_prefix = (
            f"{resolved_settings.wardrobe_storage_prefix}/"
            f"{user_id}/{job_paths.job_id}"
        )
        input_object_name = build_storage_object_name(
            output_filename=None,
            prefix=storage_prefix,
            default_name="input",
        )
        output_object_name = build_storage_object_name(
            output_filename=None,
            prefix=storage_prefix,
            default_name="output",
        )
        progress_client = GlamifyProgressClient(resolved_settings)
        input_url_future = progress_client.upload_input_background(
            content=input_jpeg,
            object_name=input_object_name,
            content_type=JPEG_CONTENT_TYPE,
        )

        run_result = coordinator.run(
            lambda: get_wardrobe_runner(resolved_settings).run_extract(
                input_image_path=str(job_paths.input_path),
                prompt=prompt,
                garment_type=garment_type,
                output_path=str(job_paths.output_path),
            ),
        )

        output_image = run_result.image.convert("RGB")
        output_jpeg = _image_to_jpeg_bytes(output_image)
        job_paths.output_path.write_bytes(output_jpeg)
        marqo_result = coordinator.run(
            lambda: get_marqo_fashion_client().classify(
                image=output_image,
                garment_type=garment_type,
            ),
        )
        category_key, category_label, category_score, category_source = (
            _resolve_wardrobe_category(
                garment_type=garment_type,
                category_key=marqo_result.category_key,
                category_label=marqo_result.category_label,
                score=marqo_result.score,
                applied=marqo_result.applied,
            )
        )

        metadata = {
            "prompt": prompt,
            "prompt_source": "override" if _has_prompt_override(payload.prompt) else "default",
            "requested_type": garment_type,
        }
        classification = {
            "primary_category": _primary_category_for_marqo_result(
                garment_type,
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
        progress_client.submit_output_and_progress_background(
            bearer_token=bearer_token,
            progress_id=job_paths.job_id,
            input_url_future=input_url_future,
            output_content=output_jpeg,
            output_object_name=output_object_name,
            output_content_type=JPEG_CONTENT_TYPE,
            classification=classification,
            marqo=marqo_metadata,
            metadata=metadata,
        )

        return WardrobeAnalyzeResponse(
            status=http_status.OK,
            message="",
            data=WardrobeAnalyzeResult(
                id=job_paths.job_id,
                type=payload.type,
                image=base64.b64encode(output_jpeg).decode("ascii"),
                category=category_key,
                categoryLabel=_display_category_label(category_label),
            ),
        )
    except WardrobeValidationError as exc:
        return _error_response(http_status.UNPROCESSABLE_CONTENT, str(exc))
    except QueueFullError:
        return _error_response(http_status.SERVICE_UNAVAILABLE, "Wardrobe queue is full.")
    except QueueTimeoutError:
        return _error_response(
            http_status.SERVICE_UNAVAILABLE,
            "Timed out while waiting for wardrobe execution.",
        )
    except (FashionDetectionRuntimeError, MarqoClassificationRuntimeError):
        return _error_response(
            http_status.INTERNAL_SERVER_ERROR,
            "Wardrobe validation runtime failed.",
        )
    except (WardrobeGenerationError, WardrobeRuntimeError):
        return _error_response(
            http_status.INTERNAL_SERVER_ERROR,
            "Wardrobe runtime failed to initialize or execute.",
        )
    except Exception:
        return _error_response(http_status.INTERNAL_SERVER_ERROR, "Wardrobe request failed.")
    finally:
        if job_paths is not None:
            cleanup_directory(job_paths.job_dir)


@lru_cache(maxsize=1)
def get_fashion_detection_client() -> FashionDetectionClient:
    return FashionDetectionClient()


@lru_cache(maxsize=1)
def get_marqo_fashion_client() -> MarqoFashionSiglipClient:
    return MarqoFashionSiglipClient()


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
    try:
        image = Image.open(BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise WardrobeValidationError("image must be a valid PNG or JPEG image.") from exc
    if str(image.format or "").upper() not in wardrobe_constants.ALLOWED_IMAGE_FORMATS:
        raise WardrobeValidationError("Only PNG and JPEG images are supported.")
    return image.convert("RGB")


def _validate_min_dimensions(image: Image.Image) -> None:
    if (
        image.width < wardrobe_constants.MIN_IMAGE_EDGE_PX
        or image.height < wardrobe_constants.MIN_IMAGE_EDGE_PX
    ):
        raise WardrobeValidationError(
            "Image width and height must both be at least "
            f"{wardrobe_constants.MIN_IMAGE_EDGE_PX}px.",
        )


def _resize_to_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    width, height = int(image.width), int(image.height)
    longest = max(width, height)
    if longest <= max_edge:
        return image.copy()
    scale = float(max_edge) / float(longest)
    resized = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(resized, Image.Resampling.LANCZOS)


def _image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _resolve_generation_prompt(prompt: str | None, garment_type: str) -> str:
    if _has_prompt_override(prompt):
        return " ".join(str(prompt).split()).strip()
    return wardrobe_constants.PROMPT_BY_TYPE[garment_type]


def _has_prompt_override(prompt: str | None) -> bool:
    return bool(str(prompt or "").strip())


def _build_wardrobe_job_paths(root_dir: Path) -> WardrobeJobPaths:
    job_id = str(uuid4())
    job_dir = ensure_directory(root_dir / job_id)
    return WardrobeJobPaths(
        job_id=job_id,
        job_dir=job_dir,
        input_path=job_dir / "input.jpg",
        output_path=job_dir / "output.jpg",
    )


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

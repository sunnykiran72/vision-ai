from __future__ import annotations

from functools import lru_cache

from app.clients.minicpm_v46 import MiniCPMRuntimeError, MiniCPMV46Client
from app.constants import http_status
from app.constants import minicpm as minicpm_constants
from app.models.minicpm import (
    MiniCPMGarmentRequest,
    MiniCPMGarmentResponse,
    MiniCPMGarmentResult,
)
from app.services.wardrobe import WardrobeValidationError, _decode_base64_image


def run_minicpm_garment_request(
    payload: MiniCPMGarmentRequest,
) -> MiniCPMGarmentResponse:
    try:
        garment_type = payload.type.value
        image = _decode_base64_image(payload.image)
        prompt = _resolve_prompt(payload.prompt, garment_type)
        result = get_minicpm_v46_client().describe_garment(
            image=image,
            prompt=prompt,
        )
        return MiniCPMGarmentResponse(
            status=http_status.OK,
            message="",
            data=MiniCPMGarmentResult(
                type=payload.type,
                description=result.text,
                prompt=prompt,
                model=result.model_id,
                metadata={
                    "latency_ms": result.latency_ms,
                    "device": result.device,
                    "dtype": result.dtype,
                    "downsample_mode": result.downsample_mode,
                    "max_new_tokens": result.max_new_tokens,
                    "max_slice_nums": result.max_slice_nums,
                    "prompt_source": "override" if _has_prompt(payload.prompt) else "default",
                },
            ),
        )
    except WardrobeValidationError as exc:
        return _error_response(http_status.UNPROCESSABLE_CONTENT, str(exc))
    except MiniCPMRuntimeError:
        return _error_response(http_status.INTERNAL_SERVER_ERROR, "MiniCPM runtime failed.")
    except Exception:
        return _error_response(http_status.INTERNAL_SERVER_ERROR, "MiniCPM request failed.")


@lru_cache(maxsize=1)
def get_minicpm_v46_client() -> MiniCPMV46Client:
    return MiniCPMV46Client()


def _resolve_prompt(prompt: str | None, garment_type: str) -> str:
    if _has_prompt(prompt):
        return " ".join(str(prompt).split()).strip()
    return minicpm_constants.PROMPT_BY_TYPE[garment_type]


def _has_prompt(prompt: str | None) -> bool:
    return bool(str(prompt or "").strip())


def _error_response(status: int, message: str) -> MiniCPMGarmentResponse:
    return MiniCPMGarmentResponse(status=status, message=message, data=None)

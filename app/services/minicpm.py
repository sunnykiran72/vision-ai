from __future__ import annotations

from app.clients.minicpm_vllm import MiniCPMRuntimeError, get_minicpm_client
from app.constants import http_status
from app.constants import wardrobe as wardrobe_constants
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
        result = get_minicpm_client().describe_garment(
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


def _resolve_prompt(prompt: str | None, garment_type: str) -> str:
    if _has_prompt(prompt):
        return " ".join(str(prompt).split()).strip()
    return wardrobe_constants.MINICPM_PROMPT_BY_TYPE[garment_type]


def _has_prompt(prompt: str | None) -> bool:
    return bool(str(prompt or "").strip())


def _error_response(status: int, message: str) -> MiniCPMGarmentResponse:
    return MiniCPMGarmentResponse(status=status, message=message, data=None)

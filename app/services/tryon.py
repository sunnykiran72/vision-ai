from __future__ import annotations

from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, UnidentifiedImageError

from app.clients.qwen_tryon_aitk import TryonGenerationError, TryonRuntimeError
from app.clients.storage import AzureStorageClient
from app.config import Settings, get_settings
from app.constants import http_status
from app.models.tryon import TryonProduct, TryonRequest, TryonResponse, TryonResponseData
from app.runtime.coordinator import QueueFullError, QueueTimeoutError
from app.runtime.tryon_runtime import (
    get_tryon_execution_coordinator,
    get_tryon_runner,
)
from app.services.tryon_routing import TryonRoutingDecision, resolve_tryon_route
from app.utils.media_utils import (
    build_storage_object_name,
    build_tryon_job_media_paths,
    cleanup_directory,
    download_media_from_url,
)
from app.utils.tryon_collage import (
    ProductReferenceInput,
    build_product_reference,
)

# Legacy free-form prompt sections — used only when TRYON_USE_SPECIALISTS=false.
TRYON_SINGLE_REFERENCE_PROMPT = "Apply the reference garment from image 2 to the person in image 1."
TRYON_MULTI_REFERENCE_PROMPT = "Apply the reference garments from image 2 to the person in image 1."
TRYON_IDENTITY_CLAUSE = (
    "Preserve the person's face, identity, body proportions, pose, and background."
)
TRYON_TOP_SECTION_TEMPLATE = "Top: {prompt}."
TRYON_BOTTOM_SECTION_TEMPLATE = "Bottom: {prompt}."
TRYON_DRESS_SECTION_TEMPLATE = "Dress: {prompt}."
TRYON_OUTER_SECTION_TEMPLATE = "Outer: {prompt}."
TRYON_GENERIC_SECTION_TEMPLATE = "{label}: {prompt}."


def run_tryon_request(
    payload: TryonRequest,
    *,
    settings: Settings | None = None,
    user_id: str,
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
        user_width, user_height = int(user_image.width), int(user_image.height)
        output_width, output_height = _bucket_dimensions(
            user_width,
            user_height,
            int(resolved_settings.tryon_dimension_multiple),
        )

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

        job_paths = build_tryon_job_media_paths(Path(resolved_settings.tryon_work_root))
        user_image.save(job_paths.person_path, format="JPEG", quality=95)
        product_reference.image.save(job_paths.garment_reference_path, format="JPEG", quality=95)

        routing_decision: TryonRoutingDecision | None = None
        if resolved_settings.tryon_use_specialists:
            routing_decision = resolve_tryon_route(payload.products, resolved_settings)
            prompt_text = _build_specialist_prompt(
                payload.products,
                routing_decision,
                resolved_settings,
            )
        else:
            prompt_text = _build_tryon_prompt(payload)

        run_result = get_tryon_execution_coordinator(resolved_settings).run(
            lambda: get_tryon_runner(resolved_settings).run_tryon(
                person_image_path=str(job_paths.person_path),
                garment_reference_path=str(job_paths.garment_reference_path),
                prompt=prompt_text,
                steps=resolved_steps,
                guidance_scale=resolved_guidance_scale,
                seed=resolved_seed,
                output_path=str(job_paths.output_path),
                output_width=output_width,
                output_height=output_height,
                lora_key=routing_decision.lora_key if routing_decision else None,
            ),
        )

        output_image = run_result.image.convert("RGB")
        if output_image.size != (user_width, user_height):
            output_image = output_image.resize(
                (user_width, user_height),
                Image.Resampling.LANCZOS,
            )
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
            f"{user_id}/{job_paths.job_id}"
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
                        "guidance_rescale": float(resolved_settings.tryon_guidance_rescale),
                        "do_cfg_norm": bool(resolved_settings.tryon_do_cfg_norm),
                        "network_multiplier": float(resolved_settings.tryon_lora_scale),
                    },
                    "reference": {
                        "product_reference_mode": product_reference.mode,
                        "control_order": {
                            "ctrl_img_1": "person",
                            "ctrl_img_2": "garment_reference",
                        },
                    },
                    "routing": {
                        "use_specialists": bool(resolved_settings.tryon_use_specialists),
                        "lora_key": routing_decision.lora_key if routing_decision else None,
                        "trigger_caption": (
                            routing_decision.trigger_caption if routing_decision else None
                        ),
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
                        "inference_width": output_width,
                        "inference_height": output_height,
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
    except TryonGenerationError as exc:
        return _error_response(
            http_status.BAD_REQUEST,
            "No image was generated by the try-on runtime.",
            {
                "feature": "tryon",
                "user_image": str(payload.user_image),
                "error": str(exc),
            },
        )
    except TryonRuntimeError as exc:
        return _error_response(
            http_status.INTERNAL_SERVER_ERROR,
            "Try-on runtime failed to initialize or execute.",
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


def _build_specialist_prompt(
    products: list[TryonProduct],
    routing: TryonRoutingDecision,
    settings: Settings,
) -> str:
    sections = _build_specialist_product_sections(products, routing.lora_key)
    parts = [f"{routing.trigger_caption.rstrip('.')}."]
    if sections:
        parts.append(sections)
    parts.append(settings.tryon_prompt_identity_clause)
    return " ".join(part for part in parts if part).strip()


def _build_specialist_product_sections(
    products: list[TryonProduct],
    lora_key: str,
) -> str:
    if lora_key == "multi":
        return _build_ordered_product_sections(products)
    if len(products) != 1:
        return ""
    product = products[0]
    description = _format_product_prompt(product.prompt)
    if not description:
        return ""
    label = _category_label_for_lora(lora_key)
    return f"{label}: {description}."


def _category_label_for_lora(lora_key: str) -> str:
    if lora_key == "top":
        return "Top"
    if lora_key == "bottom":
        return "Bottom"
    if lora_key == "dress":
        return "Dress"
    return "Garment"


def _build_ordered_product_sections(products: list[TryonProduct]) -> str:
    priority = {"top": 0, "outer": 0, "dress": 1, "bottom": 2}
    ordered = sorted(
        enumerate(products),
        key=lambda item: (priority.get(item[1].type.value, 99), item[0]),
    )
    sections: list[str] = []
    for _index, product in ordered:
        description = _format_product_prompt(product.prompt)
        if not description:
            continue
        label = "Top" if product.type.value == "outer" else product.type.value.capitalize()
        sections.append(f"{label}: {description}.")
    return " ".join(sections)


def _bucket_dimensions(width: int, height: int, multiple: int) -> tuple[int, int]:
    if multiple <= 1:
        return int(width), int(height)
    bucketed_w = max(multiple, int(round(width / multiple)) * multiple)
    bucketed_h = max(multiple, int(round(height / multiple)) * multiple)
    return bucketed_w, bucketed_h


def _build_tryon_prompt(payload: TryonRequest) -> str:
    prompt_prefix = (
        TRYON_SINGLE_REFERENCE_PROMPT
        if len(payload.products) == 1
        else TRYON_MULTI_REFERENCE_PROMPT
    )
    prompt_sections = _build_ordered_product_descriptions(payload)
    return f"{prompt_prefix} {prompt_sections} {TRYON_IDENTITY_CLAUSE}".strip()


def _build_ordered_product_descriptions(payload: TryonRequest) -> str:
    priority = {"top": 0, "outer": 0, "dress": 1, "bottom": 2}
    ordered_products = sorted(
        enumerate(payload.products),
        key=lambda item: (priority[item[1].type.value], item[0]),
    )
    return " ".join(
        _build_product_prompt_section(product.type.value, product.prompt)
        for _index, product in ordered_products
    )


def _build_product_prompt_section(product_type: str, prompt: str) -> str:
    normalized_prompt = _format_product_prompt(prompt)
    if product_type == "top":
        return TRYON_TOP_SECTION_TEMPLATE.format(prompt=normalized_prompt)
    if product_type == "bottom":
        return TRYON_BOTTOM_SECTION_TEMPLATE.format(prompt=normalized_prompt)
    if product_type == "dress":
        return TRYON_DRESS_SECTION_TEMPLATE.format(prompt=normalized_prompt)
    if product_type == "outer":
        return TRYON_OUTER_SECTION_TEMPLATE.format(prompt=normalized_prompt)
    return TRYON_GENERIC_SECTION_TEMPLATE.format(
        label=product_type.capitalize(),
        prompt=normalized_prompt,
    )


def _format_product_prompt(prompt: str) -> str:
    return str(prompt).strip().rstrip(".!?").strip()


def _error_response(status_code: int, message: str, metadata: dict[str, object]) -> TryonResponse:
    return TryonResponse(
        status=status_code,
        message=message,
        data=TryonResponseData(
            url=None,
            metadata=metadata,
        ),
    )

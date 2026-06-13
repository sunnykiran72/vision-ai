from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from time import perf_counter

import httpx
from PIL import Image, UnidentifiedImageError

from app.clients.qwen_diffusers_engine import (
    WardrobeDiffusersGenerationError,
    WardrobeDiffusersRuntimeError,
)
from app.clients.storage import AzureStorageClient
from app.config import Settings, get_settings
from app.constants import http_status
from app.constants import tryon as tryon_constants
from app.models.tryon import TryonProduct, TryonRequest, TryonResponse, TryonResponseData
from app.runtime.coordinator import QueueFullError, QueueTimeoutError
from app.runtime.system_coordinator import get_system_execution_coordinator
from app.runtime.tryon_runtime import get_tryon_runner
from app.runtime.upscale_runtime import get_upscale_execution_coordinator, get_upscale_runner
from app.services.tryon_routing import TryonRoutingDecision, resolve_tryon_route
from app.utils.media_utils import (
    build_storage_object_name,
    download_media_from_url,
)
from app.utils.tryon_collage import (
    ProductReferenceInput,
    build_product_reference,
)

logger = logging.getLogger("glamify-ai")


class TryonImageInputError(ValueError):
    def __init__(self, *, kind: str, url: str, reason: str) -> None:
        self.kind = kind
        self.url = url
        self.reason = reason
        super().__init__(f"{kind} image is invalid or could not be downloaded: {reason}")


@dataclass(frozen=True)
class _OpenedTryonImage:
    image: Image.Image
    source_url: str
    content_type: str | None


def run_tryon_request(
    payload: TryonRequest,
    *,
    settings: Settings | None = None,
    user_id: str,
    upscale_override: bool | None = None,
) -> TryonResponse:
    resolved_settings = settings or get_settings()
    # Per-request `?upscale=` query param overrides the server default; omitted -> server default.
    do_upscale = (
        bool(resolved_settings.tryon_upscale_after_qwen)
        if upscale_override is None
        else bool(upscale_override)
    )
    total_started = perf_counter()
    timings: dict[str, float] = {}
    try:
        job_id = uuid.uuid4().hex
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

        download_started = perf_counter()
        user_opened, product_opened = _download_tryon_images(payload)
        timings["download_seconds"] = _elapsed(download_started)
        logger.info(
            "Try-on image downloads completed in %.3fs (products=%d)",
            timings["download_seconds"],
            len(product_opened),
        )

        user_image = user_opened.image
        user_width, user_height = int(user_image.width), int(user_image.height)
        output_width, output_height = user_width, user_height

        product_inputs: list[ProductReferenceInput] = []
        downloaded_products: list[dict[str, str]] = []
        for product, opened in zip(payload.products, product_opened, strict=True):
            product_inputs.append(
                ProductReferenceInput(image=opened.image, type=product.type.value),
            )
            downloaded_products.append(
                {
                    "image_url": str(product.image_url),
                    "type": product.type.value,
                    "prompt": product.prompt,
                },
            )

        reference_started = perf_counter()
        product_reference = build_product_reference(product_inputs)
        reference_before_resize = product_reference.image.size
        garment_reference_image = _resize_longest_side(
            product_reference.image,
            max_edge=tryon_constants.GARMENT_REFERENCE_MAX_EDGE_PX,
        )
        timings["reference_build_seconds"] = _elapsed(reference_started)
        logger.info(
            "Try-on garment reference ready in %.3fs (mode=%s, size=%sx%s -> %sx%s)",
            timings["reference_build_seconds"],
            product_reference.mode,
            reference_before_resize[0],
            reference_before_resize[1],
            garment_reference_image.width,
            garment_reference_image.height,
        )

        prompt_started = perf_counter()
        routing_decision = resolve_tryon_route(payload.products, resolved_settings)
        prompt_text = _build_specialist_prompt(
            payload.products,
            routing_decision,
            resolved_settings,
        )
        timings["prompt_route_seconds"] = _elapsed(prompt_started)
        logger.info(
            "Try-on prompt/routing ready in %.3fs (lora=%s, products=%d)",
            timings["prompt_route_seconds"],
            routing_decision.lora_key,
            len(payload.products),
        )

        qwen_started = perf_counter()
        run_result = get_system_execution_coordinator(resolved_settings).run(
            lambda: get_tryon_runner(resolved_settings).run_tryon(
                person_image=user_image,
                garment_reference_image=garment_reference_image,
                prompt=prompt_text,
                steps=resolved_steps,
                guidance_scale=resolved_guidance_scale,
                seed=resolved_seed,
                output_width=output_width,
                output_height=output_height,
                lora_key=routing_decision.lora_key if routing_decision else None,
            ),
        )
        timings["qwen_generation_queued_wall_seconds"] = _elapsed(qwen_started)
        timings["qwen_generation_seconds"] = float(run_result.wall_seconds)
        logger.info(
            "Try-on Qwen generation completed in %.3fs (queued_wall=%.3fs, lora=%s, steps=%d)",
            timings["qwen_generation_seconds"],
            timings["qwen_generation_queued_wall_seconds"],
            routing_decision.lora_key,
            resolved_steps,
        )

        resize_started = perf_counter()
        output_image = run_result.image.convert("RGB")
        if output_image.size != (user_width, user_height):
            output_image = output_image.resize(
                (user_width, user_height),
                Image.Resampling.LANCZOS,
            )
        timings["output_resize_seconds"] = _elapsed(resize_started)
        qwen_output_size = {"width": int(output_image.width), "height": int(output_image.height)}
        upscale_metadata: dict[str, object] = {
            "enabled": do_upscale,
            "mode": "disabled",
            "override": upscale_override,
            "server_default": bool(resolved_settings.tryon_upscale_after_qwen),
        }
        if do_upscale:
            upscale_started = perf_counter()
            # In-memory tensor handoff: Qwen output PIL -> SeedVR2 -> PIL, with no intermediate
            # PNG save/reload on disk (saves the ~0.35s round-trip). Same in-memory path that
            # /v1/upscale uses; verified pixel-identical to the file route at 2496 (compiled).
            upscale_result = get_upscale_execution_coordinator(resolved_settings).run(
                lambda: get_upscale_runner(resolved_settings).run_tensor(
                    image=output_image,
                    target_long_edge=int(resolved_settings.tryon_upscale_target_long_edge),
                ),
            )
            output_image = upscale_result.image.convert("RGB")
            before_downscale_size = {
                "width": int(output_image.width),
                "height": int(output_image.height),
            }
            output_image = _resize_to_long_edge(
                output_image,
                target_long_edge=int(resolved_settings.tryon_final_output_long_edge),
            )
            timings["seedvr2_upscale_seconds"] = float(upscale_result.wall_seconds)
            timings["seedvr2_upscale_wall_seconds"] = _elapsed(upscale_started)
            upscale_metadata = {
                "enabled": True,
                "mode": "seedvr2_inline_tensor",
                "override": upscale_override,
                "server_default": bool(resolved_settings.tryon_upscale_after_qwen),
                "model_variant": upscale_result.model_variant,
                "runner_backend": upscale_result.runner_backend,
                "target_long_edge": int(upscale_result.target_long_edge),
                "derived_short_edge": int(upscale_result.derived_short_edge),
                "qwen_output_size": qwen_output_size,
                "upscaled_size_before_downscale": before_downscale_size,
                "final_long_edge": int(resolved_settings.tryon_final_output_long_edge),
                "wall_seconds": float(upscale_result.wall_seconds),
            }
            logger.info(
                "Try-on SeedVR2 inline upscale completed in %.3fs (%sx%s -> %sx%s -> %sx%s)",
                timings["seedvr2_upscale_wall_seconds"],
                qwen_output_size["width"],
                qwen_output_size["height"],
                before_downscale_size["width"],
                before_downscale_size["height"],
                output_image.width,
                output_image.height,
            )

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
            f"{user_id}/{job_id}"
        )
        object_name = build_storage_object_name(
            output_filename=None,
            prefix=storage_prefix,
            default_name="output",
        )
        encode_started = perf_counter()
        output_buffer = BytesIO()
        output_image.save(
            output_buffer,
            format="JPEG",
            quality=tryon_constants.JPEG_QUALITY,
            subsampling=0,
        )
        output_bytes = output_buffer.getvalue()
        timings["output_jpeg_encode_seconds"] = _elapsed(encode_started)
        upload_started = perf_counter()
        output_url = storage_client.upload_bytes(
            output_bytes,
            object_name=object_name,
            content_type="image/jpeg",
        )
        timings["output_upload_seconds"] = _elapsed(upload_started)
        timings["total_wall_seconds"] = _elapsed(total_started)
        logger.info(
            "Try-on completed in %.3fs (upload=%.3fs, output_bytes=%d)",
            timings["total_wall_seconds"],
            timings["output_upload_seconds"],
            len(output_bytes),
        )
        # Per-stage latency breakdown for the whole pipeline (one line, easy to scan).
        logger.info(
            "Try-on latency breakdown (s): download=%.3f | reference=%.3f | prompt_route=%.3f | "
            "qwen=%.3f (queued_wall=%.3f) | output_resize=%.3f | seedvr2_upscale=%.3f "
            "(wall=%.3f) | jpeg_encode=%.3f | upload=%.3f || total=%.3f",
            timings.get("download_seconds", 0.0),
            timings.get("reference_build_seconds", 0.0),
            timings.get("prompt_route_seconds", 0.0),
            timings.get("qwen_generation_seconds", 0.0),
            timings.get("qwen_generation_queued_wall_seconds", 0.0),
            timings.get("output_resize_seconds", 0.0),
            timings.get("seedvr2_upscale_seconds", 0.0),
            timings.get("seedvr2_upscale_wall_seconds", 0.0),
            timings.get("output_jpeg_encode_seconds", 0.0),
            timings.get("output_upload_seconds", 0.0),
            timings.get("total_wall_seconds", 0.0),
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
                        "network_multiplier": float(resolved_settings.tryon_lora_scale),
                        "upscale_after_qwen": do_upscale,
                        "upscale_override": upscale_override,
                        "upscale_server_default": bool(
                            resolved_settings.tryon_upscale_after_qwen,
                        ),
                        "upscale_target_long_edge": int(
                            resolved_settings.tryon_upscale_target_long_edge,
                        ),
                        "final_output_long_edge": int(
                            resolved_settings.tryon_final_output_long_edge,
                        ),
                    },
                    "reference": {
                        "product_reference_mode": product_reference.mode,
                        "garment_reference_max_edge": tryon_constants.GARMENT_REFERENCE_MAX_EDGE_PX,
                        "garment_reference_size_before_resize": {
                            "width": int(reference_before_resize[0]),
                            "height": int(reference_before_resize[1]),
                        },
                        "garment_reference_size": {
                            "width": int(garment_reference_image.width),
                            "height": int(garment_reference_image.height),
                        },
                        "control_order": {
                            "image_1": "person",
                            "image_2": "garment_reference",
                        },
                    },
                    "routing": {
                        "lora_key": routing_decision.lora_key,
                        "trigger_caption": routing_decision.trigger_caption,
                    },
                    "runner": {
                        **run_result.metadata,
                        "wall_seconds": float(run_result.wall_seconds),
                    },
                    "upscale": upscale_metadata,
                    "storage": {
                        "uploaded": True,
                        "url": output_url,
                        "bytes": len(output_bytes),
                    },
                    "timings": timings,
                    "job": {
                        "job_id": job_id,
                    },
                    "output": {
                        "width": int(output_image.width),
                        "height": int(output_image.height),
                        "inference_width": output_width,
                        "inference_height": output_height,
                        "qwen_width": qwen_output_size["width"],
                        "qwen_height": qwen_output_size["height"],
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
    except TryonImageInputError as exc:
        return _error_response(
            http_status.UNPROCESSABLE_CONTENT,
            f"{exc.kind.capitalize()} image is invalid or could not be downloaded.",
            {
                "feature": "tryon",
                "user_image": str(payload.user_image),
                "kind": exc.kind,
                "url": exc.url,
                "error": str(exc),
            },
            data=None,
        )
    except httpx.HTTPError as exc:
        return _error_response(
            http_status.UNPROCESSABLE_CONTENT,
            "Try-on image is invalid or could not be downloaded.",
            {
                "feature": "tryon",
                "user_image": str(payload.user_image),
                "error": str(exc),
            },
            data=None,
        )
    except WardrobeDiffusersGenerationError as exc:
        return _error_response(
            http_status.BAD_REQUEST,
            "No image was generated by the try-on runtime.",
            {
                "feature": "tryon",
                "user_image": str(payload.user_image),
                "error": str(exc),
            },
        )
    except ValueError as exc:
        return _error_response(
            http_status.BAD_REQUEST,
            str(exc),
            {
                "feature": "tryon",
                "user_image": str(payload.user_image),
                "error": str(exc),
            },
        )
    except WardrobeDiffusersRuntimeError as exc:
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


def _build_specialist_prompt(
    products: list[TryonProduct],
    routing: TryonRoutingDecision,
    settings: Settings,
) -> str:
    # Each garment type has one full template with the LoRA trigger baked in (kept exact).
    # Single types inject the product description into {garment}; multi joins the products
    # dynamically into {garment_list} (e.g. "Top: ... and Bottom: ...").
    templates = tryon_constants.TRYON_PROMPT_TEMPLATE_BY_TYPE
    if routing.lora_key == "multi":
        garment_list = _build_multi_garment_list(products) or "reference garments"
        return templates["multi"].format(garment_list=garment_list)
    template = templates.get(routing.lora_key, templates["top"])
    return template.format(garment=_single_garment_phrase(products))


def _single_garment_phrase(products: list[TryonProduct]) -> str:
    for product in products:
        description = _format_product_prompt(product.prompt)
        if description:
            return description
    return "reference garment"


def _build_multi_garment_list(products: list[TryonProduct]) -> str:
    # Canonical ordering (top/outer first, then dress, then bottom), stable within a tie.
    priority = {"top": 0, "outer": 0, "dress": 1, "bottom": 2}
    ordered = sorted(
        enumerate(products),
        key=lambda item: (priority.get(item[1].type.value, 99), item[0]),
    )
    parts: list[str] = []
    for _index, product in ordered:
        label = (
            "Top"
            if product.type.value == "outer"
            else tryon_constants.TRYON_GARMENT_LABEL_BY_TYPE.get(
                product.type.value,
                product.type.value.capitalize(),
            )
        )
        description = _format_product_prompt(product.prompt)
        parts.append(f"{label}: {description}" if description else label)
    return " and ".join(parts)


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


def _build_tryon_prompt(payload: TryonRequest) -> str:
    prompt_prefix = (
        tryon_constants.SINGLE_REFERENCE_PROMPT
        if len(payload.products) == 1
        else tryon_constants.MULTI_REFERENCE_PROMPT
    )
    prompt_sections = _build_ordered_product_descriptions(payload)
    return f"{prompt_prefix} {prompt_sections} {tryon_constants.IDENTITY_CLAUSE}".strip()


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
        return tryon_constants.TOP_SECTION_TEMPLATE.format(prompt=normalized_prompt)
    if product_type == "bottom":
        return tryon_constants.BOTTOM_SECTION_TEMPLATE.format(prompt=normalized_prompt)
    if product_type == "dress":
        return tryon_constants.DRESS_SECTION_TEMPLATE.format(prompt=normalized_prompt)
    if product_type == "outer":
        return tryon_constants.OUTER_SECTION_TEMPLATE.format(prompt=normalized_prompt)
    return tryon_constants.GENERIC_SECTION_TEMPLATE.format(
        label=product_type.capitalize(),
        prompt=normalized_prompt,
    )


def _format_product_prompt(prompt: str) -> str:
    return str(prompt).strip().rstrip(".!?").strip()


def _download_tryon_images(
    payload: TryonRequest,
) -> tuple[_OpenedTryonImage, list[_OpenedTryonImage]]:
    max_workers = max(
        1,
        min(tryon_constants.DOWNLOAD_MAX_WORKERS, 1 + len(payload.products)),
    )
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tryon-download") as pool:
        user_future = pool.submit(
            _download_and_open_image,
            kind="user",
            url=str(payload.user_image),
        )
        product_futures = [
            pool.submit(
                _download_and_open_image,
                kind="garment",
                url=str(product.image_url),
            )
            for product in payload.products
        ]
        user_image = user_future.result()
        product_images = [future.result() for future in product_futures]
    return user_image, product_images


def _download_and_open_image(*, kind: str, url: str) -> _OpenedTryonImage:
    try:
        downloaded = download_media_from_url(url)
        image = Image.open(BytesIO(downloaded.content)).convert("RGB")
        image.load()
        return _OpenedTryonImage(
            image=image,
            source_url=url,
            content_type=downloaded.content_type,
        )
    except (httpx.HTTPError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise TryonImageInputError(kind=kind, url=url, reason=str(exc)) from exc


def _resize_longest_side(image: Image.Image, *, max_edge: int) -> Image.Image:
    width, height = int(image.width), int(image.height)
    longest = max(width, height)
    if longest <= int(max_edge):
        return image.convert("RGB")
    scale = float(max_edge) / float(longest)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.convert("RGB").resize(new_size, Image.Resampling.LANCZOS)


def _resize_to_long_edge(image: Image.Image, *, target_long_edge: int) -> Image.Image:
    width, height = int(image.width), int(image.height)
    longest = max(width, height)
    if target_long_edge <= 0 or longest <= 0 or longest == int(target_long_edge):
        return image.convert("RGB")
    scale = float(target_long_edge) / float(longest)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.convert("RGB").resize(new_size, Image.Resampling.LANCZOS)


def _elapsed(started: float) -> float:
    return float(round(perf_counter() - started, 3))


def _error_response(
    status_code: int,
    message: str,
    metadata: dict[str, object],
    *,
    data: TryonResponseData | None | object = ...,
) -> TryonResponse:
    response_data = (
        TryonResponseData(
            url=None,
            metadata=metadata,
        )
        if data is ...
        else data
    )
    return TryonResponse(
        status=status_code,
        message=message,
        data=response_data,
    )

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings
from app.constants import wardrobe as wardrobe_constants
from app.runtime.wardrobe_types import WardrobeRunResult, WardrobeRuntimeStatus

logger = logging.getLogger("glamify-ai")

BACKEND_NAME = "diffusers_qwen_image_edit_plus"
WARDROBE_CATEGORIES: tuple[str, ...] = ("top", "bottom", "dress")


class WardrobeDiffusersRuntimeError(RuntimeError):
    pass


class WardrobeDiffusersGenerationError(RuntimeError):
    pass


def resize_input_for_model(
    image: Image.Image,
    max_side: int = wardrobe_constants.PREPROCESS_MAX_EDGE_PX,
) -> Image.Image:
    """Cap the longest side to ``max_side`` and round the other side to /16.

    This is an exact port of ``resize_input_for_model`` in the standalone diffusers
    tester. Images already within ``max_side`` are returned unchanged (no rounding),
    so the pixels fed to the pipeline match the reference byte-for-byte.
    """
    width, height = int(image.width), int(image.height)
    if max(width, height) <= max_side:
        return image
    if width >= height:
        new_width = max_side
        new_height = max(16, round((height * max_side / width) / 16) * 16)
    else:
        new_height = max_side
        new_width = max(16, round((width * max_side / height) / 16) * 16)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


class QwenDiffusersWardrobeEngine:
    """Single resident Qwen-Image-Edit-Plus diffusers runtime for garment extraction.

    Mirrors the standalone diffusers tester: one base pipeline kept resident, with the
    per-category extraction LoRA swapped via ``set_adapters`` per request. Unlike the
    tester (lazy-load across ~18 LoRAs), wardrobe carries only top/bottom/dress, so all
    three are loaded eagerly during warmup before the API serves traffic.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_path = Path(settings.qwen_image_edit_model_path).expanduser()
        self._lora_paths: dict[str, Path] = {
            "top": Path(settings.wardrobe_lora_top_path).expanduser(),
            "bottom": Path(settings.wardrobe_lora_bottom_path).expanduser(),
            "dress": Path(settings.wardrobe_lora_dress_path).expanduser(),
        }
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self._loaded_loras: set[str] = set()
        self._active_lora: str | None = None
        self._device = "cpu"
        self._dtype_name = "float32"
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    # ----- lifecycle -------------------------------------------------------

    def warmup(self) -> None:
        self._ensure_pipeline()
        for category in WARDROBE_CATEGORIES:
            self._ensure_lora(category)
        first = WARDROBE_CATEGORIES[0]
        warm = Image.new(
            "RGB",
            (wardrobe_constants.OUTPUT_WIDTH, wardrobe_constants.OUTPUT_HEIGHT),
            "white",
        )
        started = time.perf_counter()
        self._generate(
            category=first,
            images=warm,
            prompt=wardrobe_constants.QWEN_EXTRACT_PROMPT_TEMPLATE_BY_TYPE[first].format(
                caption="a garment",
            ),
            steps=wardrobe_constants.GENERATION_STEPS,
            seed=wardrobe_constants.GENERATION_SEED,
            width=wardrobe_constants.OUTPUT_WIDTH,
            height=wardrobe_constants.OUTPUT_HEIGHT,
            lora_scale=wardrobe_constants.GENERATION_NETWORK_MULTIPLIER,
        )
        logger.info(
            "Qwen wardrobe diffusers runtime ready (device=%s, loras=%s, warm=%.1fs)",
            self._device,
            sorted(self._loaded_loras),
            time.perf_counter() - started,
        )

    def status(self) -> WardrobeRuntimeStatus:
        return WardrobeRuntimeStatus(
            loaded=self._pipeline is not None,
            backend=BACKEND_NAME if self._pipeline is not None else None,
            loras_loaded=set(self._loaded_loras) >= set(WARDROBE_CATEGORIES),
        )

    # ----- inference -------------------------------------------------------

    def run_extract(
        self,
        *,
        input_image: Image.Image,
        prompt: str,
        garment_type: str,
    ) -> WardrobeRunResult:
        if garment_type not in WARDROBE_CATEGORIES:
            raise WardrobeDiffusersRuntimeError(
                f"Unknown wardrobe extraction category: {garment_type}",
            )
        started_at = time.perf_counter()
        image = self._generate(
            category=garment_type,
            images=resize_input_for_model(input_image.convert("RGB")),
            prompt=str(prompt).strip(),
            steps=wardrobe_constants.GENERATION_STEPS,
            seed=wardrobe_constants.GENERATION_SEED,
            width=wardrobe_constants.OUTPUT_WIDTH,
            height=wardrobe_constants.OUTPUT_HEIGHT,
            lora_scale=wardrobe_constants.GENERATION_NETWORK_MULTIPLIER,
        )
        metadata = {
            "backend": BACKEND_NAME,
            "architecture": "qwen_image_edit_plus",
            "engine": "diffusers",
            "model_source": str(self._model_path),
            "checkpoint_path": str(self._lora_paths[garment_type]),
            "lora_key": garment_type,
            "lora_rank": wardrobe_constants.LORA_RANK,
            "lora_alpha": wardrobe_constants.LORA_ALPHA,
            "lora_scale": wardrobe_constants.GENERATION_NETWORK_MULTIPLIER,
            "true_cfg_scale": wardrobe_constants.GENERATION_TRUE_CFG_SCALE,
            "seed": wardrobe_constants.GENERATION_SEED,
            "steps": wardrobe_constants.GENERATION_STEPS,
            "control_order": {"image_1": "garment_input"},
            "output_size": {
                "width": wardrobe_constants.OUTPUT_WIDTH,
                "height": wardrobe_constants.OUTPUT_HEIGHT,
            },
            "dtype": self._dtype_name,
            "device": self._device,
        }
        return WardrobeRunResult(
            image=image,
            metadata=metadata,
            wall_seconds=float(round(time.perf_counter() - started_at, 3)),
        )

    def generate_preview(
        self,
        *,
        garment_type: str,
        image: Image.Image,
        prompt: str,
        steps: int,
        seed: int,
        width: int,
        height: int,
        lora_scale: float,
    ) -> tuple[Image.Image, float]:
        """Direct extraction used by the diffusers test endpoint (no Marqo/storage)."""
        if garment_type not in WARDROBE_CATEGORIES:
            raise WardrobeDiffusersRuntimeError(
                f"Unknown wardrobe extraction category: {garment_type}",
            )
        started_at = time.perf_counter()
        out = self._generate(
            category=garment_type,
            images=resize_input_for_model(image.convert("RGB")),
            prompt=str(prompt).strip(),
            steps=int(steps),
            seed=int(seed),
            width=int(width),
            height=int(height),
            lora_scale=float(lora_scale),
        )
        return out, float(round(time.perf_counter() - started_at, 3))

    # ----- internals -------------------------------------------------------

    def _generate(
        self,
        *,
        category: str,
        images: Any,
        prompt: str,
        steps: int,
        seed: int,
        width: int,
        height: int,
        lora_scale: float,
    ) -> Image.Image:
        self._ensure_pipeline()
        self._ensure_lora(category)
        if self._pipeline is None or self._torch is None:
            raise WardrobeDiffusersRuntimeError("Qwen wardrobe diffusers pipeline is not loaded.")
        with self._infer_lock:
            self._pipeline.set_adapters([category], [float(lora_scale)])
            generator = self._torch.Generator(device=self._device).manual_seed(int(seed))
            with self._torch.inference_mode():
                result = self._pipeline(
                    image=images,
                    prompt=prompt,
                    true_cfg_scale=float(wardrobe_constants.GENERATION_TRUE_CFG_SCALE),
                    num_inference_steps=int(steps),
                    height=int(height),
                    width=int(width),
                    generator=generator,
                )
            self._active_lora = category
        produced = getattr(result, "images", None) or []
        if not produced:
            raise WardrobeDiffusersGenerationError("Qwen wardrobe pipeline returned no images.")
        image: Image.Image = produced[0].convert("RGB")
        return image

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        with self._load_lock:
            if self._pipeline is not None:
                return
            self._load_pipeline()

    def _load_pipeline(self) -> None:
        if not self._model_path.exists():
            raise WardrobeDiffusersRuntimeError(
                f"Qwen model path does not exist: {self._model_path}",
            )
        try:
            import torch  # type: ignore[import-not-found]
            from diffusers import (  # type: ignore[import-not-found]
                QwenImageEditPlusPipeline,
            )
        except Exception as exc:  # pragma: no cover - environment dependency
            raise WardrobeDiffusersRuntimeError(
                "Wardrobe diffusers backend requires torch and diffusers in the environment.",
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = _resolve_torch_dtype(
            torch_module=torch,
            requested=self._settings.qwen_image_edit_dtype,
            device=self._device,
        )
        self._dtype_name = str(dtype).replace("torch.", "")

        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            str(self._model_path),
            torch_dtype=dtype,
        ).to(self._device)

        self._torch = torch
        self._pipeline = pipeline
        logger.info(
            "Loaded Qwen-Image-Edit-Plus diffusers pipeline (device=%s, dtype=%s)",
            self._device,
            self._dtype_name,
        )

    def _ensure_lora(self, category: str) -> None:
        if category in self._loaded_loras:
            return
        if self._pipeline is None:
            raise WardrobeDiffusersRuntimeError("Qwen wardrobe diffusers pipeline is not loaded.")
        path = self._lora_paths[category]
        if not path.exists():
            raise WardrobeDiffusersRuntimeError(
                f"Wardrobe extraction LoRA missing for '{category}': {path}",
            )
        try:
            from safetensors.torch import load_file  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - environment dependency
            raise WardrobeDiffusersRuntimeError(
                "Wardrobe diffusers backend requires safetensors in the runtime environment.",
            ) from exc

        state_dict = load_file(str(path))
        # Remap AI-Toolkit `diffusion_model.*` keys onto the diffusers `transformer.*`
        # namespace, exactly as the standalone tester does.
        remapped = {
            (
                "transformer." + key[len("diffusion_model.") :]
                if key.startswith("diffusion_model.")
                else key
            ): value
            for key, value in state_dict.items()
        }
        self._pipeline.load_lora_weights(remapped, adapter_name=category)
        self._loaded_loras.add(category)
        logger.info("Loaded wardrobe extraction LoRA '%s' from %s", category, path)


def _resolve_torch_dtype(*, torch_module: Any, requested: str, device: str) -> Any:
    if device != "cuda":
        return torch_module.float32
    normalized = str(requested or "bfloat16").strip().lower()
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp8": "float8_e4m3fn",
        "float8": "float8_e4m3fn",
        "float8_e4m3fn": "float8_e4m3fn",
        "fp32": "float32",
        "float32": "float32",
    }
    dtype_name = aliases.get(normalized)
    if dtype_name is None:
        supported = ", ".join(sorted(set(aliases)))
        raise WardrobeDiffusersRuntimeError(
            f"Unsupported QWEN_IMAGE_EDIT_DTYPE '{requested}'. Supported values: {supported}.",
        )
    if dtype_name == "float8_e4m3fn" and not hasattr(torch_module, "float8_e4m3fn"):
        raise WardrobeDiffusersRuntimeError(
            "QWEN_IMAGE_EDIT_DTYPE=float8_e4m3fn requires a PyTorch build with float8_e4m3fn.",
        )
    return getattr(torch_module, dtype_name)

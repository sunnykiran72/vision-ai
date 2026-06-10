from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings
from app.constants import wardrobe as wardrobe_constants
from app.runtime.tryon_types import TryonRunResult, TryonRuntimeStatus
from app.runtime.wardrobe_types import WardrobeRunResult, WardrobeRuntimeStatus

logger = logging.getLogger("glamify-ai")

BACKEND_NAME = "diffusers_qwen_image_edit_plus"
WARDROBE_CATEGORIES: tuple[str, ...] = ("top", "bottom", "dress")
TRYON_CATEGORIES: tuple[str, ...] = ("top", "bottom", "dress", "multi")


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
        self._tryon_lora_paths: dict[str, Path] = {
            "top": Path(settings.tryon_lora_top_path).expanduser(),
            "bottom": Path(settings.tryon_lora_bottom_path).expanduser(),
            "dress": Path(settings.tryon_lora_dress_path).expanduser(),
            "multi": Path(settings.tryon_lora_multi_path).expanduser(),
        }
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self._loaded_loras: set[str] = set()
        self._active_lora: str | None = None
        self._compiled = False
        self._fp8_quantized = False
        self._quantization = "none"
        self._device = "cpu"
        self._dtype_name = "float32"
        self._load_lock = threading.Lock()
        self._compile_lock = threading.Lock()
        self._quantize_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    # ----- lifecycle -------------------------------------------------------

    def warmup(self) -> None:
        self._ensure_pipeline()
        for category in WARDROBE_CATEGORIES:
            self._ensure_lora("wardrobe", category)
        if self._settings.qwen_fp8:
            for category in self._configured_tryon_lora_keys():
                self._ensure_lora("tryon", category)
        warm = Image.new(
            "RGB",
            (wardrobe_constants.OUTPUT_WIDTH, wardrobe_constants.OUTPUT_HEIGHT),
            "white",
        )
        started = time.perf_counter()
        warm_categories = (
            WARDROBE_CATEGORIES if self._settings.qwen_compile else WARDROBE_CATEGORIES[:1]
        )
        for category in warm_categories:
            self._generate(
                category=category,
                images=warm,
                prompt=wardrobe_constants.QWEN_EXTRACT_PROMPT_TEMPLATE_BY_TYPE[category].format(
                    caption="a garment",
                ),
                steps=wardrobe_constants.GENERATION_STEPS,
                seed=wardrobe_constants.GENERATION_SEED,
                width=wardrobe_constants.OUTPUT_WIDTH,
                height=wardrobe_constants.OUTPUT_HEIGHT,
                lora_scale=wardrobe_constants.GENERATION_NETWORK_MULTIPLIER,
            )
        logger.info(
            "Qwen wardrobe diffusers runtime ready "
            "(device=%s, dtype=%s, quantization=%s, compile=%s, loras=%s, warm=%.1fs)",
            self._device,
            self._dtype_name,
            self._quantization,
            self._compiled,
            sorted(self._loaded_loras),
            time.perf_counter() - started,
        )

    def warmup_tryon(self) -> None:
        self._ensure_pipeline()
        for category in self._expected_tryon_lora_keys():
            self._ensure_lora("tryon", category)
        logger.info(
            "Qwen tryon diffusers adapters ready (device=%s, dtype=%s, loras=%s)",
            self._device,
            self._dtype_name,
            sorted(
                adapter
                for adapter in self._loaded_loras
                if adapter.startswith("tryon_")
            ),
        )

    def status(self) -> WardrobeRuntimeStatus:
        return WardrobeRuntimeStatus(
            loaded=self._pipeline is not None,
            backend=BACKEND_NAME if self._pipeline is not None else None,
            loras_loaded={_adapter_name("wardrobe", key) for key in WARDROBE_CATEGORIES}
            <= self._loaded_loras,
        )

    def tryon_status(self) -> TryonRuntimeStatus:
        expected = self._expected_tryon_lora_keys()
        return TryonRuntimeStatus(
            loaded=self._pipeline is not None,
            backend=BACKEND_NAME if self._pipeline is not None else None,
            lora_loaded={_adapter_name("tryon", key) for key in expected} <= self._loaded_loras,
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
            "quantization": self._quantization,
            "fp8": self._fp8_quantized,
            "device": self._device,
            "compiled": self._compiled,
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

    def run_tryon(
        self,
        *,
        person_image: Image.Image,
        garment_reference_image: Image.Image,
        prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        output_width: int,
        output_height: int,
        lora_key: str | None = None,
    ) -> TryonRunResult:
        category = self._resolve_tryon_lora_key(lora_key)
        started_at = time.perf_counter()
        image = self._generate(
            namespace="tryon",
            category=category,
            images=[person_image.convert("RGB"), garment_reference_image.convert("RGB")],
            prompt=str(prompt).strip(),
            steps=int(steps),
            seed=int(seed),
            width=int(output_width),
            height=int(output_height),
            lora_scale=float(self._settings.tryon_lora_scale),
            true_cfg_scale=float(guidance_scale),
        )
        metadata = {
            "backend": BACKEND_NAME,
            "architecture": "qwen_image_edit_plus",
            "engine": "diffusers",
            "feature": "tryon",
            "model_source": str(self._model_path),
            "checkpoint_path": str(self._tryon_lora_paths[category]),
            "lora_key": category,
            "lora_rank": int(self._settings.tryon_lora_rank),
            "lora_alpha": int(self._settings.tryon_lora_alpha),
            "lora_scale": float(self._settings.tryon_lora_scale),
            "true_cfg_scale": float(guidance_scale),
            "seed": int(seed),
            "steps": int(steps),
            "control_order": {
                "image_1": "person",
                "image_2": "garment_reference",
            },
            "output_size": {
                "width": int(output_width),
                "height": int(output_height),
            },
            "dtype": self._dtype_name,
            "quantization": self._quantization,
            "fp8": self._fp8_quantized,
            "device": self._device,
            "compiled": self._compiled,
        }
        return TryonRunResult(
            image=image,
            metadata=metadata,
            wall_seconds=float(round(time.perf_counter() - started_at, 3)),
        )

    # ----- internals -------------------------------------------------------

    def _generate(
        self,
        *,
        namespace: str = "wardrobe",
        category: str,
        images: Any,
        prompt: str,
        steps: int,
        seed: int,
        width: int,
        height: int,
        lora_scale: float,
        true_cfg_scale: float = wardrobe_constants.GENERATION_TRUE_CFG_SCALE,
    ) -> Image.Image:
        self._ensure_pipeline()
        self._ensure_lora(namespace, category)
        if self._pipeline is None or self._torch is None:
            raise WardrobeDiffusersRuntimeError("Qwen wardrobe diffusers pipeline is not loaded.")
        self._ensure_fp8_quantized()
        self._ensure_compiled()
        adapter = _adapter_name(namespace, category)
        with self._infer_lock:
            self._pipeline.set_adapters([adapter], [float(lora_scale)])
            generator = self._torch.Generator(device=self._device).manual_seed(int(seed))
            with self._torch.inference_mode():
                result = self._pipeline(
                    image=images,
                    prompt=prompt,
                    true_cfg_scale=float(true_cfg_scale),
                    num_inference_steps=int(steps),
                    height=int(height),
                    width=int(width),
                    generator=generator,
                )
            self._active_lora = adapter
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

    def _ensure_compiled(self) -> None:
        if self._compiled or not self._settings.qwen_compile:
            return
        if self._device != "cuda":
            logger.info("Skipping Qwen torch.compile because device is %s", self._device)
            return
        if self._pipeline is None or self._torch is None:
            raise WardrobeDiffusersRuntimeError("Qwen wardrobe diffusers pipeline is not loaded.")
        with self._compile_lock:
            if self._compiled or not self._settings.qwen_compile:
                return
            transformer = getattr(self._pipeline, "transformer", None)
            if transformer is None or not hasattr(transformer, "compile_repeated_blocks"):
                raise WardrobeDiffusersRuntimeError(
                    "QWEN_COMPILE=1 requires pipeline.transformer.compile_repeated_blocks "
                    "(diffusers >= 0.32).",
                )
            # compile_repeated_blocks compiles the single repeated transformer block once
            # and reuses that graph across every block, all loaded LoRA adapters, and
            # variable input shapes. The previous per-block torch.compile(dynamic=False)
            # loop recompiled on every new input size / adapter switch, so the fp8 GEMM
            # never stayed fused and inference ran eager (~1.6s/step) at request time.
            # Benchmarked on RTX PRO 6000 (832x1248, 15 steps): warm ~7.8s with 3 live
            # adapters (vs ~18-22s eager), stable across adapter switches and input shapes.
            transformer.compile_repeated_blocks(fullgraph=False)
            self._compiled = True
            logger.info(
                "Compiled Qwen transformer via compile_repeated_blocks(fullgraph=False).",
            )

    def _ensure_fp8_quantized(self) -> None:
        if self._fp8_quantized or not self._settings.qwen_fp8:
            return
        if self._device != "cuda":
            raise WardrobeDiffusersRuntimeError(
                "QWEN_FP8=1 requires CUDA. Use QWEN_FP8=0 for CPU/local tests.",
            )
        if self._pipeline is None:
            raise WardrobeDiffusersRuntimeError("Qwen wardrobe diffusers pipeline is not loaded.")
        with self._quantize_lock:
            if self._fp8_quantized or not self._settings.qwen_fp8:
                return
            for namespace, category in self._configured_lora_keys():
                self._ensure_lora(namespace, category)
            try:
                from torchao.quantization import (  # type: ignore[import-not-found]
                    Float8DynamicActivationFloat8WeightConfig,
                    quantize_,
                )
            except Exception as exc:  # pragma: no cover - environment dependency
                raise WardrobeDiffusersRuntimeError(
                    "QWEN_FP8=1 requires torchao. Install torchao in the GPU runtime stack.",
                ) from exc
            transformer = getattr(self._pipeline, "transformer", None)
            if transformer is None:
                raise WardrobeDiffusersRuntimeError(
                    "QWEN_FP8=1 requires pipeline.transformer to be available.",
                )
            quantize_(transformer, Float8DynamicActivationFloat8WeightConfig())
            self._fp8_quantized = True
            self._quantization = "torchao_float8_dynamic_activation_float8_weight"
            logger.info("Quantized Qwen transformer with torchao fp8 dynamic activation/weight.")

    def _ensure_lora(self, namespace: str, category: str) -> None:
        adapter = _adapter_name(namespace, category)
        if adapter in self._loaded_loras:
            return
        if self._fp8_quantized:
            raise WardrobeDiffusersRuntimeError(
                "Cannot load a new Qwen LoRA after QWEN_FP8 quantization. "
                "Configure all required wardrobe/try-on LoRA paths before warmup.",
            )
        if self._pipeline is None:
            raise WardrobeDiffusersRuntimeError("Qwen wardrobe diffusers pipeline is not loaded.")
        path = self._lora_path(namespace, category)
        if not path.exists():
            raise WardrobeDiffusersRuntimeError(
                f"{namespace.capitalize()} LoRA missing for '{category}': {path}",
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
        self._pipeline.load_lora_weights(remapped, adapter_name=adapter)
        self._loaded_loras.add(adapter)
        logger.info(
            "Loaded %s LoRA '%s' as adapter '%s' from %s",
            namespace,
            category,
            adapter,
            path,
        )

    def _lora_path(self, namespace: str, category: str) -> Path:
        if namespace == "wardrobe":
            return self._lora_paths[category]
        if namespace == "tryon":
            return self._tryon_lora_paths[category]
        raise WardrobeDiffusersRuntimeError(f"Unknown LoRA namespace: {namespace}")

    def _expected_tryon_lora_keys(self) -> tuple[str, ...]:
        return TRYON_CATEGORIES

    def _configured_tryon_lora_keys(self) -> tuple[str, ...]:
        return tuple(
            category
            for category in TRYON_CATEGORIES
            if str(self._tryon_lora_paths[category]).strip()
            and self._tryon_lora_paths[category] != Path(".")
        )

    def _configured_lora_keys(self) -> tuple[tuple[str, str], ...]:
        keys: list[tuple[str, str]] = []
        for category in WARDROBE_CATEGORIES:
            path = self._lora_paths[category]
            if str(path).strip() and path != Path("."):
                keys.append(("wardrobe", category))
        keys.extend(("tryon", category) for category in self._configured_tryon_lora_keys())
        return tuple(keys)

    def _resolve_tryon_lora_key(self, lora_key: str | None) -> str:
        resolved = str(lora_key or "").strip().lower()
        if resolved not in TRYON_CATEGORIES:
            raise WardrobeDiffusersRuntimeError(
                "lora_key is required and must be one of " + ", ".join(TRYON_CATEGORIES) + ".",
            )
        return resolved


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
        "fp32": "float32",
        "float32": "float32",
    }
    if normalized in {"fp8", "float8", "float8_e4m3fn"}:
        raise WardrobeDiffusersRuntimeError(
            "QWEN_IMAGE_EDIT_DTYPE=float8 is not a valid diffusers load dtype. "
            "Use QWEN_IMAGE_EDIT_DTYPE=bfloat16 with QWEN_FP8=1 so torchao can quantize "
            "the loaded transformer with fp8 scales.",
        )
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


def _adapter_name(namespace: str, category: str) -> str:
    return f"{namespace}_{category}"

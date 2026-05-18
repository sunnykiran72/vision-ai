from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings

logger = logging.getLogger("glamify-ai")


def _align_to_model_grid(value: int, *, base: int = 8) -> int:
    raw = max(int(value), int(base))
    return max(int(base), (raw // int(base)) * int(base))


@dataclass(frozen=True)
class QwenImageEditRuntimeStatus:
    loaded: bool
    backend: str | None
    lora_loaded: bool


@dataclass(frozen=True)
class QwenImageEditRunResult:
    image: Image.Image
    metadata: dict[str, Any]
    wall_seconds: float


class QwenImageEditClient:
    def __init__(
        self,
        settings: Settings,
        *,
        lora_path: str,
        lora_weight_name: str,
        lora_scale: float,
        adapter_name: str,
    ) -> None:
        self._settings = settings
        self._lora_path = str(lora_path or "").strip()
        self._lora_weight_name = str(lora_weight_name or "").strip()
        self._lora_scale = float(lora_scale)
        self._adapter_name = str(adapter_name or "").strip() or "default"
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self._device = "cpu"
        self._dtype_name = "float32"
        self._lora_loaded = False
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def warmup(self) -> None:
        self._ensure_ready()
        image = Image.new("RGB", (512, 512), color="white")
        try:
            self.run_tryon(
                garment_reference_image=image,
                user_image=image,
                prompt="Put the provided garments on the person while preserving identity.",
                steps=2,
                guidance_scale=1.0,
                seed=1,
                output_width=512,
                output_height=512,
            )
        except Exception as exc:
            logger.warning("Qwen image-edit warmup failed: %s", exc)

    def status(self) -> QwenImageEditRuntimeStatus:
        return QwenImageEditRuntimeStatus(
            loaded=self._pipeline is not None,
            backend=self._device if self._pipeline is not None else None,
            lora_loaded=self._lora_loaded,
        )

    def run_edit(
        self,
        image: Image.Image,
        *,
        prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        output_width: int | None = None,
        output_height: int | None = None,
    ) -> QwenImageEditRunResult:
        self._ensure_ready()
        if self._pipeline is None or self._torch is None:
            raise RuntimeError("Qwen image-edit pipeline is not loaded.")

        prepared = image.convert("RGB")
        run_kwargs: dict[str, Any] = {
            "image": prepared,
            "prompt": str(prompt or "").strip(),
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance_scale),
        }

        generator = self._torch.Generator(
            device=self._device if self._device == "cuda" else "cpu",
        ).manual_seed(int(seed))
        run_kwargs["generator"] = generator

        if output_width is not None and output_height is not None:
            run_kwargs["width"] = _align_to_model_grid(int(output_width), base=16)
            run_kwargs["height"] = _align_to_model_grid(int(output_height), base=16)

        started_at = time.perf_counter()
        with self._infer_lock, self._torch.inference_mode():
            result = self._pipeline(**run_kwargs)
        images = getattr(result, "images", None) or []
        if not images:
            raise RuntimeError("Qwen image-edit pipeline returned no images.")

        metadata = {
            "model_source": str(self._settings.qwen_image_edit_model_path),
            "lora_path": self._lora_path,
            "lora_weight_name": self._lora_weight_name,
            "lora_scale": float(self._lora_scale),
            "lora_loaded": bool(self._lora_loaded),
            "device": self._device,
            "dtype": self._dtype_name,
            "steps": int(steps),
            "guidance_scale": float(guidance_scale),
            "seed": int(seed),
            "requested_output_size": (
                {"width": int(output_width), "height": int(output_height)}
                if output_width is not None and output_height is not None
                else None
            ),
        }
        return QwenImageEditRunResult(
            image=images[0],
            metadata=metadata,
            wall_seconds=float(round(time.perf_counter() - started_at, 3)),
        )

    def run_tryon(
        self,
        *,
        garment_reference_image: Image.Image,
        user_image: Image.Image,
        prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        output_width: int | None = None,
        output_height: int | None = None,
    ) -> QwenImageEditRunResult:
        self._ensure_ready()
        if self._pipeline is None or self._torch is None:
            raise RuntimeError("Qwen image-edit pipeline is not loaded.")

        garment_reference = garment_reference_image.convert("RGB")
        user_reference = user_image.convert("RGB")
        run_kwargs: dict[str, Any] = {
            "image": [garment_reference, user_reference],
            "prompt": str(prompt or "").strip(),
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance_scale),
        }

        generator = self._torch.Generator(
            device=self._device if self._device == "cuda" else "cpu",
        ).manual_seed(int(seed))
        run_kwargs["generator"] = generator

        if output_width is not None and output_height is not None:
            run_kwargs["width"] = _align_to_model_grid(int(output_width), base=16)
            run_kwargs["height"] = _align_to_model_grid(int(output_height), base=16)

        started_at = time.perf_counter()
        with self._infer_lock, self._torch.inference_mode():
            result = self._pipeline(**run_kwargs)
        images = getattr(result, "images", None) or []
        if not images:
            raise RuntimeError("Qwen image-edit pipeline returned no images.")

        metadata = {
            "model_source": str(self._settings.qwen_image_edit_model_path),
            "lora_path": self._lora_path,
            "lora_weight_name": self._lora_weight_name,
            "lora_scale": float(self._lora_scale),
            "lora_loaded": bool(self._lora_loaded),
            "device": self._device,
            "dtype": self._dtype_name,
            "steps": int(steps),
            "guidance_scale": float(guidance_scale),
            "seed": int(seed),
            "input_mode": "separate_garment_and_user_images",
            "requested_output_size": (
                {"width": int(output_width), "height": int(output_height)}
                if output_width is not None and output_height is not None
                else None
            ),
        }
        return QwenImageEditRunResult(
            image=images[0],
            metadata=metadata,
            wall_seconds=float(round(time.perf_counter() - started_at, 3)),
        )

    def _ensure_ready(self) -> None:
        if self._pipeline is not None:
            return
        with self._load_lock:
            if self._pipeline is None:
                self._load_pipeline()

    def _load_pipeline(self) -> None:
        torch_module = __import__("torch")
        diffusers_module = __import__("diffusers", fromlist=["DiffusionPipeline"])
        diffusion_pipeline_cls = diffusers_module.DiffusionPipeline

        requested_device = "cuda" if torch_module.cuda.is_available() else "cpu"
        self._device = requested_device
        torch_dtype = torch_module.bfloat16 if self._device == "cuda" else torch_module.float32
        self._dtype_name = str(torch_dtype).replace("torch.", "")

        source = str(Path(self._settings.qwen_image_edit_model_path).expanduser())
        kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        if self._device == "cuda":
            kwargs["device_map"] = "cuda"

        pipe = diffusion_pipeline_cls.from_pretrained(source, **kwargs)
        if self._device != "cuda":
            pipe = pipe.to(self._device)

        if self._lora_path or self._lora_weight_name:
            if self._lora_weight_name:
                pipe.load_lora_weights(
                    self._lora_path,
                    weight_name=self._lora_weight_name,
                    adapter_name=self._adapter_name,
                )
            else:
                pipe.load_lora_weights(self._lora_path, adapter_name=self._adapter_name)
            self._lora_loaded = True
            if hasattr(pipe, "set_adapters"):
                try:
                    pipe.set_adapters(
                        [self._adapter_name],
                        adapter_weights=[float(self._lora_scale)],
                    )
                except Exception:
                    pipe.set_adapters(
                        self._adapter_name,
                        adapter_weights=[float(self._lora_scale)],
                    )

        self._pipeline = pipe
        self._torch = torch_module
        logger.info(
            "Qwen image-edit pipeline ready (device=%s, lora_loaded=%s, adapter=%s)",
            self._device,
            self._lora_loaded,
            self._adapter_name,
        )

from __future__ import annotations

import threading
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from PIL import Image

from app.constants import minicpm as minicpm_constants


class MiniCPMRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class MiniCPMDescription:
    text: str
    latency_ms: int
    model_id: str
    device: str
    dtype: str
    downsample_mode: str
    max_new_tokens: int
    max_slice_nums: int


class MiniCPMV46Client:
    """Lazy MiniCPM-V 4.6 garment descriptor client."""

    def __init__(
        self,
        *,
        model_id: str = minicpm_constants.MODEL_ID,
        device: str = minicpm_constants.DEFAULT_DEVICE,
        dtype: str = minicpm_constants.DEFAULT_DTYPE,
        downsample_mode: str = minicpm_constants.DEFAULT_DOWNSAMPLE_MODE,
        max_slice_nums: int = minicpm_constants.DEFAULT_MAX_SLICE_NUMS,
        max_new_tokens: int = minicpm_constants.DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self._model_id = model_id
        self._requested_device = device
        self._requested_dtype = dtype
        self._downsample_mode = downsample_mode
        self._max_slice_nums = int(max_slice_nums)
        self._max_new_tokens = int(max_new_tokens)
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device = "unloaded"
        self._dtype = dtype
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def describe_garment(
        self,
        *,
        image: Image.Image,
        prompt: str,
    ) -> MiniCPMDescription:
        self.ensure_ready()
        if self._model is None or self._processor is None or self._torch is None:
            raise MiniCPMRuntimeError("MiniCPM runtime is not loaded.")

        rgb = image.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb},
                    {"type": "text", "text": str(prompt).strip()},
                ],
            },
        ]
        started = perf_counter()
        try:
            with self._infer_lock, self._torch.inference_mode():
                inputs = self._processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    downsample_mode=self._downsample_mode,
                    max_slice_nums=self._max_slice_nums,
                ).to(self._model.device)
                generated_ids = self._model.generate(
                    **inputs,
                    downsample_mode=self._downsample_mode,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                )
                input_length = int(inputs["input_ids"].shape[-1])
                generated_ids_trimmed = [ids[input_length:] for ids in generated_ids]
                decoded = self._processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
        except Exception as exc:
            raise MiniCPMRuntimeError(f"MiniCPM garment description failed: {exc}") from exc

        text = " ".join(str(decoded[0] if decoded else "").split()).strip()
        if not text:
            raise MiniCPMRuntimeError("MiniCPM returned an empty garment description.")

        return MiniCPMDescription(
            text=text,
            latency_ms=round((perf_counter() - started) * 1000),
            model_id=self._model_id,
            device=self._device,
            dtype=self._dtype,
            downsample_mode=self._downsample_mode,
            max_new_tokens=self._max_new_tokens,
            max_slice_nums=self._max_slice_nums,
        )

    def ensure_ready(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            self._load_model()

    def _load_model(self) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForImageTextToText,
                AutoProcessor,
            )
        except Exception as exc:
            raise MiniCPMRuntimeError(
                "MiniCPM requires torch and transformers in the runtime environment.",
            ) from exc

        device_map: Any = self._requested_device
        if device_map == "auto":
            device_map = "auto"
        elif device_map in {"cuda", "cpu"}:
            device_map = {"": device_map}
        else:
            device_map = "auto"

        model_kwargs: dict[str, Any] = {
            "torch_dtype": self._requested_dtype,
            "device_map": device_map,
        }
        if self._requested_dtype == "bf16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif self._requested_dtype == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif self._requested_dtype == "fp32":
            model_kwargs["torch_dtype"] = torch.float32

        try:
            processor = AutoProcessor.from_pretrained(self._model_id)
            model = AutoModelForImageTextToText.from_pretrained(
                self._model_id,
                **model_kwargs,
            )
            model.eval()
        except Exception as exc:
            raise MiniCPMRuntimeError(
                f"Failed to load MiniCPM model {self._model_id}: {exc}",
            ) from exc

        self._torch = torch
        self._processor = processor
        self._model = model
        self._device = str(getattr(model, "device", "auto"))
        self._dtype = str(getattr(model, "dtype", self._requested_dtype))

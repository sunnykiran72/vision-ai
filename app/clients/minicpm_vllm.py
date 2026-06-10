from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings, get_settings
from app.constants import wardrobe as wardrobe_constants

logger = logging.getLogger("glamify-ai")


class MiniCPMRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class MiniCPMCaption:
    text: str
    latency_ms: int
    model_id: str


def _resize_long(
    image: Image.Image,
    target: int = wardrobe_constants.MINICPM_RESIZE_LONG_PX,
) -> Image.Image:
    width, height = image.size
    if max(width, height) <= target:
        return image
    if width >= height:
        new_width, new_height = target, max(8, round(height * target / width))
    else:
        new_height, new_width = target, max(8, round(width * target / height))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


class MiniCPMVllmClient:
    """In-process MiniCPM-V garment captioner via vLLM.

    A faithful port of the validated reference engine: vLLM loads the model in this process and
    coexists on the GPU beside the resident Qwen model (capped via gpu_memory_utilization, eager,
    single-sequence). Algorithmic tuning is in ``app/constants/wardrobe.py``; deployment-specific
    model and memory settings come from ``Settings``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = str(settings.minicpm_model_path or "").strip()
        self._llm: Any | None = None
        self._tokenizer: Any | None = None
        self._sampling_params: Any | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._llm is not None and self._tokenizer is not None

    def warmup(self) -> None:
        self.ensure_ready()

    def ensure_ready(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            self._load_model()

    def describe_garment(self, *, image: Image.Image, prompt: str) -> MiniCPMCaption:
        self.ensure_ready()
        if self._llm is None or self._tokenizer is None or self._sampling_params is None:
            raise MiniCPMRuntimeError("MiniCPM vLLM runtime is not loaded.")

        rgb = _resize_long(
            image.convert("RGB"),
            target=self._settings.minicpm_resize_long_px,
        )
        messages = [{"role": "user", "content": "(<image>./</image>)\n" + str(prompt).strip()}]
        started = time.perf_counter()
        try:
            with self._infer_lock:
                try:
                    text = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    text = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                outputs = self._llm.generate(
                    {"prompt": text, "multi_modal_data": {"image": rgb}},
                    self._sampling_params,
                    use_tqdm=False,
                )
        except Exception as exc:
            raise MiniCPMRuntimeError(f"MiniCPM caption failed: {exc}") from exc

        raw = str(outputs[0].outputs[0].text).strip()
        # MiniCPM-V 4.5 is Qwen3-based and can emit <think>...</think>; strip any reasoning block.
        if "</think>" in raw:
            raw = raw.rsplit("</think>", 1)[-1].strip()
        caption = " ".join(raw.replace("<think>", "").split()).strip()
        if not caption:
            raise MiniCPMRuntimeError("MiniCPM returned an empty caption.")
        return MiniCPMCaption(
            text=caption,
            latency_ms=round((time.perf_counter() - started) * 1000),
            model_id=self._model,
        )

    def _load_model(self) -> None:
        if not self._model:
            raise MiniCPMRuntimeError("MINICPM_MODEL_PATH is not configured.")
        # A local path is validated; an HF repo id is loaded from the cache.
        if "/" in self._model and Path(self._model).expanduser().exists():
            self._model = str(Path(self._model).expanduser())
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
            from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - environment dependency
            raise MiniCPMRuntimeError(
                "MiniCPM requires vllm and transformers in the runtime environment.",
            ) from exc

        logger.info("Loading MiniCPM-V via vLLM: %s", self._model)
        tokenizer = AutoTokenizer.from_pretrained(self._model, trust_remote_code=True)
        llm_kwargs: dict[str, Any] = {
            "model": self._model,
            "trust_remote_code": True,
            "gpu_memory_utilization": self._settings.minicpm_gpu_memory_utilization,
            "max_model_len": self._settings.minicpm_max_model_len,
            "limit_mm_per_prompt": {"image": 1},
            "dtype": self._settings.minicpm_dtype,
            "enforce_eager": self._settings.minicpm_enforce_eager,
            "max_num_seqs": 1,
            "mm_processor_kwargs": {
                "max_slice_nums": self._settings.minicpm_max_slice_nums,
            },
        }
        if self._settings.minicpm_kv_cache_dtype:
            llm_kwargs["kv_cache_dtype"] = self._settings.minicpm_kv_cache_dtype
            llm_kwargs["calculate_kv_scales"] = self._settings.minicpm_calculate_kv_scales
        if self._settings.minicpm_attention_backend:
            llm_kwargs["attention_config"] = {
                "backend": self._settings.minicpm_attention_backend,
            }
        llm = LLM(**llm_kwargs)
        stop_ids: list[int] | None
        try:
            candidate_ids = [
                tokenizer.convert_tokens_to_ids(token)
                for token in ["<|im_end|>", "<|endoftext|>"]
            ]
            stop_ids = [s for s in candidate_ids if isinstance(s, int) and s >= 0] or None
        except Exception:
            stop_ids = None
        self._sampling_params = SamplingParams(
            temperature=wardrobe_constants.MINICPM_TEMPERATURE,
            max_tokens=self._settings.minicpm_max_tokens,
            stop_token_ids=stop_ids,
        )
        self._tokenizer = tokenizer
        self._llm = llm
        logger.info("MiniCPM-V vLLM runtime ready")


@lru_cache(maxsize=1)
def get_minicpm_client() -> MiniCPMVllmClient:
    """Process-wide singleton, shared by the wardrobe flow and the MiniCPM dev endpoint."""
    return MiniCPMVllmClient(get_settings())

from __future__ import annotations

import io
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image


MODEL_PATH = os.environ.get("AWQ_MODEL_PATH", "/workspace/models/minicpm-v-4_5-awq")
HOST = os.environ.get("AWQ_HOST", "127.0.0.1")
PORT = int(os.environ.get("AWQ_PORT", "8020"))
GPU_UTIL = float(os.environ.get("AWQ_GPU_UTILIZATION", "0.10"))
MAX_MODEL_LEN = int(os.environ.get("AWQ_MAX_MODEL_LEN", "2048"))
MAX_TOKENS = int(os.environ.get("AWQ_MAX_TOKENS", "100"))
RESIZE_LONG = int(os.environ.get("AWQ_RESIZE_LONG", "1024"))
MAX_SLICE_NUMS = int(os.environ.get("AWQ_MAX_SLICE_NUMS", "4"))
DTYPE = os.environ.get("AWQ_DTYPE", "auto")


PROMPT_BY_TYPE = {
    "top": (
        "You are a fashion product specialist writing a precise description that will be used to "
        "regenerate this garment as an image. Describe ONLY the upper-body garment (the top / "
        "shirt / jacket) in this image. Completely ignore the person (face, hair, skin, body, "
        "midriff, pose), the lower-body garment (trousers/pants/skirt), footwear, accessories, "
        "and the background - do not mention them at all. Write one clear, flowing paragraph of "
        "about 30-45 words capturing every visible detail needed to recreate it: garment type, "
        "collar/neckline, closure (buttons/zip/ties), sleeves/straps or waistline/legs, fit and "
        "silhouette, fabric and texture, all colours, and the print/pattern (motif types, their "
        "colours, and where they sit). Only what is clearly visible; never guess hidden parts. "
        "Plain factual prose - no labels, no lists, no headings, no preamble."
    ),
    "bottom": (
        "You are a fashion product specialist writing a precise description that will be used to "
        "regenerate this garment as an image. Describe ONLY the lower-body garment (the trousers / "
        "pants / skirt / shorts) in this image. Completely ignore the person, the upper-body "
        "garment (top/shirt), footwear, accessories, and the background - do not mention them at "
        "all. Write one clear, flowing paragraph of about 30-45 words capturing every visible "
        "detail needed to recreate it: garment type, collar/neckline, closure (buttons/zip/ties), "
        "sleeves/straps or waistline/legs, fit and silhouette, fabric and texture, all colours, "
        "and the print/pattern (motif types, their colours, and where they sit). Only what is "
        "clearly visible; never guess hidden parts. Plain factual prose - no labels, no lists, "
        "no headings, no preamble."
    ),
    "dress": (
        "You are a fashion product specialist writing a precise description that will be used to "
        "regenerate this garment as an image. Describe ONLY the dress in this image. Completely "
        "ignore the person (face, hair, skin, body, pose), footwear, accessories, and the "
        "background - do not mention them at all. Write one clear, flowing paragraph of about "
        "30-45 words capturing every visible detail needed to recreate it: garment type, "
        "collar/neckline, closure (buttons/zip/ties), sleeves/straps or waistline/legs, fit and "
        "silhouette, fabric and texture, all colours, and the print/pattern (motif types, their "
        "colours, and where they sit). Only what is clearly visible; never guess hidden parts. "
        "Plain factual prose - no labels, no lists, no headings, no preamble."
    ),
}


@dataclass(frozen=True)
class AwqConfig:
    model_path: str
    dtype: str
    gpu_memory_utilization: float
    max_model_len: int
    max_tokens: int
    resize_long: int
    max_slice_nums: int
    kv_cache_dtype: str
    quantization: str
    enforce_eager: bool
    max_num_seqs: int
    limit_mm_per_prompt: dict[str, int]
    mm_processor_kwargs: dict[str, int]


CONFIG = AwqConfig(
    model_path=MODEL_PATH,
    dtype=DTYPE,
    gpu_memory_utilization=GPU_UTIL,
    max_model_len=MAX_MODEL_LEN,
    max_tokens=MAX_TOKENS,
    resize_long=RESIZE_LONG,
    max_slice_nums=MAX_SLICE_NUMS,
    kv_cache_dtype="auto",
    quantization="awq_marlin",
    enforce_eager=True,
    max_num_seqs=1,
    limit_mm_per_prompt={"image": 1},
    mm_processor_kwargs={"max_slice_nums": MAX_SLICE_NUMS},
)


def resize_long(image: Image.Image, target: int = RESIZE_LONG) -> Image.Image:
    width, height = image.size
    if max(width, height) <= target:
        return image
    if width >= height:
        new_width, new_height = target, max(8, round(height * target / width))
    else:
        new_height, new_width = target, max(8, round(width * target / height))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def strip_reasoning(text: str) -> str:
    raw = text.strip()
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[-1].strip()
    return " ".join(raw.replace("<think>", "").split()).strip()


class AwqCaptioner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._load_started_at = 0.0
        self._load_finished_at = 0.0
        self._llm: Any | None = None
        self._tokenizer: Any | None = None
        self._sampling_params: Any | None = None
        self._load_error = ""

    @property
    def loaded(self) -> bool:
        return self._llm is not None and self._tokenizer is not None

    def status(self) -> dict[str, object]:
        return {
            "ready": self.loaded,
            "loading": self._load_started_at > 0 and not self.loaded and not self._load_error,
            "load_seconds": (
                round(self._load_finished_at - self._load_started_at, 3)
                if self._load_started_at and self._load_finished_at
                else None
            ),
            "load_error": self._load_error,
            "config": asdict(CONFIG),
            "prompts": PROMPT_BY_TYPE,
        }

    def ensure_loaded(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            self._load()

    def caption(self, *, image: Image.Image, prompt: str) -> tuple[str, int]:
        self.ensure_loaded()
        if self._llm is None or self._tokenizer is None or self._sampling_params is None:
            raise RuntimeError("AWQ runtime is not loaded.")
        messages = [{"role": "user", "content": "(<image>./</image>)\n" + prompt.strip()}]
        try:
            chat_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            chat_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        started = time.perf_counter()
        with self._lock:
            outputs = self._llm.generate(
                {"prompt": chat_prompt, "multi_modal_data": {"image": image}},
                self._sampling_params,
                use_tqdm=False,
            )
        return strip_reasoning(str(outputs[0].outputs[0].text)), round(
            (time.perf_counter() - started) * 1000,
        )

    def _load(self) -> None:
        self._load_error = ""
        self._load_started_at = time.perf_counter()
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams

            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            self._llm = LLM(
                model=MODEL_PATH,
                trust_remote_code=True,
                gpu_memory_utilization=GPU_UTIL,
                max_model_len=MAX_MODEL_LEN,
                limit_mm_per_prompt=CONFIG.limit_mm_per_prompt,
                dtype=DTYPE,
                enforce_eager=True,
                max_num_seqs=1,
                mm_processor_kwargs=CONFIG.mm_processor_kwargs,
            )
            try:
                candidate_ids = [
                    self._tokenizer.convert_tokens_to_ids(token)
                    for token in ["<|im_end|>", "<|endoftext|>"]
                ]
                stop_ids = [
                    token_id
                    for token_id in candidate_ids
                    if isinstance(token_id, int) and token_id >= 0
                ]
            except Exception:
                stop_ids = []
            self._sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                stop_token_ids=stop_ids or None,
            )
            self._load_finished_at = time.perf_counter()
        except Exception as exc:
            self._load_error = str(exc)
            self._load_finished_at = time.perf_counter()
            raise


captioner = AwqCaptioner()
app = FastAPI(title="MiniCPM-V-4.5 AWQ Caption Tester")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/status")
def status() -> JSONResponse:
    return JSONResponse(captioner.status())


@app.get("/api/prompts")
def prompts() -> JSONResponse:
    return JSONResponse(PROMPT_BY_TYPE)


@app.post("/api/load")
def load() -> JSONResponse:
    try:
        captioner.ensure_loaded()
    except Exception as exc:
        raise HTTPException(500, f"AWQ load failed: {exc}") from exc
    return JSONResponse(captioner.status())


@app.post("/api/caption")
async def caption(
    source: UploadFile = File(...),
    type: str = Form("top"),
    prompt: str = Form(""),
) -> JSONResponse:
    garment_type = type if type in PROMPT_BY_TYPE else "top"
    resolved_prompt = prompt.strip() or PROMPT_BY_TYPE[garment_type]
    try:
        image = resize_long(Image.open(io.BytesIO(await source.read())).convert("RGB"))
        text, latency_ms = captioner.caption(image=image, prompt=resolved_prompt)
    except Exception as exc:
        raise HTTPException(500, f"caption failed: {exc}") from exc
    return JSONResponse(
        {
            "caption": text,
            "words": len(text.split()),
            "ms": latency_ms,
            "type": garment_type,
            "prompt": resolved_prompt,
            "config": asdict(CONFIG),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)

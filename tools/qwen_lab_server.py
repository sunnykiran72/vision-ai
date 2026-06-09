"""Standalone Qwen-Image-Edit extraction lab (fp8 + compile_repeated_blocks + cache-dit).

Isolated from the main app. Warm-loads ONE category LoRA (env CATEGORY, default "top"),
fuses it, quantizes the transformer to fp8 (torchao), enables cache-dit (DBCache), and
compiles the repeated blocks. Serves an HTML page + a /run endpoint that takes
image+prompt+seed+steps+dims and returns the output image plus a full timing/metric breakdown.

Run on the pod (port 8000 is the RunPod-proxied one):
  LD_LIBRARY_PATH=<venv>/.../nvidia/cu13/lib \
  CATEGORY=top CACHE_THR=0.20 \
  python tools/qwen_lab_server.py
"""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import torch
import torch._dynamo
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "/workspace/models/qwen-image-edit-2511")
CATEGORY = os.environ.get("CATEGORY", "top")
CACHE_THR = float(os.environ.get("CACHE_THR", "0.20"))
CACHE_WARMUP = int(os.environ.get("CACHE_WARMUP", "2"))
PORT = int(os.environ.get("PORT", "8000"))
LORA_PATHS = {
    "top": "/workspace/loras/wardrobe/top_23000.safetensors",
    "bottom": "/workspace/loras/wardrobe/bottom_30000.safetensors",
    "dress": "/workspace/loras/wardrobe/dress_27000.safetensors",
}
PAGE = Path(__file__).resolve().parent / "qwen_lab.html"
WARM_IMG = os.environ.get("WARM_IMG", "/workspace/awq_test_samples/21f835e64efd23eb_top.jpg")
WARM_PROMPT = (
    "GlamTopExt. Extract top wear as a standalone product. Target regenerate garment is the garment ; "
    "Keep the garment's exact shape, fabric texture, color and print. remove the person, other clothing, "
    "background and shadows. fill skin-revealing gaps with clean white. Present it as a centered product "
    "on a pure white background with sharp and precise details of original garment."
)

app = FastAPI(title="qwen-lab")
_state: dict = {"ready": False, "pipe": None}


def _load():
    from cache_dit import DBCacheConfig, enable_cache
    from diffusers import QwenImageEditPlusPipeline
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

    t0 = time.time()
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(LORA_PATHS[CATEGORY])
    pipe.fuse_lora(lora_scale=1.0)
    # Move to GPU BEFORE quantizing: fp8 quantize on CPU is ~15min on this pod; on GPU it's ~1min.
    pipe.to("cuda")
    quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
    # cache-dit is OFF by default: its step-skipping is data-dependent, which makes torch.compile
    # recompile (~100s) whenever a new image's cache-pattern differs from warmup -> request timeouts.
    # No-cache + compile is deterministic: compiled ONCE, reused for every image (~6.5s, reliable).
    # Quality is identical to cache@0.20 (PSNR ~35). Set LAB_CACHE=1 to re-enable (benchmark only).
    if os.environ.get("LAB_CACHE", "0") == "1":
        enable_cache(
            pipe,
            cache_config=DBCacheConfig(
                residual_diff_threshold=CACHE_THR,
                max_warmup_steps=CACHE_WARMUP,
                enable_separate_cfg=False,
            ),
        )
    # Compile ON by default: fp8 is only fast WITH compile (fused fp8 GEMM ~0.3s/step vs
    # ~1.5s/step uncompiled). Combined with FIXED input dims (see _preprocess) the compile
    # is built once at warmup and reused for every image -> no per-aspect recompile.
    if os.environ.get("LAB_COMPILE", "1") == "1":
        for attr in ("cache_size_limit", "recompile_limit", "accumulated_recompile_limit"):
            try:
                setattr(torch._dynamo.config, attr, 4096)
            except Exception:
                pass
        pipe.transformer.compile_repeated_blocks(fullgraph=False)
    _state["pipe"] = pipe
    _state["cache_on_thr"] = CACHE_THR if os.environ.get("LAB_CACHE", "0") == "1" else 0.0
    _state["load_s"] = round(time.time() - t0, 1)
    # warmup with a REAL sample image (not gray) so the compiled+cache path matches real requests.
    try:
        warm_img = _preprocess(Image.open(WARM_IMG).convert("RGB"))
    except Exception:
        warm_img = Image.new("RGB", (768, 1024), (127, 127, 127))
    for _ in range(2):  # 2 warmups so the compiled cache pattern settles
        _generate(warm_img, WARM_PROMPT, 7777, 15, 832, 1248)
    _state["ready"] = True
    print(f"[qwen-lab] READY category={CATEGORY} thr={CACHE_THR} cache_on={_state['cache_on_thr']} load={_state['load_s']}s", flush=True)


def _preprocess(im: Image.Image) -> Image.Image:
    # FIXED input size so the compiled graph is reused for EVERY image (no per-aspect recompile).
    # Matches the warmup shape. Mild aspect distortion is acceptable for the extraction LoRA.
    return im.convert("RGB").resize((768, 1024), Image.LANCZOS)


def _generate(img, prompt, seed, steps, width, height):
    pipe = _state["pipe"]
    ts: list[float] = []

    def cb(p, step, t, kw):
        torch.cuda.synchronize()
        ts.append(time.time())
        return kw

    g = torch.Generator(device="cuda").manual_seed(int(seed))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    out = pipe(image=[img], prompt=prompt, true_cfg_scale=1.0,
               num_inference_steps=int(steps), height=int(height), width=int(width),
               generator=g, callback_on_step_end=cb).images[0]
    torch.cuda.synchronize()
    t1 = time.time()
    per = (ts[-1] - ts[0]) / (len(ts) - 1) if len(ts) > 1 else (t1 - t0)
    timings = {
        "total_s": round(t1 - t0, 3),
        "encode_s": round(max(0.0, ts[0] - t0 - per), 3),
        "denoise_s": round((ts[-1] - ts[0]) + per, 3),
        "decode_s": round(t1 - ts[-1], 3),
        "per_step_s": round(per, 3),
    }
    vram = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    return out, timings, vram


@app.get("/tools/qwen-lab", include_in_schema=False)
def page():
    return FileResponse(PAGE)


@app.get("/tools/qwen-lab/status")
def status():
    return {"ready": _state["ready"], "category": CATEGORY, "cache_thr": CACHE_THR,
            "cache_warmup": CACHE_WARMUP, "load_s": _state.get("load_s")}


def _set_cache(thr: float):
    """Toggle cache-dit per request. Skips if already at this threshold (avoids re-toggle)."""
    thr = float(thr or 0.0)
    if abs(thr - float(_state.get("cache_on_thr", 0.0))) < 1e-9:
        return  # already in the requested state
    import cache_dit
    from cache_dit import DBCacheConfig
    pipe = _state["pipe"]
    try:
        cache_dit.disable_cache(pipe)
    except Exception:
        pass
    if thr > 0:
        cache_dit.enable_cache(
            pipe,
            cache_config=DBCacheConfig(
                residual_diff_threshold=thr,
                max_warmup_steps=CACHE_WARMUP,
                enable_separate_cfg=False,
            ),
        )
    _state["cache_on_thr"] = thr


@app.post("/tools/qwen-lab/run")
async def run(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    seed: int = Form(7777),
    steps: int = Form(15),
    width: int = Form(832),
    height: int = Form(1248),
    cache_thr: float = Form(0.0),
):
    if not _state["ready"]:
        return JSONResponse({"ok": False, "error": "pipeline still warming up"}, status_code=503)
    raw = await image.read()
    try:
        src = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"bad image: {e}"}, status_code=400)
    in_w, in_h = src.size
    pre = _preprocess(src)
    try:
        _set_cache(cache_thr)
        out, timings, vram = _generate(pre, prompt, seed, steps, width, height)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return {
        "ok": True,
        "output_image": b64,
        "timings": timings,
        "meta": {
            "category": CATEGORY, "seed": int(seed), "steps": int(steps),
            "cache_thr": float(cache_thr), "cache_warmup": CACHE_WARMUP,
            "input_dims": f"{in_w}x{in_h}", "preprocessed_dims": f"{pre.size[0]}x{pre.size[1]}",
            "output_dims": f"{out.size[0]}x{out.size[1]}",
            "vram_gb": vram, "precision": "fp8-dynamic", "prompt": prompt,
        },
    }


if __name__ == "__main__":
    print("[qwen-lab] loading pipeline (fp8+cache+compile, ~5min first compile)...", flush=True)
    _load()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")

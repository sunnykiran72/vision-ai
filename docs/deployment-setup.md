# Vision-AI Deployment & Setup

How to install, configure, and run the bundled vision-ai service on a fresh RunPod pod backed by a
network volume. The service exposes all 4 GPU APIs (`/v1/wardrobe`, `/v1/tryon`, `/v1/upscale`,
`/v1/user_validation`) from one process on **port 8000**.

> Target: this is the manual install for testing now. The end goal is a single Docker image that
> bakes this stack so a RunPod template boots all 4 APIs instantly (see "Future: single Docker
> image"). Stage models on the network volume so they are not re-downloaded per pod.

## 0. The one compatibility gate to validate first

The two reference apps deliberately ran in **separate** Python environments:
- Qwen diffusers tester (port 8000): transformers `5.10.2`
- MiniCPM-V-4.5 vLLM (port 8010): transformers `4.57.x` (its header notes **transformers v5 breaks
  MiniCPM-V** — `tokenizer.im_start_id` was removed)

Bundling both **in one process** requires a single transformers that satisfies both. We target
**transformers `>=4.57,<5`** and must confirm the Qwen `QwenImageEditPlusPipeline` also loads under
it. `scripts/validate_gpu_stack.py` checks the imports; final proof is the service warmup loading
both models. If Qwen turns out to require transformers 5, the fallback is to run MiniCPM in a second
process/venv inside the same image (still one image, two envs) — but validate the single-env path
first.

## 1. Pod requirements

- **GPU**: H100 80 GB minimum for bf16 Qwen + fp8 MiniCPM + fp8 SeedVR2 resident together;
  H200 141 GB / Blackwell 96 GB+ is comfortable. fp8 throughput needs Hopper/Blackwell (sm_90+).
- **CUDA**: 12.8 driver (matches `torch==2.8.0+cu128`).
- **Python**: 3.12 (vLLM 0.22.1 / torch cu128 wheels). Do **not** use 3.13 on the pod.
- **Network volume**: ~200 GB+ to hold models, LoRAs, and the HF cache.

## 2. Network volume layout

Pick a volume mount (RunPod volumes usually mount at `/workspace`). Suggested layout:

```text
/workspace/
  vision-ai/                      # this repo
  models/
    qwen-image-edit-2511/         # Qwen Image Edit Plus base
    minicpm-v-4_5/                # MiniCPM-V-4.5 (or use HF id + HF_HOME cache)
    seedvr2/                      # SeedVR2 weights dir (contains the .safetensors variant)
  loras/
    wardrobe/
      top_23000.safetensors
      bottom_30000.safetensors
      dress_27000.safetensors
  seedvr2_eval/
    ComfyUI-SeedVR2_VideoUpscaler/inference_cli.py
  hf_cache/                       # HF_HOME for auto-downloaded models
```

Detector (`yainage90/fashion-object-detection`) and Marqo (`Marqo/marqo-fashionSigLIP`)
auto-download from HF into `HF_HOME` on first warmup; prefetch them to the volume to avoid cold
downloads.

## 3. Install the GPU stack

```bash
cd /workspace/vision-ai
export HF_HOME=/workspace/hf_cache
bash scripts/install_gpu_stack.sh           # pinned reference versions, Python 3.12
python3.12 scripts/validate_gpu_stack.py    # must print RESULT: PASS
```

The heavy stack (torch, vllm, transformers<5, diffusers, accelerate, safetensors, open_clip) is
installed by the script and is intentionally absent from `pyproject.toml` (the app imports it
lazily; `pyproject.toml` holds only the light web deps for local dev/CI).

vLLM launch env (the reference's validated settings for Blackwell — FlashInfer fails on sm_120):

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MINICPM_ATTENTION_BACKEND=TRITON_ATTN
```

## 4. Stage models on the volume

| Asset | Source | Used by |
| --- | --- | --- |
| Qwen-Image-Edit-2511 | HF / copy to `models/qwen-image-edit-2511` | wardrobe (+ tryon) |
| Wardrobe LoRAs (top 23k / bottom 30k / dress 27k) | trained artifacts, copy to `loras/wardrobe/` | wardrobe |
| MiniCPM-V-4.5 | `openbmb/MiniCPM-V-4_5` (HF id or local dir) | wardrobe caption |
| SeedVR2 7B fp8 | `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors` + the CLI repo | upscale |
| fashion-object-detection | `yainage90/fashion-object-detection` (auto) | wardrobe gate |
| Marqo fashionSigLIP | `Marqo/marqo-fashionSigLIP` (auto, open_clip) | wardrobe category |
| Try-on LoRAs + AI-Toolkit | only if `tryon` is in `RESIDENT_RUNTIMES` | tryon |

## 5. Configure `.env`

Copy `.env.example` to `.env` and fill it in. Minimum for a wardrobe + upscale pod:

```bash
APP_ENV=runpod
RESIDENT_RUNTIMES=wardrobe,upscale          # add 'tryon' only when its LoRAs/AI-Toolkit are staged
SYSTEM_QUEUE_MAX_SIZE=8
SYSTEM_QUEUE_WAIT_TIMEOUT_SECONDS=30

JWT_ACCESS_SECRET=<shared secret with the Glamify backend>

AZURE_STORAGE_CONNECTION_STRING=<...>
AZURE_STORAGE_CONTAINER=<base container>
AZURE_WARDROBE_INPUT_CONTAINER=wardrobe-inputs
AZURE_WARDROBE_OUTPUT_CONTAINER=wardrobe-outputs

QWEN_IMAGE_EDIT_MODEL_PATH=/workspace/models/qwen-image-edit-2511
QWEN_IMAGE_EDIT_DTYPE=bfloat16
QWEN_COMPILE=false
WARDROBE_LORA_TOP_PATH=/workspace/loras/wardrobe/top_23000.safetensors
WARDROBE_LORA_BOTTOM_PATH=/workspace/loras/wardrobe/bottom_30000.safetensors
WARDROBE_LORA_DRESS_PATH=/workspace/loras/wardrobe/dress_27000.safetensors
MINICPM_MODEL_PATH=/workspace/models/minicpm-v-4_5-awq
MINICPM_DTYPE=auto
MINICPM_GPU_MEMORY_UTILIZATION=0.10
MINICPM_KV_CACHE_DTYPE=auto
MINICPM_CALCULATE_KV_SCALES=false
MINICPM_ATTENTION_BACKEND=
MINICPM_MAX_TOKENS=100
MINICPM_MAX_MODEL_LEN=2048
MINICPM_MAX_SLICE_NUMS=6
MINICPM_RESIZE_LONG_PX=1024
MINICPM_ENFORCE_EAGER=false

UPSCALE_MODEL_PATH=/workspace/models/seedvr2
UPSCALE_MODEL_VARIANT=seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors
UPSCALE_CLI_PATH=/workspace/seedvr2_eval/ComfyUI-SeedVR2_VideoUpscaler/inference_cli.py

GLAMIFY_API_BASE_URL=<glamify backend base url>
```

Startup validation only requires the config for the runtimes in `RESIDENT_RUNTIMES`
(`app/config.py:validate_startup_settings`). `AI_TOOLKIT_ROOT` is required only when `tryon` is
resident.

## 6. Run on port 8000

```bash
cd /workspace/vision-ai
export PORT=8000
python3.12 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

`--workers 1` is mandatory: the single in-process GPU coordinator and resident models assume one
worker. Startup **warm-loads everything** before serving (Qwen base + 3 LoRAs + a warm pass,
MiniCPM via vLLM, the detector, and Marqo) and never unloads. First boot is slow (model load +
MiniCPM vLLM init + one Qwen warm pass); subsequent requests are fast.

Expose port `8000` in the RunPod template (HTTP).

## 7. Smoke tests

```bash
# health (public)
curl -s http://localhost:8000/health | jq

# diffusers extraction, no auth (parity tool)
curl -s -X POST http://localhost:8000/tools/diffusers/extract \
  -F source=@garment.jpg -F lora_key=top -o out.jpg

# full wardrobe (multipart in, JSON URL out) — needs a valid JWT
curl -s -X POST http://localhost:8000/v1/wardrobe \
  -H "Authorization: Bearer $JWT" \
  -F image=@garment.jpg -F type=top | jq
```

## 8. VRAM & dtype

- Qwen: bf16 (faithful baseline). Keep `QWEN_COMPILE=false` for production until variable
  prompt/image shapes are handled without graph-specialization spikes. `QWEN_COMPILE=true` remains
  a benchmark-only option for repeated fixed-shape prompts. MiniCPM: AWQ weights via
  vLLM, capped by `MINICPM_GPU_MEMORY_UTILIZATION=0.10`, with CUDA graphs enabled by
  `MINICPM_ENFORCE_EAGER=false`. SeedVR2: the fp8 mixed variant above plus
  `ema_vae_fp16.safetensors`.
- Warm all three, then check `nvidia-smi`. If tight: lower `MINICPM_GPU_MEMORY_UTILIZATION`,
  or drop SeedVR2 to its 3B fp8 variant via `UPSCALE_MODEL_VARIANT`.
- On the RTX PRO 6000 Blackwell 96 GB pod, the measured resident set is 88,421 MiB used after a
  SeedVR2 tiny run, leaving 8,828 MiB free. On an 80 GB card this stack is not safe as a fully
  resident all-model bundle without reducing the resident set or memory caps.
- See `docs/wardrobe-flow.md` (constants) and the dtype guidance for the bf16-vs-fp8 reasoning.

## 9. Troubleshooting

- **MiniCPM caption errors / tokenizer attribute missing**: transformers is `>=5`. Pin `<5`.
- **vLLM FlashInfer crash (sm_120 / Blackwell)**: set `MINICPM_ATTENTION_BACKEND=TRITON_ATTN`
  and keep `VLLM_USE_FLASHINFER_SAMPLER=0`. The older `VLLM_ATTENTION_BACKEND` and
  `VLLM_USE_FLASHINFER` environment variables are not recognized by this vLLM build.
- **CUDA OOM at warmup**: lower the MiniCPM GPU utilization constant, use the SeedVR2 3B variant,
  or set the SeedVR2 tensor offload to CPU (see `app/clients/seedvr2.py`).
- **Startup config error**: a required env for a resident runtime is missing/invalid — the message
  lists which key.
- **Slow first request**: expected if warmup was skipped; warmup runs automatically in the app
  lifespan, so let startup finish before sending traffic.

## Future: single Docker image

To make RunPod deploys instant and repeatable:

1. Base image: a CUDA 12.8 + Python 3.12 image (e.g. an official PyTorch cu128 image).
2. `COPY` this repo and `RUN bash scripts/install_gpu_stack.sh` to bake the GPU stack into a layer.
3. Do **not** bake model weights into the image — keep them on the network volume and mount it, so
   the image stays small and models are shared across pods.
4. Entry point runs uvicorn on `8000` with one worker; warmup loads the volume-mounted models.
5. Publish as a RunPod template with port `8000` exposed and the volume + `.env` wired in.

This keeps the heavy, slow-changing layers (deps) baked and the fast-changing parts (code via
volume or thin rebuild, models via volume) cheap to update — so you stop re-installing per pod.

# Glamify Image AI RunPod Setup

Working setup notes for the fresh `sporting_violet_lemming` pod. This is the operational checklist
for building the production `glamify-image-ai` service on port `8000`.

## Pod facts

- Pod name: `sporting_violet_lemming`
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, about 96 GB VRAM
- Driver: 580.142
- OS: Ubuntu 24.04
- Python: 3.12.3
- Existing base package: torch 2.8.0+cu128 was already installed globally
- Network volume: `/workspace`

## Directory layout

```text
/workspace/
  glamify-image-ai/              # synced app repository
  .venvs/glamify-image-ai/       # uv-managed Python 3.12 environment
  hf_cache/                      # Hugging Face cache and auth
  models/
    qwen-image-edit-2511/
    minicpm-v-4_5/
    seedvr2/
  loras/
    wardrobe/
    tryon/
  seedvr2_eval/
    ComfyUI-SeedVR2_VideoUpscaler/
  logs/
    glamify-image-ai/
  tmp/
    glamify/
```

## Install order

1. Install latest `uv`.
2. Create an isolated Python 3.12 environment:

   ```bash
   uv venv /workspace/.venvs/glamify-image-ai --python /usr/bin/python3.12
   ```

3. Install torch first, pinned to the CUDA 12.8 wheel:

   ```bash
   uv pip install --python /workspace/.venvs/glamify-image-ai/bin/python \
     "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
   ```

4. Install vLLM, then re-pin transformers below v5:

   ```bash
   uv pip install --python /workspace/.venvs/glamify-image-ai/bin/python "vllm==0.22.1"
   uv pip install --python /workspace/.venvs/glamify-image-ai/bin/python "transformers>=4.57,<5"
   ```

   On this pod, `vllm==0.22.1` replaced torch `2.8.0+cu128` with torch `2.11.0` and initially
   installed transformers `5.10.2`. The transformers re-pin is mandatory for MiniCPM-V-4.5.

5. Install Qwen diffusers, SeedVR2 support dependencies, Marqo/open_clip, and app dependencies.
   `diffusers==0.38.0` requires a safetensors prerelease, so uv needs explicit prerelease allow:

   ```bash
   uv pip install --python /workspace/.venvs/glamify-image-ai/bin/python \
     --prerelease=allow "diffusers==0.38.0" "accelerate>=1.0" "safetensors>=0.8.0rc0"
   uv pip install --python /workspace/.venvs/glamify-image-ai/bin/python "open_clip_torch>=2.24"
   uv pip install --python /workspace/.venvs/glamify-image-ai/bin/python -e /workspace/glamify-image-ai
   ```

## Installed package snapshot

Current uv environment after install:

| Package | Version |
| --- | --- |
| torch | `2.11.0` |
| vLLM | `0.22.1` |
| transformers | `4.57.6` |
| diffusers | `0.38.0` |
| safetensors | `0.8.0rc1` |
| accelerate | `1.13.0` |
| open_clip_torch | `3.3.0` |
| fastapi | `0.136.3` |
| uvicorn | `0.49.0` |

Validation result:

```text
scripts/validate_gpu_stack.py -> RESULT: PASS
torch: 2.11.0+cu130, cuda=True
transformers: 4.57.6
diffusers: 0.38.0, QwenImageEditPlusPipeline importable
vLLM: 0.22.1
open_clip: 3.3.0
safetensors: importable
```

vLLM prints a warning that its transformers-v4 path is deprecated and will be removed in a future
vLLM release. We keep transformers `<5` for now because MiniCPM-V-4.5 breaks on transformers v5.

## Runtime settings

Required Blackwell/vLLM runtime flags:

```bash
export HF_HOME=/workspace/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

MiniCPM is loaded in-process through vLLM. The production defaults are:

```bash
MINICPM_MODEL_PATH=openbmb/MiniCPM-V-4_5
MINICPM_DTYPE=bfloat16
MINICPM_KV_CACHE_DTYPE=fp8
MINICPM_CALCULATE_KV_SCALES=true
MINICPM_ATTENTION_BACKEND=TRITON_ATTN
```

On this Blackwell pod, vLLM 0.22.1 ignores the older `VLLM_ATTENTION_BACKEND` environment variable
and may auto-select FlashInfer. Force `MINICPM_ATTENTION_BACKEND=TRITON_ATTN` so MiniCPM does not
enter the FlashInfer path.

SeedVR2 uses the fp8 mixed model variant:

```bash
UPSCALE_MODEL_PATH=/workspace/models/seedvr2
UPSCALE_MODEL_VARIANT=seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors
UPSCALE_CLI_PATH=/workspace/seedvr2_eval/ComfyUI-SeedVR2_VideoUpscaler/inference_cli.py
```

The SeedVR2 CLI also requires `ema_vae_fp16.safetensors` in the same model directory. The verified
`/workspace/models/seedvr2` contents are:

| file | size | note |
|---|---:|---|
| `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors` | 7.9 GB | DiT model |
| `ema_vae_fp16.safetensors` | 479 MB | required VAE, SHA256 `20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1` |

## Validation gates

Run these before model warmup:

```bash
cd /workspace/glamify-image-ai
/workspace/.venvs/glamify-image-ai/bin/python scripts/validate_gpu_stack.py
```

Expected:

- torch imports and sees CUDA
- transformers is `<5`
- `diffusers.QwenImageEditPlusPipeline` imports
- vLLM imports
- open_clip imports
- safetensors imports

Then run startup validation for the selected `RESIDENT_RUNTIMES`. Full warmup is the final proof,
because it actually loads Qwen, MiniCPM, detector, Marqo, and optionally SeedVR2/try-on.

## Remaining staging work

- Qwen Image Edit Plus is staged at `/workspace/models/qwen-image-edit-2511` (~54 GB).
- MiniCPM-V-4.5 is staged at `/workspace/models/minicpm-v-4_5` (~17 GB).
- Fashion detector is staged at `/workspace/models/fashion-object-detection` (~169 MB).
- Marqo fashionSigLIP is staged at `/workspace/models/marqo-fashionSigLIP` (~4.6 GB).
- SeedVR2 CLI is staged at `/workspace/seedvr2_eval/ComfyUI-SeedVR2_VideoUpscaler` (~74 MB).
- SeedVR2 fp8 mixed DiT + VAE are staged at `/workspace/models/seedvr2` (~8.4 GB).
- SeedVR2 lightweight warmup passes with backend `cuda`.

## Resident VRAM measurement

Measured on `sporting_violet_lemming` with RTX PRO 6000 Blackwell 96 GB, torch `2.11.0+cu130`,
vLLM `0.22.1`, Qwen bf16, MiniCPM bf16 weights + fp8 KV cache, and SeedVR2 7B fp8 mixed:

| stage | used VRAM | free VRAM |
|---|---:|---:|
| Initial context | 560 MiB | 96,689 MiB |
| Qwen Image Edit base bf16 | 55,606 MiB | 41,643 MiB |
| MiniCPM vLLM bf16 + fp8 KV | 79,467 MiB | 17,782 MiB |
| Fashion detector | 79,631 MiB | 17,618 MiB |
| Marqo fashionSigLIP | 80,011 MiB | 17,238 MiB |
| SeedVR2 tiny run | 88,421 MiB | 8,828 MiB |

This measurement does not include wardrobe LoRAs because `/workspace/loras/wardrobe` is still empty.
It does include MiniCPM's `MINICPM_GPU_MEMORY_UTILIZATION = 0.27` code cap. That cap should not be
removed for the all-resident pod, because otherwise vLLM can reserve more of the remaining VRAM.
- Stage wardrobe LoRAs into `/workspace/loras/wardrobe`.
- Fill `.env` with Azure, JWT, Glamify backend URL, and selected resident runtimes.
- Start the service with the RunPod launch script:

  ```bash
  cd /workspace/glamify-image-ai
  cp .env.runpod.example .env
  # Fill secrets and LoRA paths in .env before launching.
  scripts/runpod_start.sh
  ```

  The script sets the Blackwell/vLLM flags and runs uvicorn with one worker on port `8000`.

## Current blockers before full service warmup

- No `.env` exists on the pod yet.
- Wardrobe LoRAs are not staged yet:
  - `/workspace/loras/wardrobe/top_23000.safetensors`
  - `/workspace/loras/wardrobe/bottom_30000.safetensors`
  - `/workspace/loras/wardrobe/dress_27000.safetensors`
- Try-on/AI-Toolkit assets are not staged yet. Keep `tryon` out of `RESIDENT_RUNTIMES` until those
  paths exist.

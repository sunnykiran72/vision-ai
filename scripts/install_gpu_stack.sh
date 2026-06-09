#!/usr/bin/env bash
# Install the GPU runtime stack for the bundled vision-ai service (Qwen diffusers + MiniCPM vLLM +
# SeedVR2 + detector/Marqo). These heavy deps are intentionally NOT in pyproject.toml; the app
# imports them lazily so local dev stays light. Run this on the GPU pod (Python 3.12).
#
# Validated reference versions (RunPod, CUDA 12.8, Blackwell/Hopper):
#   python 3.12 · torch 2.8.0+cu128 · vllm 0.22.1 · transformers 4.57.x (<5) · diffusers 0.38.0
#
# IMPORTANT compatibility gate:
#   MiniCPM-V-4.5 via vLLM REQUIRES transformers < 5 (v5 removed tokenizer.im_start_id and breaks
#   MiniCPM-V). The Qwen diffusers reference happened to run on transformers 5.10.2, so bundling
#   both in ONE environment hinges on Qwen Image Edit also loading under transformers 4.57.x.
#   This is the first thing to validate (see scripts/validate_gpu_stack.py).
set -euo pipefail

PY="${PYTHON:-python3.12}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

echo ">>> Using interpreter: $($PY --version)"
$PY -m pip install --upgrade pip wheel

# 1) torch first, matching the CUDA build vLLM expects.
$PY -m pip install "torch==2.8.0" --index-url "$TORCH_INDEX"

# 2) vLLM (pulls its own pinned deps; install after torch so the CUDA build is kept).
$PY -m pip install "vllm==0.22.1"

# 3) transformers pinned < 5 for MiniCPM-V. Re-pin in case vLLM moved it.
$PY -m pip install "transformers>=4.57,<5"

# 4) diffusers stack for the Qwen extraction engine.
$PY -m pip install "diffusers==0.38.0" "accelerate>=1.0" "safetensors>=0.4"

# 5) open_clip for the Marqo fashionSigLIP classifier.
$PY -m pip install "open_clip_torch>=2.24"

# 6) OpenCV for user-image blur/sharpness scoring.
$PY -m pip install "opencv-python-headless>=4.10,<5"

# 7) the service's own (light) deps.
$PY -m pip install \
  "fastapi>=0.115,<1.0" "uvicorn[standard]>=0.30,<1.0" "python-multipart>=0.0.9,<1.0" \
  "pydantic>=2.8,<3.0" "pydantic-settings>=2.4,<3.0" "PyJWT>=2.9,<3.0" \
  "httpx>=0.27,<1.0" "pillow>=10.4,<12.0" "structlog>=24.4,<26.0" \
  "azure-storage-blob>=12.26,<13.0"

echo ">>> Done. Now run: $PY scripts/validate_gpu_stack.py"

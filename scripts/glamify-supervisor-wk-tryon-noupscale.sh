#!/usr/bin/env bash
# POD-LOCAL launcher for WK (ra4nfsl1lrajm5, RTX PRO 6000 96GB) repurposed as a TRYON / NO-UPSCALE test pod.
#
# Goal: prove + evaluate try-on with SeedVR2 completely OUT of the picture.
#   - RESIDENT_RUNTIMES="wardrobe,tryon"  -> loads COMPILED fp8 Qwen (wardrobe+tryon LoRAs) + MiniCPM.
#   - upscale runtime is NOT resident and there are NO UPSCALE_* vars -> SeedVR2 is never loaded.
#   - TRYON_UPSCALE_AFTER_QWEN="false"   -> server default is no-upscale; tryon uploads the raw Qwen
#     JPEG at the INPUT dimensions (Qwen already outputs JPEG, so no extra work).
#
# /tmp is pod-local; this does NOT touch muji or the shared .env. NOTE: while WK runs in this mode its
# /v1/upscale (4096) API is OFFLINE — the backend download-4k path must not target WK during the test.
set -uo pipefail
cd /workspace/glamify-image-ai
while true; do
  echo "===== glamify-wk-tryon start $(date -Is) ====="
  env \
    RESIDENT_RUNTIMES="wardrobe,tryon" \
    QWEN_IMAGE_EDIT_DTYPE="bfloat16" \
    QWEN_FP8="true" \
    QWEN_COMPILE="true" \
    MINICPM_ENFORCE_EAGER="true" \
    MINICPM_KV_CACHE_DTYPE="auto" \
    TRYON_UPSCALE_AFTER_QWEN="false" \
    STARTUP_PARALLEL_WARMUP="0" \
    TORCHINDUCTOR_CACHE_DIR="/workspace/.torchinductor_cache" \
    TRITON_CACHE_DIR="/workspace/.triton_cache" \
    TORCHINDUCTOR_FX_GRAPH_CACHE="1" \
    HF_HOME="/workspace/hf_cache" \
    VLLM_USE_FLASHINFER_SAMPLER="0" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    TOKENIZERS_PARALLELISM="false" \
    bash scripts/runpod_start.sh
  code=$?
  echo "===== glamify-wk-tryon exited code=$code $(date -Is); restart 5s ====="
  sleep 5
done

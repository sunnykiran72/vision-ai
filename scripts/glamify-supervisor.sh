#!/usr/bin/env bash
# Persistent supervisor for glamify-image-ai on a RunPod GPU pod.
#
# Lives in the repo (on the shared /workspace volume) so it survives pod recreation — unlike the old
# ephemeral /tmp/glamify-supervisor.sh. To AUTO-START on every (re)created pod, set the RunPod
# template/pod "onstart"/Docker command to:
#     bash /workspace/glamify-image-ai/scripts/glamify-supervisor.sh
# It relaunches the API if it ever exits, so a crash self-heals.
#
# NOTE: env here mirrors the proven production config — do not add unnecessary settings. The SeedVR2
# block (UPSCALE_COMPILE/WARMUP/WARMUP_EDGES) is what makes the 2730 upscale fast + prewarmed.
set -uo pipefail
cd /workspace/glamify-image-ai
while true; do
  echo "===== glamify-image-ai start $(date -Is) ====="
  env \
    RESIDENT_RUNTIMES="wardrobe,tryon,upscale" \
    MINICPM_ENFORCE_EAGER="true" \
    MINICPM_KV_CACHE_DTYPE="auto" \
    QWEN_IMAGE_EDIT_DTYPE="bfloat16" \
    QWEN_FP8="true" \
    QWEN_COMPILE="true" \
    UPSCALE_COMPILE="1" \
    UPSCALE_WARMUP="1" \
    UPSCALE_WARMUP_EDGES="2730" \
    STARTUP_PARALLEL_WARMUP="1" \
    TORCHINDUCTOR_CACHE_DIR="/workspace/.torchinductor_cache" \
    TRITON_CACHE_DIR="/workspace/.triton_cache" \
    TORCHINDUCTOR_FX_GRAPH_CACHE="1" \
    HF_HOME="/workspace/hf_cache" \
    VLLM_USE_FLASHINFER_SAMPLER="0" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    TOKENIZERS_PARALLELISM="false" \
    bash scripts/runpod_start.sh
  code=$?
  echo "===== glamify-image-ai exited code=$code $(date -Is); restarting in 5s ====="
  sleep 5
done

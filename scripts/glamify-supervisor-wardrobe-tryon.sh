#!/usr/bin/env bash
# Persistent supervisor for a WARDROBE + TRYON + USER_VALIDATION pod (NO upscale).
#
# Use this on a pod that should host wardrobe extraction, try-on (top/bottom/dress), and user
# image validation — but NOT the SeedVR2 upscaler. The Qwen-Image-Edit weights are shared by
# wardrobe and try-on; user_validation rides the same GPU queue via its person detector and needs
# no resident runtime (its routes are always registered; the detector loads on first call).
#
# Differences vs the all-in-one production supervisor (scripts/glamify-supervisor.sh):
#   - RESIDENT_RUNTIMES drops "upscale" -> SeedVR2 is never loaded (frees ~10-12 GB VRAM).
#   - TRYON_ENABLED_SPECIALISTS drops "multi" -> only top/bottom/dress LoRAs are staged.
#   - TRYON_UPSCALE_AFTER_QWEN=false -> try-on returns the Qwen output directly. REQUIRED: with no
#     resident upscaler, leaving inline upscale on would cold-load (or fail) SeedVR2 per request.
#   - The UPSCALE_COMPILE/WARMUP/WARMUP_EDGES block is gone (nothing to compile/prewarm).
#   - /ready gates only on wardrobe+tryon here (see app/routes/health.py), so the pod becomes
#     routable once those two are warm.
#
# Qwen is still fp8 + torch.compile (the proven fast config: ~6.5s @ 15 steps, -33% VRAM).
#
# To AUTO-START on every (re)created pod, set the RunPod template/pod "onstart"/Docker command to:
#     bash /workspace/glamify-image-ai/scripts/glamify-supervisor-wardrobe-tryon.sh
# It relaunches the API if it ever exits, so a crash self-heals.
set -uo pipefail
cd /workspace/glamify-image-ai
while true; do
  echo "===== glamify-image-ai (wardrobe+tryon) start $(date -Is) ====="
  env \
    RESIDENT_RUNTIMES="wardrobe,tryon" \
    TRYON_ENABLED_SPECIALISTS="top,bottom,dress" \
    TRYON_UPSCALE_AFTER_QWEN="false" \
    MINICPM_ENFORCE_EAGER="true" \
    MINICPM_KV_CACHE_DTYPE="auto" \
    QWEN_IMAGE_EDIT_DTYPE="bfloat16" \
    QWEN_FP8="true" \
    QWEN_COMPILE="true" \
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
  echo "===== glamify-image-ai (wardrobe+tryon) exited code=$code $(date -Is); restarting in 5s ====="
  sleep 5
done

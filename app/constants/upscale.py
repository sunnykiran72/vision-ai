from __future__ import annotations

# SeedVR2 DiT variants accepted by the pinned ComfyUI-SeedVR2 CLI build on the pod
# (authoritative source: src.utils.model_registry.get_available_dit_models()).
#
# NOTE: this CLI build does NOT accept the "pure" 7B FP8 files
# (seedvr2_ema_7b_fp8_e4m3fn / _sharp_fp8_e4m3fn). The only supported 7B FP8
# builds are the "_mixed_block35_fp16" ones. Keep this list in lockstep with the
# registry, otherwise the CLI argparse rejects the name with SystemExit.
#
# The `present` flag is resolved at runtime by scanning the model directory.
KNOWN_SEEDVR2_VARIANTS: tuple[dict[str, str], ...] = (
    {
        "filename": "seedvr2_ema_3b-Q4_K_M.gguf",
        "label": "3B - GGUF Q4_K_M (smallest/fastest)",
        "model": "3B",
        "precision": "gguf-q4",
        "approx_size": "~2.0 GB",
    },
    {
        "filename": "seedvr2_ema_3b-Q8_0.gguf",
        "label": "3B - GGUF Q8_0",
        "model": "3B",
        "precision": "gguf-q8",
        "approx_size": "~3.6 GB",
    },
    {
        "filename": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "label": "3B - FP8 (e4m3fn, current prod default)",
        "model": "3B",
        "precision": "fp8",
        "approx_size": "3.4 GB",
    },
    {
        "filename": "seedvr2_ema_3b_fp16.safetensors",
        "label": "3B - FP16",
        "model": "3B",
        "precision": "fp16",
        "approx_size": "6.8 GB",
    },
    {
        "filename": "seedvr2_ema_7b-Q4_K_M.gguf",
        "label": "7B - GGUF Q4_K_M",
        "model": "7B",
        "precision": "gguf-q4",
        "approx_size": "~4.5 GB",
    },
    {
        "filename": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
        "label": "7B - FP8 mixed block35",
        "model": "7B",
        "precision": "fp8-mixed",
        "approx_size": "8.5 GB",
    },
    {
        "filename": "seedvr2_ema_7b_fp16.safetensors",
        "label": "7B - FP16 (quality ceiling)",
        "model": "7B",
        "precision": "fp16",
        "approx_size": "16.5 GB",
    },
    {
        "filename": "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
        "label": "7B sharp - GGUF Q4_K_M",
        "model": "7B-sharp",
        "precision": "gguf-q4",
        "approx_size": "~4.5 GB",
    },
    {
        "filename": "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
        "label": "7B sharp - FP8 mixed block35",
        "model": "7B-sharp",
        "precision": "fp8-mixed",
        "approx_size": "8.5 GB",
    },
    {
        "filename": "seedvr2_ema_7b_sharp_fp16.safetensors",
        "label": "7B sharp - FP16",
        "model": "7B-sharp",
        "precision": "fp16",
        "approx_size": "16.5 GB",
    },
)

KNOWN_SEEDVR2_VARIANT_FILENAMES: frozenset[str] = frozenset(
    variant["filename"] for variant in KNOWN_SEEDVR2_VARIANTS
)

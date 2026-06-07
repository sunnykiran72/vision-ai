"""Probe resident VRAM use for the RunPod production stack.

This intentionally loads the heavy runtimes in one process and reports `nvidia-smi`
after each stage. It does not require service `.env` or wardrobe LoRAs.

Run on the pod:
    /workspace/.venvs/glamify-image-ai/bin/python scripts/probe_resident_vram.py
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from PIL import Image


def report(label: str) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    print(f"VRAM after {label}: {output} MiB used,free,total,util%", flush=True)


def main() -> None:
    print(
        "Starting resident VRAM probe. LoRAs are not loaded because /workspace/loras is empty.",
        flush=True,
    )
    report("initial")

    import torch
    from diffusers import QwenImageEditPlusPipeline

    qwen = QwenImageEditPlusPipeline.from_pretrained(
        "/workspace/models/qwen-image-edit-2511",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    report("Qwen Image Edit base bf16")

    from app.clients.minicpm_vllm import MiniCPMVllmClient
    from app.config import Settings

    minicpm = MiniCPMVllmClient(
        Settings(
            MINICPM_MODEL_PATH="/workspace/models/minicpm-v-4_5",
            MINICPM_DTYPE="bfloat16",
            MINICPM_KV_CACHE_DTYPE="fp8",
            MINICPM_CALCULATE_KV_SCALES=True,
            MINICPM_ATTENTION_BACKEND="TRITON_ATTN",
        ),
    )
    minicpm.warmup()
    report("MiniCPM vLLM bf16 + fp8 KV")

    from app.clients.fashion_detection import get_fashion_detection_client
    from app.clients.marqo_fashion import get_marqo_fashion_client

    get_fashion_detection_client().ensure_ready()
    report("fashion detector")
    get_marqo_fashion_client().ensure_ready()
    report("Marqo fashionSigLIP")

    from app.clients.seedvr2 import SeedVR2Client

    seed = SeedVR2Client(
        Settings(
            UPSCALE_MODEL_PATH="/workspace/models/seedvr2",
            UPSCALE_MODEL_VARIANT="seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
            UPSCALE_CLI_PATH="/workspace/seedvr2_eval/ComfyUI-SeedVR2_VideoUpscaler/inference_cli.py",
        ),
    )
    probe_dir = Path("/workspace/tmp/glamify/vram-probe")
    probe_dir.mkdir(parents=True, exist_ok=True)
    input_path = probe_dir / "input.jpg"
    output_path = probe_dir / "output.png"
    log_path = probe_dir / "seedvr2.log"
    Image.new("RGB", (512, 512), (128, 128, 128)).save(input_path, quality=95)

    started = time.perf_counter()
    try:
        result = seed.run(
            input_path=input_path,
            output_path=output_path,
            log_path=log_path,
            target_long_edge=512,
        )
        print(
            "SeedVR2 probe result: "
            f"output={result.output_width}x{result.output_height} "
            f"wall={result.wall_seconds}s backend={result.runner_backend}",
            flush=True,
        )
    except Exception as exc:
        print(f"SeedVR2 probe failed: {type(exc).__name__}: {exc}", flush=True)
        if log_path.exists():
            print(log_path.read_text(errors="replace")[-4000:], flush=True)
        raise

    print(f"SeedVR2 probe elapsed: {time.perf_counter() - started:.1f}s", flush=True)
    report("SeedVR2 tiny run")
    print("Probe complete.", flush=True)

    # Keep references alive until the final report is printed.
    _ = (qwen, minicpm, seed)


if __name__ == "__main__":
    main()

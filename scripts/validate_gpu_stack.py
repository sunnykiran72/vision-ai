"""Validate the bundled GPU runtime stack before starting the service.

Checks that every heavy dependency imports in ONE environment and that the known compatibility
gate holds (transformers < 5 for MiniCPM-V, plus the Qwen diffusers pipeline class is importable).
This does not load model weights; full model validation happens at service warmup.

Run on the GPU pod:  python3.12 scripts/validate_gpu_stack.py
"""

from __future__ import annotations

import sys


def _check(label: str, fn) -> bool:
    try:
        detail = fn()
        print(f"[ok]   {label}: {detail}")
        return True
    except Exception as exc:  # noqa: BLE001 - report every failure
        print(f"[FAIL] {label}: {exc}")
        return False


def main() -> int:
    ok = True

    def torch_info() -> str:
        import torch

        return f"{torch.__version__} cuda={torch.cuda.is_available()}"

    def transformers_info() -> str:
        import transformers

        major = int(transformers.__version__.split(".")[0])
        if major >= 5:
            raise RuntimeError(
                f"transformers {transformers.__version__} >= 5 breaks MiniCPM-V; pin <5",
            )
        return transformers.__version__

    def diffusers_info() -> str:
        import diffusers
        from diffusers import QwenImageEditPlusPipeline  # noqa: F401

        return f"{diffusers.__version__} (QwenImageEditPlusPipeline importable)"

    def vllm_info() -> str:
        import vllm
        from vllm import LLM, SamplingParams  # noqa: F401

        return getattr(vllm, "__version__", "unknown")

    def open_clip_info() -> str:
        import open_clip

        return getattr(open_clip, "__version__", "unknown")

    def safetensors_info() -> str:
        from safetensors.torch import load_file  # noqa: F401

        return "load_file importable"

    ok &= _check("torch", torch_info)
    ok &= _check("transformers (<5)", transformers_info)
    ok &= _check("diffusers + Qwen pipeline", diffusers_info)
    ok &= _check("vllm", vllm_info)
    ok &= _check("open_clip (Marqo)", open_clip_info)
    ok &= _check("safetensors", safetensors_info)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

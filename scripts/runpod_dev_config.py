from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

WARDROBE_RUNS = {
    "top": (
        "qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1",
    ),
    "bottom": (
        "qwen_garment_extract_bottom119_glambtmext_rank16_b1_extend_22k_to_30k_lr815e5_live",
        "qwen_garment_extract_bottom119_glambtmext_rank16_b1_continue_12k_to_22k_lr815e5_v1",
    ),
    "dress": (
        "qwen_garment_extract_dress125_glamdressext_rank16_b1_continue_15k_to_30k_lr125e5_v1",
    ),
}

TRYON_RUNS = {
    "top": "qwen_lora_glamtoptryon_v2_res1024_batch1_rank16_fresh",
    "bottom": "qwen_lora_glambottomtryon_v2_res1024_batch1_rank16_fresh",
}

DEFAULT_SEARCH_ROOTS = ("/mnt", "/workspace", "/runpod-volume")
DEFAULT_MODEL_CANDIDATES = (
    "/mnt/models/qwen-image-edit-2511",
    "/workspace/models/qwen-image-edit-2511",
)
DEFAULT_AITK_CANDIDATES = (
    "/workspace/ai-toolkit",
    "/app/ai-toolkit",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve RunPod Glamify dev checkpoint paths and write an env file.",
    )
    parser.add_argument("--write", default=".env.runpod-dev", help="Output env file path.")
    parser.add_argument(
        "--search-root",
        action="append",
        dest="search_roots",
        help="Root to search. Can be passed multiple times.",
    )
    parser.add_argument(
        "--qwen-model-path",
        default=os.getenv("QWEN_IMAGE_EDIT_MODEL_PATH", ""),
        help="Qwen Image Edit model path.",
    )
    parser.add_argument(
        "--ai-toolkit-root",
        default=os.getenv("AI_TOOLKIT_ROOT", ""),
        help="AI-Toolkit checkout path.",
    )
    parser.add_argument(
        "--glamify-api-base-url",
        default=os.getenv("GLAMIFY_API_BASE_URL", ""),
        help="Glamify backend base URL for the real API server.",
    )
    args = parser.parse_args()

    search_roots = tuple(args.search_roots or DEFAULT_SEARCH_ROOTS)
    env_values = {
        "APP_ENV": "runpod-dev",
        "QWEN_IMAGE_EDIT_MODEL_PATH": args.qwen_model_path
        or _first_existing(DEFAULT_MODEL_CANDIDATES),
        "AI_TOOLKIT_ROOT": args.ai_toolkit_root or _first_existing(DEFAULT_AITK_CANDIDATES),
        "WARDROBE_QUEUE_MAX_SIZE": "4",
        "WARDROBE_WORK_ROOT": "/tmp/glamify/wardrobe",
        "WARDROBE_STORAGE_PREFIX": "wardrobe_output/wardrobe",
        "TRYON_USE_SPECIALISTS": "true",
        "TRYON_ENABLED_SPECIALISTS": "top,bottom,dress,multi",
        "TRYON_LORA_RANK": "16",
        "TRYON_LORA_ALPHA": "16",
        "TRYON_LORA_SCALE": "1.0",
        "TRYON_DEFAULT_SEED": "43",
        "TRYON_DEFAULT_STEPS": "25",
        "TRYON_DEFAULT_GUIDANCE_SCALE": "1.0",
        "TRYON_GUIDANCE_RESCALE": "0.0",
        "TRYON_DO_CFG_NORM": "false",
        "TRYON_SAMPLER": "flowmatch",
        "TRYON_QUEUE_MAX_SIZE": "4",
        "TRYON_QUEUE_WAIT_TIMEOUT_SECONDS": "30",
        "TRYON_WORK_ROOT": "/tmp/glamify/tryon",
        "TRYON_STORAGE_PREFIX": "wardrobe_output/tryon",
        "GLAMIFY_API_BASE_URL": args.glamify_api_base_url,
    }

    for key, run_names in WARDROBE_RUNS.items():
        env_values[f"WARDROBE_LORA_{key.upper()}_PATH"] = _latest_checkpoint_for_runs(
            run_names,
            search_roots,
        )

    tryon_top = _latest_checkpoint(TRYON_RUNS["top"], search_roots)
    tryon_bottom = _latest_checkpoint(TRYON_RUNS["bottom"], search_roots)
    env_values["TRYON_LORA_TOP_PATH"] = tryon_top
    env_values["TRYON_LORA_BOTTOM_PATH"] = tryon_bottom
    env_values["TRYON_LORA_DRESS_PATH"] = tryon_top
    env_values["TRYON_LORA_MULTI_PATH"] = tryon_top

    output_path = Path(args.write)
    output_path.write_text(_format_env(env_values), encoding="utf-8")
    print(f"Wrote {output_path}")
    for key in sorted(env_values):
        value = env_values[key]
        if key.endswith("_PATH") or key in {"AI_TOOLKIT_ROOT", "QWEN_IMAGE_EDIT_MODEL_PATH"}:
            print(f"{key}={value}")
    print(
        "Note: TRYON_LORA_DRESS_PATH and TRYON_LORA_MULTI_PATH intentionally reuse "
        "TRYON_LORA_TOP_PATH for dev warmup/OOM testing only.",
    )


def _first_existing(candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return candidates[0]


def _latest_checkpoint(run_name: str, search_roots: tuple[str, ...]) -> str:
    matches: list[Path] = []
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        command = [
            "find",
            str(root_path),
            "-path",
            f"*{run_name}*",
            "-type",
            "f",
            "-name",
            "*.safetensors",
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        for line in result.stdout.splitlines():
            path = Path(line.strip())
            if path.name.startswith("optimizer"):
                continue
            matches.append(path)
    if not matches:
        raise SystemExit(f"No safetensors checkpoint found for run: {run_name}")
    return str(max(matches, key=_checkpoint_score))


def _latest_checkpoint_for_runs(run_names: tuple[str, ...], search_roots: tuple[str, ...]) -> str:
    missing: list[str] = []
    for run_name in run_names:
        try:
            return _latest_checkpoint(run_name, search_roots)
        except SystemExit:
            missing.append(run_name)
    raise SystemExit(
        "No safetensors checkpoint found for any run: " + ", ".join(missing),
    )


def _checkpoint_score(path: Path) -> tuple[int, float]:
    numbers = [int(value) for value in re.findall(r"(?<![a-zA-Z])(\d{3,})(?![a-zA-Z])", str(path))]
    step = max(numbers) if numbers else 0
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return step, mtime


def _format_env(values: dict[str, str]) -> str:
    lines = ["# Generated by scripts/runpod_dev_config.py"]
    for key in sorted(values):
        lines.append(f"{key}={_quote_env(values[key])}")
    return "\n".join(lines) + "\n"


def _quote_env(value: str) -> str:
    text = str(value)
    if not text or re.search(r"\s|['\"`$]", text):
        return "'" + text.replace("'", "'\"'\"'") + "'"
    return text


if __name__ == "__main__":
    main()

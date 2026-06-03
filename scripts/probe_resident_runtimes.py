from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.runtime.tryon_runtime import warmup_tryon_runtime
from app.runtime.wardrobe_runtime import warmup_wardrobe_runtime

RuntimeWarmup = Callable[[], None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Warm selected resident runtimes and print GPU memory snapshots.",
    )
    parser.add_argument(
        "--runtimes",
        nargs="+",
        choices=("wardrobe", "tryon"),
        default=("wardrobe", "tryon"),
        help="Runtimes to warm in order.",
    )
    args = parser.parse_args()

    settings = get_settings()
    warmups: dict[str, RuntimeWarmup] = {
        "wardrobe": lambda: warmup_wardrobe_runtime(settings),
        "tryon": lambda: warmup_tryon_runtime(settings),
    }

    _print_gpu("start")
    for runtime_name in args.runtimes:
        started = time.perf_counter()
        print(f"\n== Warming {runtime_name} ==")
        warmups[runtime_name]()
        elapsed = time.perf_counter() - started
        print(f"== {runtime_name} warmed in {elapsed:.2f}s ==")
        _print_gpu(f"after {runtime_name}")
    print("\nResident runtime probe completed.")


def _print_gpu(label: str) -> None:
    print(f"\n-- GPU memory: {label} --")
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr.strip() or "nvidia-smi unavailable")
        return
    print(result.stdout.strip())


if __name__ == "__main__":
    main()

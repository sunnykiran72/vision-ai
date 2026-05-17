from __future__ import annotations

import importlib.util
import io
import os
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from PIL import Image

from app.config import Settings


@dataclass(frozen=True)
class SeedVR2RunResult:
    output_path: Path
    output_width: int
    output_height: int
    wall_seconds: float
    target_long_edge: int
    derived_short_edge: int
    model_variant: str
    log_path: Path
    runner_backend: str


@dataclass(frozen=True)
class SeedVR2RuntimeStatus:
    loaded: bool
    backend: str | None


class SeedVR2Client:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._cli_module: ModuleType | None = None
        self._runner_cache: dict[str, object] = {}
        self._device_list: list[str] | None = None
        self._backend: str | None = None
        self._init_lock = threading.Lock()
        self._run_lock = threading.Lock()

    def run(
        self,
        *,
        input_path: Path,
        output_path: Path,
        log_path: Path,
        target_long_edge: int,
    ) -> SeedVR2RunResult:
        cli_path = Path(self._settings.upscale_cli_path)
        model_dir = Path(self._settings.upscale_model_path)
        model_variant = self._settings.upscale_model_variant.strip()

        if not cli_path.exists():
            raise FileNotFoundError(f"SeedVR2 CLI not found: {cli_path}")
        if not model_dir.exists():
            raise FileNotFoundError(f"SeedVR2 model directory not found: {model_dir}")
        if not model_variant:
            raise ValueError("UPSCALE_MODEL_VARIANT is not configured.")

        with Image.open(input_path) as image:
            input_width, input_height = image.size
        input_long_edge = max(int(input_width), int(input_height))
        input_short_edge = min(int(input_width), int(input_height))
        if input_long_edge <= 0 or input_short_edge <= 0:
            raise ValueError("Invalid input image dimensions.")

        scale = float(target_long_edge) / float(input_long_edge)
        derived_short_edge = max(256, int(round(float(input_short_edge) * scale)))

        self._ensure_loaded(cli_path)
        assert self._cli_module is not None
        assert self._device_list is not None
        assert self._backend is not None

        request_args = self._build_args(
            cli_path=cli_path,
            model_dir=model_dir,
            model_variant=model_variant,
            input_path=input_path,
            output_path=output_path,
            derived_short_edge=derived_short_edge,
            target_long_edge=target_long_edge,
        )

        capture = io.StringIO()
        started_at = time.perf_counter()
        with self._run_lock:
            with redirect_stdout(capture), redirect_stderr(capture):
                frames = self._cli_module.process_single_file(
                    str(input_path),
                    request_args,
                    device_list=self._device_list,
                    output_path=str(output_path),
                    format_auto_detected=False,
                    runner_cache=self._runner_cache,
                )
        elapsed = time.perf_counter() - started_at
        combined_log = capture.getvalue()
        log_path.write_text((combined_log or "")[-250000:], encoding="utf-8")

        if int(frames) <= 0:
            raise RuntimeError("SeedVR2 did not produce any output frames.")
        if not output_path.exists():
            raise RuntimeError("SeedVR2 finished without producing an output file.")

        with Image.open(output_path) as output_image:
            output_width, output_height = output_image.size

        return SeedVR2RunResult(
            output_path=output_path,
            output_width=int(output_width),
            output_height=int(output_height),
            wall_seconds=float(round(elapsed, 3)),
            target_long_edge=int(target_long_edge),
            derived_short_edge=int(derived_short_edge),
            model_variant=model_variant,
            log_path=log_path,
            runner_backend=self._backend,
        )

    def warmup(self) -> None:
        cli_path = Path(self._settings.upscale_cli_path)
        model_dir = Path(self._settings.upscale_model_path)
        model_variant = self._settings.upscale_model_variant.strip()

        if not cli_path.exists():
            raise FileNotFoundError(f"SeedVR2 CLI not found: {cli_path}")
        if not model_dir.exists():
            raise FileNotFoundError(f"SeedVR2 model directory not found: {model_dir}")
        if not model_variant:
            raise ValueError("UPSCALE_MODEL_VARIANT is not configured.")

        self._ensure_loaded(cli_path)

    def status(self) -> SeedVR2RuntimeStatus:
        return SeedVR2RuntimeStatus(
            loaded=(
                self._cli_module is not None
                and self._device_list is not None
                and self._backend is not None
            ),
            backend=self._backend,
        )

    def _ensure_loaded(self, cli_path: Path) -> None:
        if (
            self._cli_module is not None
            and self._device_list is not None
            and self._backend is not None
        ):
            return

        with self._init_lock:
            if (
                self._cli_module is not None
                and self._device_list is not None
                and self._backend is not None
            ):
                return

            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            startup_capture = io.StringIO()
            with redirect_stdout(startup_capture), redirect_stderr(startup_capture):
                module = self._load_module(cli_path)
            backend = str(module.get_gpu_backend())
            device_list = ["0"] if backend == "cuda" else ["cpu"]
            self._cli_module = module
            self._backend = backend
            self._device_list = device_list

    def _load_module(self, cli_path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(
            "seedvr2_inference_cli_runtime",
            str(cli_path),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import SeedVR2 CLI from: {cli_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "debug"):
            module.debug.enabled = False
        return module

    def _build_args(
        self,
        *,
        cli_path: Path,
        model_dir: Path,
        model_variant: str,
        input_path: Path,
        output_path: Path,
        derived_short_edge: int,
        target_long_edge: int,
    ) -> object:
        assert self._cli_module is not None
        argv = [
            str(cli_path),
            str(input_path),
            "--output",
            str(output_path),
            "--output_format",
            "jpg",
            "--dit_model",
            model_variant,
            "--model_dir",
            str(model_dir),
            "--resolution",
            str(int(derived_short_edge)),
            "--max_resolution",
            str(int(target_long_edge)),
            "--batch_size",
            "1",
            "--cache_dit",
            "--cache_vae",
            "--dit_offload_device",
            "none",
            "--vae_offload_device",
            "none",
            "--tensor_offload_device",
            "cpu",
        ]
        original_argv = list(sys.argv)
        try:
            sys.argv = argv
            return self._cli_module.parse_arguments()
        finally:
            sys.argv = original_argv


@lru_cache(maxsize=1)
def get_seedvr2_client(
    upscale_model_path: str,
    upscale_model_variant: str,
    upscale_cli_path: str,
) -> SeedVR2Client:
    settings = Settings(
        UPSCALE_MODEL_PATH=upscale_model_path,
        UPSCALE_MODEL_VARIANT=upscale_model_variant,
        UPSCALE_CLI_PATH=upscale_cli_path,
    )
    return SeedVR2Client(settings)

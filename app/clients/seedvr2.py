from __future__ import annotations

import importlib.util
import io
import logging
import os
import sys
import tempfile
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from PIL import Image

from app.config import Settings

logger = logging.getLogger("glamify-ai")

# Output long edges to pre-warm torch.compile at startup (prod /v1/upscale presets: 2k, 4k).
# Override with UPSCALE_WARMUP_EDGES="2048,4096" or disable pre-warm with UPSCALE_WARMUP=0.
_DEFAULT_WARMUP_EDGES: tuple[int, ...] = (2048, 4096)
# Persistent compile caches on the shared volume so the (slow) one-time inductor codegen is
# reused across restarts AND across pods sharing /workspace (same GPU arch + venv). Without
# this, inductor writes to ephemeral /tmp and recompiles from scratch on every restart.
_PERSISTENT_COMPILE_CACHE_DIRS: dict[str, str] = {
    "TORCHINDUCTOR_CACHE_DIR": "/workspace/.torchinductor_cache",
    "TRITON_CACHE_DIR": "/workspace/.triton_cache",
    "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
}
# torch.compiler "mega-cache": a single portable blob of ALL compile artifacts (inductor
# codegen + triton + autotune). The automatic inductor FX-graph cache MISSES across restarts
# for SeedVR2's custom fp8 DiT/VAE (its key includes per-process state), so every restart
# recompiles (~340s). We snapshot the blob to the shared volume after prewarm and reload it on
# startup, turning that ~340s recompile into a fast load. Override path with UPSCALE_MEGACACHE.
_MEGACACHE_PATH = "/workspace/.seedvr2_compile_megacache.bin"


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
        # Run the (slow, per-shape torch.compile) prewarm in a background daemon thread so it
        # NEVER blocks API startup/health. Wardrobe + try-on become available immediately; the
        # first upscale request simply waits on the shared _run_lock until the compile lands.
        if os.environ.get("UPSCALE_WARMUP", "1") != "0":
            threading.Thread(
                target=self._prewarm,
                name="seedvr2-prewarm",
                daemon=True,
            ).start()

    def _prewarm(self) -> None:
        """Run a synthetic upscale at each prod target resolution so torch.compile is paid at
        startup (and cached to the volume), not on the first user request. Best-effort: a
        failure here is logged and never blocks startup. Disable with UPSCALE_WARMUP=0."""
        if os.environ.get("UPSCALE_WARMUP", "1") == "0":
            return
        raw = os.environ.get("UPSCALE_WARMUP_EDGES", "")
        try:
            edges = tuple(int(x) for x in raw.split(",") if x.strip()) or _DEFAULT_WARMUP_EDGES
        except ValueError:
            edges = _DEFAULT_WARMUP_EDGES
        # 2:3 portrait synthetic input — the dominant wardrobe/try-on output aspect.
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "warm_in.png"
            Image.new("RGB", (832, 1248), (127, 127, 127)).save(src, format="PNG")
            for edge in edges:
                started = time.perf_counter()
                try:
                    self.run(
                        input_path=src,
                        output_path=Path(tmp) / f"warm_out_{edge}.png",
                        log_path=Path(tmp) / f"warm_{edge}.log",
                        target_long_edge=int(edge),
                    )
                    logger.info(
                        "SeedVR2 prewarm at long_edge=%d done in %.1fs",
                        edge,
                        time.perf_counter() - started,
                    )
                except Exception as exc:  # noqa: BLE001 - prewarm must never block startup
                    logger.warning("SeedVR2 prewarm at long_edge=%d failed: %s", edge, exc)
        # Persist the freshly-compiled artifacts so the next restart reloads instead of
        # recompiling (~340s -> fast load).
        self._save_compile_cache()

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
        # torch.compile enablement fixes for SeedVR2 on this stack:
        # (1) numz #502: a torch.compiled model is wrapped in OptimizedModule, which has no
        #     __bool__, so `if model:` truthiness checks fall back to __len__ and raise
        #     ("CompatibleDiT does not support len()"). Make compiled modules truthy.
        # (2) cudnn.benchmark selects the fastest Conv3d algorithms for the fixed-shape VAE.
        # (3) Persist the inductor/triton compile cache to the shared volume. Must be set before
        #     the first torch.compile. setdefault so an explicit launch-env value still wins.
        for key, value in _PERSISTENT_COMPILE_CACHE_DIRS.items():
            os.environ.setdefault(key, value)
            if key.endswith("_DIR"):
                try:
                    Path(value).mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass
        try:
            import torch as _torch
            from torch._dynamo.eval_frame import OptimizedModule

            if not hasattr(OptimizedModule, "__bool__"):
                OptimizedModule.__bool__ = lambda self: True  # type: ignore[attr-defined]
            _torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        # Reload persisted compile artifacts BEFORE the first torch.compile (prewarm/first
        # request), so the per-shape codegen is a fast load instead of a ~340s recompile.
        self._load_compile_cache()
        return module

    def _load_compile_cache(self) -> None:
        """Best-effort reload of the torch.compiler mega-cache snapshot. Must run before the
        first compile. Failure is logged and never blocks startup."""
        if os.environ.get("UPSCALE_COMPILE", "1") == "0":
            return
        try:
            import torch

            path = Path(os.environ.get("UPSCALE_MEGACACHE", _MEGACACHE_PATH))
            if not path.exists():
                logger.info("SeedVR2 compile mega-cache absent (%s); will compile fresh", path)
                return
            data = path.read_bytes()
            torch.compiler.load_cache_artifacts(data)
            logger.info("SeedVR2 compile mega-cache loaded (%.1f MB)", len(data) / 1e6)
        except Exception as exc:  # noqa: BLE001 - cache reload must never block startup
            logger.warning("SeedVR2 compile mega-cache load failed: %s", exc)

    def _save_compile_cache(self) -> None:
        """Best-effort snapshot of all torch.compile artifacts to the shared volume so the next
        restart reloads them. Called after prewarm has compiled every target shape."""
        if os.environ.get("UPSCALE_COMPILE", "1") == "0":
            return
        try:
            import torch

            artifacts = torch.compiler.save_cache_artifacts()
            if not artifacts:
                logger.info("SeedVR2 compile mega-cache: nothing to save")
                return
            data, _info = artifacts
            path = Path(os.environ.get("UPSCALE_MEGACACHE", _MEGACACHE_PATH))
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
            logger.info("SeedVR2 compile mega-cache saved (%.1f MB)", len(data) / 1e6)
        except Exception as exc:  # noqa: BLE001 - cache save must never block startup
            logger.warning("SeedVR2 compile mega-cache save failed: %s", exc)

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
        # Keep DiT + VAE RESIDENT on the GPU between requests. The CLI has a trap: with
        # --cache_dit/--cache_vae, an offload device of "none" is silently rewritten to "cpu"
        # (see inference_cli.py::_parse_offload_device, cache_enabled branch). That makes the
        # VAE bounce GPU->CPU->GPU and re-convert fp16->bf16 (~2.6s) on EVERY request. Pointing
        # the offload device at the GPU itself ("0") makes the cache a no-op that keeps weights
        # resident: ~5.3s -> ~3.9s per 2730 upscale (single-image). On CPU-only, fall back.
        # Single-image upscales are small enough to keep the latent/output tensors on-GPU too.
        offload_device = "0" if self._backend == "cuda" else "none"
        argv = [
            str(cli_path),
            str(input_path),
            "--output",
            str(output_path),
            "--output_format",
            "png",
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
            offload_device,
            "--vae_offload_device",
            offload_device,
            "--tensor_offload_device",
            offload_device,
        ]
        # Compile ONLY the prewarmed shapes (UPSCALE_WARMUP_EDGES, e.g. "2730" for try-on). Those
        # are fixed (2:3-only) so the one static graph is reused → ~2.4s. EVERY other target (e.g.
        # the standalone 4096 path) runs EAGER: 4096's ~20 GB compile workspace OOMs co-resident
        # with Qwen, but eager 4096 — which still gets the offload="0" resident fix above — fits
        # (~12.6 s). Set UPSCALE_COMPILE=0 to disable compile everywhere.
        compile_edges = {
            int(edge)
            for edge in os.environ.get("UPSCALE_WARMUP_EDGES", "2730").split(",")
            if edge.strip().isdigit()
        }
        if os.environ.get("UPSCALE_COMPILE", "1") != "0" and int(target_long_edge) in compile_edges:
            argv += ["--compile_dit", "--compile_vae"]
        original_argv = list(sys.argv)
        try:
            sys.argv = argv
            return self._cli_module.parse_arguments()
        finally:
            sys.argv = original_argv


@lru_cache(maxsize=4)
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

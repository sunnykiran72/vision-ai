# SeedVR2 2730 Upscale Optimization — Complete Handoff

> **Read this first.** This document is **self-contained** — it assumes you (a fresh AI assistant or
> engineer) have **zero prior context** and no access to any prior chat or memory. Everything needed
> to understand, verify, reproduce, operate, and continue the work is here.
>
> **One-sentence summary:** The try-on SeedVR2 image-upscale at long-edge **2730** was **5–7s** and is
> now **~2.4s**. The cause was **not** torch.compile — it was the VAE being offloaded to CPU and
> re-converted fp16→bf16 on *every* request. Fix = keep DiT+VAE **resident on the GPU**, plus compile
> + prewarm the single (2:3-only) output shape.

- **Date:** 2026-06-10
- **Hardware:** RunPod pod, **RTX PRO 6000 Blackwell, 96 GB** (sm_120), torch **2.11.0+cu130**
- **Project (local repo):** `glamify-image-ai` — FastAPI service; this file lives at
  `docs/seedvr2-2730-optimization.md`
- **Pod deploy path:** `/workspace/glamify-image-ai` (a checkout of the same repo on the pod)

---

## 0. System overview (what this service is)

`glamify-image-ai` is a FastAPI app serving fashion AI on **one GPU**, with three "resident runtimes"
loaded at startup (env `RESIDENT_RUNTIMES="wardrobe,tryon,upscale"`):

- **wardrobe / tryon** — Qwen-Image-Edit-2511 diffusers (fp8 + torch.compile) with PEFT LoRAs.
- **upscale** — **SeedVR2 3B fp8** image upscaler (numz `ComfyUI-SeedVR2_VideoUpscaler`), called
  in-process. **This doc is about the upscaler.**
- MiniCPM-V (vLLM) is also loaded for validation.

All heavy GPU work is **serialized through a single process-wide execution coordinator** so Qwen and
SeedVR2 never run concurrently (they'd stack activation peaks and OOM the 96 GB card).

**Request flow for try-on:** user photo → Qwen generates the try-on image (at the *user's* photo
dimensions) → **SeedVR2 upscales it to long-edge 2730** → delivered.

Endpoints: `/v1/*` need a Bearer token; **`/tools/*` are unauthenticated** (handy for testing). The
app listens on `:8000`; nginx fronts `:8001`.

---

## 1. Goal & measured outcome

| | Before | After |
|---|---|---|
| Try-on upscale @2730 (warm, in prod) | 5–7s | **~2.4s** (measured 2.38–2.65s) |
| Upscale mode | eager + VAE bounced to CPU | resident-on-GPU + static compile + prewarmed |
| Delivered output | upscale→3072 then downscale→2048 | **upscale→2730, deliver 2730** |
| Peak VRAM for one 2730 upscale | ~34 GB (would OOM beside Qwen) | **~24 GB** (fits; ~29 GB free) |

**"Under 2s at 2730" is physically impossible on this card.** Isolated compiled warm breakdown:
DiT ~1.5s + VAE decode 0.58s + encode/overhead ~0.3s ≈ **~2.4s**. The 2730 output is ~1820×2730 ≈
5 MP and the VAE decode is memory-bandwidth bound. The only way below ~2.4s is a smaller output edge.

---

## 2. Root cause analysis (how we know it wasn't compile)

We profiled an **isolated** (whole-GPU) **eager** 2730 run with the CLI's debug timing on. It was
**still ~5.3s**, so contention/compile were not the story. Per-phase:

| Phase | Time | Type |
|---|---|---|
| DiT generation | ~2.1s | real compute |
| VAE **decode** | 0.92s | real compute (fast) |
| VAE encode | 0.08s | real compute |
| **VAE weights converted fp16→bf16** | **~2.6s** | **pure overhead, every request** |
| VAE moved CPU↔GPU (×4 per request) | ~0.9s | **pure overhead** |

**The trap (root cause):** the SeedVR2 CLI function `_parse_offload_device(offload_arg, …,
cache_enabled)` in
`/workspace/seedvr2_eval/ComfyUI-SeedVR2_VideoUpscaler/inference_cli.py` does:

```python
if offload_arg == "none":
    return "cpu" if cache_enabled else None   # <-- THE TRAP
```

We pass `--cache_vae --cache_dit` (caching ON) **and** `--vae_offload_device none`. Because caching is
on, `none` is **silently rewritten to `cpu`** → the VAE is evicted to CPU between the encode and decode
phases and **re-materialized + re-converted fp16→bf16 (~2.6s)** on the way back, every single request.

**The fix:** set the offload device to the **GPU itself (`"0"`)**. It's not `none` so it dodges the
trap, and "offloading to cuda:0" is a **no-op that keeps the model resident** in the already-converted
dtype. Result: no CPU bounce, no re-convert. This alone took **5.3s → 3.9s** (eager). Adding static
compile took it to **~2.4s**. As a bonus it **lowered peak VRAM** (~34 GB → ~24 GB), which is why it
now co-exists with the resident Qwen stack (~64 GB) inside 96 GB.

---

## 3. Exact changes made

There are three places: **(A)** repo code, **(B)** the pod's launch env, **(C)** the pod's `.env`.

### (A) Code — `app/clients/seedvr2.py`

`seedvr2.py` wraps the SeedVR2 CLI: it imports the CLI module in-process, builds an argv, and calls
`process_single_file(...)`. Two changes were made.

**A1 — Resident-on-GPU (THE fix).** In `_build_args(...)`, the offload devices were changed from
`none/none/cpu` to a single GPU device. Current verbatim code:

```python
# Keep DiT + VAE RESIDENT on the GPU between requests. The CLI has a trap: with
# --cache_dit/--cache_vae, an offload device of "none" is silently rewritten to "cpu"
# (see inference_cli.py::_parse_offload_device, cache_enabled branch). That makes the
# VAE bounce GPU->CPU->GPU and re-convert fp16->bf16 (~2.6s) on EVERY request. Pointing
# the offload device at the GPU itself ("0") makes the cache a no-op that keeps weights
# resident: ~5.3s -> ~3.9s per 2730 upscale (single-image). On CPU-only, fall back.
offload_device = "0" if self._backend == "cuda" else "none"
argv = [
    str(cli_path), str(input_path), "--output", str(output_path),
    "--output_format", "png",
    "--dit_model", model_variant, "--model_dir", str(model_dir),
    "--resolution", str(int(derived_short_edge)),       # short edge (derived from input aspect)
    "--max_resolution", str(int(target_long_edge)),     # long edge target (2730)
    "--batch_size", "1",
    "--cache_dit", "--cache_vae",
    "--dit_offload_device", offload_device,              # was "none"
    "--vae_offload_device", offload_device,              # was "none"
    "--tensor_offload_device", offload_device,           # was "cpu"
]
if os.environ.get("UPSCALE_COMPILE", "1") != "0":
    argv += ["--compile_dit", "--compile_vae"]
```

**A2 — Mega-cache (KEPT but INEFFECTIVE — see §7.1).** Added a module constant and two methods to try
to skip the ~300s boot recompile by persisting torch's compile artifacts. It loads/saves a ~130 MB
blob correctly **but does not actually skip the recompile** (cache key unstable for fp8 ops). Verbatim:

```python
_MEGACACHE_PATH = "/workspace/.seedvr2_compile_megacache.bin"

def _load_compile_cache(self) -> None:   # called at end of _load_module(), BEFORE first compile
    if os.environ.get("UPSCALE_COMPILE", "1") == "0":
        return
    try:
        import torch
        path = Path(os.environ.get("UPSCALE_MEGACACHE", _MEGACACHE_PATH))
        if not path.exists():
            logger.info("SeedVR2 compile mega-cache absent (%s); will compile fresh", path); return
        torch.compiler.load_cache_artifacts(path.read_bytes())
        logger.info("SeedVR2 compile mega-cache loaded ...")
    except Exception as exc:
        logger.warning("SeedVR2 compile mega-cache load failed: %s", exc)

def _save_compile_cache(self) -> None:   # called at end of _prewarm(), AFTER compiling all shapes
    if os.environ.get("UPSCALE_COMPILE", "1") == "0":
        return
    try:
        import torch
        artifacts = torch.compiler.save_cache_artifacts()
        if not artifacts: return
        data, _info = artifacts
        path = Path(os.environ.get("UPSCALE_MEGACACHE", _MEGACACHE_PATH))
        tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_bytes(data); tmp.replace(path)
        logger.info("SeedVR2 compile mega-cache saved ...")
    except Exception as exc:
        logger.warning("SeedVR2 compile mega-cache save failed: %s", exc)
```

**Already present from earlier work (do not remove):**
- `OptimizedModule.__bool__ = lambda self: True` monkeypatch — fixes numz issue #502 (a compiled
  module wrapped in `OptimizedModule` has no `__bool__`, so `if model:` falls back to `__len__` and
  raises "CompatibleDiT does not support len()").
- `torch.backends.cudnn.benchmark = True` — picks fastest Conv3d algos (costs ~30s autotune the first
  time each new shape is seen).
- Persistent inductor/triton cache dirs via env (`TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor_cache`,
  `TRITON_CACHE_DIR=/workspace/.triton_cache`, `TORCHINDUCTOR_FX_GRAPH_CACHE=1`).
- `_prewarm()` daemon thread (started by `warmup()` when `UPSCALE_WARMUP!=0`) that runs a synthetic
  upscale at each edge in `UPSCALE_WARMUP_EDGES` using an **832×1248 (2:3)** image, so the per-shape
  compile happens at boot, not on the first user request.

### (B) Launch env — pod supervisor `/tmp/glamify-supervisor.sh`

A bash `while true` loop that relaunches the app. Changed the SeedVR2 env (Qwen env untouched):

```diff
-    UPSCALE_COMPILE="0" \
-    UPSCALE_WARMUP="0" \
+    UPSCALE_COMPILE="1" \
+    UPSCALE_WARMUP="1" \
+    UPSCALE_WARMUP_EDGES="2730" \
```

Full relevant block (for reference; Qwen lines unchanged):
```
env \
  RESIDENT_RUNTIMES="wardrobe,tryon,upscale" \
  QWEN_FP8="true" QWEN_COMPILE="true" QWEN_IMAGE_EDIT_DTYPE="bfloat16" \
  MINICPM_ENFORCE_EAGER="true" MINICPM_KV_CACHE_DTYPE="auto" \
  UPSCALE_COMPILE="1" UPSCALE_WARMUP="1" UPSCALE_WARMUP_EDGES="2730" \
  TORCHINDUCTOR_CACHE_DIR="/workspace/.torchinductor_cache" \
  TRITON_CACHE_DIR="/workspace/.triton_cache" TORCHINDUCTOR_FX_GRAPH_CACHE="1" \
  HF_HOME="/workspace/hf_cache" VLLM_USE_FLASHINFER_SAMPLER="0" \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TOKENIZERS_PARALLELISM="false" \
  bash scripts/runpod_start.sh
```

### (C) Pipeline target — pod `.env` (`/workspace/glamify-image-ai/.env`)

```diff
-TRYON_UPSCALE_TARGET_LONG_EDGE=3072
-TRYON_FINAL_OUTPUT_LONG_EDGE=2048
+TRYON_UPSCALE_TARGET_LONG_EDGE=2730
+TRYON_FINAL_OUTPUT_LONG_EDGE=2730
```

Without (C), try-on still upscales to 3072 and the fast prewarmed 2730 graph is never used. The app
reads `.env` via pydantic (`SettingsConfigDict(env_file=".env")`).

---

## 4. Why this config is correct — the compile / shape logic (IMPORTANT)

- `--compile_dit --compile_vae` use torch.compile in **static mode (`dynamic=False`)**. A compiled
  graph is bound to **one exact output shape**. A different long-edge (2730 vs 4096) **or** a different
  input aspect ratio → a **different shape → a separate ~300s cold compile** (then cached in-process).
- **Try-on output dimensions = the user's uploaded photo dimensions** (see
  `app/services/tryon.py` ~lines 95–96 and 171–173). In general that's arbitrary aspects, which would
  make static compile **thrash** (a 300s GPU-blocking compile per new aspect).
- **The product guarantees inputs are 2:3 only.** So every 2730 upscale is the **same shape
  (1820×2730)** → the single prewarmed graph hits every request → ~2.4s. The prewarm image is
  **832×1248 (2:3)**, which yields exactly 1820×2730, matching real requests.
- ⚠️ **If a non-2:3 image ever reaches the upscaler, it triggers a one-off ~300s GPU-blocking compile
  for that shape** (and blocks Qwen too via the shared coordinator). Keep inputs 2:3. If that can't be
  guaranteed, add a guard: pad/letterbox to 2:3 before upscaling, or fall back to eager
  (`UPSCALE_COMPILE=0`, ~3.9s, any shape) for off-shape inputs.
- To add another fixed size later (e.g. a 4096 standalone path): `UPSCALE_WARMUP_EDGES="2730,4096"`.

---

## 5. Pod access (no external context needed)

> ⚠️ **The pod's public IP and port are EPHEMERAL** — they change when the pod restarts. The values
> below were valid on 2026-06-10. **Always re-check the current host:port in the RunPod web console →
> the pod → "Connect" → "Direct TCP" (the SSH over exposed TCP) before relying on them.**

- **WK pod** (the active one, all work was here): RunPod id `ra4nfsl1lrajm5`.
  Last known: `root@157.157.221.29 -p 50769`.
- **Server pod** (exists, do NOT touch unless told): RunPod id `dt3jjdcekx1lvl`.
- **Both pods share one network volume mounted at `/workspace`** (so cache files, models, the
  `glamify-image-ai` checkout, and `.env` are common).
- **SSH key:** `~/.ssh/runpod_qge_ed25519` (on the operator's machine).
- The app runs from `/workspace/glamify-image-ai`, venv at
  `/workspace/.venvs/glamify-image-ai/bin/python`.
- **nvrtc requirement:** the first upscale crashes unless `LD_LIBRARY_PATH` includes the cu13 libs:
  `/workspace/.venvs/glamify-image-ai/lib/python3.12/site-packages/nvidia/cu13/lib`. The supervisor
  sets the process env; any standalone script you run must set it too.

Convenience shell var used in commands below:
```bash
SSH="ssh -o BatchMode=yes -o ServerAliveInterval=15 -i ~/.ssh/runpod_qge_ed25519 -p 50769 root@157.157.221.29"
```

---

## 6. How to operate

### 6.1 Verify it's healthy and fast
```bash
# health
$SSH 'curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/health'

# live 2730 latency via the unauthenticated lab endpoint (uses a test image already on the pod)
$SSH 'curl -s -F image=@/workspace/tmp/test_832x1248.png \
  -F model_variant=seedvr2_ema_3b_fp8_e4m3fn.safetensors \
  -F pre_resize_max_edge=4096 -F output_max_edge=2730 \
  http://127.0.0.1:8000/tools/upscale-lab/run \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[\"timings\"][\"upscale_seconds\"],\"s\",d[\"output\"])"'
# expected: ~2.4 s, output 1820x2730   (if test image missing: create a 832x1248 RGB PNG there)

# watch boot / prewarm / mega-cache lines
$SSH 'grep -iE "prewarm|mega-cache" /workspace/glamify-supervisor.log | tail'
```

### 6.2 Restart the service
> **GOTCHA:** doing kill + GPU-drain + relaunch in **one** SSH command repeatedly dropped the
> connection (exit 255). **Run these as three separate SSH calls.**
```bash
# 1) stop supervisor loop + the app
$SSH 'pkill -f "[g]lamify-supervisor.sh"; sleep 1; kill $(pgrep -f "[u]vicorn app.main" | head -1) 2>/dev/null'
# 2) drain GPU + clear orphan vLLM engine processes
$SSH 'for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 4; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader'
# 3) relaunch (detached)
$SSH 'nohup setsid bash /tmp/glamify-supervisor.sh > /workspace/glamify-supervisor.log 2>&1 </dev/null & disown; echo relaunched'
```
**Boot ≈ 10–15 min:** MiniCPM + Qwen fp8 warm (~170s) + SeedVR2 prewarm (~290–340s, see §7.1).
`/health` returns `200` *before* prewarm finishes; **upscale requests block until prewarm completes**
(they queue on the SeedVR2 run-lock).

### 6.3 Useful greps in the boot log (`/workspace/glamify-supervisor.log`)
- `SeedVR2 prewarm at long_edge=2730 done in NNNs` — prewarm finished.
- `SeedVR2 compile mega-cache loaded/saved/absent` — mega-cache activity.
- `Qwen ... runtime ready ... warm=NNNs` — Qwen ready.

---

## 7. Known limitations & open items

### 7.1 Boot recompile ~290–340s every cold restart — UNSOLVED
Every cold restart, the prewarm spends ~290–340s compiling. We tried torch's **mega-cache**
(`torch.compiler.save_cache_artifacts` / `load_cache_artifacts`): the 130 MB blob saves and loads
correctly, but the compile **still recompiles** (294s *with* the cache ≈ 281s *without*). Reasons:
(a) torch's compile cache **key is not stable across processes** for SeedVR2's custom fp8 DiT/VAE, so
the loaded artifacts are never matched; (b) a chunk of the time is **cuDNN autotune** + **GPU
contention with Qwen's concurrent boot**, neither of which any artifact cache removes.
**Impact is limited:** it's **one-time per restart**, runs in a background thread (doesn't block
`/health`), and **does NOT affect the ~2.4s per-request speed.** Practical mitigation: **restart less
often.** Possible future experiment: run the SeedVR2 prewarm **serialized after** Qwen finishes
warming (the isolated, uncontended compile was ~108s) — might roughly halve boot compile.

### 7.2 Mega-cache code is dead weight
Since it gives no benefit, it can be removed from `app/clients/seedvr2.py`: delete `_load_compile_cache`,
`_save_compile_cache`, their two call sites (end of `_load_module`, end of `_prewarm`), and
`_MEGACACHE_PATH`. It is harmless if left (best-effort try/except), only wastes ~130 MB write + a
read each boot.

### 7.3 Persistence — these changes are NOT permanent yet
- `/tmp/glamify-supervisor.sh` is **ephemeral** (lost if the pod is recreated). The
  `UPSCALE_COMPILE=1 / UPSCALE_WARMUP=1 / UPSCALE_WARMUP_EDGES=2730` env **must be baked into the real
  bootstrap/deploy** (wherever the supervisor is generated, e.g. `scripts/runpod_start.sh` or the pod
  template).
- `.env` (`TRYON_UPSCALE_TARGET_LONG_EDGE=2730`, `TRYON_FINAL_OUTPUT_LONG_EDGE=2730`) lives on the
  shared volume; persists unless the volume is reset.
- The `app/clients/seedvr2.py` change is in the **repo working tree** (local + on the pod) but was
  **NOT git-committed** as of 2026-06-10. **Commit it** so it survives.

### 7.4 The 2:3-only assumption (see §4)
Off-aspect inputs cause a one-off ~300s GPU-blocking compile. Add a guard if 2:3 can't be guaranteed.

---

## 8. Models, files & facts (reference)

**Model dir `/workspace/models/seedvr2`:**
- `seedvr2_ema_3b_fp8_e4m3fn.safetensors` — **DiT, fp8, 3.2 GB. THIS is the model in use.**
- `ema_vae_fp16.safetensors` — **VAE, fp16, 479 MB.** The *only* VAE; shared by all DiT variants;
  cast to bf16 at load (that cast was the ~2.6s overhead in §2). **There is no fp8 VAE** — only the
  DiT was ever quantized. At runtime: `DiT=float8_e4m3fn, VAE=bfloat16, compute=bfloat16`.
- `seedvr2_ema_7b_*` (×2, 7.9 GB) — 7B variants, **unused** (slower).

**Key paths:**
- SeedVR2 CLI: `/workspace/seedvr2_eval/ComfyUI-SeedVR2_VideoUpscaler/inference_cli.py`
  (numz ComfyUI-SeedVR2 v2.5.x).
- Wrapper: `app/clients/seedvr2.py` (the file we changed).
- Runtime wiring: `app/runtime/warmup.py` (`warmup_resident_runtimes` → `warmup_upscale_runtime`),
  `app/runtime/upscale_runtime.py`, `app/main.py` (calls warmup at startup).
- Lab endpoint: `app/routes/upscale_lab.py` + `app/services/upscale_lab.py` (POST
  `/tools/upscale-lab/run`, form fields `image`, `model_variant`, `pre_resize_max_edge`,
  `output_max_edge`).
- Boot log: `/workspace/glamify-supervisor.log`. Supervisor: `/tmp/glamify-supervisor.sh`.

**Measured numbers (2730, this session):**
- eager + CPU-bounce (original): **5.3s** isolated / 5–7s in prod
- eager + resident-on-GPU: **3.9s**
- resident + static compile (warm): **2.79s** isolated, **~2.4s** in prod
- compile build (cold, one shape): ~108s isolated / ~280–340s in prod
- peak VRAM for one 2730 upscale: **~24 GB**

---

## 9. Standalone profiler (recreate to re-measure without restarting the app)

This script measures eager vs compiled, resident vs CPU-offload, per-phase timing, and peak VRAM —
**loading only SeedVR2 (not Qwen)**. To run it you must **free the GPU first** (stop the app per §6.2),
because SeedVR2 at 2730 needs ~24 GB and won't fit beside the resident ~64 GB Qwen stack. Save as
`/workspace/tmp/sv2_profile.py` and run with the venv python.

```python
import os, sys, time, io
from contextlib import redirect_stdout, redirect_stderr
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # REQUIRED: match the app or it core-dumps
os.environ["LD_LIBRARY_PATH"] = ("/workspace/.venvs/glamify-image-ai/lib/python3.12/site-packages/nvidia/cu13/lib:"
                                 + os.environ.get("LD_LIBRARY_PATH", ""))   # nvrtc fix
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/workspace/.torchinductor_cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/workspace/.triton_cache")
import importlib.util, torch
CLI = "/workspace/seedvr2_eval/ComfyUI-SeedVR2_VideoUpscaler/inference_cli.py"
MODEL_DIR, VARIANT = "/workspace/models/seedvr2", "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
INP, LONG_IN, SHORT_IN = "/workspace/tmp/test_832x1248.png", 1248, 832
spec = importlib.util.spec_from_file_location("sv2cli", CLI); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.debug.enabled = True                                   # turn on per-phase timing logs
try:
    from torch._dynamo.eval_frame import OptimizedModule
    if not hasattr(OptimizedModule, "__bool__"): OptimizedModule.__bool__ = lambda self: True   # numz #502
except Exception: pass
torch.backends.cudnn.benchmark = True
device_list = ["0"] if str(m.get_gpu_backend()) == "cuda" else ["cpu"]; runner_cache = {}
def run(edge, off, compile_on, tag):
    short = int(round(SHORT_IN * (edge / LONG_IN)))
    argv = [CLI, INP, "--output", f"/workspace/tmp/prof_{tag}.png", "--output_format", "png",
            "--dit_model", VARIANT, "--model_dir", MODEL_DIR, "--resolution", str(short),
            "--max_resolution", str(edge), "--batch_size", "1", "--cache_dit", "--cache_vae",
            "--dit_offload_device", off, "--vae_offload_device", off, "--tensor_offload_device", off]
    if compile_on: argv += ["--compile_dit", "--compile_vae"]
    old = list(sys.argv); torch.cuda.reset_peak_memory_stats(); cap = io.StringIO(); t = time.perf_counter(); err = None
    try:
        sys.argv = argv; args = m.parse_arguments()
        with redirect_stdout(cap), redirect_stderr(cap):
            m.process_single_file(INP, args, device_list=device_list, output_path=f"/workspace/tmp/prof_{tag}.png",
                                  format_auto_detected=False, runner_cache=runner_cache)
    except Exception as e: err = repr(e)[:240]
    finally: sys.argv = old
    print(f"[{tag}] edge={edge} off={off} compile={compile_on} wall={time.perf_counter()-t:6.2f}s "
          f"peak={torch.cuda.max_memory_allocated()/1e9:5.1f}GB err={err}", flush=True)
    for ln in cap.getvalue().splitlines():
        if any(k in ln.lower() for k in ("processing time","vae decode:","vae encode","converted to bfloat16")):
            print("   |", ln.strip()[:150], flush=True)
# off="0" = resident on GPU (FAST). off="none"+cache silently becomes "cpu" (the SLOW original).
run(2730, "0", False, "eager_1"); run(2730, "0", False, "eager_2")           # warm eager ~3.9s
if os.environ.get("DO_COMPILE", "0") == "1":
    run(2730, "0", True, "compiled_BUILD"); run(2730, "0", True, "compiled_warm1")  # warm compiled ~2.8s
print("DONE", flush=True)
```
Run: `$SSH 'DO_COMPILE=1 /workspace/.venvs/glamify-image-ai/bin/python -u /workspace/tmp/sv2_profile.py'`
(measure peak VRAM externally by sampling `nvidia-smi --query-gpu=memory.used --format=csv,noheader`
in a loop while it runs; the in-script `peak` only counts torch-tracked tensors and reads low).

---

## 10. Quick start for the next assistant

1. Re-check the pod host:port in the RunPod console; set `$SSH` (§5).
2. Confirm state: health + a live 2730 request (§6.1). Expect ~2.4s, 1820×2730.
3. If asked to make it permanent: **commit** `app/clients/seedvr2.py`; bake the supervisor env
   (§3B) into the real deploy; keep `.env` 2730/2730 (§3C).
4. If asked to reduce boot time: see §7.1 (serialize prewarm after Qwen). Don't expect the mega-cache
   to help.
5. If asked to add a resolution/aspect: prewarm it via `UPSCALE_WARMUP_EDGES`, and remember static
   compile = one graph per shape (§4).
6. **Don't** chase "<2s at 2730" — it's below the physical decode floor (§1).
```

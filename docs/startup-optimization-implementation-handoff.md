# Startup-Time Reduction — Implementation Handoff (self-contained)

> **For a fresh AI assistant / engineer with ZERO prior context.** Everything needed to implement is
> here. Goal: **cut cold-start from ~13–15 min toward ~3–4 min** (theoretical floor ~1.5–2 min) for
> the `glamify-image-ai` service, **without changing inference latency or output quality at all.**
>
> **THE HARD RULE (read twice):** every change here must be **startup-only**. After *any* change you
> must prove a warm try-on/upscale is **byte-for-byte the same speed and output** as before (see §8
> validation). If latency or output moves, the change touched inference — **revert it.**
>
> Companion docs in this repo (read for full background): `docs/seedvr2-2730-optimization.md` (the
> already-shipped 2730 upscale work) and `docs/startup-and-tryon-upscale-optimization-plan.md` (the
> broader plan; this doc is the focused, actionable subset).
> Dates/measurements: 2026-06-10, RTX PRO 6000 Blackwell 96 GB, torch 2.11.0+cu130.

---

## 1. What this service is (context)

`glamify-image-ai` is a FastAPI app on **one GPU** with three resident runtimes warmed at startup
(`RESIDENT_RUNTIMES="wardrobe,tryon,upscale"`):
- **Qwen-Image-Edit-2511** (diffusers) — fp8 (torchao) + `torch.compile` + PEFT LoRAs → wardrobe & try-on.
- **SeedVR2 3B fp8** upscaler (numz ComfyUI-SeedVR2 CLI, in-process) → upscales try-on output to 2730.
- **MiniCPM-V** (vLLM subprocess) — garment description; **not used by the try-on path.**
All GPU work serializes through one process-wide lock (one GPU = one op at a time). Endpoints:
`/v1/*` need a Bearer JWT; `/health` + `/ready` + `/tools/*` are public.

**Try-on flow (the hot path):** validated user image (always **832×1248**, 2:3) → Qwen generates at
832×1248 → SeedVR2 upscales to **1820×2730** (~2.4 s) → deliver 2730.

---

## 2. Pod access (inline — no external context needed)

> ⚠️ **Public host:port is EPHEMERAL** (changes on pod restart). Values below were valid 2026-06-10.
> **Re-check in the RunPod console → pod → Connect → "SSH over exposed TCP" before relying on them.**

- Pod (current): id `muji5i1u5jctux`. Last known: `root@157.157.221.177 -p 15018`.
- **SSH key: `~/.ssh/runpod_qge_ed25519`** (NOT `id_ed25519` — that one is rejected).
- App on `:8000`, nginx on `:8001`. Shared **MooseFS network volume at `/workspace`** (~350–700 MB/s)
  holds the repo checkout, models, `.env`, logs — common to all pods.
- Repo on pod: `/workspace/glamify-image-ai`. venv python:
  `/workspace/.venvs/glamify-image-ai/bin/python`. Models: `/workspace/models/`.
- **nvrtc requirement:** first upscale crashes unless `LD_LIBRARY_PATH` includes
  `/workspace/.venvs/glamify-image-ai/lib/python3.12/site-packages/nvidia/cu13/lib` (the supervisor
  sets it; standalone scripts must too).

Example (inline the full ssh each time — see §9 gotcha about shell variables):
```bash
ssh -o BatchMode=yes -o ServerAliveInterval=15 -i ~/.ssh/runpod_qge_ed25519 -p 15018 root@157.157.221.177 'curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/ready'
```

---

## 3. Current state (baseline you start from)

**Already shipped & validated (DO NOT regress these — they define inference speed/quality):**
- SeedVR2 resident-on-GPU (`offload_device="0"` in `app/clients/seedvr2.py::_build_args`) → upscale **2.4 s** warm.
- `UPSCALE_COMPILE=1`, `UPSCALE_WARMUP=1`, `UPSCALE_WARMUP_EDGES=2730` (compile + prewarm the one 2:3 shape).
- `.env`: `TRYON_UPSCALE_TARGET_LONG_EDGE=2730`, `TRYON_FINAL_OUTPUT_LONG_EDGE=2730`.
- Readiness gating: `GET /ready` returns 200 only when wardrobe+tryon loaded, upscale **prewarmed**,
  and not degraded (`SeedVR2Client._prewarmed` flag). `/health` is liveness-only.
- Hang watchdog in `app/runtime/coordinator.py` (`SYSTEM_EXECUTION_TIMEOUT_SECONDS`, default 300).
- Validation enforces exact **832×1248** (`app/services/user_validation.py::_normalize_user_image`).
- Mega-cache code was **removed** (it didn't reuse compile across restarts — don't re-add it).
- Persistent supervisor: `scripts/glamify-supervisor.sh` (env + `while true` relaunch).

**Baseline cold start ≈ 13–15 min**, broken down (measured):
| Phase | Time |
|---|---|
| Qwen: imports + load 54 GB bf16 (network) + quantize(0.5s) + compile + warm gens | ~3–7 min (cache-dependent) |
| MiniCPM-V (vLLM) load + engine init | ~2–2.5 min |
| Fashion detector + SigLIP + tryon adapters | ~0.5 min |
| **→ /health 200** | ~early |
| SeedVR2 prewarm (2730 compile) — **currently runs LAST, sequential** | ~108–294 s isolated (700 s+ under contention) |
| **→ /ready 200 (fully warm)** | **~13–15 min** |

Key measured facts (so you don't re-discover):
- `quantize_` = **0.5 s** → persisting fp8-to-skip-quantize is **pointless** (do not pursue).
- Qwen cost is **loading** (54 GB over network ~156 s cold) + **compile/warm gens** (~111–127 s).
- SeedVR2 prewarm is **compile-bound**; torch mega-cache does **not** reuse it across restarts (proven).
- `/workspace` is network (~350–700 MB/s); local `/` overlay is ~13 GB/s but **ephemeral**.

---

## 4. THE INVARIANT — what must never change (or you change inference)

These knobs/values determine latency & quality. **Leave them exactly as-is:**
- `UPSCALE_COMPILE=1`, `QWEN_COMPILE=true`, `QWEN_FP8=true` (going eager = slower; don't).
- `offload_device="0"` resident fix in `seedvr2.py`.
- Model variants, dtypes (`QWEN_IMAGE_EDIT_DTYPE=bfloat16`), the torchao quantize config.
- Prewarm shape (2730 / 832×1248), `TRYON_UPSCALE_TARGET_LONG_EDGE=2730`, `TRYON_FINAL_OUTPUT=2730`.
- The number of inference steps, guidance, LoRA scales.

**Startup optimizations may ONLY change:** WHEN/ORDER things load & warm, HOW MANY warm generations
run at boot, WHERE model files live (faster disk), and whether compile artifacts are *precompiled*
(AOTInductor — and only if proven numerically identical). Nothing that alters the served graph.

**Litmus test after every change (see §8):** warm upscale still **2.4 s**, output still **1820×2730**;
warm try-on Qwen-gen time unchanged. If not → revert.

---

## 5. Implementation plan — ordered by value ÷ risk

Do these **one at a time**, each behind a flag, each followed by a monitored boot (§8). Stop when the
startup target is met. Expected stack: baseline ~13–15 min → ~3–4 min (Steps 1–3), ~1.5–2 min (4–5).

### STEP 1 — Parallel warmup v2 (overlap SeedVR2 compile with Qwen load) — biggest single win
**Status:** code exists, flag `STARTUP_PARALLEL_WARMUP` (default 0). Attempt #1 **crash-looped** —
NOT memory (peak only 60 GB), but an **import-order bug**: starting the upscale prewarm first imports
the SeedVR2/ComfyUI stack **before** Qwen, which leaves `bitsandbytes` half-initialized so Qwen's PEFT
LoRA load dies with `AttributeError: module 'bitsandbytes' has no attribute 'nn'` → segfault → loop.

**The v2 fix (do this, then enable the flag):** force a clean, ordered import of the bnb/peft stack at
process startup **before any SeedVR2 import**, so SeedVR2 loading first can't poison it. Add to
`app/main.py` (top of `create_app`, before warmup) or a new `app/runtime/preimport.py` called first:
```python
# Pre-import the bitsandbytes/PEFT stack BEFORE SeedVR2 is ever imported, so parallel warmup
# (which may load the SeedVR2/ComfyUI stack first) cannot leave bitsandbytes half-initialized.
try:
    import bitsandbytes  # noqa
    import bitsandbytes.nn  # noqa  # ensures bnb.nn exists for peft's hasattr check
    import peft  # noqa
except Exception:
    pass
```
Then the existing parallel path (in `app/runtime/warmup.py`, gated by `STARTUP_PARALLEL_WARMUP`)
starts `warmup_upscale_runtime` first; its prewarm is a daemon thread whose compile **self-staggers on
free VRAM** (`SeedVR2Client._wait_for_vram_headroom`, already implemented:
`UPSCALE_PREWARM_MIN_FREE_GB` default 26, `UPSCALE_PREWARM_WAIT_TIMEOUT_S` default 300).
**Validation:** monitored boot with the flag on, sampling `nvidia-smi` peak every 1 s. Pass if:
peak < ~90 GB, **no** `bitsandbytes`/Traceback/OOM in the log, `/ready` reaches 200, and §8 latency
unchanged. **Saving: ~3–5 min** (SeedVR2 compile + MiniCPM hide under Qwen → cold ≈ Qwen's chain).
**Rollback:** `STARTUP_PARALLEL_WARMUP=0`.

### STEP 2 — Fewer Qwen warm generations — low risk
Qwen warmup runs ~7 warm generations (one per wardrobe+tryon category) to trigger
`compile_repeated_blocks` + warm the path (~111–127 s). The compile is shared across adapters
(`compile_repeated_blocks` compiles the single repeated block once), so you likely need only **1–2**
warm gens to trigger it, not 7. In `app/clients/qwen_diffusers_engine.py::warmup`, reduce
`warm_categories` to the minimum that still triggers the compile + each LoRA path needed.
**Validation:** after boot, first real try-on of EACH category must be normal speed (no first-request
compile stall) — if a category recompiles on first use, you cut too many. **Saving: ~60–90 s.**
**Rollback:** restore the full warm list.

### STEP 3 — Take MiniCPM off the try-on readiness path — low/med risk
MiniCPM is **not used by try-on**. If this pod's job is try-on, don't let MiniCPM's ~2–2.5 min gate
readiness: warm it **asynchronously** (background thread / lazy on first `/dev/minicpm` use) and keep
`/ready` gated only on wardrobe+tryon+upscale (it already is — but confirm MiniCPM isn't loaded
synchronously inside the wardrobe warmup chain; if it is, move it to a background thread).
**Validation:** `/ready` flips green without waiting for MiniCPM; a `/dev/minicpm` call still works
(first one may be slower). **Saving: removes ~2–2.5 min from the readiness critical path** (if MiniCPM
was on it). **Rollback:** restore synchronous MiniCPM warmup.

### STEP 4 — Load-trim (cut the 54 GB network read) — higher effort, VALIDATE QUALITY
The Qwen 54 GB bf16 network read (~156 s cold) is the remaining floor. Two options:
- **4a (lower risk): bake models into the Docker image** so they're on local NVMe (~13 GB/s) from boot
  → load ~10–30 s, no per-pod network copy. Cost: large image; build/registry work. No model change.
- **4b (needs a quality spike): load a pre-saved fp8 file (20 GB)** instead of 54 GB bf16. We measured
  the save works (20.4 GB) **but torch warned "Unable to import torchao Tensor objects"** → the reload
  is NOT guaranteed bit-correct. **MANDATORY:** before trusting, generate the same try-on with the
  reloaded fp8 vs the current path and confirm **identical output** (pixel diff ≈ 0). If not identical,
  abandon 4b. (Reload pattern: build transformer on `meta`, `load_state_dict(..., assign=True)`.)
**Saving: ~1.5–2 min.** **Rollback:** load from the original 54 GB path.

### STEP 5 — AOTInductor precompile (kill the boot compile) — highest effort/risk, OPTIONAL
torch.compile artifacts don't persist across restarts (mega-cache proven dead). The only real
"compile once, load forever" path is **AOTInductor** (`torch.export` → a standalone `.so` per fixed
shape on the volume, loaded at boot). Targets: SeedVR2 DiT/VAE at 1820×2730, and/or Qwen's repeated
block. **Risk:** custom fp8 ops may not export cleanly — spike it in isolation first.
**MANDATORY quality check:** the `.so` output must be numerically identical to the live-compiled path
(§8). **Saving: ~3–5 min** (no boot compile; MiniCPM becomes the bottleneck). **Rollback:** fall back
to live `torch.compile` if the `.so` is missing/invalid.

### ALTERNATIVE (recommended for autoscaling): WARM POOL — zero boot-floor fighting
Cold-boot is already **hidden from users by readiness gating**. For autoscaling responsiveness, keep a
**minimum of 1–2 pods always `/ready`** and scale the pool *ahead* of demand (use the coordinator's
`waiting_jobs` as the scale signal). This makes the cold-boot time irrelevant to UX and avoids all the
risk above. Do this first if the goal is "add servers in parallel without users feeling it."

---

## 6. The floor — how low it can realistically go
| Stacked config | Cold `/ready` | Risk |
|---|---|---|
| Current (sequential) | ~13–15 min | shipped |
| + Step 1 (parallel v2) | ~4–5 min | medium |
| + Steps 2–3 (fewer warm gens, MiniCPM async) | ~2.5–3.5 min | low |
| + Step 4 (load-trim) | ~2–3 min | med/high |
| + Step 5 (AOTInductor) | ~1.5–2 min (MiniCPM-bound) | high |
| **Below ~1.5 min** | **not achievable** | imports + vLLM init + ≥1 warm pass floor |
**Realistic target: ~3–4 min (Steps 1–3).** Sub-2-min needs Steps 4–5 (effort/risk). None of these
change inference.

---

## 7. Boot / restart procedure (and crash recovery)
Supervisor: `scripts/glamify-supervisor.sh` (env + `while true` relaunch). **Boot ≈ 10–15 min.**
```bash
# stop (run as SEPARATE ssh calls — see §9):
ssh ... 'pkill -f "[g]lamify-supervisor.sh"; sleep 1; kill $(pgrep -f "[u]vicorn app.main"|head -1) 2>/dev/null'
ssh ... 'for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 3; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
# start:
ssh ... 'nohup setsid bash /workspace/glamify-image-ai/scripts/glamify-supervisor.sh > /workspace/glamify-supervisor.log 2>&1 </dev/null & disown; echo started'
```
**If it crash-loops** (multiple `glamify-image-ai start` lines + Tracebacks): read the traceback
(`sed -n '1,60p' /workspace/glamify-supervisor.log`), flip the offending flag off (e.g.
`STARTUP_PARALLEL_WARMUP=0` via `sed -i` on the supervisor), kill + relaunch.

---

## 8. Validation methodology (run after EVERY change)
1. **Monitored boot** — before relaunch, start a 1 s VRAM sampler; after boot, check peak:
   ```bash
   ssh ... 'nohup setsid bash -c "for i in \$(seq 1 1200); do echo \"\$(date +%s) \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\"; sleep 1; done" > /workspace/tmp/mem.log 2>&1 </dev/null & disown'
   # after boot: sort -n -k2 /workspace/tmp/mem.log | tail -1   # peak MiB, must be < ~90000
   ```
2. **No errors:** `grep -iE "Traceback|OutOfMemory|bitsandbytes|degraded" /workspace/glamify-supervisor.log`.
3. **/ready flips to 200**; record cold time vs the 13–15 min baseline.
4. **INFERENCE UNCHANGED (the hard rule):** warm upscale latency + output shape:
   ```bash
   ssh ... 'curl -s -F image=@/workspace/tmp/test_832x1248.png -F model_variant=seedvr2_ema_3b_fp8_e4m3fn.safetensors \
     -F pre_resize_max_edge=4096 -F output_max_edge=2730 http://127.0.0.1:8000/tools/upscale-lab/run \
     | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[\"timings\"][\"upscale_seconds\"], d[\"output\"])"'
   # MUST be ~2.4s and 1820x2730. If slower/different -> the change touched inference -> REVERT.
   ```
   For Steps 4b/5 also do a pixel-diff of a real try-on output vs the pre-change output (must be ~0).
5. **Rollback** is the flag/env for Steps 1–3; restore the file/load-path for 4–5.

---

## 9. Gotchas & lessons (will save you hours)
- **SSH shell:** the local shell does NOT word-split a `$SSH` variable — **inline the full ssh command**
  each call, or it errors "no such file or directory".
- **`pkill -f` self-matches** your own ssh command and drops the connection (exit 255). Use the bracket
  trick (`"[g]lamify-supervisor.sh"`) or kill by explicit PID; run kill and relaunch as **separate**
  ssh calls.
- **Import order (Step 1):** SeedVR2/ComfyUI imported before Qwen poisons `bitsandbytes` → Qwen LoRA
  crash. Pre-import bnb/peft first (Step 1 fix).
- **Don't re-add the torch mega-cache** — it loads but never reuses the compile (294 s with vs 281 s
  without). Dead end.
- **Prewarm time is inflated by concurrent traffic** (saw 294 s → 703 s when test upscales ran during
  it). Measure prewarm on an idle boot.
- **fp8 reload (Step 4b)** threw a torchao deserialize warning — never ship it without a bit-identical
  output check.
- **`/ready` vs `/health`:** route the load balancer/autoscaler at **`/ready`** (fully warm), keep
  `/health` for liveness. This is what makes the first request always fast regardless of boot length.

---

## 10. Key files
- `app/clients/seedvr2.py` — SeedVR2 client: `_build_args` (offload `0` resident fix; **don't change**),
  `_prewarm` + `_wait_for_vram_headroom` (VRAM-gated prewarm), `_prewarmed` flag, `status()`.
- `app/runtime/warmup.py` — `warmup_resident_runtimes` (the `STARTUP_PARALLEL_WARMUP` ordering).
- `app/clients/qwen_diffusers_engine.py` — `warmup` (warm_categories → Step 2), `_load_pipeline`,
  `_ensure_fp8_quantized` (quantize 0.5 s), `_ensure_compiled` (`compile_repeated_blocks`).
- `app/runtime/coordinator.py` — execution lock + hang watchdog (`degraded`).
- `app/routes/health.py` — `/health`, `/ready`.
- `app/main.py` — lifespan → `warmup_resident_runtimes` (add the bnb pre-import here for Step 1).
- `app/services/user_validation.py` — `_normalize_user_image` (832×1248; **don't change**).
- `scripts/glamify-supervisor.sh` — launch env + relaunch loop.
- `app/config.py` — `SYSTEM_EXECUTION_TIMEOUT_SECONDS`, queue knobs.
- `.env` (on pod) — `TRYON_UPSCALE_TARGET_LONG_EDGE=2730`, `TRYON_FINAL_OUTPUT_LONG_EDGE=2730`.

## 11. One-line summary
Cut cold start (~13–15 min) via **startup-only** changes — parallel warmup v2 (pre-import bnb to fix
the crash) + fewer warm gens + MiniCPM-async → **~3–4 min**; optional load-trim/AOTInductor → ~1.5–2 min
— while keeping `UPSCALE_COMPILE/QWEN_COMPILE/QWEN_FP8` and all model config identical so **inference
latency (upscale 2.4 s) and output quality never change**. For autoscaling, prefer a **warm pool**.

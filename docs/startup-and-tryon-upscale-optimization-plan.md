# Startup Time & Try-on Upscale — Optimization Plan (PLAN ONLY, not yet implemented)

> Companion to `docs/seedvr2-2730-optimization.md` (which is already done & deployed). This doc plans
> two follow-ups requested 2026-06-10: **(1) cut the ~16 min cold-start** and **(2) the try-on
> internal upscale "taking more time."** No pod changes were made for this doc — it's for review first.
>
> **Hard constraint from the owner:** none of this may change the behavior or speed of the standalone
> **`/v1/upscale`** API. (It shares the SeedVR2 client, so see §C for the no-regression analysis.)
>
> **Confirmed fact:** try-on user images are **always 2:3**. So the try-on upscale output is always
> **1820×2730**, which is the single prewarmed shape → ~2.4s after warmup, no per-aspect recompile.

---

## 0. Measured cold-start breakdown (from the 03:49:54 boot in `/workspace/glamify-supervisor.log`)

| Phase | Wall | Notes |
|---|---|---|
| Qwen `from_pretrained` (≈60 GB bf16, network volume) + LoRAs + `quantize_` fp8 | **~5.6 min** | start→"Compiled Qwen" 03:49:54→03:55:33 |
| Qwen `compile_repeated_blocks` + 159 s warm generations | **~2.5 min** | →03:58:02 "wardrobe ready warm=159.3s" |
| MiniCPM-V (vLLM) load + engine init | **~2 min** | weights 28.5 s + init; →04:00:23 |
| Fashion detector + SigLIP + tryon adapters | **~0.5 min** | →04:00:35 |
| **App health 200 ("startup complete")** | **≈11 min mark** | 04:00:53 |
| SeedVR2 2730 prewarm compile (background) | **~5 min** | 294 s; →04:05:47 fully warm |
| **Fully warm (first tryon fast)** | **≈16 min mark** | 04:05:47 |

**Two giants:** Qwen (~8 min) and SeedVR2 prewarm (~5 min). Everything below targets these.

---

## A0. MEASURED FINDINGS (2026-06-10 spike on idle pod) — READ BEFORE A1/A2

A standalone spike loaded the exact prod config (`QwenImageEditPlusPipeline.from_pretrained(...,
torch_dtype=bfloat16).to("cuda")` + `quantize_(transformer, Float8DynamicActivationFloat8WeightConfig())`)
and timed each step:
- **`from_pretrained` + `.to(cuda)` (54 GB bf16 from network volume): 156 s** ← the Qwen cost
- **`quantize_`: 0.5 s** ← effectively free
- save fp8 state_dict: 21 s, file = **20.4 GB** (and torch warned "Unable to import torchao Tensor
  objects" → fp8 checkpoint reload is not guaranteed clean)

Disk: `/workspace` is **MooseFS network** (`mfs#eur-is-1.runpod.net`, ~0.35–0.7 GB/s). Local `/`
overlay is **13.4 GB/s read** but only 150 GB and **ephemeral** (lost on pod recreation).

**Consequences (these override the earlier A2/A3 framing):**
- **A2 "persist fp8 to skip quantize" is POINTLESS** — quantize is 0.5 s. Drop it.
- The Qwen cost is **loading 54 GB over the network** (156 s). A smaller **fp8 file (20 GB)** would cut
  this to ~60 s, but reload reliability (torchao subclasses) + live-LoRA structure is unproven → spike
  before trusting.
- **Local-disk staging gives no cold-pod win** (a fresh autoscaled pod still reads from network once).
  The only way to put models on fast local disk for a *cold* pod is **baking them into the Docker
  image**.
- The rest of boot is **compile/warm-bound** (Qwen warm 159 s + SeedVR2 prewarm 108–294 s + MiniCPM
  ~120 s) and compile artifacts **do not persist reliably** (mega-cache proven dead; AOTInductor is the
  only real option, risky).

**Revised recommendation:** the cold boot is ~7 min compile-bound floor; don't expect
artifact-persistence to erase it. Win via **(1) readiness gating + (2) a warm minimum pool**; treat
load-trim (fp8 file / image-baked models) and AOTInductor as optional, measured, later steps.

---

## A. Startup-time plan (ordered by value ÷ risk)

### A1. (DO FIRST — diagnostic) Split the Qwen ~5.6 min into load-vs-quantize
We don't yet know how much of the 5.6 min is **disk read of 60 GB bf16** vs **`quantize_` on GPU**.
Add temporary timing logs around the three steps in `app/clients/qwen_diffusers_engine.py`:
`_load_pipeline` (`from_pretrained`), `_ensure_lora`, `_ensure_fp8_quantized` (`quantize_`).
This decides whether A2 (persist fp8) or A4 (faster disk) is the bigger lever. ~15 min of work, no risk.

### A2. (BIGGEST LEVER) Persist the fp8-quantized Qwen transformer to disk
Today every boot loads bf16 then re-runs `quantize_(transformer,
Float8DynamicActivationFloat8WeightConfig())` on the GPU. Instead, **quantize once, save, reload**:
- **One-time:** after `quantize_`, save the quantized transformer state dict to the volume
  (`torch.save(transformer.state_dict(), "/workspace/models/qwen-fp8/transformer_fp8.pt")` — torchao
  quantized tensors are tensor *subclasses*, so use `torch.save`, **not** safetensors).
- **Each boot:** build the transformer on `meta`/empty, then `load_state_dict(..., assign=True)` from
  the fp8 file, skipping `quantize_` entirely and reading a **much smaller file** (~15–20 GB vs 60 GB).
- **Expected:** removes the GPU re-quantization and ~2/3 of the transformer disk read. Plausibly
  **−3 to −5 min**.
- **Risks / unknowns (needs a validation spike):** torchao checkpoint load is **version-pinned**
  (must reload with the same torchao/torch); requires `weights_only=False`; the diffusers
  `QwenImageEditPlusPipeline` must accept a pre-quantized transformer (load transformer separately,
  then construct the pipeline with it). LoRAs currently load as live PEFT adapters *before* quantize
  (see `_ensure_fp8_quantized` calls `_ensure_lora`) — confirm adapter behavior with a pre-quantized
  base (adapters stay unfused/dynamic, so they should still attach; verify).
- **Fallback:** if pre-quantized load is fragile, at least cache to **local NVMe** (A4).

### A3. Persist the Qwen compile (and verify it actually reloads)
`_ensure_compiled` calls `transformer.compile_repeated_blocks(fullgraph=False)`. This is inductor
codegen, same family as the SeedVR2 compile that **did not** reuse its cache across restarts
(see seedvr2 doc §7.1). Action: measure whether the Qwen compile reuses `TORCHINDUCTOR_CACHE_DIR`
across restarts (look at the gap "Compiled Qwen"→"ready"). If it recompiles, the 159 s warm is partly
codegen we can't easily cache — treat as **known cost**, don't over-invest (mega-cache didn't help
SeedVR2). Lower-value than A2.

### A4. Stage models on local NVMe instead of the network volume
`/workspace` is a shared **network** volume; reading 60 GB (Qwen) + 8 GB (MiniCPM) + 3 GB (SeedVR2)
over it is slow. If the pod has fast local disk, copy models there on first boot and load from local.
- **Pro:** can dramatically cut load time if the volume is the bottleneck (confirm with A1 + a
  `dd`/`hdparm` read test).
- **Con:** breaks the "models shared across pods" convenience; re-copy on each new pod (one-time per
  pod, can run in background). Combine with A2 (smaller fp8 files copy faster).

### A5. SeedVR2 prewarm (~5 min) — make it not gate the first try-on
It already runs in a background daemon thread (doesn't block `/health`). The only impact is the first
try-on upscale blocks until it finishes. Options (pick per priority):
- **Keep as-is** (simplest): health is up at ~11 min; tryon upscale fast after ~16 min.
- **Lazy (skip prewarm):** set `UPSCALE_WARMUP=0`. Boot is ~5 min shorter, but the **first** real
  tryon upscale pays the ~300 s compile instead. Net: same total, just shifted to first request.
- **Best:** keep prewarm, but **start it earlier / in parallel** with Qwen+MiniCPM load so the 5 min
  overlaps the other ~11 min instead of being appended (see A6). Caveat: GPU memory during overlap —
  SeedVR2 compile peaks ~24 GB; Qwen load peaks high; verify they co-fit (96 GB) or stagger.

### A6. Overlap independent load phases
Currently strictly sequential: Qwen (8m) → MiniCPM (2m) → detectors → SeedVR2 prewarm (5m). MiniCPM
runs in its **own vLLM subprocess** and the detectors are small; their *disk load* can overlap Qwen's
compile/warm (CPU/IO-bound) without GPU conflict. Carefully overlapping could save ~2–4 min. Higher
complexity/risk (startup ordering, GPU memory races) — do after A2/A4.

### A7. (IMPORTANT, separate from raw speed) Make the service auto-start & persist
Right now the app is started by `/tmp/glamify-supervisor.sh`, which is **ephemeral** — a new pod (like
`muji5i1u5jctux`) comes up with **no server at all** (that's why it was idle). Bake the launch into the
RunPod template's onstart / `scripts/runpod_start.sh`, including the SeedVR2 env
(`UPSCALE_COMPILE=1 UPSCALE_WARMUP=1 UPSCALE_WARMUP_EDGES=2730`) and the resident-fix code. This
doesn't cut the 16 min but ensures the pod **serves on boot** without a manual 15–20 min babysit.

### Startup plan summary
1. A1 measure (load vs quantize). 2. A2 persist fp8 weights (biggest). 3. A4 local NVMe if disk-bound.
4. A6 overlap loads. 5. A5 prewarm overlap. 6. A7 auto-start. A3 only if it measures as reusable.

**Realistic target:** ~16 min → roughly **6–9 min** (A2+A4+A6) without sacrificing the 2.4s/run or
quality. Going lower means dropping fp8/compile (rejected — they're why inference is fast).

---

## B. Try-on internal upscale "taking more time"

### B1. Most likely cause: the prewarm window, not a per-request problem
Because inputs are **always 2:3**, the upscale output is always 1820×2730 = the prewarmed shape, so a
**warm** tryon upscale is ~2.4s (identical to the `/tools/upscale-lab` measurement). The slowness you
saw is almost certainly because the try-on was run **within the ~5 min SeedVR2 prewarm window** after a
cold boot — the upscale call blocks on the prewarm's run-lock until compile finishes. **Fixing §A
(boot time / prewarm overlap) removes this.**

### B2. Confirm with per-request instrumentation (do before any "fix")
Add structured logging in `tryon.py` around the upscale block (~lines 182–207) and/or in
`seedvr2.py.run`: log `input WxH`, `output WxH`, `upscale_seconds`, and a **recompile flag**
(`TORCH_LOGS=recompiles`, or wrap the call and detect a >30 s duration as "compiled this shape").
Run 3–5 real try-ons **after** full warmup and read the split `qwen_generation_seconds` vs
`seedvr2_upscale_seconds`. Expected after warm: Qwen ~3–4s, upscale ~2.4s. If upscale ≫ 2.4s warm,
escalate; otherwise it was the prewarm window (confirms B1).

### B3. Defensive (cheap insurance): clamp non-2:3 to the prewarmed shape
Even though inputs are "always 2:3," add a guard so a stray off-aspect image can never trigger a 300 s
GPU-blocking compile: before upscaling, if the aspect isn't 2:3 within a small tolerance, **letterbox
/ center-pad to 2:3** (then crop back after), so the upscaler always sees 1820×2730. Pure safety; no
effect in the normal path.

---

## C. No-regression analysis for `/v1/upscale` (the owner's hard constraint)

The standalone `/v1/upscale` and try-on share `app/clients/seedvr2.py` + the same execution
coordinator. Impact of the already-shipped changes and this plan:

- **Resident-on-GPU fix (`offload_device="0"`):** global, helps **both** paths equally (faster + lower
  VRAM). No behavior change to outputs. ✅ No regression — it's an improvement for `/v1/upscale` too.
- **`UPSCALE_COMPILE=1` + prewarm 2730:** prewarm only compiles the **2730** shape. `/v1/upscale` uses
  its own target presets (`_resolve_target_long_edge`: 2k→2048, 4k→4096). Those shapes are **not**
  prewarmed, so a `/v1/upscale` request still pays a one-time compile per preset on first use **exactly
  as it did before this work** — no new regression. If desired, prewarm them too:
  `UPSCALE_WARMUP_EDGES="2730,2048,4096"` (longer boot; only if `/v1/upscale` cold-start matters).
- **A2/A3/A4 (Qwen fp8/compile/disk):** Qwen-only; do not touch the SeedVR2 client → **zero** effect
  on `/v1/upscale`.
- **B3 letterbox guard:** must be implemented **only in the try-on call path** (`tryon.py`), **not** in
  `seedvr2.py.run`, so `/v1/upscale` semantics (arbitrary aspect, exact output) are untouched.

**Rule for implementation:** put try-on-specific shaping in `tryon.py`; keep `seedvr2.py` generic.

---

## D. Validation & rollback (when we do implement, after approval + a boot)

1. After A1, share the load-vs-quantize numbers before building A2.
2. A2 behind a flag (e.g. `QWEN_FP8_CACHE=1`): if the pre-quantized load fails, fall back to live
   `quantize_` automatically (try/except), so a bad cache never bricks startup.
3. Measure: cold-boot wall to (a) health 200, (b) first warm tryon. Compare to the §0 baseline.
4. Verify `/v1/tryon` output unchanged (same pixels/quality) and run a `/v1/upscale` 2048 + 4096 to
   confirm no regression (§C).
5. Rollback: each change is env-gated or file-scoped; revert the env/flag or restore the file.

---

## F. Architecture for autoscaling: fully-warm readiness, fast start, no cold first-user

Goal (owner, 2026-06-10): a new server should come up **with everything preloaded AND warm**, serve a
**fast first request** (no cold compile for user #1), and **not stall one request behind another** —
so we can autoscale by adding identical servers in parallel.

### F0. The single-GPU reality (sets the whole design)
There is one process-wide `BoundedExecutionCoordinator` with a single `_execution_lock`
(`app/runtime/system_coordinator.py` + `coordinator.py`). Every GPU op (Qwen gen, SeedVR2 upscale,
fashion/person detection) serializes through it. **On one GPU this is required** — Qwen (~64 GB) +
SeedVR2 (~24 GB) + MiniCPM (~8 GB) cannot run concurrently without OOM. So:
- **Per pod: GPU work is serial. This is correct; keep the lock.** You cannot remove head-of-line
  blocking *within* a pod for GPU-bound work — the GPU is one resource.
- **Throughput/parallelism = horizontal scale** (more pods behind a load balancer). Each pod handles
  one GPU op at a time; the fleet handles N in parallel.
- Therefore the per-pod requirements are: (1) start fast, (2) become fully warm, (3) only accept
  traffic once warm, (4) keep per-request latency low + the queue bounded/fair.

### F1. (HIGHEST VALUE) Split liveness vs readiness — gate traffic on FULLY-WARM
Today `main.py` warms Qwen+MiniCPM **synchronously** (uvicorn "startup complete" ~11 min) but the
SeedVR2 prewarm is a **background daemon thread** that finishes ~5 min later (~16 min). If the load
balancer routes on "startup complete"/`/health`, **user #1's upscale stalls on the prewarm lock**.
This is the root of "first user is slow" and "requests stuck."

Fix — two distinct probes:
- **`/health` = liveness:** process is up (for the orchestrator to restart a dead pod). Cheap, always
  200 once the app is running.
- **`/ready` = readiness:** returns 200 **only when every runtime is loaded AND warm**, including
  **SeedVR2 prewarm done**. The autoscaler/load balancer must route traffic **only to `/ready` pods.**

Implementation:
- Add a `prewarmed: bool` flag on `SeedVR2Client`, set `True` at the end of `_prewarm()` (and surface
  it via `get_upscale_runtime_status`). Mirror "compiled+warm" flags for Qwen/MiniCPM (mostly exist
  via `status().loaded`; add a `warm` flag set after warmup generations).
- New route `GET /ready` → 200 iff wardrobe/tryon warm + MiniCPM ready + upscale `prewarmed`. Else 503.
- Net effect: a new pod is "not ready" for its whole warm-up, then flips to ready **fully warm** →
  **every request it ever serves, including the first, is fast.** No cold-compile ever hits a user.
- This also makes A5 (prewarm timing) moot for users: traffic simply doesn't arrive until prewarm done.

### F2. Start fast so "ready" comes quickly (autoscaling responsiveness)
Readiness gating means a new pod is useless until warm, so **warm time = scale-up latency.** Cutting
the 16 min (Part A: persist fp8 Qwen weights A2, local NVMe A4, overlap loads A6) directly improves
autoscaling responsiveness. Target ~6–9 min. (If even faster scale-up is needed later, see F4.)

### F3. Make the in-pod queue predictable (fairness + bounded wait)
Keep the single execution lock, but tighten it for production load:
- **FIFO fairness:** `threading.Lock` is not guaranteed FIFO; under load a request can be starved.
  Consider a single-worker queue (`queue.Queue` + one dedicated GPU worker thread) so requests are
  served strictly in arrival order with predictable wait = (queue_depth × op_time).
- **Keep only GPU work under the lock.** Downloads, PIL resize, JPEG/PNG encode, Azure upload must run
  **outside** `coordinator.run(...)` (they already are in `tryon.py`/`user_validation.py` — verify and
  keep it that way) so the GPU isn't idle-held during I/O.
- **One tryon = two lock acquisitions** (Qwen, then SeedVR2) with a gap where another request can
  interleave. That's good for *fairness* but means a tryon's tail latency can include a wait between
  its two stages. Decide policy: interleave (max fairness/throughput) vs hold-GPU-for-whole-tryon
  (predictable single-request latency). Default: keep interleaving; revisit only if p99 suffers.
- Bounded queue already exists (`system_queue_max_size`, default 8, with QueueFull/Timeout) — size it
  to your latency SLO: max wait ≈ `max_queue_size × per_op_seconds`. With autoscaling, keep the queue
  SHORT and let the autoscaler add pods when `waiting_jobs` rises (scale signal).

### F4. Structural choice: replicated monolith vs split services
- **Option 1 — replicate the monolith (RECOMMENDED to start).** Each pod runs validation+tryon+upscale,
  identical, fully warm, behind a LB; scale by adding pods. Pros: simplest; in-process tryon→upscale
  (no network hop / image transfer). Cons: each pod loads ALL models (~96 GB, full warm) and scales
  coarsely; you scale the whole bundle even if only one stage is hot.
- **Option 2 — split into dedicated fleets (do later, if a stage becomes the bottleneck).** Separate
  Qwen-tryon pods, SeedVR2-upscale pods, validation/detection pods; each scales independently and warms
  faster (SeedVR2 alone warms in ~1–2 min vs the 8-min Qwen stack). Cons: tryon→upscale becomes an HTTP
  hop with a large-image transfer (latency + storage + retry/backpressure complexity).
- **Recommendation:** ship Option 1 (readiness-gated, fast-start monolith replicas) now; it satisfies
  "fully warm + first-user fast + autoscale in parallel." Move the **upscale** stage to its own fleet
  (Option 2, partial) only if upscale demand decouples from tryon demand or upscale warm-time/scaling
  needs to be independent. Keep the SeedVR2 client generic so it can be lifted into its own service
  later without touching the tryon contract.

### F5. Validation: optionally enforce exact 832×1248 (owner-approved "if needed")
`_normalize_user_image` (user_validation.py) currently scales the **long edge to 1248** but does **not**
crop to 2:3 — so the exact 832×1248 invariant depends on the frontend sending true 2:3. To make it
bulletproof (and guarantee the prewarmed fast path), add a **center-crop to exact 2:3 then resize to
exactly 832×1248** in `_normalize_user_image` only. **Do NOT touch the tryon path** — tryon keeps
trusting the validated image (owner's explicit requirement). This guarantees every tryon input is
exactly 832×1248 → upscale output always exactly 1820×2730 → always the prewarmed graph → never a
recompile. Small, isolated, in the validation service only.

### F6. Implementation order for Part F
1. **F1 readiness gating** (biggest UX win; small, safe) — `prewarmed` flag + `/ready` route + point
   the LB/autoscaler at `/ready`.
2. **F5 validation 832×1248 enforcement** (small, isolated, removes the only shape-drift risk).
3. **F2 = Part A fast start** (fp8 persistence, etc.) to shrink scale-up latency.
4. **F3 fairness/queue tuning** under real load.
5. **F4 Option-2 split** only if/when a stage's scaling decouples.

---

## G. Parallel startup orchestration (OOM-safe) — PLAN

Goal: warm everything in parallel to cut cold start, **without** coincident VRAM peaks causing OOM.
Today warmup is strictly sequential: Qwen (~7 min) → MiniCPM (~2.5 min) → detectors → SeedVR2 prewarm
(~3–5 min) ≈ 13–15 min. The tasks are independent (they only share the GPU), so they can overlap.

### G1. Resource model per warmup task (what each actually consumes)
| Task | Network read | GPU compute | VRAM peak (est.) | Where it runs |
|---|---|---|---|---|
| Qwen `from_pretrained` (bf16) + `.to(cuda)` | **54 GB** (I/O-bound, GPU ~idle) | low | **~54–64 GB transient** (bf16 in flight) | main thread |
| Qwen `quantize_` → fp8 | – | 0.5 s | brief bf16+fp8 overlap, then settles | main thread |
| Qwen `compile_repeated_blocks` + warm gens | – | **high** (~2 min) | ~40 GB resident | main thread |
| MiniCPM-V (vLLM) load + engine init | 8 GB | medium | ~8 GB | **separate subprocess** |
| Fashion detector + SigLIP | small | low | ~2–3 GB | main thread |
| SeedVR2 load + prewarm compile (2730) | 3 GB | **high** (inductor, ~108–294 s) | **~24 GB** (compile workspace) | daemon thread |

Two key facts:
1. The shared MooseFS volume is **~700 MB/s total** — parallel *reads* don't speed up I/O (same bytes
   over one pipe). The win is overlapping **GPU work with another task's I/O-wait**, and overlapping
   the two long **GPU-compute** phases (Qwen warm ≈ SeedVR2 compile).
2. The **only dangerous coincidence** is Qwen's bf16 **load/quantize peak (~54–64 GB)** landing at the
   same instant as SeedVR2's **compile peak (~24 GB)** → ~78–88 GB, and transient quantize spikes
   could push toward the 96 GB ceiling. (The earlier OOM was concurrent prewarm during Qwen load,
   with the *old* 34 GB SeedVR2 peak; the resident fix lowered it to ~24 GB, but we still must avoid
   stacking the two peaks.)

### G2. Strategy — overlap by complementary resource, stagger the one peak collision
- **Launch all three at t0:** MiniCPM subprocess (independent), Qwen warmup (main thread), SeedVR2
  prewarm thread. SeedVR2 + Qwen *model loads* (I/O) and MiniCPM proceed together.
- **Gate SeedVR2's COMPILE on free-VRAM headroom** (the elegant, decoupled part): before compiling,
  the SeedVR2 prewarm thread waits until `torch.cuda.mem_get_info()` free ≥ a configurable headroom
  (e.g. **30 GB**), polling every ~2 s with a timeout fallback (e.g. 600 s → compile anyway). This
  **auto-staggers** without coupling to Qwen internals: while Qwen is mid-load/quantize (low free
  VRAM) SeedVR2 waits; the moment Qwen settles (bf16 freed after quantize, resident ~40 GB) free VRAM
  rises past 30 GB and SeedVR2 compiles — overlapping Qwen's compile+warm phase. Worst-case coincident
  peak ≈ 40 (Qwen) + 24 (SeedVR2) + 8 (MiniCPM) ≈ **72 GB < 96 GB**.
- Why a dynamic headroom gate (not a fixed sleep or a Qwen-internal signal): it self-adapts to
  whatever the real memory curve is, needs no hook into the Qwen engine, and is safe across model/size
  changes.

### G3. Implementation design (small, isolated, reversible)
- **Orchestrator:** extend `warmup_resident_runtimes` (`app/runtime/warmup.py`) to start tasks
  concurrently instead of calling them in series:
  - start `warmup_upscale_runtime` **first** (it only kicks the daemon prewarm thread — non-blocking),
  - start MiniCPM warmup (already a subprocess; don't block on it — join at the end),
  - run Qwen wardrobe+tryon warmup on the main thread,
  - join MiniCPM at the end so `lifespan` only returns once liveness deps are up. (SeedVR2 stays async;
    `/ready` gates on its `prewarmed` flag — already implemented.)
- **VRAM headroom gate in `SeedVR2Client._prewarm`:** before the compile loop, call a helper
  `_wait_for_vram_headroom(min_free_gb, timeout_s)` that polls `torch.cuda.mem_get_info()`. Config:
  `UPSCALE_PREWARM_MIN_FREE_GB` (default 30), `UPSCALE_PREWARM_WAIT_TIMEOUT_S` (default 600). Gate is a
  no-op if CUDA unavailable.
- **Error isolation (your "no errors should stop the server"):** each parallel task wrapped in
  try/except. Non-fatal tasks (SeedVR2 prewarm, detectors) log + continue on failure (prewarm already
  does; `/ready` simply stays false if it never warms). Qwen/MiniCPM failure remains fatal (raise) —
  a pod that can't serve tryon should fail its boot and be recycled, not serve broken.
- **Concurrency safety:** warmups touch different models; the only shared global is
  `torch.backends.cudnn.benchmark` / the OptimizedModule patch (idempotent). No new locks needed; the
  request-time system coordinator is untouched.
- **Feature flag:** `STARTUP_PARALLEL_WARMUP=1` (default on after validation; set 0 to fall back to the
  current sequential order instantly).

### G4. Expected result & floor
- Overlapping the two ~3–5 min GPU phases (Qwen warm ∥ SeedVR2 compile) and MiniCPM:
  **~13–15 min → ~9–11 min** cold `/ready`.
- The **floor** stays the serial, unavoidable parts: Qwen 54 GB network load (~156 s) + imports +
  MiniCPM load. To go below ~9 min you'd need the load-trim (fp8-file / image-baked models, Part A0)
  and/or AOTInductor — separate, later.

### G5. Validation (mandatory before shipping — OOM history)
1. Implement behind `STARTUP_PARALLEL_WARMUP=1`.
2. Boot once while sampling `nvidia-smi --query-gpu=memory.used` every 1 s through the whole warmup;
   record **peak**. Pass if peak < ~90 GB with no CUDA OOM in logs.
3. Confirm `/ready` flips to 200 and the cold time dropped vs the 13–15 min baseline.
4. Confirm tryon + upscale latency unchanged (2.4 s upscale, normal tryon) — parallelism must not
   change steady-state behavior.
5. If peak is unsafe: raise `UPSCALE_PREWARM_MIN_FREE_GB` (more conservative stagger) or flip
   `STARTUP_PARALLEL_WARMUP=0` (full sequential) — instant rollback.

### G6. Order of work
1. VRAM headroom gate in `_prewarm` (safe even with current sequential order).
2. Parallel orchestrator in `warmup_resident_runtimes` behind the flag.
3. Monitored boot (G5). Tune `min_free_gb`. Ship if green.

### G7. ATTEMPT #1 RESULT (2026-06-10) — FAILED on import order, NOT memory. Reverted.
First monitored boot with `STARTUP_PARALLEL_WARMUP=1` **crash-looped**. Root cause was **not** OOM —
the VRAM design held (peak only ~60 GB). The killer was **import order**: starting the upscale
prewarm *first* imports the SeedVR2 CLI stack **before** Qwen, which poisons the `bitsandbytes` import
that Qwen's PEFT LoRA loading needs:
```
warmup_wardrobe -> _ensure_lora -> load_lora_weights -> peft -> import bnb ->
AttributeError: module 'bitsandbytes' has no attribute 'nn'  -> startup fails -> segfault -> loop
```
So the SeedVR2/ComfyUI stack and `bitsandbytes`/PEFT have an **import-order coupling**: SeedVR2 must
not be imported before Qwen's LoRAs load. Reverted to `STARTUP_PARALLEL_WARMUP=0` (sequential, proven)
→ service restored. The parallel code stays, flag-gated OFF.

**v2 options (if we revisit — the memory design is fine, only import order needs fixing):**
- **A (targeted):** force a clean `import bitsandbytes; import bitsandbytes.nn; import peft` at app
  startup *before* any SeedVR2 import, so SeedVR2 loading first can't leave bnb half-initialized. Then
  the parallel order is safe. Small; needs a monitored-boot spike to confirm it actually fixes it.
- **B (structural):** keep Qwen warmup first (pipeline+LoRA load → bnb imported cleanly), then start
  the SeedVR2 prewarm to overlap only Qwen's *compile+warm* phase. Needs hooking after Qwen's
  lora-load (more invasive).

**Recommendation:** since readiness gating already hides cold-boot from users, the ~3–5 min saving is
"nice to have," and parallel proved fragile. Bank the validated Batch-1 wins; pursue parallel v2
(Option A) only if cold-boot latency becomes a real autoscaling cost. The VRAM-headroom gate
(`_wait_for_vram_headroom`) is harmless and stays for when/if v2 is enabled.

---

## E. Open questions to resolve during implementation
- A2: does `torchao` in this venv round-trip a saved fp8 state dict cleanly? (spike on the pod.)
- A4: is `/workspace` actually the load bottleneck? (A1 + a raw read test answers it.)
- A6: can SeedVR2 prewarm overlap Qwen warm within 96 GB, or must it stagger? (memory probe.)
- B2: real warm tryon `seedvr2_upscale_seconds` — confirms B1 vs a deeper issue.

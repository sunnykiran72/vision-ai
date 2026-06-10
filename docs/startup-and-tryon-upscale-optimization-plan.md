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

## E. Open questions to resolve during implementation
- A2: does `torchao` in this venv round-trip a saved fp8 state dict cleanly? (spike on the pod.)
- A4: is `/workspace` actually the load bottleneck? (A1 + a raw read test answers it.)
- A6: can SeedVR2 prewarm overlap Qwen warm within 96 GB, or must it stagger? (memory probe.)
- B2: real warm tryon `seedvr2_upscale_seconds` — confirms B1 vs a deeper issue.

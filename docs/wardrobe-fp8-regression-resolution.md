# Wardrobe FP8 Regression — Root Cause & Resolution Plan

> For a fresh session. The `fp8_version` branch made `/v1/wardrobe` **slower** (~10s → ~20s), not
> faster. This doc explains exactly why, with evidence, and gives the fix paths in priority order.

---

## 1. Symptom

`/v1/wardrobe`, 4 identical requests on the pod (Pod A `dt3jjdcekx1lvl`, RTX PRO 6000 Blackwell), warm:

| Run | Total | Detector | MiniCPM | **Qwen** | Upload |
|---:|---:|---:|---:|---:|---:|
| 1 | 35.96s | 4.66s (warmup) | 0.43s | **30.34s** | 0.50s |
| 2 | 20.41s | 0.05s | 0.29s | **19.76s** | 0.29s |
| 3 | 18.80s | 0.04s | 0.28s | **18.19s** | 0.27s |
| 4 | 23.07s | 0.04s | 0.28s | **22.47s** | 0.26s |

Detector/MiniCPM/upload are fine after warmup. **Qwen generation is the bottleneck at 18–22s.**
`qwen_generation_queued_wall_seconds == qwen_generation_seconds` → not queue wait, it's real generation.

This is a **regression**: the pre-fp8 bf16 wardrobe was ~10s. Enabling fp8 made it ~2× slower.

---

## 2. Root cause (confirmed)

**FP8 is running in EAGER mode at inference — the `torch.compile` is not taking effect — and eager
fp8 is slower than bf16.**

### Evidence
- **Per-step math:** steps = 12 (current API). `~20s / 12 ≈ 1.67s/step`. That is the **eager fp8**
  rate (measured earlier: fp8 no-compile ≈ 1.3–1.5s/step). **Compiled fp8 is ~0.3–0.4s/step → ~5s**.
  So inference is eager, not compiled.
- The warmup log *does* say `Compiled Qwen transformer blocks (dynamic=False): 60 blocks` and
  `quantization=torchao_float8...`, so quantize + a compile call both ran — **but the compiled
  graph isn't being used (or isn't fusing) at request time.**
- Benchmark reference (the ~6.5s number that motivated fp8) was produced with a **FUSED single
  LoRA + `pipe.transformer.compile_repeated_blocks(fullgraph=False)`** — i.e. **no PEFT adapters**.

### Why the compile doesn't fuse here
The prod engine (`app/clients/qwen_diffusers_engine.py`) — for multi-category support — uses:
- **Unfused PEFT LoRA adapters** (`load_lora_weights(..., adapter_name=...)`, switched per request
  via `set_adapters`), and
- a **custom per-block loop** `blocks[i] = torch.compile(block, dynamic=False)` (not diffusers'
  `compile_repeated_blocks`).

The most likely failure (to confirm — see §4): **unfused PEFT LoRA adapter layers wrap the
torchao-quantized `Linear`, so torch.compile graph-breaks / falls back around the adapter dispatch
and the fp8 GEMM never gets fused.** torchao fp8 only beats bf16 *when the fp8 matmul is fused by
the compiler*; with the adapter wrapper in the way it stays eager → ~1.67s/step.

### The principle that was violated
> **FP8 WITHOUT effective compile is SLOWER than bf16** (≈20s vs ≈10s). FP8 is only a win when the
> compiler fuses the fp8 GEMM, and that fusion needs **fused weights (no live PEFT adapters)**.

So enabling `QWEN_FP8=1` on top of the **adapter-based** multi-category engine produced the
worst case: fp8 overhead, no fp8 fusion.

### The fundamental tension
| Need | Requires |
|---|---|
| Multi-category in one resident model | **Unfused adapters** + `set_adapters` per request |
| Fast fp8 (fused GEMM via compile) | **Fused weights** (no live adapters) |

These two pull in opposite directions. You cannot have both with a single shared adapter-based model.

---

## 3. Resolution options (priority order)

### Option A — IMMEDIATE un-regress: turn fp8 OFF for wardrobe (bf16)
Set `QWEN_FP8=0` (keep bf16). Wardrobe returns to **~10s** (bf16, adapters work fine; bf16 doesn't
need compile to be reasonable). This is the safe rollback — **do this first** so prod isn't at 20s.
- Cost: loses the fp8 VRAM saving (back to ~62GB) and the hoped-for speedup.
- Risk: none — this is the known-good pre-fp8 behavior.

### Option B — REAL fp8 speedup: fused-per-category models (~6.5s)
Build the fp8 path the way the benchmark proved fast: **fuse each category's LoRA into the base,
then quantize + compile — one model per category** (top/bottom/dress). No live adapters → fp8 fuses.
- Per request: route to the matching pre-built fp8+compiled model.
- VRAM: 3 fp8 transformers (~10GB each) + shared text-encoder/VAE ≈ ~40–50GB → fits in 96GB. Verify.
- Startup: 3× (fuse→GPU-quantize→compile→prewarm). Slow one-time on Pod A (~per category minutes).
- This is the **only known path to ~6.5s fp8 with multiple categories**. Most work, best result.

### Option C — Make adapter+fp8+compile actually fuse (investigate, uncertain)
Try, in order, measuring per-step each time (eager ≈1.6s vs fused ≈0.4s):
1. Replace the custom per-block loop with **`pipe.transformer.compile_repeated_blocks(fullgraph=False)`**
   (the path the benchmark used) — see if it fuses even with adapters.
2. Try **fusing the active adapter at warmup** for a single category and compiling, to confirm fused
   fp8 actually hits ~0.4s/step on this engine (isolate adapter-vs-fusion).
3. Check torchao + PEFT + torch.compile compatibility (graph-break logs:
   `TORCH_LOGS=graph_breaks,recompiles`). If PEFT breaks fusion, Option C is a dead end → use B.

---

## 4. Diagnostic steps for the new chat (do these to confirm before changing)

All read-only / quick. Pod A, SSH key per the active RunPod Connect tab. Use
`nohup setsid` for any server, bracket-grep `"[q]wen"`, and check GPU is free before launching.

1. **Confirm eager vs compiled per-step.** From the wardrobe response timings: `qwen_generation_seconds / steps`.
   ~1.6s/step = eager (bug present). ~0.4s/step = compiled (then the problem is elsewhere).
2. **A/B fp8 vs bf16 latency** with one quick toggle: restart the API with `QWEN_FP8=0` and re-run
   `/v1/wardrobe`. If bf16 ≈10s < fp8 ≈20s, that *proves* fp8-eager is the regression → ship Option A now.
3. **Isolate adapter vs fusion** (Option C #2): in a standalone script, load ONE wardrobe LoRA,
   `fuse_lora()`, `quantize_`, `compile_repeated_blocks`, generate at 12 steps. If that hits
   ~5–6s, fp8 fusion works without adapters → confirms adapters are the blocker → Option B is correct.
4. Capture `TORCH_LOGS=graph_breaks,recompiles` during one fp8 generation to see if the adapter
   path graph-breaks.

---

## 5. Recommendation

1. **Now:** Option A — `QWEN_FP8=0`, restore ~10s wardrobe (un-regress prod). Wardrobe was never the
   place fp8 paid off, given the adapter requirement.
2. **Then:** decide if ~6.5s is worth Option B (fused-per-category) — meaningful VRAM/complexity, the
   only way to get fp8's win with multiple categories.
3. Keep **SeedVR2 3B fp8 + compile** as is — it's a *single* model (no per-request adapter swap), so
   its fp8+compile fuses correctly (~3.6s @3072). The fp8-eager problem is **specific to the
   adapter-based Qwen wardrobe engine**, not SeedVR2.

---

## 6. Key files
- `app/clients/qwen_diffusers_engine.py` — `_ensure_fp8_quantized`, `_ensure_compiled` (the custom
  per-block `torch.compile` loop), `_ensure_lora` (adapter load), `set_adapters` per request.
- `app/config.py` — `qwen_fp8`, `qwen_compile`, the `qwen_fp8 → qwen_compile` guard.
- `app/runtime/wardrobe_runtime.py` — runner cache key includes `qwen_fp8`.
- `.env` on pod — `QWEN_FP8=true`, `QWEN_COMPILE=true`, `QWEN_IMAGE_EDIT_DTYPE=bfloat16`, steps=12.
- Benchmarks/recipe context: `docs/inference-optimization-seedvr2-qwen-fp8.md` (§2 — note its recipe
  used **fused single LoRA**, which is why it hit 6.5s and prod does not).

## 7. One-line summary
**Wardrobe regressed because fp8 was enabled on the adapter-based multi-LoRA engine, where
`torch.compile` can't fuse the fp8 GEMM → fp8 runs eager (~1.67s/step) which is slower than bf16.
Fix: turn fp8 off for wardrobe now (Option A, ~10s), or go fused-per-category for real fp8 (~6.5s).**

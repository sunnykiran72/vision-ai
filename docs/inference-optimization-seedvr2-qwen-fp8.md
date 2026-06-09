# Inference Optimization — SeedVR2 (3B upscaler) & Qwen-Image-Edit-2511 (FP8 wardrobe)

> Self-contained handoff doc. Captures everything done, the measured numbers, the gotchas, the
> working configs, file locations, and open items. Written so a fresh session can act without
> re-deriving context.

---

## 0. Environment & hardware (read first)

**Two RunPod pods, both RTX PRO 6000 Blackwell (sm_120), sharing ONE network volume `/workspace`:**

| Pod | ID | Edition | Notes |
|---|---|---|---|
| **A** | `dt3jjdcekx1lvl` | **Server Edition** | ~3× slower CPU (model-prep/compile slow). Current Qwen lab runs here. |
| **B** | `ra4nfsl1lrajm5` | **Workstation Edition** | Faster; was used for SeedVR2 + Qwen benchmarks. Had a wedged GPU process at end of session. |

- IPs/ports are **RunPod-ephemeral** (change on restart). Use the key shown in the active RunPod Connect tab. The 2026-06-09 Pod A screenshot shows **`~/.ssh/id_ed25519`**; older notes/scripts may still reference **`~/.ssh/runpod_qge_ed25519`**.
- Software (the venv `/workspace/.venvs/glamify-image-ai`): **Python 3.12, torch 2.11.0+cu130 (CUDA 13), diffusers 0.38.0, transformers 4.57.6, torchao 0.17, cache-dit 1.3.11**. The venv has **no `pip`** — use `uv pip install --python /workspace/.venvs/glamify-image-ai/bin/python <pkg>`.
- Repo on volume: `/workspace/glamify-image-ai`. Models: `/workspace/models/`. LoRAs: `/workspace/loras/wardrobe/`.
- **`hf` CLI + `HF_HUB_ENABLE_HF_TRANSFER=1`** are installed → fast HF downloads.

### SSH survival rules (important — caused most of the pain)
- **SSH drops with exit 255 frequently.** Long-running poll loops over one SSH connection die. Mitigations:
  - Add `-o ServerAliveInterval=15 -o ServerAliveCountMax=2000` to keep long polls alive.
  - **Run servers detached: `nohup setsid bash script.sh > log 2>&1 < /dev/null & disown`** — otherwise an SSH drop kills the child before it detaches.
- **`grep`/`pkill` self-match trap:** `pkill -f qwen_lab_server` matches *your own SSH command line* and kills it (looks like "no output" / phantom "2 servers"). Always use the **bracket trick**: `grep "[q]wen_lab_server.py"`, and kill by PID filtered to `/bin/python` (so your `bash -c` shell is excluded).
- Stuck CUDA processes can be `D`-state (uninterruptible) and survive `kill -9`; only a pod restart clears them. Always confirm GPU is actually free (`nvidia-smi --query-gpu=memory.used`) before launching, to avoid OOM from zombies.

---

## 1. SeedVR2 — 3B upscaler

### 1.1 What it is / where
- ComfyUI-SeedVR2 CLI at `/workspace/seedvr2_eval/ComfyUI-SeedVR2_VideoUpscaler/inference_cli.py` (numz build, v2.5.23).
- Our client: `app/clients/seedvr2.py` (calls the CLI's `process_single_file`/`parse_arguments` in-process).
- Prod path: `app/services/upscale.py` (`/v1/upscale`), presets `metric 2k→2048`, `4k→4096` (long edge). The client already does **arbitrary scaling** (`--resolution <short_edge>` + `--max_resolution <long_edge>`), not fixed 2×.
- **A/B lab** (committed): `app/routes/upscale_lab.py`, `app/services/upscale_lab.py`, `app/constants/upscale.py`, `tools/upscale_lab.html` → `GET /tools/upscale-lab` (unauthenticated, under public `/tools` prefix). Commits `d852de7`, `5e2bf22` on branch `wardrobe-diffusers-inference`.

### 1.2 Variant facts (authoritative = `src.utils.model_registry.get_available_dit_models()`)
The CLI's `--dit_model` has a FIXED choices list. **Only `_mixed_block35_fp16` 7B FP8 builds are valid** — the "pure" `seedvr2_ema_7b_fp8_e4m3fn` is NOT accepted (a bad name → `sys.exit()` → `SystemExit`, which `except Exception` does NOT catch → killed the worker; the lab route now guards `SystemExit`).

| File | Repo | Notes |
|---|---|---|
| `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | `numz/SeedVR2_comfyUI` | 3B FP8 |
| `seedvr2_ema_3b_fp16.safetensors` | numz | 3B FP16 |
| `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors` | **`AInVFX/SeedVR2_comfyUI`** | **7B FP8 — current PROD** |
| `seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors` | AInVFX | 7B "sharp" FP8 |
| `seedvr2_ema_7b_fp16.safetensors` / `_sharp_fp16` | numz | FP16 ceilings |
| GGUF Q4_K_M / Q8_0 (3B/7B/sharp) | AInVFX | quantized |

**Prod uses 7B mixed (NOT sharp).** Models on pod: `/workspace/models/seedvr2/`. `UPSCALE_MODEL_PATH=/workspace/models/seedvr2`.

### 1.3 CRITICAL Blackwell nvrtc fix (affects PROD)
SeedVR2 JIT-compiles a reduction kernel via nvrtc at inference; on torch 2.11+cu130/sm_120 it needs `libnvrtc-builtins.so.13.0`, which exists at `<venv>/lib/python3.12/site-packages/nvidia/cu13/lib/` but is **not on the loader path**. Symptom: `nvrtc: error: failed to open libnvrtc-builtins.so.13.0` → first real upscale crashes.
**Fix (must be in the prod launch env):**
```
export LD_LIBRARY_PATH=/workspace/.venvs/glamify-image-ai/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

### 1.4 torch.compile fix (numz issue #502) — committed
torch.compile of the DiT wraps it in `OptimizedModule`, which has no `__bool__`; numz code does `if model:`/`len(...)` truthiness checks → `TypeError: CompatibleDiT does not support len()` on cross-request reuse. **One-line universal fix** (in `seedvr2.py._load_module`): make compiled modules truthy.
```python
from torch._dynamo.eval_frame import OptimizedModule
if not hasattr(OptimizedModule, "__bool__"):
    OptimizedModule.__bool__ = lambda self: True
```
Client also sets `torch.backends.cudnn.benchmark = True`, adds `--compile_dit/--compile_vae` gated by env `UPSCALE_COMPILE` (default on), and bumped `get_seedvr2_client` lru_cache to `maxsize=4` (keep 3B+7B resident, no recompile on switch).

### 1.5 Measured latency (warm, steady-state)
- **3B FP8 @ 3072 (2048×3072), compiled = ~3.6s** ← lossless floor on this GPU. Uncompiled 5.7s.
- 3B @ 2048 ≈ 2.6s · 3B @ 2560 ≈ 2.5s · 3B @ 1536 ≈ 2s.
- 7B mixed @ 2048 ≈ 6–7s · @ 1536 ≈ 5.9s. (7B is ~2× the DiT cost; VAE is shared.)
- **3B is ~2× faster than 7B** and the community + our tests prefer 3B's look (7B undertrained / plastic).
- Latency scales ~**linearly with output pixel count**.

### 1.6 What was tested for going below 3.6s (lossless) — all exhausted
- `compile_mode` default = `max-autotune` = `reduce-overhead` → **all ~3.6s** (compute-bound autotune can't beat default; VAE is the wall).
- `tensor_offload_device none` → **slower** than `cpu`; reverted.
- GPU clock-lock (`nvidia-smi -lgc`) → **blocked** (no permission on RunPod).
- **The VAE decode is memory-bandwidth-bound (~70% of time at 3072).** No lossless lever moves it. Profiled: at 3072, VAE encode 1.55s + DiT 3.64s(cold) + VAE decode 2.30s.
- **Conclusion: ~3.5s is the lossless floor at 3072 on this GPU.** Below that needs: lower res (2560→~2.5s), lossy precision (FP4 — blocked, needs nvcc, quality risk), or HBM GPU (H200 → ~2s; the VAE is bandwidth-bound so HBM ~2.7× helps).

### 1.7 Recommendation (SeedVR2)
Ship **3B FP8 @ target resolution + torch.compile**, with: `LD_LIBRARY_PATH` (nvrtc), `UPSCALE_COMPILE=1`, and **startup pre-warm at the exact prod resolution** (compile recompiles per output shape; first request per shape pays ~2 min). For ~2.5s, use out-edge ~2560.

---

## 2. Qwen-Image-Edit-2511 — FP8 wardrobe extraction

### 2.1 What it is / where
- Model: `/workspace/models/qwen-image-edit-2511` (transformer = 39 GB bf16, ~20B params).
- Prod engine: `app/clients/qwen_diffusers_engine.py` → `QwenImageEditPlusPipeline`, **bf16**, swaps 1 of 3 category LoRAs per request.
- LoRAs: `/workspace/loras/wardrobe/{top_23000,bottom_30000,dress_27000}.safetensors`.
- Generation: **seed 7777, steps 10 (prod), `true_cfg_scale=1.0` (NO CFG → 1 forward/step), output 832×1248.** Extraction prompt templates in `app/constants/wardrobe.py` (`GlamTopExt`/`GlamBtmExt`/`GlamDressExt`).
- **Interactive lab (standalone):** `tools/qwen_lab_server.py` + `tools/qwen_lab.html` → `GET /tools/qwen-lab`. Inputs: image, prompt, seed, steps, output W×H, cache dropdown (0 / 0.2). Returns output image + metrics (total/encode/denoise/decode/per-step/VRAM) + history.

### 2.2 The working FP8 recipe
```python
pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(LORA); pipe.fuse_lora(lora_scale=1.0)   # fuse BEFORE quantize
pipe.to("cuda")                                                # GPU FIRST (see gotcha 2.4.5)
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
# optional cache (see 2.4.3):
# import cache_dit; from cache_dit import DBCacheConfig
# cache_dit.enable_cache(pipe, cache_config=DBCacheConfig(residual_diff_threshold=0.2,
#                        max_warmup_steps=2, enable_separate_cfg=False))
pipe.transformer = torch.compile(pipe.transformer)            # full compile (no cache)
# OR with cache:  pipe.transformer.compile_repeated_blocks(fullgraph=False)
```

### 2.3 Measured latency (15 steps, top LoRA fused, 832×1248)
| Config | Latency | VRAM | Quality vs bf16 |
|---|---|---|---|
| bf16 no-compile | 10.1s | 62 GB | reference |
| bf16 + compile | 8.9s | 62 GB | — |
| **fp8 no-compile** | **20.4s** ⚠️ | 42 GB | — (fp8 is SLOWER without compile) |
| **fp8 + compile** | **6.5s** | **42 GB** | PSNR 28–34, visually identical |
| **fp8 + compile, cache 0** | **~6.5s** | **42 GB** | best current production candidate |
| fp8 + compile + cache 0.20 | ~4–6s, but unstable | 42 GB | cache behavior is not reliable enough for prod |
| fp8 + compile + cache 0.25 | ~3.8–4.2s, but unstable | 42 GB | higher quality risk |

- **fp8+compile with cache 0 @15 steps ≈ same latency as current bf16@10 steps (~6.6s) → more accuracy, −33% VRAM, and deterministic latency.**
- Latency is **linear in steps** (~0.43s/step compiled): 8→4.0s, 10→4.5s, 12→5.4s, 15→6.7s (uncompiled-cache numbers; compiled is lower). Pipeline profile: 95% is the DiT denoise loop; VAE+encoder are negligible (~0.3s).
- All 3 LoRAs (top/bottom/dress) load+fuse+quantize fine under fp8. Top has the strongest visual validation so far; bottom/dress still need final quality sign-off. Benchmark outputs live at `/workspace/qwen_bench_out/*.png`.

### 2.4 CRITICAL gotchas
1. **fp8 only beats bf16 WITH `torch.compile`.** Uncompiled fp8 = 20s (quant/dequant overhead unfused). The compile fuses the fp8 GEMM (~0.3s/step vs ~1.5s/step).
2. **diffusers `TorchAoConfig("float8dq")` (string) FAILS** on diffusers 0.38 (`quant_type must be an AOBaseConfig instance`). Use **`torchao.quantize_` directly**.
3. **cache-dit + FULL `torch.compile` crashes** (`InternalTorchDynamoError: Polyfill handler __eq__ ... not traceable`, torch 2.11). Workaround: **`pipe.transformer.compile_repeated_blocks(fullgraph=False)`** instead of `torch.compile(pipe.transformer)`.
4. **cache-dit + compile recompiles on cache-pattern change.** cache-dit's step-skipping is data-dependent; with compile it can recompile (~65s first time / intermittent ~12s spikes). **Mitigation: FIXED input dims** (lab resizes every input to 768×1024) helps, but does not fully remove spikes. **No-cache + compile is fully deterministic (~6.5–6.9s, no spikes) and is now the recommended production baseline.**
5. **Quantize on GPU, not CPU.** `quantize_` on a CPU model = ~15 min on Pod A (slow CPU); `pipe.to("cuda")` *before* `quantize_` → ~1 min. Peak VRAM during quantize fits in 96 GB.
6. **`enable_separate_cfg=False`** in DBCacheConfig (because `true_cfg_scale=1.0` = no separate CFG pass). The auto-default `True` mis-aligns the cache.
7. Compile recompiles per **output dimension** too → pre-warm at the prod output size.

### 2.5 Variant / approach choices (and why)
- **Use torchao runtime fp8 on OUR base** — most faithful to the LoRA training, works with the existing pipeline.
- **AVOID pre-quantized fp8 checkpoints** (`1038lab/Qwen-Image-Edit-2511-FP8`, `xms991/...`): they have **baked-in LoRAs / modified base** → conflict with our extraction LoRAs.
- **AVOID Lightning distill** (`lightx2v/Qwen-Image-Edit-2511-Lightning`, 4-step): the LoRAs were trained for normal multi-step sampling; Lightning changes step dynamics → degrades extraction. (Its fp8 is also reportedly broken.)
- **vLLM does NOT help here:** quantization is **not supported for Qwen-Image-*Edit*** in vLLM (only base Qwen-Image), and on a single GPU it offers nothing beyond cache-dit + compile (which we already have). Its value is multi-GPU parallelism / throughput batching.

### 2.6 Sub-3s analysis
At 15 steps, ~6.5s with cache 0 is the dependable floor on the current single RTX PRO 6000 Blackwell setup. Below that needs a real tradeoff: **TaylorSeer calibrator** (cache-dit feature, predicts skipped steps — best shot at ~3–3.5s near-lossless, UNTESTED), **FP4/int4-style quantization** (quality/kernel risk), **fewer steps** (user rejected — 10 steps had quality issues, 15 is the quality bar), or **faster/more GPUs**.

### 2.7 Lab server config (current, Pod A)
Launch script `/workspace/run_qwen_lab.sh` env used for cache benchmarks: `CATEGORY=top CACHE_THR=0.2 CACHE_WARMUP=2 PORT=8000 LAB_COMPILE=1 LAB_CACHE=1 LD_LIBRARY_PATH=<cu13> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The checked-in lab server defaults to `LAB_CACHE=0` for deterministic no-cache runs; set `LAB_CACHE=1` to enable cache during warmup/benchmarking. It warms with a real sample image (2 warmups). URL: `https://<podid>-8000.proxy.runpod.net/tools/qwen-lab`. The lab loads only the `CATEGORY` LoRA at startup (one category at a time; switch via env + restart). The `cache` dropdown toggles per-request via `enable_cache`/`disable_cache` (guarded to skip if already at that threshold).

### 2.8 Recommendation (Qwen wardrobe)
For prod now: **fp8 + compile + cache 0**, 15 steps, ~6.5–6.9s, 42 GB (−33%). This is the best current balance because no-cache is generating the better/reliable image and avoids cache-dit's recompile/spike behavior. Requires: GPU-quantize, `LD_LIBRARY_PATH` (nvrtc), pre-warm at prod resolution (fixed dims), and per-category fp8 models (3 LoRAs can't dynamically swap on a quantized+compiled model — build/pre-warm one per category). Keep cache 0.20/0.25 as R&D only, not as the production path.

---

## 3. Current state at end of session
- **SeedVR2 lab**: committed to repo (`5e2bf22`); compile #502 fix + nvrtc fix + cudnn + maxsize in `app/clients/seedvr2.py`. ~3.6s @3072 reliable.
- **Qwen lab**: last known on **Pod A (`dt3jjdcekx1lvl`) port 8000**, `/tools/qwen-lab`. Deterministic mode (`LAB_CACHE=0`) is ~6.5s/image and is the preferred prod baseline. Cache benchmark mode (`LAB_CACHE=1`, cache 0.2) gives ~5–6s/image with occasional ~12s re-fire and should stay R&D-only. Files in repo: `tools/qwen_lab_server.py`, `tools/qwen_lab.html` (NOT yet committed — should be committed).
- Benchmark artifacts on Pod volume: `/workspace/qwen_bench_out/` (output PNGs, JSON, logs), `/workspace/seedvr2_eval/`, scripts `/workspace/qwen_*.py`.
- Pod B (`ra4nfsl1lrajm5`) has a wedged 60 GB GPU process — needs a pod restart to clear.

## 4. Open items / next steps
1. **Commit** this doc + `tools/qwen_lab_server.py` + `tools/qwen_lab.html` to the repo.
2. **Port to prod**: ensure the SeedVR2 prod launch includes the nvrtc `LD_LIBRARY_PATH`; keep the `SystemExit` and `OptimizedModule.__bool__` guards in the `/v1/upscale` path; integrate fp8 (torchao + compile + pre-warm) into `qwen_diffusers_engine.py` behind a flag.
3. **Qwen per-category fp8**: implement one warmed fp8+compiled no-cache model per category, or separate workers per category, so prod does not swap LoRAs on a quantized+compiled model.
4. **Untested wins**: TaylorSeer calibrator for Qwen (possible sub-3.5s near-lossless, but cache is currently unreliable); FP8 compile mode/block-compile variants; lower output size if product accepts it; SeedVR2 2560 out-edge for ~2.5s.
5. **Visual quality sign-off** on bottom/dress garments under fp8 (only `top` samples existed on the pod).

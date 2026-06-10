# FP8 Version Implementation Status

Branch: `fp8_version`

This document records the code changes made for the SeedVR2 3B FP8, Qwen fp8,
try-on inline upscale, and related runtime setup work. The branch has been synced to
RunPod pod `select_green_guineafowl` (`dt3jjdcekx1lvl`) for live testing.

## Scope

The current branch updates the API service code for three production paths:

- `/v1/upscale`
- `/v1/tryon`
- Shared Qwen Image Edit runtime used by wardrobe and try-on

It also updates the GPU stack install validation so the fp8 Qwen path has the runtime
dependency it needs.

## Upscale API Changes

Files:

- `app/models/upscale.py`
- `app/config.py`
- `app/constants/upscale.py`

Changes:

- `/v1/upscale` request default changed from `2k` to `4k`.
- Default SeedVR2 model variant changed from the 7B mixed FP8 model to:

```text
seedvr2_ema_3b_fp8_e4m3fn.safetensors
```

- The known variant label now marks the 3B FP8 model as the current production default.
- The 7B mixed FP8 variant remains supported, but is no longer the default.

SeedVR2 trial config kept:

- `UPSCALE_COMPILE` remains enabled by default in the SeedVR2 client path.
- `--compile_dit` and `--compile_vae` remain enabled unless `UPSCALE_COMPILE=0`.
- `--dit_offload_device none` is kept.
- `--vae_offload_device none` is kept.
- `--tensor_offload_device cpu` is kept because the benchmark doc says `none` was slower.
- Runner cache stays at `maxsize=4`, so switching variants can keep warmed runners.

## Try-On API Changes

Files:

- `app/constants/tryon.py`
- `app/config.py`
- `app/services/tryon.py`

New constants:

```text
UPSCALE_AFTER_QWEN = True
UPSCALE_TARGET_LONG_EDGE_PX = 3072
FINAL_OUTPUT_LONG_EDGE_PX = 2048
```

New env-backed settings:

```text
TRYON_UPSCALE_AFTER_QWEN
TRYON_UPSCALE_TARGET_LONG_EDGE
TRYON_FINAL_OUTPUT_LONG_EDGE
```

Flow after the change:

1. Download user image and garment images in parallel.
2. Build the garment reference image or collage.
3. Resize garment reference long edge to `768`.
4. Run Qwen try-on at the input user image dimensions.
5. Save Qwen output to a temporary image.
6. Run SeedVR2 inline to upscale the Qwen output to `3072` long edge.
7. Downscale the SeedVR2 output to `2048` long edge.
8. Upload the final JPEG to Azure.
9. Return metadata including Qwen size, upscaled size, final size, model variant, and timings.

Important size rule:

- Qwen inference still uses the exact user image dimensions.
- The final saved try-on output is now the post-SeedVR2 image with long edge `2048`.

Metadata added:

- `metadata.upscale.enabled`
- `metadata.upscale.mode`
- `metadata.upscale.model_variant`
- `metadata.upscale.runner_backend`
- `metadata.upscale.target_long_edge`
- `metadata.upscale.derived_short_edge`
- `metadata.upscale.qwen_output_size`
- `metadata.upscale.upscaled_size_before_downscale`
- `metadata.upscale.final_long_edge`
- `metadata.upscale.wall_seconds`
- `metadata.timings.seedvr2_upscale_seconds`
- `metadata.timings.seedvr2_upscale_wall_seconds`
- `metadata.output.qwen_width`
- `metadata.output.qwen_height`

## Qwen FP8 Runtime Changes

Files:

- `app/clients/qwen_diffusers_engine.py`
- `app/runtime/wardrobe_runtime.py`
- `app/config.py`
- `app/constants/wardrobe.py`

New env-backed setting:

```text
QWEN_FP8
```

Behavior:

- Normal mode remains `QWEN_IMAGE_EDIT_DTYPE=bfloat16`.
- `QWEN_FP8=1` enables torchao fp8 quantization.
- `QWEN_FP8=1` can run with or without `QWEN_COMPILE=1`.
- `QWEN_COMPILE=1` is only appropriate when the runtime path is known to reuse the same compiled
  graph. The multi-adapter API path can trigger repeated compile specialization, so the current
  live API test uses `QWEN_COMPILE=0`.
- The fp8 path does not pass `torch.float8_e4m3fn` as `torch_dtype`.
- Wardrobe Qwen extraction now uses `GENERATION_STEPS = 12`.
- The pipeline still loads in bf16 first, then quantizes the transformer with torchao:

```python
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
```

This matches the trial finding that Qwen fp8 is a quantization path, not a diffusers
load dtype.

Safety guard added:

- `QWEN_IMAGE_EDIT_DTYPE=fp8`, `float8`, or `float8_e4m3fn` now raises a clear error.
- The correct config is:

```text
QWEN_IMAGE_EDIT_DTYPE=bfloat16
QWEN_FP8=1
QWEN_COMPILE=0  # multi-adapter API path
```

LoRA handling:

- Before quantization, the engine loads configured wardrobe LoRAs.
- If `QWEN_FP8=1`, the engine also preloads configured try-on LoRAs before quantization.
- After fp8 quantization, loading a new LoRA is blocked with a clear runtime error.

Reason:

- The trial docs identified dynamic LoRA loading/swapping after quantized+compiled fp8 as
  a risky area. The branch makes the failure explicit instead of silently loading a late
  adapter into a quantized runtime.

Metadata added to Qwen outputs:

- `dtype`
- `quantization`
- `fp8`
- `compiled`

## GPU Stack Changes

Files:

- `scripts/install_gpu_stack.sh`
- `scripts/validate_gpu_stack.py`

Changes:

- `torchao>=0.14` added to the GPU stack install script.
- Validation now checks that torchao imports and exposes:

```python
Float8DynamicActivationFloat8WeightConfig
quantize_
```

Reason:

- Qwen fp8 requires torchao quantization. It cannot be validated by checking only PyTorch
  float8 dtype support.

## MiniCPM AWQ Runtime Fix

File:

- `app/clients/minicpm_vllm.py`

Change:

- The MiniCPM vLLM client now sets this default before importing vLLM:

```text
VLLM_USE_FLASHINFER_SAMPLER=0
```

Reason:

- vLLM `0.22.1` still defaults its top-k/top-p sampler to FlashInfer.
- On the current Blackwell RunPod stack, MiniCPM AWQ startup failed during vLLM profiling with:

```text
RuntimeError: FlashInfer requires GPUs with sm75 or higher
```

- The older `VLLM_USE_FLASHINFER=0` launch variable is not recognized by this vLLM build.
- The validated reference MiniCPM app used `VLLM_USE_FLASHINFER_SAMPLER=0`; this branch now
  applies that guard in code so the full API does not depend on a manual shell export.

## Tests Added Or Updated

Files:

- `tests/test_tryon_service.py`
- `tests/test_wardrobe_diffusers.py`
- `tests/test_wardrobe_runtime.py`

Coverage added:

- Try-on can still run with inline upscale disabled for existing behavior tests.
- Inline try-on upscale path verifies:
  - Qwen output starts at user dimensions.
  - SeedVR2 receives target long edge `3072`.
  - Final uploaded output is downscaled to long edge `2048`.
  - Metadata includes SeedVR2 model variant and output sizes.
- Qwen dtype tests now reject fp8 as a `torch_dtype`.
- Wardrobe runner cache includes `QWEN_FP8`, so bf16 and fp8 runtimes cannot accidentally
  reuse the same cached runner.

## Local Validation Completed

Commands run locally:

```bash
python3 -m py_compile app/services/tryon.py app/clients/qwen_diffusers_engine.py app/runtime/wardrobe_runtime.py app/config.py app/models/upscale.py app/constants/tryon.py app/constants/upscale.py scripts/validate_gpu_stack.py
uv run ruff check app/services/tryon.py app/clients/qwen_diffusers_engine.py app/runtime/wardrobe_runtime.py app/config.py app/models/upscale.py app/constants/tryon.py app/constants/upscale.py scripts/validate_gpu_stack.py tests/test_tryon_service.py tests/test_wardrobe_diffusers.py tests/test_wardrobe_runtime.py tests/test_upscale_service.py
bash -n scripts/install_gpu_stack.sh
uv run pytest -q
```

Remote validation on `select_green_guineafowl` before server launch:

```bash
/workspace/.venvs/glamify-image-ai/bin/python scripts/validate_gpu_stack.py
```

Result:

```text
RESULT: PASS
```

Result:

```text
90 passed
```

## Live Pod Validation And Failure Analysis

Validated on RunPod pod `select_green_guineafowl` (`dt3jjdcekx1lvl`) on June 9, 2026.

Runtime config:

```text
QWEN_IMAGE_EDIT_DTYPE=bfloat16
QWEN_FP8=1
QWEN_COMPILE=0
UPSCALE_MODEL_VARIANT=seedvr2_ema_3b_fp8_e4m3fn.safetensors
MINICPM_MODEL_PATH=/workspace/models/minicpm-v-4_5-awq
MINICPM_MAX_TOKENS=100
MINICPM_MAX_SLICE_NUMS=6
VLLM_USE_FLASHINFER_SAMPLER=0
TRYON_UPSCALE_AFTER_QWEN=1
TRYON_UPSCALE_TARGET_LONG_EDGE=3072
TRYON_FINAL_OUTPUT_LONG_EDGE=2048
```

Validation completed:

- `scripts/validate_gpu_stack.py` passed with torch `2.11.0+cu130`, diffusers `0.38.0`,
  transformers `4.57.6`, vLLM `0.22.1`, torchao `0.17.0`, and OpenCV `4.13.0`.
- Qwen pipeline loaded in bf16, then all wardrobe LoRAs loaded before fp8 quantization.
- Qwen transformer quantized with torchao fp8 dynamic activation/weight.
- MiniCPM AWQ loaded with `max_slice_nums=6` and vLLM logged:
  `FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0`.
- `/health` returned `status: ok` and reported loaded `wardrobe`, `tryon`, and `upscale`
  runtimes in both compile and no-compile runs.
- Warm resident VRAM after health OK was about `65,954 MiB / 97,887 MiB`.

### What Failed

The lab result and the API result are not equivalent.

The lab result that looked good was:

```text
Qwen lab, single category / single LoRA path:
fp8 + torch.compile + 15 steps -> about 6.5-6.8s
```

The full API path is different:

```text
One resident Qwen pipeline
Three PEFT adapters loaded: top, bottom, dress
Per request: pipeline.set_adapters([...])
Per request: real MiniCPM caption text changes the prompt
Per request: real garment image differs from the synthetic warmup image
```

With `QWEN_COMPILE=1`, startup did compile and warm all three wardrobe categories, but that did
not make the live API request fast:

| Attempt | Config | Result |
|---|---|---|
| Startup warmup, top | `QWEN_FP8=1`, `QWEN_COMPILE=1`, 12 steps | first step compiled; total about `4:54` |
| Startup warmup, bottom | same | first step compiled; total about `5:33` |
| Startup warmup, dress | same | first step compiled; total about `4:14` |
| First real `/v1/wardrobe` top request | same | triggered another compile path; curl timed out at `240s`; server later completed after about `5:27` Qwen time |
| Repeated real `/v1/wardrobe` top request | same | again started at Qwen `0/12` and re-specialized instead of reusing the expected compiled path |

So the exact issue is **not the change from 15 steps to 12 steps**. The issue is
`torch.compile` graph reuse in the full multi-adapter PEFT API path. The fused/single-LoRA lab
path can reuse the compiled graph, but the shared API path with `set_adapters(...)` and variable
real prompts/images re-specializes and repeatedly pays compile cost.

This made `QWEN_COMPILE=1` unusable for the current one-process multi-adapter API server.

### Stable Correction

The stable API config is:

```text
QWEN_IMAGE_EDIT_DTYPE=bfloat16
QWEN_FP8=1
QWEN_COMPILE=0
```

This keeps the fp8 VRAM benefit but avoids repeated TorchInductor compilation.

Measured same-image repeated top smoke test after restarting with `QWEN_COMPILE=0`:

| Run | Total wall | Fashion detector | MiniCPM AWQ caption | Qwen fp8 no-compile, 12 steps | Output upload |
|---:|---:|---:|---:|---:|---:|
| 1 | `35.96s` | `4.66s` | `0.43s` | `30.34s` | `0.50s` |
| 2 | `20.41s` | `0.05s` | `0.29s` | `19.76s` | `0.29s` |
| 3 | `18.80s` | `0.04s` | `0.28s` | `18.19s` | `0.27s` |
| 4 | `23.07s` | `0.04s` | `0.28s` | `22.47s` | `0.26s` |

This proves the API is stable and that the earlier high detector/MiniCPM timings were mostly
first-real-request warmup. After warmup, detector is about `40-50ms` and MiniCPM is about
`280-430ms`. The remaining bottleneck is actual Qwen generation, not queueing: the metadata showed
`qwen_generation_queued_wall_seconds` equal to `qwen_generation_seconds` for these runs.

The stable pod remained healthy after the loop, with the queue empty and warm VRAM around
`50.7 GB / 97.9 GB`.

This is still slower than the lab. To recover the lab speed, we should not use one shared
adapter-switching Qwen runtime with compile. The likely production direction is one of:

- fused/category-specific Qwen workers with one LoRA active per process and `QWEN_COMPILE=1`
- separate top/bottom/dress workers or pods if latency isolation is more important than simplicity
- a proven compile strategy for PEFT adapter switching, if we can make graph reuse deterministic

Still pending:

- Live `/v1/wardrobe` request tests for bottom and dress. Top was tested successfully with
  `QWEN_COMPILE=0`.
- Live `/v1/upscale` 4k request test with SeedVR2 3B.
- Live `/v1/tryon` request tests for top, bottom, dress, and multi.
- Visual quality sign-off for wardrobe and try-on outputs.

## Expected Live Request Checks

After health is OK, verify:

- `/v1/wardrobe`
- `/v1/upscale`
- `/v1/tryon`

For `/v1/tryon`, inspect response metadata:

- `runner.fp8`
- `runner.quantization`
- `runner.compiled`
- `upscale.model_variant`
- `upscale.target_long_edge`
- `upscale.upscaled_size_before_downscale`
- `output.width`
- `output.height`
- `timings.seedvr2_upscale_seconds`
- `timings.total_wall_seconds`

## Current Confidence

High confidence:

- Local implementation compiles and tests pass.
- SeedVR2 default/model/preset changes match the trial notes.
- Qwen fp8 uses the correct torchao quantization mechanism instead of invalid float8 dtype.
- Try-on inline upscale behavior is covered by tests.

Needs live pod proof:

- Actual fp8 Qwen warmup with all configured LoRAs.
- Actual VRAM usage.
- Actual latency.
- Visual quality for wardrobe and try-on outputs.

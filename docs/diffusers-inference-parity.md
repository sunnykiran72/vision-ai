# Wardrobe Diffusers Inference Parity

## Purpose

The `/v1/wardrobe` garment extraction now runs on a **diffusers** `QwenImageEditPlusPipeline`
backend instead of AI-Toolkit. This backend is a faithful port of the standalone diffusers
tester that produces the validated extraction quality on the pod.

This document records the reference, the exact inference contract we replicate, and how it
maps onto this service.

> **Production note.** The 8000/8010 webapps are dev-pod **reference testers only**. Production is
> a dedicated pod running only this vision-ai service, with both Qwen (diffusers) and MiniCPM-V
> (in-process vLLM) bundled in one process on one port. We replicate the testers' *code/inference*,
> never call their URLs. MiniCPM-V is documented in `docs/wardrobe-flow.md`; its tester is vendored
> at `reference/minicpm_vllm_webapp/app.py`.

## Reference Source

The validated inference is the "Qwen Diffusers Tester" served on pod port `8000`:

- Process: `python3 app.py`
- Location on pod: `/mnt/tryon-data/inference_apps/qwen_diffusers_webapp/app.py`
- A read-only copy is vendored at `reference/qwen_diffusers_webapp/app.py` for reference only
  (not imported by the service).

Runtime versions the reference runs against (system `python3` on the pod):

| Package | Version |
| --- | --- |
| python | 3.12.3 |
| torch | 2.8.0+cu128 |
| diffusers | 0.38.0 |
| transformers | 5.10.2 |

These are **not** declared in `pyproject.toml`; they are provided by the pod environment and
imported lazily, matching the existing AI-Toolkit pattern.

## Exact Inference Contract (replicated)

**Pipeline**: `diffusers.QwenImageEditPlusPipeline.from_pretrained(model, torch_dtype=bfloat16).to("cuda")`,
a single resident base model with the per-category LoRA swapped in per request.

**LoRA load (once per category)**:

```python
state_dict = load_file(path)                       # safetensors
remapped = {
    ("transformer." + k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k): v
    for k, v in state_dict.items()
}
pipe.load_lora_weights(remapped, adapter_name=category)
```

**Per request**: `pipe.set_adapters([category], [lora_scale])`.

**Generation core**:

```python
generator = torch.Generator(device="cuda").manual_seed(seed)
with torch.inference_mode():
    pipe(
        image=<single PIL>,            # extraction has one control image
        prompt=prompt,
        true_cfg_scale=1.0,
        num_inference_steps=steps,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
```

**Input resize** (`resize_input_for_model`, exact port): cap longest side to `1024`; if a resize
is needed, round the other side to the nearest multiple of 16; images already within `1024` are
returned unchanged (no rounding). LANCZOS.

**Output size**: the requested `width x height` (wardrobe uses `832 x 1248`), independent of the
input aspect ratio.

## Mapping Onto This Service

| Concern | Reference | This service |
| --- | --- | --- |
| Engine | `app.py` `Engine` | `app/clients/qwen_diffusers_engine.py` `QwenDiffusersWardrobeEngine` |
| Pipeline | `QwenImageEditPlusPipeline` bf16/cuda | same |
| LoRA load | lazy across ~18 LoRAs | **eager**: top/bottom/dress loaded at warmup before serving |
| LoRA swap | `set_adapters([key],[scale])` | same |
| Steps / seed / scale | form params | `constants/wardrobe.py` (`GENERATION_STEPS`, `GENERATION_SEED`, `GENERATION_NETWORK_MULTIPLIER`) |
| `true_cfg_scale` | `1.0` | `wardrobe.GENERATION_TRUE_CFG_SCALE = 1.0` |
| Output size | form `width x height` | `OUTPUT_WIDTH x OUTPUT_HEIGHT` = `832 x 1248` |
| Input resize | `resize_input_for_model` | `resize_input_for_model` (same module, same logic) |
| Prompt (test endpoint) | `PROMPTS[category]` | `wardrobe.PROMPT_BY_TYPE[type]` (identical strings) |
| Prompt (`/v1/wardrobe`) | n/a | MiniCPM caption embedded in `QWEN_EXTRACT_PROMPT_TEMPLATE_BY_TYPE[type]` |

The wardrobe service feeds the **in-memory** preprocessed image to the engine (not a reloaded
JPEG), so the pixels reaching the pipeline match the reference. The `/tools/diffusers/extract`
test endpoint uses the short reference prompt for direct 8000 parity; the live `/v1/wardrobe`
flow adds the MiniCPM garment caption (see `docs/wardrobe-flow.md`).

### Eager loading

8000 stays lazy because it carries ~18 LoRAs; wardrobe carries only three, so the warmup path
(`warmup_wardrobe_runtime` -> `QwenDiffusersWardrobeEngine.warmup`) loads the base pipeline and the
top/bottom/dress LoRAs, then runs one warm pass, all **before** the API accepts requests.

## LoRAs (pod paths)

| Type | Steps | File |
| --- | --- | --- |
| top | 23000 | `/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/.../*_000023000.safetensors` |
| bottom | 30000 | `/mnt/qwen-garment-extract/outputs/qwen_garment_extract_bottom119_glambtmext_rank16_b1_continue_12k_to_22k_lr815e5_v1/...v1.safetensors` |
| dress | 27000 | `/mnt/qwen-garment-extract/outputs/qwen_garment_extract_dress125_glamdressext_rank16_b1_continue_15k_to_30k_lr125e5_v1/.../*_000027000.safetensors` |

Wire these through `WARDROBE_LORA_TOP_PATH` / `WARDROBE_LORA_BOTTOM_PATH` / `WARDROBE_LORA_DRESS_PATH`
and the model through `QWEN_IMAGE_EDIT_MODEL_PATH=/mnt/models/qwen-image-edit-2511`.

## Try-on

Try-on is **unchanged** in this pass — it still runs on the AI-Toolkit backend. The diffusers
backend will be extended to try-on once its LoRAs are decided. The AI-Toolkit code is left intact.

## Running And Testing

On the dedicated vision-ai pod the service runs on port `8000` (see
`docs/deployment-setup.md` for full install/run steps):

```bash
PORT=8000 scripts/bootstrap.sh --no-sync
```

For extraction-only testing you can keep just the wardrobe runtime resident with
`RESIDENT_RUNTIMES=wardrobe`.

### Parity test endpoints (unauthenticated, under `/tools`)

Mirror the reference's multipart contract so output can be A/B compared against the reference:

- `GET /tools/diffusers/loras` -> available categories.
- `POST /tools/diffusers/extract` -> multipart `source` (file), `lora_key` (`top|bottom|dress`),
  optional `steps`, `seed`, `width`, `height`, `lora_scale`, `prompt`; returns `image/jpeg` with
  `X-Extract-*` metadata headers. Shares the resident engine and the system GPU coordinator.

```bash
curl -s -X POST "http://localhost:8000/tools/diffusers/extract" \
  -F source=@garment.jpg -F lora_key=top -o out.jpg
```

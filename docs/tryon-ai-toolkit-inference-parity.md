# Try-On AI-Toolkit Inference Parity

> Historical note: this document records the pre-diffusers AI-Toolkit parity work. The current
> `/v1/tryon` runtime shares the resident diffusers Qwen backend with wardrobe; see
> `docs/tryon-flow.md` for the current implementation.

## Purpose

This document defines what must change in `/v1/tryon` to match the quality we saw from the AI-Toolkit training samples and the RunPod Gradio tester.

The current production preparation is structurally good: route, service, runtime, queueing, storage, and request isolation are in place. The accuracy gap is mainly from the runtime path and sampling contract, not from the API shape.

## Current Implementation

Current live flow:

- `app/routes/tryon.py` receives `/v1/tryon`.
- `app/services/tryon.py` downloads the user image and garment images.
- `app/utils/tryon_collage.py` creates one garment reference image.
- `app/clients/qwen_image_edit.py` loads Qwen Image Edit through `diffusers.DiffusionPipeline`.
- The LoRA is loaded with `pipe.load_lora_weights(...)`.
- The pipeline receives two images through `image=[garment_reference, user_reference]`.
- The output is uploaded through Azure storage.

This is a valid prototype, but it is not the same runtime path that generated the stronger AI-Toolkit samples.

## Parity Problems

### 1. LoRA loading path is different

The validated RunPod tester used the AI-Toolkit network path:

- base model loaded through AI-Toolkit model classes
- LoRA loaded through `LoRASpecialNetwork`
- rank and alpha set to `64`
- network attached to the Qwen model before generation
- generation called through `sd.generate_images(...)`

The current production client uses direct Diffusers LoRA loading:

```python
pipe.load_lora_weights(...)
pipe.set_adapters(...)
```

For this LoRA, that direct path produced visibly worse outputs in testing. Production should use an AI-Toolkit-compatible backend for try-on quality parity.

This does not mean the generic Qwen Diffusers client should be removed. It should remain the generic image-edit/wardrobe path. Try-on parity should be implemented as a separate backend so the wardrobe path does not inherit AI-Toolkit-specific assumptions.

### 2. Control image order is reversed

Training and AI-Toolkit samples use this semantic order:

- image 1 / `ctrl_img_1`: person image
- image 2 / `ctrl_img_2`: garment reference image

Training prompts also say:

```text
Apply the reference garment from image 2 to the person in image 1.
```

Current production code sends:

```python
image=[garment_reference, user_reference]
```

That is the opposite of the training prompt and sample config. The production AI-Toolkit path must pass:

```text
ctrl_img_1 = user/person image
ctrl_img_2 = garment/reference image
```

### 3. Defaults do not match the validated samples

Current config defaults are tuned for speed/prototype behavior:

- `TRYON_LORA_SCALE=1.5`
- `TRYON_DEFAULT_SEED=44`
- `TRYON_DEFAULT_STEPS=8`
- `TRYON_DEFAULT_GUIDANCE_SCALE=2.5`

Validated sample parity used:

- LoRA/network multiplier: `1.0`
- seed: `43` for repeatable evaluation
- steps: `25` for current quality checks
- guidance scale: `1`
- guidance rescale: `0`
- `do_cfg_norm: false`
- output size: `1024x1536`

Production can later expose preview/final quality modes, but the parity baseline should first match the validated setup.

### 4. Prompt wording is not training-matched

Current prompt starts with:

```text
Put the provided garments onto the person.
```

Training/eval prompts are more specific:

```text
Apply the reference garment from image 2 to the person in image 1.
Preserve the person's face, identity, body proportions, pose, and background.
```

The API service should build prompts in the same style as training, including category-specific product text.

### 5. AI-Toolkit generation expects control image paths

The validated AI-Toolkit generation path uses `GenerateImageConfig` with file paths:

- `ctrl_img_1`
- `ctrl_img_2`

Current service only saves the garment reference to a request-local path. It keeps the user image as a PIL image. For the AI-Toolkit backend, both controls should be saved in the request job directory and passed as paths.

## Target Runtime

Create a dedicated try-on AI-Toolkit client instead of overloading the generic Diffusers client.

Recommended new file:

```text
app/clients/qwen_tryon_aitk.py
```

The existing `app/clients/qwen_image_edit.py` can remain as a generic Diffusers fallback, but it should not be the quality path for `/v1/tryon`.

Ownership boundary:

- `app/clients/qwen_image_edit.py`: generic Qwen Image Edit / wardrobe / fallback Diffusers client
- `app/clients/qwen_tryon_aitk.py`: try-on-specific AI-Toolkit parity client
- `app/runtime/tryon_runtime.py`: selects the try-on backend and hides that choice from route/service code

Do not mutate `QwenImageEditClient.run_edit(...)` or the generic Diffusers loading path to satisfy try-on parity. That would couple wardrobe/edit behavior to the try-on LoRA runtime.

The AI-Toolkit client should:

- load AI-Toolkit from `AI_TOOLKIT_ROOT`
- load Qwen Image Edit 2511 from `QWEN_IMAGE_EDIT_MODEL_PATH`
- initialize model architecture as `qwen_image_edit_plus`
- create `LoRASpecialNetwork` with rank `64` and alpha `64`
- load the selected `.safetensors` checkpoint
- keep the base model and network resident in memory
- reload only the LoRA/network when the checkpoint path changes
- serialize inference with the existing coordinator/runner lock
- generate through AI-Toolkit `sd.generate_images(...)`

Rank and alpha are part of the selected checkpoint contract. For the current `qwen_tryon_1496_full_clean_review_v1` checkpoints they are `64` and `64`. They should stay configurable because a future LoRA checkpoint may be trained with a different rank or alpha.

Expected core generation contract:

```python
GenerateImageConfig(
    prompt=prompt,
    ctrl_img_1=person_image_path,
    ctrl_img_2=garment_reference_path,
    seed=seed,
    width=1024,
    height=1536,
    num_inference_steps=steps,
    network_multiplier=1.0,
    guidance_scale=1.0,
    guidance_rescale=0.0,
    do_cfg_norm=False,
)
```

Use sampler:

```text
flowmatch
```

## Required Config

Add or adjust these `Settings` fields in `app/config.py`:

```text
TRYON_BACKEND=ai_toolkit
AI_TOOLKIT_ROOT=/mnt/tryon-data/ai-toolkit
QWEN_IMAGE_EDIT_MODEL_PATH=/mnt/models/qwen-image-edit-2511
TRYON_LORA_PATH=/mnt/tryon-data/runs/qwen-tryon-1496-full-clean-v1/output_resume_from_200_samples8/qwen_tryon_1496_full_clean_review_v1/qwen_tryon_1496_full_clean_review_v1_000003600.safetensors
TRYON_LORA_WEIGHT_NAME=
TRYON_LORA_RANK=64
TRYON_LORA_ALPHA=64
TRYON_LORA_SCALE=1.0
TRYON_DEFAULT_SEED=43
TRYON_DEFAULT_STEPS=25
TRYON_DEFAULT_GUIDANCE_SCALE=1.0
TRYON_GUIDANCE_RESCALE=0.0
TRYON_DO_CFG_NORM=false
TRYON_SAMPLER=flowmatch
TRYON_OUTPUT_WIDTH=1024
TRYON_OUTPUT_HEIGHT=1536
TRYON_PREVIEW_WIDTH=896
TRYON_PREVIEW_HEIGHT=1344
TRYON_PREVIEW_STEPS=20
```

Notes:

- `TRYON_LORA_PATH` should be a direct `.safetensors` checkpoint path for the AI-Toolkit try-on backend.
- `TRYON_LORA_WEIGHT_NAME` is only for the direct Diffusers fallback convention. It should be empty for `TRYON_BACKEND=ai_toolkit`.
- For production randomness, allow request seed to be omitted later. For parity tests, keep seed fixed.
- `896x1344` can be kept as a faster preview size because it is still 2:3, but final quality checks should use `1024x1536`.

Backend-aware startup validation:

- Always require the non-try-on service fields that are already globally required: auth, Azure storage, wardrobe runtime config, and upscale runtime config.
- Try-on fields must stop being globally required. Validate them based on `TRYON_BACKEND`.
- If `TRYON_BACKEND=ai_toolkit`, require `AI_TOOLKIT_ROOT`, `QWEN_IMAGE_EDIT_MODEL_PATH`, `TRYON_LORA_PATH`, `TRYON_LORA_RANK`, `TRYON_LORA_ALPHA`, `TRYON_OUTPUT_WIDTH`, and `TRYON_OUTPUT_HEIGHT`.
- If `TRYON_BACKEND=diffusers`, require the current Diffusers fields: `QWEN_IMAGE_EDIT_MODEL_PATH`, `TRYON_LORA_PATH`, and optionally `TRYON_LORA_WEIGHT_NAME`.
- If `TRYON_BACKEND=disabled`, skip try-on model warmup and report try-on as unavailable in health metadata.
- Reject unknown backend values during startup validation.

Do not keep unconditional `TRYON_LORA_WEIGHT_NAME` validation for the AI-Toolkit backend. The two backends use different LoRA path conventions.
Do not keep unconditional `TRYON_LORA_PATH` or model-path validation for try-on when `TRYON_BACKEND=disabled`.

## Backend Switch Contract

`app/runtime/tryon_runtime.py` must own backend selection.

Required behavior:

- `TRYON_BACKEND=ai_toolkit` returns `QwenTryonAitkClient`.
- `TRYON_BACKEND=diffusers` returns the existing `QwenImageEditClient`.
- `TRYON_BACKEND=disabled` does not warm a model and `/v1/tryon` returns a controlled unavailable response.
- Default backend selection must be explicit. Use `ai_toolkit` for parity deployments and `diffusers` only as a fallback/prototype mode.

Disabled API response contract:

```json
{
  "status": 503,
  "message": "Try-on backend is disabled.",
  "data": {
    "url": null,
    "metadata": {
      "feature": "tryon",
      "backend": "disabled"
    }
  }
}
```

The route should keep the existing pattern: service returns `status=503`, and the route applies that status to the HTTP response.

Startup behavior:

- `app/runtime/warmup.py` should call try-on warmup only when try-on backend is not `disabled`.
- AI-Toolkit import/path errors must fail startup in parity deployments, not silently fall back to Diffusers.
- Local development can set `TRYON_BACKEND=disabled` if AI-Toolkit is not installed.

Runner cache key:

- include backend name
- include base model path
- include AI-Toolkit root for `ai_toolkit`
- include LoRA checkpoint path
- include rank, alpha, scale, sampler, output size, and CFG flags

This prevents a cached Diffusers runner or stale checkpoint from being reused after config changes.

Coordinator cache key:

- include queue max size
- include queue wait timeout
- optionally include backend name for strict separation
- do not include model path, checkpoint path, rank, alpha, sampler, output size, or CFG fields

The coordinator controls admission and queue timing. It should not be duplicated for every checkpoint or model config unless queue behavior changes.

## Required Service Changes

In `app/services/tryon.py`:

- create try-on-specific job paths or extend `JobMediaPaths`
- save the user image as `person.jpg` inside the request job directory
- save the garment reference as `garment_reference.jpg`
- build the prompt using the training-style template
- call the try-on runner with control paths, not only PIL images
- pass explicit output size from config
- record control order in metadata

The current generic media helper only exposes one `input_path` and one `output_path`. The implementation must not overload `input_path` for multiple controls.

Recommended try-on path contract:

```text
job_dir/
  person.jpg
  garment_reference.jpg
  output.jpg
  metadata.json
```

`metadata.json` is optional debug/audit output only. The AI-Toolkit runtime must not depend on it for generation. Because the current service cleanup removes the request job directory after completion, durable metadata belongs in the API response metadata and external logs/storage, not this temp file.

Implementation options:

- add `build_tryon_job_media_paths(...)` in `app/utils/media_utils.py`
- or extend `JobMediaPaths` with optional `person_path` and `garment_reference_path`

The first option is cleaner because try-on has a real two-control input contract while upscale and generic edit are single-input flows.

Metadata should include:

```json
{
  "backend": "ai_toolkit_exact",
  "architecture": "qwen_image_edit_plus",
  "control_order": {
    "ctrl_img_1": "person",
    "ctrl_img_2": "garment_reference"
  },
  "lora_rank": 64,
  "lora_alpha": 64,
  "network_multiplier": 1.0,
  "guidance_scale": 1.0,
  "guidance_rescale": 0.0,
  "do_cfg_norm": false,
  "output_size": {
    "width": 1024,
    "height": 1536
  }
}
```

## Prompt Template

Use this structure:

```text
Apply the reference garment from image 2 to the person in image 1. {product_descriptions}
Preserve the person's face, identity, body proportions, pose, and background.
```

For multiple garments:

```text
Apply the reference garments from image 2 to the person in image 1. Top: {top_prompt}. Bottom: {bottom_prompt}. Preserve the person's face, identity, body proportions, pose, and background.
```

For dress:

```text
Apply the reference garment from image 2 to the person in image 1. Dress: {dress_prompt}. Preserve the person's face, identity, body proportions, pose, and background.
```

Do not use generic styling words unless they came from the product prompt. The LoRA was trained to follow concrete garment descriptions.

## Parity And Production Profiles

Separate validation defaults from product defaults.

Parity validation profile:

- fixed seed `43`
- `1024x1536`
- 25 steps
- LoRA/network multiplier `1.0`
- guidance scale `1.0`
- guidance rescale `0.0`
- `do_cfg_norm=false`
- exact checkpoint under test

Production default profile:

- may use random seed or request-provided seed
- should still start with LoRA/network multiplier `1.0`
- should use the selected production checkpoint
- may offer preview/final modes once parity is confirmed
- should not enable CFG by default until it improves a fixed sample suite

Do not optimize production defaults before matching the AI-Toolkit sample behavior. First achieve parity, then tune speed and cost.

## Runtime Design

Keep the existing resident-runner pattern:

- FastAPI process owns one warm try-on runner.
- Startup warmup loads the model once.
- The coordinator limits queue depth.
- The runner lock serializes GPU generation.
- Request-local files isolate inputs and outputs.

Avoid per-request model reload. That would make latency unusable.

Backend migration gate:

- merge the AI-Toolkit client behind `TRYON_BACKEND=ai_toolkit`
- keep `diffusers` fallback available during migration
- deploy with `ai_toolkit` only on images/pods that include AI-Toolkit and the selected checkpoint
- use `disabled` for environments that should boot without try-on GPU dependencies
- do not silently downgrade from `ai_toolkit` to `diffusers`; that would hide quality regressions

Checkpoint switching:

- In production, use one selected checkpoint by env/config.
- For internal testing, allow checkpoint path reload only when explicitly changed.
- When the checkpoint changes, reload the LoRA/network, not the base Qwen model.

## Expected Latency

The exact AI-Toolkit path is quality-first, not sub-10-second.

At `1024x1536` and `25` steps, expect roughly the same class of latency as the RunPod tester unless the GPU is much stronger or the backend is optimized.

Speed options:

- `896x1344`, 20-25 steps for preview
- selected checkpoint with 20-25 steps for normal testing
- only use 35+ steps for final QA
- keep model resident and warmed
- do not enable CFG unless it is proven useful

Training more steps may improve LoRA quality, but it will not magically make 4-step inference production-ready. A separate distilled/few-step model strategy would be needed for that.

## Tests To Add

Add tests for:

- prompt builder uses the training-style text
- service saves and passes `person` as control 1
- service saves and passes `garment_reference` as control 2
- config defaults match parity values
- runtime status reports `backend=ai_toolkit_exact`
- metadata includes checkpoint, rank, alpha, scale, control order, output size, and CFG disabled fields
- direct Diffusers backend is not selected when `TRYON_BACKEND=ai_toolkit`
- `TRYON_BACKEND=ai_toolkit` selects `QwenTryonAitkClient`
- `TRYON_BACKEND=diffusers` selects the existing `QwenImageEditClient`
- `TRYON_BACKEND=disabled` skips try-on warmup and returns HTTP `503` with message `"Try-on backend is disabled."`
- startup validation requires different fields per backend
- runtime cache keys differ when backend, checkpoint, rank, alpha, sampler, output size, or CFG fields change
- coordinator cache keys stay queue-based and do not include checkpoint/model details

Existing service tests can keep using a fake runner, but the fake should assert control order explicitly.

## Validation Plan

Use the same sample set used in AI-Toolkit training:

```text
/Users/kiran/Documents/Codex/2026-05-12/so-i-m-preparing-a-dataset/dataset/tryon_test_samples_8_with_prompts.csv
```

For each selected checkpoint:

- run the AI-Toolkit training sample output
- run the RunPod Gradio tester
- run `/v1/tryon` with the same person image, garment reference, prompt, seed, steps, LoRA scale, and output size
- compare results visually

The goal is behavioral parity, not pixel-perfect identity. Small nondeterminism can happen across environments. The important checks are:

- correct garment category is applied
- garment silhouette is complete
- texture/detail transfers reasonably
- face identity stays stable
- pose and background are preserved
- hands/fingers do not degrade more than the AI-Toolkit sample
- output roughness is comparable to the selected checkpoint

## Implementation Order

1. Add backend-aware config fields in `app/config.py`.
2. Update `validate_startup_settings(...)` for `ai_toolkit`, `diffusers`, and `disabled`.
3. Add try-on-specific media paths in `app/utils/media_utils.py`.
4. Add `app/clients/qwen_tryon_aitk.py`.
5. Update `app/runtime/tryon_runtime.py` to choose backend from `TRYON_BACKEND`.
6. Update `app/runtime/warmup.py` so disabled try-on does not warm and AI-Toolkit failures are not silently hidden.
7. Update `app/services/tryon.py` to save both controls and pass explicit output size.
8. Update prompt builder to match training templates.
9. Update tests for backend switching, startup validation, control order, config, and metadata.
10. Run a local/unit test pass.
11. Deploy to the inference pod and compare the 8 fixed samples against Gradio output.

## Decision

For production-quality try-on, use the AI-Toolkit-compatible inference path.

The current direct Diffusers implementation should be treated as a fallback/prototype path until it can reproduce the same output quality. The validated quality path is the same path used by training samples: Qwen Image Edit 2511 plus AI-Toolkit `LoRASpecialNetwork`, with person as image 1 and garment reference as image 2.

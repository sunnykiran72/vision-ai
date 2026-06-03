# Try-On Flow Update — Single LoRA to Specialist Multi-LoRA Architecture

## Purpose

This document defines how `/v1/tryon` should evolve from the current single-LoRA implementation (one general try-on LoRA referenced by `TRYON_LORA_PATH`) to a **specialist multi-LoRA architecture** with one LoRA per garment category, all loaded resident in GPU memory and switched per request.

It covers:
- target architecture
- per-category LoRA contract (paths, triggers, ranks, sampling parameters)
- runtime loading strategy (single base model + 4 resident LoRAs)
- request routing logic
- AI-Toolkit sample-output parity (production contract)
- configuration changes required
- migration plan and acceptance criteria
- pod SSH details for inspecting the trained models

This document does **not** implement code. It is a specification that the implementation work will follow.

Companion documents:
- [tryon-flow.md](tryon-flow.md) — current single-LoRA implementation
- [tryon-ai-toolkit-inference-parity.md](tryon-ai-toolkit-inference-parity.md) — AI-Toolkit parity requirements that continue to apply per-LoRA in the new architecture

---

## 1. Current State (single LoRA)

Today, `/v1/tryon` uses **one** LoRA for all categories. One AI-Toolkit runtime is loaded with that LoRA at startup. All requests route to the same model regardless of garment category.

The current free-form prompt template builds something like:

```text
Apply the reference garment from image 2 to the person in image 1.
Top: red structured jacket.
Bottom: black straight trousers.
Preserve the person's face, identity, body proportions, pose, and background.
```

### Why we are changing this

The unified rank-32 LoRA had real failures: `top` requests sometimes extended into dresses (erasing the original lower-body clothing), shape transfer was unreliable at `guidance_scale: 1.0`, and decision-boundary outputs were unstable across passes.

The fix is to physically separate categories into specialist LoRAs. Each specialist trains only on its own category's data, eliminating cross-category confusion at the model level. The disambiguation moves from "model decides from text trigger" to "application decides which LoRA to load."

---

## 2. Target Architecture

Four specialist LoRAs, all at **rank 16**, each trained with AI-Toolkit on a category-isolated dataset.

| Category | LoRA name | Rank | Dataset rows | Trigger caption |
|---|---|---|---|---|
| `top` | `qwen_lora_glamtoptryon_v1` | 16 | 230 | `Apply GlamifyTopTryon on this person` |
| `bottom` | `qwen_lora_glambottomtryon_v1` | 16 | 259 | `Apply GlamifyBottomTryon on this person` |
| `dress` | `qwen_lora_dress_v1` | 16 | 351 | `Apply GlamifyDressTryon on this person` |
| `multi` | `qwen_lora_glammultitryon_v1` | 16 | 297 | `Apply GlamifyMultiTryon on this person` |

All four share these properties (verified against the live training configs on each pod):

| Property | Value |
|---|---|
| Base model | `Qwen-Image-Edit-2511` at `QWEN_IMAGE_EDIT_MODEL_PATH` |
| Architecture | `qwen_image_edit_plus` |
| Control image order | `ctrl_img_1 = person`, `ctrl_img_2 = garment_reference` |
| Training resolution | 1280 |
| Optimizer | adamw8bit |
| Learning rate | 1e-4, cosine to `eta_min: 1e-5` |
| Training dtype | bf16 |
| Noise scheduler | flowmatch |
| caption_dropout_rate | 0 (single trigger per LoRA; no dropout needed) |
| Safetensors file size | ~282 MB per LoRA at rank 16 |
| Combined LoRA weight footprint | ~1.13 GB across all 4 |

---

## 3. Sample-Output Parity Contract (production must match this)

The single most important rule of the new system: **production inference must produce the same output that the AI-Toolkit training samples produce at the same checkpoint.**

Why this matters:
- We use the AI-Toolkit training samples (the `samples/` folder produced every 250 steps) as the acceptance criterion
- If production diverges from those samples, sample quality cannot predict production quality
- A direct `diffusers.load_lora_weights(...)` path produces visibly different outputs than the AI-Toolkit `LoRASpecialNetwork` path, even for the same checkpoint file

### 3.1 Sampling parameters (locked to training-sample values)

```text
arch                       = qwen_image_edit_plus
sampler                    = flowmatch
sample_steps               = 25
guidance_scale             = 1.0
guidance_rescale           = 0.0
do_cfg_norm                = false
network_multiplier         = 1.0
seed                       = 43 for parity testing; request-provided for production
```

These values are not negotiable for parity. They are verified against the live `sample` block of each pod's training config.

### 3.2 Output dimensions — match user image

Training samples are generated at fixed `1024 × 1536` (3:2 portrait). **Production output should use the user image's own dimensions** rather than a fixed size, so try-on results return at the same aspect ratio and resolution the user sent.

Implementation rule:

```text
output_width  = user_image.width
output_height = user_image.height
```

Notes on this choice:
- Qwen-Image-Edit-Plus is dimension-flexible at inference and handles arbitrary aspect ratios
- Returning the original dimensions removes the need for client-side cropping/resizing
- If the user image is far from the 3:2 aspect the model saw at training time, output quality may shift slightly; this is acceptable for production realism

Practical implementation detail (implemented): the service buckets the inference dimensions to the nearest multiple of `TRYON_DIMENSION_MULTIPLE` (default 64) before calling `sd.generate_images(...)`, then resizes the model output back to the user image's exact original dimensions before storage upload. AI-Toolkit's internal bucketing already does this during training, so feeding bucketed dimensions matches the training contract; the post-resize keeps the response shape equal to the request shape.

### 3.3 Prompt structure (default trigger + product detail + identity preservation)

Each specialist must be invoked with its **specialist trigger caption** as the first sentence. The rest of the prompt is a structured product description plus an identity-preservation clause. This combination keeps the model anchored to:
1. the correct specialist behavior (via the trigger)
2. the specific garment in the request (via the product description)
3. the person's identity (via the preservation clause, reducing hallucination of body proportions / height / face)

#### Prompt template — single garment

```text
{trigger_caption}. {category_label}: {product_description}. Preserve the person's face, identity, body proportions, pose, and background.
```

Examples:

```text
Apply GlamifyTopTryon on this person. Top: red structured jacket with notched lapels. Preserve the person's face, identity, body proportions, pose, and background.
```

```text
Apply GlamifyBottomTryon on this person. Bottom: black straight tailored trousers. Preserve the person's face, identity, body proportions, pose, and background.
```

```text
Apply GlamifyDressTryon on this person. Dress: navy blue knee-length wrap dress. Preserve the person's face, identity, body proportions, pose, and background.
```

#### Prompt template — multi-garment

```text
Apply GlamifyMultiTryon on this person. Top: {top_description}. Bottom: {bottom_description}. Preserve the person's face, identity, body proportions, pose, and background.
```

Or for any other multi combination (handled by the collage), list each product type and its description in priority order:

```text
priority order: top / outer  >  dress  >  bottom
```

If a description for a garment is missing in the request, omit that category-label section rather than emit an empty `Top: .` clause.

#### Why this structure

- The **trigger** activates the correct specialist behavior baked in during training
- The **product description** anchors the model to the actual requested garment so it does not improvise color/cut details from the control image alone (Qwen-Image-Edit-Plus understands natural-language clothing descriptions well)
- The **identity-preservation clause** is standard for Qwen-Image-Edit and reduces hallucination of height, face, and pose

Note for validation: the specialists were trained on the trigger caption alone (no product details). Adding product descriptions at inference is off-training-distribution but in-distribution for the base Qwen-Image-Edit-Plus. Validate empirically during shadow rollout (Phase 2) that this prompt structure does not degrade sample-parity output for the same person/garment input.

### 3.4 Control image contract (unchanged from current)

```text
ctrl_img_1 = person.jpg               (user image, saved as JPEG to the request job directory)
ctrl_img_2 = garment_reference.jpg    (collapsed garment reference)
```

The existing garment-reference collage logic in `app/utils/tryon_collage.py` continues to apply:

- Single garment → use directly
- Top + bottom → vertical collage
- Other multi-garment combinations → horizontal collage

The specialist LoRAs were trained on garment-reference images shaped the same way, so the collage rules are part of the validated input contract.

---

## 4. Category Routing

Simple, deterministic, based on `len(products)`.

```text
if len(products) == 1:
    use the LoRA matching product.type:
        product.type == "top"    -> TOP specialist
        product.type == "outer"  -> TOP specialist (outerwear is upper-body)
        product.type == "bottom" -> BOTTOM specialist
        product.type == "dress"  -> DRESS specialist

if len(products) >= 2:
    use the MULTI specialist
    pass the collapsed garment-reference collage as ctrl_img_2
```

That is the complete routing rule. The garment-reference collage in `app/utils/tryon_collage.py` handles every multi-garment combination, so the routing layer does not need to enumerate combinations.

### 4.1 Routing function shape

A small routing helper (location: either inside `app/services/tryon.py` or a new `app/services/tryon_routing.py`) returns:

```text
RoutingDecision {
    lora_key:        Literal["top", "bottom", "dress", "multi"]
    trigger_caption: str   (the specialist trigger caption, e.g. "Apply GlamifyTopTryon on this person")
}
```

The route/service calls this before prompt construction so the rest of the service knows which LoRA to invoke and which trigger to lead the prompt with.

### 4.2 Outerwear is mapped to TOP

The request body allows `type: "outer"`. The training data does not have a separate "outer" specialist, so outerwear requests use the TOP specialist. The prompt's category label remains `Top: ...` regardless of whether the source product type is `top` or `outer`.

---

## 5. Runtime Loading Strategy — All 4 LoRAs Resident

The deployment target uses a large GPU (e.g., H100 NVL 96 GB) so all 4 specialists can stay resident alongside the base model with no swap or reload cost between requests.

### 5.1 Plan

- Load the base model **once** at server startup using AI-Toolkit (`qwen_image_edit_plus`)
- Create one resident `LoRASpecialNetwork` shell (rank/alpha shared across all 4 specialists), patch the base model with `apply_to(...)` once
- Pre-load each specialist's `.safetensors` checkpoint into a cached state-dict on the same device as the network, keyed by category (`top`, `bottom`, `dress`, `multi`)
- Keep all 4 cached state-dicts resident in GPU memory for the life of the process
- On each request:
  1. Look up the routing decision
  2. If the active specialist differs from the routed category, call `network.load_state_dict(cached_sd[category], strict=False)` — this is a tensor copy, sub-millisecond on H100
  3. Build the prompt
  4. Call `sd.generate_images(...)` with the request's controls and the bucketed inference dimensions
  5. Resize the model output back to the original user-image dimensions before returning

#### Why one network with 4 cached state-dicts (not 4 network instances)

AI-Toolkit's `LoRASpecialNetwork` exposes `apply_to(...)` (which patches the base model with LoRA hooks) and `load_weights(...)` / `load_state_dict(...)`, but there is no `restore_from(...)` or detach API. Calling `apply_to` on a second network instance does not cleanly detach the first.

Since all 4 specialists share the same architecture (rank 16, alpha 16, `qwen_image_edit_plus`), the LoRA module shape is identical. Caching 4 state-dicts on GPU and swapping them into the single resident network has the same observable behaviour as "4 instances with attach/detach": the LoRA weights are resident, the switch is a tensor copy. This is the path the production client implements.

### 5.2 GPU memory budget

| Component | VRAM |
|---|---|
| Qwen-Image-Edit-Plus base model (bf16) | ~40 GB |
| 4 × rank-16 LoRA networks resident | ~1.1 GB |
| Activations / KV cache at 1024×1536 inference, batch 1 | ~10–15 GB |
| **Total peak** | **~55 GB** on H100 NVL (96 GB available) |

Plenty of headroom. No quantization needed. No swap-from-disk overhead on category switching.

### 5.3 Concurrency

The existing coordinator + runner lock pattern (see `tryon-flow.md` § "Runtime layer") continues to apply unchanged. Only one inference runs at a time; LoRA attach/detach happens between inferences, not during.

### 5.4 Runner cache key (for cache invalidation)

The resident runner cache key must include the paths of **all 4** LoRA checkpoints plus the base model path. If any of those values change, the cached runner is invalidated and the next request triggers a fresh load.

---

## 6. Configuration Changes

### 6.1 `app/config.py` — additions

```text
# Specialist LoRA paths (one per category)
TRYON_LORA_TOP_PATH    = /mnt/tryon-data/releases/specialists/v1/top/glamtoptryon_v1_rank16_<step>.safetensors
TRYON_LORA_BOTTOM_PATH = /mnt/tryon-data/releases/specialists/v1/bottom/glambottomtryon_v1_rank16_<step>.safetensors
TRYON_LORA_DRESS_PATH  = /mnt/tryon-data/releases/specialists/v1/dress/dress_v1_rank16_<step>.safetensors
TRYON_LORA_MULTI_PATH  = /mnt/tryon-data/releases/specialists/v1/multi/glammultitryon_v1_rank16_<step>.safetensors

# Specialist trigger captions (the first sentence of the prompt)
TRYON_PROMPT_TRIGGER_TOP    = "Apply GlamifyTopTryon on this person"
TRYON_PROMPT_TRIGGER_BOTTOM = "Apply GlamifyBottomTryon on this person"
TRYON_PROMPT_TRIGGER_DRESS  = "Apply GlamifyDressTryon on this person"
TRYON_PROMPT_TRIGGER_MULTI  = "Apply GlamifyMultiTryon on this person"

# Identity-preservation clause (appended at the end of every prompt)
TRYON_PROMPT_IDENTITY_CLAUSE = "Preserve the person's face, identity, body proportions, pose, and background."

# Shared rank/alpha (same for all 4 specialists in v1)
TRYON_LORA_RANK  = 16
TRYON_LORA_ALPHA = 16

# AI-Toolkit parity fields (unchanged from tryon-ai-toolkit-inference-parity.md)
TRYON_BACKEND               = ai_toolkit
AI_TOOLKIT_ROOT             = /mnt/tryon-data/ai-toolkit
QWEN_IMAGE_EDIT_MODEL_PATH  = /mnt/models/qwen-image-edit-2511
TRYON_LORA_SCALE            = 1.0
TRYON_DEFAULT_SEED          = 43
TRYON_DEFAULT_STEPS         = 25
TRYON_DEFAULT_GUIDANCE_SCALE = 1.0
TRYON_GUIDANCE_RESCALE      = 0.0
TRYON_DO_CFG_NORM           = false
TRYON_SAMPLER               = flowmatch

# Specialist runtime gate + bucketing knob
TRYON_USE_SPECIALISTS       = true      # false during Phase 0; flipped to true in Phase 2
TRYON_DIMENSION_MULTIPLE    = 64        # bucket inference dims to a multiple of this; output resized back to original
```

### 6.2 Removed / scoped from config

- `TRYON_OUTPUT_WIDTH`, `TRYON_OUTPUT_HEIGHT` — removed; replaced by "use user image dimensions" (see §3.2)
- `TRYON_PREVIEW_WIDTH`, `TRYON_PREVIEW_HEIGHT`, `TRYON_PREVIEW_STEPS` — removed; preview/final mode is no longer a runtime knob now that output dimensions track the user image
- `TRYON_LORA_PATH` (single path) — retained for the legacy path used when `TRYON_USE_SPECIALISTS=false` (Phase 0/1 fallback). Phase 3 retires it entirely.

The free-form prompt section templates in `app/services/tryon.py` (`TRYON_SINGLE_REFERENCE_PROMPT`, `TRYON_MULTI_REFERENCE_PROMPT`, `TRYON_TOP_SECTION_TEMPLATE`, etc.) are kept in code only as the prompt builder for the legacy `TRYON_USE_SPECIALISTS=false` branch. Phase 3 removes them.

### 6.3 Startup validation

If `TRYON_BACKEND=ai_toolkit`, validate at startup that all 4 specialist LoRA files exist and are readable. **Fail startup with a clear error if any specialist is missing** — do not silently fall back to a partial specialist set.

---

## 7. Migration Plan

### Phase 0 — current state

Single unified LoRA stays in `TRYON_LORA_PATH`. No code changes.

### Phase 1 — runtime infrastructure for multi-LoRA

Extend `app/clients/qwen_tryon_aitk.py` and `app/runtime/tryon_runtime.py`:

- Base model loaded once
- 4 `LoRASpecialNetwork` instances keyed by category, all loaded at startup
- `set_active_lora(category)` method swaps the attached network
- Output dimensions come from the user image, not from config
- Prompt construction follows §3.3 (trigger + product detail + identity clause)

Add the routing function (§4.1).

Behind a feature flag `TRYON_USE_SPECIALISTS=false` so production is unaffected until the flag is flipped.

### Phase 2 — shadow rollout per category

With `TRYON_USE_SPECIALISTS=true`, validate the chain end-to-end against AI-Toolkit sample parity. Roll out specialist by specialist:

1. `dress` — already validated through 2250 steps with strong sample quality, lowest risk
2. `bottom` — best training loss improvement (0.061 → 0.046), clean shape transfer at CFG 1.0
3. `top` — validates the "no top→dress confusion" claim in production
4. `multi` — last; flat training loss but visual progression; needs the most careful watching

Each rollout is gated on the acceptance criteria in §9.

### Phase 3 — retire the unified path

Once all 4 specialists are stable in production:

- Remove `TRYON_USE_SPECIALISTS` flag (specialists become the only AI-Toolkit path)
- Remove `TRYON_LORA_PATH` from `TRYON_BACKEND=ai_toolkit` config
- Free-form prompt section templates removed from code if they have no other consumer

---

## 8. Targets

The build should achieve:

1. **Production outputs match AI-Toolkit sample outputs** at the same checkpoint for the same category at the same CFG, with only image-codec-level differences
2. **No top→dress extension** — `top` requests preserve the original lower-body clothing
3. **Correct shape transfer at `guidance_scale: 1.0`** — production does not rely on high CFG to mask training weakness
4. **Output returned at the user image's own dimensions and aspect ratio**, not stretched/cropped to a fixed size
5. **Per-category iteration is decoupled** — improving the top LoRA does not require retraining or redeploying the other 3
6. **Inference latency unchanged** compared to the current single-LoRA path (only one LoRA active per inference; switching is sub-millisecond)

---

## 9. Acceptance Tests

Before promoting any specialist to production:

| Test | Pass criteria |
|---|---|
| Sample parity | Production output for a held-out person+garment pair matches the AI-Toolkit training sample image at the same checkpoint, within JPEG-codec pixel differences |
| Top preservation | For 10 held-out top requests, all 10 outputs preserve the original lower-body clothing — no dress extension |
| Shape transfer at CFG 1.0 | For 10 held-out bottom requests (shorts→pants, pants→shorts, etc.), the output garment length matches the input garment, not the input person |
| Output dimensions | Output width/height exactly match the input user image dimensions |
| Latency | p50 inference latency within 10% of the current single-LoRA path |
| Concurrent safety | 50 concurrent requests with rotating categories all succeed with no LoRA-swap race conditions |
| Disabled-state safety | `TRYON_BACKEND=disabled` still returns the controlled `503` response defined in `tryon-ai-toolkit-inference-parity.md` |

---

## 10. Pod / Training Fleet SSH Reference

Three RunPod instances are currently running specialist training. The dress specialist completed earlier on a 4th pod (since reassigned to bottom training; checkpoints still on that pod's overlay disk — see Section 10.5).

SSH key for all four pods is `~/.ssh/runpod_ed25519`.

### 10.1 Pod — `confident_emerald_tuna` (TOP specialist)

```text
SSH:          ssh root@213.173.111.96 -p 35162 -i ~/.ssh/runpod_ed25519
ID:           1vuy5wt1bqoclx
HTTP UI:      port 8675 (Bearer "password")
Job name:     qwen_lora_glamtoptryon_v1_rank16_batch4_dev5000
Output dir:   /app/ai-toolkit/output/qwen_lora_glamtoptryon_v1_rank16_batch4_dev5000/
Dataset:      /mnt/tryon-data/datasets/qwen_lora_glamtoptryon_v1/
Eval samples: /mnt/tryon-data/datasets/qwen_lora_glamtoptryon_v1/eval_samples/
Trigger:      "Apply GlamifyTopTryon on this person"
Dataset rows: 230
```

### 10.2 Pod — `integral_azure_lamprey` (BOTTOM specialist)

```text
SSH:          ssh root@216.81.245.23 -p 18058 -i ~/.ssh/runpod_ed25519
ID:           b36u6p468hcqyj
HTTP UI:      port 8675 (Bearer "password")
Job name:     qwen_lora_glambottomtryon_v1_rank16_batch4_dev5000
Output dir:   /app/ai-toolkit/output/qwen_lora_glambottomtryon_v1_rank16_batch4_dev5000/
Dataset:      /mnt/tryon-data/datasets/qwen_lora_glambottomtryon_v1/
Eval samples: /mnt/tryon-data/datasets/qwen_lora_glambottomtryon_v1/eval_samples/
Trigger:      "Apply GlamifyBottomTryon on this person"
Dataset rows: 259
```

### 10.3 Pod — `still_salmon_rodent` (MULTI specialist)

```text
SSH:          ssh root@216.81.245.23 -p 35506 -i ~/.ssh/runpod_ed25519
ID:           jsvttsxunqrstq
HTTP UI:      port 8675 (Bearer "password")
Job name:     qwen_lora_glammultitryon_v1_rank16_batch4_dev5000
Output dir:   /app/ai-toolkit/output/qwen_lora_glammultitryon_v1_rank16_batch4_dev5000/
Dataset:      /mnt/tryon-data/datasets/qwen_lora_glammultitryon_v1/
Eval samples: /mnt/tryon-data/datasets/qwen_lora_glammultitryon_v1/eval_samples/
Trigger:      "Apply GlamifyMultiTryon on this person"
Dataset rows: 297
```

### 10.4 Pod 2 — DRESS specialist (completed earlier; checkpoints on overlay disk)

The dress specialist completed training to step 2250 on the pod that is now running BOTTOM. Its checkpoints are still on that pod's local overlay disk.

```text
Pod:          integral_azure_lamprey (same SSH as Section 10.2)
Job name:     qwen_lora_dress_v1_rank16_dev5000
Output dir:   /app/ai-toolkit/output_local/qwen_lora_dress_v1_rank16_dev5000/
Dataset:      /mnt/tryon-data/datasets/qwen-lora-dress-v1/
Trigger:      "Apply GlamifyDressTryon on this person"
Dataset rows: 351
Checkpoints:  step 250 / 500 / 750 / 1000 / 1250 / 1500 / 1750 / 2000 / 2250
```

**Important:** the dress checkpoints live on `/app/ai-toolkit/output_local/` which is the pod's local overlay disk, **not** the shared `/mnt/` volume. If the pod is stopped or recycled, those checkpoints are lost. Before promoting a dress checkpoint to production, copy the chosen file to the shared release path:

```bash
ssh -i ~/.ssh/runpod_ed25519 -p 18058 root@216.81.245.23 \
  'mkdir -p /mnt/tryon-data/releases/specialists/v1/dress && \
   cp /app/ai-toolkit/output_local/qwen_lora_dress_v1_rank16_dev5000/qwen_lora_dress_v1_rank16_dev5000_000002000.safetensors \
      /mnt/tryon-data/releases/specialists/v1/dress/dress_v1_rank16_step2000.safetensors'
```

### 10.5 Production release path layout

Production AI service should consume LoRAs from the shared volume under a stable, versioned layout — not from per-pod overlay disks or per-job training output directories.

```text
/mnt/tryon-data/releases/specialists/v1/top/glamtoptryon_v1_rank16_<step>.safetensors
/mnt/tryon-data/releases/specialists/v1/bottom/glambottomtryon_v1_rank16_<step>.safetensors
/mnt/tryon-data/releases/specialists/v1/dress/dress_v1_rank16_<step>.safetensors
/mnt/tryon-data/releases/specialists/v1/multi/glammultitryon_v1_rank16_<step>.safetensors
```

`releases/specialists/v1/<category>/` paths are stable and versioned. Training-output paths under `/app/ai-toolkit/output/<job_name>/` are not stable — checkpoints there can be cleaned up when training is rerun.

### 10.6 Useful read-only commands

List checkpoints (run after SSH onto the pod):
```bash
ls -lh /app/ai-toolkit/output/<job_name>/*.safetensors
```

Inspect job config via the AI-Toolkit API:
```bash
curl -s -H "Authorization: Bearer password" http://127.0.0.1:8675/api/jobs | \
    python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin), indent=2))" | less
```

Inspect loss curve (run on the pod):
```bash
python3 - <<'PY'
import sqlite3
db = "/app/ai-toolkit/output/<job_name>/loss_log.db"
con = sqlite3.connect(db)
for s, v in con.execute("select step, value_real from metrics where key='loss/loss' order by step desc limit 20"):
    print(s, v)
PY
```

Pull a checkpoint to local for production deployment:
```bash
scp -i ~/.ssh/runpod_ed25519 -P <port> -o StrictHostKeyChecking=no \
    root@<host>:/app/ai-toolkit/output/<job_name>/<job_name>_000001500.safetensors \
    /local/destination/
```

---

## 11. Summary

The new try-on architecture:

- **4 rank-16 specialist LoRAs**, one per garment category, all resident in GPU memory
- **One AI-Toolkit runtime** with the base model loaded once; LoRA networks attached/detached per request
- **Application-layer routing**: single-product → category-specific LoRA, multi-product → MULTI specialist (collage handles all combinations)
- **Prompt structure**: trigger caption + product description + identity-preservation clause
- **Output dimensions = user image dimensions** (no fixed resize)
- **Sampling parameters locked** to the AI-Toolkit training-sample values: flowmatch / 25 steps / CFG 1.0 / CFG rescale 0 / do_cfg_norm false / network multiplier 1.0 / seed 43 for parity
- **Versioned LoRA paths** at `/mnt/tryon-data/releases/specialists/v1/<category>/`
- **Phased rollout** behind a `TRYON_USE_SPECIALISTS` feature flag

The single most important property to preserve through implementation:

**production output at a given checkpoint ≡ AI-Toolkit sample output at the same checkpoint**

Every choice in this document serves that property. If any production change risks breaking that equivalence, validate against the sample images before shipping.

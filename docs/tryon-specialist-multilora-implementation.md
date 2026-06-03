# Try-On Specialist Multi-LoRA — Implementation

## Purpose

This document describes the **as-built** state of the `/v1/tryon` specialist multi-LoRA implementation in the Python AI service.

It is a counterpart to:
- [update-tryon-flow.md](update-tryon-flow.md) — the design specification for the migration
- [tryon-flow.md](tryon-flow.md) — the original single-LoRA flow
- [tryon-ai-toolkit-inference-parity.md](tryon-ai-toolkit-inference-parity.md) — parity requirements for the AI-Toolkit runtime

What is documented here is the code that actually exists on disk: file paths, function names, env-var aliases, and runtime behavior verified by reading the source.

The implementation lives behind a feature flag (`TRYON_USE_SPECIALISTS`, default `false`). When the flag is `false`, the legacy single-LoRA path continues to operate unchanged.

---

## 1. High-Level Summary

| Capability | Status | Implementation |
|---|---|---|
| Feature flag for specialists | implemented | `Settings.tryon_use_specialists` (`TRYON_USE_SPECIALISTS`) |
| 4 specialist LoRA paths in config | implemented | `tryon_lora_{top,bottom,dress,multi}_path` |
| Specialist trigger captions in config | implemented | `tryon_prompt_trigger_{top,bottom,dress,multi}` |
| Identity-preservation clause in config | implemented | `tryon_prompt_identity_clause` |
| Category routing | implemented | `app/services/tryon_routing.py::resolve_tryon_route` |
| Multi-LoRA resident loading (all 4) | implemented | `QwenTryonAitkClient._load_specialist_state_dicts` |
| Per-request LoRA switching | implemented | `QwenTryonAitkClient.set_active_specialist` |
| Specialist prompt construction | implemented | `_build_specialist_prompt` in `app/services/tryon.py` |
| Output dimensions = user image dimensions | implemented | `_bucket_dimensions` + post-inference resize |
| Startup validation of specialist paths | implemented | `validate_startup_settings` in `app/config.py` |
| Warmup that loads all 4 specialists | implemented | `QwenTryonAitkClient.warmup` |
| Backward compatibility (single LoRA) | preserved | Branch on `tryon_use_specialists` |
| Response metadata exposes routing | implemented | `metadata.routing` in `TryonResponse` |

---

## 2. Configuration (`app/config.py`)

The following `Settings` fields drive the specialist implementation. All are bound to environment variables via Pydantic aliases.

### 2.1 Feature flag

```python
tryon_use_specialists: bool = Field(default=False, alias="TRYON_USE_SPECIALISTS")
```

When `False`, the service uses the legacy single-LoRA path and the free-form prompt builder.
When `True`, the service uses the specialist routing + LoRA switching + structured trigger prompt.

### 2.2 Specialist checkpoint paths

```python
tryon_lora_top_path:    str = Field(default="", alias="TRYON_LORA_TOP_PATH")
tryon_lora_bottom_path: str = Field(default="", alias="TRYON_LORA_BOTTOM_PATH")
tryon_lora_dress_path:  str = Field(default="", alias="TRYON_LORA_DRESS_PATH")
tryon_lora_multi_path:  str = Field(default="", alias="TRYON_LORA_MULTI_PATH")
```

Each must point at the `.safetensors` file produced by AI-Toolkit for that specialist (rank 16 in v1).

### 2.3 Trigger captions

```python
tryon_prompt_trigger_top:    str = "Apply GlamifyTopTryon on this person"
tryon_prompt_trigger_bottom: str = "Apply GlamifyBottomTryon on this person"
tryon_prompt_trigger_dress:  str = "Apply GlamifyDressTryon on this person"
tryon_prompt_trigger_multi:  str = "Apply GlamifyMultiTryon on this person"
```

Aliases: `TRYON_PROMPT_TRIGGER_{TOP,BOTTOM,DRESS,MULTI}`. Each default value matches the exact caption the corresponding specialist LoRA was trained with.

### 2.4 Identity-preservation clause

```python
tryon_prompt_identity_clause: str = (
    "Preserve the person's face, identity, body proportions, pose, and background."
)
```

Alias: `TRYON_PROMPT_IDENTITY_CLAUSE`. Appended at the end of every specialist prompt.

### 2.5 Sampling defaults (matched to AI-Toolkit training samples)

```python
tryon_default_seed:           int   = 43
tryon_default_steps:          int   = 25
tryon_default_guidance_scale: float = 1.0
tryon_guidance_rescale:       float = 0.0
tryon_do_cfg_norm:            bool  = False
tryon_sampler:                str   = "flowmatch"
tryon_lora_rank:              int   = 16
tryon_lora_alpha:             int   = 16
tryon_lora_scale:             float = 1.0
```

These defaults exactly reproduce the `sample` block of every training pod's running config (`flowmatch` sampler, 25 sample steps, `guidance_scale: 1.0`, `do_cfg_norm: false`, `network_multiplier: 1.0`, seed 43). The request body may override seed / steps / guidance_scale.

### 2.6 Output dimensioning

```python
tryon_dimension_multiple: int = 64
```

Alias: `TRYON_DIMENSION_MULTIPLE`. The service buckets the user image's `(width, height)` to the nearest multiple of this number before calling inference, then resizes the result back to the user's original exact dimensions. AI-Toolkit's training-time bucketing already snaps to multiples of 64; matching this at inference keeps production aligned with the training contract.

### 2.7 Startup validation

`validate_startup_settings` in `app/config.py` requires the 4 specialist LoRA paths to be non-empty **only when** `TRYON_USE_SPECIALISTS=true`. If specialists are enabled and any path is missing, startup fails with a clear error.

Otherwise (`TRYON_USE_SPECIALISTS=false`) those fields can be empty without breaking startup.

---

## 3. Routing (`app/services/tryon_routing.py`)

A small dataclass and one resolver function:

```python
TryonLoraKey = Literal["top", "bottom", "dress", "multi"]

@dataclass(frozen=True)
class TryonRoutingDecision:
    lora_key: TryonLoraKey
    trigger_caption: str

def resolve_tryon_route(
    products: list[TryonProduct],
    settings: Settings,
) -> TryonRoutingDecision: ...
```

### Routing rules (as implemented)

```text
if len(products) >= 2:
    return ("multi",   trigger_caption=settings.tryon_prompt_trigger_multi)

if product.type == "top":     return ("top",    trigger_caption=settings.tryon_prompt_trigger_top)
if product.type == "outer":   return ("top",    trigger_caption=settings.tryon_prompt_trigger_top)
if product.type == "bottom":  return ("bottom", trigger_caption=settings.tryon_prompt_trigger_bottom)
if product.type == "dress":   return ("dress",  trigger_caption=settings.tryon_prompt_trigger_dress)
otherwise:                    raise ValueError(...)
```

Properties:
- 2 or more products → always MULTI (collage handles the combination)
- `outer` is mapped to the TOP specialist (no separate outer LoRA exists in v1)
- Unsupported product types fail loudly with `ValueError`

The resolver is pure and side-effect-free, easy to unit-test.

---

## 4. Service Layer (`app/services/tryon.py`)

### 4.1 Branch on the feature flag

```python
routing_decision: TryonRoutingDecision | None = None
if resolved_settings.tryon_use_specialists:
    routing_decision = resolve_tryon_route(payload.products, resolved_settings)
    prompt_text = _build_specialist_prompt(
        payload.products, routing_decision, resolved_settings
    )
else:
    prompt_text = _build_tryon_prompt(payload)
```

The specialist path produces a routing decision and a structured prompt. The legacy path uses the free-form prompt builder (unchanged from before this migration).

### 4.2 Specialist prompt structure

`_build_specialist_prompt(products, routing, settings)` produces:

```text
{trigger_caption}. {product_section_or_sections}. {identity_clause}
```

#### Single-category path (`top` / `bottom` / `dress`)

`_build_specialist_product_sections` returns:

```text
{Label}: {product.prompt}.
```

where `Label` is `Top`, `Bottom`, or `Dress` based on the routed `lora_key`.

#### Multi-category path

`_build_ordered_product_sections` orders the products by priority and produces:

```text
Top: {top description}. Dress: {dress description}. Bottom: {bottom description}.
```

Ordering rule:
- priority 0: `top` / `outer` (outer renders as label "Top")
- priority 1: `dress`
- priority 2: `bottom`
- ties broken by original request order

If a product's `prompt` is empty after stripping, that section is omitted (no `Top: .` artifact).

### 4.3 Output dimensioning

```python
user_width, user_height = user_image.width, user_image.height
output_width, output_height = _bucket_dimensions(
    user_width, user_height,
    settings.tryon_dimension_multiple,
)
```

`_bucket_dimensions` rounds each dimension to the nearest `multiple` (default 64), with a floor at the multiple itself:

```python
def _bucket_dimensions(width, height, multiple):
    if multiple <= 1:
        return int(width), int(height)
    return (
        max(multiple, round(width  / multiple) * multiple),
        max(multiple, round(height / multiple) * multiple),
    )
```

Inference runs at `(output_width, output_height)`. After inference, the service resizes the output back to the user's exact original dimensions:

```python
if output_image.size != (user_width, user_height):
    output_image = output_image.resize(
        (user_width, user_height),
        Image.Resampling.LANCZOS,
    )
    output_image.save(job_paths.output_path, format="JPEG", quality=95)
```

This means:
- The model sees an AI-Toolkit-bucketed shape (training-aligned)
- The client receives the response shape it sent (no aspect-ratio surprise)
- Both inference and final dimensions are recorded in the response metadata

### 4.4 Runner call

The service calls the resident runner via the coordinator:

```python
run_result = get_tryon_execution_coordinator(settings).run(
    lambda: get_tryon_runner(settings).run_tryon(
        person_image_path=str(job_paths.person_path),
        garment_reference_path=str(job_paths.garment_reference_path),
        prompt=prompt_text,
        steps=resolved_steps,
        guidance_scale=resolved_guidance_scale,
        seed=resolved_seed,
        output_path=str(job_paths.output_path),
        output_width=output_width,
        output_height=output_height,
        lora_key=routing_decision.lora_key if routing_decision else None,
    ),
)
```

`lora_key` is included only when specialists are active. The runner uses it to choose which LoRA to attach.

### 4.5 Response metadata

The response includes a `routing` block:

```json
"routing": {
    "use_specialists": true,
    "lora_key": "top",
    "trigger_caption": "Apply GlamifyTopTryon on this person"
}
```

And dimensioning information:

```json
"output": {
    "width":             <user width>,
    "height":            <user height>,
    "inference_width":   <bucketed width>,
    "inference_height":  <bucketed height>
}
```

The `runner` block (from `QwenTryonAitkClient.run_tryon`) additionally carries the active `checkpoint_path`, `lora_key`, `lora_rank`, `lora_alpha`, `sampler`, and the full `control_order` mapping.

---

## 5. Runtime Client (`app/clients/qwen_tryon_aitk.py`)

This is the AI-Toolkit-backed try-on client. It is responsible for:
- loading AI-Toolkit and Qwen-Image-Edit-Plus
- creating the `LoRASpecialNetwork`
- caching all 4 specialist weight sets
- switching the active specialist per request
- calling `pipeline.generate_images(...)` with the parity-locked configuration

### 5.1 Construction

```python
class QwenTryonAitkClient:
    SPECIALIST_CATEGORIES = ("top", "bottom", "dress", "multi")

    def __init__(self, settings: Settings):
        self._use_specialists = bool(settings.tryon_use_specialists)
        self._specialist_paths = {
            "top":    Path(settings.tryon_lora_top_path),
            "bottom": Path(settings.tryon_lora_bottom_path),
            "dress":  Path(settings.tryon_lora_dress_path),
            "multi":  Path(settings.tryon_lora_multi_path),
        }
        # also tracks: pipeline, network, loaded_checkpoint,
        # specialist_state_dicts, active_specialist, locks, etc.
```

### 5.2 Base model + network setup (one-time, lazy)

`_load_runtime()` runs once on first request (lazy) or during warmup. It does:

1. `os.chdir(AI_TOOLKIT_ROOT)` and `sys.path.insert(0, AI_TOOLKIT_ROOT)` to match the validated RunPod startup pattern
2. Imports `toolkit.config_modules`, `toolkit.lora_special`, `toolkit.train_tools`, `toolkit.util.get_model`
3. Builds `ModelConfig(arch="qwen_image_edit_plus", quantize=False, low_vram=False, ...)`
4. Builds `NetworkConfig(type="lora", linear=rank, linear_alpha=alpha)`
5. Loads the Qwen model class returned by `get_model_class(model_config)`
6. Moves the pipeline to `cuda:0` with `bf16` dtype
7. Creates **one** `LoRASpecialNetwork` instance attached to the base model
8. Calls `network.apply_to(text_encoder, unet, ...)` to hook the network into the model

The result: one pipeline, one network, attached and live. The 4 specialists share this same network — only its weights are swapped at request time.

### 5.3 Specialist weight caching (the resident-LoRA pattern)

`_load_specialist_state_dicts()` runs once at warmup and populates a dict of cloned state dicts:

```python
for category in SPECIALIST_CATEGORIES:
    self._network.load_weights(str(self._specialist_paths[category]))
    snapshot = OrderedDict(
        (key, tensor.detach().clone())
        for key, tensor in self._network.state_dict().items()
    )
    self._specialist_state_dicts[category] = snapshot
# Attach the first specialist by default
self._network.load_state_dict(self._specialist_state_dicts["top"], strict=False)
self._active_specialist = "top"
```

This means:
- All 4 specialist weight sets are cloned to memory (one cloned `OrderedDict[str, Tensor]` per category)
- Memory cost is ~4 × (rank-16 weight size) ≈ ~4 × 280 MB ≈ 1.1 GB of LoRA weight residency
- The base model exists once
- Per-request switching is a `network.load_state_dict(snapshot, strict=False)` call — sub-millisecond, no disk read

### 5.4 Switching the active specialist

```python
def set_active_specialist(self, category: str) -> None:
    if not self._use_specialists:
        return
    if category not in SPECIALIST_CATEGORIES:
        raise TryonRuntimeError(f"Unknown try-on specialist category: {category}")
    self._ensure_ready()
    self._load_specialist_state_dicts()
    if self._active_specialist == category:
        return
    state = self._specialist_state_dicts[category]
    self._network.load_state_dict(state, strict=False)
    self._active_specialist = category
```

Properties:
- Idempotent — calling with the already-active category is a no-op
- Lazy-safe — if specialists were not yet cached, this populates the cache first
- Disabled-flag-safe — if `tryon_use_specialists=False`, this is a no-op (legacy path takes over)

### 5.5 Inference entry point

`run_tryon(...)` performs:

1. If specialists are enabled and `lora_key` is provided → `set_active_specialist(lora_key)`
2. If specialists are disabled → fall back to single-LoRA `_load_checkpoint_if_needed()` (loads from `TRYON_LORA_PATH`)
3. Build `GenerateImageConfig(...)` with all the parity-locked fields:

   ```python
   GenerateImageConfig(
       prompt=prompt.strip(),
       width=output_width,
       height=output_height,
       negative_prompt="",
       seed=int(seed),
       guidance_scale=float(guidance_scale),
       guidance_rescale=float(settings.tryon_guidance_rescale),
       num_inference_steps=int(steps),
       network_multiplier=float(settings.tryon_lora_scale),
       output_path=str(output_file),
       output_ext="jpg",
       ctrl_img_1=str(person_image_path),       # person
       ctrl_img_2=str(garment_reference_path),  # garment
       do_cfg_norm=bool(settings.tryon_do_cfg_norm),
   )
   ```

4. Call `pipeline.generate_images([conf], sampler=settings.tryon_sampler)` inside `torch.inference_mode()` and an inference lock
5. Open the produced JPEG, attach a metadata dict, return a `TryonRunResult`

### 5.6 Locks and concurrency

- `_load_lock` — guards `_load_runtime` from concurrent first-load
- `_infer_lock` — serializes `pipeline.generate_images(...)` calls; only one inference at a time

The route-level `BoundedExecutionCoordinator` already bounds queue size and wait time before calls reach the runner, so the inference lock is effectively a guard for safety rather than for throughput control.

### 5.7 Status reporting

```python
def status(self) -> TryonRuntimeStatus:
    lora_loaded = (
        bool(self._specialist_state_dicts)
        if self._use_specialists
        else self._loaded_checkpoint is not None
    )
    return TryonRuntimeStatus(
        loaded=self._pipeline is not None and self._network is not None,
        backend="ai_toolkit_exact" if self._pipeline is not None else None,
        lora_loaded=lora_loaded,
    )
```

The health endpoint surfaces this, so an operator can check whether (a) the runtime is up and (b) the specialists are actually loaded.

---

## 6. Runtime Cache (`app/runtime/tryon_runtime.py`)

The runner is cached by `lru_cache(maxsize=8)`, keyed by **all** parameters that affect runtime behavior, including:

- `ai_toolkit_root`
- `qwen_image_edit_model_path`
- `tryon_use_specialists`
- `tryon_lora_path` (single-LoRA path for the legacy fallback)
- `tryon_lora_top_path`, `tryon_lora_bottom_path`, `tryon_lora_dress_path`, `tryon_lora_multi_path`
- `tryon_lora_rank`, `tryon_lora_alpha`, `tryon_lora_scale`
- `tryon_sampler`, `tryon_guidance_rescale`, `tryon_do_cfg_norm`

Changing any of these (e.g., swapping one specialist path to a new checkpoint) invalidates the cached runner. The next call constructs a fresh `QwenTryonAitkClient` and reloads the base model + specialists.

The bounded execution coordinator is cached independently by `(max_queue_size, queue_wait_timeout_seconds)` so a checkpoint change does not unnecessarily reset the request queue.

`warmup_tryon_runtime(settings)` calls `runner.warmup()`, which loads the base model and (if specialists are enabled) populates all 4 cached state dicts. This is invoked from `app/runtime/warmup.py` during application startup.

---

## 7. Request Flow at Runtime

When a request hits `POST /v1/tryon` with `TRYON_USE_SPECIALISTS=true`:

```text
1. app/routes/tryon.py
   - JWT middleware validates auth
   - Route accepts TryonRequest
   - Dispatches to run_tryon_request via run_in_threadpool

2. app/services/tryon.py :: run_tryon_request
   - Resolves defaults from Settings (seed=43, steps=25, CFG=1.0, etc.)
   - Downloads user_image, opens as RGB
   - Buckets (user_width, user_height) -> (output_width, output_height)
   - Downloads each product image
   - Builds ProductReferenceInput list
   - Calls build_product_reference -> collapsed garment_reference image
   - Saves person.jpg and garment_reference.jpg under the request job_dir
   - resolve_tryon_route(products, settings) -> routing_decision
   - _build_specialist_prompt(products, routing_decision, settings) -> prompt_text

3. app/runtime/coordinator.py
   - Coordinator admits the request (queue bound)
   - Calls runner.run_tryon(...) with lora_key from routing_decision

4. app/clients/qwen_tryon_aitk.py :: run_tryon
   - set_active_specialist(lora_key) -> swaps in the right state_dict
   - Builds GenerateImageConfig with parity-locked fields
   - Acquires inference lock; runs pipeline.generate_images(..., sampler="flowmatch")
   - Opens the produced JPEG, builds metadata, returns TryonRunResult

5. app/services/tryon.py (back from runner)
   - If output dimensions != user dimensions, resize the output to user dimensions
   - Save resized output back to job_dir
   - Upload to Azure via AzureStorageClient
   - Build TryonResponse with full metadata (routing, runner, output, request, etc.)

6. Cleanup
   - The request-local job_dir is removed in finally
   - Only the Azure-uploaded image persists
```

When `TRYON_USE_SPECIALISTS=false`, steps 2 and 4 instead use the legacy free-form prompt builder and the single-LoRA checkpoint at `TRYON_LORA_PATH`. Everything else is identical.

---

## 8. Parity With AI-Toolkit Samples

The implementation matches the locked training-sample contract:

| Parameter | Training pod | Implementation source |
|---|---|---|
| `arch` | `qwen_image_edit_plus` | `ModelConfig(arch="qwen_image_edit_plus", ...)` |
| `sampler` | `flowmatch` | `Settings.tryon_sampler="flowmatch"`, passed to `generate_images(..., sampler=...)` |
| `sample_steps` | 25 | `Settings.tryon_default_steps=25` |
| `guidance_scale` | 1.0 | `Settings.tryon_default_guidance_scale=1.0` |
| `guidance_rescale` | 0.0 | `Settings.tryon_guidance_rescale=0.0` |
| `do_cfg_norm` | false | `Settings.tryon_do_cfg_norm=False` |
| `network_multiplier` | 1.0 | `Settings.tryon_lora_scale=1.0` |
| `seed` | 43 | `Settings.tryon_default_seed=43` |
| Control image order | `ctrl_img_1=person`, `ctrl_img_2=garment` | `GenerateImageConfig(ctrl_img_1=person_image_path, ctrl_img_2=garment_reference_path)` |
| Training resolution bucket | multiples of 64 | `Settings.tryon_dimension_multiple=64` + `_bucket_dimensions` |
| LoRA rank/alpha | 16/16 | `Settings.tryon_lora_rank=16`, `tryon_lora_alpha=16` |
| Trigger captions | exact strings from training data | hardcoded defaults in `Settings.tryon_prompt_trigger_{top,bottom,dress,multi}` |
| LoRA loader | `LoRASpecialNetwork.load_weights(...)` | matches AI-Toolkit's own training loader |

The only deliberate divergence from the trained samples is at the prompt level — production prompts append a product description and the identity clause to the trigger caption. The trigger remains the first sentence of the prompt, which preserves the specialist activation.

---

## 9. Configuration Examples

### 9.1 Run in specialist mode

```env
TRYON_BACKEND=ai_toolkit
TRYON_USE_SPECIALISTS=true

AI_TOOLKIT_ROOT=/mnt/tryon-data/ai-toolkit
QWEN_IMAGE_EDIT_MODEL_PATH=/mnt/models/qwen-image-edit-2511

TRYON_LORA_TOP_PATH=/mnt/tryon-data/releases/specialists/v1/top/glamtoptryon_v1_rank16_step1500.safetensors
TRYON_LORA_BOTTOM_PATH=/mnt/tryon-data/releases/specialists/v1/bottom/glambottomtryon_v1_rank16_step1500.safetensors
TRYON_LORA_DRESS_PATH=/mnt/tryon-data/releases/specialists/v1/dress/dress_v1_rank16_step2000.safetensors
TRYON_LORA_MULTI_PATH=/mnt/tryon-data/releases/specialists/v1/multi/glammultitryon_v1_rank16_step2000.safetensors

TRYON_LORA_RANK=16
TRYON_LORA_ALPHA=16
TRYON_LORA_SCALE=1.0

TRYON_DEFAULT_SEED=43
TRYON_DEFAULT_STEPS=25
TRYON_DEFAULT_GUIDANCE_SCALE=1.0
TRYON_GUIDANCE_RESCALE=0.0
TRYON_DO_CFG_NORM=false
TRYON_SAMPLER=flowmatch
TRYON_DIMENSION_MULTIPLE=64
```

### 9.2 Run in legacy single-LoRA mode (backward compatibility)

```env
TRYON_BACKEND=ai_toolkit
TRYON_USE_SPECIALISTS=false

TRYON_LORA_PATH=/mnt/.../unified_lora.safetensors
TRYON_LORA_RANK=32
TRYON_LORA_ALPHA=32
# specialist paths can stay empty when use_specialists=false
```

In this mode the routing and structured prompt are bypassed and the original free-form prompt builder is used.

---

## 10. Where Each Implementation Piece Lives

| Concern | File | Symbol |
|---|---|---|
| Feature flag, config fields, startup validation | `app/config.py` | `Settings`, `validate_startup_settings` |
| Category-to-LoRA routing | `app/services/tryon_routing.py` | `TryonRoutingDecision`, `resolve_tryon_route` |
| Specialist prompt construction | `app/services/tryon.py` | `_build_specialist_prompt`, `_build_specialist_product_sections`, `_build_ordered_product_sections`, `_category_label_for_lora` |
| Dimension bucketing | `app/services/tryon.py` | `_bucket_dimensions` |
| Service orchestration | `app/services/tryon.py` | `run_tryon_request` |
| AI-Toolkit runtime client | `app/clients/qwen_tryon_aitk.py` | `QwenTryonAitkClient`, `SPECIALIST_CATEGORIES` |
| All-specialists-resident loader | `app/clients/qwen_tryon_aitk.py` | `_load_specialist_state_dicts` |
| Per-request LoRA switch | `app/clients/qwen_tryon_aitk.py` | `set_active_specialist` |
| Inference call | `app/clients/qwen_tryon_aitk.py` | `run_tryon` |
| Runner cache + warmup | `app/runtime/tryon_runtime.py` | `get_tryon_runner`, `_get_tryon_runner_cached`, `warmup_tryon_runtime` |
| Coordinator (queue) | `app/runtime/tryon_runtime.py` | `get_tryon_execution_coordinator` |
| Health surface | `app/services/health.py` (via `runtime.status()`) | `TryonRuntimeStatus` |

---

## 11. Operational Notes

### 11.1 Cold-start cost

When `TRYON_USE_SPECIALISTS=true`, warmup loads the base model **once** and then sequentially loads + snapshots **four** specialist weight sets. Expect roughly 30–60 sec of additional warmup time vs the single-LoRA path. After warmup, per-request LoRA switching is essentially free.

### 11.2 Memory footprint

| Component | Approx VRAM |
|---|---|
| Qwen-Image-Edit-Plus base (bf16) | ~40 GB |
| `LoRASpecialNetwork` (rank 16, active) | ~280 MB |
| 4 cached specialist state-dicts (CPU/GPU memory) | ~1.1 GB |
| Inference activations at 1024×1536 batch 1 | ~10–15 GB |
| Total peak | ~55 GB on H100 NVL (96 GB available) |

This pattern (clone state dicts vs allocate 4 networks) was chosen specifically so the LoRA hook structure remains attached to the base model once; only weights swap.

### 11.3 Backward compatibility guarantees

- If `TRYON_USE_SPECIALISTS=false`, the legacy single-LoRA path runs unchanged. The free-form prompt templates (`TRYON_SINGLE_REFERENCE_PROMPT`, `TRYON_TOP_SECTION_TEMPLATE`, etc.) remain in `app/services/tryon.py` for this case.
- The startup validation only requires the 4 specialist paths when the flag is on.
- The route, JWT middleware, queue coordinator, storage upload, cleanup, and metadata shape are unchanged from the original `tryon-flow.md` design — only the prompt builder + LoRA layer are swapped behind the flag.

### 11.4 Switching back to unified LoRA in production

Setting `TRYON_USE_SPECIALISTS=false` (and restarting the service) takes the system back to single-LoRA mode without any code change. Useful for rolling back if a specialist release regresses.

---

## 12. Spec vs Implementation — Two Implementation-Only Choices

The implementation matches the spec in [update-tryon-flow.md](update-tryon-flow.md). Two implementation-only choices worth documenting:

1. **One `LoRASpecialNetwork` + cached state dicts, not four `LoRASpecialNetwork` instances.** The spec described "4 resident LoRAs." The implementation achieves the same effective behavior by attaching one network to the base model and swapping its `state_dict` per request. This avoids duplicating the hook structure and keeps a single inference path. Memory footprint and switching cost both meet the spec.

2. **Bucketed dimensions are implemented in the service layer**, not in the runtime client. The client just receives `(output_width, output_height)` and uses them. The bucketing + post-resize logic lives in `app/services/tryon.py` (`_bucket_dimensions` + the `output_image.resize(...)` block). This keeps the runtime client framework-agnostic and matches AI-Toolkit's own bucketing during training.

Neither change affects the production parity contract.

---

## 13. Acceptance Test Coverage (matching §9 of `update-tryon-flow.md`)

For each item, where in the code it can be exercised:

| Test | Where to validate |
|---|---|
| Sample parity | Compare an output of `run_tryon_request(...)` to the AI-Toolkit `samples/` JPEG at the same checkpoint + same controls + seed=43 + steps=25 + CFG=1.0 |
| Top preservation | Send 10 top requests with bottom clothing visible in the user image; the response output should retain the original bottom |
| Shape transfer at CFG 1.0 | Send 10 bottom requests with garment != input bottom shape; output garment shape should follow the garment ctrl, not the person |
| Output dimensions = user image dimensions | Inspect `metadata.output.width / height` vs `metadata.output.inference_width / inference_height`; final size should match the user image |
| Concurrent safety | Send 50 concurrent requests with rotating categories; verify all succeed and `_infer_lock` serializes properly |
| Latency | Measure p50 with `TRYON_USE_SPECIALISTS=true` vs `false`; per-request overhead from `set_active_specialist` should be sub-millisecond after warmup |
| Disabled-state | Set `TRYON_BACKEND=disabled` (per existing parity doc) and confirm the controlled 503 response still ships |

---

## 14. Summary

The specialist multi-LoRA implementation is in place and gated by `TRYON_USE_SPECIALISTS`. The 4 specialists share one `LoRASpecialNetwork` attached to one resident Qwen-Image-Edit-Plus base model; their weights are cached as cloned state dicts and swapped per request based on a deterministic routing decision keyed on `products[].type`. Production output dimensions match the user image (with internal bucketing for training-aligned inference). All sampling parameters default to the validated training-sample contract.

When the flag is off, the legacy single-LoRA + free-form prompt path runs unchanged, providing a safe rollback target.

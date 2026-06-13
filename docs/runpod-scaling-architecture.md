# RunPod Scaling Architecture — Glamify Image AI

> System design for taking the single-pod `glamify-image-ai` service to MVP scale
> (5k–10k daily users) on RunPod, with the existing Azure backend as control plane.
> Date: 2026-06-11. Grounded in the current codebase (`app/`, `docs/`) and measured
> numbers from `docs/runpod-glamify-image-ai-setup.md` and
> `docs/startup-optimization-implementation-handoff.md`.

---

## 1. Requirements

### Functional

- Serve the four GPU APIs: `/v1/wardrobe`, `/v1/tryon`, `/v1/user_validation`, `/v1/upscale`.
- Preserve current warm latencies: wardrobe 6–7 s, tryon ~15 s (incl. 2730 upscale),
  validation 3–5 s, standalone upscale 12–15 s.
- Async delivery is acceptable for the slow endpoints (job id + poll/webhook) — confirmed.
- Existing product backend stays on Azure; images already flow through Azure Blob.

### Non-functional

- **Warm is mandatory.** Cold start is 13–15 min today (~3–4 min after the planned
  startup optimizations). A request must never wait on a cold pod.
- **Unknown traffic.** 5k–10k DAU estimated for the first 1–2 months; peak concurrency unknown.
  The design must scale both directions without re-architecture.
- **EU compliance (GDPR).** GPU processing and image storage must stay in EU regions,
  with a DPA in place and strict data-retention behavior on ephemeral workers.
- **Budget flexible**, but cost must scale with usage, not with fear of usage.

### Constraints from the codebase (do not fight these)

- One GPU = one job at a time. All GPU work serializes through
  `BoundedExecutionCoordinator` (queue 8, wait timeout 30 s, hang watchdog 300 s).
  This is correct for a shared-weights diffusion runtime — scaling means **more pods,
  not more parallelism per pod**.
- Wardrobe and tryon share the same resident Qwen-Image-Edit-2511 weights (~54 GB bf16,
  fp8 in production) with per-task LoRA adapters. They cannot be split across workers
  without doubling the heaviest model.
- Full resident set (Qwen + MiniCPM AWQ + detector + SigLIP + SeedVR2) peaks ~88 GB →
  needs the 96 GB class (RTX PRO 6000 / "6000s PRO").
- `/ready` already gates on full warmup and degrades on a wedged GPU — this is exactly
  the readiness contract an autoscaled fleet needs. Keep it.

---

## 2. Recommendation in one paragraph

Run the GPU layer as **RunPod Serverless endpoints in an EU data center** (EU-RO-1 or
EU-SE-1 — wherever 96 GB workers and network volumes coexist), split into a **heavy
endpoint** (Qwen wardrobe/tryon + SeedVR2 upscale, 96 GB workers) and a **light endpoint**
(MiniCPM validation stack, 24 GB workers). Keep a small floor of always-warm workers and
let flex workers absorb spikes, scaled on queue delay. The Azure backend becomes the
job orchestrator: it accepts API calls, enqueues RunPod jobs with a webhook, stores job
state, and serves results from Azure Blob (EU region). Once 4–6 weeks of real traffic
data exists, move the steady base load onto reserved RunPod **pods** behind the same job
API to cut the floor cost roughly in half — the async boundary makes that swap invisible
to the product.

Why serverless first, not a pod fleet + own load balancer: a pod fleet needs you to build
and operate queueing, autoscaling, health-based routing, pod recycling, and webhook
infrastructure yourself — several weeks of work that competes with the MVP. RunPod
serverless ships all of it (bounded queue, queue-delay autoscaler, webhooks, per-second
billing, FlashBoot). At MVP traffic the serverless premium is small in absolute terms;
at scale you revisit (see §9).

---

## 3. High-level design

```text
                         AZURE (EU region) — control plane
  ┌──────────────────────────────────────────────────────────────────┐
  │  Product backend (existing)                                      │
  │   ├─ POST /v1/tryon|wardrobe|upscale  → create job, 202 {job_id} │
  │   ├─ POST /v1/user_validation        → sync passthrough (runsync)│
  │   ├─ GET  /v1/jobs/{id}              → status/result (poll)      │
  │   ├─ Job store (Postgres/Cosmos table: job_id, user_id, status,  │
  │   │            input/output blob keys, timings, error)           │
  │   └─ POST /internal/runpod-webhook   → job completion callback   │
  │  Azure Blob (EU): input images (short-lived SAS) + outputs       │
  └────────────┬─────────────────────────────────┬───────────────────┘
               │ /run (async, webhook)           │ /runsync
               ▼                                 ▼
  ┌─────────────────────────────┐   ┌─────────────────────────────┐
  │ RunPod Serverless: HEAVY    │   │ RunPod Serverless: LIGHT    │
  │ EU DC, 96 GB workers        │   │ EU DC, 24 GB workers (L4/   │
  │ Qwen fp8 + LoRAs + SeedVR2  │   │ A5000): MiniCPM AWQ +       │
  │ wardrobe | tryon | upscale  │   │ detector + SigLIP           │
  │ active floor + flex burst   │   │ 1 active + flex burst       │
  │ queue-delay autoscale       │   │                             │
  │ models on EU network volume │   │                             │
  └─────────────────────────────┘   └─────────────────────────────┘
```

Request flow (tryon, the hot path):

1. App → Azure backend `POST /v1/tryon` with the validated user image reference.
2. Backend writes job row (`queued`), uploads/locates input in Blob, calls RunPod
   `/run` with `{job payload, short-lived SAS urls}` and a webhook URL. Returns
   `202 {job_id}` to the app immediately.
3. RunPod queues the job; a warm worker picks it up, pulls input via SAS, runs
   Qwen gen (832×1248) → SeedVR2 to 1820×2730 (~2.4 s), uploads output to Blob.
4. RunPod fires the webhook → backend marks `succeeded` + output key → app gets it
   via poll (or push notification). Warm end-to-end target: **≤ 20 s** for tryon,
   of which ~15 s is GPU — same as today.
5. Validation stays synchronous via `/runsync` on the light endpoint (3–5 s fits
   comfortably inside an HTTP request).

### Why split heavy and light

- Today a 3–5 s validation can sit behind a 15 s tryon in the single GPU queue.
  Splitting removes the worst tail latency from the cheapest, most frequent call.
- MiniCPM AWQ (+ detector + SigLIP) fits in 24 GB → ~6× cheaper per second than the
  96 GB class, and its cold start is ~2–2.5 min instead of 13–15.
- It frees ~10–12 GB on the heavy worker — headroom for SeedVR2 peaks, and it keeps
  the door open to 80 GB GPUs (H100/A100) if 96 GB availability gets tight in EU DCs.
- Cost of the split: maintain two worker images and the wardrobe flow must call the
  light endpoint for its captioning step (service-to-service hop, ~tens of ms).
  Acceptable; the wardrobe captioning call can also be made by the Azure orchestrator
  between steps.

---

## 4. Capacity model (explicit assumptions — revisit with real data)

Per-job GPU seconds (measured, warm): validation 4 s (light), wardrobe 6.5 s,
tryon 15 s, standalone upscale 13 s. Assumed session = 1 validation + 2 wardrobe +
2 tryon + 1 upscale ≈ **56 heavy GPU-seconds**. Assume 20% of a day's sessions land
in the peak hour. One worker does one job at a time, target 70% utilization.

| Scenario | DAU | Generating users | Sessions/day | Peak concurrent GPU demand | Heavy workers at peak |
|---|---:|---:|---:|---:|---:|
| Low | 5,000 | 20% | 1,000 | ~3.1 | **4–5** |
| Mid | 7,500 | 30% | 2,250 | ~7.0 | **10** |
| High | 10,000 | 50% | 5,000 | ~15.6 | **22** |

Formula to re-run with live numbers:
`workers = (peak_sessions_per_hour × 56 s) / 3600 / 0.70`.

Reading: this is **not** a 1–2 GPU problem at the mid scenario. A 15 s tryon
serialized per GPU means one worker clears only ~4 tryons/minute. The architecture
must make adding/removing the Nth worker boring — which is the whole argument for
the serverless queue + autoscaler.

### Suggested endpoint settings (heavy)

- Active (always-warm) floor: **2 workers** at launch (raise after week 1 data;
  scale-to-zero is forbidden by the warm-mandatory rule).
- Max workers: **12** at launch (cap blast radius and spend; raise alongside data).
- Autoscale: queue-delay based; target P95 queue wait ≤ 10 s.
- Idle timeout: **10–15 min** (cold start is minutes; never thrash workers down
  seconds after a spike). FlashBoot on — it can resume recently-stopped workers
  fast, but treat it as opportunistic, not a substitute for the active floor.
- Execution timeout: 300 s (matches the in-repo watchdog).
- Light endpoint: 1 active + flex burst, max ~5.

---

## 5. Cost model (RunPod list prices, June 2026)

96 GB serverless ("6000s PRO"): **$0.00111/s** flex ≈ $4.00/hr; active workers get
a negotiated discount (assume ~30% → ~$67/day vs $96/day flex, confirm with RunPod
sales). 24 GB (L4/A5000): $0.00019/s. RTX PRO 6000 **pods**: ~$1.79/hr secure
on-demand (~$43/day), spot from ~$0.89/hr. Network volume: $0.07/GB/mo
(~$10/mo for ~140 GB of models/LoRAs).

Per-job GPU cost: tryon **$0.017**, wardrobe $0.007, upscale $0.014, validation
$0.0008 → a full session ≈ **$0.06**. Unit economics are healthy; spend scales
linearly with usage.

| Monthly estimate | Low | Mid | High |
|---|---:|---:|---:|
| Variable compute (sessions × $0.06) | ~$1.9k | ~$4.2k | ~$9.3k |
| Warm floor (2 active heavy + 1 light)* | ~$4.5k | ~$4.5k | ~$4.5k |
| **Total order of magnitude** | **~$6k** | **~$8–9k** | **~$13–14k** |

\* The floor partially absorbs variable load (an active worker processing jobs isn't
extra spend), so real totals land below the naive sum — treat the table as an upper
band. The floor is also the main cost lever: the same 2 warm workers as **pods**
cost ~$2.6k/mo instead of ~$4.5k — that's the §9 phase-2 move, worth ~$2k/mo once
traffic justifies operating it.

Guardrails: RunPod's default $80/hr account spend cap actually helps here (12 heavy
flex workers ≈ $48/hr); set billing alerts and keep the max-worker cap deliberate.

---

## 6. What changes in this repo

Small, additive changes — the route/service/runtime structure already anticipates this.

1. **`handler_heavy.py` / `handler_light.py`** (new, ~100 lines each): RunPod SDK
   handlers that call the existing service functions directly (`services/wardrobe.py`,
   `services/tryon.py`, `services/upscale.py`, `services/user_validation.py`),
   bypassing FastAPI/uvicorn inside the worker. Warmup reuses
   `warmup_resident_runtimes()` at worker start. `RESIDENT_RUNTIMES` selects the
   profile per image (`wardrobe,tryon,upscale` vs validation-only).
2. **Keep the FastAPI app** unchanged for local dev, the current pod, and as the
   fallback deployment mode. The serverless handler is a second entrypoint, not a fork.
3. **Job orchestration on Azure** (new, in the product backend, not this repo):
   job table, `/v1/jobs/{id}`, webhook receiver, RunPod API client, retry policy
   (1 retry on worker failure; idempotency via server-generated `job_id` — the
   request-isolation rules in `docs/gpu-runtime-and-concurrency.md` already require
   exactly this ownership tuple).
4. **Worker images**: bake venv + code + CUDA deps into the image; models + LoRAs
   stay on an **EU network volume** (54 GB Qwen pulls at ~350–700 MB/s ≈ 2–4 min).
   Continue the startup-optimization plan (13–15 → 3–4 min) — on serverless, cold
   start is billed and gates scale-up speed, so that work pays for itself directly.
5. **Queue semantics move outward**: the in-process bounded queue stays as a
   last-line guard (it's per-worker), but saturation handling (429/503) largely
   shifts to the RunPod endpoint queue + the Azure job layer (which can hold jobs
   far longer than 30 s without breaking clients, since the API is async now).

---

## 7. EU compliance posture

- Pin the serverless endpoints and the network volume to EU data centers
  (EU-RO-1 / EU-SE-1 / other EU-\*); RunPod documents GDPR compliance for EU regions
  and provides a security/compliance filter at deploy time. Sign RunPod's DPA.
- Keep all persistent user images in **Azure Blob EU**; workers receive short-lived
  SAS URLs, write outputs back to Blob, and keep only ephemeral job-local files —
  the TTL cleanup + redaction rules already specified in
  `docs/gpu-runtime-and-concurrency.md` (no signed URLs or filesystem paths in logs,
  artifacts bound to `(tenant_id, user_id, job_id)`).
- RunPod is a US company operating EU DCs: GDPR-compatible via DPA + SCCs for most
  cases. If legal later requires an EU-incorporated processor, the async job
  boundary makes swapping the GPU vendor a contained change.
- Add a data-processing record: what leaves Azure (input image, prompt/garment
  metadata), where it's processed (named EU DC), retention (job-local, deleted on
  completion; webhook payload contains keys, not images).

## 8. Failure modes and operations

- **Wedged GPU**: watchdog flags degraded → worker stops accepting (serverless
  health), RunPod replaces it. Same mechanism as today's `/ready`, fleet-wide.
- **Worker dies mid-job**: webhook never fires → Azure job reaper times out the job
  at 120 s, retries once on a different worker, then fails with a user-visible error.
- **Queue spike beyond max workers**: jobs wait; backend exposes queue-depth-derived
  ETA on `/v1/jobs/{id}` so the app can show progress instead of erroring.
- **RunPod region outage**: jobs fail fast → feature-level kill switch in the app
  ("try-on busy, retry shortly"). A second EU region endpoint is a config change.
- **Observability**: emit per-job timings (queue wait, GPU exec, end-to-end) from the
  webhook payload into your metrics stack; alert on P95 queue wait > 15 s (scale
  ceiling too low), cold-start rate > a few/hr (floor too low), cost/day anomaly.
- **Load test before launch**: replay a synthetic peak-hour (mid scenario: ~450
  sessions/hr) against a staging endpoint; verify autoscale reaches steady state
  with P95 tryon ≤ 25 s end-to-end.

## 9. Phased roadmap

| Phase | When | What | Exit criterion |
|---|---|---|---|
| 0 | now → launch | Async job API on Azure; heavy+light serverless endpoints in EU DC; 2-active floor; load test | P95 tryon ≤ 25 s at mid-scenario load |
| 1 | launch + 2 wk | Tune floor/max from real traffic; finish startup-optimization plan (3–4 min cold start) | cold-start no longer dominates scale-up |
| 2 | launch + 4–6 wk | Move steady base load to 2–3 reserved/secure **pods** behind the same job API; serverless keeps burst | floor cost ≈ halved, zero product change |
| 3 | growth | Revisit: same-LoRA micro-batching on Qwen, 80 GB GPU option for heavy worker (post-split), multi-region EU | unit cost per session trending down |

## 10. Trade-offs made explicit

- **Serverless premium vs build time**: paying ~1.5–2× per warm GPU-hour vs pods, in
  exchange for not building LB/autoscaler/queue/recycling during MVP. Revisited in
  phase 2 where the floor moves to pods.
- **Two worker images vs one**: more CI surface, but cheaper validation, lower tail
  latency, and VRAM headroom. The single-image mode remains for dev.
- **Async API vs sync simplicity**: app needs job polling/push, but the system gains
  back-pressure, retries, vendor swapability, and honest UX during spikes.
- **96 GB dependency**: simplest fit for the resident set; the validation split is
  the hedge that opens 80 GB GPUs if EU availability of the 96 GB class tightens.

## What I'd revisit as it grows

Real peak concurrency vs the 20%-peak-hour assumption (week-1 data); whether wardrobe
captioning should move fully to the orchestrator; AOTInductor-style precompiled
artifacts to push cold start toward the ~2 min floor; reserved-capacity pricing talks
with RunPod once spend stabilizes above ~$5k/mo.

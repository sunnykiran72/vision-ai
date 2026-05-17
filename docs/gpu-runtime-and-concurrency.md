# GPU Runtime and Concurrency

## Current runtime model

The current direction is a resident in-process GPU runner.

Meaning:

- the runtime module is loaded once
- model/runtime cache is reused
- requests do not spawn a fresh subprocess for inference

This is the preferred baseline for large GPU deployments such as H100-class nodes.

## Request isolation

Every request must remain isolated from every other request.

Isolation rules:

- generate a unique job id
- create a unique job directory per request
- create separate input files per request
- create separate output files per request
- create separate log files per request

This prevents:

- file collisions
- output overwrites
- one user receiving another user’s output artifact

File isolation alone is not sufficient for production.

Additional ownership rules are required:

- every request must resolve authenticated `user_id`
- every request must resolve `tenant_id` when multi-tenant
- every job id must be server-generated
- every stored artifact must be bound to `(tenant_id, user_id, job_id)`
- every artifact retrieval must enforce authorization against that ownership tuple
- user-supplied filenames must never determine storage keys

## API concurrency vs GPU execution concurrency

These are different things.

The API should be able to accept `n` incoming requests.

That does not mean the GPU runtime should execute all `n` requests in parallel without control.

Correct design:

- API layer accepts requests
- service layer creates isolated request context
- runtime layer schedules actual model execution

## Why shared runner execution must be controlled

A resident shared runner may contain:

- shared model state
- shared runtime cache
- shared device context

Because of that, GPU execution must be coordinated.

Do not assume:

- model runtime is thread-safe
- unrestricted parallel execution is free
- large VRAM alone makes unbounded concurrency safe

## Concrete production default

Until a specific runtime is proven thread-safe under load, use this default:

- one execution lock per resident runner
- one bounded queue per runner
- finite queue size
- finite queue wait timeout
- finite execution timeout

When saturated:

- reject immediately with `429` or `503` if queue is full
- return `503` or `504` if a request cannot start or finish within configured time budgets

Do not use generic `500` for overload or queue saturation.

## Why unbounded parallel inference is not the target

The goal is not to let every incoming request hit the same runtime in uncontrolled parallel fashion.

Reasons:

- shared runtime corruption risk
- unstable latency
- unpredictable VRAM and compute contention
- harder debugging

The goal is:

- correctness first
- request isolation
- clear execution control
- future scalability without changing public API contracts

## Recommended future evolution

The code should be prepared for these internal runtime strategies:

### 1. Execution coordinator

Introduce a coordinator abstraction that the service calls instead of directly deciding concurrency.

The coordinator should own:

- runner selection
- queuing
- concurrency policy
- future worker allocation

### 2. Bounded worker pool

When concurrency tuning is needed, prefer:

- a bounded worker pool
- multiple isolated resident runners
- controlled parallelism

Do not expose the worker count in public endpoint contracts.

The worker pool count should remain internal and tunable by deployment, not by API input.

### 3. Startup warmup

Use startup warmup to preload required runtimes so concurrency behavior starts from a warm system state.

## Public API boundary

Concurrency policy is an internal runtime concern.

It should not be encoded into public request models.

Public endpoint contracts should not expose:

- number of workers
- queue mode
- offload mode
- runner mode
- runtime parallelism knobs

## Process and GPU mapping

Concurrency design must be aligned with process topology.

Required guidance:

- avoid naïve multi-worker Uvicorn deployment for resident GPU runners
- each worker process may load its own model copy
- single-GPU default should be one API process per GPU-backed runtime set
- multi-GPU scaling should be explicit and documented

This prevents accidental VRAM duplication and unstable startup behavior.

## Logging and redaction

Per-request logging is useful, but logs must not leak sensitive values.

Redaction requirements:

- redact raw signed URLs and SAS tokens
- redact internal filesystem paths from user-visible responses
- avoid logging raw prompts where not required
- avoid logging secrets or full auth context

Safe correlation fields:

- `request_id`
- `job_id`
- `user_id`
- `tenant_id`

Unsafe fields to emit directly:

- raw image contents
- secrets
- signed URLs
- storage connection details

## Retention and cleanup

Request-local directories require explicit cleanup policy.

Required behavior:

- define TTL cleanup for temporary request directories
- define retention windows for successful and failed jobs
- define deletion behavior for uploaded artifacts when applicable
- protect the service from disk-pressure growth over time

## Relevance to the other core APIs

This runtime model is not specific to `/v1/upscale`.

The same request-isolation and execution-coordination pattern should later apply to:

- `/v1/tryon`
- `/v1/wardrobe`
- `/v1/user_validation`

## LLM review checklist

- Is the distinction between API concurrency and runtime concurrency clear?
- Is request isolation sufficient to prevent cross-user output mixing?
- Are runtime concurrency concerns kept out of public contracts?
- Is a coordinator or worker-pool evolution path clearly defined?
- Is this design appropriate for H100-style deployment?
- Are process topology and redaction rules explicit enough?
- Are cleanup and retention policies defined?
- Is this reusable across the other three core APIs?

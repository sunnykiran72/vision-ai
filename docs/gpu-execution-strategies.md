# GPU Execution Strategies

## Purpose

This document records the current execution choice for GPU-backed APIs in this service and the future upgrade options.

It exists to answer three questions clearly:

- what are we doing now
- why are we doing it now
- when should we switch to a stronger runtime boundary

## Current decision

The current service uses a resident in-process runner for GPU-backed inference.

That means:

- the FastAPI process loads the model runtime
- the model stays warm in memory
- requests call the runner through service code
- execution concurrency is controlled internally by a coordinator and runner lock

This is the current baseline for:

- `/v1/upscale`

The same pattern is intended for:

- `/v1/tryon`
- `/v1/wardrobe`
- `/v1/user_validation`

## Why we are keeping the in-process runner now

The in-process runner is the right current choice because:

- it is the simplest high-performance baseline
- it keeps integration overhead low
- it reuses loaded model state efficiently
- it avoids per-request process startup cost
- it is enough while the main priority is building the core APIs cleanly

For H100-class deployment, this gives a good first implementation path as long as the runtime is stable.

## What the in-process runner is good at

- fast warm-path inference
- low orchestration complexity
- simple startup warmup
- simple model cache reuse
- small code surface area

## What the in-process runner is bad at

- hard cancellation of a stuck GPU job
- strong fault isolation between model execution and API lifecycle
- safe recovery when one inference call hangs inside the runtime
- strict per-job kill-and-restart behavior

Important limitation:

If one GPU job hangs inside the in-process runner, timing out the HTTP request does not guarantee that the GPU work has actually stopped.

## What "runtime boundary" means

Runtime boundary means: where the actual model execution lives.

There are two main options:

### 1. In-process runner

The model runtime lives inside the API process.

Flow:

- API receives request
- service prepares isolated job workspace
- service calls runner directly
- runner executes inference in the API process

### 2. Worker process boundary

The model runtime lives in a separate worker process.

Flow:

- API receives request
- service prepares isolated job workspace
- API sends execution request to worker
- worker runs inference
- worker returns result

If the worker hangs, the worker can be killed and restarted without killing the API process.

## Strategy comparison

### In-process runner

Pros:

- lowest integration complexity
- lowest call overhead
- fast warm reuse
- easy to reason about while building core APIs

Cons:

- harder to kill stuck jobs safely
- API process and model process are the same thing
- one bad inference can stall that runner path

Best for:

- first clean implementation
- low operational complexity
- trusted runtime behavior

### Worker process boundary

Pros:

- stronger fault isolation
- killable stuck jobs
- safer timeout enforcement
- easier multi-worker scaling
- better production recovery behavior

Cons:

- more orchestration complexity
- inter-process communication required
- slightly more overhead
- more components to supervise

Best for:

- strict timeout enforcement
- safer recovery from stuck GPU jobs
- more production-hardened execution

## Best-practice recommendation by stage

### Current stage

Use:

- resident in-process runner
- startup warmup
- bounded queue
- request-local job isolation
- mandatory cleanup

This is the correct stage for the current service while the API contracts and internal boundaries are still being built out.

### Later production-hardening stage

Move to:

- worker process boundary for model execution
- supervisor-managed worker restart
- bounded worker pool if concurrency tuning is needed

This should happen when any of these become true:

- stuck GPU jobs are observed
- stricter per-job timeout enforcement is required
- one runner stall becomes operationally unacceptable
- multiple parallel GPU workers are needed safely

## Recommended future migration path

Do not rewrite route and service layers when moving to worker processes.

Keep these layers stable:

- `routes/`
- `models/`
- `services/`
- `storage` client
- media helpers
- auth and ownership flow

Only change the execution backend behind the runtime boundary:

- current: in-process resident runner
- future: supervised worker process runner

That keeps the public API contract stable while strengthening the execution model.

## What not to do

Do not keep adding timeout logic inside the in-process runner and assume that solves stuck GPU jobs.

Why:

- request timeout is not the same as inference cancellation
- a stuck runtime can still hold the execution path
- forced interruption inside a shared in-process runtime is not a safe general strategy

If real preemption is required, move the model execution into a separate worker process boundary.

## Practical guidance for this repo

For now:

- keep the in-process runner
- keep the coordinator
- keep startup warmup
- keep strict job isolation and cleanup

Later:

- introduce a worker process boundary only for the model execution layer
- keep all public API and orchestration contracts unchanged

## LLM review checklist

- Does this document clearly distinguish current and future execution strategies?
- Is the reason for choosing in-process now explicit?
- Are the limitations of in-process execution stated honestly?
- Is the migration path to worker processes clear and minimally disruptive?
- Is the guidance reusable for the other GPU-backed APIs?

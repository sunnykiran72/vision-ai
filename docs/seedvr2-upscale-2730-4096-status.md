# SeedVR2 Upscale — 2730 (try-on) & 4096 (standalone 4K) — Status & Decisions

> Self-contained record of everything done/measured for SeedVR2 in the wardrobe/try-on/upscale
> stack. Hardware: RunPod **RTX PRO 6000 Blackwell, 96 GB** (Workstation pod `ra4nfsl1lrajm5` and
> Server pod `dt3jjdcekx1lvl`, sharing one `/workspace` network volume). Model:
> `seedvr2_ema_3b_fp8_e4m3fn.safetensors` (3B FP8). Client: `app/clients/seedvr2.py`.

---

## A. Pod access & sign-in

> ⚠️ RunPod IPs/ports are **ephemeral** — they change on every pod stop/restart. Re-copy from the
> pod's **Connect** tab if SSH fails. Both pods mount the **same `/workspace` network volume**, so
> the repo, models, LoRAs, `.env`, and compile caches are shared between them.

### Pods
| Pod (name) | ID | Edition | Direct-TCP SSH |
|---|---|---|---|
| **WK** `sporting_violet_lemming` (current prod-candidate, fast CPU) | `ra4nfsl1lrajm5` | Workstation | `ssh root@157.157.221.29 -p 50769 -i ~/.ssh/runpod_qge_ed25519` |
| **Server** `select_green_guineafowl` (~3× slower CPU) | `dt3jjdcekx1lvl` | Server | `ssh root@157.157.221.177 -p 16470 -i ~/.ssh/runpod_qge_ed25519` |

- **SSH key: `~/.ssh/runpod_qge_ed25519`** (authenticates on both pods). The Connect tab shows
  `id_ed25519`, but `runpod_qge_ed25519` is the one that works here.
- RunPod proxy SSH (no SCP/SFTP): `ssh ra4nfsl1lrajm5-<hash>@ssh.runpod.io -i ~/.ssh/id_ed25519`.
- File transfer to the shared volume (no scp on the proxy): `cat localfile | ssh … 'cat > /workspace/…'`.

### Web tools (RunPod HTTP proxy, port 8000, `/tools/*` is **unauthenticated**)
- Try-on lab: `https://ra4nfsl1lrajm5-8000.proxy.runpod.net/tools/tryon-lab`
- Upscale lab: `https://ra4nfsl1lrajm5-8000.proxy.runpod.net/tools/upscale-lab`
- Wardrobe/extract lab: `…/tools/qwen-lab` · `POST …/tools/diffusers/extract`
- Health: `…/health` · Jupyter: port **8888** (token shown in the pod's Connect tab / jupyter process).
- Production `/v1/*` endpoints (`/v1/wardrobe`, `/v1/tryon`, `/v1/upscale`) **require a Bearer token**
  (validated against the Glamify backend); the `/tools/*` labs do not.

### Key paths (all on the shared `/workspace`)
| What | Path |
|---|---|
| App repo | `/workspace/glamify-image-ai` |
| venv (no `pip` — use `uv pip install --python …`) | `/workspace/.venvs/glamify-image-ai` |
| Models | `/workspace/models/{qwen-image-edit-2511, minicpm-v-4_5-awq, seedvr2}` |
| LoRAs | `/workspace/loras/wardrobe/`, `/workspace/loras/tryon/` |
| SeedVR2 CLI | `/workspace/seedvr2_eval/ComfyUI-SeedVR2_VideoUpscaler/inference_cli.py` |
| Env file | `/workspace/glamify-image-ai/.env` (secrets — not in git) |
| Start script | `scripts/runpod_start.sh` (sets nvrtc `LD_LIBRARY_PATH`, loads `.env`) |
| Launch supervisor | **`/tmp/glamify-supervisor.sh`** ⚠️ pod-local & ephemeral (NOT on the volume; recreate after a pod restart) |
| Server log | `/workspace/glamify-image-ai/server_wk.log` |
| Compile cache | `/workspace/.torchinductor_cache`, `/workspace/.triton_cache` |

### Start / stop / restart the API
```bash
# stop (kill supervisor + uvicorn + vLLM, by PID):
ps -eo pid,cmd | grep -E "glamify-supervisor|uvicorn app.main|EngineCore"
kill -9 <supervisor_pid> <uvicorn_pid>; pkill -9 -f EngineCore

# start (detached; supervisor auto-restarts the API on crash):
cd /workspace/glamify-image-ai && rm -f server_wk.log \
  && nohup setsid bash /tmp/glamify-supervisor.sh > server_wk.log 2>&1 </dev/null & disown

# the supervisor's env block is the source of truth for QWEN_*/UPSCALE_* flags.
```

---

## 0. TL;DR (the decisions)

- **SeedVR2 runs EAGER (no `torch.compile`).** `UPSCALE_COMPILE=0`. Compile gave only ~20–35%
  steady-state but caused **severe instability** (per-shape recompiles that monopolize the single
  GPU for minutes → request timeouts/OOM). Not worth it.
- **No startup pre-warm.** `UPSCALE_WARMUP=0`. The background pre-warm thread ran SeedVR2
  concurrently with Qwen and collided. Upscale now compiles/auto-tunes lazily on first use.
- **Qwen and SeedVR2 share ONE GPU execution slot** (`get_upscale_execution_coordinator` now
  delegates to `get_system_execution_coordinator`). They are both heavy diffusion models on one
  GPU — running them at the same time stacks their memory peaks and OOMs. Serializing fixes it.
- **Try-on upscales DIRECTLY to 2048** (the final size). We previously upscaled to **2730 then
  downscaled to 2048** — wasting ~2.4 s generating pixels we threw away. Direct 2048 ≈ **2.6 s**.
- **Standalone 4K (4096) is the only tight path** — it OOMs when co-resident with the full stack
  and is intrinsically ~11–12 s. Treat it separately (lower target, its own worker, or accept it).

---

## 1. The two scenarios

| Scenario | Where | Input | Upscale target | Output delivered |
|---|---|---|---|---|
| **Try-on inline upscale** | `app/services/tryon.py` (and `tryon_lab`) | Qwen try-on output (= user image dims, e.g. 1056×1584) | **2048** (was 2730) | 2048 long edge JPEG |
| **Standalone 4K** | `/v1/upscale` (`app/services/upscale.py`) | user image | **4096** (`metric=4k`) | 4096 long edge JPEG |

Both call the same `SeedVR2Client.run(target_long_edge=...)`. Cost is driven by the **output**
resolution (SeedVR2 generates at the target res), not the input.

---

## 2. Measured latency (RTX PRO 6000, SeedVR2 3B FP8, EAGER, steady-state)

Output long edge → wall time (input ~1056×1584; cost scales with output pixels):

| Output long edge | Output dims | Eager | Compiled (ref) |
|---|---|---|---|
| 1536 | 1024×1536 | **1.7 s** | ~1.6 s |
| **2048** | 1364×2046 | **2.6 s** ✅ | ~2.1 s |
| 2560 | 1706×2560 | 4.4 s | ~3.3 s |
| 2730 | 1820×2730 | 5.0 s | ~4.5 s |
| 3072 | 2048×3072 | 6.1 s | ~4.5 s (lab floor 3.6 s) |
| 4096 | 2730×4096 | 11–12.6 s | ~10 s |

Notes:
- **Latency is ~linear in output pixels** above ~2048; the **VAE decode is memory-bandwidth-bound**
  and is ~70 % of the time at 3072 — no lossless lever moves it.
- **First call at a NEW output shape** pays a one-time **cuDNN benchmark autotune (~30–40 s)**
  (separate from torch.compile), then is steady. `cudnn.benchmark=True` is set in the client.

---

## 3. Why we dropped `torch.compile` for SeedVR2

torch.compile *looked* attractive (lab floor ~3.6 s @3072) but on the live, co-resident stack it
broke things:

1. **Compile is ~20–35 %, not 2×.** Code comment + measurement: 3072 eager 6.1 s → compiled ~4.5 s.
   The VAE-decode wall caps the win.
2. **Compile cost is huge and per-shape.** ~7–12 min of CPU codegen **per distinct output shape**
   (even on the fast Workstation CPU; ~20 min on the Server CPU). Inference inputs vary, so new
   shapes keep triggering it.
3. **A compile monopolizes the single GPU.** While a shape compiles (minutes), it holds the GPU →
   concurrent/queued requests **time out** (`QueueTimeoutError`) and curl times out.
4. **Persistent inductor cache helps but doesn't fix it.** `TORCHINDUCTOR_CACHE_DIR=/workspace`
   cut a cold compile from ~467 s → ~181 s on a fresh process, but **Dynamo re-traces the graph
   and cuDNN re-benchmarks every process** — so it never becomes "instant", and the per-shape churn
   remains.
5. **Startup pre-warm made it worse.** Pre-warming 2730+4096 at boot either blocked health for
   ~20 min (if it held the slot) or ran **concurrently** with Qwen (if backgrounded) → OOM/500s.

**Decision: eager.** ~1.5–2 s slower at steady state, but **stable, predictable, no GPU monopoly,
no prewarm needed.** For try-on at 2048 that is **2.6 s eager** — already under target.

---

## 4. The 4096 (4K) memory problem

Co-resident VRAM at rest (96 GB GPU):

```
API process (Qwen fp8 ~42 + SeedVR2 ~10 + compile/cuDNN workspace ~14):  ~66 GB
MiniCPM-V vLLM (gpu_memory_utilization=0.1):                              ~8.6 GB
────────────────────────────────────────────────────────────────────────────
Total at rest:                                                           ~75 GB  → ~20 GB free
```

- **4K needs ~10.7 GB contiguous activation** for one 2730×4096 frame. With ~10.3 GB free at the
  moment it tried, it **OOM'd by ~0.4 GB** — a **fragmentation + peak** problem, not raw capacity.
- **2730/2048 fit comfortably** (smaller activation) — that's why the try-on path is fine.
- Going eager (no compile) frees the ~14 GB compile workspace, which **helps 4K fit** when run
  alone (serialized via the shared coordinator).

**4K options:** (a) keep it but rely on serialization + `expandable_segments`; (b) cap standalone
upscale at 3072 (~6 s, comfortable); (c) run 4K on a **dedicated upscale worker** with the full
96 GB to itself. The try-on path does **not** need 4K.

---

## 5. Fixes applied (code + config)

| Change | File | Why |
|---|---|---|
| Try-on upscale target **2730 → 2048** (direct, no downscale) | `app/constants/tryon.py` | 5.0 s → **2.6 s**, same final image |
| **Shared GPU coordinator** for upscale | `app/runtime/upscale_runtime.py` | Qwen + SeedVR2 never run concurrently → no stacked peaks/OOM |
| **Non-blocking** startup pre-warm (then disabled) | `app/clients/seedvr2.py` | prewarm never blocks health; currently off via env |
| Persistent inductor cache (kept, now moot under eager) | `app/clients/seedvr2.py` | `TORCHINDUCTOR_CACHE_DIR=/workspace` |
| Final output **JPEG@95**, intermediate **PNG** | `tryon.py` / `upscale.py` | consistent + small delivery, no intermediate recompression |

**Live runtime config (supervisor env on WK):**
```
RESIDENT_RUNTIMES="wardrobe,tryon,upscale"
QWEN_FP8="true"  QWEN_COMPILE="true"          # Qwen unchanged — keeps ~6 s
UPSCALE_COMPILE="0"                           # SeedVR2 EAGER
UPSCALE_WARMUP="0"                            # no background prewarm
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

---

## 6. Current end-to-end numbers (WK, this config)

| Path | Steady-state |
|---|---|
| Wardrobe extract @12 | **~6.1 s** |
| Try-on Qwen only (1056×1584) | **~13.8 s** |
| Try-on **upscale to 2048** | **~2.6 s** |
| Try-on total (Qwen + 2048 upscale + overhead) | **~16.4 s** |
| Standalone 4K (4096) | ~11–12 s (tight on memory co-resident) |

One-time costs: first try-on **per new image size** pays a Qwen recompile (~60 s once); first
upscale **per new output shape** pays a cuDNN autotune (~30 s once).

---

## 7. Open items / levers

- **Try-on total (~16 s) is dominated by Qwen, not upscale.** Upscale is already ~2.6 s. To lower
  the total: fewer Qwen steps (8 ≈ ~9 s, quality risk), cap person input to 832×1248 (~8–10 s
  Qwen), or a distilled few-step try-on model.
- **Kill the per-shape one-time costs**: normalize the try-on input to a fixed size so Qwen compiles
  once and SeedVR2 autotunes once.
- **4K**: decide between 3072-cap, dedicated worker, or accept ~12 s.

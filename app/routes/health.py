from fastapi import APIRouter, Response

from app.config import get_enabled_resident_runtimes, get_settings
from app.constants import http_status
from app.models.health import HealthResponse
from app.runtime.system_coordinator import get_system_execution_coordinator
from app.runtime.tryon_runtime import get_tryon_runtime_status
from app.runtime.upscale_runtime import get_upscale_runtime_status
from app.runtime.wardrobe_runtime import get_wardrobe_runtime_status
from app.services.health import build_health_response

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    return build_health_response(get_settings())


@router.get("/ready")
async def readiness(response: Response) -> dict:
    """Readiness probe for the load balancer / autoscaler.

    Returns 200 only when every runtime THIS pod is configured to host (``RESIDENT_RUNTIMES``) is
    warm — including the SeedVR2 prewarm (compile), which finishes asynchronously after FastAPI
    startup completes. Point the LB/autoscaler at /ready so a pod only receives traffic once fully
    warm → the first request it ever serves is fast (no cold compile). A pod that doesn't host a
    given runtime is never gated on it (e.g. a wardrobe+tryon pod with no upscale stays routable).
    /health stays liveness-only (process up). Read-only status checks; never triggers load.
    """
    settings = get_settings()
    enabled = get_enabled_resident_runtimes(settings)
    components: dict[str, bool] = {}
    checks: list[bool] = []

    if "wardrobe" in enabled:
        wardrobe_loaded = get_wardrobe_runtime_status(settings).runner.loaded
        components["wardrobe_loaded"] = wardrobe_loaded
        checks.append(wardrobe_loaded)

    if "tryon" in enabled:
        tryon_loaded = get_tryon_runtime_status(settings).runner.loaded
        components["tryon_loaded"] = tryon_loaded
        checks.append(tryon_loaded)

    if "upscale" in enabled:
        upscale = get_upscale_runtime_status(settings).runner
        components["upscale_loaded"] = upscale.loaded
        components["upscale_prewarmed"] = upscale.prewarmed
        checks.extend([upscale.loaded, upscale.prewarmed])

    # A wedged GPU op (hang watchdog) marks the shared coordinator degraded -> drop from the pool.
    degraded = get_system_execution_coordinator(settings).snapshot().degraded
    components["gpu_degraded"] = degraded

    ready = bool(all(checks) and not degraded)
    response.status_code = http_status.OK if ready else http_status.SERVICE_UNAVAILABLE
    return {
        "ready": ready,
        "components": components,
    }

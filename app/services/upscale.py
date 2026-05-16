from app.models.upscale import UpscaleResponse


def build_upscale_placeholder() -> UpscaleResponse:
    return UpscaleResponse(
        status="not_implemented",
        message="Upscale flow scaffold is ready. Implementation will be added next.",
        feature="upscale",
    )

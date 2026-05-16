from app.models.tryon import TryonResponse


def build_tryon_placeholder() -> TryonResponse:
    return TryonResponse(
        status="not_implemented",
        message="Try-on flow scaffold is ready. Implementation will be added next.",
        feature="tryon",
    )

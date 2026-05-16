from app.models.wardrobe import WardrobeAnalyzeResponse


def build_wardrobe_placeholder() -> WardrobeAnalyzeResponse:
    return WardrobeAnalyzeResponse(
        status="not_implemented",
        message="Wardrobe flow scaffold is ready. Implementation will be added next.",
        feature="wardrobe",
    )

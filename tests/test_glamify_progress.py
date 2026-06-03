from app.clients.glamify_progress import _wardrobe_progress_endpoint_url


def test_wardrobe_progress_endpoint_accepts_base_url() -> None:
    assert (
        _wardrobe_progress_endpoint_url("https://api.example.com")
        == "https://api.example.com/wardrobe/progress"
    )


def test_wardrobe_progress_endpoint_accepts_full_endpoint_url() -> None:
    assert (
        _wardrobe_progress_endpoint_url("https://api.example.com/wardrobe/progress")
        == "https://api.example.com/wardrobe/progress"
    )

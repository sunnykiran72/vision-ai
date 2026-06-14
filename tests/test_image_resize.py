from __future__ import annotations

from PIL import Image

from app.utils import image_resize


def _striped_bw(width: int, height: int) -> Image.Image:
    """Alternating black/white 1px vertical stripes — averages to a midtone when shrunk."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    px = img.load()
    for x in range(width):
        if x % 2 == 0:
            for y in range(height):
                px[x, y] = (255, 255, 255)
    return img


def test_resize_image_returns_rgb_and_exact_size() -> None:
    src = Image.new("L", (200, 300), 128)  # non-RGB input
    out = image_resize.resize_image(src, (100, 150))
    assert out.mode == "RGB"
    assert out.size == (100, 150)


def test_resize_image_noop_when_size_matches_but_normalizes_mode() -> None:
    src = Image.new("RGBA", (64, 64), (10, 20, 30, 255))
    out = image_resize.resize_image(src, (64, 64))
    assert out.mode == "RGB"
    assert out.size == (64, 64)


def test_gamma_correct_downscale_is_brighter_than_naive_srgb() -> None:
    """Black+white averaged in linear light (~188) is much brighter than naive
    sRGB averaging (~127). This proves the resize happens in linear space."""
    src = _striped_bw(120, 120)
    gamma = image_resize.resize_image(src, (12, 12), gamma_correct=True, sharpen=False)
    naive = src.resize((12, 12), image_resize.LANCZOS)

    gamma_mean = sum(gamma.convert("L").getdata()) / (12 * 12)
    naive_mean = sum(naive.convert("L").getdata()) / (12 * 12)

    assert gamma_mean > 160, f"expected linear-light midtone ~188, got {gamma_mean:.1f}"
    assert naive_mean < 145, f"expected naive sRGB midtone ~127, got {naive_mean:.1f}"
    assert gamma_mean - naive_mean > 30


def test_downscale_photo_sharpens_on_reduction() -> None:
    src = _striped_bw(120, 120)
    sharpened = image_resize.downscale_photo(src, (40, 40))
    plain = image_resize.resize_image(src, (40, 40), sharpen=False)
    # Sharpening widens the tonal range (more contrast) vs the unsharpened version.
    s_lo, s_hi = min(sharpened.convert("L").getdata()), max(sharpened.convert("L").getdata())
    p_lo, p_hi = min(plain.convert("L").getdata()), max(plain.convert("L").getdata())
    assert (s_hi - s_lo) >= (p_hi - p_lo)
    assert sharpened.size == (40, 40)


def test_sharpen_skipped_on_upscale() -> None:
    """Upscaling with sharpen=True must NOT sharpen (would amplify artifacts)."""
    src = Image.new("RGB", (50, 50), (120, 120, 120))
    up = image_resize.resize_image(src, (100, 100), sharpen=True)
    # A flat image stays flat — sharpening a flat upscale would not change it anyway,
    # so assert behaviour via a gradient instead.
    grad = Image.new("L", (50, 50))
    grad.putdata([((x % 50) * 5) % 256 for x in range(50 * 50)])
    grad = grad.convert("RGB")
    up_sharp = image_resize.resize_image(grad, (100, 100), sharpen=True)
    up_plain = image_resize.resize_image(grad, (100, 100), sharpen=False)
    assert list(up_sharp.getdata()) == list(up_plain.getdata())
    assert up.size == (100, 100)


def test_resize_to_long_edge_only_shrinks_by_default() -> None:
    src = Image.new("RGB", (800, 1200), (10, 10, 10))
    # within cap -> untouched
    assert image_resize.resize_to_long_edge(src, 2000).size == (800, 1200)
    # above cap -> longest side becomes max_edge, aspect preserved
    out = image_resize.resize_to_long_edge(src, 600)
    assert max(out.size) == 600
    assert out.size == (400, 600)


def test_long_edge_and_height_are_gamma_only_no_sharpen_by_default() -> None:
    """Non-upload helpers must change ONLY gamma vs old behaviour: no sharpening.
    Their default output must equal a plain gamma-correct resize (sharpen=False)."""
    src = _striped_bw(120, 160)
    le = image_resize.resize_to_long_edge(src, 60)  # -> 45x60
    expect_le = image_resize.resize_image(src, (45, 60), gamma_correct=True, sharpen=False)
    assert list(le.getdata()) == list(expect_le.getdata())

    rh = image_resize.resize_to_height(src, 80)  # -> 60x80
    expect_rh = image_resize.resize_image(src, (60, 80), gamma_correct=True, sharpen=False)
    assert list(rh.getdata()) == list(expect_rh.getdata())


def test_resize_to_height_preserves_aspect() -> None:
    src = Image.new("RGB", (900, 1200), (10, 10, 10))
    out = image_resize.resize_to_height(src, 600)
    assert out.size == (450, 600)
    # exact-height input returned as-is (RGB)
    assert image_resize.resize_to_height(out, 600).size == (450, 600)

"""Shared, detail-preserving image resizer used across the app.

Every place that downscales a user/garment photo should go through here so the
behaviour is identical and testable in one spot, instead of each call site
hand-rolling ``image.resize(..., LANCZOS)``.

What makes this better than a plain Lanczos resize (validated empirically across
10 garment photos — see ``review_outputs/downscale_ab``):

1. **Gamma-correct (linear-light) resampling.** Pillow's ``resize`` averages
   sRGB-encoded values, which darkens and muddies fine high-frequency detail
   (hair, fabric weave, edges). We decode sRGB -> linear, resample, then
   re-encode. This is a correctness fix, biggest at large downscale ratios.
2. **A light unsharp mask after a downscale** to restore the micro-contrast that
   any reduction softens. No detail is invented — it only recovers what the
   reduction blurred. Applied on downscales only (never on upscales).

numpy is imported lazily (the same pattern the blur check already uses). If it
is unavailable the resizer transparently falls back to a plain Lanczos resize so
it can never break on import in a numpy-less environment.
"""

from __future__ import annotations

from PIL import Image, ImageFilter

LANCZOS = Image.Resampling.LANCZOS

# Unsharp-mask strength applied after a downscale. Tuned on real garment photos:
# strong enough to restore crispness, mild enough to avoid halos. Lower PERCENT
# (~30-40) for a softer look, higher (~60) for more bite.
SHARPEN_RADIUS = 1.0
SHARPEN_PERCENT = 50
SHARPEN_THRESHOLD = 0


_SRGB_TO_LINEAR_LUT = None  # 256-entry float32 LUT, built once on first use


def _srgb_to_linear_lut(np):
    """256-entry sRGB->linear table. The input is 8-bit, so only 256 values exist —
    a table lookup replaces a per-pixel ``**2.4`` over millions of pixels (the cost)."""
    global _SRGB_TO_LINEAR_LUT
    if _SRGB_TO_LINEAR_LUT is None:
        c = np.arange(256, dtype=np.float32) / 255.0
        _SRGB_TO_LINEAR_LUT = np.where(
            c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4
        ).astype(np.float32)
    return _SRGB_TO_LINEAR_LUT


def _gamma_correct_resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Lanczos resample in linear light. Falls back to plain Lanczos without numpy.

    Performance (this is on the user_validation + tryon hot paths, so it matters):
      * sRGB->linear is a 256-entry LUT on the (large) source — no per-pixel ``**2.4``.
      * the resize itself uses cv2 (all 3 channels at once) when available, which is
        ~90ms vs ~210ms for PIL's 3 separate float passes, and faster than the old
        plain-Lanczos path (~130ms). Falls back to per-channel PIL float if cv2 is absent.
      * the ``**(1/2.4)`` re-encode runs only on the small resized output.
    """
    try:
        import numpy as np
    except Exception:  # numpy missing -> still produce a correct (if less ideal) result
        return image.resize(size, LANCZOS)

    lut = _srgb_to_linear_lut(np)
    linear = lut[np.asarray(image, dtype=np.uint8)]  # HxWx3 float32, no per-pixel pow

    try:
        import cv2

        # Lanczos for downscale and upscale alike (matches the validated "B" treatment).
        linear_resized = cv2.resize(
            linear, (size[0], size[1]), interpolation=cv2.INTER_LANCZOS4
        )
    except Exception:  # cv2 missing -> per-channel PIL float resize (slower but correct)
        channels = [
            np.asarray(Image.fromarray(linear[:, :, c]).resize(size, LANCZOS), dtype=np.float32)
            for c in range(3)
        ]
        linear_resized = np.stack(channels, axis=-1)

    linear_resized = np.clip(linear_resized, 0.0, 1.0)
    # linear -> sRGB (small array now)
    srgb_out = np.where(
        linear_resized <= 0.0031308,
        linear_resized * 12.92,
        1.055 * (linear_resized ** (1.0 / 2.4)) - 0.055,
    )
    return Image.fromarray(np.clip(srgb_out * 255.0 + 0.5, 0, 255).astype(np.uint8))


def _sharpen(image: Image.Image) -> Image.Image:
    """Light unsharp mask. cv2 path (gaussian + addWeighted) is ~4x faster than PIL's
    UnsharpMask and visually equivalent at these settings; PIL is the fallback."""
    try:
        import cv2
        import numpy as np

        arr = np.asarray(image)
        blurred = cv2.GaussianBlur(arr, (0, 0), SHARPEN_RADIUS)
        amount = SHARPEN_PERCENT / 100.0
        sharpened = cv2.addWeighted(arr, 1.0 + amount, blurred, -amount, 0)
        return Image.fromarray(sharpened)
    except Exception:
        return image.filter(
            ImageFilter.UnsharpMask(SHARPEN_RADIUS, SHARPEN_PERCENT, SHARPEN_THRESHOLD)
        )


def resize_image(
    image: Image.Image,
    size: tuple[int, int],
    *,
    gamma_correct: bool = True,
    sharpen: bool = False,
) -> Image.Image:
    """Resize ``image`` to ``size`` (width, height), always returning RGB.

    ``gamma_correct`` resamples in linear light (recommended for everything).
    ``sharpen`` applies a light unsharp mask, but only when the operation is a
    genuine downscale — sharpening an upscale just amplifies interpolation
    artifacts, so it is skipped there even if requested.
    """
    image = image.convert("RGB")
    size = (int(size[0]), int(size[1]))
    if image.size == size:
        return image

    is_downscale = (size[0] * size[1]) < (image.width * image.height)
    result = _gamma_correct_resize(image, size) if gamma_correct else image.resize(size, LANCZOS)
    if sharpen and is_downscale:
        result = _sharpen(result)
    return result


def downscale_photo(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Detail-preserving resize for user/garment PHOTOS: gamma-correct + light sharpen.

    This is the canonical "best" path established by the A/B tests. Use it whenever
    you are fitting a real photograph to a model input or display size.
    """
    return resize_image(image, size, gamma_correct=True, sharpen=True)


def resize_to_long_edge(
    image: Image.Image,
    max_edge: int,
    *,
    sharpen: bool = False,
    only_shrink: bool = True,
) -> Image.Image:
    """Scale ``image`` so its longest side is ``max_edge``, preserving aspect ratio.

    With ``only_shrink`` (default) images already within ``max_edge`` are returned
    untouched — matching the previous cap-the-longest-side behaviour. ``sharpen``
    defaults off so the only change versus the old plain-Lanczos path is gamma
    correctness; opt in explicitly for the detail-preserving photo treatment.
    """
    width, height = int(image.width), int(image.height)
    longest = max(width, height)
    if only_shrink and longest <= int(max_edge):
        return image.convert("RGB")
    scale = float(max_edge) / float(longest)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return resize_image(image, size, gamma_correct=True, sharpen=sharpen)


def resize_to_height(
    image: Image.Image, target_height: int, *, sharpen: bool = False
) -> Image.Image:
    """Scale ``image`` to ``target_height``, preserving aspect ratio.

    ``sharpen`` defaults off — the only change versus the old plain-Lanczos path is
    gamma correctness.
    """
    if int(image.height) == int(target_height):
        return image.convert("RGB")
    ratio = float(target_height) / float(image.height)
    size = (max(1, round(float(image.width) * ratio)), int(target_height))
    return resize_image(image, size, gamma_correct=True, sharpen=sharpen)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

TARGET_W = 1056
TARGET_H = 1584
PADDING = 24
SEPARATOR_H = 20


@dataclass(frozen=True)
class ProductReferenceInput:
    image: Image.Image
    type: str


@dataclass(frozen=True)
class TryonCollageResult:
    image: Image.Image
    mode: str
    product_count: int


def resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    if image.height == target_height:
        return image
    ratio = float(target_height) / float(image.height)
    target_width = max(1, int(round(float(image.width) * ratio)))
    resample: Any
    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.LANCZOS
    else:
        resample = 3
    return image.resize((target_width, target_height), resample)


def build_product_reference(products: list[ProductReferenceInput]) -> TryonCollageResult:
    if not products:
        raise ValueError("At least one product image is required.")

    if len(products) == 1:
        return TryonCollageResult(
            image=products[0].image.convert("RGB"),
            mode="single_product",
            product_count=1,
        )

    by_type = {item.type: item.image.convert("RGB") for item in products}
    if set(by_type.keys()) == {"top", "bottom"} and len(products) == 2:
        return TryonCollageResult(
            image=_build_top_bottom_collage(
                top_img=by_type["top"],
                bottom_img=by_type["bottom"],
                top_ratio=0.5,
            ),
            mode="top_bottom_vertical_collage",
            product_count=2,
        )

    target_height = max(int(item.image.height) for item in products)
    normalized_products = [
        resize_to_height(item.image.convert("RGB"), target_height)
        for item in products
    ]
    board_width = sum(int(image.width) for image in normalized_products)
    board = Image.new("RGB", (board_width, target_height), "white")

    offset_x = 0
    for image in normalized_products:
        board.paste(image, (offset_x, 0))
        offset_x += int(image.width)

    return TryonCollageResult(
        image=board,
        mode="multi_product_horizontal_collage",
        product_count=len(products),
    )


def compose_reference_with_user(
    product_reference: Image.Image,
    user_image: Image.Image,
) -> TryonCollageResult:
    target_height = max(int(product_reference.height), int(user_image.height))
    reference = resize_to_height(product_reference.convert("RGB"), target_height)
    user = resize_to_height(user_image.convert("RGB"), target_height)
    board = Image.new("RGB", (int(reference.width) + int(user.width), target_height), "white")
    board.paste(reference, (0, 0))
    board.paste(user, (int(reference.width), 0))
    return TryonCollageResult(
        image=board,
        mode="products_left_user_right",
        product_count=1,
    )


def _contain_no_upscale(src: Image.Image, max_w: int, max_h: int) -> Image.Image:
    src = src.convert("RGB")
    scale = min(float(max_w) / float(src.width), float(max_h) / float(src.height), 1.0)
    target_w = max(1, int(round(float(src.width) * scale)))
    target_h = max(1, int(round(float(src.height) * scale)))
    return resize_to_height(src.resize((target_w, target_h)), target_h)


def _contain_limited_upscale(
    src: Image.Image,
    max_w: int,
    max_h: int,
    *,
    max_upscale: float,
) -> Image.Image:
    src = src.convert("RGB")
    scale = min(
        float(max_w) / float(src.width),
        float(max_h) / float(src.height),
        float(max_upscale),
    )
    target_w = max(1, int(round(float(src.width) * scale)))
    target_h = max(1, int(round(float(src.height) * scale)))
    resample: Any
    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.LANCZOS
    else:
        resample = 3
    return src.resize((target_w, target_h), resample)


def _place_center(
    canvas: Image.Image,
    src: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    max_upscale: float,
    min_fill_for_no_upscale: float,
    valign: str = "center",
) -> None:
    src_area = float(src.width * src.height)
    box_area = float(max(1, w * h))
    src_fill = src_area / box_area
    if src_fill < float(min_fill_for_no_upscale):
        fitted = _contain_limited_upscale(src, w, h, max_upscale=float(max_upscale))
    else:
        fitted = _contain_no_upscale(src, w, h)
    ox = x + (w - fitted.width) // 2
    if valign == "top":
        oy = y
    elif valign == "bottom":
        oy = y + (h - fitted.height)
    else:
        oy = y + (h - fitted.height) // 2
    canvas.paste(fitted, (ox, oy))


def _build_top_bottom_collage(
    top_img: Image.Image,
    bottom_img: Image.Image,
    *,
    top_ratio: float,
) -> Image.Image:
    panel_total_h = TARGET_H - SEPARATOR_H
    ratio = max(0.35, min(0.65, float(top_ratio)))

    top_h = int(round(panel_total_h * ratio))
    top_h = max(int(panel_total_h * 0.35), min(int(panel_total_h * 0.65), top_h))
    bottom_h = panel_total_h - top_h
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), (255, 255, 255))

    top_box = (PADDING, PADDING, TARGET_W - PADDING * 2, top_h - PADDING * 2)
    bottom_y = top_h + SEPARATOR_H
    bottom_box = (PADDING, bottom_y + PADDING, TARGET_W - PADDING * 2, bottom_h - PADDING * 2)

    top_ar = float(top_img.height) / max(1.0, float(top_img.width))
    bottom_ar = float(bottom_img.height) / max(1.0, float(bottom_img.width))

    top_upscale = 1.35 if top_ar < 0.90 else 1.22 if top_ar < 1.20 else 1.15
    if bottom_ar < 0.85:
        bottom_upscale = 2.0
        bottom_min_fill = 0.62
    elif bottom_ar < 1.05:
        bottom_upscale = 1.75
        bottom_min_fill = 0.58
    else:
        bottom_upscale = 1.45
        bottom_min_fill = 0.50

    _place_center(
        canvas,
        top_img,
        *top_box,
        max_upscale=top_upscale,
        min_fill_for_no_upscale=0.45,
        valign="bottom",
    )
    _place_center(
        canvas,
        bottom_img,
        *bottom_box,
        max_upscale=bottom_upscale,
        min_fill_for_no_upscale=bottom_min_fill,
        valign="top",
    )
    return canvas

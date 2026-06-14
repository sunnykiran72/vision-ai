from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageChops

from app.utils.image_resize import resize_image
from app.utils.image_resize import resize_to_height as shared_resize_to_height

OUTER_PADDING = 12
INNER_PADDING = 0
COLLAGE_GAP = 12
MAX_ROW_ITEM_W = 640
MAX_ROW_ITEM_H = 720
MAX_VERTICAL_SECTION_W = 720
MAX_VERTICAL_SECTION_H = 720
TRIM_TOLERANCE = 12
TRIM_MARGIN = 10


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
    return shared_resize_to_height(image, target_height)


def normalize_product_type(product_type: str) -> str:
    normalized = str(product_type).strip().lower()
    if normalized == "outer":
        return "top"
    return normalized


def build_product_reference(products: list[ProductReferenceInput]) -> TryonCollageResult:
    if not products:
        raise ValueError("At least one product image is required.")

    if len(products) == 1:
        return TryonCollageResult(
            image=products[0].image.convert("RGB"),
            mode="single_product",
            product_count=1,
        )

    grouped_products = _group_products_by_type(products)
    tops = grouped_products.get("top", [])
    bottoms = grouped_products.get("bottom", [])
    dresses = grouped_products.get("dress", [])

    if len(tops) == 1 and len(bottoms) == 1 and not dresses:
        return TryonCollageResult(
            image=_build_top_bottom_collage(tops=tops, bottom_img=bottoms[0]),
            mode="top_bottom_vertical_collage",
            product_count=2,
        )

    if len(tops) == 2 and len(bottoms) == 1 and not dresses:
        return TryonCollageResult(
            image=_build_top_bottom_collage(tops=tops, bottom_img=bottoms[0]),
            mode="two_tops_bottom_mixed_collage",
            product_count=3,
        )

    if len(tops) == 2 and not bottoms and not dresses:
        return TryonCollageResult(
            image=_build_horizontal_collage(tops),
            mode="two_tops_horizontal_collage",
            product_count=2,
        )

    if len(dresses) == 1 and len(tops) == 1 and not bottoms:
        return TryonCollageResult(
            image=_build_horizontal_collage([tops[0], dresses[0]]),
            mode="top_dress_horizontal_collage",
            product_count=2,
        )

    if len(dresses) == 1 and len(bottoms) == 1 and not tops:
        return TryonCollageResult(
            image=_build_compact_vertical_collage([dresses[0], bottoms[0]]),
            mode="dress_bottom_vertical_collage",
            product_count=2,
        )

    if len(dresses) == 1 and len(tops) == 2 and not bottoms:
        return TryonCollageResult(
            image=_build_horizontal_collage([*tops, dresses[0]]),
            mode="two_tops_dress_horizontal_collage",
            product_count=3,
        )

    if len(dresses) == 1 and tops and len(tops) <= 2 and len(bottoms) == 1:
        outfit_stack = _build_top_bottom_collage(tops=tops, bottom_img=bottoms[0])
        return TryonCollageResult(
            image=_build_horizontal_collage([outfit_stack, dresses[0]]),
            mode=(
                "dress_with_top_bottom_stack_collage"
                if len(tops) == 1
                else "dress_with_two_tops_bottom_stack_collage"
            ),
            product_count=len(products),
        )

    return TryonCollageResult(
        image=_build_horizontal_collage([_prepare_collage_image(item.image) for item in products]),
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


def _group_products_by_type(
    products: list[ProductReferenceInput],
) -> dict[str, list[Image.Image]]:
    grouped: dict[str, list[Image.Image]] = {}
    for item in products:
        grouped.setdefault(normalize_product_type(item.type), []).append(
            _prepare_collage_image(item.image),
        )
    return grouped


def _prepare_collage_image(image: Image.Image) -> Image.Image:
    return _trim_empty_border(image).convert("RGB")


def _trim_empty_border(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"}:
        alpha = image.getchannel("A")
        bbox = alpha.getbbox()
        if bbox is not None:
            return image.crop(_expand_bbox(bbox, image.size, TRIM_MARGIN))

    rgb = image.convert("RGB")
    bg_color = _average_corner_color(rgb)
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, bg_color))
    mask = diff.point(lambda value: 255 if value > TRIM_TOLERANCE else 0).convert("L")
    bbox = mask.getbbox()
    if bbox is None:
        return rgb
    return rgb.crop(_expand_bbox(bbox, rgb.size, TRIM_MARGIN))


def _average_corner_color(image: Image.Image) -> tuple[int, int, int]:
    corners: list[tuple[int, int, int]] = []
    for point in (
        (0, 0),
        (image.width - 1, 0),
        (0, image.height - 1),
        (image.width - 1, image.height - 1),
    ):
        pixel = image.getpixel(point)
        if isinstance(pixel, tuple):
            corners.append((int(pixel[0]), int(pixel[1]), int(pixel[2])))
        elif isinstance(pixel, int | float):
            value = int(pixel)
            corners.append((value, value, value))
        else:
            corners.append((0, 0, 0))
    return (
        int(round(sum(pixel[0] for pixel in corners) / 4)),
        int(round(sum(pixel[1] for pixel in corners) / 4)),
        int(round(sum(pixel[2] for pixel in corners) / 4)),
    )


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    size: tuple[int, int],
    margin: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = size
    return (
        max(0, left - margin),
        max(0, top - margin),
        min(width, right + margin),
        min(height, bottom + margin),
    )


def _fit_to_box(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(
        float(max_width) / float(image.width),
        float(max_height) / float(image.height),
        1.0,
    )
    target_width = max(1, int(round(float(image.width) * scale)))
    target_height = max(1, int(round(float(image.height) * scale)))
    if target_width == image.width and target_height == image.height:
        return image
    # Gamma-correct only — no sharpen (matches the previous plain-Lanczos behaviour,
    # the sole change here is linear-light resampling).
    return resize_image(image, (target_width, target_height), sharpen=False)


def _build_horizontal_collage(
    images: list[Image.Image],
    *,
    padding: int = OUTER_PADDING,
) -> Image.Image:
    if not images:
        raise ValueError("At least one image is required.")

    prepared = [_fit_to_box(image, MAX_ROW_ITEM_W, MAX_ROW_ITEM_H) for image in images]
    width = sum(image.width for image in prepared) + COLLAGE_GAP * (len(prepared) - 1)
    height = max(image.height for image in prepared)
    canvas = Image.new("RGB", (width + padding * 2, height + padding * 2), "white")

    offset_x = padding
    for image in prepared:
        offset_y = padding + (height - image.height) // 2
        canvas.paste(image, (offset_x, offset_y))
        offset_x += image.width + COLLAGE_GAP

    return canvas


def _build_compact_vertical_collage(
    images: list[Image.Image],
    *,
    padding: int = OUTER_PADDING,
) -> Image.Image:
    if not images:
        raise ValueError("At least one image is required.")

    prepared = [
        _fit_to_box(image, MAX_VERTICAL_SECTION_W, MAX_VERTICAL_SECTION_H)
        for image in images
    ]
    content_width = max(image.width for image in prepared)
    content_height = sum(image.height for image in prepared) + COLLAGE_GAP * (len(prepared) - 1)
    width = content_width + padding * 2
    height = content_height + padding * 2
    canvas = Image.new("RGB", (width, height), "white")

    offset_y = padding
    for image in prepared:
        offset_x = padding + (content_width - image.width) // 2
        canvas.paste(image, (offset_x, offset_y))
        offset_y += image.height + COLLAGE_GAP

    return canvas


def _build_top_bottom_collage(tops: list[Image.Image], bottom_img: Image.Image) -> Image.Image:
    top_section = _build_horizontal_collage(tops, padding=INNER_PADDING)
    return _build_compact_vertical_collage([top_section, bottom_img], padding=INNER_PADDING)

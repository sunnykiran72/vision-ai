from __future__ import annotations

from PIL import Image

from app.utils.tryon_collage import (
    COLLAGE_GAP,
    ProductReferenceInput,
    build_product_reference,
    compose_reference_with_user,
)

RED = (224, 32, 32)
BLUE = (32, 64, 224)
GREEN = (32, 160, 80)
YELLOW = (224, 184, 32)
WHITE = (255, 255, 255)


def _solid_image(width: int, height: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def _bordered_image(
    width: int,
    height: int,
    color: tuple[int, int, int],
    inset: int,
) -> Image.Image:
    image = Image.new("RGB", (width, height), WHITE)
    for y in range(inset, height - inset):
        for x in range(inset, width - inset):
            image.putpixel((x, y), color)
    return image


def _color_bbox(image: Image.Image, color: tuple[int, int, int]) -> tuple[int, int, int, int]:
    pixels = image.load()
    min_x = image.width
    min_y = image.height
    max_x = -1
    max_y = -1

    for y in range(image.height):
        for x in range(image.width):
            if pixels[x, y] == color:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    if max_x == -1:
        raise AssertionError(f"Color {color} was not found in image")

    return min_x, min_y, max_x, max_y


def test_build_product_reference_returns_single_product_without_collage() -> None:
    product = _solid_image(240, 320, RED)

    result = build_product_reference([ProductReferenceInput(image=product, type="top")])

    assert result.mode == "single_product"
    assert result.product_count == 1
    assert result.image.size == product.size
    assert result.image.getpixel((0, 0)) == RED


def test_build_product_reference_returns_single_bottom_without_collage() -> None:
    product = _solid_image(180, 300, BLUE)

    result = build_product_reference([ProductReferenceInput(image=product, type="bottom")])

    assert result.mode == "single_product"
    assert result.product_count == 1
    assert result.image.size == product.size
    assert result.image.getpixel((0, 0)) == BLUE


def test_build_product_reference_creates_top_bottom_vertical_collage() -> None:
    top = _solid_image(500, 400, RED)
    bottom = _solid_image(420, 700, BLUE)

    result = build_product_reference(
        [
            ProductReferenceInput(image=top, type="top"),
            ProductReferenceInput(image=bottom, type="bottom"),
        ],
    )

    assert result.mode == "top_bottom_vertical_collage"
    assert result.product_count == 2

    red_bbox = _color_bbox(result.image, RED)
    blue_bbox = _color_bbox(result.image, BLUE)

    assert red_bbox[3] < blue_bbox[1]
    assert red_bbox[0] < red_bbox[2]
    assert blue_bbox[0] < blue_bbox[2]
    assert blue_bbox[1] - red_bbox[3] <= COLLAGE_GAP + 1


def test_build_product_reference_creates_two_tops_bottom_mixed_collage() -> None:
    left_top = _solid_image(300, 360, RED)
    right_top = _solid_image(300, 360, GREEN)
    bottom = _solid_image(360, 500, BLUE)

    result = build_product_reference(
        [
            ProductReferenceInput(image=left_top, type="top"),
            ProductReferenceInput(image=right_top, type="top"),
            ProductReferenceInput(image=bottom, type="bottom"),
        ],
    )

    assert result.mode == "two_tops_bottom_mixed_collage"
    assert result.product_count == 3

    red_bbox = _color_bbox(result.image, RED)
    green_bbox = _color_bbox(result.image, GREEN)
    blue_bbox = _color_bbox(result.image, BLUE)

    assert red_bbox[2] < green_bbox[0]
    assert red_bbox[3] < blue_bbox[1]
    assert green_bbox[3] < blue_bbox[1]
    assert blue_bbox[1] - max(red_bbox[3], green_bbox[3]) <= COLLAGE_GAP + 1


def test_build_product_reference_creates_two_tops_horizontal_collage() -> None:
    left_top = _solid_image(300, 360, RED)
    right_top = _solid_image(300, 360, GREEN)

    result = build_product_reference(
        [
            ProductReferenceInput(image=left_top, type="top"),
            ProductReferenceInput(image=right_top, type="top"),
        ],
    )

    assert result.mode == "two_tops_horizontal_collage"

    red_bbox = _color_bbox(result.image, RED)
    green_bbox = _color_bbox(result.image, GREEN)

    assert red_bbox[2] < green_bbox[0]
    assert red_bbox[1] == green_bbox[1]


def test_build_product_reference_treats_outer_as_top_for_collage_layout() -> None:
    outer = _solid_image(500, 400, RED)
    bottom = _solid_image(420, 700, BLUE)

    result = build_product_reference(
        [
            ProductReferenceInput(image=outer, type="outer"),
            ProductReferenceInput(image=bottom, type="bottom"),
        ],
    )

    assert result.mode == "top_bottom_vertical_collage"
    assert result.product_count == 2

    red_bbox = _color_bbox(result.image, RED)
    blue_bbox = _color_bbox(result.image, BLUE)

    assert red_bbox[3] < blue_bbox[1]


def test_build_product_reference_creates_top_dress_horizontal_collage() -> None:
    top = _solid_image(300, 360, RED)
    dress = _solid_image(320, 620, YELLOW)

    result = build_product_reference(
        [
            ProductReferenceInput(image=top, type="top"),
            ProductReferenceInput(image=dress, type="dress"),
        ],
    )

    assert result.mode == "top_dress_horizontal_collage"

    red_bbox = _color_bbox(result.image, RED)
    yellow_bbox = _color_bbox(result.image, YELLOW)

    assert red_bbox[2] < yellow_bbox[0]


def test_build_product_reference_trims_empty_borders_for_multi_garment_collage() -> None:
    top = _bordered_image(120, 120, RED, 48)
    dress = _bordered_image(120, 120, YELLOW, 48)

    result = build_product_reference(
        [
            ProductReferenceInput(image=top, type="top"),
            ProductReferenceInput(image=dress, type="dress"),
        ],
    )

    assert result.mode == "top_dress_horizontal_collage"
    assert result.image.width < 140

    red_bbox = _color_bbox(result.image, RED)
    yellow_bbox = _color_bbox(result.image, YELLOW)

    assert red_bbox[2] < yellow_bbox[0]


def test_build_product_reference_creates_dress_bottom_vertical_collage() -> None:
    dress = _solid_image(320, 620, YELLOW)
    bottom = _solid_image(360, 500, BLUE)

    result = build_product_reference(
        [
            ProductReferenceInput(image=dress, type="dress"),
            ProductReferenceInput(image=bottom, type="bottom"),
        ],
    )

    assert result.mode == "dress_bottom_vertical_collage"

    yellow_bbox = _color_bbox(result.image, YELLOW)
    blue_bbox = _color_bbox(result.image, BLUE)

    assert yellow_bbox[3] < blue_bbox[1]
    assert blue_bbox[1] - yellow_bbox[3] <= COLLAGE_GAP + 1


def test_build_product_reference_bounds_wide_bottom_in_vertical_collage() -> None:
    dress = _solid_image(320, 620, YELLOW)
    wide_bottom = _solid_image(2000, 400, BLUE)

    result = build_product_reference(
        [
            ProductReferenceInput(image=dress, type="dress"),
            ProductReferenceInput(image=wide_bottom, type="bottom"),
        ],
    )

    assert result.mode == "dress_bottom_vertical_collage"
    assert result.image.width <= 744

    yellow_bbox = _color_bbox(result.image, YELLOW)
    blue_bbox = _color_bbox(result.image, BLUE)

    assert yellow_bbox[3] < blue_bbox[1]
    assert blue_bbox[2] - blue_bbox[0] <= 720


def test_build_product_reference_creates_two_tops_dress_horizontal_collage() -> None:
    first_top = _solid_image(300, 360, RED)
    second_top = _solid_image(300, 360, GREEN)
    dress = _solid_image(320, 620, YELLOW)

    result = build_product_reference(
        [
            ProductReferenceInput(image=dress, type="dress"),
            ProductReferenceInput(image=first_top, type="top"),
            ProductReferenceInput(image=second_top, type="top"),
        ],
    )

    assert result.mode == "two_tops_dress_horizontal_collage"

    red_bbox = _color_bbox(result.image, RED)
    green_bbox = _color_bbox(result.image, GREEN)
    yellow_bbox = _color_bbox(result.image, YELLOW)

    assert red_bbox[2] < green_bbox[0]
    assert green_bbox[2] < yellow_bbox[0]


def test_build_product_reference_bounds_wide_item_in_horizontal_collage() -> None:
    top = _solid_image(300, 360, RED)
    wide_dress = _solid_image(2000, 500, YELLOW)

    result = build_product_reference(
        [
            ProductReferenceInput(image=top, type="top"),
            ProductReferenceInput(image=wide_dress, type="dress"),
        ],
    )

    assert result.mode == "top_dress_horizontal_collage"

    red_bbox = _color_bbox(result.image, RED)
    yellow_bbox = _color_bbox(result.image, YELLOW)

    assert red_bbox[2] < yellow_bbox[0]
    assert yellow_bbox[2] - yellow_bbox[0] <= 640


def test_build_product_reference_creates_dress_with_top_bottom_stack_collage() -> None:
    top = _solid_image(300, 360, RED)
    bottom = _solid_image(360, 500, BLUE)
    dress = _solid_image(320, 620, YELLOW)

    result = build_product_reference(
        [
            ProductReferenceInput(image=dress, type="dress"),
            ProductReferenceInput(image=top, type="top"),
            ProductReferenceInput(image=bottom, type="bottom"),
        ],
    )

    assert result.mode == "dress_with_top_bottom_stack_collage"

    red_bbox = _color_bbox(result.image, RED)
    blue_bbox = _color_bbox(result.image, BLUE)
    yellow_bbox = _color_bbox(result.image, YELLOW)

    assert red_bbox[3] < blue_bbox[1]
    assert blue_bbox[2] < yellow_bbox[0]


def test_build_product_reference_creates_dress_with_two_tops_bottom_stack_collage() -> None:
    first_top = _solid_image(300, 360, RED)
    second_top = _solid_image(300, 360, GREEN)
    bottom = _solid_image(360, 500, BLUE)
    dress = _solid_image(320, 620, YELLOW)

    result = build_product_reference(
        [
            ProductReferenceInput(image=dress, type="dress"),
            ProductReferenceInput(image=first_top, type="top"),
            ProductReferenceInput(image=second_top, type="top"),
            ProductReferenceInput(image=bottom, type="bottom"),
        ],
    )

    assert result.mode == "dress_with_two_tops_bottom_stack_collage"

    red_bbox = _color_bbox(result.image, RED)
    green_bbox = _color_bbox(result.image, GREEN)
    blue_bbox = _color_bbox(result.image, BLUE)
    yellow_bbox = _color_bbox(result.image, YELLOW)

    assert red_bbox[2] < green_bbox[0]
    assert red_bbox[3] < blue_bbox[1]
    assert green_bbox[3] < blue_bbox[1]
    assert blue_bbox[2] < yellow_bbox[0]


def test_build_product_reference_creates_horizontal_collage_for_other_product_sets() -> None:
    first = _solid_image(100, 200, RED)
    second = _solid_image(80, 100, BLUE)
    third = _solid_image(50, 200, GREEN)

    result = build_product_reference(
        [
            ProductReferenceInput(image=first, type="top"),
            ProductReferenceInput(image=second, type="shoe"),
            ProductReferenceInput(image=third, type="accessory"),
        ],
    )

    assert result.mode == "multi_product_horizontal_collage"
    assert result.product_count == 3

    red_bbox = _color_bbox(result.image, RED)
    blue_bbox = _color_bbox(result.image, BLUE)
    green_bbox = _color_bbox(result.image, GREEN)

    assert red_bbox[2] < blue_bbox[0]
    assert blue_bbox[2] < green_bbox[0]


def test_compose_reference_with_user_places_reference_left_and_user_right() -> None:
    reference = _solid_image(150, 100, RED)
    user = _solid_image(100, 200, BLUE)

    result = compose_reference_with_user(reference, user)

    assert result.mode == "products_left_user_right"
    assert result.product_count == 1
    assert result.image.size == (400, 200)
    assert result.image.getpixel((0, 0)) == RED
    assert result.image.getpixel((300, 0)) == BLUE

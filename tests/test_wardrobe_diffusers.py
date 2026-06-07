from __future__ import annotations

import pytest
from PIL import Image

from app.clients.qwen_diffusers_engine import (
    QwenDiffusersWardrobeEngine,
    WardrobeDiffusersRuntimeError,
    resize_input_for_model,
)
from app.config import Settings


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((1600, 1200), (1024, 768)),
        ((1200, 1600), (768, 1024)),
        ((2000, 1000), (1024, 512)),
        ((800, 900), (800, 900)),  # within max side -> unchanged, no /16 rounding
        ((1024, 1024), (1024, 1024)),
    ],
)
def test_resize_input_for_model_matches_reference(
    size: tuple[int, int],
    expected: tuple[int, int],
) -> None:
    resized = resize_input_for_model(Image.new("RGB", size))
    assert resized.size == expected


def test_run_extract_rejects_unknown_category_without_loading() -> None:
    engine = QwenDiffusersWardrobeEngine(
        Settings(
            QWEN_IMAGE_EDIT_MODEL_PATH="/model",
            WARDROBE_LORA_TOP_PATH="/top.safetensors",
            WARDROBE_LORA_BOTTOM_PATH="/bottom.safetensors",
            WARDROBE_LORA_DRESS_PATH="/dress.safetensors",
        ),
    )
    with pytest.raises(WardrobeDiffusersRuntimeError):
        engine.run_extract(
            input_image=Image.new("RGB", (512, 512)),
            prompt="x",
            garment_type="outer",
        )

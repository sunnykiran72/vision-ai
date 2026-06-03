from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image


@dataclass(frozen=True)
class WardrobeRunResult:
    image: Image.Image
    metadata: dict[str, Any]
    wall_seconds: float


@dataclass(frozen=True)
class WardrobeRuntimeStatus:
    loaded: bool
    backend: str | None
    loras_loaded: bool


class WardrobeRunner(Protocol):
    def warmup(self) -> None:
        pass

    def status(self) -> WardrobeRuntimeStatus:
        pass

    def run_extract(
        self,
        *,
        input_image_path: str,
        prompt: str,
        garment_type: str,
        output_path: str,
    ) -> WardrobeRunResult:
        pass

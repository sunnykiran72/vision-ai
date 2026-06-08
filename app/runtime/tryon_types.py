from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image


@dataclass(frozen=True)
class TryonRuntimeStatus:
    loaded: bool
    backend: str | None
    lora_loaded: bool


@dataclass(frozen=True)
class TryonRunResult:
    image: Image.Image
    metadata: dict[str, Any]
    wall_seconds: float


class TryonRunner(Protocol):
    def warmup(self) -> None: ...
    def status(self) -> TryonRuntimeStatus: ...
    def run_tryon(
        self,
        *,
        person_image: Image.Image,
        garment_reference_image: Image.Image,
        prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        output_width: int,
        output_height: int,
        lora_key: str | None = None,
    ) -> TryonRunResult: ...

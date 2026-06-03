from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.config import Settings
from app.models.tryon import TryonProduct

TryonLoraKey = Literal["top", "bottom", "dress", "multi"]


@dataclass(frozen=True)
class TryonRoutingDecision:
    lora_key: TryonLoraKey
    trigger_caption: str


def resolve_tryon_route(
    products: list[TryonProduct],
    settings: Settings,
) -> TryonRoutingDecision:
    if len(products) >= 2:
        return TryonRoutingDecision(
            lora_key="multi",
            trigger_caption=settings.tryon_prompt_trigger_multi,
        )

    product_type = products[0].type.value
    if product_type in ("top", "outer"):
        return TryonRoutingDecision(
            lora_key="top",
            trigger_caption=settings.tryon_prompt_trigger_top,
        )
    if product_type == "bottom":
        return TryonRoutingDecision(
            lora_key="bottom",
            trigger_caption=settings.tryon_prompt_trigger_bottom,
        )
    if product_type == "dress":
        return TryonRoutingDecision(
            lora_key="dress",
            trigger_caption=settings.tryon_prompt_trigger_dress,
        )
    raise ValueError(f"Unsupported try-on product type: {product_type}")

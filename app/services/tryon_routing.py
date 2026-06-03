from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.config import Settings, get_enabled_tryon_specialists
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
        return _ensure_enabled(
            settings,
            TryonRoutingDecision(
                lora_key="multi",
                trigger_caption=settings.tryon_prompt_trigger_multi,
            ),
        )

    product_type = products[0].type.value
    if product_type in ("top", "outer"):
        return _ensure_enabled(
            settings,
            TryonRoutingDecision(
                lora_key="top",
                trigger_caption=settings.tryon_prompt_trigger_top,
            ),
        )
    if product_type == "bottom":
        return _ensure_enabled(
            settings,
            TryonRoutingDecision(
                lora_key="bottom",
                trigger_caption=settings.tryon_prompt_trigger_bottom,
            ),
        )
    if product_type == "dress":
        return _ensure_enabled(
            settings,
            TryonRoutingDecision(
                lora_key="dress",
                trigger_caption=settings.tryon_prompt_trigger_dress,
            ),
        )
    raise ValueError(f"Unsupported try-on product type: {product_type}")


def _ensure_enabled(
    settings: Settings,
    decision: TryonRoutingDecision,
) -> TryonRoutingDecision:
    if settings.tryon_use_specialists and decision.lora_key not in get_enabled_tryon_specialists(
        settings,
    ):
        raise ValueError(f"Try-on specialist is not enabled: {decision.lora_key}")
    return decision

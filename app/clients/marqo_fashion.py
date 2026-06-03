from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass
from typing import Any

from PIL import Image

from app.constants import wardrobe as wardrobe_constants
from app.constants.wardrobe import MarqoCategoryCandidate

logger = logging.getLogger("glamify-ai")


class MarqoClassificationRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class MarqoClassificationResult:
    applied: bool
    category_key: str
    category_label: str
    score: float
    min_confidence: float
    top_matches: list[dict[str, object]]
    reason: str


class MarqoFashionSiglipClient:
    def __init__(
        self,
        *,
        model_id: str = wardrobe_constants.MARQO_MODEL_ID,
        min_confidence: float = wardrobe_constants.MARQO_CONFIDENCE_THRESHOLD,
        top_k: int = wardrobe_constants.MARQO_TOP_K,
    ) -> None:
        self._model_id = model_id
        self._min_confidence = float(min_confidence)
        self._top_k = int(top_k)
        self._image_preprocess: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device = "cpu"
        self._dtype: Any | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return (
            self._model is not None
            and self._image_preprocess is not None
            and self._tokenizer is not None
        )

    def classify(self, *, image: Image.Image, garment_type: str) -> MarqoClassificationResult:
        candidates = wardrobe_constants.MARQO_CANDIDATES_BY_TYPE.get(str(garment_type), ())
        if not candidates:
            return MarqoClassificationResult(
                applied=False,
                category_key="",
                category_label="",
                score=0.0,
                min_confidence=self._min_confidence,
                top_matches=[],
                reason="no_candidates",
            )

        labels = [candidate.label for candidate in candidates]
        scores = self._score_labels(image=image, labels=labels)
        by_label: dict[str, MarqoCategoryCandidate] = {
            candidate.label: candidate for candidate in candidates
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if self._top_k > 0:
            ranked = ranked[: self._top_k]

        top_matches: list[dict[str, object]] = []
        for label, score in ranked:
            candidate = by_label.get(label)
            if candidate is None:
                continue
            top_matches.append(
                {
                    "category_key": candidate.key,
                    "category_label": candidate.label,
                    "parent_key": candidate.parent_key,
                    "parent_label": candidate.parent_label,
                    "score": float(score),
                },
            )

        if not top_matches:
            return MarqoClassificationResult(
                applied=False,
                category_key="",
                category_label="",
                score=0.0,
                min_confidence=self._min_confidence,
                top_matches=[],
                reason="no_ranked_matches",
            )

        best = top_matches[0]
        raw_score = best.get("score")
        score = float(raw_score) if isinstance(raw_score, int | float | str) else 0.0
        applied = score >= self._min_confidence
        return MarqoClassificationResult(
            applied=applied,
            category_key=str(best.get("category_key") or ""),
            category_label=str(best.get("category_label") or ""),
            score=score,
            min_confidence=self._min_confidence,
            top_matches=top_matches,
            reason="applied" if applied else "below_threshold",
        )

    def _score_labels(self, *, image: Image.Image, labels: list[str]) -> dict[str, float]:
        self._ensure_ready()
        if (
            self._model is None
            or self._image_preprocess is None
            or self._tokenizer is None
            or self._torch is None
        ):
            raise MarqoClassificationRuntimeError("Marqo fashionSigLIP is not loaded.")

        rgb = image.convert("RGB")
        with self._infer_lock, self._torch.inference_mode():
            image_tensor = self._image_preprocess(rgb).unsqueeze(0).to(
                device=self._device,
                dtype=self._dtype,
            )
            text_tensor = self._tokenizer(labels).to(self._device)
            image_features = self._model.encode_image(image_tensor, normalize=True)
            text_features = self._model.encode_text(text_tensor, normalize=True)
            logits = 100.0 * (image_features @ text_features.T)
            probs = logits.softmax(dim=-1).detach().float().cpu().squeeze(0).tolist()
        return {label: float(score) for label, score in zip(labels, probs, strict=False)}

    def _ensure_ready(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            try:
                torch = importlib.import_module("torch")
                open_clip = importlib.import_module("open_clip")
            except Exception as exc:
                raise MarqoClassificationRuntimeError(
                    f"Unable to import Marqo fashionSigLIP dependencies: {exc}",
                ) from exc

            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._dtype = torch.float16 if self._device == "cuda" else torch.float32
            model_id = self._model_id
            if not model_id.startswith("hf-hub:"):
                model_id = f"hf-hub:{model_id}"
            logger.info(
                "Loading wardrobe Marqo fashionSigLIP from %s on %s",
                model_id,
                self._device,
            )
            model, _, image_preprocess = open_clip.create_model_and_transforms(
                model_id,
                precision="fp16" if self._device == "cuda" else "fp32",
            )
            model = model.to(self._device)
            model.eval()
            self._model = model
            self._image_preprocess = image_preprocess
            self._tokenizer = open_clip.get_tokenizer(model_id)

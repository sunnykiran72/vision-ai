from __future__ import annotations

import importlib
import logging
import threading
from typing import Any

from PIL import Image

from app.constants import wardrobe as wardrobe_constants

logger = logging.getLogger("glamify-ai")


class FashionDetectionRuntimeError(RuntimeError):
    pass


class FashionDetectionClient:
    def __init__(
        self,
        *,
        model_id: str = wardrobe_constants.FASHION_DETECTION_MODEL_ID,
        threshold: float = wardrobe_constants.FASHION_DETECTION_THRESHOLD,
    ) -> None:
        self._model_id = model_id
        self._threshold = float(threshold)
        self._processor: Any | None = None
        self._model: Any | None = None
        self._id2label: dict[int, str] = {}
        self._device = "cpu"
        self._torch: Any | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._processor is not None and self._model is not None

    def has_garment(self, image: Image.Image) -> bool:
        return bool(self.detect(image))

    def detect(self, image: Image.Image) -> list[dict[str, object]]:
        self._ensure_ready()
        if self._processor is None or self._model is None or self._torch is None:
            raise FashionDetectionRuntimeError("Fashion detector is not loaded.")

        rgb = image.convert("RGB")
        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with self._infer_lock, self._torch.inference_mode():
            outputs = self._model(**inputs)

        processed = self._processor.post_process_object_detection(
            outputs,
            threshold=self._threshold,
            target_sizes=[(rgb.height, rgb.width)],
        )
        if not processed:
            return []

        result = processed[0]
        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("labels")
        if boxes is None or scores is None or labels is None:
            return []

        detections: list[dict[str, object]] = []
        for box, score, label_idx in zip(boxes, scores, labels, strict=False):
            x0, y0, x1, y1 = [int(round(float(v))) for v in box.tolist()]
            if x1 <= x0 or y1 <= y0:
                continue
            class_id = int(label_idx.item() if hasattr(label_idx, "item") else label_idx)
            detections.append(
                {
                    "bbox": [x0, y0, x1, y1],
                    "score": float(score.item() if hasattr(score, "item") else score),
                    "label": self._id2label.get(class_id, str(class_id)),
                    "class_id": class_id,
                    "source": "fashion_object_detection",
                },
            )
        return detections

    def _ensure_ready(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            try:
                torch = importlib.import_module("torch")
                transformers = importlib.import_module("transformers")
                AutoImageProcessor = transformers.AutoImageProcessor
                AutoModelForObjectDetection = transformers.AutoModelForObjectDetection
            except Exception as exc:
                raise FashionDetectionRuntimeError(
                    f"Unable to import fashion detector dependencies: {exc}",
                ) from exc

            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "Loading wardrobe fashion detector from %s on %s",
                self._model_id,
                self._device,
            )
            self._processor = AutoImageProcessor.from_pretrained(self._model_id)
            self._model = AutoModelForObjectDetection.from_pretrained(self._model_id).to(
                self._device,
            )
            self._model.eval()
            raw_id2label = getattr(self._model.config, "id2label", {}) or {}
            self._id2label = {int(key): str(value) for key, value in raw_id2label.items()}

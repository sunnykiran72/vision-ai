from __future__ import annotations

import importlib
import logging
import threading
from functools import lru_cache
from typing import Any

from PIL import Image

from app.constants import user_validation as constants

logger = logging.getLogger("glamify-ai")


class PersonDetectionRuntimeError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_person_detection_client() -> PersonDetectionClient:
    return PersonDetectionClient()


class PersonDetectionClient:
    def __init__(
        self,
        *,
        model_id: str = constants.PERSON_DETECTION_MODEL_ID,
        score_threshold: float = constants.PERSON_DETECTION_SCORE_THRESHOLD,
    ) -> None:
        self._model_id = model_id
        self._score_threshold = float(score_threshold)
        self._processor: Any | None = None
        self._model: Any | None = None
        self._id2label: dict[int, str] = {}
        self._device = "cpu"
        self._dtype = "float32"
        self._torch: Any | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def is_loaded(self) -> bool:
        return self._processor is not None and self._model is not None

    def ensure_ready(self) -> None:
        self._ensure_ready()

    def detect(self, image: Image.Image) -> list[dict[str, object]]:
        self._ensure_ready()
        if self._processor is None or self._model is None or self._torch is None:
            raise PersonDetectionRuntimeError("Person detector is not loaded.")

        rgb = image.convert("RGB")
        inputs = self._processor(images=rgb, return_tensors="pt")
        # The image processor emits float32 pixel_values, but on CUDA the model is loaded in
        # fp16. Cast floating-point inputs to the model's dtype (leave integer masks as-is) or
        # conv2d raises "Input type (FloatTensor) and weight type (HalfTensor) should be the same".
        model_dtype = next(self._model.parameters()).dtype
        inputs = {
            key: (
                value.to(self._device, dtype=model_dtype)
                if value.is_floating_point()
                else value.to(self._device)
            )
            for key, value in inputs.items()
        }

        with self._infer_lock, self._torch.inference_mode():
            outputs = self._model(**inputs)

        target_sizes = self._torch.tensor([(rgb.height, rgb.width)], device=self._device)
        processed = self._processor.post_process_object_detection(
            outputs,
            threshold=self._score_threshold,
            target_sizes=target_sizes,
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
        image_area = float(rgb.width * rgb.height)
        for box, score, label_idx in zip(boxes, scores, labels, strict=False):
            x1, y1, x2, y2 = [round(float(v), 2) for v in box.detach().cpu().tolist()]
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(label_idx.item() if hasattr(label_idx, "item") else label_idx)
            label = self._id2label.get(class_id, str(class_id))
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            detections.append(
                {
                    "label": label,
                    "class_id": class_id,
                    "score": round(float(score.item() if hasattr(score, "item") else score), 4),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "metrics": {
                        "width_ratio": round(width / rgb.width, 4),
                        "height_ratio": round(height / rgb.height, 4),
                        "area_ratio": round((width * height) / image_area, 4),
                        "top_ratio": round(y1 / rgb.height, 4),
                        "bottom_ratio": round(y2 / rgb.height, 4),
                    },
                    "source": "rtdetr_person_detection",
                },
            )
        detections.sort(key=lambda item: float(item["score"]), reverse=True)
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
                raise PersonDetectionRuntimeError(
                    f"Unable to import person detector dependencies: {exc}",
                ) from exc

            self._torch = torch
            if torch.cuda.is_available():
                self._device = "cuda"
                dtype = torch.float16
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self._device = "mps"
                dtype = torch.float32
            else:
                self._device = "cpu"
                dtype = torch.float32
            self._dtype = str(dtype).replace("torch.", "")
            logger.info(
                "Loading user image person detector from %s on %s",
                self._model_id,
                self._device,
            )
            self._processor = AutoImageProcessor.from_pretrained(self._model_id)
            self._model = AutoModelForObjectDetection.from_pretrained(
                self._model_id,
                dtype=dtype,
            ).to(self._device)
            self._model.eval()
            raw_id2label = getattr(self._model.config, "id2label", {}) or {}
            self._id2label = {int(key): str(value) for key, value in raw_id2label.items()}

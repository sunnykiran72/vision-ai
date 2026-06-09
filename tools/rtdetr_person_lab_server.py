from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

MODEL_ID = "PekingU/rtdetr_r50vd_coco_o365"
PAGE = Path(__file__).resolve().parent / "rtdetr_person_lab.html"
PORT = 8770

app = FastAPI(title="RT-DETR Person Validation Lab")
_state: dict[str, Any] = {
    "ready": False,
    "loading": False,
    "load_error": "",
    "processor": None,
    "model": None,
    "torch": None,
    "device": "cpu",
    "dtype": "float32",
    "load_seconds": 0.0,
}


@app.get("/", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(PAGE)


@app.get("/status")
def status() -> dict[str, Any]:
    return {
        "ready": _state["ready"],
        "loading": _state["loading"],
        "load_error": _state["load_error"],
        "model_id": MODEL_ID,
        "device": _state["device"],
        "dtype": _state["dtype"],
        "load_seconds": _state["load_seconds"],
    }


@app.post("/detect")
async def detect(
    image: Annotated[UploadFile, File()],
    score_threshold: Annotated[float, Form()] = 0.30,
    multi_person_score_threshold: Annotated[float, Form()] = 0.45,
    min_height_ratio: Annotated[float, Form()] = 0.50,
    min_area_ratio: Annotated[float, Form()] = 0.12,
    min_bottom_ratio: Annotated[float, Form()] = 0.68,
    blur_threshold: Annotated[float, Form()] = 80.0,
) -> JSONResponse:
    load_error = _ensure_loaded()
    if load_error:
        return JSONResponse({"ok": False, "error": load_error}, status_code=500)

    raw = await image.read()
    try:
        decoded = Image.open(io.BytesIO(raw))
        decoded.load()
        source = decoded.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        return JSONResponse(
            {"ok": False, "error": f"Invalid image: {exc}"},
            status_code=422,
        )

    started = time.perf_counter()
    blur = _blur_metadata(source)
    boxes = _detect_boxes(source, score_threshold=score_threshold)
    timings = {"detect_seconds": round(time.perf_counter() - started, 4)}
    decision = _decide(
        boxes=boxes,
        image_size=source.size,
        blur=blur,
        multi_person_score_threshold=multi_person_score_threshold,
        min_height_ratio=min_height_ratio,
        min_area_ratio=min_area_ratio,
        min_bottom_ratio=min_bottom_ratio,
        blur_threshold=blur_threshold,
    )

    return JSONResponse(
        {
            "ok": True,
            "model": {
                "id": MODEL_ID,
                "device": _state["device"],
                "dtype": _state["dtype"],
            },
            "image": {
                "filename": image.filename,
                "content_type": image.content_type,
                "width": source.width,
                "height": source.height,
            },
            "thresholds": {
                "score_threshold": score_threshold,
                "multi_person_score_threshold": multi_person_score_threshold,
                "min_height_ratio": min_height_ratio,
                "min_area_ratio": min_area_ratio,
                "min_bottom_ratio": min_bottom_ratio,
                "blur_threshold": blur_threshold,
            },
            "blur": blur,
            "boxes": boxes,
            "decision": decision,
            "timings": timings,
        }
    )


def _ensure_loaded() -> str:
    if _state["ready"]:
        return ""
    if _state["loading"]:
        return "Model is already loading. Retry in a few seconds."
    _state["loading"] = True
    _state["load_error"] = ""
    started = time.perf_counter()
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float32
        else:
            device = "cpu"
            dtype = torch.float32

        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForObjectDetection.from_pretrained(MODEL_ID, torch_dtype=dtype)
        model.to(device)
        model.eval()

        _state.update(
            {
                "ready": True,
                "processor": processor,
                "model": model,
                "torch": torch,
                "device": device,
                "dtype": str(dtype).replace("torch.", ""),
                "load_seconds": round(time.perf_counter() - started, 3),
            }
        )
        return ""
    except Exception as exc:
        _state["load_error"] = f"{type(exc).__name__}: {exc}"
        return _state["load_error"]
    finally:
        _state["loading"] = False


def _detect_boxes(image: Image.Image, *, score_threshold: float) -> list[dict[str, Any]]:
    torch = _state["torch"]
    processor = _state["processor"]
    model = _state["model"]
    device = _state["device"]
    if torch is None or processor is None or model is None:
        raise RuntimeError("Detector is not loaded.")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    target_sizes = torch.tensor([(image.height, image.width)], device=device)
    results = processor.post_process_object_detection(
        outputs,
        threshold=float(score_threshold),
        target_sizes=target_sizes,
    )[0]
    id2label = getattr(model.config, "id2label", {})
    boxes: list[dict[str, Any]] = []
    for score, label_id, box in zip(
        results["scores"],
        results["labels"],
        results["boxes"],
        strict=True,
    ):
        label = str(id2label.get(int(label_id), label_id))
        coords = [round(float(v), 2) for v in box.detach().cpu().tolist()]
        x1, y1, x2, y2 = coords
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        boxes.append(
            {
                "label": label,
                "score": round(float(score), 4),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "metrics": {
                    "width_ratio": round(width / image.width, 4),
                    "height_ratio": round(height / image.height, 4),
                    "area_ratio": round((width * height) / (image.width * image.height), 4),
                    "top_ratio": round(y1 / image.height, 4),
                    "bottom_ratio": round(y2 / image.height, 4),
                },
            }
        )
    boxes.sort(key=lambda item: float(item["score"]), reverse=True)
    return boxes


def _blur_metadata(image: Image.Image) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np

        gray = np.asarray(image.convert("L"))
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return {
            "method": "opencv_laplacian_variance",
            "score": round(laplacian_variance, 3),
            "available": True,
        }
    except Exception as exc:
        return {
            "method": "opencv_laplacian_variance",
            "score": None,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _decide(
    *,
    boxes: list[dict[str, Any]],
    image_size: tuple[int, int],
    blur: dict[str, Any],
    multi_person_score_threshold: float,
    min_height_ratio: float,
    min_area_ratio: float,
    min_bottom_ratio: float,
    blur_threshold: float,
) -> dict[str, Any]:
    person_boxes = [
        box for box in boxes if str(box["label"]).strip().lower() in {"person", "human"}
    ]
    strong_people = [
        box for box in person_boxes if float(box["score"]) >= multi_person_score_threshold
    ]
    primary = person_boxes[0] if person_boxes else None
    checks: dict[str, Any] = {
        "person_present": primary is not None,
        "single_dominant_person": len(strong_people) <= 1,
        "main_subject": False,
        "body_coverage": False,
        "not_blurry": True,
    }
    reasons: list[str] = []

    if primary is None:
        reasons.append("No person detection passed the score threshold.")
    else:
        metrics = primary["metrics"]
        height_ratio = float(metrics["height_ratio"])
        area_ratio = float(metrics["area_ratio"])
        bottom_ratio = float(metrics["bottom_ratio"])
        checks["main_subject"] = area_ratio >= min_area_ratio
        checks["body_coverage"] = (
            height_ratio >= min_height_ratio and bottom_ratio >= min_bottom_ratio
        )
        if not checks["main_subject"]:
            reasons.append(
                f"Person area ratio {area_ratio:.3f} is below {min_area_ratio:.3f}."
            )
        if not checks["body_coverage"]:
            reasons.append(
                "Person box does not cover enough body height/lower-body region "
                f"(height={height_ratio:.3f}, bottom={bottom_ratio:.3f})."
            )

    if len(strong_people) > 1:
        reasons.append(f"Detected {len(strong_people)} strong person boxes.")

    blur_score = blur.get("score")
    if blur_score is not None and float(blur_score) < blur_threshold:
        checks["not_blurry"] = False
        reasons.append(f"Blur score {float(blur_score):.1f} is below {blur_threshold:.1f}.")

    accepted = all(bool(value) for value in checks.values())
    return {
        "accepted": accepted,
        "message": "accepted" if accepted else "rejected",
        "reasons": reasons,
        "checks": checks,
        "primary_person": primary,
        "person_count": len(person_boxes),
        "strong_person_count": len(strong_people),
        "image_size": {"width": image_size[0], "height": image_size[1]},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")

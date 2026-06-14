from __future__ import annotations

ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

NORMALIZED_LONG_EDGE_PX = 1328
# Aspect-ratio floor: after scaling the long edge to 1328, reject images whose short edge falls
# below this (i.e. too extreme/panoramic to be a usable portrait). A true 2:3 input lands at 885.
MIN_NORMALIZED_EDGE_PX = 700
# Fixed try-on model input. Both dims are multiples of 16 as the Qwen/VAE pipeline requires
# (880 = 55x16, 1328 = 83x16). Validation center-crops to this aspect and outputs EXACTLY
# 880x1328, so every /v1/tryon request hits the same compiled shape (one warmup, no per-request
# recompile) and the user image is passed to the model unresized. Frontend already sends ~2:3,
# so the crop is a near no-op in practice.
NORMALIZED_TARGET_WIDTH = 880
NORMALIZED_TARGET_HEIGHT = 1328
JPEG_QUALITY = 95
AZURE_UPLOAD_TIMEOUT_SECONDS = 60
STORAGE_PREFIX = "inputs"

PERSON_DETECTION_MODEL_ID = "PekingU/rtdetr_r50vd_coco_o365"

PERSON_DETECTION_SCORE_THRESHOLD = 0.10
PRIMARY_PERSON_SCORE_THRESHOLD = 0.50
PRIMARY_PERSON_MIN_HEIGHT_RATIO = 0.35
PRIMARY_PERSON_MIN_AREA_RATIO = 0.12
PRIMARY_PERSON_MIN_BOTTOM_RATIO = 0.40
BLUR_SCORE_THRESHOLD = 45.0

SECONDARY_LARGE_PERSON_SCORE_THRESHOLD = 0.70
SECONDARY_LARGE_PERSON_AREA_RATIO = 0.20

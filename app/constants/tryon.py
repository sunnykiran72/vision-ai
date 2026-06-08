DEFAULT_SEED = 43
DEFAULT_STEPS = 12
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_LORA_RANK = 64
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_SCALE = 1.0

GARMENT_REFERENCE_MAX_EDGE_PX = 768
JPEG_QUALITY = 95
DOWNLOAD_MAX_WORKERS = 8

STORAGE_PREFIX = "wardrobe_output/tryon"

SINGLE_REFERENCE_PROMPT = "Apply the reference garment from image 2 to the person in image 1."
MULTI_REFERENCE_PROMPT = "Apply the reference garments from image 2 to the person in image 1."
IDENTITY_CLAUSE = "Preserve the person's face, identity, body proportions, pose, and background."

PROMPT_TRIGGER_TOP = "Apply GlamifyTopTryon on this person"
PROMPT_TRIGGER_BOTTOM = "Apply GlamifyBottomTryon on this person"
PROMPT_TRIGGER_DRESS = "Apply GlamifyDressTryon on this person"
PROMPT_TRIGGER_MULTI = "Apply GlamifyMultiTryon on this person"

TOP_SECTION_TEMPLATE = "Top: {prompt}."
BOTTOM_SECTION_TEMPLATE = "Bottom: {prompt}."
DRESS_SECTION_TEMPLATE = "Dress: {prompt}."
OUTER_SECTION_TEMPLATE = "Outer: {prompt}."
GENERIC_SECTION_TEMPLATE = "{label}: {prompt}."

DEFAULT_SEED = 7777
DEFAULT_STEPS = 12
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_SCALE = 1.0

GARMENT_REFERENCE_MAX_EDGE_PX = 768
JPEG_QUALITY = 95
DOWNLOAD_MAX_WORKERS = 8
UPSCALE_AFTER_QWEN = True
# Upscale DIRECTLY to the final size (no supersample-then-downscale). Generating 2730 then
# downscaling to 2048 wasted ~2.4s on pixels we throw away; direct 2048 is ~2.6s (SeedVR2 eager).
UPSCALE_TARGET_LONG_EDGE_PX = 2048
FINAL_OUTPUT_LONG_EDGE_PX = 2048

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

# Final try-on generation prompts. The LoRA trigger token (GlamifyTopTryon, etc.) is baked into
# each LoRA at training time and MUST stay exact. `{garment}` is the single product description;
# `{garment_list}` is the dynamically-joined multi product list (e.g. "Top: ... and Bottom: ...").
# Roles: image 1 = the person, image 2 = the garment reference (collage for multi).
TRYON_PROMPT_TEMPLATE_BY_TYPE = {
    "top": (
        "Apply GlamifyTopTryon on this person. Replace the entire top garment on the person in "
        "image 1 with the {garment} from image 2. Remove any outer layer or jacket completely if "
        "present. Strictly preserve the person's face, identity, hair, skin tone, body shape, "
        "body size, body proportions, hands, pose and the background exactly; change only the top "
        "garment, fitting it naturally to the body with realistic drape."
    ),
    "bottom": (
        "Apply GlamifyBottomTryon on this person. Replace the entire bottom garment on the person "
        "in image 1 with the {garment} from image 2. Strictly preserve the person's face, "
        "identity, hair, skin tone, body shape, body size, body proportions, hands, pose and the "
        "background exactly; change only the bottom garment, fitting it naturally to the body with "
        "realistic drape."
    ),
    "dress": (
        "Apply GlamifyDressTryon on this person. Replace the person's entire outfit in image 1 "
        "with the {garment} from image 2. Remove any outer layer or jacket completely if present. "
        "Strictly preserve the person's face, identity, hair, skin tone, body shape, body size, "
        "body proportions, hands, pose and the background exactly; replace the full outfit with "
        "the dress from image 2, fitting it naturally to the body with realistic drape."
    ),
    "multi": (
        "Apply GlamifyMultiTryon on this person. Replace the person's outfit in image 1 with the "
        "{garment_list} from image 2. Remove any outer layer or jacket completely if present. "
        "Strictly preserve the person's face, identity, hair, skin tone, body shape, body size, "
        "body proportions, hands, pose and the background exactly; change only the specified "
        "garments, fitting them naturally to the body with realistic drape."
    ),
}

# Section labels used when building the multi garment list.
TRYON_GARMENT_LABEL_BY_TYPE = {
    "top": "Top",
    "bottom": "Bottom",
    "dress": "Dress",
    "outer": "Outer",
}

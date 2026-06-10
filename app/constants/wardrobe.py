from __future__ import annotations

from dataclasses import dataclass

ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png"}
MIN_IMAGE_EDGE_PX = 350
PREPROCESS_MAX_EDGE_PX = 1024

FASHION_DETECTION_MODEL_ID = "yainage90/fashion-object-detection"
FASHION_DETECTION_THRESHOLD = 0.25

MARQO_MODEL_ID = "Marqo/marqo-fashionSigLIP"
MARQO_CONFIDENCE_THRESHOLD = 0.20
MARQO_TOP_K = 5

LORA_RANK = 16
LORA_ALPHA = 16
QUEUE_WAIT_TIMEOUT_SECONDS = 30
GLAMIFY_API_TIMEOUT_SECONDS = 20
AZURE_UPLOAD_TIMEOUT_SECONDS = 60

# MiniCPM-V garment captioner, loaded in-process via vLLM. These mirror the validated reference
# config and coexist on one GPU beside the resident Qwen model. Memory budgeting is environment
# specific and lives in Settings as MINICPM_GPU_MEMORY_UTILIZATION.
MINICPM_MAX_TOKENS = 90
MINICPM_MAX_SLICE_NUMS = 4
MINICPM_MAX_MODEL_LEN = 4096
MINICPM_TEMPERATURE = 0.0
MINICPM_DTYPE = "bfloat16"
MINICPM_RESIZE_LONG_PX = 1024

OUTPUT_WIDTH = 832
OUTPUT_HEIGHT = 1248
GENERATION_SEED = 7777
GENERATION_STEPS = 12
GENERATION_GUIDANCE_SCALE = 1.0
GENERATION_GUIDANCE_RESCALE = 0.0
GENERATION_NETWORK_MULTIPLIER = 1.0
GENERATION_SAMPLER = "flowmatch"
GENERATION_DO_CFG_NORM = False

# Diffusers (QwenImageEditPlusPipeline) inference. The wardrobe path mirrors the
# standalone diffusers tester exactly: a single resident base model with the
# per-category extraction LoRA swapped in. `true_cfg_scale` is the pipeline's real
# classifier-free-guidance scale (1.0 disables negative-prompt guidance, matching
# the tester). LoRA scale reuses GENERATION_NETWORK_MULTIPLIER above.
GENERATION_TRUE_CFG_SCALE = 1.0


# --- Wardrobe static prompts -------------------------------------------------
# MiniCPM garment-description prompts. MiniCPM returns one factual caption describing only
# the requested garment. The caption is embedded into the Qwen extraction template below and
# is also sent to the Glamify backend as `promptDescription`.
MINICPM_PROMPT_BY_TYPE = {
    "top": """
You are a fashion product specialist writing a precise prompt used to regenerate this
garment as an image. Describe ONLY the top garment. Completely ignore the person (face, hair,
skin, body, midriff, pose), the lower-body garment, footwear, accessories, and the background.
Describe the garment in its own natural form, not how it is styled, tied or cinched.

Write ONE flowing prompt of ~15-55 words using concrete, factual terms only. Every attribute
you mention must have a concrete value; never output an attribute word on its own. Do not repeat
the same detail. No filler or marketing words, no hedging, no labels, lists or JSON.

Cover the following ONLY WHEN clearly visible; skip anything not present or not visible, and
never guess hidden parts: garment type/subtype; neckline and collar style; sleeve or strap
style, length and cuffs; shoulder style; closure type, count and placement; fit and silhouette;
length and hem shape/finish; fabric, material and texture; colour(s) and where each sits; print
or pattern and its placement; structural details; pockets; and trims, embellishments and
hardware. In addition, call out at least two of the top's most distinctive visible features —
any defining detail the list above does not already cover, so its unique look is never missed.
Still never guess hidden or unclear parts.""",

    "bottom": """
You are a fashion product specialist writing a precise prompt used to regenerate this
garment as an image. Describe ONLY the bottom garment. Completely ignore the person (face, hair,
skin, body, waist, midriff, legs, pose), the upper-body garment, footwear, accessories, and the
background. Describe the garment in its own natural form, not how it is styled, tucked, rolled
or cuffed.

Write ONE flowing prompt of ~15-55 words using concrete, factual terms only. Every attribute
you mention must have a concrete value; never output an attribute word on its own. Do not repeat
the same detail. No filler or marketing words, no hedging, no labels, lists or JSON.

Cover the following ONLY WHEN clearly visible; skip anything not present or not visible, and
never guess hidden parts: garment type/subtype; waistband style and rise; closure type, count
and placement; fit and silhouette; leg or skirt opening, length and hem shape/finish; fabric,
material and texture; colour(s) and where each sits; print or pattern and its placement;
structural details (pleats, darts, seams, panels, belt loops); pockets; and trims, embellishments
and hardware. In addition, call out at least two of the bottom's most distinctive visible
features — any defining detail the list above does not already cover, so its unique look is
never missed. Still never guess hidden or unclear parts.""",

    "dress": """
You are a fashion product specialist writing a precise prompt used to regenerate this
garment as an image. Describe ONLY the dress. Completely ignore the person (face, hair, skin,
body, pose), footwear, accessories (including any belt or sash worn over it), any jacket or outer
layer worn over it, and the background. Describe the garment in its own natural form, not how it
is styled, tied or cinched.

Write ONE flowing prompt of ~15-55 words using concrete, factual terms only. Every attribute
you mention must have a concrete value; never output an attribute word on its own. Do not repeat
the same detail. No filler or marketing words, no hedging, no labels, lists or JSON.

Cover the following ONLY WHEN clearly visible; skip anything not present or not visible, and
never guess hidden parts: garment type/subtype; neckline and collar style; sleeve or strap style,
length and cuffs; shoulder style; bodice and waist construction; closure type, count and
placement; fit and silhouette; skirt style, length and hem shape/finish; fabric, material and
texture; colour(s) and where each sits; print or pattern and its placement; structural details;
pockets; and trims, embellishments and hardware. In addition, call out at least two of the
dress's most distinctive visible features — any defining detail the list above does not already
cover, so its unique look is never missed. Still never guess hidden or unclear parts.""",

}

# Qwen extraction prompt templates. `{caption}` is replaced with the MiniCPM caption. Only the
# leading trigger sentence differs per garment type; the rest is identical.
QWEN_EXTRACT_PROMPT_TEMPLATE_BY_TYPE = {
    "top": (
        "GlamTopExt. Extract top wear as a standalone product. "
        "Target regenerate garment is {caption} ; "
        "Keep the garment's exact shape, fabric texture, color and print. "
        "remove the person, other clothing, background and shadows. "
        "fill skin-revealing gaps with clean white. "
        "Present it as a centered product on a pure white background "
        "with sharp and precise details of original garment."
    ),
    "bottom": (
        "GlamBtmExt. Extract bottom wear as a standalone product. "
        "Target regenerate garment is {caption} ; "
        "Keep the garment's exact shape, fabric texture, color and print. "
        "remove the person, other clothing, background and shadows. "
        "fill skin-revealing gaps with clean white. "
        "Present it as a centered product on a pure white background "
        "with sharp and precise details of original garment."
    ),
    "dress": (
        "GlamDressExt. Extract dress as a standalone product. "
        "Target regenerate garment is {caption} ; "
        "Keep the garment's exact shape, fabric texture, color and print. "
        "remove the person, other clothing, background and shadows. "
        "fill skin-revealing gaps with clean white. "
        "Present it as a centered product on a pure white background "
        "with sharp and precise details of original garment."
    ),
}


@dataclass(frozen=True)
class MarqoCategoryCandidate:
    key: str
    label: str
    parent_key: str
    parent_label: str


MARQO_CANDIDATES_BY_TYPE: dict[str, tuple[MarqoCategoryCandidate, ...]] = {
    "top": (
        MarqoCategoryCandidate("t_shirts", "T-Shirts", "tops", "Tops"),
        MarqoCategoryCandidate("long_sleeve_t_shirts", "Long Sleeve T-Shirts", "tops", "Tops"),
        MarqoCategoryCandidate("polo_shirts", "Polo Shirts", "tops", "Tops"),
        MarqoCategoryCandidate("crop_tops", "Crop Tops", "tops", "Tops"),
        MarqoCategoryCandidate("blouses", "Blouses", "tops", "Tops"),
        MarqoCategoryCandidate("shirts", "Shirts", "tops", "Tops"),
        MarqoCategoryCandidate("sweatshirts", "Sweatshirts", "tops", "Tops"),
        MarqoCategoryCandidate("hoodies", "Hoodies", "tops", "Tops"),
        MarqoCategoryCandidate("sweaters", "Sweaters", "tops", "Tops"),
        MarqoCategoryCandidate("sweater_vests", "Sweater Vests", "tops", "Tops"),
        MarqoCategoryCandidate("bodysuits", "Bodysuits", "tops", "Tops"),
        MarqoCategoryCandidate("knitwear", "Knitwear", "tops", "Tops"),
        MarqoCategoryCandidate("corsets", "Corsets", "tops", "Tops"),
        MarqoCategoryCandidate("tunics", "Tunics", "tops", "Tops"),
        MarqoCategoryCandidate("bustiers", "Bustiers", "tops", "Tops"),
        MarqoCategoryCandidate("sleeveless_tops", "Sleeveless Tops", "tops", "Tops"),
        MarqoCategoryCandidate("tank_tops_and_camis", "Tank Tops & Camis", "tops", "Tops"),
        MarqoCategoryCandidate("coats", "Coats", "outerwear", "Outerwear"),
        MarqoCategoryCandidate("trench_coats", "Trench Coats", "outerwear", "Outerwear"),
        MarqoCategoryCandidate("blazers", "Blazers", "outerwear", "Outerwear"),
        MarqoCategoryCandidate("jackets", "Jackets", "outerwear", "Outerwear"),
        MarqoCategoryCandidate("varsity_jackets", "Varsity Jackets", "outerwear", "Outerwear"),
        MarqoCategoryCandidate("biker_jackets", "Biker Jackets", "outerwear", "Outerwear"),
        MarqoCategoryCandidate("cardigans", "Cardigans", "outerwear", "Outerwear"),
        MarqoCategoryCandidate("parkas", "Parkas", "outerwear", "Outerwear"),
        MarqoCategoryCandidate("down_jackets", "Down Jackets", "outerwear", "Outerwear"),
        MarqoCategoryCandidate("puffer_jackets", "Puffer Jackets", "outerwear", "Outerwear"),
        MarqoCategoryCandidate("capes", "Capes", "outerwear", "Outerwear"),
        MarqoCategoryCandidate("ponchos", "Ponchos", "outerwear", "Outerwear"),
        MarqoCategoryCandidate("leather_jackets", "Leather Jackets", "outerwear", "Outerwear"),
        MarqoCategoryCandidate("bomber_jackets", "Bomber Jackets", "outerwear", "Outerwear"),
        MarqoCategoryCandidate("denim_jackets", "Denim Jackets", "outerwear", "Outerwear"),
        MarqoCategoryCandidate("windbreakers", "Windbreakers", "outerwear", "Outerwear"),
        MarqoCategoryCandidate("vests", "Vests", "layering_pieces", "Layering Pieces"),
        MarqoCategoryCandidate("shawls", "Shawls", "layering_pieces", "Layering Pieces"),
        MarqoCategoryCandidate("shrugs", "Shrugs", "layering_pieces", "Layering Pieces"),
        MarqoCategoryCandidate("boleros", "Boleros", "layering_pieces", "Layering Pieces"),
        MarqoCategoryCandidate("suits", "Suits", "office_wear_formal", "Office Wear / Formal"),
        MarqoCategoryCandidate(
            "knit_tops",
            "Knit Tops",
            "office_wear_formal",
            "Office Wear / Formal",
        ),
        MarqoCategoryCandidate(
            "work_dresses",
            "Work Dresses",
            "office_wear_formal",
            "Office Wear / Formal",
        ),
        MarqoCategoryCandidate(
            "structured_dresses",
            "Structured Dresses",
            "office_wear_formal",
            "Office Wear / Formal",
        ),
    ),
    "bottom": (
        MarqoCategoryCandidate("pants", "Pants", "bottoms", "Bottoms"),
        MarqoCategoryCandidate("trousers", "Trousers", "bottoms", "Bottoms"),
        MarqoCategoryCandidate("dress_pants", "Dress Pants", "bottoms", "Bottoms"),
        MarqoCategoryCandidate("track_pants", "Track Pants", "bottoms", "Bottoms"),
        MarqoCategoryCandidate("leggings", "Leggings", "bottoms", "Bottoms"),
        MarqoCategoryCandidate("sweatpants", "Sweatpants", "bottoms", "Bottoms"),
        MarqoCategoryCandidate("shorts", "Shorts", "bottoms", "Bottoms"),
        MarqoCategoryCandidate("jeans", "Jeans", "bottoms", "Bottoms"),
        MarqoCategoryCandidate("palazzos", "Palazzos", "bottoms", "Bottoms"),
        MarqoCategoryCandidate("jeggings", "Jeggings", "bottoms", "Bottoms"),
        MarqoCategoryCandidate("skorts", "Skorts", "bottoms", "Bottoms"),
        MarqoCategoryCandidate("cargo_pants", "Cargo Pants", "bottoms", "Bottoms"),
        MarqoCategoryCandidate("wide_leg_pants", "Wide-Leg Pants", "bottoms", "Bottoms"),
        MarqoCategoryCandidate("mini_skirts", "Mini Skirts", "skirts", "Skirts"),
        MarqoCategoryCandidate("midi_skirts", "Midi Skirts", "skirts", "Skirts"),
        MarqoCategoryCandidate("maxi_skirts", "Maxi Skirts", "skirts", "Skirts"),
        MarqoCategoryCandidate("a_line_skirts", "A-Line Skirts", "skirts", "Skirts"),
        MarqoCategoryCandidate("pencil_skirts", "Pencil Skirts", "skirts", "Skirts"),
        MarqoCategoryCandidate("pleated_skirts", "Pleated Skirts", "skirts", "Skirts"),
        MarqoCategoryCandidate("wrap_skirts", "Wrap Skirts", "skirts", "Skirts"),
        MarqoCategoryCandidate("denim_skirts", "Denim Skirts", "skirts", "Skirts"),
        MarqoCategoryCandidate(
            "sports_tops",
            "Sports Tops",
            "activewear_sportswear",
            "Activewear / Sportswear",
        ),
        MarqoCategoryCandidate(
            "sports_jackets",
            "Sports Jackets",
            "activewear_sportswear",
            "Activewear / Sportswear",
        ),
        MarqoCategoryCandidate(
            "gym_leggings",
            "Gym Leggings",
            "activewear_sportswear",
            "Activewear / Sportswear",
        ),
        MarqoCategoryCandidate(
            "bike_shorts",
            "Bike Shorts",
            "activewear_sportswear",
            "Activewear / Sportswear",
        ),
        MarqoCategoryCandidate(
            "tracksuits",
            "Tracksuits",
            "activewear_sportswear",
            "Activewear / Sportswear",
        ),
        MarqoCategoryCandidate(
            "tennis_skirts",
            "Tennis Skirts",
            "activewear_sportswear",
            "Activewear / Sportswear",
        ),
    ),
    "dress": (
        MarqoCategoryCandidate("day_dresses", "Day Dresses", "dresses", "Dresses"),
        MarqoCategoryCandidate("t_shirt_dresses", "T-Shirt Dresses", "dresses", "Dresses"),
        MarqoCategoryCandidate("shirt_dresses", "Shirt Dresses", "dresses", "Dresses"),
        MarqoCategoryCandidate("sweater_dresses", "Sweater Dresses", "dresses", "Dresses"),
        MarqoCategoryCandidate("jacket_dresses", "Jacket Dresses", "dresses", "Dresses"),
        MarqoCategoryCandidate("party_dresses", "Party Dresses", "dresses", "Dresses"),
        MarqoCategoryCandidate("mini_dresses", "Mini Dresses", "dresses", "Dresses"),
        MarqoCategoryCandidate("maxi_dresses", "Maxi Dresses", "dresses", "Dresses"),
        MarqoCategoryCandidate("slip_dresses", "Slip Dresses", "dresses", "Dresses"),
        MarqoCategoryCandidate("bodycon_dresses", "Bodycon Dresses", "dresses", "Dresses"),
        MarqoCategoryCandidate("casual_dresses", "Casual Dresses", "dresses", "Dresses"),
        MarqoCategoryCandidate("evening_dresses", "Evening Dresses", "dresses", "Dresses"),
        MarqoCategoryCandidate("midi_dresses", "Midi Dresses", "dresses", "Dresses"),
        MarqoCategoryCandidate("strapless_dresses", "Strapless Dresses", "dresses", "Dresses"),
        MarqoCategoryCandidate(
            "off_shoulder_dresses",
            "Off-Shoulder Dresses",
            "dresses",
            "Dresses",
        ),
        MarqoCategoryCandidate("wrap_dresses", "Wrap Dresses", "dresses", "Dresses"),
        MarqoCategoryCandidate("lounge_sets", "Lounge Sets", "loungewear", "Loungewear"),
        MarqoCategoryCandidate("lounge_pants", "Lounge Pants", "loungewear", "Loungewear"),
        MarqoCategoryCandidate("lounge_tops", "Lounge Tops", "loungewear", "Loungewear"),
        MarqoCategoryCandidate(
            "oversized_hoodies",
            "Oversized Hoodies",
            "loungewear",
            "Loungewear",
        ),
        MarqoCategoryCandidate("soft_knit_sets", "Soft Knit Sets", "loungewear", "Loungewear"),
        MarqoCategoryCandidate("jumpsuits", "Jumpsuits", "sets_one_pieces", "Sets & One-Pieces"),
        MarqoCategoryCandidate("rompers", "Rompers", "sets_one_pieces", "Sets & One-Pieces"),
        MarqoCategoryCandidate("playsuits", "Playsuits", "sets_one_pieces", "Sets & One-Pieces"),
        MarqoCategoryCandidate(
            "two_piece_sets",
            "Two-Piece Sets",
            "sets_one_pieces",
            "Sets & One-Pieces",
        ),
        MarqoCategoryCandidate(
            "matching_sets",
            "Matching Sets",
            "sets_one_pieces",
            "Sets & One-Pieces",
        ),
        MarqoCategoryCandidate("co_ords", "Co-ords", "sets_one_pieces", "Sets & One-Pieces"),
    ),
}

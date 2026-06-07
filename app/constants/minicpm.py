from __future__ import annotations

MODEL_ID = "openbmb/MiniCPM-V-4.6"
DEFAULT_DEVICE = "auto"
DEFAULT_DTYPE = "auto"
DEFAULT_DOWNSAMPLE_MODE = "16x"
DEFAULT_MAX_SLICE_NUMS = 36
DEFAULT_MAX_NEW_TOKENS = 160
REQUEST_TIMEOUT_SECONDS = 120

PROMPT_BY_TYPE = {
    "top": """Task: Describe only the TOP garment using clear, visible construction facts.

Return format:
- Exactly one line.
- Plain text only, no JSON, bullets, labels, or headings.
- 35 to 75 words.
- Keep the slot order below and include only slots with clear visual evidence.
- Mention only directly visible evidence.

Ordered slots:
1) top category,
2) neckline or upper-edge shape,
3) sleeve or strap configuration,
4) shoulder coverage,
5) torso silhouette/construction,
6) hem shape and endpoint,
7) 1-5 standout visible accurate detail.

Optional slots:
- closure type/placement,
- seam/panel structure,
- asymmetry trait.

Positive-only writing style:
- Use affirmative factual statements.
- Describe what is visibly present and useful for reconstruction.
- Keep wording concise, specific, and non-repetitive.

Hard exclusions:
- No guessing or inferred hidden structure.
- No directional words.
- No non-top details.
- No body/background/camera/styling terms.
- No color details.
- No absence phrasing.""",
    "bottom": """Task: Describe only the BOTTOM garment using clear, visible construction facts.

Return format:
- Exactly one line.
- Plain text only, no JSON, bullets, labels, or headings.
- 35 to 75 words.
- Keep the slot order below and include only slots with clear visual evidence.
- Mention only directly visible evidence.

Ordered slots:
1) bottom category,
2) waistband or upper-edge shape,
3) rise/opening shape,
4) hip or seat shaping,
5) leg or skirt-panel silhouette/construction,
6) hem shape and endpoint,
7) 1-5 standout visible accurate detail.

Optional slots:
- closure type/placement,
- seam/panel structure,
- asymmetry trait.

Positive-only writing style:
- Use affirmative factual statements.
- Describe what is visibly present and useful for reconstruction.
- Keep wording concise, specific, and non-repetitive.

Hard exclusions:
- No guessing or inferred hidden structure.
- No directional words.
- No non-bottom details.
- No body/background/camera/styling terms.
- No color details.
- No absence phrasing.""",
    "dress": """Task: Describe only the DRESS garment using clear, visible construction facts.

Return format:
- Exactly one line.
- Plain text only, no JSON, bullets, labels, or headings.
- 35 to 75 words.
- Keep the slot order below and include only slots with clear visual evidence.
- Mention only directly visible evidence.

Ordered slots:
1) dress category,
2) neckline or upper-edge shape,
3) sleeve or strap configuration,
4) shoulder coverage,
5) bodice and waist construction,
6) skirt silhouette, length, hem shape and endpoint,
7) 1-5 standout visible accurate detail.

Optional slots:
- closure type/placement,
- seam/panel structure,
- asymmetry trait.

Positive-only writing style:
- Use affirmative factual statements.
- Describe what is visibly present and useful for reconstruction.
- Keep wording concise, specific, and non-repetitive.

Hard exclusions:
- No guessing or inferred hidden structure.
- No directional words.
- No non-dress details.
- No body/background/camera/styling terms.
- No color details.
- No absence phrasing.""",
}

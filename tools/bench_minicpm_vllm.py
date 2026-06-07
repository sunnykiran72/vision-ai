from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image


PROMPT_BY_TYPE = {
    "top": (
        "You are a fashion product specialist writing a precise description that will be used to "
        "regenerate this garment as an image. Describe ONLY the upper-body garment (the top / "
        "shirt / jacket) in this image. Completely ignore the person (face, hair, skin, body, "
        "midriff, pose), the lower-body garment (trousers/pants/skirt), footwear, accessories, "
        "and the background - do not mention them at all. Write one clear, flowing paragraph of "
        "about 30-45 words capturing every visible detail needed to recreate it: garment type, "
        "collar/neckline, closure (buttons/zip/ties), sleeves/straps or waistline/legs, fit and "
        "silhouette, fabric and texture, all colours, and the print/pattern (motif types, their "
        "colours, and where they sit). Only what is clearly visible; never guess hidden parts. "
        "Plain factual prose - no labels, no lists, no headings, no preamble."
    ),
    "bottom": (
        "You are a fashion product specialist writing a precise description that will be used to "
        "regenerate this garment as an image. Describe ONLY the lower-body garment (the trousers / "
        "pants / skirt / shorts) in this image. Completely ignore the person, the upper-body "
        "garment (top/shirt), footwear, accessories, and the background - do not mention them at "
        "all. Write one clear, flowing paragraph of about 30-45 words capturing every visible "
        "detail needed to recreate it: garment type, collar/neckline, closure (buttons/zip/ties), "
        "sleeves/straps or waistline/legs, fit and silhouette, fabric and texture, all colours, "
        "and the print/pattern (motif types, their colours, and where they sit). Only what is "
        "clearly visible; never guess hidden parts. Plain factual prose - no labels, no lists, "
        "no headings, no preamble."
    ),
    "dress": (
        "You are a fashion product specialist writing a precise description that will be used to "
        "regenerate this garment as an image. Describe ONLY the dress in this image. Completely "
        "ignore the person (face, hair, skin, body, pose), footwear, accessories, and the "
        "background - do not mention them at all. Write one clear, flowing paragraph of about "
        "30-45 words capturing every visible detail needed to recreate it: garment type, "
        "collar/neckline, closure (buttons/zip/ties), sleeves/straps or waistline/legs, fit and "
        "silhouette, fabric and texture, all colours, and the print/pattern (motif types, their "
        "colours, and where they sit). Only what is clearly visible; never guess hidden parts. "
        "Plain factual prose - no labels, no lists, no headings, no preamble."
    ),
}


def resize_long(image: Image.Image, target: int) -> Image.Image:
    width, height = image.size
    if max(width, height) <= target:
        return image
    if width >= height:
        new_width, new_height = target, max(8, round(height * target / width))
    else:
        new_height, new_width = target, max(8, round(width * target / height))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def strip_reasoning(text: str) -> str:
    raw = text.strip()
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[-1].strip()
    return " ".join(raw.replace("<think>", "").split()).strip()


def infer_type(path: Path, fallback: str) -> str:
    name = path.name.lower()
    for garment_type in ("top", "bottom", "dress"):
        if garment_type in name:
            return garment_type
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--type", default="top", choices=sorted(PROMPT_BY_TYPE))
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-util", type=float, default=0.12)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=90)
    parser.add_argument("--resize-long", type=int, default=1024)
    parser.add_argument("--max-slice-nums", type=int, default=4)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        dtype=args.dtype,
        enforce_eager=True,
        max_num_seqs=1,
        mm_processor_kwargs={"max_slice_nums": args.max_slice_nums},
    )
    try:
        candidate_ids = [
            tokenizer.convert_tokens_to_ids(token) for token in ["<|im_end|>", "<|endoftext|>"]
        ]
        stop_ids = [token_id for token_id in candidate_ids if isinstance(token_id, int) and token_id >= 0]
    except Exception:
        stop_ids = []
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_ids or None,
    )

    for raw_image_path in args.image:
        image_path = Path(raw_image_path)
        garment_type = infer_type(image_path, args.type)
        prompt = PROMPT_BY_TYPE[garment_type]
        image = resize_long(Image.open(image_path).convert("RGB"), args.resize_long)
        messages = [{"role": "user", "content": "(<image>./</image>)\n" + prompt}]
        try:
            chat_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            chat_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        started = time.perf_counter()
        outputs = llm.generate(
            {"prompt": chat_prompt, "multi_modal_data": {"image": image}},
            sampling_params,
            use_tqdm=False,
        )
        latency_ms = round((time.perf_counter() - started) * 1000)
        caption = strip_reasoning(str(outputs[0].outputs[0].text))
        print(
            json.dumps(
                {
                    "model": args.model,
                    "image": str(image_path),
                    "type": garment_type,
                    "latency_ms": latency_ms,
                    "words": len(caption.split()),
                    "caption": caption,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

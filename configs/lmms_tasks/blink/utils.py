import re
from io import BytesIO

from PIL import Image

TARGET_SIZE = (336, 336)

_LETTERS = "ABCD"


def _load_image(img_dict):
    """Load a PIL Image from HuggingFace parquet image dict {bytes, path}."""
    if img_dict is None:
        return None
    if isinstance(img_dict, dict) and "bytes" in img_dict:
        return Image.open(BytesIO(img_dict["bytes"])).convert("RGB").resize(TARGET_SIZE)
    # Already a PIL Image (shouldn't happen, but be safe)
    if hasattr(img_dict, "convert"):
        return img_dict.convert("RGB").resize(TARGET_SIZE)
    return None


def blink_doc_to_visual(doc):
    """Return list of all present images (1–4). LlavaHf.flatten handles multi-image."""
    imgs = []
    for slot in ("image_1", "image_2", "image_3", "image_4"):
        img = _load_image(doc.get(slot))
        if img is not None:
            imgs.append(img)
    return imgs if imgs else [Image.new("RGB", TARGET_SIZE)]


def blink_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt", "\nAnswer with the option's letter from the given choices directly."
    )

    prompt_ctx = str(doc.get("prompt", "")).strip()
    question = str(doc.get("question", "")).strip()
    choices = doc.get("choices", [])

    parts = []
    if prompt_ctx:
        parts.append(prompt_ctx)
    if question:
        parts.append(question)

    options = []
    for i, c in enumerate(choices):
        if i < len(_LETTERS):
            options.append(f"({_LETTERS[i]}) {c}")
    if options:
        parts.append("\n".join(options))

    return f"{pre_prompt}{'  '.join(parts)}{post_prompt}"


def blink_process_results(doc, results):
    pred = results[0].strip()
    # Accept "(A)", "A", "A.", "A)" etc.
    m = re.search(r"\b([A-D])\b", pred.upper())
    pred_letter = m.group(1) if m else (pred[0].upper() if pred else "")

    gt = str(doc.get("answer", "")).strip()
    gt_m = re.search(r"([A-D])", gt.upper())
    gt_letter = gt_m.group(1) if gt_m else gt.upper()

    return {"accuracy": 1.0 if pred_letter == gt_letter else 0.0}

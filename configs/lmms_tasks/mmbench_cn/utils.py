from io import BytesIO

from PIL import Image

TARGET_SIZE = (336, 336)


def _load_image(img_field):
    if img_field is None:
        return None
    if isinstance(img_field, dict) and "bytes" in img_field:
        return Image.open(BytesIO(img_field["bytes"])).convert("RGB").resize(TARGET_SIZE)
    if hasattr(img_field, "convert"):
        return img_field.convert("RGB").resize(TARGET_SIZE)
    return None


def mmbench_cn_doc_to_visual(doc):
    img = _load_image(doc.get("image"))
    return [img] if img else []


def mmbench_cn_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc["question"]
    hint = doc.get("hint", "")
    if hint and str(hint).lower() not in ("nan", "none", ""):
        question = f"{hint}\n{question}"
    choices = []
    for letter in "ABCD":
        val = doc.get(letter, "")
        if val and str(val).lower() not in ("nan", "none", ""):
            choices.append(f"{letter}. {val}")
    options = "\n".join(choices)
    return f"{pre_prompt}{question}\n{options}{post_prompt}"


def mmbench_cn_process_results(doc, results):
    pred = results[0].strip()
    if len(pred) > 1:
        pred = pred[0]
    gt = str(doc["answer"]).strip().upper()
    return {"accuracy": 1.0 if pred.upper() == gt else 0.0}

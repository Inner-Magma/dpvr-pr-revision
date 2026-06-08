LETTERS = "ABCDEFGHIJ"
TARGET_SIZE = (336, 336)  # LLaVA-1.5 fixed vision encoder resolution


def scienceqa_doc_to_visual(doc):
    raw = doc.get("image")
    if raw is None:
        from PIL import Image as _PIL
        return [_PIL.new("RGB", TARGET_SIZE)]
    return [raw.convert("RGB").resize(TARGET_SIZE)]


def scienceqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    choices = doc["choices"]
    question = doc["question"]
    options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"{pre_prompt}{question}\n{options}{post_prompt}"


def scienceqa_doc_to_target(doc):
    return LETTERS[int(doc["answer"])]


def scienceqa_process_results(doc, results):
    pred = results[0].strip()
    if len(pred) > 1:
        pred = pred[0]
    gt = LETTERS[int(doc["answer"])]
    return {"mc_accuracy": 1.0 if pred.upper() == gt else 0.0}

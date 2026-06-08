import os

from datasets import load_dataset

GQA_RAW_IMAGE_DATASET = None
GQA_ID2IMAGE = None

_GQA_DEFAULT_PATH = "lmms-lab/GQA"


def gqa_doc_to_visual(doc):
    global GQA_RAW_IMAGE_DATASET
    global GQA_ID2IMAGE
    if GQA_RAW_IMAGE_DATASET is None:
        gqa_path = os.environ.get("GQA_IMAGE_DATASET_PATH", _GQA_DEFAULT_PATH)
        GQA_RAW_IMAGE_DATASET = load_dataset(
            gqa_path,
            "testdev_all_images",
            split="testdev",
        )
        GQA_ID2IMAGE = {}
        for row in GQA_RAW_IMAGE_DATASET:
            GQA_ID2IMAGE[row["id"]] = row["image"].convert("RGB")
    image = GQA_ID2IMAGE[doc["imageId"]]
    return [image]


def gqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{question}{post_prompt}"


def gqa_process_results(doc, results):
    pred = results[0].strip().rstrip(".").lower()
    gt_ans = doc["answer"].lower()
    score = 1.0 if pred == gt_ans else 0.0
    return {"exact_match": score}

import json


def seed_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    images = doc["image"]
    # SEED-Bench video samples expose multiple frames; LLaVA-1.5 handles only one
    # image per sample.  Always return the first frame to avoid the
    # "image features and image tokens do not match" crash.
    if isinstance(images, (list, tuple)):
        img = images[0]
    else:
        img = images
    img = img.convert("RGB")
    # Cap resolution to 1024px on the long edge before handing off to the
    # image processor.  The processor will resize to 336×336 regardless; this
    # pre-downscale avoids spending CPU time on 4K→336 resampling.
    max_side = max(img.size)
    if max_side > 1024:
        scale = 1024.0 / max_side
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        img = img.resize((new_w, new_h))
    return [img]


def seed_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    question += "\n" + f"A. {doc['choice_a']}\n"
    question += f"B. {doc['choice_b']}\n"
    question += f"C. {doc['choice_c']}\n"
    question += f"D. {doc['choice_d']}"
    return f"{question}\nAnswer with the option's letter from the given choices directly."


def seed_process_result(doc, result):
    pred = result[0].strip()
    if len(pred) > 1:
        pred = pred[0]
    answer = doc["answer"]
    data_type = doc["data_type"]

    return {f"seed_{data_type}": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}, f"seed_all": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}}


def seed_aggregation_result(results):
    total_count = 0
    total_correct = 0
    for result in results:
        if result["pred"].lower().strip() == result["answer"].lower().strip():
            total_correct += 1
        total_count += 1
    return total_correct / total_count


def seed_aggregation_result_all(results):
    score = seed_aggregation_result(results)
    stored_results = []
    for result in results:
        stored_results.append({"question_id": result["question_id"], "prediction": result["pred"]})
    with open("./seed_submission.json", "w") as f:
        json.dump(stored_results, f, indent=4)
    print("Storing files for seed_submission ...")

    return score


def seed_doc_to_text_mc(doc):
    question = doc["question"]
    return f"{question} Answer :"


def seed_doc_to_choice(doc):
    return [doc["choice_a"], doc["choice_b"], doc["choice_c"], doc["choice_d"]]


def seed_doc_to_mc_target(doc):
    answer2choice = {"A": "choice_a", "B": "choice_b", "C": "choice_c", "D": "choice_d"}
    return doc[answer2choice[doc["answer"]]]

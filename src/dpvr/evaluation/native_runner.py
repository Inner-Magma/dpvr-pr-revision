"""Native evaluation runner — no lmms-eval dependency.

Supports yes/no benchmarks (MME, POPE) and open-ended VQA (GQA, TextVQA,
VQAv2, SEED-Bench, ScienceQA). Loads a TokenDiversionLlava checkpoint and
runs model.generate() directly, so it works on air-gapped machines.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return s.strip().lower()


def _yn_accuracy(predictions: list[str], targets: list[str]) -> float:
    """Accuracy for yes/no questions (case-insensitive prefix match)."""
    correct = 0
    for pred, gt in zip(predictions, targets):
        p = _norm(pred)
        g = _norm(gt)
        if p.startswith("yes") and g.startswith("yes"):
            correct += 1
        elif p.startswith("no") and g.startswith("no"):
            correct += 1
    return correct / max(len(targets), 1)


def _vqa_accuracy(predictions: list[str], targets: list[str | list[str]]) -> float:
    """Soft VQA accuracy: min(count_matching_annotations/3, 1)."""
    scores = []
    for pred, gt in zip(predictions, targets):
        p = _norm(pred)
        if isinstance(gt, list):
            count = sum(1 for g in gt if _norm(g) in p or p in _norm(g))
            scores.append(min(count / 3.0, 1.0))
        else:
            scores.append(1.0 if _norm(gt) in p or p in _norm(gt) else 0.0)
    return sum(scores) / max(len(scores), 1)


def _mc_accuracy(predictions: list[str], targets: list[str]) -> float:
    """Multiple-choice accuracy: first letter (A/B/C/D) match."""
    correct = 0
    for pred, gt in zip(predictions, targets):
        p = _norm(pred)
        g = _norm(gt)
        # Extract first A/B/C/D from prediction
        m = re.search(r'\b([abcd])\b', p)
        pred_letter = m.group(1) if m else p[:1]
        correct += int(pred_letter == g[:1])
    return correct / max(len(targets), 1)


# ---------------------------------------------------------------------------
# Dataset loaders (local JSON only — no hub calls)
# ---------------------------------------------------------------------------

def _load_local_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_parquet_samples(local_path: Path, task: str) -> list[dict] | None:
    """Try to load samples from lmms-lab parquet files.

    Returns None if no parquet files found or pyarrow not available.
    """
    try:
        import pyarrow.parquet as pq
        from io import BytesIO
        from PIL import Image as PILImage
    except ImportError:
        return None

    # Prefer the canonical 'data/' subdirectory to avoid loading duplicate
    # files when a dataset stores both a split-level dir (Full/) and a data/ dir.
    data_subdir = local_path / "data"
    search_root = data_subdir if data_subdir.is_dir() else local_path
    parquet_files = sorted(search_root.glob("*.parquet"))
    if not parquet_files:
        parquet_files = sorted(local_path.rglob("*.parquet"))
    if not parquet_files:
        return None

    samples = []
    for pf in sorted(parquet_files):
        table = pq.read_table(str(pf))
        cols = table.column_names

        for i in range(len(table)):
            row = {c: table.column(c)[i].as_py() for c in cols}

            # Extract image: lmms-lab stores bytes under 'image' or 'image.bytes'
            image_obj = None
            if "image" in row:
                img_val = row["image"]
                if isinstance(img_val, dict) and "bytes" in img_val:
                    image_obj = PILImage.open(BytesIO(img_val["bytes"])).convert("RGB")
                elif isinstance(img_val, bytes):
                    image_obj = PILImage.open(BytesIO(img_val)).convert("RGB")

            # Map task-specific column names.
            question = row.get("question") or row.get("text") or row.get("conversations", "")
            answer = (row.get("answer") or row.get("label") or
                      row.get("gt_answer") or "")
            choices = row.get("choices") or row.get("options") or []

            if task in ("pope",):
                qtype = "yn"
            elif choices:
                qtype = "mc"
            else:
                qtype = "open"

            samples.append({
                "image_path": None,       # None signals pre-loaded image
                "image_obj": image_obj,   # PIL image (may be None if bytes missing)
                "question": str(question),
                "answer": str(answer),
                "question_type": qtype,
                "choices": list(choices) if choices else [],
            })

    return samples if samples else None


def _iter_benchmark(
    local_path: Path,
    task: str,
    max_samples: int | None = None,
) -> list[dict]:
    """Return list of {image_path, question, answer, question_type} dicts.

    Tries parquet (lmms-lab format) first, then falls back to JSON layout.
    """
    # Try parquet first (lmms-lab datasets).
    parquet_samples = _load_parquet_samples(local_path, task)
    if parquet_samples:
        if max_samples is not None:
            parquet_samples = parquet_samples[:max_samples]
        return parquet_samples

    samples = []

    if task == "pope":
        # POPE JSON format: [{"image": "...", "text": "...", "label": "yes/no"}]
        for sub in ["adversarial", "popular", "random"]:
            p = local_path / f"pope_{sub}.json"
            if p.exists():
                for item in _load_local_json(p):
                    samples.append({
                        "image_path": local_path / "images" / item.get("image", ""),
                        "question": item["text"],
                        "answer": item["label"],
                        "question_type": "yn",
                    })
    elif task == "mme":
        # MME: subfolders per category, each with <img>.txt question files
        for qfile in sorted(local_path.rglob("*.txt")):
            lines = qfile.read_text().strip().splitlines()
            for line in lines:
                parts = line.split("\t")
                if len(parts) >= 3:
                    img_name, question, answer = parts[0], parts[1], parts[2]
                    samples.append({
                        "image_path": qfile.parent / img_name,
                        "question": question,
                        "answer": answer,
                        "question_type": "yn",
                    })
    elif task == "gqa":
        # GQA testdev balanced JSON
        qfile = local_path / "testdev_balanced_questions.json"
        if qfile.exists():
            data = json.loads(qfile.read_text())
            for qid, item in data.items():
                samples.append({
                    "image_path": local_path / "images" / item["imageId"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "question_type": "open",
                })
    elif task in ("seed_bench", "scienceqa"):
        # Generic: JSON list with keys image/question/answer(/choices)
        for p in local_path.glob("*.json"):
            for item in _load_local_json(p):
                qtype = "mc" if "choices" in item else "open"
                samples.append({
                    "image_path": local_path / "images" / item.get("image", ""),
                    "question": item["question"],
                    "answer": item.get("answer", item.get("label", "")),
                    "question_type": qtype,
                    "choices": item.get("choices", []),
                })
    else:
        # Fallback: generic JSON list
        p = local_path / f"{task}.json"
        if not p.exists():
            for candidate in local_path.glob("*.json"):
                p = candidate
                break
        if p.exists():
            for item in _load_local_json(p):
                samples.append({
                    "image_path": local_path / "images" / item.get("image", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", item.get("label", "")),
                    "question_type": "open",
                })

    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _build_prompt(question: str, choices: list[str] | None = None) -> str:
    prompt = f"USER: <image>\n{question}"
    if choices:
        opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        prompt += f"\nOptions:\n{opts}\nAnswer with the option letter."
    prompt += " ASSISTANT:"
    return prompt


@torch.no_grad()
def _greedy_decode_no_cache(model, enc, max_new_tokens: int, device: str, eos_token_id: int | None, pad_token_id: int = 0):
    """Greedy decode using repeated full forward passes (no KV-cache).

    Avoids causal-mask shape errors that arise when generate() tries to expand
    KV-cache entries after image-token expansion mid-sequence.
    """
    input_ids = enc["input_ids"]
    pixel_values = enc.get("pixel_values")
    attention_mask = enc.get("attention_mask")
    B = input_ids.shape[0]
    done = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        attn_mask = (input_ids != pad_token_id).long() if attention_mask is None else attention_mask
        out = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attn_mask,
            use_cache=False,
        )
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones(B, 1, dtype=attention_mask.dtype, device=device)], dim=1
            )
        pixel_values = None  # only pass images on the first step
        done = done | (next_tok.squeeze(-1) == eos_token_id)
        if done.all():
            break

    return [input_ids[j].tolist() for j in range(B)]


@torch.no_grad()
def _yn_logit_compare(model, enc, processor, device: str) -> list[str]:
    """Determine yes/no by comparing logits at the ASSISTANT position.

    For binary yes/no tasks (POPE, MME), this is more reliable than greedy
    decode because the model may prefer descriptive openings ("There is…")
    over "Yes/No" even when "Yes/No" are the semantically correct answers.
    We compare P(Yes) vs P(No) at the next-token position after "ASSISTANT:".
    """
    yes_ids = processor.tokenizer.encode("Yes", add_special_tokens=False)
    no_ids  = processor.tokenizer.encode("No",  add_special_tokens=False)
    yes_id = yes_ids[0] if yes_ids else -1
    no_id  = no_ids[0]  if no_ids  else -1

    out = model(
        input_ids=enc["input_ids"],
        pixel_values=enc.get("pixel_values"),
        attention_mask=enc.get("attention_mask"),
        use_cache=False,
    )
    # logits[:, -1, :] = predicted distribution AFTER the last token ("ASSISTANT:")
    logits_last = out.logits[:, -1, :]  # (B, vocab)

    preds = []
    for b in range(logits_last.shape[0]):
        score_yes = logits_last[b, yes_id].item() if yes_id >= 0 else float("-inf")
        score_no  = logits_last[b, no_id].item()  if no_id  >= 0 else float("-inf")
        preds.append("Yes" if score_yes >= score_no else "No")
    return preds


@torch.no_grad()
def _run_inference(
    model,
    processor,
    samples: list[dict],
    batch_size: int = 1,
    max_new_tokens: int = 32,
    device: str = "cuda",
    yn_logit_mode: bool = False,
    pad_token_id: int = 0,
) -> list[str]:
    from PIL import Image

    model.eval()
    predictions = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        texts, images = [], []
        for s in batch:
            texts.append(_build_prompt(s["question"], s.get("choices")))
            # Prefer pre-loaded PIL image (from parquet), fall back to file path.
            if s.get("image_obj") is not None:
                img = s["image_obj"]
            else:
                try:
                    img = Image.open(s["image_path"]).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (336, 336), (128, 128, 128))
            images.append(img)

        enc = processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        if yn_logit_mode:
            # For yes/no tasks: compare P(Yes) vs P(No) from a single forward pass.
            # More reliable than greedy decode when model prefers descriptive openings.
            preds = _yn_logit_compare(model, enc, processor, device)
            predictions.extend(preds)
        else:
            # Use greedy decode without KV-cache to avoid causal-mask shape
            # mismatches when the model expands image tokens mid-sequence.
            generated_ids = _greedy_decode_no_cache(
                model, enc, max_new_tokens, device, processor.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
            )
            for j in range(len(batch)):
                new_ids = generated_ids[j][enc["input_ids"].shape[1]:]
                predictions.append(
                    processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                )

    return predictions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_native_eval(
    checkpoint_path: str | Path,
    task: str,
    local_data_path: str | Path,
    output_path: str | Path,
    batch_size: int = 1,
    max_samples: int | None = None,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    """Evaluate a TokenDiversionLlava checkpoint on a local benchmark dataset.

    Args:
        checkpoint_path: Path to saved TokenDiversionLlava checkpoint.
        task: Benchmark name (pope, mme, gqa, seed_bench, scienceqa, …).
        local_data_path: Local directory containing benchmark data.
        output_path: Where to write per-sample predictions JSON.
        batch_size: Inference batch size (default 1 for heterogeneous images).
        max_samples: Cap for quick validation runs.
        max_new_tokens: Maximum tokens to generate per sample.
    """
    from transformers import AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(checkpoint_path)

    # Load model: use the registry to pick the correct class for each method,
    # fall back to plain LlavaForConditionalGeneration for vanilla HF model dirs.
    if (checkpoint_path / "training_config.json").exists():
        import json as _json
        from dpvr.models import build_model  # imports all classes, triggering @register_method
        _cfg = _json.loads((checkpoint_path / "training_config.json").read_text())
        model = build_model(_cfg)
        # Load checkpoint weights (sharded safetensors, single safetensors, or pytorch_model.bin).
        _index_file = checkpoint_path / "model.safetensors.index.json"
        _sf_file    = checkpoint_path / "model.safetensors"
        _state_file = checkpoint_path / "pytorch_model.bin"
        if _index_file.exists():
            from safetensors.torch import load_file as _load_file
            _index = _json.loads(_index_file.read_text())
            _state: dict = {}
            for _shard in sorted(set(_index["weight_map"].values())):
                _state.update(_load_file(str(checkpoint_path / _shard)))
        elif _sf_file.exists():
            from safetensors.torch import load_file as _load_file
            _state = _load_file(str(_sf_file))
        elif _state_file.exists():
            _state = torch.load(_state_file, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model weights found in {checkpoint_path}")
        model.load_state_dict(_state, strict=False)
        # Resolve pad_token_id generically across all method model classes.
        _inner = getattr(model, "llava_model", model)
        pad_token_id = getattr(getattr(_inner, "config", None), "pad_token_id", None) or 0
    else:
        from transformers import LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            str(checkpoint_path), torch_dtype=torch.float16
        )
        pad_token_id = model.config.pad_token_id or 0
    model = model.to(device)

    # Processor files may only be in the parent directory when using HF Trainer
    # checkpoint subdirs (checkpoint-N/ only contains model weights).
    proc_path = checkpoint_path
    if not (proc_path / "preprocessor_config.json").exists() and proc_path.parent != proc_path:
        proc_path = proc_path.parent
    processor = AutoProcessor.from_pretrained(str(proc_path))
    # LlavaProcessor.patch_size may be None when loaded from a checkpoint that
    # was saved without the full CLIP config. Inject from model.config if needed.
    if getattr(processor, "patch_size", None) is None:
        import json as _json
        cfg_path = checkpoint_path / "config.json"
        if cfg_path.exists():
            cfg_json = _json.loads(cfg_path.read_text())
            ps = cfg_json.get("vision_config", {}).get("patch_size", 14)
            processor.patch_size = ps
            processor.image_processor.patch_size = ps

    local_data_path = Path(local_data_path)
    samples = _iter_benchmark(local_data_path, task, max_samples=max_samples)
    if not samples:
        print(f"[native_runner] No samples found for task={task} at {local_data_path}")
        return {"error": "no_samples", "n_samples": 0}

    print(f"[native_runner] {task}: {len(samples)} samples, device={device}")
    # For yes/no tasks: compare logit(Yes) vs logit(No) from a single forward.
    # This is more reliable than greedy decode because a partially-trained model
    # may prefer descriptive openings ("There is…") even when Yes/No is correct.
    yn_tasks = {"pope", "mme"}
    yn_logit_mode = task in yn_tasks
    predictions = _run_inference(
        model, processor, samples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
        yn_logit_mode=yn_logit_mode,
        pad_token_id=pad_token_id,
    )

    targets = [s["answer"] for s in samples]
    qtypes = [s.get("question_type", "open") for s in samples]

    # Compute metrics per question type.
    metrics: dict[str, float] = {}
    yn_preds = [p for p, q in zip(predictions, qtypes) if q == "yn"]
    yn_tgts = [t for t, q in zip(targets, qtypes) if q == "yn"]
    if yn_preds:
        metrics["yn_accuracy"] = _yn_accuracy(yn_preds, yn_tgts)

    mc_preds = [p for p, q in zip(predictions, qtypes) if q == "mc"]
    mc_tgts = [t for t, q in zip(targets, qtypes) if q == "mc"]
    if mc_preds:
        metrics["mc_accuracy"] = _mc_accuracy(mc_preds, mc_tgts)

    open_preds = [p for p, q in zip(predictions, qtypes) if q == "open"]
    open_tgts = [t for t, q in zip(targets, qtypes) if q == "open"]
    if open_preds:
        metrics["vqa_accuracy"] = _vqa_accuracy(open_preds, open_tgts)

    # Overall accuracy (all types combined).
    all_preds_by_type: dict[str, tuple[list, list]] = {}
    for pred, tgt, qt in zip(predictions, targets, qtypes):
        all_preds_by_type.setdefault(qt, ([], []))[0].append(pred)
        all_preds_by_type.setdefault(qt, ([], []))[1].append(tgt)

    # Per-sample log.
    per_sample = [
        {"question": s["question"], "prediction": p, "answer": s["answer"],
         "image": str(s["image_path"])}
        for s, p in zip(samples, predictions)
    ]

    result = {
        "task": task,
        "n_samples": len(samples),
        "metrics": metrics,
        "per_sample": per_sample,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"[native_runner] {task}: metrics={metrics} → {output_path}")
    return result

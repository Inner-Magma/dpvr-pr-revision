"""Data loading and collation for LLaVA-format training."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from PIL import Image


def load_raw_dataset(
    json_path: str,
    subset_fraction: float = 1.0,
    seed: int = 42,
) -> dict[str, Dataset]:
    """Load a LLaVA-format JSON, return {"train": Dataset}."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if subset_fraction < 1.0:
        rng = random.Random(seed)
        n = max(1, int(len(data) * subset_fraction))
        data = rng.sample(data, n)

    # Normalise mixed-type 'id' field (real 665K data has int + str IDs).
    for item in data:
        if "id" in item and not isinstance(item["id"], str):
            item["id"] = str(item["id"])

    # Filter out text-only entries (no image field): LLaVA training expects
    # all samples in a batch to have pixel_values — mixed batches cause
    # shape mismatches in the vision tower across multi-GPU replicas.
    n_before = len(data)
    data = [d for d in data if d.get("image")]
    if len(data) < n_before:
        print(f"[load_raw_dataset] Filtered {n_before - len(data)} text-only entries "
              f"({len(data)} remaining).")

    return {"train": Dataset.from_list(data)}


class LlavaDataCollator:
    """Collate LLaVA-format raw dicts into model-ready tensors."""

    def __init__(self, processor: Any, image_base_path: str):
        self.processor = processor
        self.image_base_path = Path(image_base_path)
        # How many tokens does one <image> placeholder expand to in input_ids?
        # LlavaProcessor replaces each <image> with (image_size/patch_size)^2 tokens.
        # For LLaVA-1.5 (CLIP ViT-L/14@336): 336/14=24 → 24*24=576 patch tokens.
        # CLIPImageProcessor.patch_size is not stored in preprocessor_config.json,
        # so we derive it from crop_size + a known-good patch_size, or fall back to 576.
        try:
            ip = processor.image_processor
            h = ip.crop_size["height"]          # 336
            ps = (
                getattr(ip, "patch_size", None)
                or getattr(processor, "patch_size", None)
                or 14                           # LLaVA-1.5 CLIP ViT-L/14
            )
            self.image_patch_tokens = (h // ps) ** 2   # 576
        except Exception:
            self.image_patch_tokens = 576       # safe default for LLaVA-1.5 7B/13B

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[Image.Image] = []

        for sample in batch:
            texts.append(self._build_text(sample.get("conversations", [])))
            images.append(self._load_image(sample.get("image")))

        encoding = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        # All labels default to -100; only assistant response tokens are unmasked.
        # This matches LLaVA-1.5 official training (preprocess_v1 in llava/train/train.py)
        # where only the "ASSISTANT:" response contributes to the loss.
        labels = torch.full_like(encoding["input_ids"], -100)
        tokenizer = self.processor.tokenizer

        for i, sample in enumerate(batch):
            cumulative = ""
            for conv in sample.get("conversations", []):
                role = conv.get("from") or conv.get("role", "")
                value = conv.get("value") or conv.get("content", "")
                if role in ("human", "user"):
                    cumulative += f"USER: {value} "
                elif role in ("gpt", "assistant"):
                    prompt_end = cumulative + "ASSISTANT: "
                    answer_end = prompt_end + f"{value}</s>"
                    # Locate answer boundaries in encoding["input_ids"].
                    #
                    # L_a: end position — tokenize the full cumulative text (BOS once),
                    # then adjust for <image> expansion (1 raw token → 576 in input_ids).
                    L_a_text = len(tokenizer(answer_end, add_special_tokens=True,
                                             return_tensors="pt").input_ids[0])
                    n_img_a = answer_end.count("<image>")
                    L_a = L_a_text + n_img_a * (self.image_patch_tokens - 1)
                    L_a = min(L_a, labels.shape[1])
                    #
                    # L_p: start position — subtract the answer-only token count from L_a.
                    # Using (L_a - L_ans) instead of tokenizing the prefix avoids a
                    # tokenizer-boundary off-by-1: LLaMA's tokenizer encodes a trailing
                    # space as a standalone token in the prefix string but absorbs it into
                    # the following word's token in the full string, shifting L_p by +1.
                    L_ans = len(tokenizer(f"{value}</s>", add_special_tokens=False,
                                          return_tensors="pt").input_ids[0])
                    L_p = max(0, L_a - L_ans)
                    labels[i, L_p:L_a] = encoding["input_ids"][i, L_p:L_a]
                    cumulative = answer_end

        encoding["labels"] = labels
        return dict(encoding)

    def _load_image(self, rel_path: str | None) -> Image.Image:
        if rel_path:
            try:
                return Image.open(self.image_base_path / rel_path).convert("RGB")
            except Exception:
                pass
        return Image.new("RGB", (336, 336), color=(128, 128, 128))

    def _build_text(self, conversations: list[dict]) -> str:
        parts: list[str] = []
        for conv in conversations:
            role = conv.get("from") or conv.get("role", "")
            value = conv.get("value") or conv.get("content", "")
            if role in ("human", "user"):
                parts.append(f"USER: {value} ")
            elif role in ("gpt", "assistant"):
                parts.append(f"ASSISTANT: {value}</s>")
        return "".join(parts)


def verify_image_existence(
    json_path: str,
    image_base_path: str,
    sample_limit: int | None = None,
) -> None:
    """Check that image files referenced in a LLaVA JSON actually exist on disk.

    Prints a summary of missing / found counts. Raises RuntimeError if > 5% missing.
    """
    import json as _json
    json_path = Path(json_path)
    base = Path(image_base_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = _json.load(f)

    if sample_limit is not None:
        data = data[:sample_limit]

    missing, found = [], 0
    for item in data:
        rel = item.get("image")
        if not rel:
            continue
        p = base / rel
        if p.exists():
            found += 1
        else:
            missing.append(str(p))

    total = found + len(missing)
    pct_missing = 100.0 * len(missing) / max(total, 1)
    print(f"[prepare_data] {found}/{total} images found ({pct_missing:.1f}% missing)")
    if missing:
        print(f"  First 5 missing: {missing[:5]}")
    if pct_missing > 5.0:
        raise RuntimeError(
            f"{len(missing)}/{total} images missing ({pct_missing:.1f}%). "
            "Re-run download or adjust image_base_path."
        )
    print("[prepare_data] Dataset integrity OK.")


__all__ = ["load_raw_dataset", "LlavaDataCollator", "verify_image_existence"]

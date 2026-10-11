"""LLaVA-665k collation for Qwen2.5-VL (second backbone, C8 / E6).

Each LLaVA conversation becomes a Qwen chat: the processor's own chat template
(the default system prompt "You are a helpful assistant.", as in lmms-eval's
``qwen2_5_vl`` handler), the image placed before the text of its turn, again
as the eval handler does. The Qwen processor resizes every image within
``[min_pixels, max_pixels]`` and expands ``<|image_pad|>`` to the image's own
token count, so samples in a batch hold different numbers of image tokens.

Labels cover each assistant reply and its ``<|im_end|>``; everything else
(system, user turns, the image, padding) is -100.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

IMAGE_PLACEHOLDER = "<image>"


def to_qwen_messages(conversations: list[dict]) -> list[dict]:
    """LLaVA ``conversations`` to chat-template messages with one image entry."""
    messages: list[dict] = []
    image_used = False
    for conv in conversations:
        role = conv.get("from") or conv.get("role", "")
        value = conv.get("value") or conv.get("content", "")
        if role in ("human", "user"):
            content: list[dict] = []
            if IMAGE_PLACEHOLDER in value and not image_used:
                content.append({"type": "image"})
                image_used = True
            text = value.replace(IMAGE_PLACEHOLDER, "").strip()
            content.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": content})
        elif role in ("gpt", "assistant"):
            messages.append({"role": "assistant", "content": [{"type": "text", "text": value}]})
    if not image_used and messages and messages[0]["role"] == "user":
        # The sample has an image file but no placeholder: show it first.
        messages[0]["content"].insert(0, {"type": "image"})
    return messages


def assistant_labels(input_ids: torch.Tensor, header: list[int], end_id: int,
                     skip: int = 1) -> torch.Tensor:
    """-100 everywhere except each assistant reply and its closing end_id.

    ``header`` is the token sequence of ``<|im_start|>assistant``; the reply
    starts ``skip`` tokens later (the newline). Matching without the newline
    keeps working when the tokenizer merges it with a reply that starts with
    whitespace.
    """
    labels = torch.full_like(input_ids, -100)
    n = len(header)
    for b in range(input_ids.shape[0]):
        row = input_ids[b].tolist()
        i = 0
        while i <= len(row) - n:
            if row[i:i + n] == header:
                start = min(i + n + skip, len(row))
                end = start
                while end < len(row) and row[end] != end_id:
                    end += 1
                stop = min(end + 1, len(row))
                labels[b, start:stop] = input_ids[b, start:stop]
                i = stop
            else:
                i += 1
    return labels


class QwenVLDataCollator:
    """Collate LLaVA-format samples for Qwen2.5-VL training (right padding)."""

    def __init__(self, processor: Any, image_base_path: str, max_length: int = 4096):
        self.processor = processor
        self.image_base_path = Path(image_base_path)
        self.max_length = int(max_length)
        tok = processor.tokenizer
        tok.padding_side = "right"
        self.header = tok.encode("<|im_start|>assistant", add_special_tokens=False)
        self.end_id = tok.convert_tokens_to_ids("<|im_end|>")

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts, images = [], []
        for sample in batch:
            messages = to_qwen_messages(sample.get("conversations", []))
            texts.append(self.processor.apply_chat_template(messages, tokenize=False,
                                                            add_generation_prompt=False))
            images.append(self._load_image(sample.get("image")))
        enc = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        if enc["input_ids"].shape[1] > self.max_length:
            # Images come first in 665k, so cutting the tail keeps every image
            # token (a cut through an image would break the feature count).
            pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            n_img = (enc["input_ids"] == pad_id).sum()
            for k in ("input_ids", "attention_mask"):
                enc[k] = enc[k][:, : self.max_length]
            if (enc["input_ids"] == pad_id).sum() != n_img:
                raise ValueError(f"max_length={self.max_length} cuts through image tokens")
        labels = assistant_labels(enc["input_ids"], self.header, self.end_id)
        labels[enc["attention_mask"] == 0] = -100
        out = dict(enc)
        out["labels"] = labels
        return out

    def _load_image(self, rel_path: str | None) -> Image.Image:
        if rel_path:
            try:
                return Image.open(self.image_base_path / rel_path).convert("RGB")
            except Exception:
                pass
        return Image.new("RGB", (336, 336), color=(128, 128, 128))


__all__ = ["QwenVLDataCollator", "assistant_labels", "to_qwen_messages"]

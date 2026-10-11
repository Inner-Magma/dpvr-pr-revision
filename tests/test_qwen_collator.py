"""QwenVLDataCollator label masking and message building (no model download).

The real Qwen2.5-VL processor needs the hub; a small stand-in reproduces the
parts the collator relies on: the Qwen chat format, ``<|image_pad|>``
expansion to a per-image token count, and right padding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.data.qwen import QwenVLDataCollator, assistant_labels, to_qwen_messages  # noqa: E402

SPECIAL = ["<pad>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]


class _Tok:
    padding_side = "left"

    def __init__(self):
        self.vocab = {t: i for i, t in enumerate(SPECIAL)}

    def _id(self, piece):
        return self.vocab.setdefault(piece, len(self.vocab))

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for s in SPECIAL:
                if text.startswith(s, i):
                    ids.append(self.vocab[s])
                    i += len(s)
                    break
            else:
                j = i
                while j < len(text) and not text[j].isspace() and not any(text.startswith(s, j) for s in SPECIAL):
                    j += 1
                ids.append(self._id(text[i:max(j, i + 1)]))
                i = max(j, i + 1)
        return ids

    def convert_tokens_to_ids(self, t):
        return self.vocab[t]


class _Proc:
    """Qwen chat format; an image of w x h px gives (w//28)*(h//28) tokens."""

    def __init__(self):
        self.tokenizer = _Tok()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        out = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        for m in messages:
            body = "".join("<|vision_start|><|image_pad|><|vision_end|>" if c["type"] == "image" else c["text"]
                           for c in m["content"])
            out += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
        return out

    def __call__(self, text, images, padding=True, return_tensors="pt"):
        rows = []
        for t, im in zip(text, images):
            n = (im.size[0] // 28) * (im.size[1] // 28)
            rows.append(self.tokenizer.encode(t.replace("<|image_pad|>", "<|image_pad|>" * n)))
        S = max(map(len, rows))
        ids = torch.zeros(len(rows), S, dtype=torch.long)
        att = torch.zeros(len(rows), S, dtype=torch.long)
        for b, r in enumerate(rows):
            ids[b, :len(r)] = torch.tensor(r)
            att[b, :len(r)] = 1
        return {"input_ids": ids, "attention_mask": att,
                "image_grid_thw": torch.tensor([[1, im.size[1] // 14, im.size[0] // 14] for im in images])}


def _conv(q, a, q2=None, a2=None):
    c = [{"from": "human", "value": f"<image>\n{q}"}, {"from": "gpt", "value": a}]
    if q2:
        c += [{"from": "human", "value": q2}, {"from": "gpt", "value": a2}]
    return c


def test_messages_put_image_first_once():
    msgs = to_qwen_messages(_conv("What is it?", "A cat.", "Color?", "Black."))
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"][0] == {"type": "image"}
    assert msgs[0]["content"][1]["text"] == "What is it?"
    assert all(c["type"] == "text" for c in msgs[2]["content"])
    no_tag = to_qwen_messages([{"from": "human", "value": "Describe."}, {"from": "gpt", "value": "Ok"}])
    assert no_tag[0]["content"][0] == {"type": "image"}


def test_assistant_labels_cover_replies_only():
    ids = torch.tensor([[1, 7, 8, 5, 5, 2, 1, 9, 3, 30, 31, 2, 0, 0]])
    lab = assistant_labels(ids, header=[1, 9], end_id=2)
    assert lab[0].tolist() == [-100] * 9 + [30, 31, 2] + [-100] * 2


def test_collator_variable_image_tokens_and_labels(tmp_path):
    for name, size in (("a.jpg", (56, 56)), ("b.jpg", (112, 84))):
        Image.new("RGB", size).save(tmp_path / name)
    proc = _Proc()
    col = QwenVLDataCollator(proc, image_base_path=str(tmp_path))
    assert proc.tokenizer.padding_side == "right"
    batch = col([{"image": "a.jpg", "conversations": _conv("What is it?", "A cat.")},
                 {"image": "b.jpg", "conversations": _conv("Q1", "R1 x", "Q2", "R2")}])
    pad_id = proc.tokenizer.vocab["<|image_pad|>"]
    assert ((batch["input_ids"] == pad_id).sum(1)).tolist() == [4, 12]
    tok = proc.tokenizer
    inv = {v: k for k, v in tok.vocab.items()}
    for b, expected in enumerate([["A", "cat.", "<|im_end|>"],
                                  ["R1", "x", "<|im_end|>", "R2", "<|im_end|>"]]):
        kept = [inv[int(t)] for t in batch["labels"][b] if t != -100]
        assert [k for k in kept if k.strip()] == expected
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    assert not (batch["labels"] == pad_id).any()

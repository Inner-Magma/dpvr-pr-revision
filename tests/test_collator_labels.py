"""Unit tests: LlavaDataCollator must mask user-turn tokens in labels.

The fix ensures only ASSISTANT response tokens contribute to the loss,
matching LLaVA-1.5's official preprocess_v1 behaviour.

Run (no GPU required):
    pytest tests/test_collator_labels.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.data import LlavaDataCollator


# ---------------------------------------------------------------------------
# Minimal fake processor (no vision model required)
# ---------------------------------------------------------------------------

class _FakeProcessor:
    """Wraps a real tokenizer, stubs out image processing for label tests."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, images, text, return_tensors=None, padding=True,
                 truncation=True, max_length=2048):
        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _get_tokenizer():
    """Load LLaVA tokenizer from local model path; skip if unavailable."""
    from dpvr.utils import load_config, load_paths, resolve_paths
    try:
        cfg = load_config(str(REPO_ROOT / "configs/method/token_diversion.yaml"))
        paths = load_paths()
        cfg = resolve_paths(cfg, paths)
        model_path = cfg["model"]["base_model_path"]
    except Exception:
        pytest.skip("Cannot resolve model path from config")
    if not Path(model_path).exists():
        pytest.skip(f"Model not found at {model_path}")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path)


def _make_sample(user_text: str, assistant_text: str) -> dict:
    return {
        "image": None,
        "conversations": [
            {"from": "human", "value": user_text},
            {"from": "gpt",   "value": assistant_text},
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_assistant_tokens_unmasked():
    """Labels on the assistant response must equal input_ids (not -100)."""
    tokenizer = _get_tokenizer()
    processor = _FakeProcessor(tokenizer)
    collator = LlavaDataCollator(processor, image_base_path="/tmp")

    user_q = "What is the capital of France?"
    asst_a = "Paris."
    batch = [_make_sample(user_q, asst_a)]

    out = collator(batch)
    labels = out["labels"][0]
    input_ids = out["input_ids"][0]

    # At least the answer tokens ("Paris</s>") should be unmasked
    unmasked = (labels != -100).sum().item()
    assert unmasked > 0, "No labels unmasked — assistant response completely masked"

    # Wherever labels != -100, they must equal input_ids
    mask = labels != -100
    assert torch.all(labels[mask] == input_ids[mask]), \
        "Unmasked labels don't match input_ids"


def test_user_tokens_masked():
    """Labels on the user-turn tokens must all be -100."""
    tokenizer = _get_tokenizer()
    processor = _FakeProcessor(tokenizer)
    collator = LlavaDataCollator(processor, image_base_path="/tmp")

    user_q = "Describe this image in detail."
    asst_a = "A cat sitting on a mat."
    batch = [_make_sample(user_q, asst_a)]

    out = collator(batch)
    labels = out["labels"][0]

    # Compute L_p the same way the implementation does: L_a - L_ans.
    # (Using the old prefix-tokenization method would be off-by-1 due to
    # trailing-space boundary effects in LLaMA tokenizer.)
    L_ans = len(tokenizer(f"{asst_a}</s>", add_special_tokens=False,
                          return_tensors="pt").input_ids[0])
    full_text = f"USER: {user_q} ASSISTANT: {asst_a}</s>"
    L_a = len(tokenizer(full_text, add_special_tokens=True,
                        return_tensors="pt").input_ids[0])
    L_p = L_a - L_ans

    # Everything before the assistant answer must be -100
    assert torch.all(labels[:L_p] == -100), \
        f"User-turn tokens at positions 0..{L_p} are not fully masked"


def test_loss_token_count_matches_assistant_length():
    """(labels != -100).sum() should roughly equal the assistant token count."""
    tokenizer = _get_tokenizer()
    processor = _FakeProcessor(tokenizer)
    collator = LlavaDataCollator(processor, image_base_path="/tmp")

    user_q = "What color is the sky?"
    asst_a = "The sky is blue."
    batch = [_make_sample(user_q, asst_a)]

    out = collator(batch)
    labels = out["labels"][0]

    # Count assistant answer tokens (with EOS)
    answer_text = f"{asst_a}</s>"
    answer_ids = tokenizer(answer_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    expected_unmasked = len(answer_ids)

    actual_unmasked = (labels != -100).sum().item()
    # Allow ±2 tokens for tokenizer boundary effects
    assert abs(actual_unmasked - expected_unmasked) <= 2, (
        f"Expected ~{expected_unmasked} unmasked tokens, got {actual_unmasked}"
    )


def test_multi_turn_both_answers_unmasked():
    """For a 2-turn conversation, both assistant answers must be unmasked."""
    tokenizer = _get_tokenizer()
    processor = _FakeProcessor(tokenizer)
    collator = LlavaDataCollator(processor, image_base_path="/tmp")

    sample = {
        "image": None,
        "conversations": [
            {"from": "human", "value": "What is 1+1?"},
            {"from": "gpt",   "value": "Two."},
            {"from": "human", "value": "And 2+2?"},
            {"from": "gpt",   "value": "Four."},
        ],
    }

    out = collator([sample])
    labels = out["labels"][0]

    unmasked = (labels != -100).sum().item()
    assert unmasked > 0, "Multi-turn: no tokens unmasked"

    # Both "Two." and "Four." should contribute; should be more than a single answer
    answer1_ids = tokenizer("Two.</s>", add_special_tokens=False).input_ids
    answer2_ids = tokenizer("Four.</s>", add_special_tokens=False).input_ids
    min_expected = len(answer1_ids) + len(answer2_ids) - 2  # allow small boundary slack
    assert unmasked >= min_expected, (
        f"Multi-turn: expected at least {min_expected} unmasked, got {unmasked}"
    )


def test_pad_tokens_masked():
    """Padding tokens must remain -100 (batch with different-length sequences)."""
    tokenizer = _get_tokenizer()
    processor = _FakeProcessor(tokenizer)
    collator = LlavaDataCollator(processor, image_base_path="/tmp")

    batch = [
        _make_sample("Short?", "Yes."),
        _make_sample("A much longer user question about the world around us.", "Indeed."),
    ]

    out = collator(batch)
    labels = out["labels"]
    input_ids = out["input_ids"]

    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        pad_positions = input_ids == pad_id
        assert torch.all(labels[pad_positions] == -100), \
            "Pad token positions have non-(-100) labels"

def test_batch_padding_does_not_shift_labels():
    """Regression: 同一 batch 内长度不等时（左 padding），监督窗口不得整体左移。

    旧实现在未 padding 的文本坐标里算 L_p/L_a，却写进已左 padding 的张量，
    导致每行监督窗口左移 pad_len 位：错标 prompt 尾部 / pad token，漏标答案。
    这里对两条长度差很大的样本逐行断言：非 -100 的位置恰好等于 gold 区间 + pad。
    """
    tokenizer = _get_tokenizer()
    processor = _FakeProcessor(tokenizer)
    collator = LlavaDataCollator(processor, image_base_path="/tmp")

    batch = [
        _make_sample("Hi?", "Yes."),
        _make_sample(
            "Please describe in detail what you can see in this picture "
            "and explain every object and its spatial relationship.",
            "A small wooden table stands beside a window, with two red cups "
            "and a folded newspaper resting on its surface.",
        ),
        _make_sample("What is 2+2?", "Four."),
    ]

    out = collator(batch)
    ids, labels = out["input_ids"], out["labels"]
    pad_id = tokenizer.pad_token_id
    W = ids.shape[1]

    pads = []
    for i in range(len(batch)):
        pad = 0
        while pad < W and ids[i, pad] == pad_id:
            pad += 1
        pads.append(pad)
    assert max(pads) > 0, "expected at least one left-padded row"
    assert min(pads) == 0, "expected the longest row to be unpadded"

    for i, sample in enumerate(batch):
        pad = pads[i]

        # 真实 gold 区间（无 padding 坐标）
        cum, gold = "", []
        for c in sample["conversations"]:
            if c["from"] == "human":
                cum += f"USER: {c['value']} "
            else:
                aend = cum + "ASSISTANT: " + f"{c['value']}</s>"
                L_a = len(tokenizer(aend, add_special_tokens=True).input_ids)
                L_ans = len(tokenizer(f"{c['value']}</s>", add_special_tokens=False).input_ids)
                gold.append((L_a - L_ans, L_a))
                cum = aend

        expected = torch.zeros(W, dtype=torch.bool)
        for (gs, ge) in gold:
            expected[gs + pad:ge + pad] = True
        masked = labels[i] != -100

        assert torch.equal(masked, expected), (
            f"row {i}: unmasked positions differ from gold+pad "
            f"(pad={pad}, masked={int(masked.sum())}, expected={int(expected.sum())})"
        )
        # 额外保证：答案 token 必须被监督
        for (gs, ge) in gold:
            assert masked[gs + pad:ge + pad].all()
        # 非监督位置不得落在 padding 上
        assert torch.all(labels[i][:pad] == -100), f"row {i}: pad positions are supervised"

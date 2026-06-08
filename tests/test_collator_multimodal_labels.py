"""Multimodal label-masking test: verifies labels land on the assistant
response and NOT on the expanded <image> patch tokens.

Uses the real AutoProcessor (not FakeProcessor) so that <image> expansion
is realistic.  Requires the model tokenizer to be available at the configured
base_model_path.

Run (no GPU required):
    pytest tests/test_collator_multimodal_labels.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.data import LlavaDataCollator


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _get_real_processor():
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
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_path)


def _sample_with_image(user_text: str, assistant_text: str) -> dict:
    return {
        "image": None,  # fallback to gray placeholder in collator
        "conversations": [
            {"from": "human", "value": user_text},
            {"from": "gpt",   "value": assistant_text},
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_multimodal_labels_land_on_response_not_image():
    """With <image> in user turn, unmasked tokens must decode to the assistant
    response only — not to image-patch placeholders or user-prompt fragments."""
    processor = _get_real_processor()
    collator = LlavaDataCollator(processor, image_base_path="/tmp")

    sample = _sample_with_image(
        user_text="<image>\nWhat color is dominant?",
        assistant_text="Red.",
    )
    out = collator([sample])
    labels    = out["labels"][0]
    input_ids = out["input_ids"][0]

    # 1. At least one token unmasked; response is short (< 50 tokens).
    unmasked = (labels != -100).sum().item()
    assert unmasked > 0,  "No labels unmasked — response is fully masked"
    assert unmasked < 50, f"Too many unmasked tokens ({unmasked}); likely landed on image area"

    # 2. Decoded unmasked tokens must not contain prompt artefacts.
    unmasked_ids = input_ids[labels != -100].tolist()
    decoded = processor.tokenizer.decode(unmasked_ids)
    bad_markers = ["USER", "ASSISTANT", "<image>"]
    for marker in bad_markers:
        assert marker not in decoded, (
            f"Decoded unmasked tokens contain prompt marker {marker!r}: {decoded!r}"
        )
    # Must contain "Red" (the actual answer content).
    assert "Red" in decoded, f"Expected 'Red' in decoded, got: {decoded!r}"

    # 3. Unmasked region must NOT be dominated by the image_token_id.
    img_tok_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    img_token_count = sum(1 for t in unmasked_ids if t == img_tok_id)
    assert img_token_count == 0, (
        f"Unmasked region contains {img_token_count} image-placeholder tokens "
        f"— labels landed in the image expansion area"
    )


def test_multimodal_labels_consistent_with_tokenonly():
    """With NO <image> in user turn the masking must still be correct (regression)."""
    processor = _get_real_processor()
    collator = LlavaDataCollator(processor, image_base_path="/tmp")

    sample = _sample_with_image(
        user_text="What is 2+2?",  # no <image>
        assistant_text="Four.",
    )
    out = collator([sample])
    labels    = out["labels"][0]
    input_ids = out["input_ids"][0]

    unmasked_ids = input_ids[labels != -100].tolist()
    decoded = processor.tokenizer.decode(unmasked_ids)

    assert "Four" in decoded, f"Expected 'Four' in decoded, got: {decoded!r}"
    assert "USER" not in decoded, f"USER present in decoded: {decoded!r}"
    assert "ASSISTANT" not in decoded, f"ASSISTANT present in decoded: {decoded!r}"


def test_image_patch_tokens_attr_computed_correctly():
    """LlavaDataCollator.image_patch_tokens should be 576 for LLaVA-1.5."""
    processor = _get_real_processor()
    collator = LlavaDataCollator(processor, image_base_path="/tmp")
    assert collator.image_patch_tokens == 576, (
        f"Expected image_patch_tokens=576 for LLaVA-1.5, got {collator.image_patch_tokens}"
    )

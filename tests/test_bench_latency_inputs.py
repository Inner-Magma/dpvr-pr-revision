"""bench_latency builds inputs the LLaVA forward accepts (image tokens = patches)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "revision"))

import bench_latency  # noqa: E402


def test_inputs_match_tiny_llava():
    from transformers import (
        CLIPVisionConfig,
        LlamaConfig,
        LlavaConfig,
        LlavaForConditionalGeneration,
    )

    cfg = LlavaConfig(
        vision_config=CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                       num_attention_heads=2, image_size=56, patch_size=14),
        text_config=LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                num_attention_heads=2, num_key_value_heads=2, vocab_size=128),
        image_token_index=100,
    )
    model = LlavaForConditionalGeneration(cfg).to(torch.bfloat16).eval()
    inputs, n_img = bench_latency.make_inputs(cfg, batch=2, n_txt=8, device="cpu")
    assert n_img == 16
    assert inputs["input_ids"].shape == (2, 1 + 16 + 7)
    assert int((inputs["input_ids"] == 100).sum()) == 2 * 16
    with torch.inference_mode():
        out = model(**inputs, use_cache=False)
    assert out.logits.shape[:2] == (2, 24)

"""bench_generate forces exactly N new tokens and runs on a tiny LLaVA (CPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "revision"))

import bench_generate  # noqa: E402
import bench_latency  # noqa: E402


def test_timed_generate_exact_length(monkeypatch):
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    cfg = LlavaConfig(
        vision_config=CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                       num_attention_heads=2, image_size=28, patch_size=14),
        text_config=LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                num_attention_heads=2, num_key_value_heads=2, vocab_size=128,
                                eos_token_id=2),
        image_token_index=100,
    )
    model = LlavaForConditionalGeneration(cfg).eval()
    inputs, _ = bench_latency.make_inputs(cfg, 1, 6, "cpu")
    inputs["pixel_values"] = inputs["pixel_values"].float()
    for cache in (True, False):
        ms = bench_generate.timed_generate(model, inputs, 5, cache, pad_id=0)
        assert ms > 0

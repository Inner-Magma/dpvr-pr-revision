"""Analytic FLOPs must equal torch's FlopCounterMode on real forward passes."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dpvr.profiling.analytic_flops import (  # noqa: E402
    PRESETS,
    LMDims,
    VisionDims,
    image_share_per_layer,
    prefill_flops,
)


def _count(fn) -> int:
    """Matmul FLOPs of fn(), minus the RoPE frequency outer product (not a model matmul)."""
    with FlopCounterMode(display=False) as fc:
        fn()
    rope = max([sum(v.values()) for k, v in fc.get_flop_counts().items()
                if k.split(".")[-1] in ("rotary_emb", "LlamaRotaryEmbedding")] or [0])
    return fc.get_total_flops() - rope


def _eager(config) -> None:
    # FlopCounterMode has no formula for the CPU SDPA kernel; eager attention is plain matmuls.
    config._attn_implementation = "eager"


def test_vanilla_llava_matches_flop_counter():
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

    vis = VisionDims(hidden=32, intermediate=64, layers=2, image_size=56, patch_size=14)
    lm = LMDims(hidden=48, intermediate=96, layers=3, heads=4, kv_heads=4, vocab=200)
    cfg = LlavaConfig(
        vision_config=CLIPVisionConfig(hidden_size=vis.hidden, intermediate_size=vis.intermediate,
                                       num_hidden_layers=vis.layers, num_attention_heads=2,
                                       image_size=vis.image_size, patch_size=vis.patch_size),
        text_config=LlamaConfig(hidden_size=lm.hidden, intermediate_size=lm.intermediate,
                                num_hidden_layers=lm.layers, num_attention_heads=lm.heads,
                                num_key_value_heads=lm.kv_heads, vocab_size=lm.vocab),
        image_token_index=150,
    )
    _eager(cfg)
    _eager(cfg.text_config)
    _eager(cfg.vision_config)
    model = LlavaForConditionalGeneration(cfg).eval()
    n_img, n_txt = vis.patches, 9
    ids = torch.cat([torch.tensor([[1]]), torch.full((1, n_img), 150),
                     torch.randint(3, 100, (1, n_txt - 1))], dim=1)
    pv = torch.randn(1, 3, vis.image_size, vis.image_size)
    with torch.no_grad():
        got = _count(lambda: model(input_ids=ids, pixel_values=pv, attention_mask=torch.ones_like(ids)))
    want = prefill_flops(lm, n_img, n_txt, attention="dense", vision=vis)["total"]
    assert got == want


@pytest.mark.parametrize("split_layer,fusion_count", [(2, 1), (2, 2), (3, 1)])
def test_dpvr_lf_matches_flop_counter(split_layer, fusion_count):
    from dpvr.models.token_diversion_x3_fusion import TokenDiversionX3FusionLlava

    hidden, n_layers, vocab = 64, 5, 300
    torch.manual_seed(0)
    m = TokenDiversionX3FusionLlava._build_test_instance(
        hidden_size=hidden, n_layers=n_layers, n_heads=4, vocab=vocab,
        split_layer=split_layer, vision_depth=1, fusion_count=fusion_count,
    ).eval()
    _eager(m.llava_model.language_model.config)
    S = 16  # the test instance marks the first S // 2 positions as image tokens
    ids = torch.randint(0, vocab, (1, S))
    with torch.no_grad():
        got = _count(lambda: m(input_ids=ids, pixel_values=torch.zeros(1),
                               attention_mask=torch.ones_like(ids)))
    lm = LMDims(hidden=hidden, intermediate=hidden * 4, layers=n_layers, heads=4, kv_heads=4,
                vocab=vocab)
    want = prefill_flops(lm, S // 2, S - S // 2, method="dpvr_lf", split_layer=split_layer,
                         fusion_count=fusion_count, attention="dense", vision=None)["total"]
    assert got == want


def test_limits_and_ordering():
    d = PRESETS["llava-1.5-7b"]
    van = prefill_flops(d, 576, 128)["total"]
    # Splitting at the last layer with every remaining layer fused is vanilla.
    assert prefill_flops(d, 576, 128, method="lf_no_branch", split_layer=31, fusion_count=1)["total"] == van
    lf = prefill_flops(d, 576, 128, method="dpvr_lf", split_layer=18)["total"]
    vtw = prefill_flops(d, 576, 128, method="vtw", split_layer=18)["total"]
    nob = prefill_flops(d, 576, 128, method="lf_no_branch", split_layer=18)["total"]
    assert vtw < nob < lf < van
    assert prefill_flops(d, 576, 128, method="freeze_image", split_layer=18)["total"] == van
    causal = prefill_flops(d, 576, 128, attention="causal")["total"]
    assert causal < van
    assert 0.80 < image_share_per_layer(d, 576, 128) < 0.83


def test_invalid_split():
    with pytest.raises(ValueError):
        prefill_flops(PRESETS["llava-1.5-7b"], 576, 128, method="dpvr_lf", split_layer=32)

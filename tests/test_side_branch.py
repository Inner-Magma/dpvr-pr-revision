"""Side-branch variants (C9): parameter counts, trainability, init, gradient flow."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dpvr.models.side_branch import (  # noqa: E402
    LoRALinear,
    MLPBranchLayer,
    build_side_branch,
    set_branch_trainable,
)
from dpvr.models.token_diversion_x3_fusion import TokenDiversionX3FusionLlava  # noqa: E402

CFG = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=4, vocab_size=100)
SOURCE = torch.nn.ModuleList([LlamaDecoderLayer(CFG, layer_idx=i) for i in range(4)])


def _n(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def test_param_counts():
    layer = _n(SOURCE[0])
    t = build_side_branch(CFG, 2, depth=2, dtype=torch.float32)
    assert _n(t) == 2 * layer
    m = build_side_branch(CFG, 2, branch_type="mlp", dtype=torch.float32)
    assert _n(m) == _n(SOURCE[0].mlp) + _n(SOURCE[0].post_attention_layernorm)
    lo = build_side_branch(CFG, 2, branch_type="lora", source_layers=SOURCE, lora_rank=4,
                           dtype=torch.float32)
    set_branch_trainable(lo, "lora")
    d, f = CFG.hidden_size, CFG.intermediate_size
    expect = 4 * 4 * (d + d) + 4 * ((d + f) * 2 + (f + d))  # attn q,k,v,o + mlp gate,up,down
    assert _n(lo, trainable_only=True) == expect


def test_layer_init_copies_weights():
    t = build_side_branch(CFG, 2, branch_init="layer", source_layers=SOURCE, dtype=torch.float32)
    for (k, a), (_, b) in zip(t[0].state_dict().items(), SOURCE[2].state_dict().items()):
        assert torch.equal(a, b), k
    m = build_side_branch(CFG, 1, branch_type="mlp", branch_init="layer", source_layers=SOURCE,
                          dtype=torch.float32)
    assert torch.equal(m[0].mlp.down_proj.weight, SOURCE[1].mlp.down_proj.weight)
    lo = build_side_branch(CFG, 1, branch_type="lora", source_layers=SOURCE, dtype=torch.float32)
    assert torch.equal(lo[0].self_attn.q_proj.base.weight, SOURCE[1].self_attn.q_proj.weight)
    assert lo[0].self_attn.q_proj.base.weight.data_ptr() != SOURCE[1].self_attn.q_proj.weight.data_ptr()


def test_lora_trainability():
    lo = build_side_branch(CFG, 1, branch_type="lora", source_layers=SOURCE, dtype=torch.float32)
    set_branch_trainable(lo, "lora")
    for name, p in lo.named_parameters():
        assert p.requires_grad == ("lora_" in name), name


def test_invalid_options():
    with pytest.raises(ValueError):
        build_side_branch(CFG, 1, branch_type="conv")
    with pytest.raises(ValueError):
        build_side_branch(CFG, 1, branch_type="lora")  # no source layers
    with pytest.raises(ValueError):
        build_side_branch(CFG, 3, depth=2, branch_init="layer", source_layers=SOURCE)


def _x3(**kw):
    torch.manual_seed(0)
    return TokenDiversionX3FusionLlava._build_test_instance(
        hidden_size=64, n_layers=4, n_heads=4, vocab=200, split_layer=2, **kw)


def _loss(model):
    ids = torch.randint(0, 200, (2, 12), generator=torch.Generator().manual_seed(3))
    out = model(input_ids=ids, pixel_values=torch.zeros(1), attention_mask=torch.ones_like(ids),
                labels=ids)
    return out


@pytest.mark.parametrize("kw", [
    {}, {"branch_type": "mlp"}, {"branch_type": "lora"},
    {"branch_init": "layer"}, {"vision_depth": 2},
])
def test_gradient_reaches_branch(kw):
    model = _x3(**kw)
    set_branch_trainable(model.single_transformer, kw.get("branch_type", "transformer"))
    _loss(model).loss.backward()
    grads = {n: p.grad for n, p in model.single_transformer.named_parameters() if p.requires_grad}
    assert grads
    # LoRA's A gets zero gradient at init because B starts at zero; every other weight moves.
    moving = {n: g for n, g in grads.items() if "lora_A" not in n}
    assert all(g is not None and g.abs().sum() > 0 for g in moving.values()), \
        [n for n, g in moving.items() if g is None or g.abs().sum() == 0]


def test_lora_at_init_equals_layer_copy():
    lora = _x3(branch_type="lora")
    copy_ = _x3(branch_init="layer")
    with torch.no_grad():
        torch.testing.assert_close(_loss(lora).logits, _loss(copy_).logits)


def test_mlp_branch_is_residual():
    layer = MLPBranchLayer(CFG)
    torch.nn.init.zeros_(layer.mlp.down_proj.weight)
    h = torch.randn(1, 3, CFG.hidden_size)
    torch.testing.assert_close(layer(h, position_embeddings=None, attention_mask=None), h)


def test_lora_linear_zero_init():
    base = torch.nn.Linear(8, 6)
    lo = LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(3, 8)
    torch.testing.assert_close(lo(x), base(x))

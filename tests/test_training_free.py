"""Training-free variants (C3): limiting cases and equivalences.

- split at the last layer, or no image -> exactly vanilla
- lf_no_branch == DPVR-LF's own forward with the side branch removed
- vtw == vanilla where, from layer s on, text cannot attend to image keys
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dpvr.models.token_diversion_x3_fusion import TokenDiversionX3FusionLlava  # noqa: E402
from dpvr.models.training_free import TrainingFreeLlava  # noqa: E402

IMG = slice(2, 8)
N_LAYERS, HIDDEN, VOCAB = 5, 64, 500


def _x3(split_layer=2, fusion_count=1):
    torch.manual_seed(0)
    m = TokenDiversionX3FusionLlava._build_test_instance(
        hidden_size=HIDDEN, n_layers=N_LAYERS, n_heads=4, vocab=VOCAB,
        split_layer=split_layer, vision_depth=1, fusion_count=fusion_count,
    )
    embed = nn.Embedding(VOCAB, HIDDEN)

    def prepare_multimodal_input(input_ids, pixel_values=None):
        B, S = input_ids.shape
        emb = embed(input_ids).float()
        if pixel_values is None:
            return emb, None
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, IMG] = True
        return emb, mask.unsqueeze(-1).expand(-1, -1, HIDDEN)

    m.llava_model.model = types.SimpleNamespace(prepare_multimodal_input=prepare_multimodal_input)
    m._embed = embed
    return m.eval()


def _tf(x3, variant, split_layer, fusion_count=1):
    m = TrainingFreeLlava._from_llava(x3.llava_model, x3.config, split_layer,
                                      fusion_count, variant=variant)
    # The test LLaVA is a namespace, not a module: register its weights so
    # .device / .parameters() work as on the real model.
    m._weights = nn.ModuleList([x3.llava_model.language_model, x3.llava_model.lm_head, x3._embed])
    return m.eval()


def _logits(model, ids, pv=True, mask=None):
    mask = torch.ones_like(ids) if mask is None else mask
    with torch.no_grad():
        return model(input_ids=ids, pixel_values=torch.zeros(1) if pv else None,
                     attention_mask=mask).logits


IDS = torch.randint(0, VOCAB, (2, 14), generator=torch.Generator().manual_seed(1))
TEXT = torch.ones(14, dtype=torch.bool)
TEXT[IMG] = False


@pytest.mark.parametrize("variant", ["vtw", "lf_no_branch", "freeze_image"])
def test_split_at_last_layer_is_vanilla(variant):
    x3 = _x3()
    ref = _logits(_tf(x3, "vanilla", 2), IDS)
    if variant != "lf_no_branch":  # lf_no_branch needs >= 1 fusion layer after the split
        torch.testing.assert_close(_logits(_tf(x3, variant, N_LAYERS), IDS), ref)
    # lf_no_branch with every deep layer fused leaves nothing text-only.
    full = _logits(_tf(x3, "lf_no_branch", 2, fusion_count=N_LAYERS - 2), IDS)
    torch.testing.assert_close(full, ref)


def test_freeze_from_last_layer_is_vanilla():
    x3 = _x3()
    ref = _logits(_tf(x3, "vanilla", 2), IDS)
    # Image positions end at their frozen state; text positions are what matters.
    got = _logits(_tf(x3, "freeze_image", N_LAYERS - 1), IDS)
    torch.testing.assert_close(got[:, TEXT], ref[:, TEXT])
    assert not torch.allclose(_logits(_tf(x3, "freeze_image", 2), IDS)[:, TEXT], ref[:, TEXT])


@pytest.mark.parametrize("variant", ["vtw", "lf_no_branch", "freeze_image"])
def test_no_image_is_vanilla(variant):
    x3 = _x3()
    ref = _logits(_tf(x3, "vanilla", 2), IDS, pv=False)
    torch.testing.assert_close(_logits(_tf(x3, variant, 2), IDS, pv=False), ref)


@pytest.mark.parametrize("split_layer,fusion_count", [(2, 1), (2, 2), (3, 1)])
def test_lf_no_branch_matches_dpvr_lf_without_side_branch(split_layer, fusion_count):
    x3 = _x3(split_layer, fusion_count)
    x3.single_transformer = nn.ModuleList()  # identity side branch
    ref = _logits(x3, IDS)
    got = _logits(_tf(x3, "lf_no_branch", split_layer, fusion_count), IDS)
    torch.testing.assert_close(got, ref)


def _vtw_reference(x3, ids, split_layer):
    """Full-sequence forward; from split_layer on, text queries cannot see image keys."""
    lm = x3.llava_model.language_model
    emb, _ = x3.llava_model.model.prepare_multimodal_input(ids, torch.zeros(1))
    B, S = ids.shape
    pos = torch.arange(S).unsqueeze(0).expand(B, -1)
    pe = lm.rotary_emb(emb, pos)
    neg = torch.finfo(emb.dtype).min / 2
    causal = torch.triu(torch.full((S, S), neg), diagonal=1).expand(B, 1, S, S).clone()
    blocked = causal.clone()
    blocked[:, :, :, IMG] = neg
    blocked[:, :, IMG, :] = causal[:, :, IMG, :]  # image rows unchanged (outputs unused)
    h = emb
    for i, layer in enumerate(lm.layers):
        out = layer(h, position_embeddings=pe, attention_mask=causal if i < split_layer else blocked,
                    past_key_values=None, cache_position=None, use_cache=False)
        h = out[0] if isinstance(out, (tuple, list)) else out
    return x3.llava_model.lm_head(lm.norm(h))


@pytest.mark.parametrize("split_layer", [1, 2, 4])
def test_vtw_equals_masking_image_keys(split_layer):
    x3 = _x3()
    with torch.no_grad():
        ref = _vtw_reference(x3, IDS, split_layer)
    got = _logits(_tf(x3, "vtw", split_layer), IDS)
    torch.testing.assert_close(got[:, TEXT], ref[:, TEXT], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(got[:, TEXT], _logits(_tf(x3, "vanilla", 2), IDS)[:, TEXT])


def test_vtw_left_padding_matches_unpadded():
    x3 = _x3()
    model = _tf(x3, "vtw", 2)
    pad = 3
    ids = IDS[:1]
    padded = torch.cat([torch.zeros(1, pad, dtype=torch.long), ids], dim=1)
    mask = torch.cat([torch.zeros(1, pad, dtype=torch.long), torch.ones_like(ids)], dim=1)

    # Image span sits at the same offset inside the real tokens.
    emb_fn = x3.llava_model.model.prepare_multimodal_input

    def shifted(input_ids, pixel_values=None):
        emb, m = emb_fn(input_ids, pixel_values)
        if m is None:
            return emb, m
        m2 = torch.zeros_like(m)
        m2[:, IMG.start + pad:IMG.stop + pad] = True
        return emb, m2

    plain = _logits(model, ids)
    x3.llava_model.model = types.SimpleNamespace(prepare_multimodal_input=shifted)
    got = _logits(model, padded, mask=mask)
    torch.testing.assert_close(got[:, pad:][:, TEXT], plain[:, TEXT], atol=1e-5, rtol=1e-5)


def test_generate_without_cache_feeds_image_each_step():
    x3 = _x3()
    model = _tf(x3, "vtw", 2)
    seen = []
    orig = model.forward

    def spy(*a, **k):
        seen.append(k.get("pixel_values") is not None)
        return orig(*a, **k)

    model.forward = spy
    ids = IDS[:1]
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                         pixel_values=torch.zeros(1), max_new_tokens=3, do_sample=False,
                         use_cache=False, pad_token_id=0, eos_token_id=VOCAB + 1)
    assert out.shape[1] == ids.shape[1] + 3
    assert seen and all(seen)


def test_invalid_split_rejected():
    x3 = _x3()
    with pytest.raises(ValueError):
        _tf(x3, "lf_no_branch", 4, fusion_count=2)
    with pytest.raises(ValueError):
        _tf(x3, "vtw", 0)

"""Cached decoding for DPVR-LF must reproduce the no-cache prefill exactly.

Reference semantics: the logits for token t+1 are those of a full prefill
(with the image) over the first t tokens. A cached decode step must return the
same logits while feeding only the new token.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.models.token_diversion_x3_fusion import (
    _DPVR_LF_TEXT_LEN,
    TokenDiversionX3FusionLlava,
)

IMG_START, IMG_END = 2, 8  # image tokens sit after a short system prefix, as in LLaVA


def _make(n_layers=5, split_layer=2, fusion_count=1, hidden_size=64, vocab=1000):
    torch.manual_seed(0)
    model = TokenDiversionX3FusionLlava._build_test_instance(
        hidden_size=hidden_size, n_layers=n_layers, n_heads=4, vocab=vocab,
        split_layer=split_layer, vision_depth=1, fusion_count=fusion_count,
    )
    embed = nn.Embedding(vocab, hidden_size)

    # Fixed image span independent of sequence length, so a longer reference
    # sequence has the image at the same positions as the prompt.
    def prepare_multimodal_input(input_ids, pixel_values=None):
        B, S = input_ids.shape
        inputs_embeds = embed(input_ids).float()
        if pixel_values is None:
            return inputs_embeds, None
        mask_2d = torch.zeros(B, S, dtype=torch.bool)
        mask_2d[:, IMG_START:IMG_END] = True
        return inputs_embeds, mask_2d.unsqueeze(-1).expand(-1, -1, hidden_size)

    model.llava_model.model = types.SimpleNamespace(
        prepare_multimodal_input=prepare_multimodal_input
    )
    model._embed = embed  # keep a reference so the embedding is part of the module
    model.eval()
    return model


def _full_logits(model, ids, pixel_values):
    mask = torch.ones_like(ids)
    with torch.no_grad():
        return model(input_ids=ids, pixel_values=pixel_values,
                     attention_mask=mask, use_cache=False).logits


@pytest.mark.parametrize("n_layers,split_layer,fusion_count", [
    (5, 2, 1),   # 2 text-only deep layers + 1 fusion layer
    (6, 2, 2),   # 2 text-only deep layers + 2 fusion layers
    (4, 2, 2),   # no text-only deep layer at all
])
def test_cached_decode_matches_full_prefill(n_layers, split_layer, fusion_count):
    model = _make(n_layers=n_layers, split_layer=split_layer, fusion_count=fusion_count)
    B, S, n_new, V = 2, 12, 5, 1000
    ids = torch.randint(0, V, (B, S + n_new))
    pixel_values = torch.randn(B, 3, 8, 8)

    ref = _full_logits(model, ids, pixel_values)

    with torch.no_grad():
        out = model(input_ids=ids[:, :S], pixel_values=pixel_values,
                    attention_mask=torch.ones(B, S, dtype=torch.long), use_cache=True)
        cache = out.past_key_values
        assert getattr(cache, _DPVR_LF_TEXT_LEN) == S - (IMG_END - IMG_START)
        torch.testing.assert_close(out.logits, ref[:, :S], atol=1e-4, rtol=1e-4)

        for k in range(n_new - 1):
            t = S + k
            step = model(
                input_ids=ids[:, t:t + 1],
                pixel_values=None,
                attention_mask=torch.ones(B, t + 1, dtype=torch.long),
                past_key_values=cache,
                cache_position=torch.tensor([t]),
                use_cache=True,
            )
            cache = step.past_key_values
            torch.testing.assert_close(step.logits[:, -1], ref[:, t], atol=1e-4, rtol=1e-4)


def test_text_only_layers_cache_text_only():
    model = _make(n_layers=5, split_layer=2, fusion_count=1)
    B, S = 1, 12
    ids = torch.randint(0, 1000, (B, S))
    with torch.no_grad():
        cache = model(input_ids=ids, pixel_values=torch.randn(B, 3, 8, 8),
                      attention_mask=torch.ones(B, S, dtype=torch.long),
                      use_cache=True).past_key_values
    n_txt = S - (IMG_END - IMG_START)
    assert cache.get_seq_length(0) == S        # shallow: image + text
    assert cache.get_seq_length(2) == n_txt    # text-only deep layer
    assert cache.get_seq_length(3) == n_txt
    assert cache.get_seq_length(4) == S        # fusion layer: image + text


def test_prepare_inputs_slices_only_with_dpvr_cache():
    model = _make()
    ids = torch.randint(0, 1000, (1, 10))
    pv = torch.randn(1, 3, 8, 8)

    first = model.prepare_inputs_for_generation(ids, pixel_values=pv,
                                                cache_position=torch.arange(10))
    assert first["input_ids"].shape[1] == 10
    assert first["pixel_values"] is pv

    cache = types.SimpleNamespace(**{_DPVR_LF_TEXT_LEN: 4})
    step = model.prepare_inputs_for_generation(ids, past_key_values=cache, pixel_values=pv,
                                               cache_position=torch.tensor([9]))
    assert step["input_ids"].shape[1] == 1
    assert "pixel_values" not in step

    no_cache = model.prepare_inputs_for_generation(ids, pixel_values=pv, use_cache=False,
                                                   cache_position=torch.tensor([9]))
    assert no_cache["input_ids"].shape[1] == 10
    assert no_cache["use_cache"] is False

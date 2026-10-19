"""Token-reduction baselines (C5) and DPVR-LF combinations (C4)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dpvr.models import token_reduction as tr  # noqa: E402
from dpvr.models.token_diversion_x3_fusion import TokenDiversionX3FusionLlava  # noqa: E402

IMG = slice(2, 8)
N_LAYERS, HIDDEN, VOCAB, S = 5, 64, 500, 14
IDS = torch.randint(0, VOCAB, (2, S), generator=torch.Generator().manual_seed(1))
TEXT = torch.ones(S, dtype=torch.bool)
TEXT[IMG] = False


def _x3(split_layer=2, fusion_count=1, seed=0):
    torch.manual_seed(seed)
    m = TokenDiversionX3FusionLlava._build_test_instance(
        hidden_size=HIDDEN, n_layers=N_LAYERS, n_heads=4, vocab=VOCAB,
        split_layer=split_layer, vision_depth=1, fusion_count=fusion_count,
    )
    embed = nn.Embedding(VOCAB, HIDDEN)

    def prepare_multimodal_input(input_ids, pixel_values=None):
        B, n = input_ids.shape
        emb = embed(input_ids).float()
        if pixel_values is None:
            return emb, None
        mask = torch.zeros(B, n, dtype=torch.bool)
        mask[:, IMG] = True
        return emb, mask.unsqueeze(-1).expand(-1, -1, HIDDEN)

    m.llava_model.model = types.SimpleNamespace(prepare_multimodal_input=prepare_multimodal_input)
    m._embed = embed
    return m.eval()


def _reduced(x3, reduction="none", params=None, lf=False):
    m = tr.ReducedLlava._from_parts(x3.llava_model, x3.config, reduction, params,
                                    lf_model=x3 if lf else None)
    m._weights = nn.ModuleList([x3.llava_model.language_model, x3.llava_model.lm_head, x3._embed])
    return m.eval()


def _out(model, ids=IDS, pv=True):
    with torch.no_grad():
        return model(input_ids=ids, pixel_values=torch.zeros(1) if pv else None,
                     attention_mask=torch.ones_like(ids))


TOL = {"atol": 2e-5, "rtol": 2e-5}


def test_none_is_vanilla():
    ref_model = _x3(split_layer=N_LAYERS - 1)
    ref_model.single_transformer = nn.ModuleList()  # identity branch at the last layer == vanilla
    ref = _out(ref_model).logits
    got = _out(_reduced(ref_model)).logits
    torch.testing.assert_close(got[:, TEXT], ref[:, TEXT], **TOL)


@pytest.mark.parametrize("split_layer,fusion_count", [(2, 1), (2, 2), (3, 1)])
def test_lf_none_matches_dpvr_lf(split_layer, fusion_count):
    x3 = _x3(split_layer, fusion_count)
    torch.testing.assert_close(_out(_reduced(x3, lf=True)).logits, _out(x3).logits, **TOL)


def test_lf_fastv_keep_all_matches_dpvr_lf():
    x3 = _x3(3, 1)
    got = _out(_reduced(x3, "fastv", {"layer": 1, "keep_ratio": 1.0}, lf=True)).logits
    torch.testing.assert_close(got, _out(x3).logits, **TOL)


def _masked_reference(x3, ids, prune_layer, kept_index):
    """Full sequence; from prune_layer on, pruned image keys are invisible."""
    lm = x3.llava_model.language_model
    emb, _ = x3.llava_model.model.prepare_multimodal_input(ids, torch.zeros(1))
    B, n, _ = emb.shape
    pe = lm.rotary_emb(emb, torch.arange(n).expand(B, -1))
    am = torch.ones(B, n, dtype=torch.bool)
    keep = torch.zeros(B, n, dtype=torch.bool)
    keep.scatter_(1, kept_index, True)
    full = tr.additive_causal_mask(am, emb.dtype)
    pruned = tr.additive_causal_mask(keep, emb.dtype)
    h = emb
    for i, layer in enumerate(lm.layers):
        h = tr._run(layer, h, pe, full if i < prune_layer else pruned)
    return x3.llava_model.lm_head(lm.norm(h))


@pytest.mark.parametrize("score", ["last", "mean"])
def test_fastv_equals_masking_pruned_keys(score):
    x3 = _x3(N_LAYERS - 1)
    out = _out(_reduced(x3, "fastv", {"layer": 2, "keep_ratio": 0.5, "score": score}))
    kept = out.kept_index
    assert kept.shape[1] == S - 3  # 6 image tokens -> 3
    with torch.no_grad():
        ref = _masked_reference(x3, IDS, 2, kept)
    ref_kept = ref.gather(1, kept[:, :, None].expand(-1, -1, ref.shape[-1]))
    text_rows = TEXT[kept[0]]
    torch.testing.assert_close(out.logits[:, text_rows], ref_kept[:, text_rows], **TOL)


def test_fastv_keeps_top_scored_image_tokens():
    x3 = _x3(N_LAYERS - 1)
    lm = x3.llava_model.language_model
    with torch.no_grad():
        emb, _ = x3.llava_model.model.prepare_multimodal_input(IDS, torch.zeros(1))
        am = torch.ones(2, S, dtype=torch.bool)
        pe = lm.rotary_emb(emb, torch.arange(S).expand(2, -1))
        h1 = tr._run(lm.layers[0], emb, pe, tr.additive_causal_mask(am, emb.dtype))
        scores = tr.image_scores(tr.attention_probs(lm.layers[1], h1, pe, am), am, "last")
        out = _out(_reduced(x3, "fastv", {"layer": 2, "keep_ratio": 0.5}))
    img_scores = scores[:, IMG]
    expect = img_scores.topk(3, dim=1).indices.sort(1).values + IMG.start
    got = out.kept_index[:, IMG.start:IMG.start + 3]
    assert torch.equal(got, expect)


def test_pdrop_stage_counts():
    x3 = _x3(N_LAYERS - 1)
    out = _out(_reduced(x3, "pdrop", {"layers": [2, 4], "keep_ratio": 0.5}))
    assert out.kept_index.shape[1] == S - 6 + 2  # 6 -> 3 -> round(1.5) = 2


def test_lf_pruning_after_split_rejected():
    x3 = _x3(2, 1)
    with pytest.raises(ValueError):
        _out(_reduced(x3, "fastv", {"layer": 3, "keep_ratio": 0.5}, lf=True))


def test_image_scores_rows():
    probs = torch.zeros(1, 2, 3, 3)
    probs[0, :, 0, 0] = 1
    probs[0, :, 1, :2] = 0.5
    probs[0, :, 2] = torch.tensor([0.2, 0.3, 0.5])
    am = torch.ones(1, 3, dtype=torch.bool)
    torch.testing.assert_close(tr.image_scores(probs, am, "last"), torch.tensor([[0.2, 0.3, 0.5]]))
    torch.testing.assert_close(tr.image_scores(probs, am, "mean"),
                               torch.tensor([[(1 + 0.5 + 0.2) / 3, (0.5 + 0.3) / 3, 0.5 / 3]]))


# ------------------------------------------------------------- vision side
def test_iqr_keep_count():
    attn = torch.cat([torch.full((1, 95), 0.01), torch.full((1, 5), 0.5)], dim=1)
    attn = attn + torch.linspace(0, 1e-4, 100)[None]
    assert tr.iqr_keep_count(attn) == 5


def test_prumerge_merges_weighted_neighbours():
    patches = torch.eye(4)
    cls_attn = torch.tensor([[0.7, 0.1, 0.15, 0.05]])
    keys = torch.tensor([[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]])
    out = tr.prumerge(patches[None], cls_attn, keys, n_keep=2, k_neighbors=1)
    # center 0 merges its nearest key (patch 1) weighted by its CLS attention 0.1;
    # center 2 merges patch 3 with weight 0.05.
    torch.testing.assert_close(out[0, 0], patches[0] + 0.1 * patches[1])
    torch.testing.assert_close(out[0, 1], patches[2] + 0.05 * patches[3])


def test_visionzip_counts_and_dominant_rows():
    torch.manual_seed(0)
    B, N, C = 1, 20, 8
    hidden = torch.randn(B, 1 + N, C)
    attn = torch.rand(B, 2, 1 + N, 1 + N).softmax(-1)
    keys = torch.randn(B, 1 + N, 2 * 4)
    out = tr.visionzip(hidden, attn, keys, dominant=5, contextual=3, keep_cls=True, key_heads=2)
    assert out.shape == (B, 1 + 5 + 3, C)
    top = attn[:, :, 0, 1:].sum(1).topk(5, dim=1).indices + 1
    idx = torch.cat([torch.zeros(1, 1, dtype=torch.long), top], 1).sort(1).values
    torch.testing.assert_close(out[:, :6], hidden[0, idx[0]][None])


def test_splice_image_tokens():
    embed = nn.Embedding(10, 4)
    ids = torch.tensor([[1, 2, 9, 9, 9, 9, 3, 4]])
    feats = torch.ones(1, 2, 4)
    emb, img = tr.splice_image_tokens(ids, embed, 9, feats)
    assert emb.shape == (1, 6, 4)
    assert img.tolist() == [[False, False, True, True, False, False]]
    torch.testing.assert_close(emb[0, 4], embed(torch.tensor(3)))
    torch.testing.assert_close(emb[0, 2:4], feats[0])


def test_splice_two_images():
    """Interleaved sample: two placeholder runs, each replaced by its own block."""
    embed = nn.Embedding(10, 4)
    ids = torch.tensor([[1, 9, 9, 2, 9, 9, 9, 3]])
    feats = [torch.full((3, 4), 0.5), torch.full((2, 4), 0.25)]
    emb, img = tr.splice_image_tokens(ids, embed, 9, feats)
    assert emb.shape == (1, 8, 4)  # 3 text + 3 + 2
    assert img.tolist() == [[False, True, True, True, False, True, True, False]]
    torch.testing.assert_close(emb[0, 1:4], feats[0])
    torch.testing.assert_close(emb[0, 4], embed(torch.tensor(2)))
    torch.testing.assert_close(emb[0, 5:7], feats[1])


def test_splice_rejects_image_run_mismatch():
    embed = nn.Embedding(10, 4)
    ids = torch.tensor([[1, 9, 9, 2, 9, 9, 3]])
    with pytest.raises(ValueError, match="placeholder runs"):
        tr.splice_image_tokens(ids, embed, 9, torch.ones(1, 2, 4))


def _tiny_llava():
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig

    from dpvr.models.modeling_llava import LlavaForConditionalGeneration

    cfg = LlavaConfig(
        vision_config=CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                       num_attention_heads=2, image_size=56, patch_size=14),
        text_config=LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                num_attention_heads=2, num_key_value_heads=2, vocab_size=128),
        image_token_index=100,
    )
    torch.manual_seed(0)
    return LlavaForConditionalGeneration(cfg).eval(), cfg


@pytest.mark.parametrize("reduction,params,n_img", [
    ("none", {}, 16),
    ("prumerge", {"n_keep": 4, "k_neighbors": 3}, 4),
    ("visionzip", {"dominant": 4, "contextual": 2}, 7),
    ("fastv", {"layer": 2, "keep_ratio": 0.5}, 8),
])
def test_end_to_end_tiny_llava(reduction, params, n_img):
    llava, cfg = _tiny_llava()
    model = tr.ReducedLlava._from_parts(llava, cfg, reduction, params)
    ids = torch.cat([torch.tensor([[1, 5]]), torch.full((1, 16), 100), torch.randint(3, 99, (1, 6))], 1)
    pv = torch.randn(1, 3, 56, 56)
    with torch.no_grad():
        out = model(input_ids=ids, pixel_values=pv, attention_mask=torch.ones_like(ids))
    assert out.logits.shape[1] == 2 + n_img + 6
    if reduction == "none":
        with torch.no_grad():
            ref = llava(input_ids=ids, pixel_values=pv, attention_mask=torch.ones_like(ids)).logits
        torch.testing.assert_close(out.logits, ref, atol=2e-5, rtol=2e-5)
    gen = model.generate(input_ids=ids, pixel_values=pv, attention_mask=torch.ones_like(ids),
                         max_new_tokens=2, do_sample=False, use_cache=False, pad_token_id=0,
                         eos_token_id=127)
    assert gen.shape[1] == ids.shape[1] + 2

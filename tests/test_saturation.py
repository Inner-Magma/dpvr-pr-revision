"""C6: model-agnostic saturation probes on a tiny random LLaVA."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dpvr.analysis import saturation as sat

IMG = 100


@pytest.fixture(scope="module")
def tiny():
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

    torch.manual_seed(0)
    cfg = LlavaConfig(
        vision_config=CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                       num_attention_heads=2, image_size=56, patch_size=14),
        text_config=LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                num_attention_heads=2, num_key_value_heads=2, vocab_size=128),
        image_token_index=IMG,
    )
    cfg._attn_implementation = "eager"
    model = LlavaForConditionalGeneration(cfg).float().eval()
    ids = torch.tensor([[1, 5, 6] + [IMG] * 16 + [7, 8, 9, 10]])
    enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
           "pixel_values": torch.randn(1, 3, 56, 56)}
    return model, sat.locate_parts(model), enc


def test_locate_parts(tiny):
    model, parts, _ = tiny
    assert parts.n_layers == 3 and parts.image_token_id == IMG
    assert parts.lm_head is model.lm_head


def test_token_masks():
    ids = torch.tensor([[1, IMG, IMG, 4, 5]])
    img, txt, query = sat.token_masks(ids, None, IMG)
    assert img.tolist() == [False, True, True, False, False]
    assert txt.tolist() == [True, False, False, True, True]
    assert query.tolist() == [False, False, False, True, True]
    with pytest.raises(ValueError):
        sat.token_masks(torch.tensor([[1, 2]]), None, IMG)


def test_metrics_match_reference(tiny):
    model, parts, enc = tiny
    res = sat.analyze_sample(model, parts, enc)
    with torch.no_grad():
        out = model(**enc, use_cache=False, output_attentions=True, output_hidden_states=True)
    img, txt, query = sat.token_masks(enc["input_ids"], None, IMG)
    # HF hidden_states[L] is post-norm, so compare the pre-norm layers only.
    hs = out.hidden_states
    for l in range(parts.n_layers - 1):
        ref = F.cosine_similarity(hs[l][0][img], hs[l + 1][0][img], dim=-1).mean()
        assert res["cos_img"][l] == pytest.approx(float(ref), abs=1e-5)
        a, b = hs[l][0][txt], hs[l + 1][0][txt]
        assert res["resid_txt"][l] == pytest.approx(float(((b - a).norm(dim=-1) / a.norm(dim=-1)).mean()), rel=1e-4)
    for l, w in enumerate(out.attentions):
        mass = w[0][:, :, img].sum(-1)
        assert res["attn"][l] == pytest.approx(float(mass[:, query].mean()), abs=1e-5)
        assert res["attn_last"][l] == pytest.approx(float(mass[:, -1].mean()), abs=1e-5)
    assert np.all((res["attn"] >= 0) & (res["attn"] <= 1))
    # logit lens: the last layer is the reference distribution
    assert res["kl_img"][-1] == pytest.approx(0.0, abs=1e-5)
    assert np.all(res["kl_txt"] >= -1e-6)
    assert res["n_img_tokens"][0] == 16


def test_sdpa_without_weights_raises(tiny):
    model, parts, enc = tiny
    model.config._attn_implementation = "sdpa"
    model.model.language_model.config._attn_implementation = "sdpa"
    try:
        with pytest.raises(RuntimeError, match="eager"):
            sat.analyze_sample(model, parts, enc)
        res = sat.analyze_sample(model, parts, enc, attention=False)
        assert "attn" not in res and len(res["cos_img"]) == 3
    finally:
        model.config._attn_implementation = "eager"
        model.model.language_model.config._attn_implementation = "eager"


def test_grid_and_pooling_helpers():
    assert sat.parse_grid("12x24") == (12, 24)
    assert sat.parse_grid("144") == (12, 12)
    assert sat.parse_grid(None) is None
    with pytest.raises(ValueError):
        sat.parse_grid("288")
    feats = torch.arange(16.0).reshape(16, 1)          # 4x4 grid, value = index
    pooled = sat.pool_features(feats, (2, 2))
    assert pooled.flatten().tolist() == [2.5, 4.5, 10.5, 12.5]
    ids = torch.tensor([[1, IMG, IMG, IMG, IMG, 2]])
    assert sat.collapse_image_run(ids, IMG, 2).tolist() == [[1, IMG, IMG, 2]]


def test_pooled_identity_matches_standard(tiny):
    model, parts, enc = tiny
    full = sat.analyze_sample(model, parts, enc)
    same = sat.analyze_sample(model, parts, sat.pool_image_tokens(model, parts, enc, (4, 4)))
    for k in sat.METRICS:
        np.testing.assert_allclose(full[k], same[k], rtol=1e-4, atol=1e-5)
    small = sat.pool_image_tokens(model, parts, enc, (2, 2))
    assert int((small["input_ids"] == IMG).sum()) == 4
    res = sat.analyze_sample(model, parts, small)
    assert res["n_img_tokens"][0] == 4


def test_transitions_and_summary():
    assert sat.detect_transition(np.array([4.0, 3.0, 1.5, 1.0, 0.0])) == 2
    assert sat.plateau_layer(np.array([0.8, 0.96, 0.9, 0.97, 0.99]), 0.95) == 3
    rng = np.random.default_rng(0)
    kl = np.linspace(4, 0, 8)[None, :] + rng.random((5, 8)) * 0.01
    s = sat.summarize({"kl_img": kl, "cos_img": np.full((5, 8), 0.97)})
    assert s["transitions"]["kl_img"]["median_curve"] == 4
    assert s["transitions"]["cos_img"]["plateau_from"] == 0
    assert s["n_layers"] == 8 and s["normalized"]["kl_img"] == 0.5
    assert len(s["per_layer"]["kl_img"]["median"]) == 8

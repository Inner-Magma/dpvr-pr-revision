"""CPU tests for DPVR-LF on Qwen2.5-VL with per-sample image token counts (C8).

A tiny random Qwen2.5-VL (5 layers, 64 hidden) built from a config, no download.

 1. route=False reproduces the stock HF forward exactly (same layers, masks, M-RoPE).
 2. Two samples with different image sizes give the same logits batched
    (left or right padded) as alone: packing and padding masks are correct.
 3. With routing, text tokens before the fusion layers do not depend on the image
    except through layers 0..s-1 (the deep text path never reads image positions).
 4. Only the side branch receives gradients.
 5. generate() runs without a KV cache and matches step-by-step argmax.
 6. branch_init=copy starts the branch from layer s.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

transformers = pytest.importorskip("transformers")
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration  # noqa: E402

from dpvr.models.token_diversion_x3_fusion_qwen import TokenDiversionX3FusionQwen25VL  # noqa: E402

IMG, VID, VS, VE, PAD = 10, 11, 12, 13, 0
N_LAYERS = 5


def _tiny_qwen(attn: str = "eager") -> Qwen2_5_VLForConditionalGeneration:
    cfg = Qwen2_5_VLConfig(
        text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=N_LAYERS,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=200,
                         max_position_embeddings=512, tie_word_embeddings=False,
                         rope_scaling={"type": "mrope", "mrope_section": [4, 2, 2]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=64, patch_size=14, spatial_merge_size=2,
                           temporal_patch_size=2, window_size=112, fullatt_block_indexes=[0],
                           in_channels=3),
        image_token_id=IMG, video_token_id=VID, vision_start_token_id=VS, vision_end_token_id=VE,
        pad_token_id=PAD,
    )
    for c in (cfg, cfg.text_config, cfg.vision_config):
        c._attn_implementation = attn
    torch.manual_seed(0)
    return Qwen2_5_VLForConditionalGeneration(cfg).eval()


def _dpvr(split_layer=2, fusion_count=1, attn="eager", **kw) -> TokenDiversionX3FusionQwen25VL:
    return TokenDiversionX3FusionQwen25VL(split_layer=split_layer, fusion_count=fusion_count,
                                          qwen_model=_tiny_qwen(attn), **kw).eval()


def _sample(h: int, w: int, n_pre: int = 3, n_post: int = 4, seed: int = 0):
    """One image of h x w patches (h*w/4 LLM tokens) between two text spans."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.cat([torch.randint(20, 200, (n_pre,), generator=g), torch.tensor([VS]),
                     torch.full((h * w // 4,), IMG), torch.tensor([VE]),
                     torch.randint(20, 200, (n_post,), generator=g)])
    pix = torch.randn(h * w, 3 * 2 * 14 * 14, generator=g)
    return ids, pix, torch.tensor([[1, h, w]])


def _batch(samples, side: str):
    S = max(len(s[0]) for s in samples)
    ids = torch.full((len(samples), S), PAD)
    att = torch.zeros(len(samples), S, dtype=torch.long)
    for i, (x, _, _) in enumerate(samples):
        sl = slice(S - len(x), S) if side == "left" else slice(0, len(x))
        ids[i, sl] = x
        att[i, sl] = 1
    pix = torch.cat([s[1] for s in samples])
    thw = torch.cat([s[2] for s in samples])
    return ids, att, pix, thw


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_unrouted_matches_hf_forward(attn):
    m = _dpvr(attn=attn, route=False)
    ids, pix, thw = _sample(4, 6)
    with torch.no_grad():
        ref = m.qwen(input_ids=ids[None], pixel_values=pix, image_grid_thw=thw).logits
        out = m(input_ids=ids[None], pixel_values=pix, image_grid_thw=thw).logits
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("fusion_count", [1, 2])
def test_variable_image_tokens_batch_invariant(side, fusion_count):
    m = _dpvr(fusion_count=fusion_count)
    samples = [_sample(4, 4, seed=1), _sample(4, 8, n_post=2, seed=2), _sample(6, 4, n_pre=1, seed=3)]
    ids, att, pix, thw = _batch(samples, side)
    assert len({int((r == IMG).sum()) for r in ids}) == 3  # 4, 8 and 6 image tokens
    with torch.no_grad():
        batched = m(input_ids=ids, attention_mask=att, pixel_values=pix, image_grid_thw=thw).logits
        for i, (x, p, t) in enumerate(samples):
            alone = m(input_ids=x[None], pixel_values=p, image_grid_thw=t).logits[0]
            got = batched[i][att[i].bool()]
            torch.testing.assert_close(got, alone, rtol=1e-4, atol=1e-4)


def test_deep_text_path_ignores_image_positions():
    """Fusion layers only at the end: perturbing image hidden states after the
    split changes nothing before the fusion stack. Checked by hooking the input
    of the first fusion layer at text positions."""
    m = _dpvr(split_layer=2, fusion_count=1)
    ids, pix, thw = _sample(4, 4)
    captured = []
    fusion = m.qwen.model.language_model.layers[N_LAYERS - 1]
    hook = fusion.register_forward_pre_hook(lambda mod, args: captured.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            m(input_ids=ids[None], pixel_values=pix, image_grid_thw=thw)
            for p in m.single_transformer.parameters():
                p.add_(torch.randn_like(p))
            m(input_ids=ids[None], pixel_values=pix, image_grid_thw=thw)
    finally:
        hook.remove()
    text = ids != IMG
    torch.testing.assert_close(captured[0][0][text], captured[1][0][text])
    assert not torch.allclose(captured[0][0][~text], captured[1][0][~text])


def test_only_side_branch_gets_gradients():
    m = _dpvr().train()
    samples = [_sample(4, 4, seed=1), _sample(4, 8, seed=2)]
    ids, att, pix, thw = _batch(samples, "right")
    labels = torch.where(att.bool() & (ids != IMG), ids, torch.full_like(ids, -100))
    out = m(input_ids=ids, attention_mask=att, pixel_values=pix, image_grid_thw=thw, labels=labels)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    for name, p in m.named_parameters():
        if name.startswith("single_transformer."):
            assert p.grad is not None and p.grad.abs().sum() > 0, name
        else:
            assert not p.requires_grad and p.grad is None, name


def test_generate_without_cache_is_greedy_recompute():
    m = _dpvr()
    m.generation_config.pad_token_id = PAD
    ids, pix, thw = _sample(4, 4)
    with torch.no_grad():
        gen = m.generate(input_ids=ids[None], attention_mask=torch.ones_like(ids)[None],
                         pixel_values=pix, image_grid_thw=thw, max_new_tokens=3,
                         do_sample=False, use_cache=False)
        seq = ids[None]
        for _ in range(3):
            nxt = m(input_ids=seq, pixel_values=pix, image_grid_thw=thw).logits[:, -1].argmax(-1)
            seq = torch.cat([seq, nxt[:, None]], dim=1)
    assert torch.equal(gen, seq)


def test_branch_init_copy():
    m = _dpvr(split_layer=3, branch_init="copy")
    src = m.qwen.model.language_model.layers[3].state_dict()
    for k, v in m.single_transformer[0].state_dict().items():
        assert torch.equal(v, src[k]), k


def test_rejects_bad_split():
    with pytest.raises(ValueError):
        _dpvr(split_layer=4, fusion_count=2)


def test_checkpoint_roundtrip_through_load_trained(tmp_path):
    """Trainer-style save (save_pretrained + training_config.json) loads back via
    checkpoint_io.load_trained with no missing keys and the trained branch."""
    import json

    from dpvr.models import checkpoint_io
    from dpvr.utils import PathConfig

    _tiny_qwen("sdpa").save_pretrained(tmp_path / "models" / "tiny-qwen")
    cfg = {"model": {"base_model": "tiny-qwen", "attn_implementation": "sdpa"},
           "method": {"name": "token_diversion_x3_fusion_qwen25vl", "split_layer": 2, "fusion_count": 1}}
    paths = PathConfig(data_root=tmp_path, model_root=tmp_path / "models", ckpt_root=tmp_path)
    from dpvr.utils import resolve_paths

    model = TokenDiversionX3FusionQwen25VL.from_config(resolve_paths(cfg, paths))
    with torch.no_grad():
        for p in model.single_transformer.parameters():
            p.add_(0.5)
    ck = tmp_path / "ckpt"
    model.save_pretrained(ck)
    (ck / "training_config.json").write_text(json.dumps(cfg))

    loaded, report = checkpoint_io.load_trained(ck, paths)
    assert isinstance(loaded, TokenDiversionX3FusionQwen25VL)
    assert report.missing == [] and report.unexpected == []
    for (n, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b), n

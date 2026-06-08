"""CPU smoke tests for TokenDiversionX3FusionLlava (X3-fusion-last variant).

5 mandatory tests verifying the design contract:
 1. test_grad_flows_to_single_transformer       — sole gradient path through layer end-1
 2. test_forward_shape_no_nan                   — output shape + numerical sanity
 3. test_text_only_deep_independence            — deep 0..end-2 layers ignore image
 4. test_layer_31_uses_out_image                — last layer's image K/V comes from out_image
 5. test_compute_saving_vs_method_a             — < 80% of Method A deep token-layer FLOPs

Plus 3 supplementary tests (loss, split-layer boundaries, decoding fallback).

Run:
    pytest tests/test_token_diversion_x3_fusion.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.models.token_diversion import TokenDiversionLlava
from dpvr.models.token_diversion_x3_fusion import TokenDiversionX3FusionLlava


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_x3(**kwargs) -> TokenDiversionX3FusionLlava:
    defaults = dict(hidden_size=64, n_layers=4, n_heads=4, vocab=1000,
                    split_layer=2, vision_depth=1)
    defaults.update(kwargs)
    return TokenDiversionX3FusionLlava._build_test_instance(**defaults)


def _make_a(**kwargs) -> TokenDiversionLlava:
    """Build a Method A test instance with the same shape as X3 for comparison.

    TokenDiversionLlava lacks a public _build_test_instance — recreate it here
    using the same mock-LLaVA pattern as B-2 / X3-fusion's helpers.
    """
    import types
    import torch.nn as nn
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from dpvr.models.modeling_llama import LlamaModel

    defaults = dict(hidden_size=64, n_layers=4, n_heads=4, vocab=1000,
                    split_layer=2, vision_depth=1)
    defaults.update(kwargs)
    hidden_size = defaults["hidden_size"]
    n_layers = defaults["n_layers"]
    n_heads = defaults["n_heads"]
    vocab = defaults["vocab"]
    split_layer = defaults["split_layer"]
    vision_depth = defaults["vision_depth"]

    llama_cfg = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        vocab_size=vocab,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        rope_scaling=None,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
    )
    obj = TokenDiversionLlava.__new__(TokenDiversionLlava)
    nn.Module.__init__(obj)
    obj.config = llama_cfg
    obj.name_or_path = ""
    obj._is_hf_initialized = True

    lm = LlamaModel(llama_cfg)
    embed = nn.Embedding(vocab, hidden_size)
    lm_head = nn.Linear(hidden_size, vocab, bias=False)

    def _mock_prepare_multimodal_input(input_ids, pixel_values=None):
        B, S = input_ids.shape
        inputs_embeds = embed(input_ids).float()
        if pixel_values is not None:
            T_img = S // 2
            mask_2d = torch.zeros(B, S, dtype=torch.bool, device=input_ids.device)
            mask_2d[:, :T_img] = True
            mask_3d = mask_2d.unsqueeze(-1).expand(-1, -1, hidden_size)
            return inputs_embeds, mask_3d
        return inputs_embeds, None

    llava_config_ns = types.SimpleNamespace(vocab_size=vocab)
    llava_mm_ns = types.SimpleNamespace(prepare_multimodal_input=_mock_prepare_multimodal_input)
    llava_model_ns = types.SimpleNamespace(
        config=llava_config_ns,
        lm_head=lm_head,
        language_model=lm,
        model=llava_mm_ns,
    )
    obj.llava_model = llava_model_ns
    obj.split_layer = split_layer
    obj.vision_depth = vision_depth
    obj.single_transformer = nn.ModuleList([
        LlamaDecoderLayer(llama_cfg, layer_idx=split_layer + i)
        for i in range(vision_depth)
    ])
    return obj


def _toy_batch(B=1, S=16, V=1000):
    input_ids = torch.randint(0, V, (B, S))
    pixel_values = torch.randn(B, 3, 64, 64)
    attention_mask = torch.ones(B, S, dtype=torch.long)
    return input_ids, pixel_values, attention_mask


# ── 5 MANDATORY TESTS ─────────────────────────────────────────────────────────

def test_grad_flows_to_single_transformer():
    """X3-fusion-last core contract: single_transformer must receive non-zero grad
    via the SOLE attention path through layer (end-1)'s text→image attention."""
    model = _make_x3(n_layers=4, split_layer=2)  # 2 deep layers: 1 text-only + 1 fusion

    B, S, V = 1, 16, 1000
    input_ids, pixel_values, attention_mask = _toy_batch(B, S, V)
    labels = torch.randint(0, V, (B, S))

    model.zero_grad()
    out = model(input_ids=input_ids, pixel_values=pixel_values,
                attention_mask=attention_mask, labels=labels)
    out.loss.backward()

    grad_q = model.single_transformer[0].self_attn.q_proj.weight.grad
    assert grad_q is not None, "single_transformer q_proj.weight.grad is None"
    assert grad_q.norm().item() > 0, (
        f"single_transformer q_proj grad is zero! norm={grad_q.norm().item()}. "
        "Layer (end-1) must overwrite hidden[image_pos] = out_image so its K/V "
        "projection on image positions sees single_transformer output."
    )

    grad_mlp = model.single_transformer[0].mlp.up_proj.weight.grad
    assert grad_mlp is not None and grad_mlp.norm().item() > 0, (
        "single_transformer mlp.up_proj grad is zero!"
    )


def test_forward_shape_no_nan():
    """Forward output shape correct; logits are finite."""
    model = _make_x3()
    model.eval()

    B, S, V = 2, 20, 1000
    input_ids, pixel_values, attention_mask = _toy_batch(B, S, V)

    with torch.no_grad():
        out = model(input_ids=input_ids, pixel_values=pixel_values,
                    attention_mask=attention_mask)

    assert out.logits is not None
    assert out.logits.shape == (B, S, V), f"got {out.logits.shape}"
    assert not torch.isnan(out.logits).any(), "logits contain NaN"
    assert not torch.isinf(out.logits).any(), "logits contain Inf"


def test_text_only_deep_independence():
    """Deep layers split_layer..end-2 are TEXT-ONLY: their output (text_h before
    layer end-1) must NOT depend on out_image.

    Strategy: capture text_h after deep loop in two forwards — one with normal
    out_image, one with out_image patched to garbage. The two text_h outputs
    must be identical (deep doesn't see out_image).

    Implementation: monkey-patch the X3 forward to expose text_h via a sidecar,
    or simpler: zero-out single_transformer (forces out_image = different value)
    and compare text_h. We use the cleanest path — directly compare against
    a re-run with single_transformer randomized.
    """
    model = _make_x3(n_layers=5, split_layer=2)  # 3 deep layers: 2 text-only + 1 fusion
    model.eval()

    B, S, V = 1, 16, 1000
    input_ids, pixel_values, attention_mask = _toy_batch(B, S, V)
    T_img = S // 2
    T_txt = S - T_img

    captured = {}

    # Hook on layer end-2 (the LAST text-only deep layer): its output IS text_h
    # right before the fusion layer end-1.
    n_layers = len(model.llava_model.language_model.layers)
    last_text_only_layer = model.llava_model.language_model.layers[n_layers - 2]

    def _capture(_module, _inputs, output):
        out_tensor = output[0] if isinstance(output, (tuple, list)) else output
        captured["text_h"] = out_tensor.detach().clone()

    handle = last_text_only_layer.register_forward_hook(_capture)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, pixel_values=pixel_values,
                  attention_mask=attention_mask)
        text_h_normal = captured["text_h"]

        # Randomize single_transformer → out_image changes drastically
        for p in model.single_transformer.parameters():
            torch.nn.init.normal_(p, std=10.0)

        with torch.no_grad():
            model(input_ids=input_ids, pixel_values=pixel_values,
                  attention_mask=attention_mask)
        text_h_perturbed = captured["text_h"]
    finally:
        handle.remove()

    # text_h must be the SAME shape (B, T_txt, H) — confirms text-only path
    assert text_h_normal.shape == text_h_perturbed.shape
    assert text_h_normal.shape[1] == T_txt, (
        f"text-only deep should process T_txt={T_txt} tokens, "
        f"got {text_h_normal.shape[1]}"
    )

    # text_h must be IDENTICAL — deep text-only forward never sees out_image
    diff = (text_h_normal - text_h_perturbed).abs().max().item()
    assert diff < 1e-4, (
        f"deep text-only output changed when out_image perturbed: max_diff={diff}. "
        "Deep layers must not depend on image content."
    )


def test_layer_31_uses_out_image():
    """Layer end-1 (the fusion layer) must consume out_image. Verify by zeroing
    single_transformer weights and checking that text-position logits change
    (image K/V at layer end-1 changes → text Q's attention output changes)."""
    model = _make_x3(n_layers=4, split_layer=2)
    model.eval()

    B, S, V = 1, 16, 1000
    T_img = S // 2
    input_ids, pixel_values, attention_mask = _toy_batch(B, S, V)

    with torch.no_grad():
        out_normal = model(input_ids=input_ids, pixel_values=pixel_values,
                           attention_mask=attention_mask)

    # Zero out single_transformer → out_image becomes ~the input image features
    # passed through a dead layer (residual passes through). This changes
    # what layer end-1 sees at image positions.
    zeroed_state = {
        k: torch.zeros_like(v) for k, v in model.single_transformer.state_dict().items()
    }
    model.single_transformer.load_state_dict(zeroed_state)

    with torch.no_grad():
        out_zeroed = model(input_ids=input_ids, pixel_values=pixel_values,
                           attention_mask=attention_mask)

    # Text positions' logits must differ — text Q at layer end-1 attends to
    # image K/V which depends on out_image (now changed).
    text_diff = (out_normal.logits[:, T_img:, :] - out_zeroed.logits[:, T_img:, :]).abs().max().item()
    assert text_diff > 0.0, (
        f"text logits unchanged when single_transformer zeroed: max_diff={text_diff}. "
        "Layer end-1 must use out_image as image K/V source."
    )

    # Image positions also change because layer end-1 image_pos input is
    # overwritten with out_image (different) → image output differs.
    img_diff = (out_normal.logits[:, :T_img, :] - out_zeroed.logits[:, :T_img, :]).abs().max().item()
    assert img_diff > 0.0, "image logits unchanged when single_transformer zeroed"


def test_compute_saving_vs_method_a():
    """Total deep-layer token-FLOPs proxy must be < 80% of Method A's.

    Counts (tokens × layer_calls) summed across all decoder layers, not just deep.
    Per-token compute is dominated by attention (linear in seq len) and FFN
    (linear in seq len), so token-layer count is a faithful proxy for FLOPs at
    fixed hidden_size.

    Method A: 14 deep layers × T_full tokens
    X3:       (deep-1) × T_txt tokens + 1 × T_full tokens

    With T_img = T_full/2, T_txt = T_full/2:
        X3 deep / A deep ≈ ((deep-1)*0.5 + 1) / deep
        For deep=14: (13*0.5 + 1)/14 = 7.5/14 ≈ 0.536 → ~46% saving.
    """
    n_layers = 16  # large enough that deep saving is meaningful
    split_layer = 2
    a_model = _make_a(n_layers=n_layers, split_layer=split_layer)
    x3_model = _make_x3(n_layers=n_layers, split_layer=split_layer)
    a_model.eval()
    x3_model.eval()

    B, S, V = 1, 32, 1000  # T_img = 16, T_txt = 16
    input_ids, pixel_values, attention_mask = _toy_batch(B, S, V)

    a_token_count = 0
    x3_token_count = 0

    def _make_counter(target):
        def _hook(_mod, inputs):
            x = inputs[0]
            target.append(x.shape[0] * x.shape[1])
        return _hook

    a_buf, x3_buf = [], []
    a_handles = [
        layer.register_forward_pre_hook(_make_counter(a_buf))
        for layer in a_model.llava_model.language_model.layers
    ]
    x3_handles = [
        layer.register_forward_pre_hook(_make_counter(x3_buf))
        for layer in x3_model.llava_model.language_model.layers
    ]

    try:
        with torch.no_grad():
            a_model(input_ids=input_ids, pixel_values=pixel_values,
                    attention_mask=attention_mask)
            x3_model(input_ids=input_ids, pixel_values=pixel_values,
                     attention_mask=attention_mask)
    finally:
        for h in a_handles + x3_handles:
            h.remove()

    a_token_count = sum(a_buf)
    x3_token_count = sum(x3_buf)

    # Restrict to deep layers: shallow stack contributes equally.
    deep_a = sum(a_buf[split_layer:])
    deep_x3 = sum(x3_buf[split_layer:])

    ratio = deep_x3 / deep_a
    assert ratio < 0.8, (
        f"X3 deep token-FLOPs ratio = {ratio:.3f} (deep_x3={deep_x3}, "
        f"deep_a={deep_a}). Expected < 0.8 (saving > 20%)."
    )
    # Sanity: total saving should be visible too
    total_ratio = x3_token_count / a_token_count
    assert total_ratio < 0.95, (
        f"Total token-FLOPs ratio {total_ratio:.3f} too close to 1 — "
        "X3 should save deep compute."
    )


# ── SUPPLEMENTARY TESTS ───────────────────────────────────────────────────────

def test_loss_computation():
    """Forward with labels returns a finite scalar loss."""
    model = _make_x3()
    model.eval()
    B, S, V = 1, 16, 1000
    input_ids, pixel_values, attention_mask = _toy_batch(B, S, V)
    labels = torch.randint(0, V, (B, S))

    with torch.no_grad():
        out = model(input_ids=input_ids, pixel_values=pixel_values,
                    attention_mask=attention_mask, labels=labels)

    assert out.loss is not None
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss), f"loss not finite: {out.loss.item()}"


def test_decoding_fallback_no_image():
    """pixel_values=None → vanilla deep forward (decoding fallback). No crash."""
    model = _make_x3()
    model.eval()
    B, S, V = 1, 16, 1000
    input_ids, _, attention_mask = _toy_batch(B, S, V)

    with torch.no_grad():
        out = model(input_ids=input_ids, pixel_values=None,
                    attention_mask=attention_mask)
    assert out.logits.shape == (B, S, V)
    assert not torch.isnan(out.logits).any()


def test_split_layer_boundary():
    """Various split_layer values must not crash."""
    B, S, V = 1, 8, 1000
    input_ids, pixel_values, attention_mask = _toy_batch(B, S, V)

    for split, n in [(1, 4), (2, 5), (3, 4)]:
        model = _make_x3(split_layer=split, n_layers=n)
        model.eval()
        with torch.no_grad():
            out = model(input_ids=input_ids, pixel_values=pixel_values,
                        attention_mask=attention_mask)
        assert out.logits.shape == (B, S, V), (
            f"split={split}, n={n}: got {out.logits.shape}"
        )


# ── fusion_count=2 ABLATION TESTS ─────────────────────────────────────────────

def test_fusion_count_2_correctness():
    """fusion_count=2 forward returns correct shape with no NaN/Inf."""
    # n_layers=5, split_layer=2 → 3 deep layers → 1 text-only + 2 fusion
    model = _make_x3(n_layers=5, split_layer=2, fusion_count=2)
    model.eval()

    B, S, V = 2, 20, 1000
    input_ids, pixel_values, attention_mask = _toy_batch(B, S, V)

    with torch.no_grad():
        out = model(input_ids=input_ids, pixel_values=pixel_values,
                    attention_mask=attention_mask)

    assert out.logits.shape == (B, S, V), f"got {out.logits.shape}"
    assert not torch.isnan(out.logits).any(), "logits contain NaN"
    assert not torch.isinf(out.logits).any(), "logits contain Inf"

    # Decoding fallback (no image) must also work with fusion_count=2.
    input_ids_d, _, attention_mask_d = _toy_batch(B, S, V)
    with torch.no_grad():
        out_d = model(input_ids=input_ids_d, pixel_values=None,
                      attention_mask=attention_mask_d)
    assert out_d.logits.shape == (B, S, V)
    assert not torch.isnan(out_d.logits).any()


def test_fusion_count_2_grad():
    """fusion_count=2: single_transformer still receives non-zero gradient.

    Gradient now flows through 2 fusion layers' attention rather than 1, so
    the magnitude should be at least as large as fusion_count=1 (often
    larger — the K=2 case is the motivation for the ablation)."""
    B, S, V = 1, 16, 1000
    input_ids, pixel_values, attention_mask = _toy_batch(B, S, V)
    labels = torch.randint(0, V, (B, S))

    torch.manual_seed(0)
    model_k1 = _make_x3(n_layers=5, split_layer=2, fusion_count=1)
    torch.manual_seed(0)
    model_k2 = _make_x3(n_layers=5, split_layer=2, fusion_count=2)

    def _run(m):
        m.zero_grad()
        out = m(input_ids=input_ids, pixel_values=pixel_values,
                attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        return m.single_transformer[0].self_attn.q_proj.weight.grad.norm().item()

    g_k1 = _run(model_k1)
    g_k2 = _run(model_k2)

    assert g_k2 > 0, f"fusion_count=2 grad is zero! got {g_k2}"
    # K=2 has 2 attention paths back to single_transformer; K=1 has 1.
    # Magnitudes can differ but both must be strictly positive and finite.
    assert g_k1 > 0


def test_fusion_count_2_compute():
    """fusion_count=2 deep token-FLOPs must lie strictly between fusion_count=1
    (lowest) and Method A (highest), confirming the correct trade-off."""
    n_layers = 16
    split_layer = 2

    a_model = _make_a(n_layers=n_layers, split_layer=split_layer)
    x3_k1 = _make_x3(n_layers=n_layers, split_layer=split_layer, fusion_count=1)
    x3_k2 = _make_x3(n_layers=n_layers, split_layer=split_layer, fusion_count=2)
    for m in (a_model, x3_k1, x3_k2):
        m.eval()

    B, S, V = 1, 32, 1000  # T_img = 16, T_txt = 16
    input_ids, pixel_values, attention_mask = _toy_batch(B, S, V)

    def _measure(model):
        buf = []

        def _hook(_mod, inputs):
            x = inputs[0]
            buf.append(x.shape[0] * x.shape[1])

        handles = [
            layer.register_forward_pre_hook(_hook)
            for layer in model.llava_model.language_model.layers
        ]
        try:
            with torch.no_grad():
                model(input_ids=input_ids, pixel_values=pixel_values,
                      attention_mask=attention_mask)
        finally:
            for h in handles:
                h.remove()
        return sum(buf), sum(buf[split_layer:])

    _, deep_a = _measure(a_model)
    _, deep_k1 = _measure(x3_k1)
    _, deep_k2 = _measure(x3_k2)

    assert deep_k1 < deep_k2 < deep_a, (
        f"deep token-FLOPs ordering wrong: K=1 ({deep_k1}) < K=2 ({deep_k2}) "
        f"< A ({deep_a}) expected."
    )
    # Saving should still be meaningful for K=2.
    ratio_k2 = deep_k2 / deep_a
    assert ratio_k2 < 0.85, (
        f"K=2 deep token-FLOPs ratio {ratio_k2:.3f} too high — "
        "should still save vs Method A."
    )


def test_fusion_count_default_matches_last1():
    """fusion_count default (=1) reproduces last-1 behavior bit-exactly."""
    torch.manual_seed(42)
    model_default = _make_x3(n_layers=4, split_layer=2)
    torch.manual_seed(42)
    model_explicit = _make_x3(n_layers=4, split_layer=2, fusion_count=1)

    # Force matching weights via state_dict copy
    model_explicit.load_state_dict(model_default.state_dict())
    model_default.eval()
    model_explicit.eval()

    B, S, V = 1, 12, 1000
    input_ids, pixel_values, attention_mask = _toy_batch(B, S, V)
    with torch.no_grad():
        out_d = model_default(input_ids=input_ids, pixel_values=pixel_values,
                              attention_mask=attention_mask)
        out_e = model_explicit(input_ids=input_ids, pixel_values=pixel_values,
                               attention_mask=attention_mask)

    diff = (out_d.logits - out_e.logits).abs().max().item()
    assert diff < 1e-5, f"default vs fusion_count=1 mismatch: max_diff={diff}"

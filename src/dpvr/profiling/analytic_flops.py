"""Closed-form prefill FLOPs for LLaVA-1.5 and the DPVR compute graphs (C7, R1.7).

Counts matrix multiplies only (2 FLOPs per multiply-add), the same quantity
``torch.utils.flop_counter.FlopCounterMode`` reports, so the two can be
checked against each other. Norms, RoPE, activations and softmax are ignored;
they are a fraction of a percent at these sizes.

Attention is counted two ways:
- ``dense``: every query against every key in its block (what a masked SDPA
  or eager kernel computes, and what FlopCounterMode counts);
- ``causal``: only the visible (query, key) pairs, the ideal a causal
  flash kernel approaches.

The saving of a DPVR variant follows from the dimensions alone, with no
latency measurement involved, so it is an independent prediction rather than
the calibrated rho of the original manuscript.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LMDims:
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    vocab: int

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads


@dataclass(frozen=True)
class VisionDims:
    hidden: int = 1024
    intermediate: int = 4096
    layers: int = 24
    image_size: int = 336
    patch_size: int = 14
    channels: int = 3

    @property
    def patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


PRESETS = {
    "llava-1.5-7b": LMDims(hidden=4096, intermediate=11008, layers=32, heads=32, kv_heads=32, vocab=32064),
    "llava-1.5-13b": LMDims(hidden=5120, intermediate=13824, layers=40, heads=40, kv_heads=40, vocab=32064),
}


def decoder_layer_flops(d: LMDims, n_q: int, n_k: int, causal: bool, dense: bool) -> int:
    """One decoder layer: n_q queries over a block of n_k keys (n_q == n_k for prefill)."""
    kv = 2 * d.kv_heads * d.head_dim
    proj = 2 * n_q * d.hidden * (d.hidden + kv) + 2 * n_q * d.hidden * d.hidden  # QKV + O
    mlp = 2 * n_q * 3 * d.hidden * d.intermediate  # SwiGLU: gate, up, down
    if dense or not causal:
        pairs = n_q * n_k
    else:  # query i sees the n_k - n_q cached keys plus i+1 new ones
        pairs = n_q * (n_k - n_q) + n_q * (n_q + 1) // 2
    attn = 2 * 2 * pairs * d.hidden  # QK^T and AV
    return proj + mlp + attn


def vision_flops(v: VisionDims, lm_hidden: int) -> int:
    """CLIP ViT (HF runs every layer) + patch embedding + LLaVA-1.5 MLP projector.

    The projector is Linear(vision -> lm_hidden), GELU, Linear(lm_hidden -> lm_hidden).
    """
    n = v.patches + 1  # + CLS
    patch = 2 * v.patches * v.hidden * v.channels * v.patch_size ** 2
    layer = 2 * n * 4 * v.hidden * v.hidden + 2 * n * 2 * v.hidden * v.intermediate + 4 * n * n * v.hidden
    proj = 2 * v.patches * v.hidden * lm_hidden + 2 * v.patches * lm_hidden * lm_hidden
    return patch + v.layers * layer + proj


def prefill_flops(
    d: LMDims,
    n_img: int,
    n_txt: int,
    *,
    method: str = "vanilla",
    split_layer: int | None = None,
    fusion_count: int = 1,
    vision_depth: int = 1,
    attention: str = "dense",
    vision: VisionDims | None = VisionDims(),
    lm_head_positions: str = "all",
) -> dict:
    """Prefill FLOPs broken down by stage.

    method: vanilla | dpvr_lf | lf_no_branch | vtw | freeze_image
    lm_head_positions: "all" (HF forward, training) or "last" (generation prefill).
    """
    if attention not in ("dense", "causal"):
        raise ValueError(attention)
    causal = attention == "causal"
    dense = attention == "dense"
    S, L = n_img + n_txt, d.layers
    if method in ("vanilla", "freeze_image"):  # freeze_image computes every position
        s, K = L, 0
    elif method == "vtw":
        s, K = int(split_layer), 0
    elif method in ("dpvr_lf", "lf_no_branch"):
        s, K = int(split_layer), int(fusion_count)
    else:
        raise ValueError(f"unknown method {method!r}")
    if not (0 < s <= L - K):
        raise ValueError(f"split_layer={s}, fusion_count={K} invalid for {L} layers")

    full = decoder_layer_flops(d, S, S, causal, dense)
    text = decoder_layer_flops(d, n_txt, n_txt, causal, dense)
    branch = decoder_layer_flops(d, n_img, n_img, causal, dense)
    out = {
        "vision": vision_flops(vision, d.hidden) if vision is not None else 0,
        "shallow": s * full,
        "side_branch": vision_depth * branch if method == "dpvr_lf" else 0,
        "text_only": (L - s - K) * text,
        "fusion": K * full,
        "lm_head": 2 * (S if lm_head_positions == "all" else 1) * d.hidden * d.vocab,
    }
    out["language_model"] = out["shallow"] + out["side_branch"] + out["text_only"] + out["fusion"]
    out["total"] = out["vision"] + out["language_model"] + out["lm_head"]
    return out


def saving_table(d: LMDims, n_img: int, n_txt: int, configs: list[dict], attention: str = "dense",
                 vision: VisionDims | None = VisionDims()) -> list[dict]:
    """Rows of {config, FLOPs, saving vs vanilla} for total / LM-only / deep stack."""
    base = prefill_flops(d, n_img, n_txt, attention=attention, vision=vision)
    rows = []
    for cfg in configs:
        f = prefill_flops(d, n_img, n_txt, attention=attention, vision=vision, **cfg)
        s = cfg.get("split_layer") or d.layers
        deep_base = (d.layers - s) * decoder_layer_flops(d, n_img + n_txt, n_img + n_txt,
                                                         attention == "causal", attention == "dense")
        deep = f["side_branch"] + f["text_only"] + f["fusion"]
        rows.append({
            **cfg,
            "tflops_total": f["total"] / 1e12,
            "saving_total": f["total"] / base["total"] - 1,
            "saving_lm": f["language_model"] / base["language_model"] - 1,
            "saving_deep": (deep / deep_base - 1) if deep_base else 0.0,
        })
    return rows


def image_share_per_layer(d: LMDims, n_img: int, n_txt: int, attention: str = "dense") -> float:
    """Fraction of one full-sequence layer's FLOPs removed by dropping image tokens.

    This is the analytic counterpart of the manuscript's calibrated rho.
    """
    c, de = attention == "causal", attention == "dense"
    full = decoder_layer_flops(d, n_img + n_txt, n_img + n_txt, c, de)
    return 1 - decoder_layer_flops(d, n_txt, n_txt, c, de) / full


def dims_from_hf_config(cfg: dict) -> tuple[LMDims, VisionDims]:
    """LMDims / VisionDims from a LLaVA-HF config.json dict."""
    t, v = cfg["text_config"], cfg["vision_config"]
    hidden = t.get("hidden_size", 4096)
    lm = LMDims(
        hidden=hidden,
        intermediate=t.get("intermediate_size", 11008),
        layers=t.get("num_hidden_layers", 32),
        heads=t.get("num_attention_heads", 32),
        kv_heads=t.get("num_key_value_heads", t.get("num_attention_heads", 32)),
        vocab=t.get("vocab_size", cfg.get("vocab_size", 32064)),
    )
    vis = VisionDims(
        hidden=v.get("hidden_size", 1024), intermediate=v.get("intermediate_size", 4096),
        layers=v.get("num_hidden_layers", 24), image_size=v.get("image_size", 336),
        patch_size=v.get("patch_size", 14),
    )
    return lm, vis


__all__ = ["LMDims", "VisionDims", "PRESETS", "decoder_layer_flops", "vision_flops",
           "prefill_flops", "saving_table", "image_share_per_layer", "dims_from_hf_config", "asdict"]

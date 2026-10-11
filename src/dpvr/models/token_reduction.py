"""Visual-token reduction baselines and their combination with DPVR-LF (C4, C5).

Two families, both training-free (inference only, like the original papers'
plug-and-play settings):

LLM-side pruning (drop image tokens between decoder layers; kept tokens keep
their original position ids):
- ``fastv``  FastV (Chen et al., ECCV 2024): at layer K keep the top
  ``keep_ratio`` image tokens ranked by the attention they receive in layer
  K-1. ``score: last`` ranks by the last token's attention (the official
  implementation), ``score: mean`` by the mean over all queries (the paper's
  description).
- ``pdrop``  PyramidDrop (Xing et al., CVPR 2025): at each stage boundary
  (default layers 8/16/24) keep ``keep_ratio`` of the remaining image tokens,
  ranked by the last instruction token's attention.

Vision-side reduction (fewer image tokens enter the LLM; the sequence is
shortened, positions are contiguous):
- ``prumerge``  LLaVA-PruMerge (Shang et al.): keep the CLIP patches whose
  [CLS] attention is an IQR outlier, then add to each kept patch the
  [CLS]-attention-weighted sum of its 32 nearest patches by key cosine.
- ``visionzip``  VisionZip (Yang et al., CVPR 2025): ``dominant`` patches with
  the highest [CLS] attention (plus [CLS] itself), and ``contextual`` tokens
  formed by merging the remaining patches into evenly spaced targets by key
  similarity.

Every method also runs on a trained DPVR-LF checkpoint (``lf_reduced``):
LLM-side pruning must happen before the split layer, vision-side reduction
simply feeds fewer image tokens to the shallow stack, side branch and fusion
layers. ``reduction: none`` reproduces DPVR-LF / vanilla exactly (tests).

The algorithms follow the official code as closely as this re-implementation
allows; each must reproduce at least one published number before it goes in
the paper (PLAN C5 acceptance). Vision-side methods and multi-stage pruning
need batch size 1 (they produce per-sample token counts); evaluation runs at
batch size 1.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, GenerationConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from .modeling_llava import LlavaForConditionalGeneration
from .registry import register_method

LLM_SIDE = ("fastv", "pdrop")
VISION_SIDE = ("prumerge", "visionzip")


# ---------------------------------------------------------------- LLM side
def additive_causal_mask(am: torch.Tensor, dtype) -> torch.Tensor:
    """(B, 1, S, S) additive mask: causal in index order, padding keys masked."""
    B, S = am.shape
    neg = torch.finfo(dtype).min / 2
    mask = torch.triu(torch.full((S, S), neg, dtype=dtype, device=am.device), diagonal=1)
    mask = mask.expand(B, 1, S, S).clone()
    return mask.masked_fill(~am[:, None, None, :], neg)


def attention_probs(layer, h_in: torch.Tensor, pos_emb, am: torch.Tensor) -> torch.Tensor:
    """Softmax attention (B, heads, S, S) that ``layer`` computes on input ``h_in``."""
    attn = layer.self_attn
    x = layer.input_layernorm(h_in)
    B, S, _ = x.shape
    hd = attn.head_dim
    q = attn.q_proj(x).view(B, S, -1, hd).transpose(1, 2)
    k = attn.k_proj(x).view(B, S, -1, hd).transpose(1, 2)
    cos, sin = pos_emb
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k = repeat_kv(k, q.shape[1] // k.shape[1])
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(hd)
    scores = scores + additive_causal_mask(am, torch.float32)
    return scores.softmax(dim=-1)


def image_scores(probs: torch.Tensor, am: torch.Tensor, score: str) -> torch.Tensor:
    """Per-key importance (B, S) from attention probabilities."""
    p = probs.mean(dim=1)  # heads -> (B, S, S)
    if score == "last":
        last = am.long().cumsum(-1).argmax(-1)  # index of the last real token
        return p[torch.arange(p.shape[0]), last]
    if score == "mean":
        w = am.float()
        return (p * w[:, :, None]).sum(1) / w.sum(1, keepdim=True)
    raise ValueError(f"score must be 'last' or 'mean', got {score!r}")


def select_keep(scores: torch.Tensor, img: torch.Tensor, keep: int) -> torch.Tensor:
    """Keep mask (B, S): every non-image position plus the top-``keep`` image positions."""
    s = scores.masked_fill(~img, float("-inf"))
    top = s.topk(keep, dim=1).indices
    out = ~img.clone()
    out.scatter_(1, top, True)
    return out


def _gather(keep: torch.Tensor, *tensors):
    n = int(keep.sum(1)[0])
    if not bool((keep.sum(1) == n).all()):
        raise ValueError("every batch row must keep the same number of tokens")
    idx = keep.nonzero()[:, 1].view(keep.shape[0], n)
    out = []
    for t in tensors:
        if t.dim() == 3:
            out.append(t.gather(1, idx[:, :, None].expand(-1, -1, t.shape[-1])))
        else:
            out.append(t.gather(1, idx))
    return idx, out


def _run(layer, hidden, pos_emb, mask):
    out = layer(hidden, position_embeddings=pos_emb, attention_mask=mask,
                past_key_values=None, cache_position=None, output_attentions=False,
                use_cache=False)
    return out[0] if isinstance(out, (tuple, list)) else out


def pruning_schedule(method: str, n_layers: int, params: dict) -> dict[int, float]:
    """layer index -> fraction of the *current* image tokens kept before that layer."""
    if method == "fastv":
        return {int(params.get("layer", 2)): float(params.get("keep_ratio", 0.5))}
    if method == "pdrop":
        layers = params.get("layers", [8, 16, 24])
        return {int(i): float(params.get("keep_ratio", 0.5)) for i in layers if 0 < int(i) < n_layers}
    return {}


def run_language_model(
    lm,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    am: torch.Tensor,
    img: Optional[torch.Tensor],
    *,
    schedule: dict[int, float] | None = None,
    score: str = "last",
    lf: dict | None = None,
):
    """Decoder stack with optional LLM-side pruning and optional DPVR-LF deep graph.

    lf = {"split_layer": s, "fusion_count": K, "branch": ModuleList} runs the
    DPVR-LF graph from layer s on (same semantics as TokenDiversionX3FusionLlava).
    Returns (hidden, kept_index) where kept_index maps output rows to input rows.
    """
    schedule = schedule or {}
    L = len(lm.layers)
    stop = lf["split_layer"] if lf else L
    if lf and any(k >= stop for k in schedule):
        raise ValueError("with DPVR-LF, pruning layers must lie before split_layer")
    B, S, H = hidden.shape
    kept = torch.arange(S, device=hidden.device).expand(B, S)
    pos_emb = lm.rotary_emb(hidden, position_ids)
    mask = additive_causal_mask(am, hidden.dtype)
    h_prev = None
    for i in range(stop):
        if i in schedule and img is not None and bool(img.any()) and h_prev is not None:
            n_img = int(img.sum(1).min())
            keep_n = max(1, round(n_img * schedule[i]))
            if keep_n < n_img:
                probs = attention_probs(lm.layers[i - 1], h_prev, prev_pos_emb, am)
                keep = select_keep(image_scores(probs, am, score), img, keep_n)
                idx, (hidden, position_ids, am, img, kept) = _gather(
                    keep, hidden, position_ids, am, img, kept)
                pos_emb = lm.rotary_emb(hidden, position_ids)
                mask = additive_causal_mask(am, hidden.dtype)
        h_prev, prev_pos_emb = hidden, pos_emb
        hidden = _run(lm.layers[i], hidden, pos_emb, mask)

    if lf and img is not None and bool(img.any()):
        hidden = _lf_deep(lm, hidden, position_ids, am, img, pos_emb, mask, lf)
    elif lf:
        for i in range(stop, L):
            hidden = _run(lm.layers[i], hidden, pos_emb, mask)
    return hidden, kept


def _lf_deep(lm, hidden, position_ids, am, img, pos_emb, mask, lf):
    """DPVR-LF from the split layer on: side branch, text-only deep, last-K fusion."""
    B, S, H = hidden.shape
    L = len(lm.layers)
    n_img = int(img.sum(1).max())
    T = S - n_img
    s, K = lf["split_layer"], lf["fusion_count"]
    # Side branch: image tokens alone, positions 0..n_img-1 (as in the trained model).
    out_image = hidden[img].view(B, n_img, H)
    ipos = torch.arange(n_img, device=hidden.device).expand(B, -1)
    ipe = lm.rotary_emb(out_image, ipos)
    imask = additive_causal_mask(torch.ones(B, n_img, dtype=torch.bool, device=hidden.device),
                                 hidden.dtype)
    for layer in lf["branch"]:
        out_image = _run(layer, out_image, ipe, imask)
    txt = ~img
    text_h = hidden[txt].view(B, T, H)
    tpe = lm.rotary_emb(text_h, position_ids[txt].view(B, T))
    tmask = additive_causal_mask(am[txt].view(B, T), hidden.dtype)
    for i in range(s, L - K):
        text_h = _run(lm.layers[i], text_h, tpe, tmask)
    hidden = hidden.clone()
    hidden[txt] = text_h.reshape(-1, H)
    hidden[img] = out_image.reshape(-1, H)
    for i in range(L - K, L):
        hidden = _run(lm.layers[i], hidden, pos_emb, mask)
    return hidden


# ------------------------------------------------------------- vision side
@contextmanager
def _capture_keys(layer):
    store = {}
    handle = layer.self_attn.k_proj.register_forward_hook(lambda m, i, o: store.__setitem__("k", o))
    try:
        yield store
    finally:
        handle.remove()


def clip_features_attention_keys(vision_tower, pixel_values, feature_layer: int = -2,
                                 key_layer: int | None = None):
    """CLIP hidden states at ``feature_layer`` (with [CLS]), that layer's attention and keys."""
    key_layer = feature_layer if key_layer is None else key_layer
    layers = vision_tower.vision_model.encoder.layers
    impl = vision_tower.config._attn_implementation
    vision_tower.config._attn_implementation = "eager"  # attention weights are needed
    try:
        with _capture_keys(layers[key_layer]) as store:
            out = vision_tower(pixel_values, output_hidden_states=True, output_attentions=True)
    finally:
        vision_tower.config._attn_implementation = impl
    return out.hidden_states[feature_layer], out.attentions[feature_layer], store["k"]


def iqr_keep_count(cls_attn: torch.Tensor) -> int:
    """PruMerge's adaptive count: patches whose [CLS] attention exceeds Q3 + 1.5 IQR."""
    flat = cls_attn.float().flatten()
    q1, q3 = torch.quantile(flat, 0.25), torch.quantile(flat, 0.75)
    upper = q3 + 1.5 * (q3 - q1)
    ratio = float((flat > upper).float().mean())
    return max(1, int(cls_attn.shape[-1] * ratio))


def prumerge(patches: torch.Tensor, cls_attn: torch.Tensor, keys: torch.Tensor,
             n_keep: int | None = None, k_neighbors: int = 32) -> torch.Tensor:
    """patches (B, N, C), cls_attn (B, N), keys (B, N, Ck) -> merged (B, n_keep, C)."""
    B, N, C = patches.shape
    n_keep = iqr_keep_count(cls_attn) if n_keep is None else n_keep
    idx = cls_attn.topk(n_keep, dim=1).indices
    keys_n = F.normalize(keys.float(), dim=-1)
    out = torch.empty(B, n_keep, C, dtype=patches.dtype, device=patches.device)
    for b in range(B):
        centers = idx[b]
        for j, c in enumerate(centers.tolist()):
            others = torch.ones(N, dtype=torch.bool, device=patches.device)
            others[c] = False
            cand = others.nonzero().squeeze(1)
            sim = keys_n[b, cand] @ keys_n[b, c]
            near = cand[sim.topk(min(k_neighbors, cand.numel())).indices]
            w = cls_attn[b, near].to(patches.dtype)[:, None]
            out[b, j] = patches[b, c] + (patches[b, near] * w).sum(0)
    return out


def visionzip(hidden_with_cls: torch.Tensor, attn: torch.Tensor, keys: torch.Tensor,
              dominant: int = 54, contextual: int = 10, keep_cls: bool = True,
              key_heads: int = 1) -> torch.Tensor:
    """hidden_with_cls (B, 1+N, C), attn (B, heads, 1+N, 1+N), keys (B, 1+N, heads*hd).

    The merge metric is the key averaged over heads, as in the official code.
    """
    B, T, C = hidden_with_cls.shape
    keys = keys.view(B, T, key_heads, -1).mean(2)
    cls_sum = attn[:, :, 0, 1:].sum(dim=1)  # (B, N)
    top = cls_sum.topk(dominant, dim=1).indices + 1
    chosen = torch.cat([torch.zeros(B, 1, dtype=top.dtype, device=top.device), top], 1) if keep_cls else top
    rest = torch.ones(B, T, dtype=torch.bool, device=hidden_with_cls.device)
    rest.scatter_(1, chosen, False)
    if not keep_cls:
        rest[:, 0] = False
    dominant_tokens = hidden_with_cls.gather(1, chosen.sort(1).values[:, :, None].expand(-1, -1, C))
    n_rest = int(rest.sum(1)[0])
    h_rest = hidden_with_cls[rest].view(B, n_rest, C)
    m_rest = keys[rest].view(B, n_rest, -1).float()
    m_rest = m_rest / m_rest.norm(dim=-1, keepdim=True)
    step = max(1, n_rest // contextual)
    targets = torch.arange(0, n_rest, step, device=h_rest.device)[:contextual]
    is_target = torch.zeros(n_rest, dtype=torch.bool, device=h_rest.device)
    is_target[targets] = True
    sim = m_rest[:, ~is_target] @ m_rest[:, targets].transpose(1, 2)
    assign = torch.zeros_like(sim).scatter_(2, sim.argmax(2, keepdim=True), 1.0)
    counts = assign.sum(1).clamp(min=1)[:, :, None]
    merged = (assign.transpose(1, 2) @ h_rest[:, ~is_target].float()) / counts
    contextual_tokens = h_rest[:, targets] + merged.to(h_rest.dtype)
    return torch.cat([dominant_tokens, contextual_tokens], dim=1)


def splice_image_tokens(input_ids, embed, image_token_id: int, features: torch.Tensor):
    """Replace the image placeholder run in input_ids with ``features`` (B=1).

    Returns inputs_embeds (1, S', H) and the image mask (1, S').
    """
    if input_ids.shape[0] != 1:
        raise ValueError("vision-side reduction needs batch size 1")
    pos = (input_ids[0] == image_token_id).nonzero().squeeze(1)
    if pos.numel() == 0:
        return embed(input_ids), None
    start, end = int(pos[0]), int(pos[-1]) + 1
    text = embed(input_ids)
    emb = torch.cat([text[:, :start], features.to(text.dtype), text[:, end:]], dim=1)
    img = torch.zeros(1, emb.shape[1], dtype=torch.bool, device=emb.device)
    img[:, start:start + features.shape[1]] = True
    return emb, img


# ------------------------------------------------------------------ models
class ReducedLlava(PreTrainedModel, GenerationMixin):
    """Vanilla LLaVA-1.5 or a trained DPVR-LF with a visual-token reduction method."""

    def __init__(self, base_model_path: str, reduction: str = "none", params: dict | None = None,
                 lf_checkpoint: str | None = None):
        super().__init__(AutoConfig.from_pretrained(base_model_path))
        self.lf_model = None
        if lf_checkpoint:
            from .checkpoint_io import load_trained

            self.lf_model, _ = load_trained(lf_checkpoint)
            self.llava_model = self.lf_model.llava_model
        else:
            self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                base_model_path, torch_dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True)
        for p in self.parameters():
            p.requires_grad_(False)
        self._configure(reduction, params or {})

    def _configure(self, reduction: str, params: dict) -> None:
        if reduction not in ("none",) + LLM_SIDE + VISION_SIDE:
            raise ValueError(f"unknown reduction {reduction!r}")
        self.reduction = reduction
        self.params = dict(params)

    @classmethod
    def from_config(cls, cfg: dict) -> "ReducedLlava":
        m = cfg.get("method", {})
        return cls(base_model_path=cfg["model"]["base_model_path"],
                   reduction=m.get("reduction", "none"), params=m.get("params") or {},
                   lf_checkpoint=m.get("checkpoint"))

    @classmethod
    def _from_parts(cls, llava_model, config, reduction="none", params=None, lf_model=None):
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.config = config
        obj.generation_config = GenerationConfig.from_model_config(config)
        obj.name_or_path = ""
        obj._is_hf_initialized = True
        obj.llava_model = llava_model
        obj.lf_model = lf_model
        obj._configure(reduction, params or {})
        return obj

    def print_trainable_parameters(self) -> None:
        print(f"trainable params: 0 | all params: {sum(p.numel() for p in self.parameters()):,}")

    def prepare_inputs_for_generation(self, input_ids, pixel_values=None, attention_mask=None,
                                      **kwargs) -> dict:
        return {"input_ids": input_ids, "pixel_values": pixel_values,
                "attention_mask": attention_mask, "use_cache": False}

    def _lf_spec(self) -> dict | None:
        if self.lf_model is None:
            return None
        return {"split_layer": self.lf_model.split_layer, "fusion_count": self.lf_model.fusion_count,
                "branch": self.lf_model.single_transformer}

    def _embed(self, input_ids, pixel_values):
        mm = self.llava_model.model
        if self.reduction not in VISION_SIDE or pixel_values is None:
            emb, mask3 = mm.prepare_multimodal_input(input_ids, pixel_values)
            return emb, (mask3.any(-1) if mask3 is not None else None)
        p = self.params
        feature_layer = int(p.get("feature_layer", -2))
        hidden, attn, keys = clip_features_attention_keys(
            mm.vision_tower, pixel_values, feature_layer, p.get("key_layer"))
        if self.reduction == "prumerge":
            feats = prumerge(hidden[:, 1:], attn[:, :, 0, 1:].mean(1), keys[:, 1:],
                             n_keep=p.get("n_keep"), k_neighbors=int(p.get("k_neighbors", 32)))
        else:
            feats = visionzip(hidden, attn, keys, int(p.get("dominant", 54)),
                              int(p.get("contextual", 10)), bool(p.get("keep_cls", True)),
                              key_heads=mm.vision_tower.config.num_attention_heads)
        feats = mm.multi_modal_projector(feats)
        return splice_image_tokens(input_ids, mm.get_input_embeddings(),
                                   self.llava_model.config.image_token_index, feats)

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, labels=None, **kwargs):
        emb, img = self._embed(input_ids, pixel_values)
        B, S, _ = emb.shape
        if attention_mask is not None and attention_mask.shape[1] == S:
            am = attention_mask.bool()
        else:
            if attention_mask is not None and not bool(attention_mask.bool().all()):
                raise ValueError("vision-side reduction does not support padding")
            am = torch.ones(B, S, dtype=torch.bool, device=emb.device)
        pos = (am.long().cumsum(-1) - 1).masked_fill(~am, 0)
        lm = self.llava_model.language_model
        schedule = (pruning_schedule(self.reduction, len(lm.layers), self.params)
                    if pixel_values is not None else {})
        hidden, kept = run_language_model(lm, emb, pos, am, img, schedule=schedule,
                                          score=self.params.get("score", "last"), lf=self._lf_spec())
        logits = self.llava_model.lm_head(lm.norm(hidden))
        out = CausalLMOutputWithPast(logits=logits)
        out.kept_index = kept
        return out


@register_method("reduced")
class ReducedVanillaLlava(ReducedLlava):
    """method.name: reduced — vanilla weights + method.reduction."""


@register_method("lf_reduced")
class ReducedLFLlava(ReducedLlava):
    """method.name: lf_reduced — trained DPVR-LF (method.checkpoint) + method.reduction."""

    @classmethod
    def from_config(cls, cfg: dict) -> "ReducedLlava":
        if not cfg.get("method", {}).get("checkpoint"):
            raise ValueError("lf_reduced needs method.checkpoint (a trained DPVR-LF run)")
        return super().from_config(cfg)

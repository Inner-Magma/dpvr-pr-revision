"""Model-agnostic visual-saturation probes (C6, R1.2 / R2.4 / R2.7).

One forward pass per sample on a vanilla HF vision-language model; for every
decoder layer l (h_0 = the layer-0 input embeddings, h_{l+1} = output of layer l)
we record, separately for image and text positions:

- ``cos``    mean cos(h_l, h_{l+1})                    (paper Figure 2a)
- ``resid``  mean ||h_{l+1} - h_l|| / ||h_l||           relative residual update
- ``attn``   text-to-image attention mass of layer l   (paper Figure 2b; text
             queries after the last image token, mean over heads and queries)
- ``attn_last`` the same for the last prompt token only
- ``kl``     logit-lens KL(p_l || p_final), full vocabulary, with
             p_l = softmax(lm_head(norm(h_{l+1})))     (paper Figure 2c)

Nothing here assumes LLaVA-1.5: the image token id comes from the config, the
image positions from ``input_ids``, and the decoder stack is located for the
llava / llava_next / qwen2(.5)_vl / internvl layouts of transformers 4.57.
``pool_image_tokens`` (LLaVA-family square grids) shrinks the image tokens to an
arbitrary grid for the token-count study (E8).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

METRICS = ("cos_img", "cos_txt", "resid_img", "resid_txt", "kl_img", "kl_txt",
           "attn", "attn_last")


@dataclass
class VLMParts:
    layers: nn.ModuleList
    norm: nn.Module
    lm_head: nn.Module
    embed_tokens: nn.Module
    image_token_id: int

    @property
    def n_layers(self) -> int:
        return len(self.layers)


def image_token_id(config) -> int:
    for key in ("image_token_id", "image_token_index"):
        v = getattr(config, key, None)
        if v is not None:
            return int(v)
    raise ValueError(f"{type(config).__name__} has no image_token_id / image_token_index")


def locate_parts(model) -> VLMParts:
    """Find the decoder stack, final norm and LM head of an HF VLM."""
    candidates = []
    inner = getattr(model, "model", None)
    for owner in (inner, model):
        if owner is None:
            continue
        lm = getattr(owner, "language_model", None)
        if lm is not None:
            candidates += [lm, getattr(lm, "model", None)]
    for lm in candidates:
        if lm is not None and hasattr(lm, "layers") and hasattr(lm, "norm"):
            head = model.get_output_embeddings()
            return VLMParts(lm.layers, lm.norm, head, lm.get_input_embeddings(),
                            image_token_id(model.config))
    raise ValueError(f"cannot locate the language model inside {type(model).__name__}")


def token_masks(input_ids: torch.LongTensor, attention_mask: torch.Tensor | None,
                img_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(img, txt, query) masks of shape (S,) for a single sample (B=1).

    query = text positions after the last image token (the instruction).
    """
    ids = input_ids[0]
    valid = torch.ones_like(ids, dtype=torch.bool) if attention_mask is None else attention_mask[0].bool()
    img = (ids == img_id) & valid
    txt = valid & ~img
    if not img.any():
        raise ValueError("sample has no image tokens; check the prompt / chat template")
    last = int(img.nonzero().max())
    query = txt.clone()
    query[: last + 1] = False
    return img, txt, query


# ---------------------------------------------------------------------------
# image-token pooling (E8)
# ---------------------------------------------------------------------------

def parse_grid(spec: str | None) -> tuple[int, int] | None:
    """'12x24' -> (12, 24); '144' -> (12, 12); None/'' -> None."""
    if not spec:
        return None
    if "x" in spec:
        h, w = spec.lower().split("x")
        return int(h), int(w)
    n = int(spec)
    r = int(round(n ** 0.5))
    if r * r != n:
        raise ValueError(f"{n} is not a square; give the grid as HxW")
    return r, r


def pool_features(feats: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """(n, d) square patch grid -> (gh*gw, d) by adaptive average pooling."""
    n, d = feats.shape
    side = int(round(n ** 0.5))
    if side * side != n:
        raise ValueError(f"{n} image features do not form a square grid")
    x = feats.T.reshape(1, d, side, side).float()
    x = F.adaptive_avg_pool2d(x, grid)
    return x.reshape(d, -1).T.to(feats.dtype)


def collapse_image_run(input_ids: torch.LongTensor, img_id: int, n_new: int) -> torch.LongTensor:
    """Replace the single contiguous image-token run of a (1, S) sequence by n_new tokens."""
    ids = input_ids[0]
    pos = (ids == img_id).nonzero().flatten()
    if len(pos) == 0 or int(pos[-1] - pos[0]) + 1 != len(pos):
        raise ValueError("expected one contiguous run of image tokens")
    a, b = int(pos[0]), int(pos[-1]) + 1
    run = torch.full((n_new,), img_id, dtype=ids.dtype, device=ids.device)
    return torch.cat([ids[:a], run, ids[b:]]).unsqueeze(0)


@torch.no_grad()
def pool_image_tokens(model, parts: VLMParts, enc: dict, grid: tuple[int, int]) -> dict:
    """LLaVA-family inputs with the image tokens pooled to ``grid``.

    The projected patch features are pooled (after the projector, before the
    LLM), so the LLM sees gh*gw image tokens at the original image position.
    Returns ``{input_ids, inputs_embeds, attention_mask}`` for ``analyze_sample``.
    """
    cfg = model.config
    if getattr(cfg, "model_type", "") != "llava":
        raise ValueError("image-token pooling is implemented for LLaVA-1.5 style models only")
    feats = model.get_image_features(
        pixel_values=enc["pixel_values"],
        vision_feature_layer=cfg.vision_feature_layer,
        vision_feature_select_strategy=cfg.vision_feature_select_strategy,
    )
    feats = feats[0]  # list (or batch) of per-image (n, d) features; B=1
    pooled = pool_features(feats, grid)
    ids = collapse_image_run(enc["input_ids"], parts.image_token_id, pooled.shape[0])
    embeds = parts.embed_tokens(ids)
    mask = ids == parts.image_token_id
    embeds[mask] = pooled.to(embeds.dtype)
    return {"input_ids": ids, "inputs_embeds": embeds,
            "attention_mask": torch.ones_like(ids)}


# ---------------------------------------------------------------------------
# per-sample probe
# ---------------------------------------------------------------------------

def _layer_out(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    return getattr(out, "hidden_states", out)


@torch.no_grad()
def _kl_rows(parts: VLMParts, h: torch.Tensor, logp_final: torch.Tensor, chunk: int) -> torch.Tensor:
    """KL(p_l || p_final) per row of h (n, d)."""
    out = []
    for i in range(0, h.shape[0], chunk):
        logp = parts.lm_head(parts.norm(h[i:i + chunk])).float().log_softmax(-1)
        out.append((logp.exp() * (logp - logp_final[i:i + chunk])).sum(-1))
    return torch.cat(out)


@torch.no_grad()
def analyze_sample(model, parts: VLMParts, inputs: dict, *, attention: bool = True,
                   logit_lens: bool = True, kl_chunk: int = 256) -> dict[str, np.ndarray]:
    """Run one sample (batch size 1) and return each metric as an (L,) array.

    ``inputs`` are the processor outputs (or ``pool_image_tokens`` outputs) and
    must contain ``input_ids``; it is passed to ``model(**inputs)`` without them
    when ``inputs_embeds`` is present. ``attention=True`` needs the model loaded
    with ``attn_implementation="eager"``.
    """
    img, txt, query = token_masks(inputs["input_ids"], inputs.get("attention_mask"),
                                  parts.image_token_id)
    L = parts.n_layers
    hs: list[torch.Tensor | None] = [None] * (L + 1)
    attn = np.full(L, np.nan)
    attn_last = np.full(L, np.nan)
    q_idx = query.nonzero().flatten()
    last_q = int(q_idx[-1]) if len(q_idx) else None

    def pre_hook(_m, args, kwargs):
        x = args[0] if args else kwargs["hidden_states"]
        hs[0] = x[0].detach()

    def make_layer_hook(i):
        def hook(_m, _a, out):
            hs[i + 1] = _layer_out(out)[0].detach()
        return hook

    def make_attn_hook(i):
        def hook(_m, _a, out):
            w = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
            if w is None:
                raise RuntimeError("attention weights unavailable: load the model with "
                                   "attn_implementation='eager' or pass attention=False")
            w = w[0].float()                                # (heads, S, S)
            to_img = w[:, :, img.to(w.device)].sum(-1)      # (heads, S)
            if last_q is not None:
                attn[i] = float(to_img[:, q_idx.to(w.device)].mean())
                attn_last[i] = float(to_img[:, last_q].mean())
        return hook

    handles = [parts.layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)]
    for i, layer in enumerate(parts.layers):
        handles.append(layer.register_forward_hook(make_layer_hook(i)))
        if attention:
            handles.append(layer.self_attn.register_forward_hook(make_attn_hook(i)))
    call = dict(inputs)
    if "inputs_embeds" in call:
        call.pop("input_ids")
    try:
        model(**call, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    res: dict[str, np.ndarray] = {}
    dev = hs[0].device
    m_img, m_txt = img.to(dev), txt.to(dev)
    for tag, m in (("img", m_img), ("txt", m_txt)):
        cos, resid = np.zeros(L), np.zeros(L)
        for l in range(L):
            a, b = hs[l][m].float(), hs[l + 1][m].float()
            cos[l] = float(F.cosine_similarity(a, b, dim=-1).mean())
            resid[l] = float(((b - a).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-6)).mean())
        res[f"cos_{tag}"], res[f"resid_{tag}"] = cos, resid
    if logit_lens:
        for tag, m in (("img", m_img), ("txt", m_txt)):
            final = hs[L][m]
            logp_final = torch.cat([
                parts.lm_head(parts.norm(final[i:i + kl_chunk])).float().log_softmax(-1)
                for i in range(0, final.shape[0], kl_chunk)])
            kl = np.zeros(L)
            for l in range(L):
                kl[l] = float(_kl_rows(parts, hs[l + 1][m], logp_final, kl_chunk).mean())
            res[f"kl_{tag}"] = kl
    if attention:
        res["attn"], res["attn_last"] = attn, attn_last
    res["n_img_tokens"] = np.array([int(img.sum())])
    res["n_txt_tokens"] = np.array([int(txt.sum())])
    return res


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def detect_transition(curve: np.ndarray) -> int:
    """First layer where a decreasing curve falls below 50% of its peak with at
    most one later reversal (the rule of scripts/preliminary_kl.py, paper §3.1)."""
    curve = np.asarray(curve, dtype=float)
    thr = 0.5 * float(np.nanmax(curve))
    n = len(curve)
    for l in range(n):
        if curve[l] < thr:
            if sum(1 for j in range(l, n - 1) if curve[j + 1] > curve[j]) <= 1:
                return l
    return int(np.nanargmin(curve))


def plateau_layer(curve: np.ndarray, tau: float) -> int | None:
    """First layer from which ``curve`` stays at or above ``tau`` (e.g. image cosine)."""
    curve = np.asarray(curve, dtype=float)
    for l in range(len(curve)):
        if np.all(curve[l:] >= tau):
            return l
    return None


def summarize(curves: dict[str, np.ndarray], cos_tau: float = 0.95) -> dict:
    """curves[metric]: (n_samples, L). Per-layer median/IQR plus transition layers."""
    out: dict = {"per_layer": {}, "transitions": {}}
    for k, arr in curves.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2:
            continue
        q25, med, q75 = np.nanpercentile(arr, [25, 50, 75], axis=0)
        out["per_layer"][k] = {"median": med.tolist(), "q25": q25.tolist(), "q75": q75.tolist(),
                               "mean": np.nanmean(arr, axis=0).tolist()}
    L = None
    for k in ("kl_img", "kl_txt"):
        if k in curves:
            arr = np.asarray(curves[k], dtype=float)
            L = arr.shape[1]
            per = np.array([detect_transition(c) for c in arr])
            out["transitions"][k] = {
                "median_curve": detect_transition(np.median(arr, axis=0)),
                "per_sample_median": float(np.median(per)),
                "per_sample_iqr": [float(np.percentile(per, 25)), float(np.percentile(per, 75))],
            }
    for k in ("resid_img", "resid_txt"):
        if k in curves:
            arr = np.asarray(curves[k], dtype=float)
            L = arr.shape[1]
            # residual updates shrink toward deep layers; report where the median
            # update falls below 50% of its peak, skipping layer 0 (the input jump)
            med = np.median(arr, axis=0)
            out["transitions"][k] = {"median_curve": 1 + detect_transition(med[1:])}
    if "cos_img" in curves:
        med = np.median(np.asarray(curves["cos_img"], dtype=float), axis=0)
        L = len(med)
        out["transitions"]["cos_img"] = {"tau": cos_tau, "plateau_from": plateau_layer(med, cos_tau)}
    if L:
        out["n_layers"] = L
        out["normalized"] = {k: (v["median_curve"] / L if v.get("median_curve") is not None else None)
                             for k, v in out["transitions"].items() if "median_curve" in v}
    return out

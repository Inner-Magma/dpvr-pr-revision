#!/usr/bin/env python
"""scripts/preliminary_kl.py — KL + logit lens preliminary analysis (v3).

Three-way triangulation across three independently-formulated metrics:

  Path B-vision  (--mode vlm):        full-vocab KL, mean over vision positions
  Path B-text    (--mode vlm):        Zero over-positions KL on VLM text positions
  Path A         (--mode text_only):  Zero over-positions KL on LLM-only text inputs

All three share the same COCO samples (control variable = whether image is present).
Triangulation figure generated via --mode triangulation (no model loading).

Zero paper formula (over-positions KL, for text positions):
  p_l[s]       = softmax(lm_head(norm(h_l)))[s, GT_token_{s+1}]  -- scalar per pos
  p_l_m_norm   = (p_l * mask_M) / sum_s(p_l * mask_M)            -- over-pos distribution
  KL(p_l_m_norm || p_final_m_norm) summed across positions

Full-vocab KL (for vision positions, which lack GT labels):
  KL(p_l || p_final) = sum_v p_l[v] * log(p_l[v] / p_final[v])  -- per position
  Mean over vision positions per sample

v3 change from v2: vision positions now use full-vocab KL (not Zero GT formula).
v1 bug: used full-vocab KL summed over ALL positions — inflated KL 100-1000x.

Model access paths (LlavaForConditionalGeneration, transformers 4.57.1, hsf):
    lm_head    : model.lm_head              -- Linear(H=4096, V=32064) for 7B
    final_norm : model.language_model.norm  -- LlamaRMSNorm
    dec_layers : model.language_model.layers -- ModuleList (32 for 7B / 40 for 13B)
    IMAGE_TOKEN_INDEX : model.config.image_token_index == 32000

Usage:
    # Path B (VLM, image-text pairs)
    CUDA_VISIBLE_DEVICES=5 python scripts/preliminary_kl.py \\
        --mode vlm --model /path/to/models/llava-1.5-7b-hf \\
        --output results/analysis_7b_kl_v3/ --n_samples 300

    # Path A (LLM-only, text only)
    CUDA_VISIBLE_DEVICES=5 python scripts/preliminary_kl.py \\
        --mode text_only --model /path/to/models/llava-1.5-7b-hf \\
        --output results/analysis_7b_kl_v3_text_only/ --n_samples 300

    # Triangulation figure (no model loading)
    python scripts/preliminary_kl.py \\
        --mode triangulation \\
        --vlm_json results/analysis_7b_kl_v3/kl_summary.json \\
        --text_only_json results/analysis_7b_kl_v3_text_only/kl_summary.json \\
        --output results/analysis_7b_kl_v3/
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_IMAGE_TOKEN_INDEX = 32000
_IMAGE_SEQ_LENGTH_DEFAULT = 576
_DATA_ROOT = Path("/path/to/llava-1.5-665k-instructions")
_EPS = 1e-6  # matches Zero paper


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_samples(n_samples: int, seed: int = 42) -> list[dict]:
    """Load n_samples image-text pairs from LLaVA-665k COCO subset."""
    json_path = _DATA_ROOT / "llava_v1_5_mix665k.json"
    print(f"[data] Loading {json_path} ...")
    with open(json_path) as f:
        all_data = json.load(f)

    coco_entries = [e for e in all_data if e.get("image", "").startswith("coco/")]
    print(f"[data] {len(coco_entries):,} COCO entries in dataset.")

    random.seed(seed)
    candidates = random.sample(coco_entries, min(n_samples * 5, len(coco_entries)))

    samples: list[dict] = []
    for entry in candidates:
        img_path = _DATA_ROOT / entry["image"]
        if not img_path.exists():
            continue
        q = None
        for turn in entry["conversations"]:
            if turn["from"] == "human":
                q = turn["value"].replace("<image>", "").strip()
                break
        if not q:
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        samples.append({"image": img, "question": q, "id": entry.get("id", "")})
        if len(samples) >= n_samples:
            break

    if len(samples) < n_samples:
        print(f"[data] WARNING: only {len(samples)} valid samples (requested {n_samples}).")
    else:
        print(f"[data] Loaded {len(samples)} samples from COCO/train2017.")
    return samples


def build_batches(
    samples: list[dict],
    processor,
    batch_size: int,
    device: str,
    max_length: int = 1024,
) -> list[dict]:
    """Build VLM batches with images (Path B)."""
    batches = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        images = [s["image"] for s in chunk]
        texts = [f"USER: <image>\n{s['question']} ASSISTANT:" for s in chunk]
        enc = processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        batches.append({k: v.to(device) for k, v in enc.items()})
    return batches


def build_text_batches(
    samples: list[dict],
    processor,
    batch_size: int,
    device: str,
    max_length: int = 1024,
) -> list[dict]:
    """Build text-only batches (Path A) — no images, no image tokens."""
    tokenizer = processor.tokenizer
    batches = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        texts = [f"USER: {s['question']} ASSISTANT:" for s in chunk]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        has_img_tok = (enc["input_ids"] == _IMAGE_TOKEN_INDEX).any().item()
        assert not has_img_tok, (
            "Tokenized text batch contains IMAGE_TOKEN_INDEX (32000). "
            "Strip <image> from questions before building text batches."
        )
        batches.append({k: v.to(device) for k, v in enc.items()})
    return batches


# ---------------------------------------------------------------------------
# 2. Vision / text token masks  (VLM mode only)
# ---------------------------------------------------------------------------

def build_vision_text_masks(
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    image_seq_length: int = _IMAGE_SEQ_LENGTH_DEFAULT,
) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """Return (vis_mask, txt_mask) of shape (B, S) on CPU.

    Asserts exactly image_seq_length vision tokens per sample — fails fast
    if the processor text format is wrong.
    """
    ids = input_ids.cpu()
    amask = attention_mask.cpu()
    vis_mask = (ids == _IMAGE_TOKEN_INDEX)
    txt_mask = (amask == 1) & (~vis_mask)

    n_vis = vis_mask.sum(dim=1)
    bad = (n_vis != image_seq_length).nonzero(as_tuple=True)[0].tolist()
    assert not bad, (
        f"Samples {bad} have {n_vis[bad].tolist()} vision tokens, "
        f"expected {image_seq_length}. "
        "Likely cause: text format does not contain exactly one <image>."
    )
    return vis_mask, txt_mask


# ---------------------------------------------------------------------------
# 3. Model accessors
# ---------------------------------------------------------------------------

def get_llm_layers(model) -> torch.nn.ModuleList:
    """Return decoder layer ModuleList — model.language_model.layers."""
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot locate decoder layers. Expected model.language_model.layers.")


def get_lm_head_and_norm(model):
    """Return (lm_head, final_norm).

    lm_head    : model.lm_head              Linear(4096, 32064) for 7B
    final_norm : model.language_model.norm  LlamaRMSNorm
    """
    return model.lm_head, model.language_model.norm


# ---------------------------------------------------------------------------
# 4. VLM KL computation (Path B)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_kl_batch(
    model,
    batch: dict,
    n_layers: int,
    device: str,
    image_seq_length: int,
    lm_head,
    final_norm,
) -> dict[str, np.ndarray]:
    """Path B: vision (full-vocab KL) + text (Zero over-positions KL).

    vision: KL(p_l || p_final) summed over vocab, mean over vision positions.
    text:   Zero formula — gather GT probs, normalize over text positions, KL.
    all:    Zero formula on all valid positions (reference).

    Returns dict 'vision' / 'text' / 'all', each shape (B, n_layers).
    """
    B = batch["input_ids"].shape[0]

    vis_mask, txt_mask = build_vision_text_masks(
        batch["input_ids"], batch["attention_mask"], image_seq_length
    )
    vis_mask_pred = vis_mask[:, :-1]   # (B, S-1) CPU bool
    txt_mask_pred = txt_mask[:, :-1]
    all_mask_pred = (batch["attention_mask"][:, :-1] == 1).cpu()

    target_ids = batch["input_ids"][:, 1:].to(device)   # (B, S-1)

    # --- capture hidden states via hooks (stored on CPU to save VRAM) ---
    layer_hs: dict[int, torch.Tensor] = {}

    def make_hook(l: int):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, (tuple, list)) else (
                out.hidden_states if hasattr(out, "hidden_states") else out
            )
            layer_hs[l] = hs.detach().cpu()
        return hook

    llm_layers = get_llm_layers(model)
    hooks = [llm_layers[l].register_forward_hook(make_hook(l)) for l in range(n_layers)]
    try:
        _ = model(
            input_ids=batch["input_ids"],
            pixel_values=batch.get("pixel_values"),
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
    finally:
        for h in hooks:
            h.remove()

    # --- final-layer log-probs (full vocab, for vision KL) and GT probs (text KL) ---
    final_hs_pred = layer_hs[n_layers - 1][:, :-1, :].to(device)
    final_logits  = lm_head(final_norm(final_hs_pred)).float()  # (B, S-1, V)
    del final_hs_pred
    final_log_p = final_logits.log_softmax(dim=-1)              # (B, S-1, V) — kept for vision KL
    del final_logits
    # GT probs via gather (avoids materializing full exp tensor)
    final_p_gt = final_log_p.gather(2, target_ids.unsqueeze(-1)).squeeze(-1).exp()  # (B, S-1)

    eps = _EPS

    def _masked_norm(p: torch.Tensor, mask_f: torch.Tensor) -> torch.Tensor:
        pm = p * mask_f
        return pm / (pm.sum(dim=-1, keepdim=True) + eps)

    txt_float  = txt_mask_pred.float().to(device)
    all_float  = all_mask_pred.float().to(device)
    vis_float  = vis_mask_pred.float().to(device)
    vis_count  = vis_float.sum(dim=-1).clamp(min=1)  # (B,) — for mean

    last_txt_norm = _masked_norm(final_p_gt, txt_float)
    last_all_norm = _masked_norm(final_p_gt, all_float)

    kl_vision = np.zeros((B, n_layers), dtype=np.float32)
    kl_text   = np.zeros((B, n_layers), dtype=np.float32)
    kl_all    = np.zeros((B, n_layers), dtype=np.float32)

    for l in range(n_layers):
        hs_pred  = layer_hs[l][:, :-1, :].to(device)
        logits_l = lm_head(final_norm(hs_pred)).float()
        del hs_pred
        log_p_l = logits_l.log_softmax(dim=-1)    # (B, S-1, V)
        del logits_l

        # Vision KL: full-vocab KL per position, mean over vision positions
        p_l_full   = log_p_l.exp()                                          # (B, S-1, V)
        kl_per_pos = (p_l_full * (log_p_l - final_log_p)).sum(dim=-1)      # (B, S-1)
        del p_l_full
        kl_vision[:, l] = (
            (kl_per_pos * vis_float).sum(dim=-1) / vis_count
        ).cpu().numpy()
        del kl_per_pos

        # Text / All KL: Zero over-positions formula (gather GT, normalize, KL)
        p_l_gt = log_p_l.gather(2, target_ids.unsqueeze(-1)).squeeze(-1).exp()  # (B, S-1)
        del log_p_l

        p_l_txt_norm = _masked_norm(p_l_gt, txt_float)
        kl_text[:, l] = (
            p_l_txt_norm.clamp(eps) *
            (p_l_txt_norm.clamp(eps) / last_txt_norm.clamp(eps)).log()
        ).sum(dim=-1).cpu().numpy()

        p_l_all_norm = _masked_norm(p_l_gt, all_float)
        kl_all[:, l] = (
            p_l_all_norm.clamp(eps) *
            (p_l_all_norm.clamp(eps) / last_all_norm.clamp(eps)).log()
        ).sum(dim=-1).cpu().numpy()

        del p_l_gt, p_l_txt_norm, p_l_all_norm
        torch.cuda.empty_cache()

    del layer_hs, final_log_p, final_p_gt, last_txt_norm, last_all_norm
    del txt_float, all_float, vis_float, vis_count
    torch.cuda.empty_cache()
    return {"vision": kl_vision, "text": kl_text, "all": kl_all}


# ---------------------------------------------------------------------------
# 5. Text-only KL computation (Path A)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_kl_text_only_batch(
    model,
    batch: dict,
    n_layers: int,
    device: str,
    lm_head,
    final_norm,
) -> dict[str, np.ndarray]:
    """Path A: forward through model.language_model only (no vision pipeline).

    KL: Zero over-positions formula on all valid (non-padding) positions.
    Returns dict 'text_only_kl' of shape (B, n_layers).
    """
    assert not (batch["input_ids"] == _IMAGE_TOKEN_INDEX).any().item(), (
        "text_only batch contains IMAGE_TOKEN_INDEX (32000). "
        "Questions must have <image> stripped before tokenisation."
    )
    assert "pixel_values" not in batch, (
        "text_only batch must not contain pixel_values."
    )

    B = batch["input_ids"].shape[0]

    # Capture hidden states via hooks — same approach as vlm mode for consistency
    layer_hs: dict[int, torch.Tensor] = {}

    def make_hook(l: int):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, (tuple, list)) else (
                out.hidden_states if hasattr(out, "hidden_states") else out
            )
            layer_hs[l] = hs.detach().cpu()
        return hook

    llm_layers = get_llm_layers(model)
    hooks = [llm_layers[l].register_forward_hook(make_hook(l)) for l in range(n_layers)]
    try:
        _ = model.language_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
    finally:
        for h in hooks:
            h.remove()

    target_ids     = batch["input_ids"][:, 1:].to(device)
    all_mask_pred  = (batch["attention_mask"][:, :-1] == 1).cpu()
    all_float      = all_mask_pred.float().to(device)

    eps = _EPS

    def _masked_norm(p: torch.Tensor, mask_f: torch.Tensor) -> torch.Tensor:
        pm = p * mask_f
        return pm / (pm.sum(dim=-1, keepdim=True) + eps)

    final_hs_pred = layer_hs[n_layers - 1][:, :-1, :].to(device)
    final_logits  = lm_head(final_norm(final_hs_pred)).float()
    del final_hs_pred
    final_p_gt   = final_logits.softmax(-1).gather(
        2, target_ids.unsqueeze(-1)
    ).squeeze(-1)
    del final_logits
    last_all_norm = _masked_norm(final_p_gt, all_float)

    kl_text_only = np.zeros((B, n_layers), dtype=np.float32)

    for l in range(n_layers):
        hs_l     = layer_hs[l][:, :-1, :].to(device)
        logits_l = lm_head(final_norm(hs_l)).float()
        del hs_l
        p_l_gt   = logits_l.softmax(-1).gather(
            2, target_ids.unsqueeze(-1)
        ).squeeze(-1)
        del logits_l

        p_l_norm = _masked_norm(p_l_gt, all_float)
        kl = (
            p_l_norm.clamp(eps) *
            (p_l_norm.clamp(eps) / last_all_norm.clamp(eps)).log()
        ).sum(dim=-1)
        kl_text_only[:, l] = kl.cpu().numpy()

        del p_l_gt, p_l_norm, kl
        torch.cuda.empty_cache()

    del layer_hs, final_p_gt, last_all_norm, all_float
    torch.cuda.empty_cache()
    return {"text_only_kl": kl_text_only}


# ---------------------------------------------------------------------------
# 6. Transition point detection
# ---------------------------------------------------------------------------

def detect_transition(kl_mean: np.ndarray) -> int:
    """First layer where KL drops below 50% of peak with ≤1 reversal."""
    max_kl    = float(kl_mean.max())
    threshold = max_kl * 0.5
    n         = len(kl_mean)

    for l in range(n):
        if kl_mean[l] < threshold:
            reversals = sum(1 for j in range(l, n - 1) if kl_mean[j + 1] > kl_mean[j])
            if reversals <= 1:
                return l

    return int(np.argmin(kl_mean))


def detect_transition_per_sample(kl_curves: np.ndarray) -> np.ndarray:
    """Apply detect_transition independently to each row of kl_curves.

    Input  : (n_samples, n_layers)
    Output : (n_samples,) int array of transition layer indices
    """
    n_samples = kl_curves.shape[0]
    result = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        result[i] = detect_transition(kl_curves[i])
    return result


def _count_tp_fallbacks(kl_curves: np.ndarray) -> int:
    """Count samples where detect_transition falls back to argmin (no clean threshold crossing)."""
    n_fallback = 0
    n = kl_curves.shape[1]
    for curve in kl_curves:
        threshold = float(curve.max()) * 0.5
        found = False
        for l in range(n):
            if curve[l] < threshold:
                reversals = sum(1 for j in range(l, n - 1) if curve[j + 1] > curve[j])
                if reversals <= 1:
                    found = True
                    break
        if not found:
            n_fallback += 1
    return n_fallback


def _transition_dist_stats(tp_per_sample: np.ndarray) -> dict:
    """Distribution statistics for an array of per-sample integer transition layers."""
    vals, counts = np.unique(tp_per_sample, return_counts=True)
    mode = int(vals[counts.argmax()])
    return {
        "median": int(np.median(tp_per_sample)),
        "q25":    float(np.percentile(tp_per_sample, 25)),
        "q75":    float(np.percentile(tp_per_sample, 75)),
        "min":    int(tp_per_sample.min()),
        "max":    int(tp_per_sample.max()),
        "mode":   mode,
    }


def _print_transition_summary(
    rows: list[tuple[str, int, int, float, float]],
    recommended_split: int,
) -> None:
    """Print formatted comparison table: mean-curve TP vs per-sample distribution."""
    w = 82
    print("\n" + "=" * w)
    print("  Transition point: mean-curve vs per-sample distribution  (v3.1)")
    print("=" * w)
    hdr = f"  {'Curve':<22}  {'Mean-curve TP':>14}  {'Median TP':>10}  {'IQR':>12}  {'Consistent?':>12}"
    print(hdr)
    print("-" * w)
    for name, mean_tp, median_tp, q25, q75 in rows:
        consistent = "YES ✓" if abs(mean_tp - median_tp) <= 2 else "NO ✗ (>2 layers)"
        iqr_str = f"[{q25:.0f}, {q75:.0f}]"
        print(f"  {name:<22}  {mean_tp:>14}  {median_tp:>10}  {iqr_str:>12}  {consistent:>12}")
    print("-" * w)
    print(f"  Recommended split_layer (median of per-sample medians) : L{recommended_split}")
    print("=" * w + "\n")


# ---------------------------------------------------------------------------
# 7. Plotting
# ---------------------------------------------------------------------------

def _single_curve_fig(
    kl_mean: np.ndarray,
    tp: int,
    label: str,
    color: str,
    title: str,
    model_name: str,
    out_path: Path,
) -> None:
    n = len(kl_mean)
    x = list(range(n))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, kl_mean, "o-", color=color, lw=2, ms=4, label=label)
    ax.axvline(tp, color=color, ls="--", lw=1.5, alpha=0.8, label=f"Transition L{tp}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in x], fontsize=7, rotation=60, ha="right")
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("KL divergence", fontsize=11)
    ax.set_title(f"{title} ({model_name})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


def plot_kl_vision_full_vocab(kl_mean: np.ndarray, tp: int, model_name: str, out_path: Path) -> None:
    _single_curve_fig(
        kl_mean, tp,
        label="Vision tokens — full-vocab KL(p_l || p_final)",
        color="#2196F3",
        title="Path B-vision: full-vocab KL (mean over vision positions)",
        model_name=model_name,
        out_path=out_path,
    )


def plot_kl_text_zero_vlm(kl_mean: np.ndarray, tp: int, model_name: str, out_path: Path) -> None:
    _single_curve_fig(
        kl_mean, tp,
        label="VLM text positions — Zero over-positions KL",
        color="#F44336",
        title="Path B-text: Zero KL on VLM text positions",
        model_name=model_name,
        out_path=out_path,
    )


def plot_kl_text_zero_llm_only(kl_mean: np.ndarray, tp: int, model_name: str, out_path: Path) -> None:
    _single_curve_fig(
        kl_mean, tp,
        label="LLM-only text — Zero over-positions KL (no image)",
        color="#4CAF50",
        title="Path A: Zero KL on LLM-only text input",
        model_name=model_name,
        out_path=out_path,
    )


def plot_kl_per_sample(
    kl_curves: np.ndarray,
    kl_mean: np.ndarray,
    label: str,
    color: str,
    model_name: str,
    out_path: Path,
    kl_median: np.ndarray | None = None,
    tp_median: int | None = None,
    tp_q25: float | None = None,
    tp_q75: float | None = None,
    tp_per_sample: np.ndarray | None = None,
    split_layer: int | None = None,
) -> None:
    n = kl_curves.shape[1]
    x = list(range(n))
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(len(kl_curves)):
        ax.plot(x, kl_curves[i], color="gray", alpha=0.12, lw=0.7)
    # scatter per-sample transition points
    if tp_per_sample is not None:
        scatter_y = kl_curves[np.arange(len(kl_curves)), tp_per_sample]
        ax.scatter(tp_per_sample, scatter_y, color=color, alpha=0.25, s=8, zorder=3,
                   label="Per-sample transition (scatter)")
    ax.plot(x, kl_mean, color=color, lw=2.5, label=f"Mean KL ({label})")
    if kl_median is not None:
        median_color = "#FF6F00" if color == "#2196F3" else "#1565C0"
        ax.plot(x, kl_median, color=median_color, lw=2.0, ls="--",
                label=f"Median KL ({label})", alpha=0.9)
    if tp_q25 is not None and tp_q75 is not None:
        ax.axvspan(tp_q25 - 0.4, tp_q75 + 0.4, alpha=0.10, color=color,
                   label=f"Transition IQR [{tp_q25:.0f}, {tp_q75:.0f}]")
    if tp_median is not None:
        ax.axvline(tp_median, color=color, ls=":", lw=1.8, alpha=0.85,
                   label=f"Median transition L{tp_median}")
    if split_layer is not None:
        ax.axvline(split_layer, color="#C62828", ls="-", lw=2.0, alpha=0.85,
                   label=f"split_layer=L{split_layer}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in x], fontsize=7, rotation=60, ha="right")
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("KL divergence (log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title(f"Per-sample KL curves — {label} ({model_name})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


def plot_transition_histogram(
    tp_per_sample: np.ndarray,
    curve_label: str,
    color: str,
    mean_curve_tp: int,
    dist: dict,
    model_name: str,
    out_path: Path,
    split_layer: int | None = None,
) -> None:
    """Histogram of per-sample transition layers with median / IQR / mean-curve TP markers."""
    max_layer = int(max(tp_per_sample.max(), mean_curve_tp))
    if split_layer is not None:
        max_layer = max(max_layer, split_layer)
    bins = np.arange(-0.5, max_layer + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(12, 4))
    n_valid = len(tp_per_sample)
    ax.hist(tp_per_sample, bins=bins, color=color, alpha=0.65, edgecolor="white", linewidth=0.5)

    ax.axvspan(dist["q25"] - 0.4, dist["q75"] + 0.4, alpha=0.12, color=color,
               label=f"IQR [{dist['q25']:.0f}, {dist['q75']:.0f}]")
    ax.axvline(dist["median"], color=color, ls="-", lw=2.2,
               label=f"Per-sample median L{dist['median']}")
    ax.axvline(mean_curve_tp, color="#D32F2F", ls="--", lw=2.0, alpha=0.85,
               label=f"Mean-curve TP L{mean_curve_tp}")
    if split_layer is not None:
        ax.axvline(split_layer, color="#C62828", ls="-", lw=2.0, alpha=0.85,
                   label=f"split_layer=L{split_layer}")

    ax.set_xlabel("Transition layer index", fontsize=11)
    ax.set_ylabel("Sample count", fontsize=11)
    ax.set_title(
        f"Per-sample transition distribution — {curve_label} ({model_name})\n"
        f"n={n_valid}, median=L{dist['median']}, mode=L{dist['mode']}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


def plot_transition_hist_combined(
    tp_vision: np.ndarray,
    tp_text_vlm: np.ndarray,
    tp_text_llm: np.ndarray | None,
    dist_vision: dict,
    dist_text_vlm: dict,
    dist_text_llm: dict | None,
    model_name: str,
    split_layer: int,
    out_path: Path,
) -> None:
    """Combined histogram overlay of vision / VLM-text / LLM-only transition distributions."""
    all_arrs = [tp_vision, tp_text_vlm]
    if tp_text_llm is not None:
        all_arrs.append(tp_text_llm)
    max_layer = max(int(a.max()) for a in all_arrs)
    max_layer = max(max_layer, split_layer)
    bins = np.arange(-0.5, max_layer + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(13, 5))

    specs = [
        (tp_vision,   dist_vision,   "#2196F3", "Vision (full-vocab)"),
        (tp_text_vlm, dist_text_vlm, "#F44336", "VLM-text (Zero KL)"),
    ]
    if tp_text_llm is not None and dist_text_llm is not None:
        specs.append((tp_text_llm, dist_text_llm, "#4CAF50", "LLM-only text (Zero KL)"))

    for arr, dist, color, label in specs:
        ax.hist(arr, bins=bins, alpha=0.40, color=color, edgecolor="white",
                linewidth=0.4, label=label)
        ax.axvline(dist["median"], color=color, ls="-", lw=2.0, alpha=0.85,
                   label=f"{label} median=L{dist['median']}")

    ax.axvline(split_layer, color="#C62828", ls="-", lw=2.2, alpha=0.9,
               label=f"split_layer=L{split_layer}")

    ax.set_xlabel("Transition layer index", fontsize=11)
    ax.set_ylabel("Sample count", fontsize=11)
    n = len(tp_vision)
    ax.set_title(
        f"Combined per-sample transition distributions ({model_name}, n={n})", fontsize=11
    )
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


def plot_kl_triangulation(
    kl_mean_vision: np.ndarray,
    kl_mean_text_vlm: np.ndarray,
    kl_mean_text_llm: np.ndarray,
    tp_vision: int,
    tp_text_vlm: int,
    tp_text_llm: int,
    model_name: str,
    split_layer: int,
    out_path: Path,
) -> None:
    """Three-way triangulation figure — paper Section 3 main figure candidate.

    Each curve normalized to [0, 1] by its own maximum.
    """
    n = len(kl_mean_vision)
    x = list(range(n))

    def _norm(arr: np.ndarray) -> np.ndarray:
        m = arr.max()
        return arr / m if m > 0 else arr

    v_n = _norm(kl_mean_vision)
    t_n = _norm(kl_mean_text_vlm)
    l_n = _norm(kl_mean_text_llm)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, v_n, "o-",  color="#2196F3", lw=2, ms=4,
            label="Vision (full-vocab KL) — Path B-vision")
    ax.plot(x, t_n, "s--", color="#F44336", lw=2, ms=4,
            label="VLM-text (Zero KL) — Path B-text")
    ax.plot(x, l_n, "^:",  color="#4CAF50", lw=2, ms=4,
            label="LLM-only text (Zero KL) — Path A")

    ax.axhline(0.5, color="gray", ls=":", lw=1.0, alpha=0.6,
               label="50% convergence threshold")
    ax.axvline(split_layer, color="#C62828", ls="--", lw=2.0, alpha=0.9,
               label=f"split = L{split_layer}")

    ax.axvline(tp_vision,   color="#2196F3", ls=":", lw=1.5, alpha=0.7,
               label=f"Vision transition L{tp_vision}")
    ax.axvline(tp_text_vlm, color="#F44336", ls=":", lw=1.5, alpha=0.7,
               label=f"VLM-text transition L{tp_text_vlm}")
    ax.axvline(tp_text_llm, color="#4CAF50", ls=":", lw=1.5, alpha=0.7,
               label=f"LLM-only transition L{tp_text_llm}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in x], fontsize=7, rotation=60, ha="right")
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Normalized KL (each curve → [0, 1])", fontsize=11)
    ax.set_title(
        f"Layer-wise prediction convergence — three-way triangulation ({model_name})",
        fontsize=12,
    )
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_ylim(-0.05, 1.12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


def plot_kl_vs_layer_dual(
    kl_mean_vision: np.ndarray,
    kl_mean_text: np.ndarray,
    kl_mean_all: np.ndarray,
    tp_vision: int,
    tp_text: int,
    model_name: str,
    out_path: Path,
) -> None:
    n = len(kl_mean_vision)
    x = list(range(n))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, kl_mean_vision, "o-",  color="#2196F3", lw=2, ms=4, label="Vision tokens KL")
    ax.plot(x, kl_mean_text,   "s--", color="#F44336", lw=2, ms=4, label="Text tokens KL")
    ax.plot(x, kl_mean_all,    "^:",  color="#4CAF50", lw=1.5, ms=3, alpha=0.7, label="All tokens KL")
    ax.axvline(tp_vision, color="#2196F3", ls="--", lw=1.5, alpha=0.8, label=f"Vision L{tp_vision}")
    ax.axvline(tp_text,   color="#F44336", ls="--", lw=1.5, alpha=0.8, label=f"Text L{tp_text}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in x], fontsize=7, rotation=60, ha="right")
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("KL divergence to final-layer prediction", fontsize=11)
    ax.set_title(f"Logit lens KL divergence ({model_name})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


# ---------------------------------------------------------------------------
# 8. Summary report writer
# ---------------------------------------------------------------------------

def _write_summary_report(
    rows: list[tuple[str, int, int, float, float]],
    recommended_split: int,
    model_name: str,
    mode: str,
    out_dir: Path,
    ts: str,
) -> None:
    """Write stdout transition summary to reports/hsf_status/ as a markdown file."""
    import datetime as _dt
    report_dir = Path(__file__).parent.parent / "reports" / "hsf_status"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"v3_1_transition_summary_{ts}.md"

    lines = [
        f"# v3.1 Transition Distribution Summary",
        f"",
        f"- **Model**: {model_name}",
        f"- **Mode**: {mode}",
        f"- **Output dir**: `{out_dir}`",
        f"- **Generated**: {_dt.datetime.now().isoformat()}",
        f"",
        f"## Transition Distribution",
        f"",
        f"| Curve | Mean-curve TP | Median TP | IQR | Consistent? |",
        f"|-------|:-------------:|:---------:|:---:|:-----------:|",
    ]
    for name, mean_tp, median_tp, q25, q75 in rows:
        consistent = "YES" if abs(mean_tp - median_tp) <= 2 else "NO (>2 layers)"
        lines.append(f"| {name} | L{mean_tp} | L{median_tp} | [{q25:.0f}, {q75:.0f}] | {consistent} |")

    lines += [
        f"",
        f"**Recommended split_layer: L{recommended_split}** (median of per-sample medians)",
        f"",
    ]
    report_path.write_text("\n".join(lines))
    print(f"[report] Saved {report_path}")


# ---------------------------------------------------------------------------
# 9. Analysis runners
# ---------------------------------------------------------------------------

def run_vlm_analysis(
    model_path: str,
    output_dir: str,
    n_samples: int,
    batch_size: int,
    device: str,
) -> None:
    """Path B: full LLaVA forward — vision (full-vocab KL) + text (Zero KL)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_path).name

    samples  = load_samples(n_samples)
    actual_n = len(samples)

    print(f"\n[model] Loading from {model_path} ...")
    t0 = time.time()
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

    n_layers         = len(get_llm_layers(model))
    image_seq_length = int(getattr(model.config, "image_seq_length", _IMAGE_SEQ_LENGTH_DEFAULT))
    lm_head, final_norm = get_lm_head_and_norm(model)

    print(
        f"[model] Loaded in {time.time() - t0:.1f}s — {n_layers} layers "
        f"(image_seq_length={image_seq_length})"
    )

    batches = build_batches(samples, processor, batch_size, device)
    print(f"\n[forward] {len(batches)} batches × up to {batch_size} samples")

    all_kl_vision: list[np.ndarray] = []
    all_kl_text:   list[np.ndarray] = []
    all_kl_all:    list[np.ndarray] = []

    for b_idx, batch in enumerate(batches):
        t0 = time.time()
        kl_dict = compute_kl_batch(
            model, batch, n_layers, device, image_seq_length, lm_head, final_norm
        )
        all_kl_vision.append(kl_dict["vision"])
        all_kl_text.append(kl_dict["text"])
        all_kl_all.append(kl_dict["all"])
        B = batch["input_ids"].shape[0]
        print(f"  [batch {b_idx + 1}/{len(batches)}] {B} samples, {time.time() - t0:.1f}s")

    kl_vision = np.concatenate(all_kl_vision, axis=0)[:actual_n]
    kl_text   = np.concatenate(all_kl_text,   axis=0)[:actual_n]
    kl_all    = np.concatenate(all_kl_all,    axis=0)[:actual_n]

    kl_mean_vision   = kl_vision.mean(axis=0)
    kl_mean_text     = kl_text.mean(axis=0)
    kl_mean_all      = kl_all.mean(axis=0)
    kl_median_vision = np.median(kl_vision, axis=0)
    kl_median_text   = np.median(kl_text,   axis=0)
    kl_median_all    = np.median(kl_all,    axis=0)

    tp_vision = detect_transition(kl_mean_vision)
    tp_text   = detect_transition(kl_mean_text)
    tp_all    = detect_transition(kl_mean_all)

    print(f"\n[results] Transition points: vision=L{tp_vision}  text=L{tp_text}  all=L{tp_all}")

    l_split = 18 if n_layers == 32 else 24
    print(f"\n--- Key KL values at split=L{l_split} ---")
    for nm, mean in [("vision", kl_mean_vision), ("text", kl_mean_text), ("all", kl_mean_all)]:
        ls = min(l_split, n_layers - 1)
        print(f"  {nm:6s}: KL[L{ls}]={mean[ls]:.4f}  KL[final]={mean[-1]:.6f}")

    # --- per-sample transition distribution (v3.1) ---
    print("\n[per-sample] Computing per-sample transition distributions ...")
    tp_vision_ps = detect_transition_per_sample(kl_vision)
    tp_text_ps   = detect_transition_per_sample(kl_text)
    tp_all_ps    = detect_transition_per_sample(kl_all)

    for curve_name, curves in [("vision", kl_vision), ("text_vlm", kl_text), ("all", kl_all)]:
        n_fb = _count_tp_fallbacks(curves)
        if n_fb > 0:
            print(f"  [WARN] {curve_name}: {n_fb}/{actual_n} samples hit argmin fallback in detect_transition")

    dist_vision = _transition_dist_stats(tp_vision_ps)
    dist_text   = _transition_dist_stats(tp_text_ps)
    dist_all    = _transition_dist_stats(tp_all_ps)

    # dump raw matrices so future re-plotting doesn't need forward pass
    np.save(str(out / "kl_matrix_vision.npy"), kl_vision)
    np.save(str(out / "kl_matrix_text.npy"),   kl_text)
    np.save(str(out / "kl_matrix_all.npy"),    kl_all)
    print(f"[npy] Saved kl_matrix_{{vision,text,all}}.npy to {out}/")

    print("\n[plot] Generating figures ...")
    plot_kl_vision_full_vocab(
        kl_mean_vision, tp_vision, model_name,
        out / "fig_kl_vision_full_vocab.pdf",
    )
    plot_kl_text_zero_vlm(
        kl_mean_text, tp_text, model_name,
        out / "fig_kl_text_zero_vlm.pdf",
    )
    plot_kl_per_sample(
        kl_vision, kl_mean_vision,
        "vision (full-vocab)", "#2196F3", model_name,
        out / "fig_kl_per_sample_vision.pdf",
        kl_median=kl_median_vision,
        tp_median=dist_vision["median"],
        tp_q25=dist_vision["q25"],
        tp_q75=dist_vision["q75"],
        tp_per_sample=tp_vision_ps,
        split_layer=l_split,
    )
    plot_kl_per_sample(
        kl_text, kl_mean_text,
        "VLM-text (Zero)", "#F44336", model_name,
        out / "fig_kl_per_sample_text.pdf",
        kl_median=kl_median_text,
        tp_median=dist_text["median"],
        tp_q25=dist_text["q25"],
        tp_q75=dist_text["q75"],
        tp_per_sample=tp_text_ps,
        split_layer=l_split,
    )
    plot_kl_vs_layer_dual(
        kl_mean_vision, kl_mean_text, kl_mean_all,
        tp_vision, tp_text, model_name,
        out / "fig_kl_vs_layer_dual.pdf",
    )
    plot_transition_histogram(
        tp_vision_ps, "vision (full-vocab)", "#2196F3",
        tp_vision, dist_vision, model_name,
        out / "fig_kl_transition_hist_vision.pdf",
        split_layer=l_split,
    )
    plot_transition_histogram(
        tp_text_ps, "VLM-text (Zero KL)", "#F44336",
        tp_text, dist_text, model_name,
        out / "fig_kl_transition_hist_text_vlm.pdf",
        split_layer=l_split,
    )
    plot_transition_histogram(
        tp_all_ps, "all tokens (Zero KL)", "#4CAF50",
        tp_all, dist_all, model_name,
        out / "fig_kl_transition_hist_all.pdf",
        split_layer=l_split,
    )
    plot_transition_hist_combined(
        tp_vision_ps, tp_text_ps, None,
        dist_vision, dist_text, None,
        model_name, l_split,
        out / "fig_kl_transition_hist_combined.pdf",
    )

    # stdout summary (vlm curves only; triangulation mode adds text_only row)
    vlm_rows = [
        ("vision (full-vocab)",  tp_vision, dist_vision["median"], dist_vision["q25"], dist_vision["q75"]),
        ("VLM-text (Zero KL)",   tp_text,   dist_text["median"],   dist_text["q25"],   dist_text["q75"]),
        ("all tokens (Zero KL)", tp_all,    dist_all["median"],    dist_all["q25"],    dist_all["q75"]),
    ]
    vlm_medians = [dist_vision["median"], dist_text["median"], dist_all["median"]]
    recommended_vlm = int(np.median(vlm_medians))
    _print_transition_summary(vlm_rows, recommended_vlm)

    # structured v3.1 JSON fields
    v3_1_dist = {
        "vision": {
            "label": "vision (full-vocab)", "n_valid": actual_n,
            "median": dist_vision["median"], "iqr": [int(dist_vision["q25"]), int(dist_vision["q75"])],
            "mode": dist_vision["mode"], "min": dist_vision["min"], "max": dist_vision["max"],
        },
        "text": {
            "label": "VLM-text (Zero KL)", "n_valid": actual_n,
            "median": dist_text["median"], "iqr": [int(dist_text["q25"]), int(dist_text["q75"])],
            "mode": dist_text["mode"], "min": dist_text["min"], "max": dist_text["max"],
        },
        "all": {
            "label": "all tokens (Zero KL)", "n_valid": actual_n,
            "median": dist_all["median"], "iqr": [int(dist_all["q25"]), int(dist_all["q75"])],
            "mode": dist_all["mode"], "min": dist_all["min"], "max": dist_all["max"],
        },
    }
    v3_recommended = {
        "from_median_vision":  dist_vision["median"],
        "from_median_text":    dist_text["median"],
        "from_median_all":     dist_all["median"],
        "from_consensus":      recommended_vlm,
        "rationale": (
            "median of per-sample medians across vision/text/all; "
            "text-mode median most consistent with LLM-only baseline"
        ),
    }

    summary = {
        "kl_version": "v3.1",
        "kl_method_vision": "full-vocab KL(p_l || p_final) per position, mean over vision positions",
        "kl_method_text":   "over-positions (Zero paper): gather GT probs, normalize, KL",
        "mode": "vlm",
        "model_path": model_path,
        "model_name": model_name,
        "data_source": "LLaVA-1.5-665k COCO subset",
        "n_samples": actual_n,
        "n_layers": n_layers,
        "batch_size": batch_size,
        "image_token_index": _IMAGE_TOKEN_INDEX,
        "image_seq_length": image_seq_length,
        "kl_mean_vision":          kl_mean_vision.tolist(),
        "kl_mean_text":            kl_mean_text.tolist(),
        "kl_mean_all":             kl_mean_all.tolist(),
        "kl_median_vision":        kl_median_vision.tolist(),
        "kl_median_text":          kl_median_text.tolist(),
        "transition_point_vision": int(tp_vision),
        "transition_point_text":   int(tp_text),
        "transition_point_all":    int(tp_all),
        # flat per-sample fields kept for backward compat
        "transition_layers_per_sample_vision": tp_vision_ps.tolist(),
        "transition_layers_per_sample_text":   tp_text_ps.tolist(),
        "transition_layers_per_sample_all":    tp_all_ps.tolist(),
        "transition_median_vision":  dist_vision["median"],
        "transition_q25_vision":     dist_vision["q25"],
        "transition_q75_vision":     dist_vision["q75"],
        "transition_min_vision":     dist_vision["min"],
        "transition_max_vision":     dist_vision["max"],
        "transition_mode_vision":    dist_vision["mode"],
        "transition_median_text":    dist_text["median"],
        "transition_q25_text":       dist_text["q25"],
        "transition_q75_text":       dist_text["q75"],
        "transition_min_text":       dist_text["min"],
        "transition_max_text":       dist_text["max"],
        "transition_mode_text":      dist_text["mode"],
        "transition_median_all":     dist_all["median"],
        "transition_q25_all":        dist_all["q25"],
        "transition_q75_all":        dist_all["q75"],
        "transition_min_all":        dist_all["min"],
        "transition_max_all":        dist_all["max"],
        "transition_mode_all":       dist_all["mode"],
        "kl_matrix_vision_path":     str(out / "kl_matrix_vision.npy"),
        "kl_matrix_text_path":       str(out / "kl_matrix_text.npy"),
        "kl_matrix_all_path":        str(out / "kl_matrix_all.npy"),
        # v3.1 structured additions
        "v3_1_transition_distribution": v3_1_dist,
        "per_sample_npy_files": [
            str(out / "kl_matrix_vision.npy"),
            str(out / "kl_matrix_text.npy"),
            str(out / "kl_matrix_all.npy"),
        ],
        "v3_recommended_split_layer": v3_recommended,
    }
    json_path = out / "kl_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[json] Saved {json_path}")

    # write summary report
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_summary_report(vlm_rows, recommended_vlm, model_name, "vlm", out, ts)

    print(f"\n[done] All outputs written to {out}/")


def run_text_only_analysis(
    model_path: str,
    output_dir: str,
    n_samples: int,
    batch_size: int,
    device: str,
) -> None:
    """Path A: forward through model.language_model only (no vision pipeline)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_path).name

    samples  = load_samples(n_samples)
    actual_n = len(samples)

    print(f"\n[model] Loading from {model_path} ...")
    t0 = time.time()
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

    n_layers = len(get_llm_layers(model))
    lm_head, final_norm = get_lm_head_and_norm(model)

    print(f"[model] Loaded in {time.time() - t0:.1f}s — {n_layers} layers")
    print(f"[mode]  text_only — forward through model.language_model (no vision pipeline)")

    batches = build_text_batches(samples, processor, batch_size, device)
    print(f"\n[forward] {len(batches)} batches × up to {batch_size} samples")

    all_kl: list[np.ndarray] = []

    for b_idx, batch in enumerate(batches):
        t0 = time.time()
        kl_dict = compute_kl_text_only_batch(
            model, batch, n_layers, device, lm_head, final_norm
        )
        all_kl.append(kl_dict["text_only_kl"])
        B = batch["input_ids"].shape[0]
        print(f"  [batch {b_idx + 1}/{len(batches)}] {B} samples, {time.time() - t0:.1f}s")

    kl_text_only = np.concatenate(all_kl, axis=0)[:actual_n]
    kl_mean      = kl_text_only.mean(axis=0)
    kl_median    = np.median(kl_text_only, axis=0)
    tp           = detect_transition(kl_mean)

    print(f"\n[results] Transition point (LLM-only): L{tp}")

    l_split = 18 if n_layers == 32 else 24
    ls = min(l_split, n_layers - 1)
    print(f"  KL[L{ls}]={kl_mean[ls]:.4f}  KL[final]={kl_mean[-1]:.6f}")

    # --- per-sample transition distribution (v3.1) ---
    print("\n[per-sample] Computing per-sample transition distribution ...")
    tp_ps   = detect_transition_per_sample(kl_text_only)
    n_fb    = _count_tp_fallbacks(kl_text_only)
    if n_fb > 0:
        print(f"  [WARN] text_only: {n_fb}/{actual_n} samples hit argmin fallback in detect_transition")
    dist    = _transition_dist_stats(tp_ps)

    # dump raw matrix
    np.save(str(out / "kl_matrix_text_only.npy"), kl_text_only)
    print(f"[npy] Saved kl_matrix_text_only.npy to {out}/")

    print("\n[plot] Generating figures ...")
    plot_kl_text_zero_llm_only(
        kl_mean, tp, model_name,
        out / "fig_kl_text_zero_llm_only.pdf",
    )
    plot_kl_per_sample(
        kl_text_only, kl_mean,
        "LLM-only text (Zero)", "#4CAF50", model_name,
        out / "fig_kl_per_sample.pdf",
        kl_median=kl_median,
        tp_median=dist["median"],
        tp_q25=dist["q25"],
        tp_q75=dist["q75"],
        tp_per_sample=tp_ps,
        split_layer=l_split,
    )
    plot_transition_histogram(
        tp_ps, "LLM-only text (Zero KL)", "#4CAF50",
        tp, dist, model_name,
        out / "fig_kl_transition_hist_text_only.pdf",
        split_layer=l_split,
    )

    # stdout summary (text_only run only)
    text_rows = [("LLM-only text (Zero KL)", tp, dist["median"], dist["q25"], dist["q75"])]
    _print_transition_summary(text_rows, dist["median"])

    # structured v3.1 JSON fields
    v3_1_dist = {
        "text_only": {
            "label": "LLM-only text (Zero KL)", "n_valid": actual_n,
            "median": dist["median"], "iqr": [int(dist["q25"]), int(dist["q75"])],
            "mode": dist["mode"], "min": dist["min"], "max": dist["max"],
        },
    }
    v3_recommended = {
        "from_median_text_only": dist["median"],
        "from_consensus":        dist["median"],
        "rationale": "LLM-only text mode provides baseline split recommendation",
    }

    summary = {
        "kl_version": "v3.1",
        "kl_method":  "over-positions (Zero paper): gather GT probs, normalize, KL",
        "mode": "text_only",
        "model_path": model_path,
        "model_name": model_name,
        "data_source": "LLaVA-1.5-665k COCO subset (text only, <image> stripped)",
        "n_samples": actual_n,
        "n_layers": n_layers,
        "batch_size": batch_size,
        "kl_mean_text_only":          kl_mean.tolist(),
        "kl_median_text_only":        kl_median.tolist(),
        "transition_point_text_only": int(tp),
        # flat per-sample fields (backward compat)
        "transition_layers_per_sample_text_only": tp_ps.tolist(),
        "transition_median_text_only":  dist["median"],
        "transition_q25_text_only":     dist["q25"],
        "transition_q75_text_only":     dist["q75"],
        "transition_min_text_only":     dist["min"],
        "transition_max_text_only":     dist["max"],
        "transition_mode_text_only":    dist["mode"],
        "kl_matrix_text_only_path":     str(out / "kl_matrix_text_only.npy"),
        # v3.1 structured additions
        "v3_1_transition_distribution": v3_1_dist,
        "per_sample_npy_files": [str(out / "kl_matrix_text_only.npy")],
        "v3_recommended_split_layer": v3_recommended,
    }
    json_path = out / "kl_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[json] Saved {json_path}")

    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_summary_report(text_rows, dist["median"], model_name, "text_only", out, ts)

    print(f"\n[done] All outputs written to {out}/")


def run_triangulation(
    vlm_json: str,
    text_only_json: str,
    output_dir: str,
) -> None:
    """Generate three-way triangulation figure from two JSON files (no model loading)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(vlm_json) as f:
        vd = json.load(f)
    with open(text_only_json) as f:
        td = json.load(f)

    kl_mean_vision   = np.array(vd["kl_mean_vision"])
    kl_mean_text_vlm = np.array(vd["kl_mean_text"])
    kl_mean_text_llm = np.array(td["kl_mean_text_only"])
    tp_vision   = int(vd["transition_point_vision"])
    tp_text_vlm = int(vd["transition_point_text"])
    tp_text_llm = int(td["transition_point_text_only"])
    model_name  = vd.get("model_name", "LLaVA-1.5")
    n_layers    = int(vd["n_layers"])
    split_layer = 18 if n_layers == 32 else 24

    print(f"[triangulation] {model_name}  n_layers={n_layers}  split=L{split_layer}")
    print(f"  Path B-vision transition:   L{tp_vision}")
    print(f"  Path B-text  transition:    L{tp_text_vlm}")
    print(f"  Path A (LLM-only) transition: L{tp_text_llm}")

    all_tp = [tp_vision, tp_text_vlm, tp_text_llm]
    gap = max(all_tp) - min(all_tp)
    consistent = gap <= 3
    print(
        f"\n  Transition gap: {gap} layers  "
        f"[{'CONSISTENT ✓ (≤3)' if consistent else 'WARNING: gap > 3 — auditor review needed'}]"
    )
    if not consistent:
        print("  [STOP] Gap > 5 layers — escalating to auditor per red-line rule 10.")

    # Combined per-sample summary table + histogram (only if v3.1 JSON has per-sample fields)
    if vd.get("kl_version", "v3") >= "v3.1" and td.get("kl_version", "v3") >= "v3.1":
        rows = [
            (
                "vision (full-vocab)",
                tp_vision,
                int(vd["transition_median_vision"]),
                float(vd["transition_q25_vision"]),
                float(vd["transition_q75_vision"]),
            ),
            (
                "VLM-text (Zero KL)",
                tp_text_vlm,
                int(vd["transition_median_text"]),
                float(vd["transition_q25_text"]),
                float(vd["transition_q75_text"]),
            ),
            (
                "all tokens (Zero KL)",
                int(vd["transition_point_all"]),
                int(vd["transition_median_all"]),
                float(vd["transition_q25_all"]),
                float(vd["transition_q75_all"]),
            ),
            (
                "LLM-only text (Zero KL)",
                tp_text_llm,
                int(td["transition_median_text_only"]),
                float(td["transition_q25_text_only"]),
                float(td["transition_q75_text_only"]),
            ),
        ]
        all_medians = [r[2] for r in rows]
        recommended = int(np.median(all_medians))
        _print_transition_summary(rows, recommended)

        # Load per-sample arrays from JSON and generate text_llm histogram + combined
        tp_vision_ps   = np.array(vd.get("transition_layers_per_sample_vision", []), dtype=np.int32)
        tp_text_vlm_ps = np.array(vd.get("transition_layers_per_sample_text",   []), dtype=np.int32)
        tp_text_llm_ps = np.array(td.get("transition_layers_per_sample_text_only", []), dtype=np.int32)

        dist_vision_ps  = _transition_dist_stats(tp_vision_ps)   if len(tp_vision_ps) else {}
        dist_text_vlm_ps = _transition_dist_stats(tp_text_vlm_ps) if len(tp_text_vlm_ps) else {}
        dist_text_llm_ps = _transition_dist_stats(tp_text_llm_ps) if len(tp_text_llm_ps) else {}

        if len(tp_text_llm_ps):
            plot_transition_histogram(
                tp_text_llm_ps, "LLM-only text (Zero KL)", "#4CAF50",
                tp_text_llm, dist_text_llm_ps, model_name,
                out / "fig_kl_transition_hist_text_llm.pdf",
                split_layer=split_layer,
            )
        if len(tp_vision_ps) and len(tp_text_vlm_ps) and len(tp_text_llm_ps):
            plot_transition_hist_combined(
                tp_vision_ps, tp_text_vlm_ps, tp_text_llm_ps,
                dist_vision_ps, dist_text_vlm_ps, dist_text_llm_ps,
                model_name, split_layer,
                out / "fig_kl_transition_hist_combined_3way.pdf",
            )

        import datetime as _dt
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        _write_summary_report(rows, recommended, model_name, "triangulation", out, ts)
    else:
        print("\n  [NOTE] Per-sample distribution stats not found in JSON (need v3.1 re-run).")

    plot_kl_triangulation(
        kl_mean_vision, kl_mean_text_vlm, kl_mean_text_llm,
        tp_vision, tp_text_vlm, tp_text_llm,
        model_name, split_layer,
        out / "fig_kl_triangulation.pdf",
    )

    # Absolute-scale three-way figure (log y)
    n = len(kl_mean_vision)
    x = list(range(n))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, kl_mean_vision,   "o-",  color="#2196F3", lw=2, ms=3, label="Vision (full-vocab)")
    ax.plot(x, kl_mean_text_vlm, "s--", color="#F44336", lw=2, ms=3, label="VLM-text (Zero)")
    ax.plot(x, kl_mean_text_llm, "^:",  color="#4CAF50", lw=2, ms=3, label="LLM-only (Zero)")
    ax.axvline(split_layer, color="#C62828", ls="--", lw=2.0, label=f"split=L{split_layer}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in x], fontsize=7, rotation=60, ha="right")
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Mean KL (log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title(f"Three-way KL — absolute scale ({model_name})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    abs_path = out / "fig_kl_three_way_absolute.pdf"
    fig.savefig(abs_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {abs_path}")

    summary = {
        "mode": "triangulation",
        "model_name": model_name,
        "n_layers": n_layers,
        "split_layer": split_layer,
        "tp_vision": tp_vision,
        "tp_text_vlm": tp_text_vlm,
        "tp_text_llm": tp_text_llm,
        "transition_gap": gap,
        "consistent_le3": consistent,
        "vlm_json_source": vlm_json,
        "text_only_json_source": text_only_json,
    }
    json_path = out / "kl_triangulation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[json] Saved {json_path}")
    print(f"\n[done] Triangulation outputs written to {out}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--mode",
        choices=["vlm", "text_only", "triangulation"],
        default="vlm",
        help=(
            "vlm: Path B — full LLaVA forward with images; "
            "text_only: Path A — LLM-only forward, no images; "
            "triangulation: generate combined figure from two JSON files (no model needed)."
        ),
    )
    ap.add_argument("--model",    help="Local path to LLaVA-1.5 model dir (vlm/text_only).")
    ap.add_argument("--output",   default="results/analysis_kl_v3/")
    ap.add_argument("--n_samples",  type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--vlm_json",       help="Path to vlm kl_summary.json (triangulation mode).")
    ap.add_argument("--text_only_json", help="Path to text_only kl_summary.json (triangulation mode).")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.mode == "triangulation":
        if not args.vlm_json or not args.text_only_json:
            raise ValueError("--mode triangulation requires --vlm_json and --text_only_json")
        run_triangulation(args.vlm_json, args.text_only_json, args.output)

    elif args.mode == "text_only":
        if not args.model:
            raise ValueError("--mode text_only requires --model")
        run_text_only_analysis(
            model_path=args.model,
            output_dir=args.output,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            device=args.device,
        )

    else:  # vlm (default)
        if not args.model:
            raise ValueError("--mode vlm requires --model")
        run_vlm_analysis(
            model_path=args.model,
            output_dir=args.output,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            device=args.device,
        )

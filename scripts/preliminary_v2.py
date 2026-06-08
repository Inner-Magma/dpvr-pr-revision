#!/usr/bin/env python
"""scripts/preliminary_v2.py — CKA preliminary analysis, Task 6 redo.

Addresses four v1 rejection issues:
1. Real data: 100-200 COCO samples from LLaVA-1.5-665k (not 8 synthetic colour blocks).
2. Correct figure: adjacent-layer CKA dual curves — vision_CKA(L_i, L_{i+1}) and
   text_CKA(L_i, L_{i+1}) overlaid (paper Section 3.4 main figure).
3. Fixed vision-token mask: in transformers 4.57.1 the processor already writes
   IMAGE_SEQ_LENGTH (576) copies of IMAGE_TOKEN_INDEX (32000) into input_ids before
   any model forward; LlavaModel.forward uses masked_scatter to replace those positions
   with image patch features — sequence length does NOT change.
   Mask = (input_ids == 32000); assert count == 576 per sample fails fast if wrong.
4. Feature-space CKA: O(N*p*q) time, O(p*q=64MB) space; self-contained in this file.
   Does NOT touch src/dpvr/analysis/cka.py.
   KL removed (v1 treated hidden-state mean vectors as logits — dimensionally invalid).

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/preliminary_v2.py \\
        --model /path/to/models/llava-1.5-7b-hf \\
        --output results/analysis_7b_v2/ \\
        --n_samples 100 \\
        --plot_mode both

Outputs in --output dir:
    fig2_cka_heatmap_vision.pdf   — 32×32 vision CKA across all layer pairs
    fig2_cka_heatmap_text.pdf     — 32×32 text CKA (supplementary)
    fig3_adjacent_cka_dual.pdf    — adjacent-layer CKA curves (supplementary)
    fig3_cka_from_l0_dual.pdf     — paper Fig 3: CKA(L0, Li) drift curves
    analysis_summary.json         — all CKA arrays + run metadata + warnings
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
# Constants — must match the loaded model config
# ---------------------------------------------------------------------------
_IMAGE_TOKEN_INDEX = 32000   # LLaVA-1.5 <image> special token id
_IMAGE_SEQ_LENGTH_DEFAULT = 576  # fallback; real value read from model.config at runtime

_DATA_ROOT = Path("/path/to/llava-1.5-665k-instructions")


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_samples(n_samples: int, seed: int = 42) -> list[dict]:
    """Load n_samples diverse image-text pairs from LLaVA-665k COCO subset.

    Only COCO images are locally available on this server; other sources
    (VG, GQA, OCR-VQA, TextVQA) are filtered out.
    """
    json_path = _DATA_ROOT / "llava_v1_5_mix665k.json"
    print(f"[data] Loading {json_path} ...")
    with open(json_path) as f:
        all_data = json.load(f)

    coco_entries = [e for e in all_data if e.get("image", "").startswith("coco/")]
    print(f"[data] {len(coco_entries):,} COCO entries in dataset.")

    random.seed(seed)
    # Over-sample candidates to allow for missing / corrupted images
    candidates = random.sample(coco_entries, min(n_samples * 5, len(coco_entries)))

    samples: list[dict] = []
    for entry in candidates:
        img_path = _DATA_ROOT / entry["image"]
        if not img_path.exists():
            continue
        # First human turn; strip <image> tag to get the bare question text
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
    """Tokenize + pad samples into model-ready batches, move to device."""
    batches = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        images = [s["image"] for s in chunk]
        # Standard LLaVA-1.5 chat format
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


# ---------------------------------------------------------------------------
# 2. Vision / text token masks
# ---------------------------------------------------------------------------

def build_vision_text_masks(
    input_ids: torch.LongTensor,       # (B, S) on any device
    attention_mask: torch.LongTensor,  # (B, S)
    image_seq_length: int = _IMAGE_SEQ_LENGTH_DEFAULT,
) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """Return boolean masks (B, S), both on CPU.

    In transformers 4.57.1, LlavaProcessor expands the single <image> placeholder
    into image_seq_length copies of _IMAGE_TOKEN_INDEX (32000) in input_ids.
    LlavaModel.forward then uses masked_scatter to overwrite those positions'
    embeddings with image-patch features — sequence length stays unchanged.

    Hidden states from model.language_model.layers[i] are therefore (B, S, H)
    where S == input_ids.shape[1].

    vis_mask : True where input_ids == IMAGE_TOKEN_INDEX (image positions)
    txt_mask : True where attention_mask==1 AND not a vision position

    Asserts image_seq_length vision tokens per sample; fails fast if the
    processor text format is wrong.
    """
    ids = input_ids.cpu()
    amask = attention_mask.cpu()

    vis_mask = (ids == _IMAGE_TOKEN_INDEX)          # (B, S) bool
    txt_mask = (amask == 1) & (~vis_mask)           # (B, S) bool

    # --- sanity check ---
    n_vis = vis_mask.sum(dim=1)  # (B,)
    bad = (n_vis != image_seq_length).nonzero(as_tuple=True)[0].tolist()
    assert not bad, (
        f"Samples {bad} have {n_vis[bad].tolist()} vision tokens, "
        f"expected {image_seq_length}.\n"
        "Likely cause: text passed to processor does not contain exactly one "
        "'<image>' placeholder per sample. Stopping."
    )

    return vis_mask, txt_mask


# ---------------------------------------------------------------------------
# 3. Feature-space linear CKA (self-contained; does NOT modify cka.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def linear_cka_features(
    X_half: torch.Tensor,  # (N, p) float16, any device
    Y_half: torch.Tensor,  # (N, q) float16, any device
    eps: float = 1e-10,
) -> float:
    """Linear CKA via feature-space Gram matrices.

        CKA(X, Y) = ||Xc.T @ Yc||_F^2 / (||Xc.T @ Xc||_F * ||Yc.T @ Yc||_F)

    O(N*p*q) time, O(p*q) space.  At d=4096 each Gram matrix is 64 MB.
    In-place centering keeps peak GPU memory ≈ 2*(N*d*4 bytes) + 192 MB.
    Inputs may be on CPU (e.g. accumulated to avoid OOM); moved to CUDA here.
    """
    if not X_half.is_cuda and torch.cuda.is_available():
        X_half = X_half.cuda()
        Y_half = Y_half.cuda()
    X = X_half.float()
    Y = Y_half.float()
    X.sub_(X.mean(0, keepdim=True))  # centre in-place
    Y.sub_(Y.mean(0, keepdim=True))
    xy = X.T @ Y   # (p, q)
    xx = X.T @ X   # (p, p)
    yy = Y.T @ Y   # (q, q)
    del X, Y
    num = (xy ** 2).sum()
    den = torch.norm(xx, "fro") * torch.norm(yy, "fro") + eps
    return (num / den).item()


# ---------------------------------------------------------------------------
# 4. Model layer accessor
# ---------------------------------------------------------------------------

def get_llm_layers(model) -> torch.nn.ModuleList:
    """Return the LlamaDecoderLayer ModuleList from the LLaVA model.

    transformers 4.57.1 structure:
      LlavaForConditionalGeneration
        .language_model  (@property) → LlavaModel.language_model → LlamaModel
        LlamaModel.layers → ModuleList[LlamaDecoderLayer]
    """
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "layers"):
        return lm.layers  # LlamaModel.layers
    # Fallback: TokenDiversionLlava or other wrappers
    if hasattr(model, "llava_model"):
        inner_lm = model.llava_model.language_model
        if hasattr(inner_lm, "layers"):
            return inner_lm.layers
    # Plain LLaMA style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(
        "Cannot locate LLaMA decoder layers in model. "
        "Expected model.language_model.layers or model.model.layers."
    )


# ---------------------------------------------------------------------------
# 5. Hidden state accumulation via forward hooks
# ---------------------------------------------------------------------------

def accumulate_hidden_states(
    model,
    batches: list[dict],
    n_layers: int,
    device: str,
    image_seq_length: int = _IMAGE_SEQ_LENGTH_DEFAULT,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], list[str]]:
    """Run all batches through the model; collect per-layer vision / text reps.

    Returns:
        vis_reps[i]: (N_vis_total, H) float16 on `device`
        txt_reps[i]: (N_txt_total, H) float16 on `device`
        warnings:    list of warning strings (empty if all checks pass)
    where N_vis_total = n_samples * image_seq_length.
    """
    vis_accum: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    txt_accum: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}

    # Mutable holder so hook closures always see the current-batch masks
    _state: dict[str, torch.Tensor | None] = {"vis": None, "txt": None}

    def _make_hook(layer_idx: int):
        def hook(module, inp, out):
            # LlamaDecoderLayer output: (hidden_states, ...) or dataclass
            if isinstance(out, (tuple, list)):
                hs = out[0]
            elif hasattr(out, "hidden_states"):
                hs = out.hidden_states
            else:
                hs = out
            vis = _state["vis"]
            txt = _state["txt"]
            if vis is None:
                return
            vis_d = vis.to(hs.device, non_blocking=True)
            txt_d = txt.to(hs.device, non_blocking=True)
            # Boolean index: flatten (B, S, H) → (n_true, H); store on CPU to
            # avoid OOM when accumulating 32 layers × n_samples × 576 × 4096 tokens.
            vis_accum[layer_idx].append(hs[vis_d].detach().half().cpu())
            txt_accum[layer_idx].append(hs[txt_d].detach().half().cpu())
        return hook

    llm_layers = get_llm_layers(model)
    hooks = [llm_layers[i].register_forward_hook(_make_hook(i)) for i in range(n_layers)]
    print(f"[accum] Hooks registered on {n_layers} layers.")

    try:
        for b_idx, batch in enumerate(batches):
            t0 = time.time()
            vis_mask, txt_mask = build_vision_text_masks(
                batch["input_ids"], batch["attention_mask"], image_seq_length
            )
            _state["vis"] = vis_mask
            _state["txt"] = txt_mask
            with torch.no_grad():
                _ = model(
                    input_ids=batch["input_ids"],
                    pixel_values=batch.get("pixel_values"),
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
            del _
            B = batch["input_ids"].shape[0]
            print(
                f"  [batch {b_idx + 1}/{len(batches)}] "
                f"{B} samples, {time.time() - t0:.1f}s"
            )
    finally:
        for h in hooks:
            h.remove()
        _state["vis"] = None
        _state["txt"] = None

    print("[accum] Concatenating per-layer reps ...")
    vis_reps = {i: torch.cat(vis_accum[i], dim=0) for i in range(n_layers)}
    txt_reps = {i: torch.cat(txt_accum[i], dim=0) for i in range(n_layers)}

    N_vis = vis_reps[0].shape[0]
    N_txt = txt_reps[0].shape[0]
    hidden_dim = vis_reps[0].shape[1]
    print(f"[accum] N_vis = {N_vis:,} tokens/layer | N_txt = {N_txt:,} tokens/layer | hidden_dim = {hidden_dim}")

    run_warnings: list[str] = []
    if N_txt < 2 * hidden_dim:
        msg = (
            f"text CKA estimate may be noisy due to undersampling "
            f"(N_txt={N_txt} < 2*d={2 * hidden_dim})"
        )
        print(f"[accum] WARNING: {msg}")
        run_warnings.append(msg)

    return vis_reps, txt_reps, run_warnings


# ---------------------------------------------------------------------------
# 6. CKA computation
# ---------------------------------------------------------------------------

def compute_adjacent_cka(
    reps: dict[int, torch.Tensor],
    n_layers: int,
    label: str,
) -> list[float]:
    """CKA(layer_i, layer_{i+1}) for i in 0 .. n_layers-2. Returns 31 values."""
    values: list[float] = []
    for i in range(n_layers - 1):
        v = linear_cka_features(reps[i], reps[i + 1])
        values.append(v)
        if (i + 1) % 8 == 0 or i == n_layers - 2:
            print(f"  [{label} adj] L{i}→L{i+1}: {v:.4f}")
    return values


def compute_full_cka_matrix(
    reps: dict[int, torch.Tensor],
    n_layers: int,
    label: str,
) -> np.ndarray:
    """All-pairs CKA matrix (n_layers × n_layers), symmetric, diagonal = 1."""
    mat = np.zeros((n_layers, n_layers), dtype=np.float32)
    n_total = n_layers * (n_layers + 1) // 2
    done = 0
    for i in range(n_layers):
        for j in range(i, n_layers):
            v = linear_cka_features(reps[i], reps[j])
            mat[i, j] = v
            mat[j, i] = v
            done += 1
            if done % 64 == 0 or done == n_total:
                print(f"  [{label} heatmap] {done}/{n_total} pairs")
    return mat


# ---------------------------------------------------------------------------
# 7. Plotting
# ---------------------------------------------------------------------------

def save_heatmap(matrix: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    n = matrix.shape[0]
    ticks = list(range(0, n, 4))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"L{t}" for t in ticks], fontsize=8)
    ax.set_yticklabels([f"L{t}" for t in ticks], fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


def save_adjacent_dual(
    vision_adj: list[float],
    text_adj: list[float],
    out_path: Path,
) -> None:
    n = len(vision_adj)
    x = list(range(n))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        x, vision_adj, "o-", color="#2196F3", lw=2, ms=4,
        label=r"Vision tokens  CKA$(L_i,\, L_{i+1})$",
    )
    ax.plot(
        x, text_adj, "s--", color="#F44336", lw=2, ms=4,
        label=r"Text tokens  CKA$(L_i,\, L_{i+1})$",
    )
    ax.axhline(0.95, color="gray", ls=":", lw=1, alpha=0.8, label="0.95 saturation")

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in x], fontsize=7, rotation=60, ha="right")
    ax.set_xlabel(r"Adjacent layer pair $i \rightarrow i+1$", fontsize=11)
    ax.set_ylabel("Linear CKA", fontsize=11)
    ax.set_title(
        "Adjacent-layer CKA: vision vs text token representations (LLaVA-1.5-7B)",
        fontsize=11,
    )
    ax.set_ylim(-0.02, 1.07)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


def plot_from_l0_dual(
    vision_cka_full: np.ndarray,
    text_cka_full: np.ndarray,
    out_path: Path,
) -> None:
    """CKA(L0, Li) drift curves for vision and text tokens (paper main Fig 3)."""
    vision_from_l0 = vision_cka_full[0, :]  # shape (n_layers,)
    text_from_l0 = text_cka_full[0, :]
    n = len(vision_from_l0)
    x = list(range(n))

    # Find vision minimum over layers 1..n-1 (L0 is always 1.0 by definition)
    vis_min_idx = int(np.argmin(vision_from_l0[1:])) + 1
    vis_min_val = float(vision_from_l0[vis_min_idx])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        x, vision_from_l0, "o-", color="#2196F3", lw=2, ms=4,
        label=r"Vision tokens  CKA$(L_0,\, L_i)$",
    )
    ax.plot(
        x, text_from_l0, "s--", color="#FF9800", lw=2, ms=4,
        label=r"Text tokens  CKA$(L_0,\, L_i)$",
    )
    ax.axvline(16, color="gray", ls="--", lw=1.5, alpha=0.8, label="L16 (split_layer=18 − 2)")

    # Annotate vision minimum
    ann_x_offset = min(vis_min_idx + 2, n - 5)
    ann_y_offset = vis_min_val - 0.04
    ax.annotate(
        f"L{vis_min_idx}: min CKA(L0, Li) = {vis_min_val:.4f} (vision)",
        xy=(vis_min_idx, vis_min_val),
        xytext=(ann_x_offset, ann_y_offset),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in x], fontsize=7, rotation=60, ha="right")
    ax.set_xlabel(r"Layer index $i$", fontsize=11)
    ax.set_ylabel(r"Linear CKA$(L_0,\, L_i)$", fontsize=11)
    ax.set_title(
        r"CKA$(L_0,\, L_i)$: representation drift from layer 0 — vision vs text (LLaVA-1.5-7B)",
        fontsize=11,
    )
    y_lo = max(0.0, min(float(vision_from_l0.min()), float(text_from_l0.min())) - 0.05)
    ax.set_ylim(y_lo, 1.07)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


# ---------------------------------------------------------------------------
# 8. Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    model_path: str,
    output_dir: str,
    n_samples: int,
    batch_size: int,
    device: str,
    plot_mode: str = "both",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    samples = load_samples(n_samples)
    actual_n = len(samples)

    # ---- model ----
    print(f"\n[model] Loading from {model_path} ...")
    t0 = time.time()
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    n_layers = len(get_llm_layers(model))
    # Read image_seq_length from model config; fall back to 576 for LLaVA-1.5
    image_seq_length = int(getattr(model.config, "image_seq_length", _IMAGE_SEQ_LENGTH_DEFAULT))
    print(
        f"[model] Loaded in {time.time() - t0:.1f}s  —  {n_layers} decoder layers  "
        f"on {device}  (image_seq_length={image_seq_length})"
    )

    # ---- batches ----
    batches = build_batches(samples, processor, batch_size, device)
    B0 = batches[0]["input_ids"].shape
    print(f"\n[forward] {len(batches)} batches × {batch_size}  (input_ids shape per batch: {B0})")

    # ---- accumulate hidden states ----
    vis_reps, txt_reps, run_warnings = accumulate_hidden_states(
        model, batches, n_layers, device, image_seq_length
    )
    N_vis = vis_reps[0].shape[0]
    N_txt = txt_reps[0].shape[0]

    # Free model weights before CKA computation — saves ~14 GB VRAM for the
    # pair-wise matrix multiplications (accum is already on CPU).
    del model
    torch.cuda.empty_cache()
    print("[mem] Model freed; GPU cache cleared before CKA computation.")

    # ---- adjacent CKA ----
    print(f"\n[cka] Adjacent-layer CKA — vision  (N={N_vis:,} tokens) ...")
    vision_cka_adj = compute_adjacent_cka(vis_reps, n_layers, "vision")

    print(f"\n[cka] Adjacent-layer CKA — text  (N={N_txt:,} tokens) ...")
    text_cka_adj = compute_adjacent_cka(txt_reps, n_layers, "text")

    # ---- full heatmaps ----
    print(f"\n[cka] Full 32×32 CKA heatmap — vision  ({n_layers*(n_layers+1)//2} pairs) ...")
    vision_cka_full = compute_full_cka_matrix(vis_reps, n_layers, "vision")

    print(f"\n[cka] Full 32×32 CKA heatmap — text ...")
    text_cka_full = compute_full_cka_matrix(txt_reps, n_layers, "text")

    # ---- plots ----
    print(f"\n[plot] Generating figures (plot_mode={plot_mode}) ...")
    save_heatmap(
        vision_cka_full,
        "Vision-token CKA across all layer pairs  (LLaVA-1.5-7B)",
        out / "fig2_cka_heatmap_vision.pdf",
    )
    save_heatmap(
        text_cka_full,
        "Text-token CKA across all layer pairs  (LLaVA-1.5-7B)",
        out / "fig2_cka_heatmap_text.pdf",
    )
    if plot_mode in ("adjacent", "both"):
        save_adjacent_dual(vision_cka_adj, text_cka_adj, out / "fig3_adjacent_cka_dual.pdf")
    if plot_mode in ("from_l0", "both"):
        plot_from_l0_dual(vision_cka_full, text_cka_full, out / "fig3_cka_from_l0_dual.pdf")

    # ---- from_l0 key values ----
    vision_from_l0 = vision_cka_full[0, :].tolist()
    text_from_l0 = text_cka_full[0, :].tolist()

    # ---- summary JSON ----
    summary = {
        "model_path": model_path,
        "data_source": "LLaVA-1.5-665k COCO subset  (coco/train2017/)",
        "n_samples": actual_n,
        "n_layers": n_layers,
        "batch_size": batch_size,
        "plot_mode": plot_mode,
        "image_token_index": _IMAGE_TOKEN_INDEX,
        "image_seq_length": image_seq_length,
        "total_vision_tokens_per_layer": N_vis,
        "total_text_tokens_per_layer": N_txt,
        "warnings": run_warnings,
        "vision_cka_adj": vision_cka_adj,
        "text_cka_adj": text_cka_adj,
        "vision_cka_from_l0": vision_from_l0,
        "text_cka_from_l0": text_from_l0,
        "vision_cka_full": vision_cka_full.tolist(),
        "text_cka_full": text_cka_full.tolist(),
    }
    json_path = out / "analysis_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[json] Saved {json_path}")

    # ---- key stats ----
    vis_late = float(np.mean(vision_cka_adj[16:]))
    txt_late = float(np.mean(text_cka_adj[16:]))
    sat = vis_late >= 0.95
    print(f"\n--- Key CKA values (adjacent) ---")
    print(f"vision_cka_adj[16]  (L16→L17) : {vision_cka_adj[16]:.4f}")
    print(f"vision_cka_adj mean (L16→L31) : {vis_late:.4f}")
    print(f"text_cka_adj[16]   (L16→L17)  : {text_cka_adj[16]:.4f}")
    print(f"text_cka_adj mean  (L16→L31)  : {txt_late:.4f}")
    print(f"Vision saturation ≥0.95 after L16? {'YES ✓' if sat else 'NO — check data!'}")
    print(f"\n--- Key CKA values (from L0) ---")
    print(f"vision CKA(L0, L16) : {vision_from_l0[16]:.4f}")
    print(f"vision CKA(L0, L31) : {vision_from_l0[31]:.4f}")
    print(f"text  CKA(L0, L16)  : {text_from_l0[16]:.4f}")
    print(f"text  CKA(L0, L31)  : {text_from_l0[31]:.4f}")
    if run_warnings:
        print(f"\n--- WARNINGS ---")
        for w in run_warnings:
            print(f"  ⚠  {w}")
    print(f"\n[done] Results written to {out}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, help="Local path to LLaVA-1.5 model dir.")
    ap.add_argument("--output", default="results/analysis_7b_v2/", help="Output directory.")
    ap.add_argument("--n_samples", type=int, default=100, help="Number of real samples.")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string.",
    )
    ap.add_argument(
        "--plot_mode",
        choices=["adjacent", "from_l0", "both"],
        default="both",
        help=(
            "Which Fig 3 variant(s) to produce. "
            "'adjacent' = CKA(Li, Li+1) supplementary curve; "
            "'from_l0' = CKA(L0, Li) paper main figure; "
            "'both' = both (default)."
        ),
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_analysis(
        model_path=args.model,
        output_dir=args.output,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        device=args.device,
        plot_mode=args.plot_mode,
    )

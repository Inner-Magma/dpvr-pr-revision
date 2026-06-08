#!/usr/bin/env python
"""Preliminary analysis: CKA heatmap + vision-vs-text KL curve (Paper Fig. 2/3).

Loads a pre-trained LLaVA-1.5 model, runs a tiny image-text batch through it,
extracts per-layer hidden states for vision and text tokens, then:
  - Fig. 2: 32×32 layer-vs-layer CKA heatmap (all vision token representations)
  - Fig. 3: Per-layer KL divergence between vision-token and text-token distributions

Usage:
    python scripts/preliminary.py \\
        --model /path/to/models/llava-1.5-7b-hf \\
        --output results/analysis/ \\
        [--n_samples 4] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from PIL import Image

from dpvr.analysis.cka import linear_cka
from dpvr.analysis.hidden_states import extract_hidden_states
from dpvr.analysis.kl_divergence import kl_divergence_logits
from dpvr.analysis.plots import plot_cka_heatmap, plot_kl_curve


# ---------------------------------------------------------------------------
# Tiny synthetic batch builder
# ---------------------------------------------------------------------------

def _build_batch(processor, n_samples: int, device: str):
    """Create a small image-text batch for analysis."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (128, 0, 128), (255, 165, 0), (0, 255, 255), (192, 192, 192)]
    color_names = ["red", "green", "blue", "yellow",
                   "purple", "orange", "cyan", "silver"]
    questions = [
        f"USER: <image>\nWhat is the dominant color? ASSISTANT: The dominant color is {color_names[i % len(color_names)]}."
        for i in range(n_samples)
    ]
    images = [
        Image.new("RGB", (336, 336), color=colors[i % len(colors)])
        for i in range(n_samples)
    ]
    enc = processor(
        images=images,
        text=questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,  # LLaVA image tokens expand to 576; must not truncate them
    )
    return {k: v.to(device) for k, v in enc.items()}


# ---------------------------------------------------------------------------
# Vision-token mask extraction
# ---------------------------------------------------------------------------

def _get_vision_mask(model, input_ids: torch.LongTensor) -> torch.BoolTensor:
    """Return True for positions that carry image tokens in the embedded sequence.

    LLaVA-HF uses image_token_index = 32000 (the IMAGE_TOKEN_INDEX constant).
    """
    IMAGE_TOKEN_INDEX = getattr(
        model.config, "image_token_index",
        getattr(model.config, "image_token_id", 32000),
    )
    return input_ids == IMAGE_TOKEN_INDEX


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(model_path: str, output_dir: str, n_samples: int, device: str) -> None:
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    print(f"[preliminary] Loading model from {model_path} ...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(model_path)
    n_layers = len(model.language_model.layers)
    print(f"[preliminary] {n_layers} LLaMA layers, device={device}")

    batch = _build_batch(processor, n_samples, device)
    vision_mask = _get_vision_mask(model, batch["input_ids"])  # (B, S)

    # Extract hidden states from all layers.
    print("[preliminary] Extracting hidden states ...")
    hidden: dict[int, torch.Tensor] = extract_hidden_states(
        model,
        input_ids=batch["input_ids"],
        pixel_values=batch.get("pixel_values"),
        attention_mask=batch.get("attention_mask"),
        layers=list(range(n_layers)),
    )
    # hidden[i]: (B, S, H)  — float32, on CPU

    # Separate vision vs text tokens per layer.
    vis_states: dict[int, torch.Tensor] = {}   # (n_vis_tokens, H) flattened over batch
    txt_states: dict[int, torch.Tensor] = {}

    for layer_idx, hs in hidden.items():
        B, S, H = hs.shape
        v_mask = vision_mask.cpu()            # (B, S) bool
        t_mask = ~v_mask & (batch["attention_mask"].cpu() == 1)

        v_reps = hs[v_mask]   # (N_vis, H)
        t_reps = hs[t_mask]   # (N_txt, H)

        if v_reps.shape[0] > 0:
            vis_states[layer_idx] = v_reps.float()
        if t_reps.shape[0] > 0:
            txt_states[layer_idx] = t_reps.float()

    layers_with_both = sorted(
        set(vis_states.keys()) & set(txt_states.keys())
    )
    print(f"[preliminary] Layers with both vision + text tokens: {len(layers_with_both)}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Fig. 2
    # CKA heatmap: vision_layer_i vs vision_layer_j across all layer pairs.
    print("[preliminary] Computing CKA heatmap ...")
    vis_layers = sorted(vis_states.keys())
    n = len(vis_layers)
    cka_matrix = torch.zeros(n, n)
    for i, li in enumerate(vis_layers):
        for j, lj in enumerate(vis_layers):
            if i <= j:
                v = linear_cka(vis_states[li], vis_states[lj])
                cka_matrix[i, j] = v
                cka_matrix[j, i] = v

    labels = [f"L{l}" for l in vis_layers]
    plot_cka_heatmap(
        cka_matrix.numpy(),
        x_labels=labels,
        y_labels=labels,
        title="Vision-token CKA similarity across layers",
        out_path=out / "fig2_cka_heatmap.pdf",
    )
    print(f"[preliminary] Saved {out}/fig2_cka_heatmap.pdf")

    # ------------------------------------------------------------------ Fig. 3
    # Per-layer KL divergence: vision-token hidden states vs text-token hidden states.
    # We convert hidden states → pseudo-logits via a linear projection (use mean
    # cosine similarity as a proxy) for interpretability.
    # Simpler: compare the L2-normalised distributions over the hidden dimension.
    print("[preliminary] Computing vision-vs-text KL divergence per layer ...")
    kl_values = []
    layer_indices = []
    for layer_idx in layers_with_both:
        vis = vis_states[layer_idx]   # (N_vis, H)
        txt = txt_states[layer_idx]   # (N_txt, H)

        # Subsample to same size for fair comparison.
        n_sub = min(vis.shape[0], txt.shape[0], 256)
        vis_sub = vis[:n_sub]
        txt_sub = txt[:n_sub]

        # Treat each hidden dimension as a "logit" → compare distributions.
        # We add a (1, N_sub) dimension to treat it as batch=1, T=N_sub.
        vis_logits = vis_sub.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H)
        txt_logits = txt_sub.mean(dim=0, keepdim=True).unsqueeze(0)

        kl = kl_divergence_logits(vis_logits, txt_logits).item()
        kl_values.append(abs(kl))
        layer_indices.append(layer_idx)

    plot_kl_curve(
        kl_values,
        layer_indices,
        label="Vision-vs-Text token KL divergence",
        out_path=out / "fig3_kl_vision_text.pdf",
    )
    print(f"[preliminary] Saved {out}/fig3_kl_vision_text.pdf")

    # Save summary JSON for downstream use.
    summary = {
        "n_layers": n_layers,
        "n_samples": n_samples,
        "model_path": model_path,
        "vis_layers": vis_layers,
        "kl_values": {str(l): v for l, v in zip(layer_indices, kl_values)},
        "cka_diagonal": [float(cka_matrix[i, i]) for i in range(n)],
    }
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[preliminary] Done. Results in {out}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Local LLaVA model path.")
    ap.add_argument("--output", default="results/analysis/", help="Output directory.")
    ap.add_argument("--n_samples", type=int, default=4,
                    help="Number of synthetic samples (more → smoother CKA, slower).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        model_path=args.model,
        output_dir=args.output,
        n_samples=args.n_samples,
        device=args.device,
    )

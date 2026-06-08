"""Visualize self-attention maps extracted from a LLaMA/LLaVA layer."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def plot_attention_map(
    attn_weights: torch.Tensor,
    tokens: list[str],
    layer: int,
    head: int = 0,
    out_path: Optional[Path] = None,
):
    """Save or show a single attention head heatmap.

    Args:
        attn_weights: (num_heads, S, S) or (S, S) attention weight tensor.
        tokens: list of token strings for axis labels.
        layer: layer index (for title).
        head: which head to visualize if multi-head input.
        out_path: if given, save figure here; else plt.show().
    """
    import matplotlib.pyplot as plt

    if attn_weights.dim() == 3:
        attn = attn_weights[head].float().cpu().numpy()
    else:
        attn = attn_weights.float().cpu().numpy()

    S = len(tokens)
    attn = attn[:S, :S]

    fig, ax = plt.subplots(figsize=(max(6, S // 3), max(5, S // 3)))
    im = ax.imshow(attn, cmap="Blues", vmin=0)
    ax.set_xticks(range(S))
    ax.set_xticklabels(tokens, rotation=90, fontsize=6)
    ax.set_yticks(range(S))
    ax.set_yticklabels(tokens, fontsize=6)
    ax.set_title(f"Layer {layer} head {head} attention")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)

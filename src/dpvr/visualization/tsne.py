"""t-SNE visualization of hidden-state token clouds."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def plot_tsne(
    hidden_states: torch.Tensor,
    labels: Optional[list[str]] = None,
    title: str = "t-SNE of token representations",
    perplexity: float = 30.0,
    out_path: Optional[Path] = None,
):
    """Run t-SNE on (N, H) hidden states and plot the 2-D embedding.

    Args:
        hidden_states: (N, H) float tensor.
        labels: optional per-point color labels (e.g. "image", "text").
        title: plot title.
        perplexity: t-SNE perplexity.
        out_path: if given, save; else plt.show().
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    X = hidden_states.float().cpu().numpy()
    perp = min(perplexity, max(1, X.shape[0] - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    coords = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    if labels is not None:
        unique = sorted(set(labels))
        cmap = plt.get_cmap("tab10")
        for i, lbl in enumerate(unique):
            mask = np.array([l == lbl for l in labels])
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       label=lbl, color=cmap(i), s=10, alpha=0.7)
        ax.legend(markerscale=2)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)

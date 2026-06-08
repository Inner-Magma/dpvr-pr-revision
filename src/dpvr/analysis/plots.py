"""Plotting utilities for representation analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def plot_cka_heatmap(
    cka_matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title: str = "CKA similarity",
    out_path: Optional[Path] = None,
):
    """Save or show a heatmap of CKA values between two sets of layers."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(x_labels)), max(5, len(y_labels))))
    im = ax.imshow(cka_matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="CKA")
    plt.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_kl_curve(
    kl_values: list[float],
    layer_indices: list[int],
    label: str = "KL divergence",
    out_path: Optional[Path] = None,
):
    """Save or show a line plot of per-layer KL divergence."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layer_indices, kl_values, marker="o", label=label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL divergence")
    ax.set_title(label)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)

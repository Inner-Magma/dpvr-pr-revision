"""Centered Kernel Alignment (CKA) between representation matrices."""
from __future__ import annotations

import torch


def _center(K: torch.Tensor) -> torch.Tensor:
    """Double-center a kernel matrix."""
    n = K.shape[0]
    col_mean = K.mean(dim=0, keepdim=True)
    row_mean = K.mean(dim=1, keepdim=True)
    grand_mean = K.mean()
    return K - col_mean - row_mean + grand_mean


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between (n, p) and (n, q) representation matrices.

    Returns a scalar in [0, 1]: 1 means identical geometry.
    """
    X = X.double()
    Y = Y.double()
    KX = X @ X.T
    KY = Y @ Y.T
    KX_c = _center(KX)
    KY_c = _center(KY)
    hsic_xy = (KX_c * KY_c).sum()
    hsic_xx = (KX_c * KX_c).sum()
    hsic_yy = (KY_c * KY_c).sum()
    denom = (hsic_xx * hsic_yy).sqrt()
    if denom < 1e-10:
        return 0.0
    return (hsic_xy / denom).item()

"""KL-divergence between two sets of logits (token-distribution comparison)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence_logits(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL(P || Q) where P and Q are softmax distributions over logits.

    Args:
        logits_p: (B, T, V) reference logits.
        logits_q: (B, T, V) comparison logits.
        reduction: "mean" over all (B*T) positions, or "none" → (B, T).

    Returns:
        Scalar KL divergence or (B, T) tensor.
    """
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)   # (B, T)
    if reduction == "none":
        return kl
    return kl.mean()

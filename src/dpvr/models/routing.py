"""Backbone-agnostic helpers for splitting a sequence into image and text rows.

DPVR-LF runs the deep layers on the text tokens only and the side branch on
the image tokens only. Both sub-sequences are taken out of the full sequence
per batch row. The original code did this with ``hidden[mask].view(B, n, H)``,
which only works when every row has the same number of image tokens (fixed
576 for LLaVA-1.5). Backbones with dynamic resolution (Qwen2.5-VL, LLaVA-1.6
any-res) give each sample a different count, so the helpers here pack each
row's selected tokens to the front, pad to the longest row, and return a
validity mask that the attention masks use to hide the padding.

For a batch whose rows all select the same number of tokens the packed tensor
equals the old ``view`` result exactly (same values, same order).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Packed:
    """Row-packed selection of a (B, S, ...) tensor.

    index[b, j] is the position in the full sequence of packed slot j of row b;
    valid[b, j] is False for padding slots (j >= number of selected tokens).
    """

    index: torch.Tensor  # (B, L) long
    valid: torch.Tensor  # (B, L) bool

    @property
    def length(self) -> int:
        return int(self.index.shape[1])


def pack_index(mask: torch.Tensor) -> Packed:
    """Packing plan for a (B, S) bool mask: selected positions first, in order."""
    if mask.dim() != 2:
        raise ValueError(f"mask must be (B, S), got {tuple(mask.shape)}")
    counts = mask.sum(dim=1)
    L = int(counts.max().item()) if mask.numel() else 0
    # Stable sort on "not selected" puts selected positions first, order kept.
    order = torch.sort((~mask).to(torch.int8), dim=1, stable=True).indices
    index = order[:, :L]
    valid = torch.arange(L, device=mask.device).unsqueeze(0) < counts.unsqueeze(1)
    # Padding slots point at position 0; they are never written back.
    index = torch.where(valid, index, torch.zeros_like(index))
    return Packed(index=index, valid=valid)


def gather_rows(x: torch.Tensor, plan: Packed, seq_dim: int = 1) -> torch.Tensor:
    """Take the planned positions out of ``x`` along ``seq_dim``.

    ``x`` has the batch on ``seq_dim - 1`` and the sequence on ``seq_dim``
    (hidden states (B, S, H) use seq_dim=1; M-RoPE position ids (3, B, S) use
    seq_dim=2).
    """
    idx = plan.index
    # Bring idx to x's rank: leading dims broadcast, trailing dims expand.
    for _ in range(seq_dim - 1):
        idx = idx.unsqueeze(0)
    while idx.dim() < x.dim():
        idx = idx.unsqueeze(-1)
    shape = list(x.shape)
    shape[seq_dim] = plan.length
    return x.gather(seq_dim, idx.expand(*shape))


def scatter_rows(base: torch.Tensor, packed: torch.Tensor, plan: Packed) -> torch.Tensor:
    """Write ``packed`` (B, L, H) back into a copy of ``base`` (B, S, H)."""
    out = base.clone()
    b = torch.arange(base.shape[0], device=base.device).unsqueeze(1).expand_as(plan.index)
    out[b[plan.valid], plan.index[plan.valid]] = packed[plan.valid].to(out.dtype)
    return out


def causal_padding_mask(valid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Additive (B, 1, L, L) mask: causal, and padding keys hidden.

    Uses ``finfo.min / 2`` rather than -inf, so a padding query row (all keys
    hidden) gives finite values instead of NaN. Its output is discarded.
    """
    B, L = valid.shape
    causal = torch.ones(L, L, dtype=torch.bool, device=valid.device).tril()
    allowed = causal.unsqueeze(0) & valid.unsqueeze(1)  # (B, L, L): query i, key j
    mask = torch.zeros(B, 1, L, L, dtype=dtype, device=valid.device)
    return mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min / 2)


__all__ = ["Packed", "pack_index", "gather_rows", "scatter_rows", "causal_padding_mask"]

"""Parameter counting utilities."""
from __future__ import annotations

import torch.nn as nn


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return trainable, frozen, and total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "frozen": total - trainable,
        "total": total,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }

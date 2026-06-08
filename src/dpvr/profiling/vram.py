"""VRAM measurement utilities (no fvcore dependency)."""
from __future__ import annotations

import torch


def peak_vram_mb(device: int | str = 0) -> float:
    """Return peak VRAM usage in MiB since last reset on the given device."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def reset_peak_vram(device: int | str = 0) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

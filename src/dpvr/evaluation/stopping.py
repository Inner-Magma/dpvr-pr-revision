"""Simple early-stopping monitor based on a validation metric."""
from __future__ import annotations


class EarlyStoppingMonitor:
    """Trigger early stopping when a metric stops improving.

    Args:
        patience: how many checks without improvement before stopping.
        min_delta: minimum improvement to count as a gain.
        mode: "max" for accuracy-like metrics, "min" for loss-like.
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4,
                 mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: float | None = None
        self._bad_count = 0

    def step(self, value: float) -> bool:
        """Return True if training should stop."""
        if self._best is None:
            self._best = value
            return False

        improved = (
            (value > self._best + self.min_delta) if self.mode == "max"
            else (value < self._best - self.min_delta)
        )
        if improved:
            self._best = value
            self._bad_count = 0
        else:
            self._bad_count += 1

        return self._bad_count >= self.patience

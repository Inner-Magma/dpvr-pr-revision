"""Estimate GPU-hours from a training_time.json written by TimingCallback."""
from __future__ import annotations

import json
from pathlib import Path


def estimate_gpu_hours(training_time_json: str | Path) -> float:
    """Return GPU-hours from a timing file produced by TimingCallback."""
    data = json.loads(Path(training_time_json).read_text())
    seconds = data.get("total_seconds", 0.0)
    num_gpus = data.get("num_gpus", 1)
    return (seconds / 3600) * num_gpus

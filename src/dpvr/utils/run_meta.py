"""Standardised result envelope for all evaluation and profiling runs.

Every result JSON written by runner.py / native_runner.py / profile_model.py
must be wrapped through make_envelope() so that aggregate_results.py can
pivot cleanly across methods × benchmarks.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _config_hash(cfg: dict) -> str:
    serialised = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialised).hexdigest()[:16]


def make_envelope(
    method: str,
    base_model: str,
    benchmark: str,
    cfg: dict,
    metrics: dict[str, Any],
    *,
    n_samples: int = 0,
    latency_ms_p50: float | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Wrap evaluation metrics in a standard result envelope.

    Args:
        method: Method name (token_diversion, lora, full_ft, fastv, …).
        base_model: Model size tag (llava-1.5-7b-hf, llava-1.5-13b-hf, …).
        benchmark: Benchmark name (pope, mme, gqa, vqav2, …).
        cfg: Resolved config dict used for this run.
        metrics: Dict of metric_name → value.
        n_samples: Number of evaluated samples.
        latency_ms_p50: Median per-sample latency in milliseconds (optional).
        run_id: Explicit run ID; defaults to YYYYMMDD_HHMMSS.
    """
    now = datetime.now(timezone.utc)
    return {
        "method": method,
        "base_model": base_model,
        "benchmark": benchmark,
        "config_hash": _config_hash(cfg),
        "run_id": run_id or now.strftime("%Y%m%d_%H%M%S"),
        "git_sha": _git_sha(),
        "metrics": metrics,
        "latency_ms_p50": latency_ms_p50,
        "n_samples": n_samples,
        "timestamp_utc": now.isoformat(),
    }


def save_envelope(envelope: dict[str, Any], output_path: str | Path) -> None:
    """Write envelope to JSON, creating parent directories as needed."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(envelope, indent=2, default=str))

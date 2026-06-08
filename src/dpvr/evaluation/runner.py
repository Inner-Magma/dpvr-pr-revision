"""Evaluation runner that shells out to lmms-eval (offline, no hub calls)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_eval(
    model_path: str | Path,
    tasks: list[str],
    output_dir: str | Path,
    batch_size: int = 1,
    num_fewshot: int = 0,
    lmms_eval_bin: str = "lmms_eval",
) -> dict:
    """Run lmms-eval on a checkpoint and return parsed results.

    Requires lmms_eval to be installed in the environment.
    All model loading is by local path — no HF Hub calls.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"

    cmd = [
        sys.executable, "-m", lmms_eval_bin,
        "--model", "llava",
        "--model_args", f"pretrained={model_path}",
        "--tasks", ",".join(tasks),
        "--batch_size", str(batch_size),
        "--num_fewshot", str(num_fewshot),
        "--output_path", str(output_dir),
        "--log_samples",
    ]
    subprocess.run(cmd, check=True)

    if results_file.exists():
        return json.loads(results_file.read_text())
    return {}

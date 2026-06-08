#!/usr/bin/env python
"""Evaluate a checkpoint on perception / hallucination benchmarks.

Primary path: native_runner (no lmms-eval needed, works air-gapped).
Fallback path: lmms-eval shell-out (requires lmms_eval installed).

Usage:
    python scripts/evaluate.py \\
        --checkpoint /path/to/checkpoints/token_diversion/llava665k_split18_d1 \\
        --tasks pope,mme,gqa \\
        --output_dir results/eval/token_diversion/ \\
        --data_root /path/to/benchmarks \\
        --method token_diversion \\
        --base_model llava-1.5-7b-hf

    # Use lmms-eval backend instead of native runner:
    python scripts/evaluate.py ... --backend lmms_eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.utils import load_config, load_paths
from dpvr.utils.run_meta import make_envelope, save_envelope


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Local checkpoint directory.")
    ap.add_argument("--tasks", required=True,
                    help="Comma-separated task names: pope,mme,gqa,seed_bench,scienceqa,vqav2,mmbench,textvqa")
    ap.add_argument("--output_dir", default="results/eval", help="Directory for result JSONs.")
    ap.add_argument("--data_root", default=None,
                    help="Root dir for benchmark data. Defaults to DATA_ROOT/benchmarks.")
    ap.add_argument("--method", default="unknown", help="Method name for envelope metadata.")
    ap.add_argument("--base_model", default="llava-1.5-7b-hf", help="Model tag for metadata.")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Cap samples per benchmark (for quick validation).")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--backend", choices=["native", "lmms_eval"], default="native",
                    help="Evaluation backend. 'native' works air-gapped; "
                         "'lmms_eval' requires lmms-eval installed.")
    ap.add_argument("--config", default=None,
                    help="YAML config path for envelope metadata (optional).")
    return ap.parse_args()


def _run_native(args, tasks: list[str], data_root: Path, output_dir: Path) -> None:
    from dpvr.evaluation.native_runner import run_native_eval

    cfg = {}
    if args.config:
        try:
            cfg = load_config(args.config)
        except Exception:
            pass

    for task in tasks:
        local_data = data_root / task
        out_file = output_dir / f"{task}_raw.json"
        result = run_native_eval(
            checkpoint_path=args.checkpoint,
            task=task,
            local_data_path=local_data,
            output_path=out_file,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
        )
        if "error" in result:
            print(f"[evaluate] SKIP {task}: {result['error']}")
            continue

        envelope = make_envelope(
            method=args.method,
            base_model=args.base_model,
            benchmark=task,
            cfg=cfg,
            metrics=result.get("metrics", {}),
            n_samples=result.get("n_samples", 0),
        )
        save_envelope(envelope, output_dir / f"{task}.json")
        print(f"[evaluate] {task}: {envelope['metrics']}")


def _run_lmms_eval(args, tasks: list[str], output_dir: Path) -> None:
    import os, subprocess

    # Force offline mode so lmms-eval never tries HF Hub downloads.
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # Register the DPVR adapter (no-op if lmms-eval not installed yet).
    try:
        import dpvr.evaluation.lmms_adapter  # noqa: F401
    except ImportError:
        pass

    try:
        from lmms_eval.__main__ import cli_evaluate

        # Monkey-patch sys.argv for the lmms_eval CLI.
        import sys as _sys
        old_argv = _sys.argv[:]
        _sys.argv = [
            "lmms_eval",
            "--model", "dpvr",
            "--model_args", f"checkpoint={args.checkpoint},method={args.method}",
            "--tasks", ",".join(tasks),
            "--batch_size", str(args.batch_size),
            "--output_path", str(output_dir),
            "--log_samples",
        ]
        if args.max_samples:
            _sys.argv += ["--limit", str(args.max_samples)]
        try:
            cli_evaluate()
        finally:
            _sys.argv = old_argv

    except ImportError:
        # Fall back to shell-out.
        from dpvr.evaluation.runner import run_eval
        results = run_eval(
            model_path=args.checkpoint,
            tasks=tasks,
            output_dir=output_dir,
            batch_size=args.batch_size,
        )
        (output_dir / "lmms_raw.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))


def main() -> None:
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_root:
        data_root = Path(args.data_root)
    else:
        paths = load_paths()
        data_root = paths.data_root / "benchmarks"

    if args.backend == "native":
        _run_native(args, tasks, data_root, output_dir)
    else:
        _run_lmms_eval(args, tasks, output_dir)


if __name__ == "__main__":
    main()

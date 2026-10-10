#!/usr/bin/env python
"""Evaluate a checkpoint (or the vanilla base model) with lmms-eval 0.5.0.

This is the single evaluation entry point for the revision. It uses the
``llava_hf`` handler with ``use_cache=False`` (the paper's protocol) through
the ``dpvr_hf`` plugin, so vanilla and DPVR rows go through identical code.

Usage:
    python scripts/revision/eval_lmms.py --checkpoint /ckpt/run --tasks eight --out /runs/E9/eval_seed1
    python scripts/revision/eval_lmms.py --checkpoint none --tasks pope,scienceqa --out /runs/E0/vanilla
    python scripts/revision/eval_lmms.py ... --limit 20          # quick smoke check

Benchmarks are read from ``--bench_root`` (default ``$BENCH_ROOT`` or
``${DATA_ROOT}/benchmarks``); the task templates in ``configs/lmms_tasks``
are copied into ``<out>/tasks`` with their ``/path/to/benchmarks`` prefix
replaced. Results: lmms-eval output under ``<out>/lmms``, and a flat
``<out>/summary.json`` with every metric plus run metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

TASK_DIR = REPO_ROOT / "configs" / "lmms_tasks"
PLACEHOLDER = "/path/to/benchmarks"

# short name -> (template directory, lmms-eval task name)
TASKS = {
    "pope": ("pope", "pope_local"),
    "mme": ("mme", "mme_local"),
    "mmbench_en": ("mmbench_en", "mmbench_en_local"),
    "mmbench_cn": ("mmbench_cn", "mmbench_cn_local"),
    "scienceqa": ("scienceqa", "scienceqa_local"),
    "seedbench": ("seedbench", "seedbench_local"),
    "blink": ("blink", "blink_local"),
    "gqa": ("gqa", "gqa_local"),
}
# The paper's eight columns (MME gives both MME-P and MME-C).
EIGHT = ["pope", "mme", "mmbench_en", "mmbench_cn", "scienceqa", "seedbench", "blink"]


def parse_tasks(spec: str) -> list[str]:
    names: list[str] = []
    for part in spec.split(","):
        part = part.strip().removesuffix("_local")
        if not part:
            continue
        if part == "eight":
            names.extend(EIGHT)
        elif part in TASKS:
            names.append(part)
        else:
            raise SystemExit(f"unknown task {part!r}; choose from eight, {', '.join(TASKS)}")
    return list(dict.fromkeys(names))


def materialize_tasks(names: list[str], bench_root: Path, dest: Path) -> list[Path]:
    """Copy task templates to dest with real benchmark paths.

    Returns the dataset paths the configs reference, for an existence check.
    """
    import re

    dest.mkdir(parents=True, exist_ok=True)
    referenced: list[Path] = []
    for name in names:
        src_dir = TASK_DIR / TASKS[name][0]
        out_dir = dest / TASKS[name][0]
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        for f in src_dir.iterdir():
            if f.suffix == ".yaml" and not f.name.endswith("_local.yaml"):
                continue  # only the *_local task, so upstream task names never clash
            text = f.read_text(encoding="utf-8")
            if f.suffix == ".yaml":
                text = text.replace(PLACEHOLDER, str(bench_root))
                referenced += [Path(m) for m in re.findall(r"(?m)(?:dataset_path:|validation:|test:)\s*(\S+)", text)
                               if m.startswith("/")]
            (out_dir / f.name).write_text(text, encoding="utf-8")
    return referenced


def collect_metrics(lmms_out: Path) -> dict[str, dict]:
    files = sorted(lmms_out.rglob("*_results.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return {}
    data = json.loads(files[-1].read_text())
    metrics = {}
    for task, vals in (data.get("results") or {}).items():
        metrics[task] = {k: v for k, v in vals.items()
                         if isinstance(v, (int, float)) and k != "alias"}
    return metrics


def run_meta(args, checkpoint: str) -> dict:
    import platform

    meta = {"checkpoint": checkpoint, "use_cache": args.use_cache, "limit": args.limit,
            "python": platform.python_version()}
    try:
        meta["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        meta["git_sha"] = "unknown"
    try:
        import torch
        import transformers
        meta["torch"] = torch.__version__
        meta["transformers"] = transformers.__version__
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        from importlib.metadata import version
        meta["lmms_eval"] = version("lmms_eval")
    except Exception:
        pass
    cfg = Path(checkpoint) / "training_config.json"
    if checkpoint != "none" and cfg.exists():
        c = json.loads(cfg.read_text())
        meta["method"] = c.get("method")
        meta["seed"] = c.get("seed")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="checkpoint dir, or 'none' for vanilla")
    ap.add_argument("--tasks", default="eight")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_model", default="llava-1.5-7b-hf",
                    help="LLaVA-HF dir, absolute or under MODEL_ROOT (processor/tokenizer source)")
    ap.add_argument("--bench_root", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--use_cache", default="False",
                    help="keep False for accuracy numbers (paper protocol)")
    ap.add_argument("--dry-run", action="store_true", help="prepare tasks and print the command only")
    args = ap.parse_args()

    from dpvr.utils import load_paths

    paths = load_paths()
    bench_root = Path(args.bench_root or os.environ.get("BENCH_ROOT") or paths.data_root / "benchmarks")
    base = Path(args.base_model)
    if not base.is_absolute():
        base = paths.model_root / base
    out = Path(args.out)
    names = parse_tasks(args.tasks)

    referenced = materialize_tasks(names, bench_root, out / "tasks")
    missing = [str(p) for p in referenced if not p.exists()]
    if missing and not args.dry_run:
        raise SystemExit(f"benchmark data not found: {missing}")
    checkpoint = args.checkpoint
    if checkpoint.lower() != "none" and not Path(checkpoint).exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if not base.exists() and not args.dry_run:
        raise SystemExit(f"base model not found: {base}")

    model_args = f"pretrained={base},checkpoint={checkpoint},use_cache={args.use_cache},dtype=bfloat16"
    cmd = [
        sys.executable, "-m", "lmms_eval",
        "--model", "dpvr_hf",
        "--model_args", model_args,
        "--tasks", ",".join(TASKS[n][1] for n in names),
        "--include_path", str(out / "tasks"),
        "--batch_size", str(args.batch_size),
        "--output_path", str(out / "lmms"),
        "--log_samples",
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    env = dict(os.environ)
    plugins = [p for p in env.get("LMMS_EVAL_PLUGINS", "").split(",") if p]
    env["LMMS_EVAL_PLUGINS"] = ",".join(dict.fromkeys(plugins + ["dpvr.lmms_plugin"]))
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")

    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return
    rc = subprocess.call(cmd, env=env)
    summary = {"meta": run_meta(args, checkpoint), "tasks": names,
               "metrics": collect_metrics(out / "lmms"), "returncode": rc}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["metrics"], indent=2))
    if rc != 0:
        sys.exit(rc)
    if not summary["metrics"]:
        sys.exit("lmms-eval finished but no *_results.json was found")


if __name__ == "__main__":
    main()

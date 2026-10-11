#!/usr/bin/env python
"""Run a revision job file on the server's GPUs.

Usage:
    python scripts/revision/run_queue.py jobs/E0.yaml --gpus 0-7 --dry-run
    python scripts/revision/run_queue.py jobs/E0.yaml --gpus 0-7 \
        --set lf7b_seed1_ckpt=/data/ckpt/token_diversion_x3_fusion/seed1
    python scripts/revision/run_queue.py jobs/E9.yaml --gpus 0-7 --only train_seed1

Run it inside tmux; it returns when every job has finished. Re-running the same
command skips jobs that already succeeded (``DONE`` marker), so after a crash
just start it again. Logs and status live under
``${CKPT_ROOT}/revision_runs/<experiment>/<job>/``.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.pipeline import Scheduler, expand_jobs, load_job_file, parse_gpu_list  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_file")
    ap.add_argument("--gpus", default=os.environ.get("DPVR_GPUS", "0-7"),
                    help="GPU pool, e.g. 0-7 or 0,1,4 (default: $DPVR_GPUS or 0-7)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a job-file var (repeatable)")
    ap.add_argument("--only", action="append", default=[],
                    help="run only these jobs (and nothing that depends on others); repeatable")
    ap.add_argument("--run_root", default=None,
                    help="default: ${CKPT_ROOT}/revision_runs")
    ap.add_argument("--busy_mb", type=int, default=2000,
                    help="treat a GPU with this much memory in use as taken by someone else")
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--force", action="store_true", help="re-run jobs that already succeeded")
    ap.add_argument("--dry-run", action="store_true", help="print the expanded jobs and exit")
    return ap.parse_args()


def _path_env() -> dict[str, str]:
    """Export DATA_ROOT / MODEL_ROOT / CKPT_ROOT so job commands can use ${...}."""
    from dpvr.utils import load_paths

    p = load_paths()
    env = {
        "DATA_ROOT": str(p.data_root),
        "MODEL_ROOT": str(p.model_root),
        "CKPT_ROOT": str(p.ckpt_root),
    }
    if p.lmms_eval_path:
        env["LMMS_EVAL_PATH"] = str(p.lmms_eval_path)
    if p.hf_endpoint:
        env["HF_ENDPOINT"] = p.hf_endpoint
    return env


def main() -> None:
    args = parse_args()
    overrides = {}
    for item in args.set:
        if "=" not in item:
            sys.exit(f"--set expects NAME=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        overrides[k] = v

    env = dict(os.environ)
    env.update(_path_env())
    env.setdefault("PYTHONUNBUFFERED", "1")
    run_root = Path(args.run_root or Path(env["CKPT_ROOT"]) / "revision_runs")

    spec = load_job_file(args.job_file)
    jobs = expand_jobs(spec, run_root, overrides)
    if args.only:
        keep = set(args.only)
        unknown = keep - {j.name for j in jobs}
        if unknown:
            sys.exit(f"--only: unknown jobs {sorted(unknown)}")
        jobs = [j for j in jobs if j.name in keep]
        for j in jobs:
            j.after = [a for a in j.after if a in keep]

    gpus = parse_gpu_list(args.gpus)
    if args.dry_run:
        print(f"experiment {spec['experiment']}, GPU pool {gpus}, run root {run_root}")
        for j in jobs:
            flag = f"  [needs --set {','.join(j.missing_vars)}]" if j.missing_vars else ""
            print(f"\n- {j.name}  gpus={j.gpus} min_free_gb={j.min_free_gb} after={j.after}{flag}")
            print(f"  {j.cmd}")
        return

    sched = Scheduler(
        jobs, gpus, run_root, str(spec["experiment"]), repo_root=REPO_ROOT,
        busy_mb=args.busy_mb, poll_s=args.poll, force=args.force, base_env=env,
    )
    signal.signal(signal.SIGINT, sched.request_stop)
    signal.signal(signal.SIGTERM, sched.request_stop)
    final = sched.run()

    print("\n[queue] summary")
    print(json.dumps(final, indent=2))
    if any(s != "done" for s in final.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

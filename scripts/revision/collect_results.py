#!/usr/bin/env python
"""Gather every revision run into small, committable result files.

Reads the queue's run directories (``${CKPT_ROOT}/revision_runs/<exp>/<job>/``)
and writes, under ``results/revision/``:

    summary/eval_table.md|csv   one row per accuracy run, the paper's eight columns
    summary/jobs.md|csv         state of every job (done / failed / blocked / running)
    <exp>/<job>/...             copies of the small outputs of each job
                                (summary.json, latency/flops json, saturation
                                summary.json, compare.md, provenance.json, ...)

Checkpoints, lmms-eval sample logs and anything over ``--max_kb`` stay on the
server. Re-running overwrites the copies, so it is safe after every experiment.

    python scripts/revision/collect_results.py
    python scripts/revision/collect_results.py --run_root /data/ckpt/revision_runs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Paper column -> (lmms-eval task, metric key). Scale: MME scores as is, the rest x100.
COLUMNS = [
    ("POPE", "pope_local", "pope_accuracy,none"),
    ("MME-P", "mme_local", "mme_perception_score,none"),
    ("MME-C", "mme_local", "mme_cognition_score,none"),
    ("MMB-EN", "mmbench_en_local", "accuracy,none"),
    ("MMB-CN", "mmbench_cn_local", "accuracy,none"),
    ("SQA", "scienceqa_local", "mc_accuracy,none"),
    ("SEED", "seedbench_local", "seed_image,none"),
    ("BLINK", "blink_local", "accuracy,none"),
]
SMALL_SUFFIXES = {".json", ".md", ".csv", ".txt", ".pdf", ".png", ".yaml"}
SKIP_DIRS = {"lmms", "tasks", "checkpoint"}


def eval_row(exp: str, job: str, summary: dict) -> dict:
    metrics = summary.get("metrics", {})
    meta = summary.get("meta", {})
    row = {"experiment": exp, "job": job,
           "checkpoint": meta.get("checkpoint", ""), "base_model": meta.get("base_model", ""),
           "limit": meta.get("limit"), "gpu": meta.get("gpu", ""), "git_sha": meta.get("git_sha", "")}
    for col, task, key in COLUMNS:
        v = metrics.get(task, {}).get(key)
        row[col] = None if v is None else (round(v, 2) if col.startswith("MME") else round(100 * v, 2))
    return row


def copy_small(src_dir: Path, dst_dir: Path, max_kb: int) -> int:
    n = 0
    for f in src_dir.rglob("*"):
        rel = f.relative_to(src_dir)
        if not f.is_file() or any(p in SKIP_DIRS or p.startswith("checkpoint-") for p in rel.parts[:-1]):
            continue
        if f.suffix not in SMALL_SUFFIXES or f.stat().st_size > max_kb * 1024:
            continue
        if f.name == "cmd.sh":
            continue
        out = dst_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        n += 1
    log = src_dir / "log.txt"
    if log.exists():  # keep the tail of the log: enough to see how a job ended
        lines = log.read_text(errors="replace").splitlines()[-200:]
        (dst_dir / "log_tail.txt").parent.mkdir(parents=True, exist_ok=True)
        (dst_dir / "log_tail.txt").write_text("\n".join(lines) + "\n")
    return n


def _md_table(rows: list[dict], cols: list[str]) -> str:
    def fmt(v):
        return "" if v is None else str(v)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    out += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def collect(run_root: Path, out_root: Path, max_kb: int = 512) -> dict:
    eval_rows, job_rows = [], []
    for status_file in sorted(run_root.glob("*/*/status.json")):
        job_dir = status_file.parent
        exp, job = job_dir.parent.name, job_dir.name
        st = json.loads(status_file.read_text())
        job_rows.append({"experiment": exp, "job": job, "state": st.get("state", ""),
                         "started": st.get("started", ""), "finished": st.get("finished", st.get("at", "")),
                         "returncode": st.get("returncode", ""), "error": st.get("error", "")})
        copy_small(job_dir, out_root / exp / job, max_kb)
        summ = job_dir / "summary.json"
        if summ.exists():
            try:
                s = json.loads(summ.read_text())
            except json.JSONDecodeError:
                continue
            if "metrics" in s:
                eval_rows.append(eval_row(exp, job, s))

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    ecols = ["experiment", "job"] + [c for c, _, _ in COLUMNS] + ["limit", "checkpoint", "gpu", "git_sha"]
    jcols = ["experiment", "job", "state", "started", "finished", "returncode", "error"]
    _csv(summary_dir / "eval_table.csv", eval_rows, ecols)
    _csv(summary_dir / "jobs.csv", job_rows, jcols)
    (summary_dir / "eval_table.md").write_text(
        "# Accuracy runs (lmms-eval 0.5.0, use_cache=False)\n\n"
        "POPE/MMB/SQA/SEED/BLINK in %, MME as scores. `limit` set = smoke run, not a result.\n\n"
        + _md_table(eval_rows, ecols[:-2]))
    counts: dict[str, int] = {}
    for r in job_rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    (summary_dir / "jobs.md").write_text(
        "# Job states\n\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + "\n\n"
        + _md_table(job_rows, jcols))
    return {"evals": len(eval_rows), "jobs": len(job_rows), "states": counts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_root", default=None, help="default: ${CKPT_ROOT}/revision_runs")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "revision"))
    ap.add_argument("--max_kb", type=int, default=512, help="largest file copied")
    args = ap.parse_args()
    if args.run_root:
        run_root = Path(args.run_root)
    else:
        from dpvr.utils import load_paths

        run_root = load_paths().ckpt_root / "revision_runs"
    if not run_root.exists():
        sys.exit(f"run root not found: {run_root}")
    print(json.dumps(collect(run_root, Path(args.out), args.max_kb), indent=2))


if __name__ == "__main__":
    main()

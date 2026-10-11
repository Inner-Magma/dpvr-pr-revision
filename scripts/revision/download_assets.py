#!/usr/bin/env python
"""Download the models, benchmarks and training JSON the revision jobs need.

Everything lands where the configs and job files expect it:

    ${MODEL_ROOT}/<dir>                       HF-format models (MODELS below)
    ${MODEL_ROOT}/reference/...               Vicuna + stage-1 projector, only for
                                              check_weight_provenance.py (kept out of
                                              the llama_subdir / projector_subdir paths
                                              on purpose: their presence changes the
                                              training start, SERVER_HANDOFF §1.4)
    ${BENCH_ROOT or DATA_ROOT/benchmarks}/... the eight lmms-eval benchmarks
    ${DATA_ROOT}/llava-1.5-665k-instructions/llava_v1_5_mix665k.json

Images of LLaVA-665k come from their original hosts: scripts/download_llava_images.sh.

    python scripts/revision/download_assets.py --list
    python scripts/revision/download_assets.py models benchmarks llava665k
    python scripts/revision/download_assets.py benchmarks --only blink
    python scripts/revision/download_assets.py --verify          # sample counts only

``HF_ENDPOINT`` (from .env, e.g. https://hf-mirror.com) is honoured. Downloads
resume; re-running skips finished items. ``--verify`` loads every benchmark the
way the task configs do and compares the sample count with the count behind
the paper's numbers; a mismatch means the numbers are not comparable with the
paper (write it into RESULTS.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# local dir under MODEL_ROOT -> (repo id, allow_patterns or None)
MODELS = {
    "llava-1.5-7b-hf": ("llava-hf/llava-1.5-7b-hf", None),
    "llava-1.5-13b-hf": ("llava-hf/llava-1.5-13b-hf", None),
    "Qwen2.5-VL-7B-Instruct": ("Qwen/Qwen2.5-VL-7B-Instruct", None),
    "llava-v1.6-mistral-7b-hf": ("llava-hf/llava-v1.6-mistral-7b-hf", None),
    "InternVL3-8B-hf": ("OpenGVLab/InternVL3-8B-hf", None),
    "reference/vicuna-7b-v1.5": ("lmsys/vicuna-7b-v1.5", None),
    "reference/llava-v1.5-7b-pretrain": (
        "liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5", ["mm_projector.bin"]),
}

# name -> (dataset repo, local dir under BENCH_ROOT, allow_patterns)
BENCHMARKS = {
    "pope": ("lmms-lab/POPE", "pope", None),
    "mme": ("lmms-lab/MME", "mme", None),
    "mmbench_en": ("lmms-lab/MMBench", "mmbench_en", ["en/*"]),
    "mmbench_cn": ("lmms-lab/MMBench", "mmbench_cn", ["cn/*"]),
    "scienceqa": ("lmms-lab/ScienceQA", "scienceqa", ["ScienceQA-IMG/*"]),
    "seedbench": ("lmms-lab/SEED-Bench", "seed-bench", None),
    "blink": ("BLINK-Benchmark/BLINK", "blink/raw", None),
}

# Samples behind the paper's numbers (lmms-eval n-samples in the old results.json).
# SEED: the 7B main rows (all seeds) were evaluated on the full SEED-Bench test
# split (17,990 questions; the paper reports seed_image). An incomplete local copy
# with 5,874 rows was used for the 13B runs and some 7B ablations, so those rows
# are not comparable with a full-split re-run (SERVER_HANDOFF §1.3).
EXPECTED = {"pope": 9000, "mme": 2374, "mmbench_en": 4329, "mmbench_cn": 4329,
            "scienceqa": 2017, "seedbench": 17990, "blink": 1901}

LLAVA_JSON = ("liuhaotian/LLaVA-Instruct-150K", "llava_v1_5_mix665k.json")


def _snapshot(repo_id: str, local_dir: Path, repo_type: str, allow=None, dry=False) -> None:
    print(f"[download] {repo_type} {repo_id} -> {local_dir}" + (f" ({allow})" if allow else ""), flush=True)
    if dry:
        return
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=str(local_dir),
                      allow_patterns=allow, max_workers=8)


def combine_blink(raw_dir: Path, out_file: Path) -> int:
    """All BLINK subtasks' val splits in one parquet (the layout blink_local.yaml reads)."""
    import pandas as pd

    parts = []
    for sub in sorted(p for p in raw_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        files = sorted(sub.glob("val-*.parquet"))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if "sub_task" not in df.columns:
            df["sub_task"] = sub.name
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"no <subtask>/val-*.parquet under {raw_dir}")
    combined = pd.concat(parts, ignore_index=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_file)
    print(f"[blink] {len(parts)} subtasks, {len(combined)} rows -> {out_file}")
    return len(combined)


def count_samples(bench_root: Path) -> dict[str, int | str]:
    """Load each benchmark like its *_local.yaml and count the evaluated split."""
    import datasets

    specs = {
        "pope": (str(bench_root / "pope"), {}, "test"),
        "mme": (str(bench_root / "mme"), {}, "test"),
        "mmbench_en": (str(bench_root / "mmbench_en/en"), {}, "validation"),
        "mmbench_cn": ("parquet", {"data_files": {"validation": str(bench_root / "mmbench_cn/cn/dev-00000-of-00001.parquet")}}, "validation"),
        "scienceqa": (str(bench_root / "scienceqa/ScienceQA-IMG"), {}, "test"),
        "seedbench": (str(bench_root / "seed-bench"), {}, "test"),
        "blink": ("parquet", {"data_files": {"validation": str(bench_root / "blink/blink_val_combined.parquet")}}, "validation"),
    }
    out: dict[str, int | str] = {}
    for name, (path, kw, split) in specs.items():
        try:
            ds = datasets.load_dataset(path, split=split, verification_mode="no_checks", **kw)
            out[name] = len(ds)
        except Exception as e:  # report and continue
            out[name] = f"error: {type(e).__name__}: {e}"[:300]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="*", metavar="{models,benchmarks,llava665k}")
    ap.add_argument("--only", action="append", default=[], help="a model dir or benchmark name (repeatable)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify", action="store_true", help="count benchmark samples against the paper's")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(REPO_ROOT / "results/revision/E0/assets.json"))
    args = ap.parse_args()
    bad = set(args.what) - {"models", "benchmarks", "llava665k"}
    if bad:
        ap.error(f"unknown item(s): {', '.join(sorted(bad))}")

    if args.list:
        for k, (r, a) in MODELS.items():
            print(f"model      {k:34s} {r}" + (f" {a}" if a else ""))
        for k, (r, d, a) in BENCHMARKS.items():
            print(f"benchmark  {k:34s} {r} -> {d}" + (f" {a}" if a else ""))
        print(f"llava665k  {LLAVA_JSON[1]:34s} {LLAVA_JSON[0]}")
        return

    from dpvr.utils import load_paths

    paths = load_paths()
    if paths.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", paths.hf_endpoint)
    bench_root = Path(os.environ.get("BENCH_ROOT") or paths.data_root / "benchmarks")
    want = lambda name: not args.only or name in args.only  # noqa: E731

    if "models" in args.what:
        for d, (repo, allow) in MODELS.items():
            if want(d):
                _snapshot(repo, paths.model_root / d, "model", allow, args.dry_run)
    if "benchmarks" in args.what:
        for name, (repo, d, allow) in BENCHMARKS.items():
            if want(name):
                _snapshot(repo, bench_root / d, "dataset", allow, args.dry_run)
        if want("blink") and not args.dry_run:
            combine_blink(bench_root / "blink/raw", bench_root / "blink/blink_val_combined.parquet")
    if "llava665k" in args.what and want("llava665k"):
        dest = paths.data_root / "llava-1.5-665k-instructions"
        print(f"[download] {LLAVA_JSON[0]}/{LLAVA_JSON[1]} -> {dest}")
        if not args.dry_run:
            from huggingface_hub import hf_hub_download

            hf_hub_download(LLAVA_JSON[0], LLAVA_JSON[1], repo_type="dataset", local_dir=str(dest))
        print("[llava665k] images: bash scripts/download_llava_images.sh "
              f"{dest}")

    if args.verify:
        counts = count_samples(bench_root)
        report = {k: {"found": counts[k], "paper": EXPECTED[k], "match": counts[k] == EXPECTED[k]}
                  for k in EXPECTED}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"bench_root": str(bench_root), "benchmarks": report}, indent=2))
        for k, v in report.items():
            print(f"{'OK ' if v['match'] else 'DIFF'} {k:11s} found={v['found']} paper={v['paper']}")
        if not all(v["match"] for v in report.values()):
            sys.exit("benchmark sample counts differ from the paper's; see SERVER_HANDOFF §1.3")


if __name__ == "__main__":
    main()

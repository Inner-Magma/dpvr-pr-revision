#!/usr/bin/env python
"""Analytic prefill FLOPs and savings for the paper's configurations (C7, R1.7).

Usage:
    python scripts/revision/analytic_flops.py --out results/revision/E5/analytic_flops.json
    python scripts/revision/analytic_flops.py --n_txt 64 --attention causal

Dimensions come from ${MODEL_ROOT}/<model>/config.json when present, otherwise
from built-in LLaVA-1.5 presets. Prints a markdown table; --out also writes JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.profiling.analytic_flops import (  # noqa: E402
    PRESETS,
    VisionDims,
    dims_from_hf_config,
    image_share_per_layer,
    saving_table,
)

CONFIGS = {
    "llava-1.5-7b": [
        {"method": "dpvr_lf", "split_layer": 18},
        {"method": "dpvr_lf", "split_layer": 18, "fusion_count": 2},
        {"method": "lf_no_branch", "split_layer": 18},
        *({"method": "vtw", "split_layer": s} for s in (8, 12, 16, 18, 20, 24)),
    ],
    "llava-1.5-13b": [
        *({"method": "dpvr_lf", "split_layer": s} for s in (24, 28, 34)),
        *({"method": "vtw", "split_layer": s} for s in (24, 28, 34)),
    ],
}


def dims(name: str):
    try:
        from dpvr.utils import load_paths

        cfg_file = load_paths().model_root / f"{name}-hf" / "config.json"
        if cfg_file.exists():
            return (*dims_from_hf_config(json.loads(cfg_file.read_text())), str(cfg_file))
    except RuntimeError:
        pass
    return PRESETS[name], VisionDims(), "preset"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_img", type=int, default=576)
    ap.add_argument("--n_txt", type=int, default=128)
    ap.add_argument("--attention", choices=["dense", "causal"], default="dense")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {"n_img": args.n_img, "n_txt": args.n_txt, "attention": args.attention, "models": {}}
    for name, configs in CONFIGS.items():
        lm, vis, source = dims(name)
        rho = image_share_per_layer(lm, args.n_img, args.n_txt, args.attention)
        rows = saving_table(lm, args.n_img, args.n_txt, configs, args.attention, vis)
        report["models"][name] = {"dims_source": source, "rho_analytic": rho, "rows": rows}
        print(f"\n### {name} (n_img={args.n_img}, n_txt={args.n_txt}, {args.attention} attention, "
              f"analytic rho={rho:.3f})\n")
        print("| method | s | K | TFLOPs | total | LM only | deep stack |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        for r in rows:
            k = r.get("fusion_count", 1) if r["method"] in ("dpvr_lf", "lf_no_branch") else "-"
            print(f"| {r['method']} | {r['split_layer']} | {k} | {r['tflops_total']:.2f} | "
                  f"{r['saving_total']:+.1%} | {r['saving_lm']:+.1%} | {r['saving_deep']:+.1%} |")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

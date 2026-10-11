#!/usr/bin/env python
"""Overlay saturation runs (E2 across families, E8 across image-token counts).

    python scripts/revision/compare_saturation.py \
        "LLaVA-1.5-7B=results/E2/llava7b" "Qwen2.5-VL-7B=results/E2/qwen25vl7b" \
        --normalize --out results/E2/compare

Writes <out>.pdf (one panel per metric, median curves, x = layer or layer/L)
and <out>.md (transition layers per run, absolute and normalized).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PANELS = ("cos_img", "resid_img", "attn", "kl_img", "kl_txt")


def load(spec: str) -> tuple[str, dict]:
    label, _, path = spec.partition("=")
    if not path:
        label, path = Path(spec).name, spec
    return label, json.loads((Path(path) / "summary.json").read_text())


def table(runs: list[tuple[str, dict]]) -> str:
    keys = sorted({k for _, s in runs for k in s["transitions"]})
    head = "| run | L | image tokens | " + " | ".join(keys) + " |"
    lines = [head, "|" + " --- |" * (3 + len(keys))]
    for label, s in runs:
        L = s.get("n_layers")
        cells = []
        for k in keys:
            t = s["transitions"].get(k, {})
            v = t.get("median_curve", t.get("plateau_from"))
            cells.append("—" if v is None else f"{v} ({v / L:.2f})")
        lines.append(f"| {label} | {L} | {s['meta'].get('image_tokens_mean')} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n\nCells: layer (layer / L). kl_*, resid_*: 50%-of-peak transition; " \
        "cos_img: first layer from which the median stays ≥ tau.\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="LABEL=RUN_DIR (RUN_DIR holds summary.json)")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--normalize", action="store_true", help="x axis = layer / L")
    args = ap.parse_args()

    runs = [load(r) for r in args.runs]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(table(runs))
    print(table(runs))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [p for p in PANELS if any(p in s["per_layer"] for _, s in runs)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.2))
    for ax, key in zip(np.atleast_1d(axes), panels):
        for label, s in runs:
            if key not in s["per_layer"]:
                continue
            med = np.asarray(s["per_layer"][key]["median"])
            x = np.arange(len(med)) / (len(med) if args.normalize else 1)
            ax.plot(x, med, label=label)
        ax.set_title(key)
        ax.set_xlabel("layer / L" if args.normalize else "layer")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[plot] {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

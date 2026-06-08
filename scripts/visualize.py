#!/usr/bin/env python
"""Generate attention maps and t-SNE plots for a checkpoint.

Usage:
    python scripts/visualize.py --config configs/method/token_diversion.yaml \
        --checkpoint /path/to/checkpoints/token_diversion/llava665k_split18_d1 \
        --output results/figures/token_diversion
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.models import build_model
from dpvr.analysis import extract_hidden_states
from dpvr.visualization import plot_tsne
from dpvr.utils import apply_cli_overrides, load_config, load_paths, resolve_paths, set_seed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="Optional checkpoint dir to load.")
    ap.add_argument("--output", default="results/figures")
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("overrides", nargs="*")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args.overrides)
    paths = load_paths()
    cfg = resolve_paths(cfg, paths)

    model = build_model(cfg)
    if args.checkpoint:
        from transformers import AutoConfig
        state = torch.load(Path(args.checkpoint) / "pytorch_model.bin",
                           map_location="cpu")
        model.load_state_dict(state, strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    dummy_ids = torch.randint(100, 1000, (1, args.seq_len), device=device)
    hs = extract_hidden_states(model, dummy_ids, layers=[args.layer])

    out_dir = Path(args.output)
    hidden = hs[args.layer][0]   # (S, H)
    labels = ["token"] * hidden.shape[0]
    plot_tsne(hidden, labels=labels,
              title=f"t-SNE layer {args.layer}",
              out_path=out_dir / f"tsne_layer{args.layer}.png")
    print(f"Saved t-SNE to {out_dir}/tsne_layer{args.layer}.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Profile a model checkpoint: parameter counts, VRAM, and FLOPs.

Usage:
    python scripts/profile_model.py --config configs/method/token_diversion.yaml \
        --output results/profiling/token_diversion.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Force transformers to finish loading before unsloth leaves it partially init'd.
from transformers.modeling_utils import PreTrainedModel as _PTM  # noqa: F401


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="results/profiling/profile.json")
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("overrides", nargs="*")
    return ap.parse_args()


def main():
    args = parse_args()

    from dpvr.utils import apply_cli_overrides, load_config, load_paths, resolve_paths
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args.overrides)
    paths = load_paths()
    cfg = resolve_paths(cfg, paths)

    from dpvr.models import build_model
    from dpvr.profiling import count_parameters, peak_vram_mb, estimate_flops
    from dpvr.profiling.vram import reset_peak_vram

    model = build_model(cfg)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    params = count_parameters(model)

    reset_peak_vram(device)
    dummy_ids = torch.zeros(1, args.seq_len, dtype=torch.long, device=device)

    try:
        flops = estimate_flops(model, dummy_ids)
    except Exception as e:
        flops = {"error": str(e)}

    vram = peak_vram_mb(device)

    results = {
        "method": cfg["method"]["name"],
        "params": params,
        "vram_peak_mb": vram,
        "flops": flops,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Training entry point.

Usage:
    python scripts/train.py --config configs/method/token_diversion.yaml \
        method.split_layer=12 training.per_device_train_batch_size=2

All experiment variations — split_layer, LoRA rank, data subset fraction, etc. —
flow through YAML + CLI overrides. No need to edit this file for ablations.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.data import LlavaDataCollator, load_raw_dataset  # noqa: E402
from dpvr.utils import (  # noqa: E402
    apply_cli_overrides,
    load_config,
    load_paths,
    resolve_paths,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a dpvr method from a YAML config.")
    ap.add_argument("--config", required=True, help="Path to a YAML config under configs/.")
    ap.add_argument(
        "overrides",
        nargs="*",
        help="Optional key.path=value overrides (e.g. method.split_layer=12).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args.overrides)

    paths = load_paths()
    cfg = resolve_paths(cfg, paths)

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    if paths.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", paths.hf_endpoint)
    if cfg.get("runtime", {}).get("detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)

    set_seed(cfg.get("seed", 42))

    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model_path"])
    tokenizer.chat_template = cfg["chat_template"]
    processor = AutoProcessor.from_pretrained(cfg["model"]["base_model_path"])
    processor.tokenizer = tokenizer

    # Registry-based model construction — no method-specific if-else here.
    from dpvr.models import build_model
    from dpvr.training import build_trainer

    model = build_model(cfg)
    model.print_trainable_parameters()

    raw_datasets = load_raw_dataset(
        cfg["data"]["json_path"],
        subset_fraction=cfg["data"].get("subset_fraction", 1.0),
    )
    collator = LlavaDataCollator(processor, image_base_path=cfg["data"]["image_base_path"])

    trainer = build_trainer(cfg, model, raw_datasets["train"], collator,
                             track_efficiency=cfg.get("runtime", {}).get("track_efficiency", False))
    trainer.train(resume_from_checkpoint=cfg["training"].get("resume_from_checkpoint"))

    t = cfg["training"]
    trainer.save_model(t["output_dir"])
    if trainer.args.should_save:
        tokenizer.save_pretrained(t["output_dir"])
        processor.save_pretrained(t["output_dir"])
    print(f"model saved to {t['output_dir']}")


if __name__ == "__main__":
    main()

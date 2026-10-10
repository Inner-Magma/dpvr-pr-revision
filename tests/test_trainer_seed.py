"""The config seed must reach the HF Trainer (data shuffle), not only torch."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.training.trainer import build_training_args


def _cfg(tmp_path, seed, **training_overrides):
    training = dict(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        optim="adamw_torch",
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        logging_steps=1,
        save_strategy="no",
        save_steps=1000,
        save_total_limit=1,
        report_to="none",
    )
    training.update(training_overrides)
    return {"seed": seed, "training": training}


def test_seed_reaches_training_args(tmp_path):
    args = build_training_args(_cfg(tmp_path, seed=3))
    assert args.seed == 3
    assert args.data_seed == 3


def test_data_seed_can_be_set_separately(tmp_path):
    args = build_training_args(_cfg(tmp_path, seed=3, data_seed=7))
    assert args.seed == 3
    assert args.data_seed == 7

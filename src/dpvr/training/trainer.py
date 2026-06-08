"""Unified trainer factory: build a HF Trainer from a resolved config dict."""
from __future__ import annotations

from transformers import Trainer, TrainingArguments

from .callbacks import EfficiencyCallback, ConfigSaveCallback


def build_training_args(cfg: dict) -> TrainingArguments:
    t = cfg["training"]
    return TrainingArguments(
        output_dir=t["output_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        num_train_epochs=t["num_train_epochs"],
        max_steps=t.get("max_steps", -1),
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        optim=t["optim"],
        bf16=t["bf16"],
        fp16=t["fp16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        report_to=t["report_to"],
        remove_unused_columns=False,
    )


def build_trainer(cfg: dict, model, train_dataset, data_collator,
                  track_efficiency: bool = False) -> Trainer:
    training_args = build_training_args(cfg)
    callbacks = [ConfigSaveCallback(cfg)]
    if track_efficiency:
        callbacks.append(EfficiencyCallback())
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

"""Training callbacks: VRAM, timing, and FLOPs tracking."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class VRAMCallback(TrainerCallback):
    """Log peak VRAM after every logging step."""

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, logs=None, **kwargs):
        if not torch.cuda.is_available():
            return
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        if logs is not None:
            logs["peak_vram_mb"] = round(peak_mb, 1)
        torch.cuda.reset_peak_memory_stats()


class TimingCallback(TrainerCallback):
    """Record total wall-clock training time."""

    def on_train_begin(self, args, state, control, **kwargs):
        self._t0 = time.time()

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        elapsed = time.time() - self._t0
        print(f"\n[TimingCallback] Total training time: {elapsed:.1f}s "
              f"({elapsed/3600:.3f} GPU-hours)")
        # Write to results/profile/ if output_dir is set.
        out = Path(args.output_dir) / "training_time.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"total_seconds": round(elapsed, 1),
                                   "gpu_hours": round(elapsed / 3600, 4)}))


class EfficiencyCallback(VRAMCallback, TimingCallback):
    """Combines VRAM + timing in one callback (P0-T2)."""
    pass


class ConfigSaveCallback(TrainerCallback):
    """Write training_config.json to every checkpoint directory.

    Enables TokenDiversionLlava.from_trained() to reconstruct the model
    without knowing the original base_model_path at load time.
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg

    def on_save(self, args: TrainingArguments, state: TrainerState,
                control: TrainerControl, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if ckpt_dir.exists():
            (ckpt_dir / "training_config.json").write_text(
                json.dumps(self._cfg, indent=2, default=str)
            )

#!/usr/bin/env python3
"""
profile_efficiency.py — Section 5.5 Efficiency Analysis data.

Profiles TD main and LoRA r=16 ckpt-144000:
  - Trainable params (M, % of total)
  - Total params (M)
  - Peak VRAM during training-step simulation (batch=1, forward+backward, with image)
  - Single forward latency (ms, batch=1, seq≈2048, avg 10 runs)

Must be run on cuda:5 only (CUDA_VISIBLE_DEVICES=5).

Usage:
  CUDA_VISIBLE_DEVICES=5 python scripts/profile_efficiency.py
  # or via nohup for background:
  nohup bash -c "CUDA_VISIBLE_DEVICES=5 python scripts/profile_efficiency.py" \
    > /tmp/profile_efficiency.out 2>&1 &
"""
from __future__ import annotations

import gc
import glob
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Ensure cuda:5 (mapped to cuda:0 inside this process after CUDA_VISIBLE_DEVICES=5 export)
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
IMAGE_TOKEN_ID = 32000  # <image> in LLaVA-1.5 tokenizer
IMAGE_SIZE = 336        # LLaVA-1.5-7B CLIP resolution
NUM_IMAGE_TOKENS = 576  # (336/14)^2 = 576 CLIP patch tokens per image

TD_CKPT = "/path/to/checkpoints/token_diversion/llava665k_split18_d1"
LORA_CKPT = "/path/to/checkpoints/lora/llava665k_r16/checkpoint-144000"


def count_params(model: torch.nn.Module) -> tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    return trainable, total, pct


def make_dummy_batch(seq_text: int = 1472) -> dict:
    """Build a representative LLaVA training batch (1 image + text).

    seq_text: number of text tokens after the image tokens.
    Total length: NUM_IMAGE_TOKENS + seq_text.
    """
    img_ids = torch.full((1, NUM_IMAGE_TOKENS), IMAGE_TOKEN_ID, dtype=torch.long)
    txt_ids = torch.randint(1, 32000, (1, seq_text), dtype=torch.long)
    input_ids = torch.cat([img_ids, txt_ids], dim=1).to(DEVICE)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :NUM_IMAGE_TOKENS] = -100  # don't compute loss on image tokens
    pixel_values = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=DTYPE, device=DEVICE)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
    }


def measure_vram_training_step(model: torch.nn.Module) -> float:
    """Peak VRAM (MiB) during a training-step simulation (forward+backward)."""
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    batch = make_dummy_batch(seq_text=512)  # shorter text for backward to avoid OOM
    try:
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        if loss is None:
            # Fallback: sum logits if loss is not returned
            loss = outputs.logits.float().sum() * 0.0
        loss.backward()
    except Exception as e:
        print(f"  [WARN] backward failed: {e}")
    finally:
        model.zero_grad(set_to_none=True)

    peak_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
    torch.cuda.empty_cache()
    return peak_mb


def measure_forward_latency(model: torch.nn.Module, n_warmup: int = 3, n_runs: int = 10) -> tuple[float, float]:
    """Mean and std forward latency (ms) over n_runs (after n_warmup warmup runs)."""
    model.eval()
    batch = make_dummy_batch(seq_text=1472)  # 576 + 1472 = 2048 total
    times_ms = []

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(**{k: v for k, v in batch.items() if k != "labels"}, use_cache=False)
            torch.cuda.synchronize(DEVICE)

        for _ in range(n_runs):
            torch.cuda.synchronize(DEVICE)
            t0 = time.perf_counter()
            _ = model(**{k: v for k, v in batch.items() if k != "labels"}, use_cache=False)
            torch.cuda.synchronize(DEVICE)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = sum(times_ms) / len(times_ms)
    std_ms = (sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5
    return mean_ms, std_ms


def profile_td_main() -> dict:
    print("\n=== Profiling TD main ===")
    from dpvr.models.token_diversion import TokenDiversionLlava

    print(f"  Loading from {TD_CKPT} ...")
    model = TokenDiversionLlava.from_trained(TD_CKPT)
    model = model.to(DEVICE)

    trainable, total, pct = count_params(model)
    print(f"  Params: trainable={trainable/1e6:.2f}M ({pct:.4f}%), total={total/1e6:.0f}M")

    print("  Measuring training-step VRAM...")
    vram_mb = measure_vram_training_step(model)
    print(f"  Peak VRAM (train step): {vram_mb:.0f} MiB")

    print("  Measuring forward latency (10 runs, seq=2048)...")
    lat_mean, lat_std = measure_forward_latency(model)
    print(f"  Forward latency: {lat_mean:.1f} ± {lat_std:.1f} ms")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": "TD-main (split_layer=18, vision_depth=1)",
        "ckpt": TD_CKPT,
        "trainable_params_M": round(trainable / 1e6, 3),
        "total_params_M": round(total / 1e6, 1),
        "trainable_pct": round(pct, 4),
        "peak_vram_train_step_MiB": round(vram_mb, 0),
        "forward_latency_ms_mean": round(lat_mean, 1),
        "forward_latency_ms_std": round(lat_std, 1),
    }


def profile_lora_preview() -> dict:
    print("\n=== Profiling LoRA r=16 ckpt-144000 (pre-merge) ===")
    from dpvr.models.lora_baseline import LoRALlava
    from safetensors.torch import load_file as safetensors_load_file

    cfg_path = os.path.join(LORA_CKPT, "training_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    print("  Building LoRALlava (pre-merge)...")
    model = LoRALlava.from_config(cfg)

    print("  Loading checkpoint shards...")
    shard_files = sorted(glob.glob(os.path.join(LORA_CKPT, "model-*-of-*.safetensors")))
    if not shard_files:
        shard_files = [os.path.join(LORA_CKPT, "model.safetensors")]
    state_dict: dict = {}
    for shard in shard_files:
        state_dict.update(safetensors_load_file(shard, device="cpu"))
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()

    model = model.to(DEVICE)

    trainable, total, pct = count_params(model)
    print(f"  Params: trainable={trainable/1e6:.2f}M ({pct:.4f}%), total={total/1e6:.0f}M")

    print("  Measuring training-step VRAM...")
    vram_mb = measure_vram_training_step(model)
    print(f"  Peak VRAM (train step): {vram_mb:.0f} MiB")

    print("  Measuring forward latency (10 runs, seq=2048)...")
    lat_mean, lat_std = measure_forward_latency(model)
    print(f"  Forward latency: {lat_mean:.1f} ± {lat_std:.1f} ms")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": "LoRA r=16 (pre-merge, adapters active)",
        "ckpt": LORA_CKPT,
        "trainable_params_M": round(trainable / 1e6, 3),
        "total_params_M": round(total / 1e6, 1),
        "trainable_pct": round(pct, 4),
        "peak_vram_train_step_MiB": round(vram_mb, 0),
        "forward_latency_ms_mean": round(lat_mean, 1),
        "forward_latency_ms_std": round(lat_std, 1),
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== profile_efficiency.py started {ts} ===")
    print(f"Device: {DEVICE} | dtype: {DTYPE}")
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print(f"Available VRAM: {torch.cuda.mem_get_info(DEVICE)[0] / 1024**3:.1f} GB free")

    results = {}

    td_result = profile_td_main()
    results["td_main"] = td_result

    lora_result = profile_lora_preview()
    results["lora_r16_preview"] = lora_result

    # Save JSON
    out_json = REPO_ROOT / "results" / "profile" / f"efficiency_{ts}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nJSON saved: {out_json}")

    # Build markdown report
    td = results["td_main"]
    lr = results["lora_r16_preview"]

    report_path = REPO_ROOT / "reports" / "hsf_status" / f"efficiency_profile_{ts}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(f"""# Efficiency Profile — {ts}

> Paper Section 5.5 "Efficiency Analysis" source data.
> Profiled on cuda:5 (RTX PRO 6000 Blackwell 97GB).
> Training-step VRAM: batch=1, 1 image (336×336) + 512 text tokens, forward+backward.
> Forward latency: batch=1, 1 image + 1472 text tokens (≈2048 total), avg 10 runs.

| Metric | TD-main | LoRA r=16 |
|--------|---------|-----------|
| Trainable params | {td['trainable_params_M']:.3f} M | {lr['trainable_params_M']:.3f} M |
| Trainable % | {td['trainable_pct']:.4f}% | {lr['trainable_pct']:.4f}% |
| Total params | {td['total_params_M']:.0f} M | {lr['total_params_M']:.0f} M |
| Peak VRAM (train step) | {td['peak_vram_train_step_MiB']:.0f} MiB | {lr['peak_vram_train_step_MiB']:.0f} MiB |
| Forward latency (mean) | {td['forward_latency_ms_mean']:.1f} ms | {lr['forward_latency_ms_mean']:.1f} ms |
| Forward latency (std) | {td['forward_latency_ms_std']:.1f} ms | {lr['forward_latency_ms_std']:.1f} ms |

TD ckpt: `{td['ckpt']}`
LoRA ckpt: `{lr['ckpt']}`
JSON: `{out_json}`
""")
    print(f"Report saved: {report_path}")
    print(f"\n=== profile_efficiency.py DONE {datetime.now().strftime('%Y%m%d_%H%M%S')} ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Prefill (single forward) latency under the revision's fixed protocol.

Protocol: bf16, ``use_cache=False``, synthetic input of one image (576 image
tokens for LLaVA-1.5) plus ``--n_txt`` text tokens, ``--warmup`` untimed
passes, then ``--iters`` timed passes with CUDA events. Weights of the DPVR
side branch are irrelevant for latency, so the model is built from the config
without a checkpoint.

Usage:
    python scripts/revision/bench_latency.py --vanilla --base_model llava-1.5-7b-hf --out lat.json
    python scripts/revision/bench_latency.py --config configs/method/token_diversion_x3_fusion.yaml \
        --batch 1 --n_txt 128 --out lat.json [key=value overrides]

Run it alone on an otherwise idle GPU; the job files give it ``min_free_gb``
close to the card's full memory for that reason.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def build(args):
    from dpvr.utils import apply_cli_overrides, load_config, load_paths, resolve_paths

    paths = load_paths()
    if args.vanilla:
        from transformers import LlavaForConditionalGeneration

        base = paths.model_root / args.base_model
        model = LlavaForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16)
        return model, model.config, "vanilla", {"base_model": args.base_model}
    cfg = resolve_paths(apply_cli_overrides(load_config(args.config), args.overrides), paths)
    from dpvr.models import build_model

    model = build_model(cfg)
    return model, model.llava_model.config, cfg["method"]["name"], {
        "config": args.config, "overrides": args.overrides, "method": cfg["method"]}


def make_inputs(config, batch: int, n_txt: int, device):
    image_token = getattr(config, "image_token_id", None) or config.image_token_index
    vc = config.vision_config
    # one feature per patch (the CLS token is dropped): 576 for CLIP-L/14 at 336 px
    n_img = (vc.image_size // vc.patch_size) ** 2
    vocab = config.text_config.vocab_size
    g = torch.Generator().manual_seed(0)
    text = torch.randint(3, min(vocab, image_token), (batch, n_txt - 1), generator=g)  # never the image id
    bos = torch.ones(batch, 1, dtype=torch.long)
    img = torch.full((batch, n_img), image_token, dtype=torch.long)
    input_ids = torch.cat([bos, img, text], dim=1).to(device)
    pixel_values = torch.randn(batch, 3, vc.image_size, vc.image_size, generator=g)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": pixel_values.to(device, torch.bfloat16),
    }, n_img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--vanilla", action="store_true")
    src.add_argument("--config")
    ap.add_argument("--base_model", default="llava-1.5-7b-hf")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--n_txt", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is required")
    device = torch.device("cuda:0")
    model, config, method, info = build(args)
    model.to(device, dtype=torch.bfloat16).eval()
    inputs, n_img = make_inputs(config, args.batch, args.n_txt, device)

    times = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(**inputs, use_cache=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.iters):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(**inputs, use_cache=False)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    result = {
        "method": method,
        **info,
        "protocol": {"batch": args.batch, "n_img": n_img, "n_txt": args.n_txt,
                     "dtype": "bfloat16", "use_cache": False,
                     "warmup": args.warmup, "iters": args.iters},
        "latency_ms": {"mean": statistics.mean(times), "std": statistics.stdev(times),
                       "p50": statistics.median(times), "min": min(times), "max": max(times)},
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    lat = result["latency_ms"]
    print(f"{method}: {lat['mean']:.2f} ± {lat['std']:.2f} ms on {result['gpu']}")


if __name__ == "__main__":
    main()

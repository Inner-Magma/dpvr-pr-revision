#!/usr/bin/env python
"""End-to-end generation latency: time to first token, decode ms/token, total (C7, E1).

Protocol: B=1, bf16, one image (576 tokens) + ``--n_txt`` prompt tokens,
greedy decoding forced to exactly N new tokens (``min_new_tokens=N``), for each
N in ``--new_tokens``. TTFT is a 1-token generate; decode ms/token is
``(total(N) - TTFT) / (N - 1)``.

Usage:
    python scripts/revision/bench_generate.py --vanilla --out gen.json
    python scripts/revision/bench_generate.py --config configs/method/token_diversion_x3_fusion.yaml \
        --use_cache true --new_tokens 64,256 --out gen.json
    # the manuscript's Table 7 path (no cache) for comparison:
    python scripts/revision/bench_generate.py --config ... --use_cache false --out gen_nocache.json

Cached decoding for DPVR-LF needs C2 (KV-cache) in the checked-out code.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "revision"))

from bench_latency import build, make_inputs  # noqa: E402


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def timed_generate(model, inputs, n_new: int, use_cache: bool, pad_id: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        **inputs, max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False,
        use_cache=use_cache, pad_token_id=pad_id,
    )
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    produced = out.shape[1] - inputs["input_ids"].shape[1]
    if produced != n_new:
        raise RuntimeError(f"asked for {n_new} new tokens, got {produced}")
    return ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--vanilla", action="store_true")
    src.add_argument("--config")
    ap.add_argument("--base_model", default="llava-1.5-7b-hf")
    ap.add_argument("--n_txt", type=int, default=128)
    ap.add_argument("--new_tokens", default="64,256")
    ap.add_argument("--use_cache", default="true")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is required")
    use_cache = _as_bool(args.use_cache)
    lengths = [int(x) for x in args.new_tokens.split(",") if x.strip()]
    args.batch = 1
    model, config, method, info = build(args)
    model.to("cuda:0", dtype=torch.bfloat16).eval()
    inputs, n_img = make_inputs(config, 1, args.n_txt, torch.device("cuda:0"))
    pad_id = getattr(config, "pad_token_id", None) or 0

    def measure(n: int) -> list[float]:
        with torch.inference_mode():
            for _ in range(args.warmup):
                timed_generate(model, inputs, n, use_cache, pad_id)
            return [timed_generate(model, inputs, n, use_cache, pad_id) for _ in range(args.iters)]

    ttft = measure(1)
    result = {
        "method": method,
        **info,
        "protocol": {"batch": 1, "n_img": n_img, "n_txt": args.n_txt, "dtype": "bfloat16",
                     "use_cache": use_cache, "warmup": args.warmup, "iters": args.iters,
                     "decoding": "greedy, min_new_tokens = max_new_tokens"},
        "ttft_ms": {"mean": statistics.mean(ttft), "std": statistics.pstdev(ttft)},
        "by_length": {},
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    for n in lengths:
        t = measure(n)
        total = statistics.mean(t)
        result["by_length"][str(n)] = {
            "total_ms": {"mean": total, "std": statistics.pstdev(t)},
            "decode_ms_per_token": (total - result["ttft_ms"]["mean"]) / max(n - 1, 1),
        }
        print(f"{method} N={n}: total {total:.1f} ms, "
              f"decode {result['by_length'][str(n)]['decode_ms_per_token']:.2f} ms/token "
              f"(cache={use_cache}, {result['gpu']})", flush=True)
    result["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

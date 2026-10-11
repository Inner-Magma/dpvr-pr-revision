#!/usr/bin/env python
"""Training-free saturation analysis for any HF vision-language model (E2, E8).

Examples:
    # LLaVA-1.5-7B, the paper's Figure 2 setting (500 LLaVA-665k samples)
    python scripts/revision/saturation.py --model llava-1.5-7b-hf --out results/E2/llava7b

    # another family (needs transformers 4.57 support; model under ${MODEL_ROOT})
    python scripts/revision/saturation.py --model Qwen2.5-VL-7B-Instruct --out ...

    # E8: image tokens pooled 24x24 -> 12x12 (144 tokens)
    python scripts/revision/saturation.py --model llava-1.5-7b-hf --image_grid 12x12 --out ...

Writes <out>/curves.npz (per-sample (n, L) arrays for every metric),
<out>/summary.json (per-layer median/IQR, transition layers, run metadata) and,
with matplotlib installed, <out>/saturation.pdf.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.analysis import saturation as sat  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def default_data() -> tuple[Path, Path]:
    from dpvr.utils import load_config, load_paths, resolve_paths

    cfg = resolve_paths(load_config(str(REPO_ROOT / "configs/_base.yaml")), load_paths())
    return Path(cfg["data"]["json_path"]), Path(cfg["data"]["image_base_path"])


def resolve_model(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    from dpvr.utils import load_paths

    return load_paths().join_model(name)


def load_samples(json_path: Path, image_root: Path, n: int, seed: int, prefix: str):
    """n random single-image LLaVA-format samples whose image file exists."""
    from PIL import Image

    with open(json_path) as f:
        data = json.load(f)
    data = [e for e in data if isinstance(e.get("image"), str) and e["image"].startswith(prefix)]
    rng = random.Random(seed)
    rng.shuffle(data)
    out = []
    for e in data:
        path = image_root / e["image"]
        if not path.exists():
            continue
        q = next((t["value"] for t in e.get("conversations", []) if t.get("from") == "human"), None)
        if not q:
            continue
        q = q.replace("<image>", "").strip()
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        out.append({"id": e.get("id", ""), "image": img, "question": q, "path": e["image"]})
        if len(out) >= n:
            break
    if len(out) < n:
        print(f"[data] WARNING: only {len(out)} usable samples (asked for {n})")
    return out


def build_prompt(processor, question: str) -> str:
    if getattr(processor, "chat_template", None):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return processor.apply_chat_template(msgs, add_generation_prompt=True)
    return f"USER: <image>\n{question} ASSISTANT:"


def plot(summary: dict, out_pdf: Path, title: str, split: int | None) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib missing; skipped")
        return
    pl = summary["per_layer"]
    panels = [("cos", "adjacent cosine"), ("resid", "relative residual update"),
              ("attn", "text→image attention mass"), ("kl", "logit-lens KL to final")]
    panels = [p for p in panels if any(k.startswith(p[0]) for k in pl)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.2))
    for ax, (key, label) in zip(np.atleast_1d(axes), panels):
        for k, v in pl.items():
            if not (k == key or k.startswith(key + "_")):
                continue
            x = np.arange(len(v["median"]))
            ax.plot(x, v["median"], label=k)
            ax.fill_between(x, v["q25"], v["q75"], alpha=0.2)
        if split is not None:
            ax.axvline(split, ls="--", c="gray")
        ax.set_xlabel("layer")
        ax.set_title(label)
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_pdf)
    print(f"[plot] {out_pdf}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model dir name under ${MODEL_ROOT}, or a path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_json", help="LLaVA-format JSON (default: LLaVA-665k from configs/_base.yaml)")
    ap.add_argument("--image_root")
    ap.add_argument("--image_prefix", default="", help="keep samples whose image path starts with this")
    ap.add_argument("--image_grid", help="pool image tokens to HxW (LLaVA-1.5 only), e.g. 12x12, 12x24")
    ap.add_argument("--max_pixels", type=int, default=640 * 28 * 28,
                    help="cap for dynamic-resolution processors (Qwen2-VL family)")
    ap.add_argument("--no_attention", action="store_true", help="skip attention mass (allows sdpa)")
    ap.add_argument("--no_logit_lens", action="store_true")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--split", type=int, help="draw a split-layer marker in the plot")
    ap.add_argument("--cos_tau", type=float, default=0.95)
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model(args.model)
    json_path, image_root = default_data() if not args.data_json else (Path(args.data_json), None)
    if args.image_root:
        image_root = Path(args.image_root)
    if image_root is None:
        ap.error("--image_root is required with --data_json")

    torch.manual_seed(args.seed)
    attn_impl = "sdpa" if args.no_attention else "eager"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=DTYPES[args.dtype], attn_implementation=attn_impl,
        device_map={"": 0} if torch.cuda.is_available() else None).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    ip = getattr(processor, "image_processor", None)
    if ip is not None and hasattr(ip, "max_pixels") and args.max_pixels:
        ip.max_pixels = args.max_pixels
        if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
            ip.size["longest_edge"] = args.max_pixels
    parts = sat.locate_parts(model)
    grid = sat.parse_grid(args.image_grid)
    device = parts.embed_tokens.weight.device

    samples = load_samples(json_path, image_root, args.n_samples, args.seed, args.image_prefix)
    curves: dict[str, list] = {}
    n_img, skipped = [], []
    t0 = time.time()
    for i, s in enumerate(samples):
        enc = processor(images=[s["image"]], text=[build_prompt(processor, s["question"])],
                        return_tensors="pt")
        enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
        if "pixel_values" in enc:
            enc["pixel_values"] = enc["pixel_values"].to(DTYPES[args.dtype])
        try:
            if grid is not None:
                enc = sat.pool_image_tokens(model, parts, enc, grid)
            res = sat.analyze_sample(model, parts, enc, attention=not args.no_attention,
                                     logit_lens=not args.no_logit_lens)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            skipped.append(s["id"])
            continue
        n_img.append(int(res.pop("n_img_tokens")[0]))
        res.pop("n_txt_tokens")
        for k, v in res.items():
            curves.setdefault(k, []).append(v)
        if (i + 1) % 50 == 0 or i + 1 == len(samples):
            print(f"[run] {i + 1}/{len(samples)}  {time.time() - t0:.0f}s", flush=True)

    arrays = {k: np.stack(v) for k, v in curves.items()}
    np.savez_compressed(out / "curves.npz", **arrays, sample_ids=np.array([s["id"] for s in samples
                                                                           if s["id"] not in skipped]))
    summary = sat.summarize(arrays, cos_tau=args.cos_tau)
    summary["meta"] = {
        "model": str(model_path), "model_type": model.config.model_type,
        "n_samples": len(n_img), "skipped_oom": skipped, "seed": args.seed,
        "image_grid": args.image_grid, "image_tokens_mean": float(np.mean(n_img)) if n_img else None,
        "image_tokens_range": [min(n_img), max(n_img)] if n_img else None,
        "image_token_id": parts.image_token_id, "dtype": args.dtype, "attn_implementation": attn_impl,
        "data_json": str(json_path), "image_prefix": args.image_prefix,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "host": platform.node(), "seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["transitions"], indent=2))
    plot(summary, out / "saturation.pdf", f"{model_path.name}  grid={args.image_grid or 'native'}",
         args.split)


if __name__ == "__main__":
    main()

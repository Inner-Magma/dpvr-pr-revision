#!/usr/bin/env python
"""Which weights did a DPVR checkpoint start from? (SERVER_HANDOFF §1.4)

At build time the model loads ``llava-1.5-7b-hf`` and then, only if the
directories exist on that machine, swaps in the original Vicuna-7B-v1.5
decoder layers (``llama_subdir``) and the stage-1 projector
(``projector_subdir``). The swapped modules stay frozen during DPVR training,
so the checkpoint still holds them bit for bit. Comparing a few frozen tensors
against both candidates answers the question without running any evaluation.

Usage:
    python scripts/revision/check_weight_provenance.py --checkpoint /ckpt/run --out prov.json

Reads only the probed tensors (safetensors are opened lazily), so it runs on
CPU in under a minute.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


class TensorDir:
    """Lazy name -> tensor lookup over a weights directory or a single file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._sf: dict[str, Path] = {}
        self._bin: dict[str, torch.Tensor] = {}
        files = [self.path] if self.path.is_file() else self._files(self.path)
        for f in files:
            if f.suffix == ".safetensors":
                from safetensors import safe_open
                with safe_open(str(f), framework="pt") as h:
                    for k in h.keys():
                        self._sf[k] = f
            else:
                self._bin.update(torch.load(f, map_location="cpu", weights_only=True, mmap=True))

    @staticmethod
    def _files(d: Path) -> list[Path]:
        sf = sorted(d.glob("*.safetensors"))
        return sf if sf else sorted(d.glob("*.bin"))

    def keys(self) -> list[str]:
        return list(self._sf) + list(self._bin)

    def get(self, key: str) -> torch.Tensor:
        if key in self._bin:
            return self._bin[key]
        from safetensors import safe_open
        with safe_open(str(self._sf[key]), framework="pt") as h:
            return h.get_tensor(key)

    def find(self, pattern: str) -> str | None:
        rx = re.compile(pattern)
        hits = [k for k in self.keys() if rx.search(k) and "vision" not in k]
        return min(hits, key=len) if hits else None


def probes(split_layer: int, n_layers: int) -> dict[str, str]:
    """Probe name -> regex matching the tensor in any of the key layouts."""
    out = {}
    for i in sorted({0, split_layer - 1, split_layer, n_layers - 1}):
        out[f"layer{i}.q_proj"] = rf"(^|\.)layers\.{i}\.self_attn\.q_proj\.weight$"
        out[f"layer{i}.down_proj"] = rf"(^|\.)layers\.{i}\.mlp\.down_proj\.weight$"
    out["final_norm"] = r"(^|\.)norm\.weight$"  # final RMSNorm; per-layer norms end in layernorm
    return out


PROJ_PROBES = {
    "projector.linear_1": (r"multi_modal_projector\.linear_1\.weight$", r"mm_projector\.0\.weight$"),
    "projector.linear_2": (r"multi_modal_projector\.linear_2\.weight$", r"mm_projector\.2\.weight$"),
}


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if a.shape != b.shape:
        return None
    return float((a.to(torch.float32) - b.to(a.dtype).to(torch.float32)).abs().max())


def compare(ckpt: TensorDir, refs: dict[str, TensorDir | None], split_layer: int,
            n_layers: int) -> dict:
    rows = []
    for name, pat in probes(split_layer, n_layers).items():
        ck = ckpt.find(pat)
        row = {"probe": name, "ckpt_key": ck}
        for ref_name in ("llava_hf", "vicuna"):
            ref = refs.get(ref_name)
            rk = ref.find(pat) if (ref is not None and ck) else None
            row[ref_name] = max_diff(ckpt.get(ck), ref.get(rk)) if rk else None
        rows.append(row)
    for name, (pat_hf, pat_stage1) in PROJ_PROBES.items():
        ck = ckpt.find(pat_hf)
        row = {"probe": name, "ckpt_key": ck}
        for ref_name, pat in (("llava_hf", pat_hf), ("stage1_projector", pat_stage1)):
            ref = refs.get(ref_name)
            rk = ref.find(pat) if (ref is not None and ck) else None
            row[ref_name] = max_diff(ckpt.get(ck), ref.get(rk)) if rk else None
        rows.append(row)
    return {"rows": rows, "verdict": verdict(rows)}


def _match(rows: list[dict], prefix: str, ref: str) -> bool | None:
    vals = [r.get(ref) for r in rows if r["probe"].startswith(prefix)]
    if not vals or any(v is None for v in vals):
        return None
    return all(v == 0.0 for v in vals)


def verdict(rows: list[dict]) -> dict:
    lm_rows = [r for r in rows if not r["probe"].startswith("projector")]

    def decide(prefix_rows, a, b):
        ma, mb = _match(prefix_rows, "", a), _match(prefix_rows, "", b)
        if ma and not mb:
            return a
        if mb and not ma:
            return b
        if ma and mb:
            return "ambiguous (both references identical)"
        if ma is None and mb is None:
            return "unknown (references missing)"
        if ma is None or mb is None:
            missing = a if ma is None else b
            return f"unknown (not the other one; {missing} reference missing)"
        return "neither (weights differ from both references)"

    return {
        "language_model": decide(lm_rows, "llava_hf", "vicuna"),
        "projector": decide([r for r in rows if r["probe"].startswith("projector")],
                            "llava_hf", "stage1_projector"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base_model", default="llava-1.5-7b-hf")
    ap.add_argument("--llama_subdir", default="llava1.5-7b-pretrained/vicuna-7b-v1.5")
    ap.add_argument("--projector_subdir",
                    default="llava1.5-7b-pretrained/llava-v1.5-7b-pretrain/mm_projector.bin")
    ap.add_argument("--vicuna", default=None, help="override: Vicuna weights dir")
    ap.add_argument("--projector", default=None, help="override: stage-1 mm_projector.bin")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from dpvr.utils import load_paths

    paths = load_paths()
    ckpt_dir = Path(args.checkpoint)
    split_layer, n_layers = 18, 32
    cfg_file = ckpt_dir / "training_config.json"
    if cfg_file.exists():
        cfg = json.loads(cfg_file.read_text())
        split_layer = int(cfg.get("method", {}).get("split_layer", split_layer))
    base = paths.model_root / args.base_model
    cfg_json = base / "config.json"
    if cfg_json.exists():
        c = json.loads(cfg_json.read_text())
        n_layers = int(c.get("text_config", {}).get("num_hidden_layers", n_layers))

    def maybe(p: Path) -> TensorDir | None:
        return TensorDir(p) if p.exists() else None

    refs = {
        "llava_hf": maybe(base),
        "vicuna": maybe(Path(args.vicuna) if args.vicuna else paths.model_root / args.llama_subdir),
        "stage1_projector": maybe(Path(args.projector) if args.projector
                                  else paths.model_root / args.projector_subdir),
    }
    report = compare(TensorDir(ckpt_dir), refs, split_layer, n_layers)
    report["checkpoint"] = str(ckpt_dir)
    report["references_found"] = {k: v is not None for k, v in refs.items()}

    print(f"{'probe':22s} {'llava_hf':>12s} {'vicuna/stage1':>14s}")
    for r in report["rows"]:
        other = r.get("vicuna", r.get("stage1_projector"))
        fmt = lambda v: "n/a" if v is None else f"{v:.3g}"
        print(f"{r['probe']:22s} {fmt(r.get('llava_hf')):>12s} {fmt(other):>14s}")
    print("verdict:", json.dumps(report["verdict"], ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

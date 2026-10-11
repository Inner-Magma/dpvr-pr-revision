#!/usr/bin/env python
"""E0 environment self-check: versions, GPUs, paths, models, data, benchmarks.

Usage:
    python scripts/revision/check_env.py                       # print the table
    python scripts/revision/check_env.py --out results/revision/E0/env_check.json

Exit code 1 when a required item fails. Optional items (13B weights, Vicuna /
stage-1 projector, GQA) only warn, because whether they should exist depends
on the weight-provenance result (SERVER_HANDOFF §1.4).
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "revision"))

IMAGE_SUBDIRS = ["coco/train2017", "gqa/images", "ocr_vqa/images", "textvqa/train_images",
                 "vg/VG_100K", "vg/VG_100K_2"]


class Checks:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, group: str, name: str, ok: bool, detail: str = "", required: bool = True):
        self.items.append({"group": group, "name": name, "ok": bool(ok),
                           "required": required, "detail": detail})

    @property
    def failed(self) -> list[dict]:
        return [i for i in self.items if i["required"] and not i["ok"]]


def locked_versions() -> dict[str, str]:
    pins = {}
    lock = REPO_ROOT / "requirements-lock.txt"
    if lock.exists():
        for line in lock.read_text().splitlines():
            m = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s#]+)", line)
            if m:
                pins[m.group(1)] = m.group(2)
    pins.setdefault("lmms_eval", "0.5.0")
    return pins


def check_versions(c: Checks) -> None:
    py = platform.python_version()
    c.add("software", "python 3.10", py.startswith("3.10."), py)
    for pkg, want in locked_versions().items():
        try:
            have = version(pkg)
        except PackageNotFoundError:
            have = None
        ok = have is not None and have.split("+")[0] == want
        c.add("software", f"{pkg}=={want}", ok, have or "not installed")


def check_gpus(c: Checks) -> list[dict]:
    import torch

    gpus = []
    c.add("gpu", "CUDA available", torch.cuda.is_available(), f"torch CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        return gpus
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        try:
            a = torch.randn(1024, 1024, device=f"cuda:{i}", dtype=torch.bfloat16)
            ok = bool(torch.isfinite(a @ a).all())
        except Exception:  # pragma: no cover - hardware specific
            ok = False
        gpus.append({"index": i, "name": p.name, "capability": f"{p.major}.{p.minor}",
                     "total_gb": round(total / 2**30, 1), "free_gb": round(free / 2**30, 1)})
        c.add("gpu", f"cuda:{i} bf16 matmul", ok,
              f"{p.name} sm_{p.major}{p.minor} {total / 2**30:.0f} GB, {free / 2**30:.0f} GB free")
    return gpus


def check_paths(c: Checks) -> None:
    from dpvr.utils import load_config, load_paths, resolve_paths

    try:
        paths = load_paths()
    except RuntimeError as e:
        c.add("paths", ".env / configs/paths.yaml", False, str(e))
        return
    for name in ("data_root", "model_root", "ckpt_root"):
        p = getattr(paths, name)
        c.add("paths", name, p.exists(), str(p))

    cfg = resolve_paths(load_config(str(REPO_ROOT / "configs/method/token_diversion_x3_fusion.yaml")), paths)
    m = cfg["model"]
    c.add("models", "llava-1.5-7b-hf", (Path(m["base_model_path"]) / "config.json").exists(),
          m["base_model_path"])
    c.add("models", "llava-1.5-13b-hf", (paths.model_root / "llava-1.5-13b-hf/config.json").exists(),
          str(paths.model_root / "llava-1.5-13b-hf"), required=False)
    for key, label in (("llama_path", "vicuna-7b-v1.5 (llama_subdir)"),
                       ("projector_weights_path", "stage-1 mm_projector.bin (projector_subdir)"),
                       ("vision_tower_path", "clip-vit-large-patch14-336 (vision_tower_subdir)")):
        p = m.get(key)
        c.add("models", label, bool(p) and Path(p).exists(),
              f"{p} — must match the §1.4 provenance result", required=False)

    d = cfg["data"]
    c.add("data", "llava_v1_5_mix665k.json", Path(d["json_path"]).exists(), d["json_path"])
    for sub in IMAGE_SUBDIRS:
        p = Path(d["image_base_path"]) / sub
        c.add("data", f"images/{sub}", p.is_dir(), str(p))

    import tempfile

    import eval_lmms

    bench_root = paths.data_root / "benchmarks"
    with tempfile.TemporaryDirectory() as tmp:
        for name in eval_lmms.EIGHT + ["gqa"]:
            refs = eval_lmms.materialize_tasks([name], bench_root, Path(tmp))
            ok = bool(refs) and all(r.exists() for r in refs)
            c.add("benchmarks", name, ok, ", ".join(map(str, refs)), required=name != "gqa")


def check_imports(c: Checks) -> None:
    import os

    os.environ.setdefault("LMMS_EVAL_PLUGINS", "dpvr.lmms_plugin")
    try:
        from lmms_eval.models import get_model
        cls = get_model("dpvr_hf")
        c.add("software", "lmms-eval plugin dpvr_hf", cls.__name__ == "DPVRHf", cls.__module__)
    except Exception as e:
        c.add("software", "lmms-eval plugin dpvr_hf", False, repr(e)[:200])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip", action="append", default=[], choices=["gpu", "paths", "imports"])
    args = ap.parse_args()

    c = Checks()
    check_versions(c)
    gpus = check_gpus(c) if "gpu" not in args.skip else []
    if "imports" not in args.skip:
        check_imports(c)
    if "paths" not in args.skip:
        check_paths(c)

    for i in c.items:
        mark = "✓" if i["ok"] else ("✗" if i["required"] else "!")
        print(f"{mark} [{i['group']}] {i['name']}: {i['detail']}")
    print(f"\n{len(c.failed)} required check(s) failed")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"gpus": gpus, "checks": c.items}, indent=2,
                                             ensure_ascii=False))
    sys.exit(1 if c.failed else 0)


if __name__ == "__main__":
    main()

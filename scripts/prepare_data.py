#!/usr/bin/env python
"""Check LLaVA-665k image integrity against the paths declared in the config."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.data import verify_image_existence  # noqa: E402
from dpvr.utils import apply_cli_overrides, load_config, load_paths, resolve_paths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/method/token_diversion.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args.overrides)
    cfg = resolve_paths(cfg, load_paths())

    verify_image_existence(cfg["data"]["json_path"], cfg["data"]["image_base_path"])


if __name__ == "__main__":
    main()

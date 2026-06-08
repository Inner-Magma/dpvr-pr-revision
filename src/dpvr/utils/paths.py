"""Machine-local path resolution.

Order of precedence (highest → lowest):

1. ``configs/paths.yaml`` (gitignored, per-machine override)
2. Environment variables loaded from ``.env``
3. Environment variables already set in the shell
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

try:  # python-dotenv is optional — silently skip if missing
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class PathConfig:
    data_root: Path
    model_root: Path
    ckpt_root: Path
    lmms_eval_path: Path | None = None
    hf_endpoint: str | None = None

    def join_data(self, rel: str | Path) -> Path:
        return self.data_root / rel

    def join_model(self, rel: str | Path) -> Path:
        return self.model_root / rel

    def join_ckpt(self, rel: str | Path) -> Path:
        return self.ckpt_root / rel


def load_paths(paths_yaml: str | Path | None = None) -> PathConfig:
    """Resolve path roots for the current machine."""
    load_dotenv(REPO_ROOT / ".env", override=False)

    values = {
        "data_root": os.environ.get("DATA_ROOT"),
        "model_root": os.environ.get("MODEL_ROOT"),
        "ckpt_root": os.environ.get("CKPT_ROOT"),
        "lmms_eval_path": os.environ.get("LMMS_EVAL_PATH"),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
    }

    candidate = Path(paths_yaml) if paths_yaml else REPO_ROOT / "configs" / "paths.yaml"
    if candidate.exists():
        with open(candidate, "r", encoding="utf-8") as f:
            for k, v in (yaml.safe_load(f) or {}).items():
                if v is not None:
                    values[k] = v

    missing = [k for k in ("data_root", "model_root", "ckpt_root") if not values[k]]
    if missing:
        raise RuntimeError(
            f"Path roots not set: {missing}. Copy .env.example → .env or "
            f"configs/paths.yaml.example → configs/paths.yaml and fill in values."
        )

    return PathConfig(
        data_root=Path(values["data_root"]).expanduser(),
        model_root=Path(values["model_root"]).expanduser(),
        ckpt_root=Path(values["ckpt_root"]).expanduser(),
        lmms_eval_path=Path(values["lmms_eval_path"]).expanduser()
        if values["lmms_eval_path"]
        else None,
        hf_endpoint=values["hf_endpoint"] or None,
    )

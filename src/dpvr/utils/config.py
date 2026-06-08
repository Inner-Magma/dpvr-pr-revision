"""Tiny YAML config loader with Hydra-style ``defaults`` list and CLI overrides.

Design goals:
  * Readable by anyone who has seen YAML — no framework to learn.
  * Supports a one-level ``defaults:`` list so method configs can inherit ``_base.yaml``.
  * Supports ``key.nested=value`` CLI overrides for ablation sweeps.

Not a general-purpose config system. If you find yourself reaching for more,
adopt Hydra/OmegaConf outright rather than growing this file.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

import yaml

from .paths import PathConfig


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config, resolving its ``defaults:`` list relative to /configs."""
    config_path = Path(config_path).resolve()
    configs_root = config_path.parent
    while configs_root.name and configs_root.name != "configs":
        configs_root = configs_root.parent
    if configs_root.name != "configs":
        configs_root = config_path.parent

    raw = _load_yaml(config_path)
    defaults = raw.pop("defaults", [])
    if not isinstance(defaults, list):
        raise ValueError(f"'defaults' in {config_path} must be a list of yaml names")

    merged: dict = {}
    for entry in defaults:
        default_path = (configs_root / entry).resolve()
        if not default_path.exists():
            raise FileNotFoundError(f"default '{entry}' referenced from {config_path} not found at {default_path}")
        merged = _deep_merge(merged, _load_yaml(default_path))
    merged = _deep_merge(merged, raw)
    return merged


def _coerce(value: str) -> Any:
    """Coerce a CLI string into a Python literal when possible (int/float/bool/list)."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def apply_cli_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``key.nested=value`` style overrides in-place (returns the cfg)."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override '{item}' must be of the form key.path=value")
        key, value = item.split("=", 1)
        target = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in target or not isinstance(target[p], dict):
                target[p] = {}
            target = target[p]
        target[parts[-1]] = _coerce(value)
    return cfg


def resolve_paths(cfg: dict, paths: PathConfig) -> dict:
    """Resolve ``*_subdir`` / ``*_rel`` keys against the machine's path roots.

    Adds (and does not remove) absolute path keys so downstream code can use them
    directly. Conventions:
      model.base_model, model.llama_subdir, model.vision_tower_subdir,
      model.projector_subdir   -> absolute paths under ${MODEL_ROOT}
      data.json_rel, data.image_rel -> absolute paths under ${DATA_ROOT}
      training.output_subdir -> absolute path under ${CKPT_ROOT}
    """
    cfg = copy.deepcopy(cfg)

    model = cfg.get("model", {})
    if model.get("base_model"):
        model["base_model_path"] = str(paths.join_model(model["base_model"]))
    if model.get("llama_subdir"):
        model["llama_path"] = str(paths.join_model(model["llama_subdir"]))
    if model.get("vision_tower_subdir"):
        model["vision_tower_path"] = str(paths.join_model(model["vision_tower_subdir"]))
    if model.get("projector_subdir"):
        model["projector_weights_path"] = str(paths.join_model(model["projector_subdir"]))

    data = cfg.get("data", {})
    if "json_rel" in data:
        data["json_path"] = str(paths.join_data(data["json_rel"]))
    if "image_rel" in data:
        data["image_base_path"] = str(paths.join_data(data["image_rel"]))

    training = cfg.get("training", {})
    if "output_subdir" in training:
        training["output_dir"] = str(paths.join_ckpt(training["output_subdir"]))

    return cfg

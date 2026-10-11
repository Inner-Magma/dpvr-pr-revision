"""Load training checkpoints on a machine other than the one that wrote them.

``from_trained`` on the model classes reads ``training_config.json`` and uses
the absolute paths stored in it, which point at the old server. The helpers
here re-resolve those paths against the current machine's roots (the config
also stores the relative ``*_subdir`` keys) and report missing / unexpected
keys instead of silently ignoring them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class LoadReport:
    method: str
    n_tensors: int
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"method={self.method} tensors={self.n_tensors} "
                f"missing={len(self.missing)} unexpected={len(self.unexpected)}")


def weight_files(checkpoint_dir: str | Path) -> list[Path]:
    """Weight files of an HF Trainer save, in load order."""
    ckpt = Path(checkpoint_dir)
    index = ckpt / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [ckpt / s for s in shards]
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (ckpt / name).exists():
            return [ckpt / name]
    bin_index = ckpt / "pytorch_model.bin.index.json"
    if bin_index.exists():
        shards = sorted(set(json.loads(bin_index.read_text())["weight_map"].values()))
        return [ckpt / s for s in shards]
    raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")


def load_state(checkpoint_dir: str | Path) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for f in weight_files(checkpoint_dir):
        if f.suffix == ".safetensors":
            from safetensors.torch import load_file
            state.update(load_file(str(f)))
        else:
            state.update(torch.load(f, map_location="cpu", weights_only=True))
    return state


def read_training_config(checkpoint_dir: str | Path, paths=None) -> dict:
    """training_config.json with model/data paths re-resolved for this machine."""
    from ..utils import load_paths, resolve_paths

    cfg_file = Path(checkpoint_dir) / "training_config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"training_config.json not found in {checkpoint_dir}")
    cfg = json.loads(cfg_file.read_text())
    return resolve_paths(cfg, paths if paths is not None else load_paths())


def load_trained(checkpoint_dir: str | Path, paths=None):
    """Build the model named in the checkpoint's config and load its weights.

    Returns ``(model, LoadReport)``. For full_ft the inner HF model is
    returned, matching ``TokenDiversionLlava.from_trained``.
    """
    import dpvr.models  # noqa: F401  (runs the @register_method decorators)

    from .registry import _REGISTRY

    cfg = read_training_config(checkpoint_dir, paths)
    method = cfg.get("method", {}).get("name", "token_diversion")
    if method not in _REGISTRY:
        raise ValueError(f"checkpoint method {method!r} is not registered: {sorted(_REGISTRY)}")
    model = _REGISTRY[method].from_config(cfg)
    state = load_state(checkpoint_dir)
    result = model.load_state_dict(state, strict=False)
    report = LoadReport(method=method, n_tensors=len(state),
                        missing=list(result.missing_keys),
                        unexpected=list(result.unexpected_keys))
    if method == "full_ft" and hasattr(model, "llava_model"):
        model = model.llava_model
    return model, report

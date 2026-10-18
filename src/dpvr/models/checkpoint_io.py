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


def _training_config_file(checkpoint_dir: Path) -> Path:
    """``training_config.json`` of a checkpoint dir, also when it sits one level down.

    Relay layouts differ: some runs keep it in the run directory next to the
    weights; LoRA / FullFT runs have the weights in the run directory but the
    config only in ``checkpoint-N/`` (or the other way round). Take the run's
    own file first, else the one in the largest ``checkpoint-N``.
    """
    own = checkpoint_dir / "training_config.json"
    if own.exists():
        return own
    steps = [d for d in checkpoint_dir.glob("checkpoint-*") if d.is_dir()]
    if steps:
        largest = max(steps, key=lambda d: int(d.name.rsplit("-", 1)[-1]) if d.name.rsplit("-", 1)[-1].isdigit() else -1)
        nested = largest / "training_config.json"
        if nested.exists():
            return nested
    raise FileNotFoundError(f"training_config.json not found in {checkpoint_dir}")


def _rebase_stale(cfg: dict, paths) -> dict:
    """Point old-server absolute paths at this machine's roots.

    The relayed configs were written on the old box and still carry
    ``/nvme/lsy/...``. ``resolve_paths`` only overwrites a path when the relative
    key (``base_model``, ``llama_subdir``, ...) survives; the run configs saved by
    the old trainer only kept ``base_model_path`` in some cases, so the stale
    value is used as is and loading dies on a directory that does not exist.
    Rewrite any ``*_path`` that does not exist by the same tail under
    MODEL_ROOT / DATA_ROOT, else by its basename.
    """
    model_root = Path(paths.model_root)
    data_root = Path(paths.data_root)
    model_keys = ("base_model_path", "llama_path", "vision_tower_path", "projector_weights_path")
    data_keys = ("json_path", "image_base_path")

    for section, keys, root in ((cfg.get("model", {}), model_keys, model_root),
                                (cfg.get("data", {}), data_keys, data_root)):
        for k in keys:
            v = section.get(k)
            if not v or Path(v).exists():
                continue
            cand = root / Path(v).name
            if cand.exists():
                section[k] = str(cand)
                continue
            tail = Path(v).parts
            for i in range(1, len(tail)):
                cand = root.joinpath(*tail[i:])
                if cand.exists():
                    section[k] = str(cand)
                    break
    return cfg


def read_training_config(checkpoint_dir: str | Path, paths=None) -> dict:
    """training_config.json with model/data paths re-resolved for this machine."""
    from ..utils import load_paths, resolve_paths

    cfg_file = _training_config_file(Path(checkpoint_dir))
    cfg = json.loads(cfg_file.read_text())
    paths = paths if paths is not None else load_paths()
    return _rebase_stale(resolve_paths(cfg, paths), paths)


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

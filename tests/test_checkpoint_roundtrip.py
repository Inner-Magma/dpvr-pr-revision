"""Phase 2 acceptance test: checkpoint save → load round-trip.

Saves the model state_dict, reloads it into a fresh instance, and asserts
that every parameter under ``llava_model.*`` and ``single_transformer.*``
matches exactly (bit-for-bit, no fp precision loss since we only move tensors
through CPU).

Run with:
    CUDA_VISIBLE_DEVICES=0 pytest tests/test_checkpoint_roundtrip.py -v
or:
    CUDA_VISIBLE_DEVICES=0 python tests/test_checkpoint_roundtrip.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.utils import load_config, load_paths, resolve_paths
from dpvr.models import TokenDiversionLlava


def _load_model_from_config() -> TokenDiversionLlava:
    cfg = load_config(str(REPO_ROOT / "configs/method/token_diversion.yaml"))
    paths = load_paths()
    cfg = resolve_paths(cfg, paths)
    return TokenDiversionLlava(
        base_model_path=cfg["model"]["base_model_path"],
        llama_path=cfg["model"].get("llama_path"),
        vision_tower_path=cfg["model"].get("vision_tower_path"),
        projector_weights_path=cfg["model"].get("projector_weights_path"),
        split_layer=cfg["method"]["split_layer"],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_checkpoint_roundtrip():
    """state_dict keys and values survive save → load exactly."""
    model_a = _load_model_from_config()
    model_a.eval()

    sd_a = {k: v.clone() for k, v in model_a.state_dict().items()}

    # Verify key namespaces are present.
    llava_keys = [k for k in sd_a if k.startswith("llava_model.")]
    single_keys = [k for k in sd_a if k.startswith("single_transformer.")]
    assert llava_keys, "No llava_model.* keys found in state_dict."
    assert single_keys, "No single_transformer.* keys found in state_dict."

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "model.pt"
        torch.save(sd_a, ckpt_path)

        model_b = _load_model_from_config()
        model_b.eval()
        sd_loaded = torch.load(ckpt_path, map_location="cpu")
        model_b.load_state_dict(sd_loaded, strict=True)

    sd_b = model_b.state_dict()

    mismatches = []
    for key in sd_a:
        if key not in sd_b:
            mismatches.append(f"  MISSING in reloaded model: {key}")
            continue
        if not torch.equal(sd_a[key].cpu(), sd_b[key].cpu()):
            diff = (sd_a[key].float() - sd_b[key].float()).abs().max().item()
            mismatches.append(f"  VALUE MISMATCH  {key}  max|diff|={diff:.2e}")

    assert not mismatches, "Checkpoint round-trip failed:\n" + "\n".join(mismatches)
    print(
        f"[PASS] {len(sd_a)} keys round-tripped exactly "
        f"({len(llava_keys)} llava_model.*, {len(single_keys)} single_transformer.*)"
    )


if __name__ == "__main__":
    test_checkpoint_roundtrip()
    print("Checkpoint round-trip check passed.")

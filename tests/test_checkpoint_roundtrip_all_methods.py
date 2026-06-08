"""Phase B audit: checkpoint save → load roundtrip for all 5 methods.

Strategy for each method:
  1. build_model(cfg) from registry
  2. torch.save(state_dict) to a temp file
  3. torch.load back from temp file
  4. Assert key sets and shapes are identical
  5. model.load_state_dict(loaded) and assert values are bit-for-bit equal

LoRA save strategy (documented here per Phase B spec):
  - LoRALlava wraps a PEFT model in self.llava_model.
  - state_dict() includes both base-model weights AND adapter (LoRA) weights
    under the PEFT-renamed keys (e.g. base_model.model.model.layers.*).
  - For real inference, adapter-only saving (model.llava_model.save_pretrained)
    is preferred (saves ~13 MB vs ~13 GB).  For roundtrip correctness, we test
    the full state_dict path here.
  - load_state_dict uses strict=True (all PEFT keys are present in the full sd).

Run:
    pytest tests/test_checkpoint_roundtrip_all_methods.py -v
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
from dpvr.models import build_model


def _load_cfg(method_name: str) -> dict:
    cfg_path = REPO_ROOT / f"configs/method/{method_name}.yaml"
    cfg = load_config(str(cfg_path))
    paths = load_paths()
    return resolve_paths(cfg, paths)


METHODS = [
    "token_diversion",
    "lora",
    "fastv",
    "full_ft",
    "td_plus_fastv",
]


@pytest.mark.parametrize("method_name", METHODS)
def test_roundtrip(method_name: str, tmp_path: Path):
    """state_dict keys and tensor shapes survive torch.save → torch.load."""
    cfg = _load_cfg(method_name)
    model = build_model(cfg)
    model.eval()

    sd_orig = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    ckpt_file = tmp_path / f"{method_name}.pt"
    torch.save(sd_orig, ckpt_file)

    sd_loaded = torch.load(ckpt_file, map_location="cpu")

    # Key-set parity.
    missing = set(sd_orig.keys()) - set(sd_loaded.keys())
    extra   = set(sd_loaded.keys()) - set(sd_orig.keys())
    assert not missing, f"[{method_name}] Keys missing after load: {sorted(missing)[:5]}"
    assert not extra,   f"[{method_name}] Extra keys after load: {sorted(extra)[:5]}"

    # Shape parity for every tensor.
    shape_errs = []
    for k in sd_orig:
        if sd_orig[k].shape != sd_loaded[k].shape:
            shape_errs.append(
                f"  {k}: orig={sd_orig[k].shape}  loaded={sd_loaded[k].shape}"
            )
    assert not shape_errs, (
        f"[{method_name}] Shape mismatches:\n" + "\n".join(shape_errs)
    )

    # Reload weights into model and verify bit-for-bit equality.
    model.load_state_dict(sd_loaded, strict=True)
    sd_after = {k: v.cpu() for k, v in model.state_dict().items()}
    value_errs = []
    for k in sd_orig:
        if not torch.equal(sd_orig[k], sd_after[k]):
            diff = (sd_orig[k].float() - sd_after[k].float()).abs().max().item()
            value_errs.append(f"  {k}  max|diff|={diff:.2e}")
    assert not value_errs, (
        f"[{method_name}] Value mismatches after reload:\n" + "\n".join(value_errs[:5])
    )

    total_keys = len(sd_orig)
    trainable  = sum(1 for k in sd_orig
                     if any(k.startswith(p) for p in
                            ("single_transformer.", "lora_A", "lora_B")))
    print(f"[{method_name}] PASS  {total_keys} keys, trainable-ish={trainable}")

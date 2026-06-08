"""Phase 2 acceptance test: training forward ≡ inference PREFILL.

Asserts that for the same weights and inputs, forward(use_cache=False) and
forward(use_cache=True) with no prior cache produce identical logits
(absolute difference < 1e-3, accounting for bfloat16 precision).

Run with:
    CUDA_VISIBLE_DEVICES=0 pytest tests/test_forward_parity.py -v
or:
    CUDA_VISIBLE_DEVICES=0 python tests/test_forward_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.utils import load_config, load_paths, resolve_paths
from dpvr.models import TokenDiversionLlava


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_model_from_config() -> TokenDiversionLlava:
    cfg = load_config(str(REPO_ROOT / "configs/method/token_diversion.yaml"))
    paths = load_paths()
    cfg = resolve_paths(cfg, paths)
    model = TokenDiversionLlava(
        base_model_path=cfg["model"]["base_model_path"],
        llama_path=cfg["model"].get("llama_path"),
        vision_tower_path=cfg["model"].get("vision_tower_path"),
        projector_weights_path=cfg["model"].get("projector_weights_path"),
        split_layer=cfg["method"]["split_layer"],
    )
    return model


def _make_dummy_batch(model: TokenDiversionLlava, device: torch.device):
    """Return a minimal (B=1) batch with image tokens."""
    from transformers import AutoProcessor
    from PIL import Image
    import numpy as np

    cfg_path = str(REPO_ROOT / "configs/method/token_diversion.yaml")
    cfg = load_config(cfg_path)
    paths = load_paths()
    cfg = resolve_paths(cfg, paths)

    processor = AutoProcessor.from_pretrained(cfg["model"]["base_model_path"])
    # Solid-red image and a minimal prompt with <image> placeholder.
    image = Image.fromarray(np.full((336, 336, 3), [200, 50, 50], dtype="uint8"))
    text = "USER: <image>\nDescribe the image. ASSISTANT:"
    encoding = processor(images=[image], text=[text], return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in encoding.items()}


# ── test ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_training_inference_prefill_parity():
    """logits from use_cache=False ≡ logits from use_cache=True (PREFILL)."""
    device = torch.device("cuda")
    model = _load_model_from_config()
    model.eval()
    model.to(device)

    batch = _make_dummy_batch(model, device)
    input_ids = batch["input_ids"]
    pixel_values = batch.get("pixel_values")
    attention_mask = batch.get("attention_mask")

    with torch.no_grad():
        out_train = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=False,
        )
        out_infer = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=True,
        )

    logits_train = out_train.logits.float()
    logits_infer = out_infer.logits.float()

    max_diff = (logits_train - logits_infer).abs().max().item()
    assert max_diff < 1e-3, (
        f"Training and inference PREFILL logits diverge: max |diff| = {max_diff:.2e} "
        f"(threshold 1e-3). Check that use_cache does not alter the forward computation."
    )
    # KV-cache should be populated in the inference run.
    assert out_infer.past_key_values is not None, (
        "use_cache=True should return populated past_key_values."
    )
    assert out_train.past_key_values is None, (
        "use_cache=False should return past_key_values=None."
    )
    print(f"[PASS] max |logits_train - logits_infer| = {max_diff:.2e}")


if __name__ == "__main__":
    test_training_inference_prefill_parity()
    print("All parity checks passed.")

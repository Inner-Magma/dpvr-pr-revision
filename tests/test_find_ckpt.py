"""scripts/revision/find_ckpt.py: picks the checkpoint dir in relayed run layouts."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("find_ckpt", REPO_ROOT / "scripts/revision/find_ckpt.py")
find_ckpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(find_ckpt)


def _ckpt(d: Path, weights: str = "model.safetensors", cfg: str = "training_config.json") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / weights).write_bytes(b"x")
    (d / cfg).write_text("{}")
    return d


def test_top_level_final_save(tmp_path):
    run = _ckpt(tmp_path / "llava665k_split18_d1")
    _ckpt(run / "checkpoint-100")
    assert find_ckpt.find_checkpoint(run) == run


def test_extra_run_level(tmp_path):
    root = tmp_path / "full_ft"
    run = _ckpt(root / "llava665k_fullft")
    _ckpt(run / "checkpoint-500")
    assert find_ckpt.find_checkpoint(root) == run


def test_only_step_checkpoints_picks_largest(tmp_path):
    root = tmp_path / "lora"
    for n in (900, 10000, 2000):
        _ckpt(root / "run" / f"checkpoint-{n}", weights="adapter_model.safetensors")
    (root / "run" / "trainer_state.json").write_text("{}")
    assert find_ckpt.find_checkpoint(root) == root / "run" / "checkpoint-10000"


def test_optimizer_only_is_not_a_checkpoint(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "optimizer.bin").write_bytes(b"x")
    (d / "training_config.json").write_text("{}")
    assert find_ckpt.find_checkpoint(tmp_path) is None


def test_missing_root(tmp_path):
    assert find_ckpt.find_checkpoint(tmp_path / "nope") is None

"""`dpvr.models.checkpoint_io`：中转 checkpoint 的旧路径与分散布局。

现象（2026-10-18，E12）：六个任务全部在加载阶段失败。
  - `pc7b_s18` 等：`OSError: Can't load the configuration of '/nvme/lsy/models/llava-1.5-7b-hf'`
    —— 旧服务器写死的绝对路径，`resolve_paths` 只在相对键存在时才会覆盖它；
  - `lora7b`：`FileNotFoundError: training_config.json not found in .../lora/llava665k_r16`
    —— 权重在运行目录、配置只在 `checkpoint-156153/` 里。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.models.checkpoint_io import _rebase_stale, _training_config_file, read_training_config  # noqa: E402


def _paths(tmp_path: Path):
    for sub in ("models/llava-1.5-7b-hf", "data/llava-1.5-665k-instructions"):
        (tmp_path / sub).mkdir(parents=True)
    (tmp_path / "models/llava-1.5-7b-hf/config.json").write_text("{}")
    (tmp_path / "data/llava-1.5-665k-instructions/x.json").write_text("{}")
    return SimpleNamespace(model_root=tmp_path / "models", data_root=tmp_path / "data",
                           ckpt_root=tmp_path / "ckpt")


def test_stale_absolute_paths_are_rebased():
    paths = SimpleNamespace(model_root=Path("/models"), data_root=Path("/data"))
    cfg = {"model": {"base_model_path": "/nvme/lsy/models/llava-1.5-7b-hf"},
           "data": {"json_path": "/nvme/lsy/data/llava-1.5-665k-instructions/llava_v1_5_mix665k.json"}}
    # 两个目标都不存在 → 保持原值，不瞎猜
    assert _rebase_stale(json.loads(json.dumps(cfg)), paths)["model"]["base_model_path"].startswith("/nvme/")


def test_stale_paths_rebased_against_roots(tmp_path):
    paths = _paths(tmp_path)
    cfg = {"model": {"base_model_path": "/nvme/lsy/models/llava-1.5-7b-hf"},
           "data": {"json_path": "/nvme/lsy/data/llava-1.5-665k-instructions/x.json"}}
    out = _rebase_stale(cfg, paths)
    assert out["model"]["base_model_path"] == str(tmp_path / "models/llava-1.5-7b-hf")
    assert out["data"]["json_path"] == str(tmp_path / "data/llava-1.5-665k-instructions/x.json")


def test_existing_path_is_kept(tmp_path):
    paths = _paths(tmp_path)
    good = str(tmp_path / "models/llava-1.5-7b-hf")
    out = _rebase_stale({"model": {"base_model_path": good}}, paths)
    assert out["model"]["base_model_path"] == good


def test_config_found_in_largest_step_dir(tmp_path):
    run = tmp_path / "llava665k_r16"
    (run / "checkpoint-900").mkdir(parents=True)
    (run / "checkpoint-156153").mkdir()
    (run / "checkpoint-156153/training_config.json").write_text("{}")
    assert _training_config_file(run) == run / "checkpoint-156153/training_config.json"


def test_config_own_file_wins(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoint-10").mkdir(parents=True)
    (run / "checkpoint-10/training_config.json").write_text("{}")
    (run / "training_config.json").write_text("{}")
    assert _training_config_file(run) == run / "training_config.json"


def test_read_training_config_end_to_end(tmp_path):
    paths = _paths(tmp_path)
    run = tmp_path / "ckpt/run"
    (run / "checkpoint-5").mkdir(parents=True)
    (run / "checkpoint-5/training_config.json").write_text(json.dumps({
        "model": {"base_model_path": "/nvme/lsy/models/llava-1.5-7b-hf"},
        "method": {"name": "token_diversion", "split_layer": 18},
    }))
    cfg = read_training_config(run, paths)
    assert cfg["model"]["base_model_path"] == str(tmp_path / "models/llava-1.5-7b-hf")
    assert cfg["method"]["split_layer"] == 18


def test_missing_config_raises(tmp_path):
    try:
        _training_config_file(tmp_path)
    except FileNotFoundError as e:
        assert "training_config.json" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError")

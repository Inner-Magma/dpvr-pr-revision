"""Revision evaluation chain: task templates, checkpoint loading, lmms-eval plugin."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "revision"))

import eval_lmms  # noqa: E402

from dpvr.models import checkpoint_io  # noqa: E402
from dpvr.utils import PathConfig  # noqa: E402


def test_eight_expands_to_paper_benchmarks():
    names = eval_lmms.parse_tasks("eight")
    assert names == eval_lmms.EIGHT
    assert eval_lmms.parse_tasks("pope_local,scienceqa,pope") == ["pope", "scienceqa"]
    with pytest.raises(SystemExit):
        eval_lmms.parse_tasks("vqav2")


def test_materialize_rewrites_paths_and_drops_upstream_yaml(tmp_path):
    refs = eval_lmms.materialize_tasks(eval_lmms.EIGHT, Path("/bench"), tmp_path)
    assert len(refs) == len(eval_lmms.EIGHT)
    assert all(str(r).startswith("/bench/") for r in refs)
    for d in tmp_path.iterdir():
        yamls = [f.name for f in d.glob("*.yaml")]
        assert len(yamls) == 1 and yamls[0].endswith("_local.yaml"), d
        assert "/path/to/benchmarks" not in (d / yamls[0]).read_text()
        assert (d / "utils.py").exists()


def test_collect_metrics_reads_latest_results(tmp_path):
    out = tmp_path / "lmms" / "model"
    out.mkdir(parents=True)
    (out / "a_results.json").write_text(json.dumps(
        {"results": {"pope_local": {"alias": "pope_local", "pope_accuracy,none": 0.85,
                                    "pope_accuracy_stderr,none": "N/A"}}}))
    assert eval_lmms.collect_metrics(tmp_path / "lmms") == {"pope_local": {"pope_accuracy,none": 0.85}}


def _paths(tmp_path) -> PathConfig:
    return PathConfig(data_root=tmp_path / "data", model_root=tmp_path / "models",
                      ckpt_root=tmp_path / "ckpt")


def test_training_config_paths_are_re_resolved(tmp_path):
    ckpt = tmp_path / "run"
    ckpt.mkdir()
    old = {"method": {"name": "token_diversion_x3_fusion"},
           "model": {"base_model": "llava-1.5-7b-hf", "base_model_path": "/old/server/llava-1.5-7b-hf",
                     "llama_subdir": "llava1.5-7b-pretrained/vicuna-7b-v1.5",
                     "llama_path": "/old/server/vicuna"},
           "training": {"output_subdir": "x"}}
    (ckpt / "training_config.json").write_text(json.dumps(old))
    cfg = checkpoint_io.read_training_config(ckpt, _paths(tmp_path))
    assert cfg["model"]["base_model_path"] == str(tmp_path / "models" / "llava-1.5-7b-hf")
    assert cfg["model"]["llama_path"] == str(tmp_path / "models" / "llava1.5-7b-pretrained/vicuna-7b-v1.5")


def test_load_state_sharded_and_single(tmp_path):
    single = tmp_path / "single"
    single.mkdir()
    save_file({"a": torch.ones(2)}, str(single / "model.safetensors"))
    assert set(checkpoint_io.load_state(single)) == {"a"}

    sharded = tmp_path / "sharded"
    sharded.mkdir()
    save_file({"a": torch.ones(2)}, str(sharded / "model-00001-of-00002.safetensors"))
    save_file({"b": torch.zeros(2)}, str(sharded / "model-00002-of-00002.safetensors"))
    (sharded / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    assert set(checkpoint_io.load_state(sharded)) == {"a", "b"}

    with pytest.raises(FileNotFoundError):
        checkpoint_io.weight_files(tmp_path)


def test_lmms_plugin_registers_dpvr_hf():
    pytest.importorskip("lmms_eval")
    code = ("from lmms_eval.models import get_model; c = get_model('dpvr_hf'); "
            "from lmms_eval.models.simple.llava_hf import LlavaHf; "
            "assert issubclass(c, LlavaHf); print(c.__module__)")
    env = dict(os.environ, LMMS_EVAL_PLUGINS="dpvr.lmms_plugin",
               PYTHONPATH=os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if r.returncode != 0 and "No module named" in r.stderr and "dpvr" not in r.stderr.split("No module named")[-1]:
        pytest.skip(f"lmms_eval dependencies missing: {r.stderr.strip().splitlines()[-1]}")
    assert r.returncode == 0, r.stderr
    assert "dpvr.lmms_plugin.models.dpvr_hf" in r.stdout

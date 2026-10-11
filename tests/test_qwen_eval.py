"""Second-backbone evaluation chain: handler choice and the dpvr_qwen25vl plugin."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "revision"))

import eval_lmms  # noqa: E402


def test_pick_backbone(tmp_path):
    pick = eval_lmms.pick_backbone
    assert pick("auto", Path("/m/llava-1.5-7b-hf"), "none") == "llava"
    assert pick("auto", Path("/m/Qwen2.5-VL-7B-Instruct"), "none") == "qwen2_5_vl"
    assert pick("auto", Path("/m/qwen2_5_vl_7b"), "none") == "qwen2_5_vl"
    assert pick("llava", Path("/m/Qwen2.5-VL-7B-Instruct"), "none") == "llava"
    ck = tmp_path / "run"
    ck.mkdir()
    (ck / "training_config.json").write_text(json.dumps(
        {"method": {"name": "token_diversion_x3_fusion_qwen25vl"}}))
    assert pick("auto", Path("/m/renamed"), str(ck)) == "qwen2_5_vl"


def _qwen_handler_module():
    pytest.importorskip("lmms_eval")
    try:
        from lmms_eval.models.simple import qwen2_5_vl
    except ImportError as e:
        pytest.skip(f"lmms_eval qwen2_5_vl dependencies missing: {e}")
    return qwen2_5_vl


def test_lmms_plugin_registers_dpvr_qwen25vl():
    _qwen_handler_module()
    code = ("from lmms_eval.models import get_model; c = get_model('dpvr_qwen25vl'); "
            "from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL; "
            "assert issubclass(c, Qwen2_5_VL); print(c.__module__)")
    env = dict(os.environ, LMMS_EVAL_PLUGINS="dpvr.lmms_plugin",
               PYTHONPATH=os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "dpvr.lmms_plugin.models.dpvr_qwen25vl" in r.stdout


def test_dpvr_qwen25vl_swaps_in_checkpoint(monkeypatch):
    mod = _qwen_handler_module()
    from dpvr.lmms_plugin.models import dpvr_qwen25vl
    from dpvr.models import checkpoint_io

    seen = {}

    def fake_init(self, **kw):
        seen.update(kw)
        self.use_cache = kw["use_cache"]
        self._model = torch.nn.Linear(2, 2)

    monkeypatch.setattr(mod.Qwen2_5_VL, "__init__", fake_init)
    h = dpvr_qwen25vl.DPVRQwen25VL(pretrained="x", checkpoint="none", max_pixels=100)
    assert seen == {"pretrained": "x", "use_cache": False, "max_pixels": 100}
    assert isinstance(h._model, torch.nn.Linear)

    trained = torch.nn.Linear(3, 3)

    class _Report:
        missing: list = []

        def summary(self):
            return "ok"

    monkeypatch.setattr(checkpoint_io, "load_trained", lambda ck: (trained, _Report()))
    h = dpvr_qwen25vl.DPVRQwen25VL(pretrained="x", checkpoint="/ckpt", use_cache="False")
    assert h._model is trained and h.use_cache is False and not h._model.training

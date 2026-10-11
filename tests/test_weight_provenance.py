"""check_weight_provenance: tells Vicuna / stage-1 starts from LLaVA-HF ones."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "revision"))

import check_weight_provenance as prov  # noqa: E402

N_LAYERS, SPLIT = 4, 2
PARAMS = {"self_attn.q_proj.weight": (8, 8), "mlp.down_proj.weight": (8, 16)}


def _lm(seed):
    g = torch.Generator().manual_seed(seed)
    w = {f"layers.{i}.{p}": torch.randn(*s, generator=g).half()
         for i in range(N_LAYERS) for p, s in PARAMS.items()}
    w["norm.weight"] = torch.randn(8, generator=g).half()
    w["layers.0.input_layernorm.weight"] = torch.randn(8, generator=g).half()
    return w


def _proj(seed):
    g = torch.Generator().manual_seed(seed)
    return {"linear_1.weight": torch.randn(8, 4, generator=g).half(),
            "linear_2.weight": torch.randn(8, 8, generator=g).half()}


def _refs(tmp_path):
    llava_lm, vicuna_lm = _lm(1), _lm(2)
    llava_proj, stage1_proj = _proj(3), _proj(4)
    hf = tmp_path / "llava-hf"
    hf.mkdir()
    sd = {f"language_model.model.{k}": v for k, v in llava_lm.items()}
    sd.update({f"multi_modal_projector.{k}": v for k, v in llava_proj.items()})
    sd["vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"] = torch.zeros(8, 8).half()
    save_file(sd, str(hf / "model.safetensors"))
    vic = tmp_path / "vicuna"
    vic.mkdir()
    torch.save({f"model.{k}": v for k, v in vicuna_lm.items()}, vic / "pytorch_model.bin")
    projf = tmp_path / "mm_projector.bin"
    torch.save({"model.mm_projector.0.weight": stage1_proj["linear_1.weight"],
                "model.mm_projector.2.weight": stage1_proj["linear_2.weight"]}, projf)
    refs = {"llava_hf": prov.TensorDir(hf), "vicuna": prov.TensorDir(vic),
            "stage1_projector": prov.TensorDir(projf)}
    return refs, (llava_lm, vicuna_lm, llava_proj, stage1_proj)


def _ckpt(tmp_path, lm, proj):
    d = tmp_path / "ckpt"
    d.mkdir()
    sd = {f"llava_model.model.language_model.{k}": v.to(torch.bfloat16) for k, v in lm.items()}
    sd.update({f"llava_model.model.multi_modal_projector.{k}": v.to(torch.bfloat16)
               for k, v in proj.items()})
    sd["llava_model.model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"] = \
        torch.ones(8, 8, dtype=torch.bfloat16)
    sd["single_transformer.0.self_attn.q_proj.weight"] = torch.ones(8, 8, dtype=torch.bfloat16)
    save_file(sd, str(d / "model.safetensors"))
    return prov.TensorDir(d)


@pytest.mark.parametrize("lm_src,proj_src,want", [
    ("vicuna", "stage1", {"language_model": "vicuna", "projector": "stage1_projector"}),
    ("llava", "llava", {"language_model": "llava_hf", "projector": "llava_hf"}),
    ("vicuna", "llava", {"language_model": "vicuna", "projector": "llava_hf"}),
])
def test_verdict(tmp_path, lm_src, proj_src, want):
    refs, (llava_lm, vicuna_lm, llava_proj, stage1_proj) = _refs(tmp_path)
    ckpt = _ckpt(tmp_path, vicuna_lm if lm_src == "vicuna" else llava_lm,
                 stage1_proj if proj_src == "stage1" else llava_proj)
    report = prov.compare(ckpt, refs, SPLIT, N_LAYERS)
    assert report["verdict"] == want
    probes = {r["probe"] for r in report["rows"]}
    assert {"layer0.q_proj", "layer1.q_proj", "layer2.q_proj", "layer3.q_proj", "final_norm"} <= probes


def test_trained_weights_are_neither(tmp_path):
    refs, _ = _refs(tmp_path)
    ckpt = _ckpt(tmp_path, _lm(9), _proj(9))
    v = prov.compare(ckpt, refs, SPLIT, N_LAYERS)["verdict"]
    assert v["language_model"].startswith("neither")
    assert v["projector"].startswith("neither")


def test_missing_reference_is_unknown(tmp_path):
    refs, (_, vicuna_lm, _, stage1_proj) = _refs(tmp_path)
    ckpt = _ckpt(tmp_path, vicuna_lm, stage1_proj)
    refs["vicuna"] = None
    v = prov.compare(ckpt, refs, SPLIT, N_LAYERS)["verdict"]
    assert v["language_model"].startswith("unknown")
    assert v["projector"] == "stage1_projector"

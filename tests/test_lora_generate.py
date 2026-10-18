"""LoRA 基线：generate 必须能拿到 logits_to_keep。

现象（2026-10-18，E12/lora7b）：lmms-eval 生成阶段报
`Tensors must have same number of dimensions: got 2 and 3`。
原因：GenerationMixin 用 `inspect.signature(forward)` 判断是否支持
logits_to_keep（`_supports_logits_to_keep()`），包装类 LoRALlava 的 forward
没有显式声明它，于是 generate 不设默认值 1，而基类
prepare_inputs_for_generation 又传 logits_to_keep=None，最终
`hidden_states[:, None, :]` 让 logits 多一维。
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.models.lora_baseline import LoRALlava  # noqa: E402


def test_lora_forward_declares_logits_to_keep():
    params = inspect.signature(LoRALlava.forward).parameters
    assert "logits_to_keep" in params
    assert params["logits_to_keep"].default == 0


def test_forward_passes_logits_to_keep_to_inner_model():
    # 不建模型：只检查透传逻辑，用假的内层模型抓 kwargs。
    import torch

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = {}

        def forward(self, **kw):
            self.seen = kw
            return kw

    obj = object.__new__(LoRALlava)  # 跳过 __init__（会加载 7B 权重）
    torch.nn.Module.__init__(obj)
    obj.llava_model = Inner()
    sentinel = torch.tensor([0])
    out = obj.forward(input_ids=torch.tensor([1]), logits_to_keep=sentinel)
    assert out["logits_to_keep"] is sentinel

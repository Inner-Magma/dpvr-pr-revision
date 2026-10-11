"""``--model dpvr_qwen25vl``: lmms-eval's own ``qwen2_5_vl`` handler with a DPVR model inside.

The second-backbone counterpart of ``dpvr_hf`` (C8 / E6). It keeps the stock
handler's prompt (system prompt, image before text), its image budget
(``min_pixels`` / ``max_pixels`` defaults), ``qwen_vl_utils`` preprocessing and
generation call, and only swaps the network for the trained checkpoint.

    --model dpvr_qwen25vl \
    --model_args pretrained=/models/Qwen2.5-VL-7B-Instruct,checkpoint=/ckpt/run,use_cache=False

``checkpoint=none`` evaluates the vanilla model through the same handler, so the
vanilla and DPVR rows of the cross-backbone table share one evaluation chain.
"""
from __future__ import annotations

import torch
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL
from loguru import logger as eval_logger


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y")


class DPVRQwen25VL(Qwen2_5_VL):
    def __init__(self, pretrained: str, checkpoint: str = "none", use_cache=False, **kwargs) -> None:
        super().__init__(pretrained=pretrained, use_cache=_as_bool(use_cache), **kwargs)
        self.checkpoint = str(checkpoint)
        if self.checkpoint.lower() in ("", "none", "vanilla"):
            eval_logger.info(f"[dpvr_qwen25vl] vanilla model {pretrained}, use_cache={self.use_cache}")
            return

        from dpvr.models.checkpoint_io import load_trained

        device = next(self._model.parameters()).device
        self._model = None
        torch.cuda.empty_cache()
        model, report = load_trained(self.checkpoint)
        eval_logger.info(f"[dpvr_qwen25vl] loaded {self.checkpoint}: {report.summary()}")
        if report.missing:
            eval_logger.warning(f"[dpvr_qwen25vl] missing keys (first 10): {report.missing[:10]}")
        self._model = model.to(device, dtype=torch.bfloat16).eval()

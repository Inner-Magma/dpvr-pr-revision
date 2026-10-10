"""``--model dpvr_hf``: lmms-eval's own ``llava_hf`` handler with a DPVR model inside.

The paper's numbers come from the ``llava_hf`` handler with ``use_cache=False``.
This class keeps that handler's prompt building, chat template, image
preprocessing and generation call, and only swaps the network: it loads the
base LLaVA-HF model given by ``pretrained`` (which supplies the processor and
tokenizer), then replaces it with the trained checkpoint.

    --model dpvr_hf \
    --model_args pretrained=/models/llava-1.5-7b-hf,checkpoint=/ckpt/run,use_cache=False

``checkpoint=none`` evaluates the vanilla base model through the same code
path, so vanilla and DPVR rows share one evaluation chain.
"""
from __future__ import annotations

import torch
from lmms_eval.models.simple.llava_hf import LlavaHf
from loguru import logger as eval_logger


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y")


class DPVRHf(LlavaHf):
    def __init__(
        self,
        pretrained: str,
        checkpoint: str = "none",
        use_cache=False,
        dtype: str = "bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(pretrained=pretrained, dtype=dtype, use_cache=_as_bool(use_cache), **kwargs)
        self.checkpoint = str(checkpoint)
        if self.checkpoint.lower() in ("", "none", "vanilla"):
            eval_logger.info(f"[dpvr_hf] vanilla model {pretrained}, use_cache={self.use_cache}")
            return

        from dpvr.models.checkpoint_io import load_trained

        base_dtype = self._model.dtype
        self._model = None
        torch.cuda.empty_cache()
        model, report = load_trained(self.checkpoint)
        eval_logger.info(f"[dpvr_hf] loaded {self.checkpoint}: {report.summary()}")
        if report.missing:
            eval_logger.warning(f"[dpvr_hf] missing keys (first 10): {report.missing[:10]}")
        model = model.to(self._device, dtype=base_dtype)
        model.eval()
        self._model = model

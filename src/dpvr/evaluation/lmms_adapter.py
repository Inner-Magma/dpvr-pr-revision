"""lmms_eval model adapter for DPVR methods.

Registers a custom model class as ``--model dpvr`` so that lmms_eval can be
invoked end-to-end via:

    python -m lmms_eval \\
        --model dpvr \\
        --model_args checkpoint=<path>,method=token_diversion \\
        --tasks pope,mme \\
        --output_path results/eval/...

Requirements:
    - lmms-eval must be installed (pip install lmms-eval, or sideload).
    - Run with HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 to avoid outbound traffic.

Supported methods:
    token_diversion, td_plus_fastv  — TokenDiversionLlava.from_trained(ckpt)
    lora                            — LoRALlava loaded via state_dict
    fastv                           — FastVLlava loaded via state_dict
    full_ft                         — FullFTLlava loaded via state_dict

Design (lmms-eval 0.5.0):
    DPVREvalModel subclasses lmms_eval.api.model.lmms directly and implements
    generate_until / loglikelihood using the same logit-comparison method as
    native_runner._yn_logit_compare.  This avoids the removed Llava base class.

NOTE on lmms-eval 0.4.x vs 0.5.0:
    Prior to 0.5.0, lmms_eval.models.llava.Llava existed and provided
    generate_until / loglikelihood.  In 0.5.0 the module was restructured;
    Llava moved to lmms_eval.models.simple.llava but requires the separate
    'llava' (LLaVA-1.5 repo) package.  We implement inference ourselves.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple


def _load_dpvr_model(checkpoint: str, method: str):
    """Load a DPVR model from a training checkpoint directory."""
    import sys, json, torch
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(REPO_ROOT / "src"))

    from dpvr.utils import load_config, load_paths, resolve_paths

    ckpt = Path(checkpoint)

    if method in ("token_diversion", "td_plus_fastv"):
        from dpvr.models.token_diversion import TokenDiversionLlava
        from dpvr.models.td_fastv import TDFastVLlava
        cls = TokenDiversionLlava if method == "token_diversion" else TDFastVLlava
        return cls.from_trained(checkpoint)

    # For methods without from_trained: build from config + load state_dict.
    cfg_file = ckpt / "training_config.json"
    if not cfg_file.exists():
        yaml_path = REPO_ROOT / f"configs/method/{method}.yaml"
        cfg = load_config(str(yaml_path))
        paths = load_paths()
        cfg = resolve_paths(cfg, paths)
    else:
        cfg = json.loads(cfg_file.read_text())

    if method == "lora":
        from dpvr.models.lora_baseline import LoRALlava
        model = LoRALlava.from_config(cfg)
    elif method == "fastv":
        from dpvr.models.fastv_baseline import FastVLlava
        model = FastVLlava.from_config(cfg)
    elif method == "full_ft":
        from dpvr.models.full_ft_baseline import FullFTLlava
        model = FullFTLlava.from_config(cfg)
    else:
        raise ValueError(f"Unknown DPVR method: {method!r}")

    state_files = list(ckpt.glob("pytorch_model*.bin")) + list(ckpt.glob("model*.safetensors"))
    if state_files:
        if state_files[0].suffix == ".bin":
            state = torch.load(state_files[0], map_location="cpu")
        else:
            from safetensors.torch import load_file
            state = load_file(str(state_files[0]))
        model.load_state_dict(state, strict=False)

    return model


try:
    from lmms_eval.api.model import lmms
    from lmms_eval.api.registry import register_model
    from transformers import AutoProcessor

    @register_model("dpvr")
    class DPVREvalModel(lmms):
        """lmms_eval 0.5.0 wrapper for DPVR checkpoints.

        model_args: checkpoint=<path>,method=<name>

        Implements generate_until and loglikelihood using the same
        logit-comparison approach as native_runner._yn_logit_compare.
        Inherits lmms base class directly (lmms_eval 0.5.0 no longer has
        a separate Llava base class that we can safely subclass).
        """

        def __init__(
            self,
            checkpoint: str,
            method: str = "token_diversion",
            device: str = "cuda",
            dtype: Optional[str] = "bfloat16",
            batch_size: int = 1,
            **kwargs,
        ):
            super().__init__()

            self._method = method
            self._checkpoint = checkpoint
            self._batch_size = batch_size

            import torch
            _dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

            dpvr_model = _load_dpvr_model(checkpoint, method)
            dpvr_model = dpvr_model.to(device, dtype=_dtype)
            dpvr_model.eval()

            self._model = dpvr_model
            self._device = torch.device(device)
            self._torch = torch

            proc_path = Path(checkpoint)
            if (proc_path / "processor_config.json").exists():
                self._processor = AutoProcessor.from_pretrained(str(proc_path))
            else:
                from dpvr.utils import load_paths
                paths = load_paths()
                base_path = paths.model_root / "llava-1.5-7b-hf"
                self._processor = AutoProcessor.from_pretrained(str(base_path))

            self._tokenizer = self._processor.tokenizer
            tok = self._tokenizer
            self._yes_id = tok.encode("Yes", add_special_tokens=False)[0]
            self._no_id = tok.encode("No", add_special_tokens=False)[0]

        # ------------------------------------------------------------------ #
        # Required lmms interface
        # ------------------------------------------------------------------ #

        def generate_until_multi_round(self, requests) -> List[str]:
            return self.generate_until(requests)

        def generate_until(self, requests) -> List[str]:
            """Greedy generation for open-ended VQA tasks."""
            from PIL import Image
            results = []
            for req in requests:
                ctx, gen_kwargs, visual_list = req.args[:3]
                max_new = gen_kwargs.get("max_new_tokens", 32)

                images = []
                if visual_list:
                    for v in visual_list:
                        if isinstance(v, Image.Image):
                            images.append(v)

                if images:
                    enc = self._processor(
                        images=images[0], text=ctx, return_tensors="pt"
                    ).to(self._device)
                else:
                    enc = self._processor(text=ctx, return_tensors="pt").to(self._device)

                with self._torch.no_grad():
                    ids = self._model.generate(**enc, max_new_tokens=max_new,
                                               do_sample=False)
                new_ids = ids[0][enc["input_ids"].shape[1]:]
                results.append(self._tokenizer.decode(new_ids, skip_special_tokens=True))
            return results

        def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
            """Log-likelihood for classification tasks (POPE, MME).

            For yes/no tasks, compare logit(Yes) vs logit(No) at the
            ASSISTANT position rather than computing full sequence log-prob.
            This matches native_runner._yn_logit_compare exactly.
            """
            from PIL import Image
            results = []
            for req in requests:
                ctx, continuation = req.args[:2]
                visual_list = req.args[2] if len(req.args) > 2 else None

                images = []
                if visual_list:
                    for v in visual_list:
                        if isinstance(v, Image.Image):
                            images.append(v)

                if images:
                    enc = self._processor(
                        images=images[0], text=ctx, return_tensors="pt"
                    ).to(self._device)
                else:
                    enc = self._processor(text=ctx, return_tensors="pt").to(self._device)

                with self._torch.no_grad():
                    out = self._model(**enc)

                logits_last = out.logits[0, -1]
                score_yes = logits_last[self._yes_id].item()
                score_no = logits_last[self._no_id].item()

                cont_norm = continuation.strip().lower()
                if cont_norm.startswith("yes"):
                    log_prob = score_yes
                    is_greedy = score_yes >= score_no
                else:
                    log_prob = score_no
                    is_greedy = score_no >= score_yes

                results.append((log_prob, is_greedy))
            return results

        # ------------------------------------------------------------------ #
        # Standard properties
        # ------------------------------------------------------------------ #

        @property
        def model(self):
            return self._model

        @property
        def tokenizer(self):
            return self._tokenizer

        @property
        def device(self):
            return self._device

        @property
        def batch_size(self):
            return self._batch_size

        @property
        def rank(self):
            return 0

        @property
        def world_size(self):
            return 1

except ImportError as _e:
    # lmms-eval not installed; adapter code ready for when it is sideloaded.
    DPVREvalModel = None

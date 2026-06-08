"""Extract per-layer hidden states from a model for a batch of inputs."""
from __future__ import annotations

from typing import Optional

import torch


@torch.no_grad()
def extract_hidden_states(
    model,
    input_ids: torch.LongTensor,
    pixel_values: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layers: Optional[list[int]] = None,
) -> dict[int, torch.Tensor]:
    """Return {layer_idx: hidden_state (B, S, H)} for the requested layers.

    Uses forward hooks so the model forward does not need modification.
    """
    hooks = []
    captured: dict[int, torch.Tensor] = {}

    def _make_hook(idx: int):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, (tuple, list)) else out
            captured[idx] = hs.detach().cpu().float()
        return hook

    llm_layers = None
    # TokenDiversionLlava wraps a LlavaForConditionalGeneration under .llava_model
    if hasattr(model, "llava_model"):
        llm_layers = model.llava_model.language_model.layers
    # transformers LlavaForConditionalGeneration exposes .language_model
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        llm_layers = model.language_model.layers
    # transformers LlavaForConditionalGeneration .model.language_model.layers
    elif hasattr(model, "model") and hasattr(getattr(model.model, "language_model", None), "layers"):
        llm_layers = model.model.language_model.layers
    # Plain LLaMA-style model
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        llm_layers = model.model.layers

    if llm_layers is None:
        raise ValueError("Cannot locate LLaMA layers in the provided model.")

    target = layers if layers is not None else list(range(len(llm_layers)))
    for idx in target:
        h = llm_layers[idx].register_forward_hook(_make_hook(idx))
        hooks.append(h)

    try:
        model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
    finally:
        for h in hooks:
            h.remove()

    return captured

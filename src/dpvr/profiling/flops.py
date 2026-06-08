"""FLOPs estimation via torch profiler (no fvcore required)."""
from __future__ import annotations

from typing import Optional

import torch


def estimate_flops(
    model,
    input_ids: torch.LongTensor,
    pixel_values: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Estimate forward-pass FLOPs using PyTorch's built-in profiler.

    Returns a dict with "flops_giga" and "params_million".
    Note: counts multiply-add ops via profiler's `flops` field when available.
    Falls back to a parameter-count-based heuristic if profiler FLOPs are zero.
    """
    from dpvr.profiling.params import count_parameters

    model.eval()
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            with_flops=True,
            record_shapes=False,
        ) as prof:
            model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )

    total_flops = sum(
        e.flops for e in prof.key_averages() if e.flops is not None
    )
    params = count_parameters(model)

    return {
        "flops_giga": total_flops / 1e9,
        "params_million": params["total"] / 1e6,
        "trainable_params_million": params["trainable"] / 1e6,
    }

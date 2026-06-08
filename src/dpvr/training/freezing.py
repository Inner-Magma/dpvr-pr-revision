"""Shared freeze strategies used by multiple methods."""
from __future__ import annotations

import torch.nn as nn


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(True)


def apply_llava_freeze_strategy(
    llava_model: nn.Module,
    strategy: str,
    split_layer: int = 0,
    trainable_modules: list[nn.Module] | None = None,
) -> None:
    """Freeze the LLaVA backbone according to strategy, then unfreeze trainable_modules.

    Strategies:
      all_but_single     — freeze entire llava_model (default for TokenDiversion)
      full_ft            — leave everything unfrozen
      shallow_unfrozen   — freeze layers split_layer..end, unfreeze 0..split_layer-1
    """
    if strategy == "full_ft":
        unfreeze_all(llava_model)
    elif strategy == "all_but_single":
        freeze_all(llava_model)
    elif strategy == "shallow_unfrozen":
        freeze_all(llava_model)
        lm = getattr(llava_model, "language_model", None)
        if lm is None:
            lm = getattr(getattr(llava_model, "model", None), "language_model", None)
        if lm is not None and hasattr(lm, "layers"):
            for i in range(split_layer):
                for p in lm.layers[i].parameters():
                    p.requires_grad_(True)
    else:
        raise ValueError(
            f"Unknown freeze_strategy '{strategy}'. "
            "Choose: all_but_single | full_ft | shallow_unfrozen"
        )

    if trainable_modules:
        for mod in trainable_modules:
            for p in mod.parameters():
                p.requires_grad_(True)

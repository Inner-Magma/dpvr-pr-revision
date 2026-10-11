"""Side-branch variants for DPVR-LF (C9, R2.5).

The default branch is ``vision_depth`` randomly initialised LlamaDecoderLayers.
Alternatives, selected with ``method.branch_type`` / ``method.branch_init``:

- ``transformer`` (default): full decoder layer(s); ``branch_init: layer``
  copies the weights of decoder layer s+i instead of a random init.
- ``mlp``: attention-free residual block ``h + MLP(RMSNorm(h))`` with the
  decoder's SwiGLU MLP (about two thirds of a layer's parameters).
- ``lora``: a frozen copy of decoder layer s+i with LoRA adapters (rank
  ``lora_rank``) on q/k/v/o and the MLP projections; only the adapters train.

Every variant takes the same call as a LlamaDecoderLayer, so the DPVR-LF
forward is unchanged.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaMLP, LlamaRMSNorm

BRANCH_TYPES = ("transformer", "mlp", "lora")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class MLPBranchLayer(nn.Module):
    """Pre-norm residual SwiGLU block with the decoder layer's call signature."""

    def __init__(self, config):
        super().__init__()
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class LoRALinear(nn.Module):
    """y = base(x) + (alpha / r) * B(A(x)); base frozen, B zero-initialised."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / rank
        self.lora_A.to(base.weight.dtype)
        self.lora_B.to(base.weight.dtype)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


def _add_lora(layer: nn.Module, rank: int, alpha: float) -> nn.Module:
    for parent in (layer.self_attn, layer.mlp):
        for name in LORA_TARGETS:
            if hasattr(parent, name):
                setattr(parent, name, LoRALinear(getattr(parent, name), rank, alpha))
    return layer


def build_side_branch(
    llama_cfg,
    split_layer: int,
    depth: int = 1,
    branch_type: str = "transformer",
    branch_init: str = "random",
    source_layers=None,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.ModuleList:
    if branch_type not in BRANCH_TYPES:
        raise ValueError(f"branch_type must be one of {BRANCH_TYPES}, got {branch_type!r}")
    if branch_init not in ("random", "layer"):
        raise ValueError(f"branch_init must be 'random' or 'layer', got {branch_init!r}")
    needs_source = branch_type == "lora" or branch_init == "layer"
    if needs_source and source_layers is None:
        raise ValueError(f"{branch_type}/{branch_init} needs the decoder layers to copy from")
    if needs_source and split_layer + depth > len(source_layers):
        raise ValueError("not enough decoder layers after split_layer to initialise the branch")

    layers = []
    for i in range(depth):
        idx = split_layer + i
        if branch_type == "lora":
            layer = _add_lora(copy.deepcopy(source_layers[idx]), lora_rank, lora_alpha)
        elif branch_type == "mlp":
            layer = MLPBranchLayer(llama_cfg)
            if branch_init == "layer":
                src = source_layers[idx]
                layer.post_attention_layernorm.load_state_dict(src.post_attention_layernorm.state_dict())
                layer.mlp.load_state_dict(src.mlp.state_dict())
        else:
            layer = LlamaDecoderLayer(llama_cfg, layer_idx=idx)
            if branch_init == "layer":
                layer.load_state_dict(source_layers[idx].state_dict())
        layers.append(layer.to(dtype))
    return nn.ModuleList(layers)


def set_branch_trainable(branch: nn.ModuleList, branch_type: str) -> None:
    """All branch weights train, except the frozen base of a LoRA branch."""
    for p in branch.parameters():
        p.requires_grad_(branch_type != "lora")
    if branch_type == "lora":
        for m in branch.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.weight.requires_grad_(True)
                m.lora_B.weight.requires_grad_(True)

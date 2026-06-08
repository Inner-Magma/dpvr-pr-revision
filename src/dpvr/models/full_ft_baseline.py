"""Full fine-tuning baseline: all LLaVA parameters trainable."""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoConfig, PreTrainedModel
from transformers.generation import GenerationMixin

from .modeling_llava import LlavaForConditionalGeneration
from .registry import register_method


@register_method("full_ft")
class FullFTLlava(PreTrainedModel, GenerationMixin):
    """LLaVA-1.5 with all parameters unfrozen (full fine-tuning baseline)."""

    main_input_name = "input_ids"

    def __init__(self, base_model_path: str):
        config = AutoConfig.from_pretrained(base_model_path)
        super().__init__(config)
        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        # Unfreeze everything.
        for p in self.llava_model.parameters():
            p.requires_grad_(True)

    @classmethod
    def from_config(cls, cfg: dict) -> "FullFTLlava":
        return cls(base_model_path=cfg["model"]["base_model_path"])

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.llava_model, "gradient_checkpointing_enable"):
            self.llava_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def print_trainable_parameters(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} | all params: {total:,} | "
              f"trainable%: {100*trainable/max(total,1):.6f}")

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      pixel_values=None, attention_mask=None,
                                      cache_position=None, **kwargs):
        return self.llava_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values,
            pixel_values=pixel_values, attention_mask=attention_mask,
            cache_position=cache_position, **kwargs,
        )

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None,
                labels=None, **kwargs):
        return self.llava_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

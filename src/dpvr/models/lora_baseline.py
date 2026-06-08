"""LoRA baseline: PEFT LoRA adapters on the LLaVA language model."""
from __future__ import annotations

import torch
from transformers import AutoConfig, PreTrainedModel
from transformers.generation import GenerationMixin

from .modeling_llava import LlavaForConditionalGeneration
from .registry import register_method


@register_method("lora")
class LoRALlava(PreTrainedModel, GenerationMixin):
    """LLaVA-1.5 with LoRA adapters on the LLM (q/k/v/o projections).

    Config knobs:
        method.lora_rank       (default 16)
        method.lora_alpha      (default 32)
        method.lora_dropout    (default 0.05)
        method.lora_target_modules  (default q_proj,k_proj,v_proj,o_proj)
    """

    main_input_name = "input_ids"

    def __init__(self, base_model_path: str, lora_rank: int = 16,
                 lora_alpha: int = 32, lora_dropout: float = 0.05,
                 target_modules: list[str] | None = None):
        from peft import LoraConfig, get_peft_model

        config = AutoConfig.from_pretrained(base_model_path)
        super().__init__(config)

        base = LlavaForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        # Freeze vision tower and projector; LoRA targets only the LLM.
        for p in base.model.vision_tower.parameters():
            p.requires_grad_(False)
        for p in base.model.multi_modal_projector.parameters():
            p.requires_grad_(False)

        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llava_model = get_peft_model(base, lora_cfg)

    @classmethod
    def from_config(cls, cfg: dict) -> "LoRALlava":
        m = cfg["method"]
        return cls(
            base_model_path=cfg["model"]["base_model_path"],
            lora_rank=m.get("lora_rank", 16),
            lora_alpha=m.get("lora_alpha", 32),
            lora_dropout=m.get("lora_dropout", 0.05),
            target_modules=m.get("lora_target_modules", None),
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.llava_model, "gradient_checkpointing_enable"):
            self.llava_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        # PEFT requires this so frozen embedding outputs carry gradients through
        # the checkpoint recomputation boundary.
        if hasattr(self.llava_model, "enable_input_require_grads"):
            self.llava_model.enable_input_require_grads()

    def print_trainable_parameters(self) -> None:
        self.llava_model.print_trainable_parameters()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      pixel_values=None, attention_mask=None,
                                      cache_position=None, **kwargs):
        base = self.llava_model.base_model.model
        return base.prepare_inputs_for_generation(
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

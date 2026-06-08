"""FastV baseline: prune low-attention visual tokens after layer K.

Reference: "An Image is Worth 1/2 Tokens after Layer 2: Plug-and-Play
Acceleration for VLLM Inference" (ECCV 2024).

At fastv_layer we compute per-token attention scores (averaged over heads)
and keep only the top-R% visual tokens.  Text tokens are never pruned.
Pruning is applied both at train and inference time for a fair comparison.

Config knobs:
    method.fastv_layer     layer index where pruning happens (default 2)
    method.fastv_ratio     fraction of visual tokens to KEEP (default 0.5)
    method.freeze_strategy full_ft | all_but_lm | ... (default full_ft)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling_llama import LlamaModel
from .modeling_llava import LlavaForConditionalGeneration
from .registry import register_method


@register_method("fastv")
class FastVLlava(PreTrainedModel, GenerationMixin):
    """LLaVA-1.5 with FastV visual-token pruning."""

    main_input_name = "input_ids"

    def __init__(self, base_model_path: str, fastv_layer: int = 2,
                 fastv_ratio: float = 0.5, freeze_strategy: str = "full_ft"):
        from ..training.freezing import apply_llava_freeze_strategy

        config = AutoConfig.from_pretrained(base_model_path)
        super().__init__(config)
        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        apply_llava_freeze_strategy(self.llava_model, strategy=freeze_strategy)
        self.fastv_layer = fastv_layer
        self.fastv_ratio = fastv_ratio

    @classmethod
    def from_config(cls, cfg: dict) -> "FastVLlava":
        m = cfg["method"]
        return cls(
            base_model_path=cfg["model"]["base_model_path"],
            fastv_layer=m.get("fastv_layer", 2),
            fastv_ratio=m.get("fastv_ratio", 0.5),
            freeze_strategy=m.get("freeze_strategy", "full_ft"),
        )

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
        inputs = self.llava_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values,
            pixel_values=pixel_values, attention_mask=attention_mask,
            cache_position=cache_position, **kwargs,
        )
        inputs["_fastv_pixel_values"] = pixel_values
        return inputs

    # ---------------------------------------------------------------- forward
    def forward(self, input_ids=None, pixel_values=None, attention_mask=None,
                labels=None, use_cache: bool = False, past_key_values=None,
                cache_position=None, **kwargs):
        if attention_mask is not None:
            attention_mask_bool = attention_mask == 1
        else:
            attention_mask_bool = None

        inputs_embeds, image_token_mask = self.llava_model.model.prepare_multimodal_input(
            input_ids, pixel_values
        )

        language_model = self.llava_model.language_model

        if attention_mask_bool is not None and cache_position is None:
            position_ids = attention_mask_bool.long().cumsum(-1) - 1
            position_ids.masked_fill_(~attention_mask_bool, 0)
        else:
            position_ids = None

        (hidden_states, causal_mask, position_ids, past_key_values,
         cache_position, position_embeddings) = language_model.prepare_input_mine(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_bool,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
        )

        has_image = image_token_mask is not None and image_token_mask.any()
        # 2D boolean mask (B, seq_len) for image positions — used after pruning
        img_mask_2d = image_token_mask.any(dim=-1) if has_image else None

        keep_idx_used = None  # tracks original positions kept after FastV pruning
        for i, layer in enumerate(language_model.layers):
            # At fastv_layer, prune low-attention visual tokens.
            if i == self.fastv_layer and has_image and img_mask_2d is not None:
                hidden_states, causal_mask, position_ids, position_embeddings, \
                    cache_position, img_mask_2d, keep_idx_used = self._prune_visual_tokens(
                        hidden_states, causal_mask, position_ids,
                        position_embeddings, cache_position, img_mask_2d,
                        language_model,
                    )
                has_image = img_mask_2d.any()

            out = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                output_attentions=False,
                use_cache=use_cache,
            )
            hidden_states = out[0] if isinstance(out, (tuple, list)) else out

        final_hidden_states = language_model.norm(hidden_states)
        logits = self.llava_model.lm_head(final_hidden_states)

        loss = None
        if labels is not None:
            if keep_idx_used is not None:
                # After pruning: logit[b, i] was computed from original position
                # keep_idx_used[b, i], so it predicts original position
                # keep_idx_used[b, i] + 1.  Gather the matching labels.
                gather_idx = (keep_idx_used + 1).clamp(max=labels.shape[1] - 1)
                shift_logits = logits.float().contiguous()           # (B, S_pruned, V)
                shift_labels = labels.gather(1, gather_idx)          # (B, S_pruned)
            else:
                shift_logits = logits[..., :-1, :].float().contiguous()
                shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, self.llava_model.config.vocab_size),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None, attentions=None,
        )

    def _prune_visual_tokens(self, hidden_states, causal_mask, position_ids,
                              position_embeddings, cache_position, img_mask_2d,
                              language_model):
        """Prune visual tokens with lowest mean attention score at this layer."""
        B, S, H = hidden_states.shape
        # Compute Q·K^T attention scores using the first layer's attention module
        # as a proxy (lightweight, no grad needed for pruning decision).
        with torch.no_grad():
            # Use rotary-embedded Q and K from a single attention head as the score.
            attn = language_model.layers[self.fastv_layer].self_attn
            cos, sin = position_embeddings
            q = attn.q_proj(hidden_states)
            k = attn.k_proj(hidden_states)
            head_dim = attn.head_dim
            q = q.view(B, S, -1, head_dim).transpose(1, 2)   # (B, H, S, d)
            k = k.view(B, S, -1, head_dim).transpose(1, 2)
            # Average attention score per token (mean over query positions).
            scores = (q @ k.transpose(-1, -2)).mean(dim=(1, 2))  # (B, S)

        # Only consider scores for image token positions.
        img_scores = scores.masked_fill(~img_mask_2d, float("-inf"))
        n_img = img_mask_2d.sum(dim=1).min().item()
        k_keep = max(1, int(n_img * self.fastv_ratio))

        # Build a keep mask: text tokens always kept, top-k image tokens kept.
        topk_idx = img_scores.topk(k_keep, dim=1).indices  # (B, k_keep)
        keep_mask = ~img_mask_2d.clone()                    # text positions = True
        keep_mask.scatter_(1, topk_idx, True)               # add top-k image

        # Filter hidden_states, masks, and position info.
        # NOTE: different batch items may have slightly different keep counts
        # due to rounding; we use the minimum to keep a rectangular tensor.
        min_keep = keep_mask.sum(dim=1).min().item()
        # Re-sort to preserve positional order.
        sorted_keep = []
        for b in range(B):
            idx = keep_mask[b].nonzero(as_tuple=True)[0][:min_keep]
            sorted_keep.append(idx)
        keep_idx = torch.stack(sorted_keep)                  # (B, min_keep)

        hidden_states = hidden_states.gather(
            1, keep_idx.unsqueeze(-1).expand(-1, -1, H))

        # Rebuild causal mask for pruned sequence.
        new_mask = None
        if causal_mask is not None:
            # causal_mask shape: (B, 1, S, S)
            new_mask = causal_mask.gather(
                2, keep_idx.unsqueeze(1).unsqueeze(-1).expand(
                    -1, causal_mask.shape[1], -1, causal_mask.shape[-1])
            ).gather(
                3, keep_idx.unsqueeze(1).unsqueeze(2).expand(
                    -1, causal_mask.shape[1], min_keep, -1)
            )

        # Rebuild position_ids.
        new_pos_ids = position_ids.gather(1, keep_idx) if position_ids is not None \
            else None

        # Rebuild position_embeddings (cos, sin).
        cos, sin = position_embeddings
        new_cos = cos.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, cos.shape[-1]))
        new_sin = sin.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, sin.shape[-1]))

        new_cache_pos = cache_position[:min_keep] if cache_position is not None \
            else None

        # Update img_mask_2d to the pruned sequence.
        new_img_mask = img_mask_2d.gather(1, keep_idx)

        return (hidden_states, new_mask, new_pos_ids,
                (new_cos, new_sin), new_cache_pos, new_img_mask, keep_idx)

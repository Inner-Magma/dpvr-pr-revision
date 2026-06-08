"""TD × FastV: TokenDiversion light-branch + FastV visual-token pruning.

The light branch (single_transformer) processes image tokens at split_layer.
FastV pruning then reduces low-attention visual tokens at fastv_layer (applied
after the light branch injects its output back into the main stream).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers import PreTrainedModel

from .token_diversion import TokenDiversionLlava
from .registry import register_method


@register_method("td_plus_fastv")
class TDFastVLlava(TokenDiversionLlava):
    """TokenDiversion with additional FastV pruning in the deep layers.

    Inherits all of TokenDiversionLlava's __init__ / forward logic and adds
    one pruning step at fastv_layer (must be > split_layer).

    Extra config knobs:
        method.fastv_layer   (default split_layer + 4)
        method.fastv_ratio   fraction of visual tokens to KEEP (default 0.5)
    """

    def __init__(self, base_model_path: str, fastv_layer: int | None = None,
                 fastv_ratio: float = 0.5, **kwargs):
        super().__init__(base_model_path, **kwargs)
        split = kwargs.get("split_layer", 18)
        self.fastv_layer = fastv_layer if fastv_layer is not None else split + 4
        self.fastv_ratio = fastv_ratio

    @classmethod
    def from_config(cls, cfg: dict) -> "TDFastVLlava":
        m = cfg["method"]
        split_layer = m.get("split_layer", 18)
        return cls(
            base_model_path=cfg["model"]["base_model_path"],
            llama_path=cfg["model"].get("llama_path"),
            vision_tower_path=cfg["model"].get("vision_tower_path"),
            projector_weights_path=cfg["model"].get("projector_weights_path"),
            split_layer=split_layer,
            vision_depth=m.get("vision_depth", 1),
            freeze_strategy=m.get("freeze_strategy", "all_but_single"),
            fastv_layer=m.get("fastv_layer", split_layer + 4),
            fastv_ratio=m.get("fastv_ratio", 0.5),
        )

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None,
                labels=None, use_cache: bool = False, past_key_values=None,
                cache_position=None, **kwargs) -> CausalLMOutputWithPast:
        # Delegate fully to parent — pruning hook injected via overridden
        # _deep_layer_hook called inside the parent's deep-layer loop.
        # Simplest approach: call parent, then re-run deep layers with pruning.
        # For cleanliness we re-implement the deep loop here with pruning.
        if attention_mask is not None:
            attention_mask = attention_mask == 1

        inputs_embeds, image_token_mask = self.llava_model.model.prepare_multimodal_input(
            input_ids, pixel_values
        )
        language_model = self.llava_model.language_model

        if attention_mask is not None and cache_position is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(~attention_mask, 0)
        else:
            position_ids = None

        (hidden_states, causal_mask, position_ids, past_key_values,
         cache_position, position_embeddings) = language_model.prepare_input_mine(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
        )

        # Shallow shared stack.
        for i in range(self.split_layer):
            out = language_model.layers[i](
                hidden_states, position_embeddings=position_embeddings,
                attention_mask=causal_mask, past_key_values=past_key_values,
                cache_position=cache_position, output_attentions=False,
                use_cache=use_cache,
            )
            hidden_states = out[0] if isinstance(out, (tuple, list)) else out

        # Light branch (TokenDiversion).
        has_image = image_token_mask is not None and image_token_mask.any()
        if has_image:
            B, _, H = hidden_states.shape
            max_img = int(image_token_mask.sum(dim=1).max().item())
            image_part_features = hidden_states[image_token_mask].view(B, max_img, H)
            image_attn_mask = torch.ones(B, max_img, dtype=torch.bool,
                                         device=hidden_states.device)
            image_pos_ids = torch.arange(max_img, device=hidden_states.device
                                         ).unsqueeze(0).expand(B, -1)
            (img_h, img_cm, _, _, img_cp, img_pe) = language_model.prepare_input_mine(
                inputs_embeds=image_part_features,
                attention_mask=image_attn_mask,
                position_ids=image_pos_ids,
                use_cache=False,
            )
            out_image = img_h
            for layer in self.single_transformer:
                out_image = layer(out_image, position_embeddings=img_pe,
                                  attention_mask=img_cm, past_key_values=None,
                                  cache_position=img_cp, output_attentions=False,
                                  use_cache=False)
                out_image = out_image[0] if isinstance(out_image, (tuple, list)) else out_image
            img_mask_2d = image_token_mask.any(dim=-1)
        else:
            out_image = None
            img_mask_2d = None

        # Deep layers with optional FastV pruning.
        keep_idx_used = None
        for i in range(self.split_layer, len(language_model.layers)):
            if out_image is not None:
                hidden_states[img_mask_2d] = out_image.reshape(-1, hidden_states.shape[-1])

            # FastV pruning at the designated layer.
            if i == self.fastv_layer and img_mask_2d is not None and img_mask_2d.any():
                hidden_states, img_mask_2d, position_embeddings, \
                    causal_mask, cache_position, keep_idx_used = \
                    self._fastv_prune(hidden_states, img_mask_2d, position_embeddings,
                                      causal_mask, cache_position, language_model, i)
                out_image = None   # No more injection after pruning.

            out = language_model.layers[i](
                hidden_states, position_embeddings=position_embeddings,
                attention_mask=causal_mask, past_key_values=past_key_values,
                cache_position=cache_position, output_attentions=False,
                use_cache=use_cache,
            )
            hidden_states = out[0] if isinstance(out, (tuple, list)) else out

        final_hidden_states = language_model.norm(hidden_states)
        logits = self.llava_model.lm_head(final_hidden_states)

        loss = None
        if labels is not None:
            if keep_idx_used is not None:
                # logit[b, i] predicts original position keep_idx_used[b, i] + 1.
                gather_idx = (keep_idx_used + 1).clamp(max=labels.shape[1] - 1)
                shift_logits = logits.float().contiguous()
                shift_labels = labels.gather(1, gather_idx)
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

    def _fastv_prune(self, hidden_states, img_mask_2d, position_embeddings,
                     causal_mask, cache_position, language_model, layer_idx):
        """Keep top-fastv_ratio image tokens by mean attention score.

        Also rebuilds causal_mask and cache_position so subsequent layers
        see a consistent (pruned) sequence length.
        """
        B, S, H = hidden_states.shape
        attn = language_model.layers[layer_idx].self_attn
        with torch.no_grad():
            q = attn.q_proj(hidden_states).view(B, S, -1, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(hidden_states).view(B, S, -1, attn.head_dim).transpose(1, 2)
            scores = (q @ k.transpose(-1, -2)).mean(dim=(1, 2))  # (B, S)

        img_scores = scores.masked_fill(~img_mask_2d, float("-inf"))
        n_img = img_mask_2d.sum(dim=1).min().item()
        k_keep = max(1, int(n_img * self.fastv_ratio))
        topk_idx = img_scores.topk(k_keep, dim=1).indices
        keep_mask = ~img_mask_2d.clone()
        keep_mask.scatter_(1, topk_idx, True)

        min_keep = keep_mask.sum(dim=1).min().item()
        keep_idx = torch.stack([
            keep_mask[b].nonzero(as_tuple=True)[0][:min_keep] for b in range(B)
        ])
        hidden_states = hidden_states.gather(
            1, keep_idx.unsqueeze(-1).expand(-1, -1, H))
        new_img_mask = img_mask_2d.gather(1, keep_idx)

        cos, sin = position_embeddings
        new_cos = cos.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, cos.shape[-1]))
        new_sin = sin.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, sin.shape[-1]))

        # Rebuild causal_mask for the pruned sequence length.
        new_causal_mask = None
        if causal_mask is not None:
            # causal_mask: (B, 1, S, S) → gather rows and cols for kept positions
            new_causal_mask = causal_mask.gather(
                2, keep_idx.unsqueeze(1).unsqueeze(-1).expand(
                    -1, causal_mask.shape[1], -1, causal_mask.shape[-1])
            ).gather(
                3, keep_idx.unsqueeze(1).unsqueeze(2).expand(
                    -1, causal_mask.shape[1], min_keep, -1)
            )

        new_cache_pos = cache_position[:min_keep] if cache_position is not None else None

        return (hidden_states, new_img_mask, (new_cos, new_sin),
                new_causal_mask, new_cache_pos, keep_idx)

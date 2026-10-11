"""DPVR-LF on Qwen2.5-VL (second backbone, C8 / E6; R1.14, R2.2).

Same routing as ``token_diversion_x3_fusion`` (the paper's DPVR-LF), on a
stock ``Qwen2_5_VLForConditionalGeneration`` from transformers 4.57.1:

- layers 0 .. s-1 (frozen): full sequence, vanilla.
- side branch (trainable, ``vision_depth`` decoder layers): image tokens only,
  taken from the hidden state after layer s-1.
- layers s .. L-K-1 (frozen): text tokens only. Image positions are not
  computed at all.
- layers L-K .. L-1 (frozen, K = ``fusion_count``): full sequence again, with
  the image positions overwritten by the side-branch output first.

Differences from the LLaVA-1.5 class, all forced by the backbone:

- Image token counts vary per sample (dynamic resolution), so sub-sequences
  are row-packed with ``routing.pack_index`` and padded; padding is hidden in
  every attention mask, so a sample gives the same logits alone or in a
  batch (``tests/test_qwen_dpvr.py``).
- Positions are Qwen's 3D M-RoPE ids from ``get_rope_index``. Both
  sub-sequences keep the ids of their tokens in the full sequence, so the
  side branch sees each image token's (t, h, w) position.
- Batch padding is masked in all layers (Qwen pads on the left at inference).
- No KV cache: every generation step recomputes the full prefix
  (``use_cache=False``, the accuracy protocol). ``prepare_inputs_for_generation``
  therefore always passes the full sequence and the image.

Attention runs with ``attn_implementation="sdpa"`` (or eager) because the
layers receive explicit 4D additive masks; flash-attention-2 does not take them.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .registry import register_method
from .routing import causal_padding_mask, gather_rows, pack_index, scatter_rows


def _hidden(out):
    return out[0] if isinstance(out, (tuple, list)) else out


@register_method("token_diversion_x3_fusion_qwen25vl")
class TokenDiversionX3FusionQwen25VL(PreTrainedModel, GenerationMixin):
    """Qwen2.5-VL with DPVR-LF routing. See the module docstring."""

    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _supports_sdpa = True  # the layers get explicit 4D masks; sdpa and eager take them

    def __init__(
        self,
        base_model_path: str | None = None,
        split_layer: int = 16,
        vision_depth: int = 1,
        fusion_count: int = 1,
        freeze_strategy: str = "all_but_single",
        branch_init: str = "random",
        attn_implementation: str = "sdpa",
        route: bool = True,
        qwen_model: nn.Module | None = None,
    ):
        if qwen_model is None:
            from transformers import Qwen2_5_VLForConditionalGeneration

            if attn_implementation not in ("sdpa", "eager"):
                raise ValueError("DPVR passes 4D masks to the layers; use sdpa or eager attention")
            qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_path, torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation, low_cpu_mem_usage=True,
            )
        super().__init__(qwen_model.config)
        self.qwen = qwen_model
        # Use the backbone's own generation defaults (e.g. its repetition
        # penalty) so vanilla and DPVR decode under the same settings.
        if getattr(qwen_model, "generation_config", None) is not None:
            self.generation_config = qwen_model.generation_config

        text_cfg = qwen_model.config.text_config
        n_total = text_cfg.num_hidden_layers
        self.split_layer = int(split_layer)
        self.vision_depth = int(vision_depth)
        self.fusion_count = int(fusion_count)
        self.route = bool(route)
        n_deep = n_total - self.split_layer
        if not (1 <= self.split_layer < n_total):
            raise ValueError(f"split_layer={split_layer} must be in [1, {n_total - 1}]")
        if not (1 <= self.fusion_count <= n_deep):
            raise ValueError(
                f"fusion_count={fusion_count} must be in [1, {n_deep}] "
                f"(deep layers between split_layer={split_layer} and end={n_total})"
            )

        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer

        layers = self._lm.layers
        self.single_transformer = nn.ModuleList([
            Qwen2_5_VLDecoderLayer(text_cfg, layer_idx=self.split_layer + i)
            for i in range(self.vision_depth)
        ])
        if branch_init == "copy":
            # Start from the frozen layers the branch replaces (s, s+1, ...).
            for i, layer in enumerate(self.single_transformer):
                layer.load_state_dict(layers[self.split_layer + i].state_dict())
        elif branch_init != "random":
            raise ValueError(f"branch_init must be random or copy, got {branch_init!r}")
        self.single_transformer.to(next(layers.parameters()).dtype)

        if freeze_strategy == "all_but_single":
            for p in self.qwen.parameters():
                p.requires_grad_(False)
        elif freeze_strategy != "full_ft":
            raise ValueError(f"freeze_strategy {freeze_strategy!r} is not supported for Qwen2.5-VL")
        for p in self.single_transformer.parameters():
            p.requires_grad_(True)

    # ---------------------------------------------------------- registry glue
    @classmethod
    def from_config(cls, cfg: dict) -> "TokenDiversionX3FusionQwen25VL":
        m = cfg["method"]
        return cls(
            base_model_path=cfg["model"]["base_model_path"],
            split_layer=m.get("split_layer", 16),
            vision_depth=m.get("vision_depth", 1),
            fusion_count=m.get("fusion_count", 1),
            freeze_strategy=m.get("freeze_strategy", "all_but_single"),
            branch_init=m.get("branch_init", "random"),
            attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
            route=m.get("route", True),
        )

    @property
    def _lm(self):
        return self.qwen.model.language_model

    # -------------------------------------------------------------------- utils
    def get_input_embeddings(self):
        return self.qwen.get_input_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # Frozen backbone layers only, as in the LLaVA class: the side branch is
        # not checkpointed, so its inputs need not require grad.
        return self.qwen.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def print_trainable_parameters(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} | all params: {total:,} | "
              f"trainable%: {100.0 * trainable / max(total, 1):.6f}")

    # -------------------------------------------------------- generation glue
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, pixel_values=None,
                                      image_grid_thw=None, **kwargs) -> dict:
        # No cache: every step sees the whole sequence and the image again.
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "pixel_values": pixel_values, "image_grid_thw": image_grid_thw,
                "use_cache": False}

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, *args, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, *args, **kwargs)
        model_kwargs["past_key_values"] = None
        return model_kwargs

    # ---------------------------------------------------------------- forward
    def _run(self, layer_ids, h, mask, pos_emb):
        layers = self._lm.layers
        for i in layer_ids:
            h = _hidden(layers[i](h, attention_mask=mask, position_embeddings=pos_emb))
        return h

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        qm = self.qwen.model
        lm = self._lm
        B, S = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        valid = attention_mask.bool()

        h = qm.get_input_embeddings()(input_ids)
        image_mask = None
        if pixel_values is not None:
            feats = torch.cat(qm.get_image_features(pixel_values, image_grid_thw), dim=0)
            feats = feats.to(h.device, h.dtype)
            mask_3d, _ = qm.get_placeholder_mask(input_ids, inputs_embeds=h, image_features=feats)
            h = h.masked_scatter(mask_3d, feats)
            image_mask = (input_ids == qm.config.image_token_id) & valid

        position_ids, _ = qm.get_rope_index(input_ids, image_grid_thw, None, attention_mask=attention_mask)
        full_mask = causal_padding_mask(valid, h.dtype)
        full_pos = lm.rotary_emb(h, position_ids)

        n_total = len(lm.layers)
        fusion_start = n_total - self.fusion_count
        h = self._run(range(self.split_layer), h, full_mask, full_pos)

        if self.route and image_mask is not None and image_mask.any():
            # Side branch on image tokens (positions as in the full sequence).
            img = pack_index(image_mask)
            img_h = gather_rows(h, img)
            img_mask = causal_padding_mask(img.valid, h.dtype)
            img_pos = lm.rotary_emb(img_h, gather_rows(position_ids, img, seq_dim=2))
            for layer in self.single_transformer:
                img_h = _hidden(layer(img_h, attention_mask=img_mask, position_embeddings=img_pos))

            # Text-only deep layers.
            txt = pack_index(~image_mask)
            txt_h = gather_rows(h, txt)
            txt_valid = txt.valid & gather_rows(valid, txt)
            txt_mask = causal_padding_mask(txt_valid, h.dtype)
            txt_pos = lm.rotary_emb(txt_h, gather_rows(position_ids, txt, seq_dim=2))
            txt_h = self._run(range(self.split_layer, fusion_start), txt_h, txt_mask, txt_pos)

            h = scatter_rows(h, txt_h, txt)
            h = scatter_rows(h, img_h, img)
            h = self._run(range(fusion_start, n_total), h, full_mask, full_pos)
        else:
            h = self._run(range(self.split_layer, n_total), h, full_mask, full_pos)

        logits = self.qwen.lm_head(lm.norm(h))

        loss = None
        if labels is not None:
            shift_logits = logits.float()[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


__all__ = ["TokenDiversionX3FusionQwen25VL"]

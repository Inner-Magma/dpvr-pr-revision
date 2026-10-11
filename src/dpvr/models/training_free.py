"""Training-free variants of the DPVR compute graph on a frozen LLaVA (C3).

These isolate what the trained side branch and the late fusion contribute
(R1.1, R1.12, R1.13). All weights are the vanilla LLaVA-1.5 weights; nothing
is trained. With L decoder layers and split layer s:

- ``vtw`` (Visual Tokens Withdrawal, Lin et al.): layers 0..s-1 run on
  image+text; image tokens are then withdrawn and layers s..L-1 run on text
  only. This is DPVR-LF with neither side branch nor fusion.
- ``lf_no_branch``: DPVR-LF's graph with the side branch replaced by the
  identity. Layers s..L-K-1 run on text only; before the last K layers the
  image positions get their layer-s hidden states back and the last K layers
  run on image+text.
- ``freeze_image``: every layer runs on image+text, but from layer s on the
  image hidden states are held at their layer-s value. Text keeps attending to
  image tokens at every depth; only the image tokens' own deep updates are
  removed. No compute saving; it tests whether those updates matter.

Prefill semantics only: generation recomputes the full sequence every step
(``use_cache=False``), which is how accuracy is evaluated anyway. Like
DPVR-LF, every batch row must carry the same number of image tokens.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, GenerationConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling_llava import LlavaForConditionalGeneration
from .registry import register_method

VARIANTS = ("vanilla", "vtw", "lf_no_branch", "freeze_image")


def _text_mask(text_attn: Optional[torch.Tensor], B: int, T: int, device, dtype) -> torch.Tensor:
    """Additive (B, 1, T, T) causal mask over the text sub-sequence, padding keys masked."""
    neg = torch.finfo(dtype).min / 2
    mask = torch.triu(torch.full((T, T), neg, dtype=dtype, device=device), diagonal=1)
    mask = mask.expand(B, 1, T, T).clone()
    if text_attn is not None:
        mask = mask.masked_fill(~text_attn[:, None, None, :], neg)
    return mask


def _run(layer, hidden, position_embeddings, attention_mask):
    out = layer(
        hidden,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=None,
        cache_position=None,
        output_attentions=False,
        use_cache=False,
    )
    return out[0] if isinstance(out, (tuple, list)) else out


class TrainingFreeLlava(PreTrainedModel, GenerationMixin):
    variant = "vanilla"
    _supports_sdpa = True

    def __init__(self, base_model_path: str, split_layer: int = 18, fusion_count: int = 1):
        super().__init__(AutoConfig.from_pretrained(base_model_path))
        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True,
        )
        for p in self.llava_model.parameters():
            p.requires_grad_(False)
        self._configure(split_layer, fusion_count)

    def _configure(self, split_layer: int, fusion_count: int) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}")
        n = len(self.llava_model.language_model.layers)
        if not (1 <= split_layer <= n):
            raise ValueError(f"split_layer={split_layer} must be in [1, {n}]")
        if self.variant == "lf_no_branch" and not (1 <= fusion_count <= n - split_layer):
            raise ValueError(f"fusion_count={fusion_count} must be in [1, {n - split_layer}]")
        self.split_layer = int(split_layer)
        self.fusion_count = int(fusion_count) if self.variant == "lf_no_branch" else 0

    @classmethod
    def from_config(cls, cfg: dict) -> "TrainingFreeLlava":
        m = cfg.get("method", {})
        return cls(
            base_model_path=cfg["model"]["base_model_path"],
            split_layer=m.get("split_layer", 18),
            fusion_count=m.get("fusion_count", 1),
        )

    @classmethod
    def _from_llava(cls, llava_model, config, split_layer: int, fusion_count: int = 1,
                    variant: str | None = None) -> "TrainingFreeLlava":
        """Wrap an existing (test) LLaVA module without loading weights."""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.config = config
        obj.generation_config = GenerationConfig.from_model_config(config)
        obj.name_or_path = ""
        obj._is_hf_initialized = True
        if variant is not None:
            obj.variant = variant
        obj.llava_model = llava_model
        obj._configure(split_layer, fusion_count)
        return obj

    def print_trainable_parameters(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: 0 | all params: {total:,} | trainable%: 0.000000")

    # -------------------------------------------------------- generation glue
    def prepare_inputs_for_generation(self, input_ids, pixel_values=None, attention_mask=None,
                                      **kwargs) -> dict:
        # No cache: every step sees the whole sequence and the image again.
        return {"input_ids": input_ids, "pixel_values": pixel_values,
                "attention_mask": attention_mask, "use_cache": False}

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        am = attention_mask.bool() if attention_mask is not None else None
        inputs_embeds, image_mask = self.llava_model.model.prepare_multimodal_input(
            input_ids, pixel_values
        )
        lm = self.llava_model.language_model
        position_ids = None
        if am is not None:
            position_ids = am.long().cumsum(-1) - 1
            position_ids.masked_fill_(~am, 0)

        hidden, causal_mask, position_ids, _, _, pos_emb = lm.prepare_input_mine(
            inputs_embeds=inputs_embeds, attention_mask=am, position_ids=position_ids,
            use_cache=False,
        )
        layers = lm.layers
        L = len(layers)
        has_image = image_mask is not None and bool(image_mask.any())
        s = self.split_layer if (has_image and self.variant != "vanilla") else L

        for i in range(s):
            hidden = _run(layers[i], hidden, pos_emb, causal_mask)

        if s < L:
            B, S, H = hidden.shape
            img = image_mask.any(dim=-1)  # (B, S)
            n_img = int(img.sum(dim=1).max())
            h_img = hidden[img].view(B, n_img, H)

            if self.variant == "freeze_image":
                for i in range(s, L):
                    hidden = _run(layers[i], hidden, pos_emb, causal_mask).clone()
                    hidden[img] = h_img.reshape(-1, H)
            else:
                txt = ~img
                T = S - n_img
                text_h = hidden[txt].view(B, T, H)
                full_pos = position_ids.expand(B, S)
                text_pos_emb = lm.rotary_emb(text_h, full_pos[txt].view(B, T))
                text_attn = am[txt].view(B, T) if am is not None else None
                tmask = _text_mask(text_attn, B, T, hidden.device, hidden.dtype)
                fusion_start = L - self.fusion_count
                for i in range(s, fusion_start):
                    text_h = _run(layers[i], text_h, text_pos_emb, tmask)
                hidden = hidden.clone()  # image positions keep their layer-s states
                hidden[txt] = text_h.reshape(-1, H)
                for i in range(fusion_start, L):
                    hidden = _run(layers[i], hidden, pos_emb, causal_mask)
        else:
            for i in range(s, L):
                hidden = _run(layers[i], hidden, pos_emb, causal_mask)

        logits = self.llava_model.lm_head(lm.norm(hidden))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                logits.float()[..., :-1, :].reshape(-1, logits.shape[-1]),
                labels[..., 1:].reshape(-1),
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


@register_method("vtw")
class VTWLlava(TrainingFreeLlava):
    variant = "vtw"


@register_method("lf_no_branch")
class LFNoBranchLlava(TrainingFreeLlava):
    variant = "lf_no_branch"


@register_method("freeze_image")
class FreezeImageLlava(TrainingFreeLlava):
    variant = "freeze_image"

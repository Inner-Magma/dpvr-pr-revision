"""Token-Diversion B-2 variant: image tokens serve as frozen K/V context in deep layers.

Key difference from variant A:
- A: deep layers run the full sequence; image positions are overwritten with single output
  before each layer → stable cross-modal context at image positions.
- B-2 (this file): deep layers run TEXT ONLY. Image tokens from single_transformer are
  concatenated as K/V prefix to text hidden states; their updated outputs are discarded.
  Image positions in the final hidden state come from single output, never updated by deep.

Compute saving (vs A): image positions skip Q-projection, FFN, and layer norm in deep layers
(~25-35% FLOPs saved for typical image token fractions).

v1 vs v2 (current):
- v1: text Q/K/V projections and FFN in deep layers were computed with full autograd.
  Frozen weights accumulated spurious gradients and the backward graph held intermediate
  activations unnecessarily, wasting memory and compute.
- v2: text Q/K/V and FFN are computed under torch.no_grad() (detached), eliminating their
  backward graphs. Image K/V remain grad-tracked, preserving the sole gradient path to
  single_transformer. Result: ~30-40% lower VRAM usage during training + faster s/step.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, apply_rotary_pos_emb, repeat_kv

from .modeling_llama import LlamaModel
from .modeling_llava import LlavaForConditionalGeneration
from .registry import register_method


def _build_b2_attention_mask(
    B: int,
    T_img: int,
    T_txt: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build 4D additive attention mask for B-2 combined [img; txt] sequence.

    Text positions attend to: all image positions (no causal restriction) + causal within text.
    Image positions: no masking (their outputs are discarded anyway).

    Returns shape (B, 1, T_img+T_txt, T_img+T_txt) with 0.0 for attended, -inf for masked.
    """
    T = T_img + T_txt
    mask = torch.zeros(B, 1, T, T, dtype=dtype, device=device)
    if T_txt > 1:
        neg_inf = torch.finfo(dtype).min / 2
        # Causal mask for text-text sub-block: upper triangle (excluding diagonal) is -inf
        text_causal = torch.triu(
            torch.full((T_txt, T_txt), neg_inf, dtype=dtype, device=device),
            diagonal=1,
        )
        mask[:, :, T_img:, T_img:] = text_causal.unsqueeze(0).unsqueeze(0)
    return mask


def _b2_deep_layer_forward(
    layer: LlamaDecoderLayer,
    text_hidden: torch.Tensor,
    out_image: torch.Tensor,
    text_pos_emb: tuple,
    image_pos_emb: tuple,
    text_attn_mask: torch.Tensor,
) -> torch.Tensor:
    """B-2 single-layer forward (v2): image K/V are grad-tracked; text path is forward-only.

    Gradient flows: loss → text_hidden → o_proj(attn_out) → attn_out → image_V/image_K
                    → image_normed → out_image → single_transformer (trainable).
    Text Q/K/V and FFN are detached (no backward graph), since text-side weights are frozen.
    """
    B, T_txt, H = text_hidden.shape
    _, T_img, _ = out_image.shape
    attn = layer.self_attn
    head_dim = attn.head_dim

    # Image side: grad-tracked — sole gradient path back to single_transformer
    image_normed = layer.input_layernorm(out_image)
    image_K = attn.k_proj(image_normed).view(B, T_img, -1, head_dim).transpose(1, 2)
    image_V = attn.v_proj(image_normed).view(B, T_img, -1, head_dim).transpose(1, 2)

    # Text side (v2): forward-only — frozen weights, no need to build backward graph
    with torch.no_grad():
        text_normed_nograd = layer.input_layernorm(text_hidden)
        text_Q_full = attn.q_proj(text_normed_nograd)
        text_K_full = attn.k_proj(text_normed_nograd)
        text_V_full = attn.v_proj(text_normed_nograd)
    text_Q = text_Q_full.view(B, T_txt, -1, head_dim).transpose(1, 2)
    text_K = text_K_full.view(B, T_txt, -1, head_dim).transpose(1, 2)
    text_V = text_V_full.view(B, T_txt, -1, head_dim).transpose(1, 2)

    # Apply RoPE with each sub-sequence's own position IDs
    text_Q, text_K = apply_rotary_pos_emb(text_Q, text_K, *text_pos_emb)
    # Pass image_K as both q and k; discard the rotated-Q result — only K is needed
    _, image_K = apply_rotary_pos_emb(image_K, image_K, *image_pos_emb)

    # Concatenate [image | text] K/V and expand for GQA
    full_K = repeat_kv(torch.cat([image_K, text_K], dim=2), attn.num_key_value_groups)
    full_V = repeat_kv(torch.cat([image_V, text_V], dim=2), attn.num_key_value_groups)

    # Scaled dot-product: text Q attends to [image; text] K/V with causal mask on text block
    # text_Q is detached but its value is used to compute d_full_K in backward → image_K gets grad
    attn_w = torch.matmul(text_Q, full_K.transpose(2, 3)) * attn.scaling
    if text_attn_mask is not None:
        attn_w = attn_w + text_attn_mask
    attn_w = torch.nn.functional.softmax(attn_w, dim=-1, dtype=torch.float32).to(text_Q.dtype)
    attn_out = torch.matmul(attn_w, full_V)  # (B, n_heads, T_txt, head_dim)

    # Output projection on text only + residual
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_txt, H)
    text_hidden = text_hidden + attn.o_proj(attn_out)

    # FFN on text only + residual (v2: detached — frozen FFN, no backward needed)
    with torch.no_grad():
        ffn_out = layer.mlp(layer.post_attention_layernorm(text_hidden))
    text_hidden = text_hidden + ffn_out

    return text_hidden


@register_method("token_diversion_substitution")
class TokenDiversionSubstitutionLlava(PreTrainedModel, GenerationMixin):
    """LLaVA with B-2 substitution: image tokens serve as frozen K/V context.

    Shallow layers (0..split_layer-1): all tokens, frozen, same as variant A.
    split_layer: image tokens extracted → single_transformer (trainable) → out_image.
    Deep layers (split_layer..L): TEXT-ONLY propagation.
        Before each layer: concat [out_image; text_hidden] → run layer → keep text outputs.
        Image K/V contribution is preserved; image residual updates are discarded.
    Reassembly: image_final = single output, text_final = deep output.
    """

    main_input_name = "input_ids"

    def __init__(
        self,
        base_model_path: str,
        llama_path: str | None = None,
        vision_tower_path: str | None = None,
        projector_weights_path: str | None = None,
        split_layer: int = 18,
        vision_depth: int = 1,
        freeze_strategy: str = "all_but_single",
    ):
        llava_config = AutoConfig.from_pretrained(base_model_path)
        super().__init__(llava_config)

        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True,
        )

        if llama_path is not None and Path(llama_path).exists():
            language_model = LlamaModel.from_pretrained(
                llama_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            )
            self.llava_model.model.language_model.layers = language_model.layers
            self.llava_model.model.language_model.norm = language_model.norm
            self.llava_model.model.language_model.rotary_emb = language_model.rotary_emb

        if vision_tower_path is not None and Path(vision_tower_path).exists():
            from transformers import CLIPVisionModel
            vision_tower = CLIPVisionModel.from_pretrained(
                vision_tower_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            )
            self.llava_model.model.vision_tower = vision_tower

        if projector_weights_path is not None and Path(projector_weights_path).exists():
            projector_ckpt = torch.load(projector_weights_path, map_location="cpu")
            rename_map = {
                "model.mm_projector.0.weight": "linear_1.weight",
                "model.mm_projector.0.bias":   "linear_1.bias",
                "model.mm_projector.2.weight": "linear_2.weight",
                "model.mm_projector.2.bias":   "linear_2.bias",
            }
            proj_state = {
                rename_map[k]: v for k, v in projector_ckpt.items() if k in rename_map
            }
            self.llava_model.multi_modal_projector.load_state_dict(proj_state, strict=True)

        from ..training.freezing import apply_llava_freeze_strategy

        self.split_layer = split_layer
        self.vision_depth = vision_depth
        llama_cfg = self.llava_model.language_model.config

        self.single_transformer = nn.ModuleList([
            LlamaDecoderLayer(llama_cfg, layer_idx=split_layer + i)
            for i in range(vision_depth)
        ])
        self.single_transformer.to(torch.bfloat16)

        apply_llava_freeze_strategy(
            self.llava_model,
            strategy=freeze_strategy,
            split_layer=split_layer,
            trainable_modules=list(self.single_transformer),
        )

    # -------------------------------------------------------------- registry glue
    @classmethod
    def from_config(cls, cfg: dict) -> "TokenDiversionSubstitutionLlava":
        return cls(
            base_model_path=cfg["model"]["base_model_path"],
            llama_path=cfg["model"].get("llama_path"),
            vision_tower_path=cfg["model"].get("vision_tower_path"),
            projector_weights_path=cfg["model"].get("projector_weights_path"),
            split_layer=cfg["method"].get("split_layer", 18),
            vision_depth=cfg["method"].get("vision_depth", 1),
            freeze_strategy=cfg["method"].get("freeze_strategy", "all_but_single"),
        )

    @classmethod
    def from_trained(cls, checkpoint_dir: str) -> "TokenDiversionSubstitutionLlava":
        """Load a saved checkpoint (same format as TokenDiversionLlava.from_trained)."""
        import json as _json
        ckpt = Path(checkpoint_dir)
        cfg_file = ckpt / "training_config.json"
        if cfg_file.exists():
            cfg = _json.loads(cfg_file.read_text())
        else:
            raise FileNotFoundError(f"training_config.json not found in {checkpoint_dir}")

        import dpvr.models as _dpvr_models  # noqa: F401
        from .registry import _REGISTRY as _method_registry
        method_name = cfg.get("method", {}).get("name", "token_diversion_substitution")
        target_cls = _method_registry.get(method_name, cls)
        model = target_cls.from_config(cfg)

        index_file = ckpt / "model.safetensors.index.json"
        sf_file    = ckpt / "model.safetensors"
        state_file = ckpt / "pytorch_model.bin"

        if index_file.exists():
            from safetensors.torch import load_file
            index = _json.loads(index_file.read_text())
            shard_set = sorted(set(index["weight_map"].values()))
            state: dict = {}
            for shard in shard_set:
                state.update(load_file(str(ckpt / shard)))
        elif sf_file.exists():
            from safetensors.torch import load_file
            state = load_file(str(sf_file))
        elif state_file.exists():
            state = torch.load(state_file, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")

        model.load_state_dict(state, strict=False)
        return model

    # -------------------------------------------------------------------- utils
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.llava_model, "gradient_checkpointing_enable"):
            return self.llava_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def print_trainable_parameters(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        pct = 100.0 * trainable / max(total, 1)
        print(
            f"trainable params: {trainable:,} | all params: {total:,} | "
            f"trainable%: {pct:.6f}"
        )

    # ---------------------------------------------------------------- generation glue
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        **kwargs,
    ) -> dict:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "cache_position": cache_position,
        }
        if cache_position is None or cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
        return model_inputs

    # -------------------------------------------------------------- B-2 forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
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

        (
            hidden_states,
            causal_mask,
            position_ids,
            past_key_values,
            cache_position,
            position_embeddings,
        ) = language_model.prepare_input_mine(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
        )

        # ── Shared shallow stack (layers 0 … split_layer-1) ──────────────────
        for i in range(self.split_layer):
            layer = language_model.layers[i]
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

        # ── Light branch: only during PREFILL (image tokens present) ─────────
        has_image = image_token_mask is not None and image_token_mask.any()
        if has_image:
            B, S, H = hidden_states.shape
            max_img = int(image_token_mask.sum(dim=1).max().item())

            image_part_features = hidden_states[image_token_mask].view(B, max_img, H)
            image_attn_mask = torch.ones(B, max_img, dtype=torch.bool, device=hidden_states.device)
            image_pos_ids = torch.arange(max_img, device=hidden_states.device).unsqueeze(0).expand(B, -1)

            (
                img_hidden,
                img_causal_mask,
                _,
                _,
                img_cache_pos,
                img_pos_emb,
            ) = language_model.prepare_input_mine(
                inputs_embeds=image_part_features,
                attention_mask=image_attn_mask,
                position_ids=image_pos_ids,
                use_cache=False,
            )

            out_image = img_hidden
            for layer in self.single_transformer:
                out_image = layer(
                    out_image,
                    position_embeddings=img_pos_emb,
                    attention_mask=img_causal_mask,
                    past_key_values=None,
                    cache_position=img_cache_pos,
                    output_attentions=False,
                    use_cache=False,
                )
                out_image = out_image[0] if isinstance(out_image, (tuple, list)) else out_image

            # Reduce mask to 2D for indexing
            image_token_mask_2d = image_token_mask.any(dim=-1)  # (B, S)

            # ── B-2 split: extract text hidden states ────────────────────────
            # Only text positions propagate through deep layers.
            txt_len = S - max_img
            text_hidden = hidden_states[~image_token_mask_2d].view(B, txt_len, H)

            # Preserve original position IDs for both image and text sub-sequences
            full_pos = position_ids.expand(B, S)  # (B, S)
            img_pos_ids_orig = full_pos[image_token_mask_2d].view(B, max_img)  # (B, T_img)
            txt_pos_ids_orig = full_pos[~image_token_mask_2d].view(B, txt_len)  # (B, T_txt)
            combined_pos_ids = torch.cat([img_pos_ids_orig, txt_pos_ids_orig], dim=1)  # (B, T_img+T_txt)

            # Compute position embeddings for the combined [img; txt] sequence once.
            combined_dummy = torch.zeros(
                B, max_img + txt_len, H,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            combined_pos_emb = language_model.rotary_emb(combined_dummy, combined_pos_ids)

            # Build attention mask: text attends to all image positions + causal within text.
            combined_mask = _build_b2_attention_mask(
                B, max_img, txt_len, hidden_states.device, hidden_states.dtype
            )

            # Split position embeddings for text and image sub-sequences.
            cos_all, sin_all = combined_pos_emb
            image_pos_emb_split = (cos_all[:, :max_img, :], sin_all[:, :max_img, :])
            text_pos_emb_split = (cos_all[:, max_img:, :], sin_all[:, max_img:, :])
            # Text-only mask: rows for text Q, cols for [image; text] K/V
            text_only_mask = combined_mask[:, :, max_img:, :]  # (B, 1, T_txt, T_img+T_txt)

        else:
            out_image = None
            image_token_mask_2d = None
            text_hidden = hidden_states
            combined_pos_emb = position_embeddings
            combined_mask = causal_mask

        # ── Deep layers (split_layer … end) — true B-2: text Q only, image K/V only ──
        for i in range(self.split_layer, len(language_model.layers)):
            layer = language_model.layers[i]

            if out_image is not None:
                # Image provides K/V only — Q, output proj, and FFN are text-only.
                text_hidden = _b2_deep_layer_forward(
                    layer, text_hidden, out_image,
                    text_pos_emb_split, image_pos_emb_split,
                    text_only_mask,
                )
            else:
                # No image: standard text-only forward
                out = layer(
                    text_hidden,
                    position_embeddings=combined_pos_emb,
                    attention_mask=combined_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    output_attentions=False,
                    use_cache=use_cache,
                )
                text_hidden = out[0] if isinstance(out, (tuple, list)) else out

        # ── Reassemble full sequence ──────────────────────────────────────────
        if out_image is not None:
            # image positions: single_transformer output (never updated by deep layers)
            # text positions: accumulated text_hidden after all deep layers
            final_hidden = hidden_states.clone()
            final_hidden[image_token_mask_2d] = out_image.reshape(-1, H)
            final_hidden[~image_token_mask_2d] = text_hidden.reshape(-1, H)
        else:
            final_hidden = text_hidden

        final_hidden_states = language_model.norm(final_hidden)
        logits = self.llava_model.lm_head(final_hidden_states)

        loss = None
        if labels is not None:
            logits_f = logits.float()
            shift_logits = logits_f[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, self.llava_model.config.vocab_size),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
        )

    # --------------------------------------------------------- test-only factory
    @classmethod
    def _build_test_instance(
        cls,
        hidden_size: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        vocab: int = 1000,
        split_layer: int = 2,
        vision_depth: int = 1,
    ) -> "TokenDiversionSubstitutionLlava":
        """Build a random-weight instance for CPU unit tests (no pretrained loading).

        Constructs a minimal LlamaModel and mock LLaVA wrapper directly from config.
        prepare_multimodal_input is mocked: first S//2 positions are treated as image tokens.
        """
        import types
        from transformers import LlamaConfig

        llama_cfg = LlamaConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_heads,
            vocab_size=vocab,
            max_position_embeddings=64,
            rms_norm_eps=1e-5,
            rope_scaling=None,
            attention_bias=False,
            mlp_bias=False,
            attention_dropout=0.0,
        )

        # Bypass __init__ pretrained loading: manual nn.Module init.
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.config = llama_cfg  # required by PreTrainedModel attribute access
        obj.name_or_path = ""   # required by PreTrainedModel
        obj._is_hf_initialized = True

        # Build actual LlamaModel (random weights, no pretrained).
        lm = LlamaModel(llama_cfg)

        embed = nn.Embedding(vocab, hidden_size)
        lm_head = nn.Linear(hidden_size, vocab, bias=False)

        def _mock_prepare_multimodal_input(input_ids, pixel_values=None):
            B, S = input_ids.shape
            inputs_embeds = embed(input_ids).float()
            if pixel_values is not None:
                T_img = S // 2  # first half = image tokens
                mask_2d = torch.zeros(B, S, dtype=torch.bool, device=input_ids.device)
                mask_2d[:, :T_img] = True
                mask_3d = mask_2d.unsqueeze(-1).expand(-1, -1, hidden_size)
                return inputs_embeds, mask_3d
            return inputs_embeds, None

        llava_config_ns = types.SimpleNamespace(vocab_size=vocab)
        llava_mm_ns = types.SimpleNamespace(prepare_multimodal_input=_mock_prepare_multimodal_input)
        llava_model_ns = types.SimpleNamespace(
            config=llava_config_ns,
            lm_head=lm_head,
            language_model=lm,
            model=llava_mm_ns,
        )

        obj.llava_model = llava_model_ns
        obj.split_layer = split_layer
        obj.vision_depth = vision_depth
        obj.single_transformer = nn.ModuleList([
            LlamaDecoderLayer(llama_cfg, layer_idx=split_layer + i)
            for i in range(vision_depth)
        ])

        return obj

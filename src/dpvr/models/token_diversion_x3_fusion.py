"""Token-Diversion X3-fusion-last variant: deep layers text-only, last K fuse.

Architecture (split_layer=18, vision_depth=1, fusion_count K, 7B has 32 layers):
- shallow 0..split_layer-1 (frozen): image+text vanilla
- single_transformer (vision_depth layers trainable): out_image = single(image_at_split)
- deep split_layer..end-K-1 (frozen, 14-K layers for 7B): TEXT-ONLY forward.
    Image processing entirely skipped (no Q/K/V/O/FFN computed for image positions
    in these layers) — true compute saving vs method A.
- deep end-K..end-1 (frozen, last K layers): full vanilla LlamaDecoderLayer forward.
    Before the FIRST of these layers, hidden[image_pos] is overwritten with
    out_image, so subsequent fusion layers' K/V projection on image positions
    originates from single_transformer output (gradient path back to
    single_transformer flows through all K fusion layers' attention).
- final reassembly: image_pos comes from the fusion stack's image output, text_pos
    from the fusion stack's text output.

fusion_count = 1 (default) reproduces the original last-1 architecture.
fusion_count = 2 is the last-2 ablation: 2 fusion paths, slightly more compute,
stronger gradient signal.

Compute saving vs Method A (K=1): ~25-30%; (K=2): ~22-27%.
Gradient signal: ~5% of A for K=1; ~10% for K=2 (2 paths vs 14).

DECODING with KV cache (use_cache=True): the prefill stores, per layer, the K/V
that layer actually computed — image+text for the shallow and fusion layers,
text only for the text-only deep layers — and tags the cache with
``_DPVR_LF_TEXT_LEN``. Each decode step then attends to exactly the context it
would see in a full prefill over the extended sequence, so cached decoding
matches the training/prefill semantics without recomputing the image path.

DECODING without a tagged cache (pixel_values is None, no image tokens): falls
back to the vanilla deep forward, identical to Method A's fallback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling_llama import LlamaModel
from .modeling_llava import LlavaForConditionalGeneration
from .registry import register_method
from .side_branch import build_side_branch, set_branch_trainable


def _build_text_only_causal_mask(
    B: int,
    T_txt: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build 4D additive causal mask for a text-only sub-sequence.

    Returns shape (B, 1, T_txt, T_txt) with 0.0 below+on diagonal, -inf above.
    """
    neg_inf = torch.finfo(dtype).min / 2
    mask = torch.zeros(B, 1, T_txt, T_txt, dtype=dtype, device=device)
    if T_txt > 1:
        triu = torch.triu(
            torch.full((T_txt, T_txt), neg_inf, dtype=dtype, device=device),
            diagonal=1,
        )
        mask = mask + triu.unsqueeze(0).unsqueeze(0)
    return mask


def _build_text_decode_mask(
    B: int,
    q_len: int,
    past_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """4D additive mask for decode steps in the text-only deep layers.

    Queries are the ``q_len`` new tokens; keys are the ``past_len`` cached text
    positions followed by the new tokens. Cached keys are always visible and the
    new tokens are causal among themselves. Like the prefill text-only mask,
    padding is not masked here, so cached decoding reproduces the prefill exactly.
    Returns shape (B, 1, q_len, past_len + q_len).
    """
    neg_inf = torch.finfo(dtype).min / 2
    mask = torch.zeros(B, 1, q_len, past_len + q_len, dtype=dtype, device=device)
    if q_len > 1:
        triu = torch.triu(
            torch.full((q_len, q_len), neg_inf, dtype=dtype, device=device),
            diagonal=1,
        )
        mask[..., past_len:] = mask[..., past_len:] + triu
    return mask


# Attribute set on the KV cache after a DPVR-LF prefill. It holds the number of
# text positions cached in the text-only deep layers.
_DPVR_LF_TEXT_LEN = "_dpvr_lf_text_len"


@register_method("token_diversion_x3_fusion")
class TokenDiversionX3FusionLlava(PreTrainedModel, GenerationMixin):
    """LLaVA with X3-fusion-last routing: deep layers text-only, last layer fuses.

    See module docstring for the full architecture description.
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
        fusion_count: int = 1,
        freeze_strategy: str = "all_but_single",
        branch_type: str = "transformer",
        branch_init: str = "random",
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
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
        self.fusion_count = int(fusion_count)
        llama_cfg = self.llava_model.language_model.config
        n_total = llama_cfg.num_hidden_layers
        n_deep = n_total - split_layer
        if not (1 <= self.fusion_count <= n_deep):
            raise ValueError(
                f"fusion_count={self.fusion_count} must be in [1, {n_deep}] "
                f"(deep layers between split_layer={split_layer} and end={n_total})"
            )

        # Side branch (C9): default is vision_depth random decoder layers; see side_branch.py.
        self.branch_type = branch_type
        self.single_transformer = build_side_branch(
            llama_cfg, split_layer, depth=vision_depth, branch_type=branch_type,
            branch_init=branch_init, source_layers=self.llava_model.language_model.layers,
            lora_rank=lora_rank, lora_alpha=lora_alpha,
        )

        apply_llava_freeze_strategy(
            self.llava_model,
            strategy=freeze_strategy,
            split_layer=split_layer,
            trainable_modules=list(self.single_transformer),
        )
        set_branch_trainable(self.single_transformer, branch_type)

    # ---------------------------------------------------------- registry glue
    @classmethod
    def from_config(cls, cfg: dict) -> "TokenDiversionX3FusionLlava":
        return cls(
            base_model_path=cfg["model"]["base_model_path"],
            llama_path=cfg["model"].get("llama_path"),
            vision_tower_path=cfg["model"].get("vision_tower_path"),
            projector_weights_path=cfg["model"].get("projector_weights_path"),
            split_layer=cfg["method"].get("split_layer", 18),
            vision_depth=cfg["method"].get("vision_depth", 1),
            fusion_count=cfg["method"].get("fusion_count", 1),
            freeze_strategy=cfg["method"].get("freeze_strategy", "all_but_single"),
            branch_type=cfg["method"].get("branch_type", "transformer"),
            branch_init=cfg["method"].get("branch_init", "random"),
            lora_rank=cfg["method"].get("lora_rank", 16),
            lora_alpha=cfg["method"].get("lora_alpha", 32.0),
        )

    @classmethod
    def from_trained(cls, checkpoint_dir: str) -> "TokenDiversionX3FusionLlava":
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
        method_name = cfg.get("method", {}).get("name", "token_diversion_x3_fusion")
        target_cls = _method_registry.get(method_name, cls)
        model = target_cls.from_config(cfg)

        index_file = ckpt / "model.safetensors.index.json"
        sf_file = ckpt / "model.safetensors"
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

    # -------------------------------------------------------- generation glue
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
        is_decode_step = cache_position is not None and cache_position[0] != 0
        if is_decode_step and getattr(past_key_values, _DPVR_LF_TEXT_LEN, None) is not None:
            # The cache already holds the prompt: feed only the new tokens.
            input_ids = input_ids[:, -cache_position.shape[0]:]
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "cache_position": cache_position,
        }
        if not is_decode_step:
            model_inputs["pixel_values"] = pixel_values
        return model_inputs

    # ---------------------------------------------------------------- forward
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

        lf_cached_decode = (
            use_cache
            and pixel_values is None
            and getattr(past_key_values, _DPVR_LF_TEXT_LEN, None) is not None
        )

        if attention_mask is not None and cache_position is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(~attention_mask, 0)
        elif attention_mask is not None and lf_cached_decode:
            # Same position ids the prefill would assign to these tokens.
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(~attention_mask, 0)
            position_ids = position_ids[:, -inputs_embeds.shape[1]:]
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

            image_token_mask_2d = image_token_mask.any(dim=-1)  # (B, S)
        else:
            out_image = None
            image_token_mask_2d = None

        # ── Deep layers: X3-fusion-last split ────────────────────────────────
        end_idx = len(language_model.layers)
        K = self.fusion_count
        fusion_start = end_idx - K  # first index of the fusion stack

        if has_image:
            text_mask_2d = ~image_token_mask_2d  # (B, S)
            T_txt = S - max_img  # uniform: every batch row has same count

            # Extract text hidden + original position_ids (preserve RoPE indices)
            text_h = hidden_states[text_mask_2d].view(B, T_txt, H)

            full_pos = position_ids.expand(B, S) if position_ids.dim() == 1 else position_ids
            text_pos_ids_orig = full_pos[text_mask_2d].view(B, T_txt)

            # Recompute position_embeddings for text-only subseq using ORIGINAL
            # position_ids — RoPE indices must match the fusion layers'.
            text_pos_emb = language_model.rotary_emb(text_h, text_pos_ids_orig)

            # Text-only causal mask
            text_attn_mask = _build_text_only_causal_mask(
                B, T_txt, hidden_states.device, hidden_states.dtype
            )

            # Text-only deep loop: split_layer..fusion_start-1. With use_cache,
            # these layers cache text-only K/V for the decode steps.
            text_cache = past_key_values if use_cache else None
            text_cache_pos = (
                torch.arange(T_txt, device=hidden_states.device) if use_cache else None
            )
            for i in range(self.split_layer, fusion_start):
                layer = language_model.layers[i]
                out = layer(
                    text_h,
                    position_embeddings=text_pos_emb,
                    attention_mask=text_attn_mask,
                    past_key_values=text_cache,
                    cache_position=text_cache_pos,
                    output_attentions=False,
                    use_cache=use_cache,
                )
                text_h = out[0] if isinstance(out, (tuple, list)) else out
            if use_cache and past_key_values is not None:
                setattr(past_key_values, _DPVR_LF_TEXT_LEN, T_txt)

            # Reassemble before fusion stack; OVERWRITE image positions with
            # out_image so the FIRST fusion layer's K/V projection on image
            # positions sees single_transformer output (gradient path source).
            hidden_states_new = hidden_states.clone()
            hidden_states_new[text_mask_2d] = text_h.reshape(-1, H)
            hidden_states_new[image_token_mask_2d] = out_image.reshape(-1, H)
            hidden_states = hidden_states_new

            # Fusion stack (last K layers): full vanilla forward, image+text fused.
            for i in range(fusion_start, end_idx):
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
        elif lf_cached_decode:
            # Cached decode step: the text-only deep layers see only cached text
            # K/V (as in the prefill); the fusion layers see the full context.
            B, q_len, _ = hidden_states.shape
            if fusion_start > self.split_layer:
                text_past = past_key_values.get_seq_length(self.split_layer)
                text_attn_mask = _build_text_decode_mask(
                    B, q_len, text_past, hidden_states.device, hidden_states.dtype
                )
                text_cache_pos = torch.arange(
                    text_past, text_past + q_len, device=hidden_states.device
                )
            for i in range(self.split_layer, fusion_start):
                layer = language_model.layers[i]
                out = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=text_attn_mask,
                    past_key_values=past_key_values,
                    cache_position=text_cache_pos,
                    output_attentions=False,
                    use_cache=True,
                )
                hidden_states = out[0] if isinstance(out, (tuple, list)) else out
            for i in range(fusion_start, end_idx):
                layer = language_model.layers[i]
                out = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    output_attentions=False,
                    use_cache=True,
                )
                hidden_states = out[0] if isinstance(out, (tuple, list)) else out
        else:
            # DECODING / no-image: vanilla deep forward (identical to Method A fallback)
            for i in range(self.split_layer, end_idx):
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

        final_hidden_states = language_model.norm(hidden_states)
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
        fusion_count: int = 1,
        branch_type: str = "transformer",
        branch_init: str = "random",
        lora_rank: int = 4,
    ) -> "TokenDiversionX3FusionLlava":
        """Build a random-weight instance for CPU unit tests (no pretrained loading).

        Mirrors TokenDiversionSubstitutionLlava._build_test_instance: minimal
        LlamaModel + mock LLaVA wrapper. _mock_prepare_multimodal_input treats
        first S//2 positions as image tokens.

        With n_layers=4, split_layer=2: 2 deep layers total. fusion_count=1 → 1
        text-only + 1 fusion. fusion_count=2 → 0 text-only + 2 fusion.
        With split_layer=2, n_layers=5: 3 deep. fusion_count=2 → 1 text-only +
        2 fusion (the canonical last-2 ablation shape).
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

        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.config = llama_cfg
        obj.name_or_path = ""
        obj._is_hf_initialized = True

        lm = LlamaModel(llama_cfg)

        embed = nn.Embedding(vocab, hidden_size)
        lm_head = nn.Linear(hidden_size, vocab, bias=False)

        def _mock_prepare_multimodal_input(input_ids, pixel_values=None):
            B, S = input_ids.shape
            inputs_embeds = embed(input_ids).float()
            if pixel_values is not None:
                T_img = S // 2
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
        obj.fusion_count = int(fusion_count)
        n_deep = n_layers - split_layer
        if not (1 <= obj.fusion_count <= n_deep):
            raise ValueError(
                f"fusion_count={obj.fusion_count} must be in [1, {n_deep}]"
            )
        obj.branch_type = branch_type
        obj.single_transformer = build_side_branch(
            llama_cfg, split_layer, depth=vision_depth, branch_type=branch_type,
            branch_init=branch_init, source_layers=lm.layers, lora_rank=lora_rank,
            lora_alpha=2 * lora_rank, dtype=torch.float32,
        )

        return obj

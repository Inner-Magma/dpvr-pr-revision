"""Token-Diversion model: vision tokens branch onto a lightweight path at split_layer.

Training and inference are unified in a single forward:
- Training / PREFILL:  use_cache=False (default) or use_cache=True with no prior cache.
- Autoregressive DECODING: use_cache=True + populated past_key_values; image_token_mask
  is None so the light branch is skipped entirely.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from .modeling_llama import LlamaModel
from .modeling_llava import LlavaForConditionalGeneration
from .registry import register_method


@register_method("token_diversion")
class TokenDiversionLlava(PreTrainedModel, GenerationMixin):
    """LLaVA with a dual-path token router.

    Shallow layers (0 … split_layer-1) are frozen and shared across modalities.
    At ``split_layer`` image tokens are pulled out of the main stream, pushed
    through ``single_transformer`` (the only trainable submodule), and written
    back into the corresponding positions before each remaining deep layer runs.

    DECODING (autoregressive generation after PREFILL):
        ``pixel_values`` is None, so ``image_token_mask`` is None and the light
        branch is skipped.  All layers run with the accumulated KV-cache.
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

        # Load full packed LLaVA model (language model + vision tower + projector).
        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True,
        )

        # Optionally swap language-model weights from a separate Vicuna/LLaMA checkpoint.
        if llama_path is not None and Path(llama_path).exists():
            language_model = LlamaModel.from_pretrained(
                llama_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            )
            self.llava_model.model.language_model.layers = language_model.layers
            self.llava_model.model.language_model.norm = language_model.norm
            self.llava_model.model.language_model.rotary_emb = language_model.rotary_emb

        # Optionally swap vision-tower weights.
        if vision_tower_path is not None and Path(vision_tower_path).exists():
            from transformers import CLIPVisionModel
            vision_tower = CLIPVisionModel.from_pretrained(
                vision_tower_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            )
            self.llava_model.model.vision_tower = vision_tower

        # Optionally load projector weights from a legacy mm_projector.bin.
        if projector_weights_path is not None and Path(projector_weights_path).exists():
            projector_ckpt = torch.load(projector_weights_path, map_location="cpu")
            rename_map = {
                "model.mm_projector.0.weight": "linear_1.weight",
                "model.mm_projector.0.bias": "linear_1.bias",
                "model.mm_projector.2.weight": "linear_2.weight",
                "model.mm_projector.2.bias": "linear_2.bias",
            }
            proj_state = {
                rename_map[k]: v for k, v in projector_ckpt.items() if k in rename_map
            }
            self.llava_model.multi_modal_projector.load_state_dict(proj_state, strict=True)

        from ..training.freezing import apply_llava_freeze_strategy

        self.split_layer = split_layer
        self.vision_depth = vision_depth
        llama_cfg = self.llava_model.language_model.config

        # Light-branch: a stack of vision_depth decoder layers (P2-T8 ablation).
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

    # ---------------------------------------------------------- registry glue
    @classmethod
    def from_config(cls, cfg: dict) -> "TokenDiversionLlava":
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
    def from_trained(cls, checkpoint_dir: str) -> "TokenDiversionLlava":
        """Load a saved TokenDiversionLlava checkpoint produced by scripts/train.py.

        Expects checkpoint_dir to contain:
          - training_config.json  (written by ConfigSaveCallback in trainer.py)
          - model weights in one of: pytorch_model.bin, model.safetensors,
            or sharded model.safetensors.index.json + model-xxxxx.safetensors

        The checkpoint is a full-model save from HF Trainer; the state_dict keys
        carry the TokenDiversionLlava prefix (llava_model.* / single_transformer.*).
        We build a fresh model from the original base_model_path (from config), then
        overwrite all keys from the checkpoint — this correctly restores both the
        frozen LLaVA weights and the trained single_transformer.
        """
        import json as _json
        ckpt = Path(checkpoint_dir)
        cfg_file = ckpt / "training_config.json"
        if cfg_file.exists():
            cfg = _json.loads(cfg_file.read_text())
        else:
            raise FileNotFoundError(
                f"training_config.json not found in {checkpoint_dir}. "
                "This file is written by ConfigSaveCallback starting from "
                "the next training run.  For the current checkpoint, create "
                "it manually: {\"model\": {\"base_model_path\": <path>}, "
                "\"method\": {\"name\": \"token_diversion\", \"split_layer\": 18, "
                "\"vision_depth\": 1, \"freeze_strategy\": \"all_but_single\"}}"
            )

        # Dispatch to the correct model class based on method name.
        # Importing dpvr.models ensures all @register_method decorators have run.
        import dpvr.models as _dpvr_models  # noqa: F401
        from .registry import _REGISTRY as _method_registry
        method_name = cfg.get("method", {}).get("name", "token_diversion")
        target_cls = _method_registry.get(method_name, cls)
        model = target_cls.from_config(cfg)

        # Determine which state-dict format the checkpoint uses.
        index_file = ckpt / "model.safetensors.index.json"
        sf_file    = ckpt / "model.safetensors"
        state_file = ckpt / "pytorch_model.bin"

        if index_file.exists():
            # Sharded safetensors (HF Trainer default for models >10 GB).
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

        # full_ft checkpoints save weights under llava_model.* keys inside a
        # FullFTLlava wrapper. LlavaHf.generate_until expects a native
        # LlavaForConditionalGeneration, so unwrap here after loading weights.
        if method_name == "full_ft" and hasattr(model, "llava_model"):
            return model.llava_model

        return model

    # ------------------------------------------------------------------ utils
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
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "cache_position": cache_position,
        }
        # Only pass pixel_values on the first step (PREFILL).
        if cache_position is None or cache_position[0] == 0:
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
        # Convert {0,1} int mask to bool; leave None as-is.
        if attention_mask is not None:
            attention_mask = attention_mask == 1

        inputs_embeds, image_token_mask = self.llava_model.model.prepare_multimodal_input(
            input_ids, pixel_values
        )

        language_model = self.llava_model.language_model

        # Build position_ids from attention_mask (training) or let prepare_input_mine
        # derive it from cache_position (DECODING).
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
            B, _, H = hidden_states.shape
            # image_token_mask is 3D (B, seq_len, H) — expanded inside
            # prepare_multimodal_input.  Sum over seq_len dim gives (B, H);
            # every column has the same count so .max() == num_image_tokens.
            max_img = int(image_token_mask.sum(dim=1).max().item())

            # Extract image-token vectors into a compact (B, max_img, H) tensor.
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
            # Reduce 3D mask to 2D (B, seq_len) for positional write-back.
            image_token_mask_2d = image_token_mask.any(dim=-1)
        else:
            out_image = None
            image_token_mask_2d = None

        # ── Deep layers (split_layer … end) ──────────────────────────────────
        # Before each deep layer, overwrite image-token positions with the light-
        # branch output so image tokens contribute as stable context while text
        # tokens evolve through the deep path.
        for i in range(self.split_layer, len(language_model.layers)):
            if out_image is not None:
                hidden_states[image_token_mask_2d] = out_image.reshape(-1, hidden_states.shape[-1])
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

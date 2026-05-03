"""
Vill -- Complete Transformer Model
-------------------------------------
Assembles all components into the full Vill decoder-only language model.
Supports both dense and Mixture-of-Experts configurations.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from vill.model.config import VillConfig
from vill.model.components import (
    RMSNorm,
    GroupedQueryAttention,
    SwiGLUFeedForward,
    MoEFeedForward,
    precompute_rope_frequencies,
)


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block.

    Structure:
        x -> RMSNorm -> GQA Attention -> residual
          -> RMSNorm -> FFN (or MoE)   -> residual
    """

    def __init__(self, config: VillConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if config.is_moe:
            self.feed_forward = MoEFeedForward(config)
        else:
            self.feed_forward = SwiGLUFeedForward(config)

        self.is_moe = config.is_moe
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states, new_kv_cache = self.attention(
            hidden_states, cos, sin, attention_mask, kv_cache
        )
        hidden_states = self.residual_dropout(hidden_states) + residual

        # Pre-norm feed-forward
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)

        aux_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.is_moe:
            hidden_states, aux_loss = self.feed_forward(hidden_states)
        else:
            hidden_states = self.feed_forward(hidden_states)

        hidden_states = self.residual_dropout(hidden_states) + residual

        return hidden_states, new_kv_cache, aux_loss


class VillModel(nn.Module):
    """
    The core Vill Transformer (without the language model head).

    Consists of:
    - Token embedding layer
    - N Transformer blocks with GQA + SwiGLU/MoE
    - Final RMSNorm
    """

    def __init__(self, config: VillConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Precompute RoPE frequency tables
        cos, sin = precompute_rope_frequencies(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        # Build causal attention mask
        if attention_mask is None and seq_len > 1:
            attention_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)

        # Move RoPE buffers to correct device
        cos = self.rope_cos.to(hidden_states.device)
        sin = self.rope_sin.to(hidden_states.device)

        new_kv_caches = []
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device)

        for i, layer in enumerate(self.layers):
            layer_kv = kv_caches[i] if kv_caches is not None else None
            hidden_states, new_kv, aux_loss = layer(
                hidden_states, cos, sin, attention_mask, layer_kv
            )
            new_kv_caches.append(new_kv)
            total_aux_loss = total_aux_loss + aux_loss

        hidden_states = self.norm(hidden_states)
        return hidden_states, new_kv_caches, total_aux_loss


class VillForCausalLM(nn.Module):
    """
    Vill language model with a causal language modeling head.

    This is the complete model used for both pre-training (next-token
    prediction) and inference (text generation).
    """

    def __init__(self, config: VillConfig):
        super().__init__()
        self.config = config
        self.model = VillModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> dict:
        hidden_states, new_kv_caches, aux_loss = self.model(
            input_ids, attention_mask, kv_caches
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if self.config.is_moe:
                loss = loss + aux_loss

        return {
            "logits": logits,
            "loss": loss,
            "kv_caches": new_kv_caches,
            "aux_loss": aux_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with top-p (nucleus) sampling.

        Args:
            input_ids: Prompt token IDs of shape (1, seq_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature. Lower = more deterministic.
            top_p: Nucleus sampling threshold.
            top_k: Top-K filtering.
            eos_token_id: Token ID that signals end of generation.

        Returns:
            Generated token IDs including the prompt.
        """
        self.eval()
        kv_caches = None
        generated = input_ids

        for _ in range(max_new_tokens):
            if kv_caches is not None:
                current_input = generated[:, -1:]
            else:
                current_input = generated

            outputs = self.forward(current_input, kv_caches=kv_caches)
            logits = outputs["logits"][:, -1, :]
            kv_caches = outputs["kv_caches"]

            if temperature > 0:
                logits = logits / temperature

                # Top-K filtering
                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_k_vals[:, -1:]] = float("-inf")

                # Top-P (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[mask] = float("-inf")
                    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

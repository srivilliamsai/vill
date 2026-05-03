"""
Vill -- Transformer Components
--------------------------------
Core building blocks: RMSNorm, RoPE, Grouped-Query Attention,
SwiGLU Feed-Forward, and Mixture of Experts.

Each component is implemented from first principles in PyTorch,
following the same design used by Llama 3, Qwen 2.5, and Mistral.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vill.model.config import VillConfig


# ---------------------------------------------------------------------------
# RMSNorm -- Root Mean Square Layer Normalization
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (Zhang & Sennrich, 2019).

    Preferred over LayerNorm in modern LLMs for its computational
    efficiency. Omits the mean-centering step and normalizes by
    the root mean square of activations only.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cosine and sine tables for Rotary Positional Embeddings.

    RoPE encodes absolute position through rotation of query/key vectors
    in pairs of dimensions. The relative position information emerges
    naturally from the dot product of rotated vectors.

    Args:
        head_dim: Dimension of each attention head.
        max_seq_len: Maximum sequence length to precompute.
        theta: Base frequency (higher values extend effective context).
        device: Target device for the tensors.

    Returns:
        Tuple of (cos, sin) tensors with shape (max_seq_len, head_dim).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    # Outer product: (seq_len, head_dim // 2)
    angles = torch.outer(positions, freqs)
    # Duplicate to cover full head_dim: (seq_len, head_dim)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def apply_rotary_embeddings(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary positional embeddings to a tensor.

    Args:
        x: Input tensor of shape (batch, num_heads, seq_len, head_dim).
        cos: Cosine table of shape (seq_len, head_dim).
        sin: Sine table of shape (seq_len, head_dim).

    Returns:
        Rotated tensor of the same shape.
    """
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # Split into even and odd pairs, then rotate
    x_even = x[..., : x.shape[-1] // 2]
    x_odd = x[..., x.shape[-1] // 2 :]
    x_rotated = torch.cat([-x_odd, x_even], dim=-1)

    return x * cos + x_rotated * sin


# ---------------------------------------------------------------------------
# Grouped-Query Attention (GQA)
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (Ainslie et al., 2023).

    Instead of maintaining separate Key and Value heads for each Query
    head, GQA shares KV heads across groups of query heads. This reduces
    the KV-cache size at inference and memory bandwidth requirements
    without significant quality loss.

    When num_kv_heads == num_attention_heads: standard Multi-Head Attention.
    When num_kv_heads == 1: Multi-Query Attention.
    """

    def __init__(self, config: VillConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Project queries, keys, values
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose to (batch, heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply rotary embeddings to Q and K
        q = apply_rotary_embeddings(q, cos, sin)
        k = apply_rotary_embeddings(k, cos, sin)

        # Handle KV-cache for autoregressive generation
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv_cache = (k, v)

        # Expand KV heads to match query head count (GQA)
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(batch_size, self.num_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(batch_size, self.num_heads, -1, self.head_dim)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)

        return self.o_proj(attn_output), new_kv_cache


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network (Shazeer, 2020).

    Combines a gating mechanism with the SiLU activation function.
    Outperforms standard ReLU/GELU FFN in most LLM benchmarks.

    The gate and up projections produce two separate transformations;
    the gate projection is activated with SiLU and element-wise multiplied
    with the up projection. The result is then projected back down.
    """

    def __init__(self, config: VillConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Mixture of Experts (MoE)
# ---------------------------------------------------------------------------

class MoERouter(nn.Module):
    """
    Top-K router for Mixture of Experts.

    For each token, the router computes a probability distribution over
    all experts and selects the top-K experts to process that token.
    Includes an auxiliary load-balancing loss to prevent expert collapse.
    """

    def __init__(self, config: VillConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.aux_loss_coeff = config.moe_aux_loss_coeff

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch * seq_len, hidden_size)
        Returns:
            router_weights: (batch * seq_len, top_k) -- normalized weights
            selected_experts: (batch * seq_len, top_k) -- expert indices
            aux_loss: scalar load-balancing loss
        """
        logits = self.gate(hidden_states)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)

        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # re-normalize

        # Auxiliary load-balancing loss
        # Encourages uniform distribution of tokens across experts
        tokens_per_expert = torch.zeros(self.num_experts, device=logits.device)
        for i in range(self.num_experts):
            tokens_per_expert[i] = (indices == i).float().sum()
        tokens_per_expert = tokens_per_expert / hidden_states.shape[0]
        avg_probs = probs.mean(dim=0)
        aux_loss = self.aux_loss_coeff * self.num_experts * (tokens_per_expert * avg_probs).sum()

        return weights.to(hidden_states.dtype), indices, aux_loss


class MoEFeedForward(nn.Module):
    """
    Mixture of Experts Feed-Forward layer.

    Contains N independent SwiGLU FFN experts. A router selects the
    top-K experts for each token. The final output is the weighted
    sum of the selected experts' outputs.
    """

    def __init__(self, config: VillConfig):
        super().__init__()
        self.router = MoERouter(config)
        self.experts = nn.ModuleList([
            SwiGLUFeedForward(config) for _ in range(config.num_experts)
        ])
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)

        weights, indices, aux_loss = self.router(flat)
        output = torch.zeros_like(flat)

        for i in range(self.num_experts):
            # Find which tokens selected this expert and in which top-k slot
            for k in range(self.top_k):
                mask = (indices[:, k] == i)
                if mask.any():
                    expert_input = flat[mask]
                    expert_output = self.experts[i](expert_input)
                    output[mask] += weights[mask, k].unsqueeze(-1) * expert_output

        return output.reshape(batch_size, seq_len, hidden_dim), aux_loss

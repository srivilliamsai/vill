"""
Vill -- Model Configuration
----------------------------
Defines all hyperparameters for the Vill Transformer architecture.
Provides preset configurations at multiple scales.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class VillConfig:
    """
    Configuration for the Vill language model.

    This mirrors the structure used by Llama, Mistral, and Qwen,
    with support for Grouped-Query Attention (GQA), Rotary Positional
    Embeddings (RoPE), SwiGLU activation, and optional Mixture of Experts.
    """

    # -- Vocabulary and Embedding --
    vocab_size: int = 32000
    hidden_size: int = 768
    max_position_embeddings: int = 2048

    # -- Transformer Layers --
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4       # GQA: fewer KV heads than query heads
    head_dim: Optional[int] = None     # Computed if not set

    # -- Feed-Forward Network --
    intermediate_size: int = 2048
    hidden_act: str = "silu"           # SwiGLU uses SiLU internally

    # -- Normalization --
    rms_norm_eps: float = 1e-6

    # -- Positional Encoding --
    rope_theta: float = 10000.0

    # -- Mixture of Experts (Optional) --
    num_experts: int = 1               # 1 = dense model, >1 = MoE
    num_experts_per_tok: int = 1       # Top-K experts activated per token
    moe_aux_loss_coeff: float = 0.01   # Load balancing coefficient

    # -- Training --
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02

    # -- Dropout (zero for pre-training, nonzero for fine-tuning) --
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0

    # -- Metadata --
    model_type: str = "vill"

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) must be <= "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible "
                f"by num_key_value_heads ({self.num_key_value_heads})"
            )

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 1

    @property
    def num_parameters_estimate(self) -> int:
        """Rough parameter count estimate."""
        embed = self.vocab_size * self.hidden_size * 2  # input + output
        attn_per_layer = (
            self.hidden_size * self.head_dim * self.num_attention_heads  # Q
            + self.hidden_size * self.head_dim * self.num_key_value_heads  # K
            + self.hidden_size * self.head_dim * self.num_key_value_heads  # V
            + self.hidden_size * self.hidden_size  # O
        )
        ffn_per_layer = 3 * self.hidden_size * self.intermediate_size  # SwiGLU has 3 matrices
        if self.is_moe:
            ffn_per_layer = ffn_per_layer * self.num_experts
        norm_per_layer = 2 * self.hidden_size
        per_layer = attn_per_layer + ffn_per_layer + norm_per_layer
        total = embed + per_layer * self.num_hidden_layers + self.hidden_size
        return total

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "VillConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# Preset Configurations
# ---------------------------------------------------------------------------

def vill_nano() -> VillConfig:
    """
    Vill-Nano: 150M parameters.
    Trainable on a single consumer GPU or Apple M1 with 8GB RAM.
    """
    return VillConfig(
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=2048,
        max_position_embeddings=2048,
    )


def vill_small() -> VillConfig:
    """
    Vill-Small: 1.5B parameters.
    Suitable for training on a single A100 or T4 GPU (Google Colab).
    """
    return VillConfig(
        vocab_size=32000,
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=5504,
        max_position_embeddings=4096,
    )


def vill_medium() -> VillConfig:
    """
    Vill-Medium: 7B parameters.
    Requires multi-GPU setup (e.g., 4x A100).
    """
    return VillConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=11008,
        max_position_embeddings=8192,
        rope_theta=500000.0,
    )


def vill_large_moe() -> VillConfig:
    """
    Vill-Large: 70B total parameters (MoE), ~13B active per token.
    Requires TPU pod or large GPU cluster.
    """
    return VillConfig(
        vocab_size=64000,
        hidden_size=4096,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        max_position_embeddings=32768,
        rope_theta=1000000.0,
        num_experts=16,
        num_experts_per_tok=2,
    )


PRESET_CONFIGS = {
    "vill-nano": vill_nano,
    "vill-small": vill_small,
    "vill-medium": vill_medium,
    "vill-large-moe": vill_large_moe,
}


def get_config(name: str) -> VillConfig:
    if name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[name]()

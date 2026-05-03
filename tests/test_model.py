"""
Vill -- Unit Tests
--------------------
Tests for model architecture, configuration, and forward pass.
"""

import pytest
import torch

from vill.model.config import VillConfig, get_config, vill_nano
from vill.model.components import (
    RMSNorm,
    precompute_rope_frequencies,
    apply_rotary_embeddings,
    GroupedQueryAttention,
    SwiGLUFeedForward,
)
from vill.model.transformer import VillForCausalLM, TransformerBlock


class TestConfig:

    def test_nano_config_valid(self):
        config = vill_nano()
        assert config.vocab_size == 32000
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.head_dim == 64

    def test_gqa_constraint(self):
        with pytest.raises(ValueError):
            VillConfig(num_attention_heads=12, num_key_value_heads=5)

    def test_all_presets_valid(self):
        for name in ["vill-nano", "vill-small", "vill-medium", "vill-large-moe"]:
            config = get_config(name)
            assert config.hidden_size > 0
            assert config.num_hidden_layers > 0

    def test_parameter_estimate(self):
        config = vill_nano()
        estimate = config.num_parameters_estimate
        assert estimate > 100_000_000  # Should be > 100M

    def test_moe_detection(self):
        dense = vill_nano()
        assert not dense.is_moe
        moe = get_config("vill-large-moe")
        assert moe.is_moe


class TestComponents:

    def test_rmsnorm_shape(self):
        norm = RMSNorm(768)
        x = torch.randn(2, 16, 768)
        out = norm(x)
        assert out.shape == x.shape

    def test_rmsnorm_normalized(self):
        norm = RMSNorm(768)
        x = torch.randn(2, 16, 768)
        out = norm(x)
        rms = out.float().pow(2).mean(-1).sqrt()
        assert rms.mean().item() < 5.0  # Reasonable magnitude

    def test_rope_frequencies_shape(self):
        cos, sin = precompute_rope_frequencies(64, 2048)
        assert cos.shape == (2048, 64)
        assert sin.shape == (2048, 64)

    def test_rope_apply(self):
        cos, sin = precompute_rope_frequencies(64, 2048)
        x = torch.randn(2, 12, 128, 64)
        out = apply_rotary_embeddings(x, cos, sin)
        assert out.shape == x.shape

    def test_swiglu_shape(self):
        config = vill_nano()
        ffn = SwiGLUFeedForward(config)
        x = torch.randn(2, 16, config.hidden_size)
        out = ffn(x)
        assert out.shape == x.shape

    def test_gqa_shape(self):
        config = vill_nano()
        attn = GroupedQueryAttention(config)
        cos, sin = precompute_rope_frequencies(config.head_dim, config.max_position_embeddings)
        x = torch.randn(2, 16, config.hidden_size)
        out, kv = attn(x, cos, sin)
        assert out.shape == x.shape


class TestModel:

    @pytest.fixture
    def small_config(self):
        return VillConfig(
            vocab_size=256,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            max_position_embeddings=64,
        )

    def test_forward_pass(self, small_config):
        model = VillForCausalLM(small_config)
        input_ids = torch.randint(0, 256, (2, 32))
        labels = torch.randint(0, 256, (2, 32))
        outputs = model(input_ids=input_ids, labels=labels)
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (2, 32, 256)
        assert outputs["loss"].item() > 0

    def test_generate(self, small_config):
        model = VillForCausalLM(small_config)
        input_ids = torch.randint(0, 256, (1, 8))
        output = model.generate(input_ids, max_new_tokens=16, temperature=1.0)
        assert output.shape[1] > 8
        assert output.shape[1] <= 24

    def test_parameter_count(self, small_config):
        model = VillForCausalLM(small_config)
        count = model.count_parameters()
        assert count > 0
        assert count == model.count_trainable_parameters()

    def test_moe_forward(self):
        config = VillConfig(
            vocab_size=256,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            max_position_embeddings=64,
            num_experts=4,
            num_experts_per_tok=2,
        )
        model = VillForCausalLM(config)
        input_ids = torch.randint(0, 256, (2, 16))
        labels = torch.randint(0, 256, (2, 16))
        outputs = model(input_ids=input_ids, labels=labels)
        assert outputs["loss"].item() > 0
        assert outputs["aux_loss"].item() >= 0

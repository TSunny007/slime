"""CPU unit tests for ``slime.utils.flops_utils``.

Verifies the FLOPs calculation helpers used for throughput reporting.
All functions are pure arithmetic on model-shape parameters, so they
run on CPU without any model weights.
"""

from __future__ import annotations

import argparse

import pytest

from slime.utils.flops_utils import (
    calculate_attention_flops,
    calculate_embedding_flops,
    calculate_fwd_flops,
    calculate_layer_flops,
    calculate_lm_head_flops,
    calculate_mlp_flops,
    calculate_output_flops,
    calculate_qkv_projection_flops,
)

NUM_GPUS = 0


def _make_dense_args(
    hidden_size=4096,
    num_attention_heads=32,
    num_query_groups=32,
    kv_channels=128,
    ffn_hidden_size=11008,
    num_layers=32,
    vocab_size=32000,
    num_experts=None,
):
    return argparse.Namespace(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        kv_channels=kv_channels,
        ffn_hidden_size=ffn_hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        num_experts=num_experts,
        multi_latent_attention=False,
    )


def _make_mla_args(
    hidden_size=4096,
    num_attention_heads=32,
    num_query_groups=32,
    kv_channels=128,
    q_lora_rank=1536,
    qk_head_dim=128,
    qk_pos_emb_head_dim=64,
    kv_lora_rank=512,
    v_head_dim=128,
    ffn_hidden_size=11008,
    num_layers=32,
    vocab_size=32000,
    num_experts=None,
):
    return argparse.Namespace(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        kv_channels=kv_channels,
        q_lora_rank=q_lora_rank,
        qk_head_dim=qk_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        kv_lora_rank=kv_lora_rank,
        v_head_dim=v_head_dim,
        ffn_hidden_size=ffn_hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        num_experts=num_experts,
        multi_latent_attention=True,
    )


# ---------------------------------------------------------------------------
# Leaf functions
# ---------------------------------------------------------------------------


class TestEmbeddingFlops:
    def test_formula(self):
        assert calculate_embedding_flops(512, 4096) == 2 * 512 * 4096


class TestLmHeadFlops:
    def test_formula(self):
        assert calculate_lm_head_flops(512, 4096, 32000) == 2 * 512 * 4096 * 32000


class TestOutputFlops:
    def test_formula(self):
        assert calculate_output_flops(512, 4096) == 2 * 512 * 4096 * 4096


class TestMlpFlops:
    def test_formula(self):
        assert calculate_mlp_flops(512, 4096, 11008) == 2 * 512 * 4096 * 11008 * 3


# ---------------------------------------------------------------------------
# QKV projection (dense vs MLA)
# ---------------------------------------------------------------------------


class TestQkvProjectionFlops:
    def test_dense(self):
        args = _make_dense_args()
        flops = calculate_qkv_projection_flops(
            args, seqlen=512, hidden_size=4096, num_attention_heads=32, num_query_groups=32
        )
        q_flops = 2 * 512 * 4096 * 32 * 128
        kv_flops = 2 * 2 * 512 * 4096 * 32 * 128
        assert flops == q_flops + kv_flops

    def test_mla_with_lora(self):
        args = _make_mla_args()
        flops = calculate_qkv_projection_flops(
            args, seqlen=512, hidden_size=4096, num_attention_heads=32, num_query_groups=32
        )
        assert flops > 0


# ---------------------------------------------------------------------------
# Attention flops
# ---------------------------------------------------------------------------


class TestAttentionFlops:
    def test_dense_attention(self):
        args = _make_dense_args()
        flops = calculate_attention_flops(args, seqlen=512, num_attention_heads=32)
        # QK^T with causal: 2 * 32 * 512 * 512 * 128 / 2
        # A*V: 32 * 512 * 512 * 128
        qkt = 2 * 32 * 512 * 512 * 128 / 2
        av = 32 * 512 * 512 * 128
        assert flops == qkt + av

    def test_mla_attention(self):
        args = _make_mla_args()
        flops = calculate_attention_flops(args, seqlen=512, num_attention_heads=32)
        assert flops > 0


# ---------------------------------------------------------------------------
# Full layer and forward pass
# ---------------------------------------------------------------------------


class TestLayerFlops:
    def test_positive(self):
        args = _make_dense_args()
        flops = calculate_layer_flops(
            args,
            seqlen=512,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=32,
            ffn_hidden_size=11008,
        )
        assert flops > 0


class TestFwdFlops:
    def test_dense_model(self):
        args = _make_dense_args()
        flops = calculate_fwd_flops([512, 1024], args)
        assert flops > 0

    def test_moe_model_with_freq_int(self):
        args = argparse.Namespace(
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=32,
            kv_channels=128,
            ffn_hidden_size=11008,
            num_layers=8,
            vocab_size=32000,
            num_experts=8,
            moe_ffn_hidden_size=1408,
            moe_router_topk=2,
            moe_shared_expert_intermediate_size=0,
            moe_layer_freq=2,
            multi_latent_attention=False,
        )
        flops = calculate_fwd_flops([512], args)
        assert flops > 0

    def test_moe_model_with_freq_list(self):
        args = argparse.Namespace(
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=32,
            kv_channels=128,
            ffn_hidden_size=11008,
            num_layers=4,
            vocab_size=32000,
            num_experts=8,
            moe_ffn_hidden_size=1408,
            moe_router_topk=2,
            moe_shared_expert_intermediate_size=0,
            moe_layer_freq=[0, 1, 0, 1],
            multi_latent_attention=False,
        )
        flops = calculate_fwd_flops([256], args)
        assert flops > 0

    def test_moe_no_freq_attr(self):
        args = argparse.Namespace(
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=32,
            kv_channels=128,
            ffn_hidden_size=11008,
            num_layers=4,
            vocab_size=32000,
            num_experts=8,
            moe_ffn_hidden_size=1408,
            moe_router_topk=2,
            multi_latent_attention=False,
        )
        flops = calculate_fwd_flops([128], args)
        assert flops > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

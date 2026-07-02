"""Shared utilities for Megatron-to-HuggingFace weight conversion.

These helpers extract the duplicated tensor-manipulation patterns that appear
across all model-specific converter modules (qwen2, llama, glm4, deepseekv3,
etc.) into a single location.  Each function owns one piece of the conversion
math so that bug-fixes and optimisations propagate uniformly.
"""

import re

import torch

# Regex shared by all converters for matching decoder layer parameters.
DECODER_LAYERS_PATTERN = re.compile(r"module\.module\.decoder\.layers\.(\d+)\.(.+)")


def get_head_dim(args) -> int:
    """Return the per-head dimension from args."""
    try:
        kv_channels = args.kv_channels
    except AttributeError:
        kv_channels = None
    if kv_channels is not None:
        return kv_channels
    return args.hidden_size // args.num_attention_heads


def get_value_num_per_group(args) -> int:
    """Return the number of value heads per query group."""
    return args.num_attention_heads // args.num_query_groups


def split_qkv_weight(param: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a fused QKV weight into separate Q, K, V projections.

    Expects ``param`` shaped for grouped-query attention:
    ``(num_query_groups * (value_num_per_group + 2) * head_dim, hidden_size)``.
    """
    head_dim = get_head_dim(args)
    value_num_per_group = get_value_num_per_group(args)
    param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
    q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
    q_param = q_param.reshape(-1, args.hidden_size)
    k_param = k_param.reshape(-1, args.hidden_size)
    v_param = v_param.reshape(-1, args.hidden_size)
    return q_param, k_param, v_param


def split_qkv_bias(param: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a fused QKV bias into separate Q, K, V biases."""
    head_dim = get_head_dim(args)
    value_num_per_group = get_value_num_per_group(args)
    param = param.view(args.num_query_groups, -1)
    q_bias, k_bias, v_bias = torch.split(
        param,
        split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
        dim=1,
    )
    return q_bias.contiguous().flatten(), k_bias.contiguous().flatten(), v_bias.contiguous().flatten()


def split_gate_up_weight(param: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a fused gate+up projection weight into (gate, up)."""
    return param.chunk(2, dim=0)


def match_decoder_layer(name: str) -> tuple[str, str] | None:
    """Match a Megatron decoder layer name, returning (layer_idx, rest) or None."""
    m = DECODER_LAYERS_PATTERN.match(name)
    if m:
        return m.group(1), m.group(2)
    return None

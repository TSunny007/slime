import re


from .conversion_utils import match_decoder_layer, split_gate_up_weight, split_qkv_weight


def convert_minimax_m2_to_hf(args, name, param):
    """Convert Megatron parameter names/tensors to HuggingFace format for MiniMax-M2.5.

    HF uses `block_sparse_moe` prefix with expert naming w1(gate)/w2(down)/w3(up).
    Custom SelfAttention uses `q_norm`/`k_norm` (not `q_layernorm`/`k_layernorm`).
    """
    # Direct mappings
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    layer = match_decoder_layer(name)
    if layer:
        layer_idx, rest = layer

        # MoE experts: linear_fc1 -> w1 (gate) + w3 (up), linear_fc2 -> w2 (down)
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = split_gate_up_weight(param)
                return [
                    (f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight", gate_weight),
                    (f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight", up_weight),
                ]
            elif rest == "linear_fc2":
                return [
                    (f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight", param),
                ]
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # Attention: o_proj
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]

        # Attention: fused QKV -> split into Q/K/V (GQA: 48 heads, 8 kv heads)
        elif rest == "self_attention.linear_qkv.weight":
            q_param, k_param, v_param = split_qkv_weight(param, args)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]

        # Input layernorm
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]

        # QK Norm (custom attention uses q_norm/k_norm, NOT q_layernorm/k_layernorm)
        elif rest == "self_attention.q_norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

        # Post-attention layernorm
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

        # Router
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.block_sparse_moe.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.block_sparse_moe.e_score_correction_bias", param)]

    raise ValueError(f"Unknown parameter name: {name}")

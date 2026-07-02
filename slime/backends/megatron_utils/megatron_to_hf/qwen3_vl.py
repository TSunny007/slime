from .conversion_utils import match_decoder_layer, split_gate_up_weight, split_qkv_bias, split_qkv_weight


def convert_qwen3vl_to_hf(args, name, param):
    if name.startswith("module.module.language_model."):
        name = "module.module." + name[len("module.module.language_model.") :]

    # (Optional safety) if you ever see extra "module." prefixes
    while name.startswith("module.module.module."):
        name = name.replace("module.module.module.", "module.module.", 1)

    if name.startswith("module.module.vision_model."):
        hf_name = "model.visual." + name[len("module.module.vision_model.") :]
        return [(hf_name, param)]

    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]

    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]

    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    layer = match_decoder_layer(name)
    if layer:
        layer_idx, rest = layer
        base = f"model.language_model.layers.{layer_idx}"

        if rest == "self_attention.linear_proj.weight":
            return [(f"{base}.self_attn.o_proj.weight", param)]

        elif rest == "self_attention.linear_qkv.weight":
            q_param, k_param, v_param = split_qkv_weight(param, args)
            return [
                (f"{base}.self_attn.q_proj.weight", q_param),
                (f"{base}.self_attn.k_proj.weight", k_param),
                (f"{base}.self_attn.v_proj.weight", v_param),
            ]

        elif rest == "self_attention.linear_qkv.bias":
            q_bias, k_bias, v_bias = split_qkv_bias(param, args)
            return [
                (f"{base}.self_attn.q_proj.bias", q_bias),
                (f"{base}.self_attn.k_proj.bias", k_bias),
                (f"{base}.self_attn.v_proj.bias", v_bias),
            ]

        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = split_gate_up_weight(param)
            return [
                (f"{base}.mlp.gate_proj.weight", gate_weight),
                (f"{base}.mlp.up_proj.weight", up_weight),
            ]

        elif rest == "mlp.linear_fc2.weight":
            return [(f"{base}.mlp.down_proj.weight", param)]

        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"{base}.input_layernorm.weight", param)]

        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"{base}.post_attention_layernorm.weight", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"{base}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"{base}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")

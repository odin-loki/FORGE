#!/usr/bin/env python3
"""Golden end-to-end test for the phi3 template: random tiny models, a float64 numpy reference, and FORGE.

  phi3       fused qkv projection, fused gate-up MLP, full-width RoPE, separate lm_head, grouped-query attention
  phi4mini   the same with partial RoPE (partial_rotary_factor 0.75) and the head tied to the embedding

  1. HF import (f32, q8_0) -> prefill logits and KV-cached greedy decode against the reference, per variant
  2. GGUF import of the phi4mini-style model (general.architecture phi3, rope.dimension_count) -> prefill logits
"""
import os
import numpy as np
import golden_util as gu

BASE = dict(architectures=["Phi3ForCausalLM"], model_type="phi3", hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=96, vocab_size=300, rms_norm_eps=1e-5, rope_theta=10000.0, hidden_act="silu",
            max_position_embeddings=128, bos_token_id=290, eos_token_id=291)
CFGS = {"phi3": dict(BASE, tie_word_embeddings=False), "phi4mini": dict(BASE, tie_word_embeddings=True, partial_rotary_factor=0.75)}

def make_weights(c, seed):
    rng = np.random.default_rng(seed)
    H, nh, nk, F, V = (c[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size")); hd = H // nh
    W = {"model.embed_tokens.weight": rng.normal(0, 0.2, (V, H)), "model.norm.weight": 1 + 0.1 * rng.normal(size=H)}   # small: a tied head must not just echo the input token
    if not c["tie_word_embeddings"]: W["lm_head.weight"] = rng.normal(0, 1 / np.sqrt(H), (V, H))
    for l in range(c["num_hidden_layers"]):
        p = f"model.layers.{l}."
        W[p + "input_layernorm.weight"] = 1 + 0.1 * rng.normal(size=H); W[p + "post_attention_layernorm.weight"] = 1 + 0.1 * rng.normal(size=H)
        W[p + "self_attn.qkv_proj.weight"] = rng.normal(0, 1 / np.sqrt(H), ((nh + 2 * nk) * hd, H))   # rows [q | k | v]
        W[p + "self_attn.o_proj.weight"] = rng.normal(0, 1 / np.sqrt(H), (H, nh * hd))
        W[p + "mlp.gate_up_proj.weight"] = rng.normal(0, 1 / np.sqrt(H), (2 * F, H))                 # rows [gate | up]
        W[p + "mlp.down_proj.weight"] = rng.normal(0, 1 / np.sqrt(F), (H, F))
    return W

def forward(c, W, ids):
    H, nh, nk, F, eps = (c[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size", "rms_norm_eps")); hd = H // nh
    rot = int(hd * c.get("partial_rotary_factor", 1.0)); s = len(ids); pos = np.arange(s, dtype=np.float64)
    prope = lambda t: np.concatenate([gu.rope(t[..., :rot], pos, c["rope_theta"]), t[..., rot:]], -1)
    x = W["model.embed_tokens.weight"][ids]
    for l in range(c["num_hidden_layers"]):
        p = f"model.layers.{l}."
        qkv = gu.rms(x, W[p + "input_layernorm.weight"], eps) @ W[p + "self_attn.qkv_proj.weight"].T
        qd, kd = nh * hd, nk * hd
        q = prope(qkv[:, :qd].reshape(s, nh, hd)); k = prope(qkv[:, qd:qd + kd].reshape(s, nk, hd)); v = qkv[:, qd + kd:].reshape(s, nk, hd)
        x = x + gu.attend(q, k, v, gu.causal_mask(s)).reshape(s, qd) @ W[p + "self_attn.o_proj.weight"].T
        gup = gu.rms(x, W[p + "post_attention_layernorm.weight"], eps) @ W[p + "mlp.gate_up_proj.weight"].T
        x = x + (gup[:, F:] * gu.silu(gup[:, :F])) @ W[p + "mlp.down_proj.weight"].T
    return gu.rms(x, W["model.norm.weight"], eps) @ W.get("lm_head.weight", W["model.embed_tokens.weight"]).T

def gguf_tensors(W):
    T = {"token_embd.weight": W["model.embed_tokens.weight"], "output_norm.weight": W["model.norm.weight"]}
    if "lm_head.weight" in W: T["output.weight"] = W["lm_head.weight"]
    names = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm", "self_attn.qkv_proj": "attn_qkv", "self_attn.o_proj": "attn_output",
             "mlp.gate_up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
    for k, v in W.items():
        if k.startswith("model.layers."): T[f"blk.{k.split('.')[2]}.{names['.'.join(k.split('.')[3:-1])]}.weight"] = v
    return T

def main():
    a = gu.args(); ck = gu.Checker(a.forge, a.work)
    prompt = [290, 72, 101, 108, 108, 111, 44, 32, 116, 104, 101, 32, 119]
    for i, (name, c) in enumerate(CFGS.items()):
        hf = os.path.join(a.work, "hf_" + name)
        W = gu.save_model(hf, make_weights(c, 200 + i), c)
        ck.decoder(hf, name, lambda ids, c=c, W=W: forward(c, W, ids), prompt, c["eos_token_id"])
        if name != "phi4mini": continue
        hd = c["hidden_size"] // c["num_attention_heads"]
        gp = gu.write_gguf(os.path.join(a.work, "phi4mini.gguf"), "phi3",
                           {"context_length": 128, "embedding_length": c["hidden_size"], "block_count": c["num_hidden_layers"],
                            "feed_forward_length": c["intermediate_size"], "attention.head_count": c["num_attention_heads"],
                            "attention.head_count_kv": c["num_key_value_heads"], "attention.layer_norm_rms_epsilon": c["rms_norm_eps"],
                            "rope.dimension_count": int(hd * c["partial_rotary_factor"]), "rope.freq_base": c["rope_theta"]}, gguf_tensors(W))
        if gp: ck.logits(ck.imp(gp, "phi4mini_gguf.fpk", None), prompt, forward(c, W, prompt)[-1], "phi4mini gguf F32")
        else: print("[skip] gguf python package not installed")
    ck.done("phi3")

if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Golden end-to-end test for the gemma / gemma2 / gemma3 templates: random tiny models of the three generations,
one float64 numpy reference parameterised by generation, and FORGE.

  gemma   RMSNorm (1 + w), embeddings * sqrt(d), GeGLU (GELU-tanh), tied head, grouped-query attention
  gemma2  + pre/post norms, attention soft-capping (cap 5: active on these weights), query_pre_attn_scalar,
            sliding window 4 on the even layers, final logit soft-capping
  gemma3  + QK-norm instead of soft-capping, sliding-window pattern 3 with a local RoPE base

  1. HF import (f32, q8_0) -> prefill logits and KV-cached greedy decode against the reference, per generation
  2. GGUF import of the gemma2 model (norm weights stored as 1 + w, as llama.cpp's converter does) -> prefill logits
The prompt (13 tokens) is longer than the sliding window, so local and global layers really differ.
"""
import os
import numpy as np
import golden_util as gu

BASE = dict(hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=96,
            vocab_size=300, rms_norm_eps=1e-6, rope_theta=10000.0, max_position_embeddings=128, bos_token_id=290, eos_token_id=291,
            hidden_activation="gelu_pytorch_tanh")
CFGS = {
    1: dict(BASE, architectures=["GemmaForCausalLM"], model_type="gemma"),
    2: dict(BASE, architectures=["Gemma2ForCausalLM"], model_type="gemma2", query_pre_attn_scalar=24, sliding_window=4,
            attn_logit_softcapping=5.0, final_logit_softcapping=3.0),
    3: dict(BASE, architectures=["Gemma3ForCausalLM"], model_type="gemma3_text", query_pre_attn_scalar=24, sliding_window=4,
            sliding_window_pattern=3, rope_theta=1000000.0, rope_local_base_freq=100.0, attn_logit_softcapping=None, final_logit_softcapping=None),
}

def make_weights(gen, seed):
    c = CFGS[gen]; rng = np.random.default_rng(seed)
    H, nh, nk, hd, F, V = (c[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size"))
    nw = lambda n: 0.2 * rng.normal(size=n)   # RMSNorm weights are stored as w; the model applies 1 + w
    W = {"model.embed_tokens.weight": rng.normal(0, 0.05, (V, H)), "model.norm.weight": nw(H)}   # small: a tied head must not just echo the input token
    for l in range(c["num_hidden_layers"]):
        p = f"model.layers.{l}."
        W[p + "input_layernorm.weight"] = nw(H); W[p + "post_attention_layernorm.weight"] = nw(H)
        if gen >= 2: W[p + "pre_feedforward_layernorm.weight"] = nw(H); W[p + "post_feedforward_layernorm.weight"] = nw(H)
        if gen == 3: W[p + "self_attn.q_norm.weight"] = nw(hd); W[p + "self_attn.k_norm.weight"] = nw(hd)
        qs = (3 if gen == 2 else 1) / np.sqrt(H)   # gemma2: large q/k so soft-capping bites
        W[p + "self_attn.q_proj.weight"] = rng.normal(0, qs, (nh * hd, H))
        W[p + "self_attn.k_proj.weight"] = rng.normal(0, qs, (nk * hd, H))
        W[p + "self_attn.v_proj.weight"] = rng.normal(0, 1 / np.sqrt(H), (nk * hd, H))
        W[p + "self_attn.o_proj.weight"] = rng.normal(0, 1 / np.sqrt(nh * hd), (H, nh * hd))
        W[p + "mlp.gate_proj.weight"] = rng.normal(0, 1 / np.sqrt(H), (F, H))
        W[p + "mlp.up_proj.weight"] = rng.normal(0, 1 / np.sqrt(H), (F, H))
        W[p + "mlp.down_proj.weight"] = rng.normal(0, 1 / np.sqrt(F), (H, F))
    return W

def forward(gen, W, ids, stats=None):
    c = CFGS[gen]; nh, nk, hd, eps = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["rms_norm_eps"]
    s = len(ids); pos = np.arange(s, dtype=np.float64)
    norm = lambda x, n: gu.rms(x, 1.0 + W[n], eps)
    x = W["model.embed_tokens.weight"][ids] * np.sqrt(c["hidden_size"])
    for l in range(c["num_hidden_layers"]):
        p = f"model.layers.{l}."
        local = gen >= 2 and (l + 1) % (2 if gen == 2 else c["sliding_window_pattern"]) != 0
        base = c["rope_local_base_freq"] if (gen == 3 and local) else c["rope_theta"]
        h = norm(x, p + "input_layernorm.weight")
        q = (h @ W[p + "self_attn.q_proj.weight"].T).reshape(s, nh, hd); k = (h @ W[p + "self_attn.k_proj.weight"].T).reshape(s, nk, hd)
        if gen == 3: q = norm(q, p + "self_attn.q_norm.weight"); k = norm(k, p + "self_attn.k_norm.weight")
        q = gu.rope(q, pos, base); k = gu.rope(k, pos, base)
        v = (h @ W[p + "self_attn.v_proj.weight"].T).reshape(s, nk, hd)
        scale = 1 / np.sqrt(hd) if gen == 1 else c["query_pre_attn_scalar"] ** -0.5
        cap = c["attn_logit_softcapping"] if gen == 2 else 0.0
        if stats is not None and cap: stats.append(np.abs(np.einsum("shd,thd->hst", q, np.repeat(k, nh // nk, axis=1)) * scale).max())
        a = gu.attend(q, k, v, gu.causal_mask(s, c["sliding_window"] if local else 0), scale, cap).reshape(s, nh * hd) @ W[p + "self_attn.o_proj.weight"].T
        if gen == 1:
            x = x + a; h = norm(x, p + "post_attention_layernorm.weight")
        else:
            x = x + norm(a, p + "post_attention_layernorm.weight"); h = norm(x, p + "pre_feedforward_layernorm.weight")
        m = (gu.gelu_tanh(h @ W[p + "mlp.gate_proj.weight"].T) * (h @ W[p + "mlp.up_proj.weight"].T)) @ W[p + "mlp.down_proj.weight"].T
        x = x + (m if gen == 1 else norm(m, p + "post_feedforward_layernorm.weight"))
    lg = norm(x, "model.norm.weight") @ W["model.embed_tokens.weight"].T
    fc = c.get("final_logit_softcapping") if gen == 2 else None
    return np.tanh(lg / fc) * fc if fc else lg

def gguf_tensors(W):
    T = {"token_embd.weight": W["model.embed_tokens.weight"], "output_norm.weight": 1 + W["model.norm.weight"]}
    names = {"input_layernorm": "attn_norm", "post_attention_layernorm": "post_attention_norm", "pre_feedforward_layernorm": "ffn_norm",
             "post_feedforward_layernorm": "post_ffw_norm", "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
             "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
    for k, v in W.items():
        if not k.startswith("model.layers."): continue
        l = k.split(".")[2]; mod = ".".join(k.split(".")[3:-1])
        T[f"blk.{l}.{names[mod]}.weight"] = 1 + v if mod.endswith("layernorm") else v
    return T

def main():
    a = gu.args(); ck = gu.Checker(a.forge, a.work)
    prompt = [290, 72, 101, 108, 108, 111, 44, 32, 116, 104, 101, 32, 119]
    for gen in (1, 2, 3):
        name = {1: "gemma", 2: "gemma2", 3: "gemma3"}[gen]; hf = os.path.join(a.work, "hf_" + name)
        W = gu.save_model(hf, make_weights(gen, 100 + gen), CFGS[gen])
        fwd = lambda ids, gen=gen, W=W: forward(gen, W, ids)
        # gemma2's q/k weights are 3x larger so that soft-capping is active, which also makes its logits the most
        # sensitive to q8_0 rounding: cosine 0.9999 on the CPU and CUDA paths, 0.9984 on the OpenCL path (f32-decoded weights)
        ck.decoder(hf, name, fwd, prompt, CFGS[gen]["eos_token_id"], q8_cos=0.997 if gen == 2 else 0.999)
        if gen == 2:
            st = []; forward(2, W, prompt, st)
            ck.ok(max(st) > CFGS[2]["attn_logit_softcapping"], f"gemma2: raw attention scores reach {max(st):.1f}, above the cap of 5 (soft-capping is exercised)")
            win = forward(2, W, prompt)[-1]; CFGS[2]["sliding_window"] = 4096
            ck.ok(np.abs(win - forward(2, W, prompt)[-1]).max() > 1e-2, "gemma2: the sliding window changes the reference logits (it is exercised)")
            CFGS[2]["sliding_window"] = 4
            c = CFGS[2]
            gp = gu.write_gguf(os.path.join(a.work, "gemma2.gguf"), "gemma2",
                               {"context_length": 128, "embedding_length": c["hidden_size"], "block_count": c["num_hidden_layers"],
                                "feed_forward_length": c["intermediate_size"], "attention.head_count": c["num_attention_heads"],
                                "attention.head_count_kv": c["num_key_value_heads"], "attention.key_length": c["head_dim"],
                                "attention.layer_norm_rms_epsilon": c["rms_norm_eps"], "attention.sliding_window": c["sliding_window"],
                                "attn_logit_softcapping": c["attn_logit_softcapping"], "final_logit_softcapping": c["final_logit_softcapping"]}, gguf_tensors(W))
            if gp:   # a GGUF carries no query_pre_attn_scalar: llama.cpp (and the importer) scale by 1 / sqrt(head_dim)
                c["query_pre_attn_scalar"] = c["head_dim"]
                ck.logits(ck.imp(gp, "gemma2_gguf.fpk", None), prompt, forward(2, W, prompt)[-1], "gemma2 gguf F32")
                c["query_pre_attn_scalar"] = 24
            else: print("[skip] gguf python package not installed")
    ck.done("gemma")

if __name__ == "__main__": main()

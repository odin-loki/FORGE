#!/usr/bin/env python3
"""Golden end-to-end test for the gpt2 template: a random tiny GPT-2 (learned positions, LayerNorm with biases,
Conv1D weights stored [in, out], head tied to the embedding), a float64 numpy reference, and FORGE.

  1. HF import (f32, q8_0) -> prefill logits and KV-cached greedy decode against the reference
  2. GGUF import (general.architecture gpt2, weights transposed as llama.cpp's converter stores them) -> prefill logits
"""
import os
import numpy as np
import golden_util as gu

CFG = dict(architectures=["GPT2LMHeadModel"], model_type="gpt2", n_embd=64, n_layer=2, n_head=4, n_positions=128, n_ctx=128,
           vocab_size=300, layer_norm_epsilon=1e-5, activation_function="gelu_new", bos_token_id=291, eos_token_id=291)

def make_weights(seed=77):
    rng = np.random.default_rng(seed); D, V = CFG["n_embd"], CFG["vocab_size"]
    ln = lambda: (1 + 0.1 * rng.normal(size=D), 0.1 * rng.normal(size=D))
    W = {"wte.weight": rng.normal(0, 0.2, (V, D)), "wpe.weight": rng.normal(0, 0.2, (CFG["n_positions"], D))}   # small: a tied head must not just echo the input token
    W["ln_f.weight"], W["ln_f.bias"] = ln()
    for l in range(CFG["n_layer"]):
        p = f"h.{l}."
        W[p + "ln_1.weight"], W[p + "ln_1.bias"] = ln(); W[p + "ln_2.weight"], W[p + "ln_2.bias"] = ln()
        for n, (i, o) in {"attn.c_attn": (D, 3 * D), "attn.c_proj": (D, D), "mlp.c_fc": (D, 4 * D), "mlp.c_proj": (4 * D, D)}.items():
            W[p + n + ".weight"] = rng.normal(0, 1 / np.sqrt(i), (i, o)); W[p + n + ".bias"] = 0.1 * rng.normal(size=o)   # Conv1D: [in, out]
    return W

def forward(W, ids):
    D, nh = CFG["n_embd"], CFG["n_head"]; hd = D // nh; eps = CFG["layer_norm_epsilon"]; s = len(ids)
    x = W["wte.weight"][ids] + W["wpe.weight"][:s]
    for l in range(CFG["n_layer"]):
        p = f"h.{l}."
        h = gu.layer_norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"], eps)
        c = h @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]
        q, k, v = (c[:, i * D:(i + 1) * D].reshape(s, nh, hd) for i in range(3))
        a = gu.attend(q, k, v, gu.causal_mask(s)).reshape(s, D)
        x = x + a @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        h = gu.layer_norm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"], eps)
        x = x + gu.gelu_tanh(h @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"]) @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
    return gu.layer_norm(x, W["ln_f.weight"], W["ln_f.bias"], eps) @ W["wte.weight"].T

def gguf_tensors(W):
    T = {"token_embd.weight": W["wte.weight"], "position_embd.weight": W["wpe.weight"], "output_norm.weight": W["ln_f.weight"], "output_norm.bias": W["ln_f.bias"]}
    names = {"ln_1": "attn_norm", "ln_2": "ffn_norm", "attn.c_attn": "attn_qkv", "attn.c_proj": "attn_output", "mlp.c_fc": "ffn_up", "mlp.c_proj": "ffn_down"}
    for k, v in W.items():
        if not k.startswith("h."): continue
        _, l, rest = k.split(".", 2); mod, kind = rest.rsplit(".", 1)
        T[f"blk.{l}.{names[mod]}.{kind}"] = v.T if (kind == "weight" and v.ndim == 2) else v   # the converter transposes Conv1D
    return T

def main():
    a = gu.args(); ck = gu.Checker(a.forge, a.work)
    hf = os.path.join(a.work, "hf")
    W = gu.save_model(hf, {"transformer." + k: v for k, v in make_weights().items()}, CFG)
    W = {k[len("transformer."):]: v for k, v in W.items()}
    prompt = [72, 101, 108, 108, 111, 44, 32, 116, 104, 101, 32, 119]
    ck.decoder(hf, "gpt2", lambda ids: forward(W, ids), prompt, CFG["eos_token_id"])
    gp = gu.write_gguf(os.path.join(a.work, "gpt2.gguf"), "gpt2",
                       {"context_length": CFG["n_positions"], "embedding_length": CFG["n_embd"], "block_count": CFG["n_layer"],
                        "attention.head_count": CFG["n_head"], "attention.layer_norm_epsilon": CFG["layer_norm_epsilon"]}, gguf_tensors(W))
    if gp: ck.logits(ck.imp(gp, "gpt2_gguf.fpk", None), prompt, forward(W, prompt)[-1], "gpt2 gguf F32")
    else: print("[skip] gguf python package not installed")
    ck.done("gpt2")

if __name__ == "__main__": main()

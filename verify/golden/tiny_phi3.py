#!/usr/bin/env python3
"""Golden end-to-end test for the phi3 template: random tiny models, a float64 numpy reference, and FORGE.

  phi3       fused qkv projection, fused gate-up MLP, full-width RoPE, separate lm_head, grouped-query attention
  phi4mini   the same with partial RoPE (partial_rotary_factor 0.75) and the head tied to the embedding

  1. HF import (f32, q8_0) -> prefill logits and KV-cached greedy decode against the reference, per variant
  2. GGUF import of the phi4mini-style model (general.architecture phi3, rope.dimension_count) -> prefill logits
  3. text -> ids -> logits: phi3 has a tokenizer.json as transformers writes one for its SentencePiece BPE model, and
     a chat template in tokenizer_config.json; phi4mini a SentencePiece tokenizer.model, and its GGUF the same
     vocabulary as tokenizer.ggml.* (model llama) with tokenizer.chat_template. `forge run --prompt` must reach the
     logits of the reference ids, and `--chat` those of the rendered template.
"""
import os
import numpy as np
import golden_util as gu

BASE = dict(architectures=["Phi3ForCausalLM"], model_type="phi3", hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=96, vocab_size=300, rms_norm_eps=1e-5, rope_theta=10000.0, hidden_act="silu",
            max_position_embeddings=128, bos_token_id=1, eos_token_id=2)
CFGS = {"phi3": dict(BASE, tie_word_embeddings=False, vocab_size=gu.tok_vocab_size("hf_bpe_llama2.json")),
        "phi4mini": dict(BASE, tie_word_embeddings=True, partial_rotary_factor=0.75, vocab_size=560)}
TOK = {"phi3": ("hf_bpe_llama2.json", "hf_bpe_llama2"), "phi4mini": ("spm_bpe_llama.model", "spm_bpe_llama")}
# Phi-3's template, cut down: "<s>" and "</s>" are added tokens of both fixtures, so the rendered text maps to ids
CHAT = "{{ bos_token }}{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}{{ eos_token }}{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"

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

def chat_piece_ids(fixture_file, text):
    """Reference ids of `text` without special tokens (`tokenizers` / `sentencepiece`); None when not installed."""
    path = os.path.join(gu.TOK_DIR, fixture_file)
    try:
        if fixture_file.endswith(".json"):
            from tokenizers import Tokenizer
            return Tokenizer.from_file(path).encode(text, add_special_tokens=False).ids
        import sentencepiece
        return sentencepiece.SentencePieceProcessor(model_file=path).encode(text)
    except ImportError:
        return None

def main():
    a = gu.args(); ck = gu.Checker(a.forge, a.work)
    prompt = [290, 72, 101, 108, 108, 111, 44, 32, 116, 104, 101, 32, 119]
    for i, (name, c) in enumerate(CFGS.items()):
        hf = os.path.join(a.work, "hf_" + name)
        W = gu.save_model(hf, make_weights(c, 200 + i), c, tokenizer=None)
        gu.use_tokenizer(hf, TOK[name][0], {"add_bos_token": True, "add_eos_token": False, "bos_token": "<s>", "eos_token": "</s>", "chat_template": CHAT})
        fwd = lambda ids, c=c, W=W: forward(c, W, ids)
        fp = ck.decoder(hf, name, fwd, prompt, c["eos_token_id"])
        text, ids = gu.tok_golden(TOK[name][1])
        if ids[0] != 1: ids = [1] + ids   # SentencePiece goldens carry no BOS; add_bos_token puts it there
        ck.text_logits(fp, text, fwd(ids)[-1], f"{name} {TOK[name][0]} ({len(ids)} ids with BOS)")
        # chat template: "<s>user:" + text + "</s>assistant:" -- the pieces between the special tokens tokenize on their own
        user, asst = chat_piece_ids(TOK[name][0], "user:"), chat_piece_ids(TOK[name][0], "assistant:")
        if user is None: print("[skip] chat template check: the tokenizers / sentencepiece package is not installed")
        else:
            full = [1] + chat_piece_ids(TOK[name][0], "user:" + text) + [2] + asst
            ck.text_logits(fp, text, fwd(full)[-1], f"{name} --chat: template from tokenizer_config.json ({len(full)} ids)", extra=["--chat"])
        if name != "phi4mini": continue
        hd = c["hidden_size"] // c["num_attention_heads"]
        gp = gu.write_gguf(os.path.join(a.work, "phi4mini.gguf"), "phi3",
                           {"context_length": 128, "embedding_length": c["hidden_size"], "block_count": c["num_hidden_layers"],
                            "feed_forward_length": c["intermediate_size"], "attention.head_count": c["num_attention_heads"],
                            "attention.head_count_kv": c["num_key_value_heads"], "attention.layer_norm_rms_epsilon": c["rms_norm_eps"],
                            "rope.dimension_count": int(hd * c["partial_rotary_factor"]), "rope.freq_base": c["rope_theta"],
                            "tokenizer.chat_template": CHAT}, gguf_tensors(W), tokenizer="gguf_llama_spm.gguf")
        if gp:
            fg = ck.imp(gp, "phi4mini_gguf.fpk", None)
            ck.logits(fg, prompt, forward(c, W, prompt)[-1], "phi4mini gguf F32")
            ck.text_logits(fg, text, fwd(ids)[-1], "phi4mini gguf tokenizer.ggml (llama: SentencePiece BPE)")
            if user is not None: ck.text_logits(fg, text, fwd(full)[-1], "phi4mini gguf --chat: tokenizer.chat_template", extra=["--chat"])
        else: print("[skip] gguf python package not installed")
    ck.done("phi3")

if __name__ == "__main__": main()

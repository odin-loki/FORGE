#!/usr/bin/env python3
"""Golden end-to-end test: a random tiny Llama in HF format, a float64 numpy reference, and FORGE.

  1. writes config.json / model.safetensors / tokenizer.json for a tiny random Llama
  2. `forge import` -> .fpk (f32 and q8_0), `forge inspect --verify`
  3. compares FORGE prefill logits with the numpy reference
  4. compares FORGE greedy generation (KV-cached decode) with numpy greedy (full recompute)
"""
import argparse, json, os, subprocess, sys
import numpy as np

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))

CFG = dict(architectures=["LlamaForCausalLM"], model_type="llama", hidden_size=64, num_hidden_layers=2,
           num_attention_heads=4, num_key_value_heads=2, intermediate_size=128, vocab_size=300,
           rms_norm_eps=1e-5, rope_theta=10000.0, max_position_embeddings=128, bos_token_id=290,
           eos_token_id=291, tie_word_embeddings=False)

def make_model(d, seed=1234):
    rng = np.random.default_rng(seed)
    H, L, nh, nk, F, V = CFG["hidden_size"], CFG["num_hidden_layers"], CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["intermediate_size"], CFG["vocab_size"]
    hd = H // nh
    W = {"model.embed_tokens.weight": rng.normal(0, 1.0, (V, H)), "model.norm.weight": 1 + 0.1 * rng.normal(size=H),
         "lm_head.weight": rng.normal(0, 1 / np.sqrt(H), (V, H))}
    for l in range(L):
        p = f"model.layers.{l}."
        W[p + "input_layernorm.weight"] = 1 + 0.1 * rng.normal(size=H)
        W[p + "post_attention_layernorm.weight"] = 1 + 0.1 * rng.normal(size=H)
        W[p + "self_attn.q_proj.weight"] = rng.normal(0, 1 / np.sqrt(H), (nh * hd, H))
        W[p + "self_attn.k_proj.weight"] = rng.normal(0, 1 / np.sqrt(H), (nk * hd, H))
        W[p + "self_attn.v_proj.weight"] = rng.normal(0, 1 / np.sqrt(H), (nk * hd, H))
        W[p + "self_attn.o_proj.weight"] = rng.normal(0, 1 / np.sqrt(H), (H, nh * hd))
        W[p + "mlp.gate_proj.weight"] = rng.normal(0, 1 / np.sqrt(H), (F, H))
        W[p + "mlp.up_proj.weight"] = rng.normal(0, 1 / np.sqrt(H), (F, H))
        W[p + "mlp.down_proj.weight"] = rng.normal(0, 1 / np.sqrt(F), (H, F))
    W = {k: v.astype(np.float32) for k, v in W.items()}
    from safetensors.numpy import save_file
    os.makedirs(d, exist_ok=True)
    save_file(W, os.path.join(d, "model.safetensors"))
    json.dump(CFG, open(os.path.join(d, "config.json"), "w"), indent=1)
    # byte-level BPE tokenizer: 256 byte tokens + merges + specials
    b2u = bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    merges = [("t", "h"), ("th", "e"), ("Ġ", "t"), ("Ġt", "he"), ("i", "n"), ("Ġ", "a"), ("e", "r"), ("o", "n"),
              ("Ġ", "s"), ("r", "e"), ("a", "n"), ("in", "g"), ("Ġ", "w"), ("o", "r"), ("Ġ", "f"), ("Ġ", "o")]
    for a, b in merges: vocab[a + b] = len(vocab)
    added = [{"id": 290, "content": "<s>", "special": True}, {"id": 291, "content": "</s>", "special": True}]
    tok = {"version": "1.0", "added_tokens": added, "model": {"type": "BPE", "vocab": vocab, "merges": [f"{a} {b}" for a, b in merges]}}
    json.dump(tok, open(os.path.join(d, "tokenizer.json"), "w"))
    return {k: v.astype(np.float64) for k, v in W.items()}

def rms(x, w, eps): return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

def rope(x, pos, base):  # x [s, h, d], neox / rotate_half
    d = x.shape[-1]; half = d // 2
    inv = base ** (-np.arange(half) * 2.0 / d)
    th = pos[:, None] * inv[None, :]
    c, s = np.cos(th)[:, None, :], np.sin(th)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)

def forward(W, ids):
    H, nh, nk = CFG["hidden_size"], CFG["num_attention_heads"], CFG["num_key_value_heads"]; hd = H // nh; eps = CFG["rms_norm_eps"]
    s = len(ids); pos = np.arange(s, dtype=np.float64)
    x = W["model.embed_tokens.weight"][ids]
    for l in range(CFG["num_hidden_layers"]):
        p = f"model.layers.{l}."
        h = rms(x, W[p + "input_layernorm.weight"], eps)
        q = rope((h @ W[p + "self_attn.q_proj.weight"].T).reshape(s, nh, hd), pos, CFG["rope_theta"])
        k = rope((h @ W[p + "self_attn.k_proj.weight"].T).reshape(s, nk, hd), pos, CFG["rope_theta"])
        v = (h @ W[p + "self_attn.v_proj.weight"].T).reshape(s, nk, hd)
        g = nh // nk
        k = np.repeat(k, g, axis=1); v = np.repeat(v, g, axis=1)
        sc = np.einsum("shd,thd->hst", q, k) / np.sqrt(hd)
        sc = sc + np.triu(np.full((s, s), -np.inf), 1)[None]
        sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
        a = np.einsum("hst,thd->shd", sc, v).reshape(s, nh * hd)
        x = x + a @ W[p + "self_attn.o_proj.weight"].T
        h = rms(x, W[p + "post_attention_layernorm.weight"], eps)
        gt = h @ W[p + "mlp.gate_proj.weight"].T
        x = x + ((gt / (1 + np.exp(-gt))) * (h @ W[p + "mlp.up_proj.weight"].T)) @ W[p + "mlp.down_proj.weight"].T
    return rms(x, W["model.norm.weight"], eps) @ W["lm_head.weight"].T

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print("FAILED:", " ".join(cmd), "\n", r.stdout, r.stderr); sys.exit(1)
    return r.stdout

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--forge", required=True); ap.add_argument("--work", default="build/golden")
    a = ap.parse_args()
    hf = os.path.join(a.work, "hf"); W = make_model(hf)
    fails = 0
    prompt = [290, 72, 101, 108, 108, 111, 44, 32, 116, 104, 101, 32, 119]
    ref = forward(W, prompt)[-1]
    for codec, tol in (("f32", 2e-3), ("q8_0", None)):
        out = os.path.join(a.work, f"tiny_{codec}.fpk")
        run([a.forge, "import", hf, "-o", out, "--codec", codec])
        print(run([a.forge, "inspect", out, "--verify"]).strip().splitlines()[-1])
        lf = os.path.join(a.work, f"logits_{codec}.f32")
        gen = run([a.forge, "run", out, "--tokens", ",".join(map(str, prompt)), "-n", "20", "--ids", "--logits", lf])
        got = np.fromfile(lf, dtype=np.float32)
        err = np.abs(got - ref).max(); cos = got @ ref / np.linalg.norm(got) / np.linalg.norm(ref)
        ok = err < tol if tol else cos > 0.999
        print(f"[{'ok' if ok else 'FAIL'}] {codec}: prefill logits max|err| {err:.2e}, cosine {cos:.6f}")
        fails += not ok
        if codec != "f32": continue
        ids = list(prompt); ref_gen = []
        for _ in range(20):
            lg = forward(W, ids)[-1]; t = int(np.argmax(lg)); ref_gen.append(t); ids.append(t)
            if t == CFG["eos_token_id"]: break
        fg = [int(t) for t in gen.split()]
        same = fg == ref_gen
        if not same:  # allow divergence only at a genuine near-tie
            i = next(i for i in range(min(len(fg), len(ref_gen))) if fg[i] != ref_gen[i])
            lg = np.sort(forward(W, prompt + ref_gen[:i])[-1]); same = lg[-1] - lg[-2] < 1e-3
        print(f"[{'ok' if same else 'FAIL'}] greedy decode (KV cache) matches reference: forge {fg[:10]}... ref {ref_gen[:10]}...")
        fails += not same
    # tokenizer + text path
    out = os.path.join(a.work, "tiny_f32.fpk")
    txt = run([a.forge, "run", out, "--prompt", "the theory of things", "-n", "4"])
    print("[ok] text prompt path:", repr(txt.strip()))
    print("golden:", "PASS" if fails == 0 else f"{fails} FAILURE(S)")
    sys.exit(1 if fails else 0)

if __name__ == "__main__": main()

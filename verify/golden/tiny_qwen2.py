#!/usr/bin/env python3
"""Golden end-to-end test for the qwen2 template: a random tiny Qwen2 (biases on the Q, K and V projections) in HF
and GGUF form, a float64 numpy reference, and FORGE.

  1. HF import (f32, q8_0) -> prefill logits and greedy decode against the reference
  2. the reference with the biases zeroed must differ (the test would not notice dropped biases otherwise)
  3. GGUF import (arch qwen2, F32; attn_{q,k,v}.bias) -> prefill logits
  4. --template llama on the same checkpoint is refused: the biases would be left over

forward() is also the reference of tiny_rope.py (inv_freq / attn_factor arguments).
"""
import argparse, json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny_llama as tl

CFG = dict(architectures=["Qwen2ForCausalLM"], model_type="qwen2", hidden_size=64, num_hidden_layers=2,
           num_attention_heads=4, num_key_value_heads=2, intermediate_size=128, vocab_size=300,
           rms_norm_eps=1e-6, rope_theta=1000000.0, max_position_embeddings=128, bos_token_id=290,
           eos_token_id=291, tie_word_embeddings=False)   # untied: a tied tiny model just repeats its last token

def make_model(d, cfg=CFG, seed=2468, biases=True, extra=None):
    """Writes config.json / model.safetensors / tokenizer.json; returns the float64 weights."""
    rng = np.random.default_rng(seed)
    H, L, nh, nk, F, V = (cfg[k] for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                                           "num_key_value_heads", "intermediate_size", "vocab_size"))
    hd = H // nh
    W = {"model.embed_tokens.weight": rng.normal(0, 1.0, (V, H)), "model.norm.weight": 1 + 0.1 * rng.normal(size=H)}
    if not cfg.get("tie_word_embeddings"): W["lm_head.weight"] = rng.normal(0, 1 / np.sqrt(H), (V, H))
    for l in range(L):
        p = f"model.layers.{l}."
        W[p + "input_layernorm.weight"] = 1 + 0.1 * rng.normal(size=H)
        W[p + "post_attention_layernorm.weight"] = 1 + 0.1 * rng.normal(size=H)
        for x, n in (("q", nh), ("k", nk), ("v", nk)):
            W[p + f"self_attn.{x}_proj.weight"] = rng.normal(0, 1 / np.sqrt(H), (n * hd, H))
            if biases: W[p + f"self_attn.{x}_proj.bias"] = rng.normal(0, 0.5, n * hd)   # large enough to matter
        W[p + "self_attn.o_proj.weight"] = rng.normal(0, 1 / np.sqrt(nh * hd), (H, nh * hd))
        W[p + "mlp.gate_proj.weight"] = rng.normal(0, 1 / np.sqrt(H), (F, H))
        W[p + "mlp.up_proj.weight"] = rng.normal(0, 1 / np.sqrt(H), (F, H))
        W[p + "mlp.down_proj.weight"] = rng.normal(0, 1 / np.sqrt(F), (H, F))
    W = {k: v.astype(np.float32) for k, v in W.items()}
    saved = dict(tl.CFG); tl.CFG.update(cfg)
    try: tl.make_model(d)  # tokenizer.json (and a llama model we overwrite)
    finally: tl.CFG.clear(); tl.CFG.update(saved)
    from safetensors.numpy import save_file
    save_file({**W, **(extra or {})}, os.path.join(d, "model.safetensors"))
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    return {k: v.astype(np.float64) for k, v in W.items()}

def rope(x, pos, inv):  # x [s, h, d], rotate_half; inv [d/2]
    half = x.shape[-1] // 2
    th = pos[:, None] * inv[None, :]
    c, s = np.cos(th)[:, None, :], np.sin(th)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)

def forward(W, ids, cfg=CFG, inv_freq=None, attn_factor=1.0):
    """Llama-family forward. Q/K/V biases when W has them. inv_freq(d, seq_len) overrides the RoPE frequencies;
    attn_factor scales cos and sin (YaRN), i.e. both q and k."""
    H, nh, nk, eps = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["rms_norm_eps"]
    hd = H // nh; s = len(ids); pos = np.arange(s, dtype=np.float64); rms = tl.rms
    inv = inv_freq(hd, s) if inv_freq else cfg.get("rope_theta", 10000.0) ** (-np.arange(hd // 2) * 2.0 / hd)
    x = W["model.embed_tokens.weight"][ids]
    for l in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{l}."
        h = rms(x, W[p + "input_layernorm.weight"], eps)
        proj = lambda n: h @ W[p + f"self_attn.{n}_proj.weight"].T + W.get(p + f"self_attn.{n}_proj.bias", 0.0)
        q = rope(proj("q").reshape(s, nh, hd), pos, inv) * attn_factor
        k = rope(proj("k").reshape(s, nk, hd), pos, inv) * attn_factor
        v = proj("v").reshape(s, nk, hd)
        g = nh // nk; k = np.repeat(k, g, axis=1); v = np.repeat(v, g, axis=1)
        sc = np.einsum("shd,thd->hst", q, k) / np.sqrt(hd) + np.triu(np.full((s, s), -np.inf), 1)[None]
        sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
        x = x + np.einsum("hst,thd->shd", sc, v).reshape(s, nh * hd) @ W[p + "self_attn.o_proj.weight"].T
        h = rms(x, W[p + "post_attention_layernorm.weight"], eps)
        gt = h @ W[p + "mlp.gate_proj.weight"].T
        x = x + ((gt / (1 + np.exp(-gt))) * (h @ W[p + "mlp.up_proj.weight"].T)) @ W[p + "mlp.down_proj.weight"].T
    head = W["lm_head.weight"] if "lm_head.weight" in W else W["model.embed_tokens.weight"]
    return rms(x, W["model.norm.weight"], eps) @ head.T

HF2GGUF = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm", "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
           "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}

def write_gguf(path, W, hf_dir, cfg=CFG, arch="qwen2", meta=None, extra=None):
    """A GGUF in HF Q/K order (NeoX RoPE, as llama.cpp writes qwen2). meta(writer) adds keys; extra: more tensors."""
    import gguf
    wr = gguf.GGUFWriter(path, arch)
    wr.add_context_length(cfg["max_position_embeddings"]); wr.add_embedding_length(cfg["hidden_size"])
    wr.add_block_count(cfg["num_hidden_layers"]); wr.add_feed_forward_length(cfg["intermediate_size"])
    wr.add_head_count(cfg["num_attention_heads"]); wr.add_head_count_kv(cfg["num_key_value_heads"])
    wr.add_layer_norm_rms_eps(cfg["rms_norm_eps"]); wr.add_rope_freq_base(cfg["rope_theta"])
    tok = json.load(open(os.path.join(hf_dir, "tokenizer.json")))
    inv = {i: t for t, i in tok["model"]["vocab"].items()}
    for a in tok["added_tokens"]: inv[a["id"]] = a["content"]
    toks = [inv.get(i, f"<unused{i}>") for i in range(cfg["vocab_size"])]
    wr.add_tokenizer_model("gpt2"); wr.add_token_list(toks); wr.add_token_types([3 if t in ("<s>", "</s>") else 1 for t in toks])
    wr.add_token_merges(tok["model"]["merges"]); wr.add_bos_token_id(cfg["bos_token_id"]); wr.add_eos_token_id(cfg["eos_token_id"])
    if meta: meta(wr)
    names = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight", "lm_head.weight": "output.weight"}
    for k, v in W.items():
        if k in names: g = names[k]
        else:
            parts = k.split("."); g = f"blk.{parts[2]}.{HF2GGUF['.'.join(parts[3:-1])]}.{parts[-1]}"
        wr.add_tensor(g, v.astype(np.float32))
    for k, v in (extra or {}).items(): wr.add_tensor(k, v)
    wr.write_header_to_file(); wr.write_kv_data_to_file(); wr.write_tensors_to_file(); wr.close()

def run_fail(cmd):
    """Runs a command that must fail; returns its output (None when it succeeded)."""
    runner = os.environ.get("FORGE_RUNNER", "").split()
    if runner and cmd and cmd[0].endswith("forge"): cmd = runner + list(cmd)
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    return None if r.returncode == 0 else r.stdout + r.stderr

def logits(forge, fp, prompt, n=1):
    """forge run -> (generated ids, prefill logits)."""
    lf = fp + ".logits.f32"
    gen = tl.run([forge, "run", fp, "--tokens", ",".join(map(str, prompt)), "-n", str(n), "--ids", "--logits", lf])
    return [int(t) for t in gen.split()], np.fromfile(lf, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--forge", required=True); ap.add_argument("--work", default="build/golden_qwen2")
    a = ap.parse_args(); run = tl.run
    hf = os.path.join(a.work, "hf"); W = make_model(hf)
    open(os.path.join(hf, "README.md"), "w").write("---\nlicense: apache-2.0\nlicense_link: https://www.apache.org/licenses/LICENSE-2.0\n---\n# tiny\n")
    open(os.path.join(hf, "LICENSE"), "w").write("Apache License, Version 2.0 (golden test text)\n")
    fails = 0
    def report(ok, msg):
        nonlocal fails
        print(f"[{'ok' if ok else 'FAIL'}] {msg}"); fails += not ok
    prompt = [290, 72, 101, 108, 108, 111, 44, 32, 116, 104, 101, 32, 119]
    ref = forward(W, prompt)[-1]
    nobias = forward({k: v for k, v in W.items() if not k.endswith(".bias")}, prompt)[-1]
    report(np.abs(ref - nobias).max() > 0.1, f"the biases matter: reference with / without differ by {np.abs(ref - nobias).max():.2f}")
    for codec, tol, cosmin in (("f32", 2e-3, None), ("q8_0", None, 0.999)):
        out = os.path.join(a.work, f"qwen2_{codec}.fpk")
        txt = run([a.forge, "import", hf, "-o", out, "--codec", codec])
        if codec == "f32": report("template 'qwen2'" in txt, f"HF Qwen2ForCausalLM maps to the qwen2 template: {txt.strip().splitlines()[0]}")
        fg, got = logits(a.forge, out, prompt, 20 if codec == "f32" else 1)
        err = np.abs(got - ref).max(); cos = got @ ref / np.linalg.norm(got) / np.linalg.norm(ref)
        report(err < tol if tol else cos > cosmin, f"hf {codec}: prefill logits max|err| {err:.2e}, cosine {cos:.6f}")
        if codec != "f32": continue
        ids = list(prompt); ref_gen = []
        for _ in range(20):
            t = int(np.argmax(forward(W, ids)[-1])); ref_gen.append(t); ids.append(t)
            if t == CFG["eos_token_id"]: break
        same = fg == ref_gen
        if not same:
            i = next(i for i in range(min(len(fg), len(ref_gen))) if fg[i] != ref_gen[i])
            lg = np.sort(forward(W, prompt + ref_gen[:i])[-1]); same = lg[-1] - lg[-2] < 1e-3
        report(same, f"greedy decode matches reference: forge {fg[:8]}... ref {ref_gen[:8]}...")
    # the licence found at import (README front matter + LICENSE) is shown by inspect and survives sign and encrypt
    f32 = os.path.join(a.work, "qwen2_f32.fpk"); key = os.path.join(a.work, "k"); sg = os.path.join(a.work, "signed.fpk"); en = os.path.join(a.work, "sealed.fpk")
    run([a.forge, "keygen", "-o", key]); run([a.forge, "sign", f32, "--key", key + ".key", "-o", sg])
    run([a.forge, "encrypt", sg, "-o", en, "--new-key", os.path.join(a.work, "m.mkey")])
    for label, fp in (("import", f32), ("sign", sg), ("encrypt", en)):
        ins = run([a.forge, "inspect", fp, "--license"])
        report("license   apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0> [LICENSE]" in ins and "Apache License, Version 2.0 (golden test text)" in ins,
               f"licence recorded and shown by inspect after {label}")
    # the llama template has no biases: importing a Qwen2 checkpoint through it must fail and name the leftovers
    out = run_fail([a.forge, "import", hf, "-o", os.path.join(a.work, "wrong.fpk"), "--template", "llama"])
    report(out is not None and "self_attn.q_proj.bias" in out and "--allow-unused" in out, "--template llama refuses the checkpoint and lists the bias tensors")
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("[skip] gguf python package not installed"); gguf = None
    if gguf:
        gp = os.path.join(a.work, "qwen2.gguf"); fp = os.path.join(a.work, "qwen2_gguf.fpk")
        write_gguf(gp, W, hf)
        txt = run([a.forge, "import", gp, "-o", fp])
        report("template 'qwen2'" in txt, "GGUF arch qwen2 maps to the qwen2 template")
        _, got = logits(a.forge, fp, prompt)
        err = np.abs(got - ref).max()
        report(err < 1e-3, f"gguf qwen2 F32: prefill logits max|err| {err:.2e}")
        names = run([a.forge, "inspect", fp, "--tensors"])
        report("extra." not in names and "layer.0.bq" in names, "gguf: biases are imported as layer.N.b{q,k,v}, nothing lands in extra.*")
    print("golden qwen2:", "PASS" if fails == 0 else f"{fails} FAILURE(S)")
    sys.exit(1 if fails else 0)

if __name__ == "__main__": main()

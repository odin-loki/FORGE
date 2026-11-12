#!/usr/bin/env python3
"""Golden end-to-end test for RoPE frequency scaling: a random tiny Llama whose config.json (or GGUF metadata) asks
for a scaling variant, a float64 numpy reference that follows transformers/modeling_rope_utils.py (transformers 5.22), and FORGE.

  HF config.json  rope_scaling / rope_parameters: linear, dynamic NTK, YaRN (default, explicit and mscale attention
                  factors), llama3; LongRoPE and unknown types are refused
  GGUF            <arch>.rope.scaling.type linear / yarn, and llama3 factors baked into rope_freqs.weight
                  (llama.cpp's converter), with llama.cpp's permuted Q/K and interleaved RoPE

Each variant must also differ from the unscaled reference, so a scaling that is silently ignored fails the test.
`--dump-tables` prints the inverse-frequency tables that tests/test_graph.cpp checks rope_inv_freq against.
"""
import argparse, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny_llama as tl
import tiny_qwen2 as tq

BASE = dict(architectures=["LlamaForCausalLM"], model_type="llama", hidden_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, intermediate_size=128, vocab_size=300,
            rms_norm_eps=1e-5, rope_theta=10000.0, max_position_embeddings=128, bos_token_id=290,
            eos_token_id=291, tie_word_embeddings=False)

# ---- transformers/modeling_rope_utils.py (5.22: _compute_*_parameters), in numpy (float64) ----------------------------------------------------
def inv_default(base, d): return 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float64) / d))

def inv_linear(base, d, factor): return inv_default(base, d) / factor

def inv_dynamic(base, d, factor, max_pos, seq_len):
    seq_len = seq_len if seq_len > max_pos else max_pos
    b = base * ((factor * seq_len / max_pos) - (factor - 1)) ** (d / (d - 2))
    return inv_default(b, d)

def get_mscale(scale, mscale=1.0): return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0

def inv_yarn(base, d, factor, orig, beta_fast=32.0, beta_slow=1.0, truncate=True):
    def corr_dim(rot): return (d * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))
    low, high = corr_dim(beta_fast), corr_dim(beta_slow)
    if truncate: low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, d - 1)
    if low == high: high += 0.001
    ramp = np.clip((np.arange(d // 2, dtype=np.float64) - low) / (high - low), 0, 1)
    pos_freqs = base ** (np.arange(0, d, 2, dtype=np.float64) / d)
    extra, interp = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    keep = 1 - ramp
    return interp * (1 - keep) + extra * keep

def inv_llama3(base, d, factor, low_freq_factor, high_freq_factor, old_ctx):
    inv = inv_default(base, d)
    low_wl, high_wl = old_ctx / low_freq_factor, old_ctx / high_freq_factor
    wl = 2 * math.pi / inv
    out = np.where(wl > low_wl, inv / factor, inv)
    smooth = (old_ctx / wl - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * out / factor + smooth * out
    medium = ~(wl < high_wl) * ~(wl > low_wl)
    return np.where(medium, smoothed, out)

# ---- the variants: (name, config.json overrides, reference inv_freq(d, seq_len), attention factor) ----------------
def variants():
    b = BASE["rope_theta"]
    yf = 256 / 64   # HF: a factor that is not given is max_position_embeddings / original_max_position_embeddings
    return [
        ("linear", dict(rope_scaling=dict(rope_type="linear", factor=4.0)), lambda d, s: inv_linear(b, d, 4.0), 1.0),
        ("linear (legacy 'type' key)", dict(rope_scaling=dict(type="linear", factor=2.5)), lambda d, s: inv_linear(b, d, 2.5), 1.0),
        ("linear (rope_parameters)", dict(rope_theta=None, rope_parameters=dict(rope_type="linear", factor=3.0, rope_theta=5000.0)),
         lambda d, s: inv_linear(5000.0, d, 3.0), 1.0),
        ("dynamic", dict(max_position_embeddings=8, rope_scaling=dict(rope_type="dynamic", factor=2.0)),
         lambda d, s: inv_dynamic(b, d, 2.0, 8, s), 1.0),
        ("yarn", dict(max_position_embeddings=256, rope_scaling=dict(rope_type="yarn", original_max_position_embeddings=64)),
         lambda d, s: inv_yarn(b, d, yf, 64), get_mscale(yf)),
        ("yarn (attention_factor, betas, no truncate)",
         dict(max_position_embeddings=256, rope_scaling=dict(rope_type="yarn", factor=8.0, attention_factor=1.3, beta_fast=16, beta_slow=2, truncate=False)),
         lambda d, s: inv_yarn(b, d, 8.0, 256, 16, 2, False), 1.3),
        ("yarn (mscale / mscale_all_dim)",
         dict(max_position_embeddings=256, rope_scaling=dict(rope_type="yarn", factor=4.0, original_max_position_embeddings=64, mscale=1.0, mscale_all_dim=0.5)),
         lambda d, s: inv_yarn(b, d, yf, 64), get_mscale(yf, 1.0) / get_mscale(yf, 0.5)),
        ("llama3", dict(rope_scaling=dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=64)),
         lambda d, s: inv_llama3(b, d, 8.0, 1.0, 4.0, 64), 1.0),
    ]

def dump_tables():
    """C++ initialisers for tests/test_graph.cpp (d = 16, base 10000)."""
    d, b = 16, 10000.0
    rows = [("linear factor 4", inv_linear(b, d, 4.0)), ("dynamic factor 2 max_pos 8 seq_len 13", inv_dynamic(b, d, 2.0, 8, 13)),
            ("yarn factor 4 orig 64", inv_yarn(b, d, 4.0, 64)), ("yarn factor 8 orig 256 betas 16/2 no truncate", inv_yarn(b, d, 8.0, 256, 16, 2, False)),
            ("llama3 factor 8 low 1 high 4 orig 64", inv_llama3(b, d, 8.0, 1.0, 4.0, 64))]
    for name, t in rows: print(f"// {name}\n{{" + ", ".join(f"{x:.17g}" for x in t) + "},")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--forge"); ap.add_argument("--work", default="build/golden_rope")
    ap.add_argument("--dump-tables", action="store_true")
    a = ap.parse_args(); run = tl.run
    if a.dump_tables: return dump_tables()
    fails = 0
    def report(ok, msg):
        nonlocal fails
        print(f"[{'ok' if ok else 'FAIL'}] {msg}"); fails += not ok
    prompt = [290, 72, 101, 108, 108, 111, 44, 32, 116, 104, 101, 32, 119]
    def compare(fp, W, cfg, inv, af, label, tol=1e-4, decode=True):
        ref = tq.forward(W, prompt, cfg, inv, af)[-1]
        plain = tq.forward(W, prompt, cfg)[-1]
        lf = fp + ".logits.f32"
        gen = run([a.forge, "run", fp, "--tokens", ",".join(map(str, prompt)), "-n", "8", "--ids", "--logits", lf, "--ctx", "64"])
        got = np.fromfile(lf, dtype=np.float32); fg = [int(t) for t in gen.split()]
        err, gap = np.abs(got - ref).max(), np.abs(ref - plain).max()
        report(err < tol and gap > 20 * tol, f"{label}: prefill logits max|err| {err:.2e} (the unscaled reference is {gap:.2f} away)")
        if not decode: return
        ids = list(prompt); ref_gen = []
        for _ in range(8):
            t = int(np.argmax(tq.forward(W, ids, cfg, inv, af)[-1])); ref_gen.append(t); ids.append(t)
            if t == cfg["eos_token_id"]: break
        same = fg == ref_gen
        if not same:
            i = next(i for i in range(min(len(fg), len(ref_gen))) if fg[i] != ref_gen[i])
            lg = np.sort(tq.forward(W, prompt + ref_gen[:i], cfg, inv, af)[-1]); same = lg[-1] - lg[-2] < 1e-3
        report(same, f"{label}: greedy decode (KV cache) matches the reference")
    # ---- Hugging Face config.json ----
    for i, (name, over, inv, af) in enumerate(variants()):
        cfg = {k: v for k, v in {**BASE, **over}.items() if v is not None}
        hf = os.path.join(a.work, f"hf{i}"); W = tq.make_model(hf, cfg, seed=100 + i, biases=False)
        fp = os.path.join(a.work, f"rope{i}.fpk")
        run([a.forge, "import", hf, "-o", fp])
        # dynamic NTK recomputes the frequencies from the positions of each call, so cached keys keep the base they
        # were rotated with (as transformers does with a KV cache): only the prefill equals a full recompute
        compare(fp, W, cfg, inv, af, f"hf {name}", decode="dynamic" not in name)
        if i == 0:
            ins = run([a.forge, "inspect", fp, "--blueprint"])
            report("opset 2" in ins and "scaling: 1, factor: 4" in ins, "a scaled import declares opset 2 and carries the rope attrs")
    # unscaled imports stay opset 1 with the same rope call as before
    hf = os.path.join(a.work, "hf_plain"); tq.make_model(hf, BASE, biases=False); fp = os.path.join(a.work, "plain.fpk")
    run([a.forge, "import", hf, "-o", fp]); ins = run([a.forge, "inspect", fp, "--blueprint"])
    report("opset 1" in ins and "scaling" not in ins and "neox: 1)" in ins, "an unscaled import stays opset 1 with the unchanged rope call")
    for name, rs, want in (("longrope", dict(rope_type="longrope", short_factor=[1.0] * 8, long_factor=[2.0] * 8), "longrope"),
                           ("unknown type", dict(rope_type="spiral", factor=2.0), "spiral"),
                           ("llama3 without its parameters", dict(rope_type="llama3", factor=8.0), "low_freq_factor")):
        hf = os.path.join(a.work, "hf_bad"); tq.make_model(hf, {**BASE, "rope_scaling": rs}, biases=False)
        out = tq.run_fail([a.forge, "import", hf, "-o", os.path.join(a.work, "bad.fpk")])
        report(out is not None and want in out, f"hf {name}: refused with a message naming '{want}'")
    # ---- GGUF ----
    try:
        import gguf
    except ImportError:
        print("[skip] gguf python package not installed"); gguf = None
    if gguf:
        b, d = BASE["rope_theta"], BASE["hidden_size"] // BASE["num_attention_heads"]
        def permuted(W):  # llama.cpp's converter permutes Q/K of arch llama so that RoPE pairs are interleaved
            nh, nk = BASE["num_attention_heads"], BASE["num_key_value_heads"]
            perm = lambda w, n: w.reshape(n, 2, w.shape[0] // n // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)
            return {k: perm(v, nh) if "q_proj" in k else perm(v, nk) if "k_proj" in k else v for k, v in W.items()}
        ll3 = inv_llama3(b, d, 8.0, 1.0, 4.0, 64)
        factors = (ll3 / inv_default(b, d)).astype(np.float32)   # what convert_hf_to_gguf.py stores as rope_freqs.weight
        cases = [
            ("linear", lambda w: (w.add_rope_scaling_type(gguf.RopeScalingType.LINEAR), w.add_rope_scaling_factor(4.0)), None,
             lambda dd, s: inv_linear(b, dd, 4.0), 1.0),
            ("yarn", lambda w: (w.add_rope_scaling_type(gguf.RopeScalingType.YARN), w.add_rope_scaling_factor(4.0), w.add_rope_scaling_orig_ctx_len(64)), None,
             lambda dd, s: inv_yarn(b, dd, 4.0, 64), get_mscale(4.0)),
            ("rope_freqs.weight (llama3)", None, {"rope_freqs.weight": factors}, lambda dd, s: inv_default(b, dd) * factors.astype(np.float64), 1.0),
        ]
        hf = os.path.join(a.work, "hf_gguf"); W = tq.make_model(hf, BASE, seed=77, biases=False)
        for i, (name, meta, extra, inv, af) in enumerate(cases):
            gp = os.path.join(a.work, f"rope{i}.gguf"); fp = os.path.join(a.work, f"rope_gguf{i}.fpk")
            tq.write_gguf(gp, permuted(W), hf, BASE, arch="llama", meta=meta, extra=extra)
            run([a.forge, "import", gp, "-o", fp])
            compare(fp, W, BASE, inv, af, f"gguf {name}")
        gp = os.path.join(a.work, "longrope.gguf")
        tq.write_gguf(gp, permuted(W), hf, BASE, arch="llama", meta=lambda w: w.add_rope_scaling_type(gguf.RopeScalingType.LONGROPE))
        out = tq.run_fail([a.forge, "import", gp, "-o", os.path.join(a.work, "bad.fpk")])
        report(out is not None and "longrope" in out, "gguf longrope: refused")
    print("golden rope scaling:", "PASS" if fails == 0 else f"{fails} FAILURE(S)")
    sys.exit(1 if fails else 0)

if __name__ == "__main__": main()

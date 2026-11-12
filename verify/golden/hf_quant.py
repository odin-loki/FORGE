#!/usr/bin/env python3
"""AWQ and GPTQ checkpoint import, against synthetic checkpoints.

The packers below are written from the layouts documented in modelfile/include/forge/import/hf_quant.hpp (AutoAWQ's
WQLinear "gemm" and AutoGPTQ's QuantLinear); the importer is NOT checked against a checkpoint written by AutoAWQ or
AutoGPTQ themselves. What this test establishes: importer and packer agree on the documented layout, the grouped
codec holds the codes without re-quantising, and the model computes with W = scale * (q - zero).

Needs numpy and safetensors.
"""
import argparse, json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny_llama as tl

AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]   # AutoAWQ: nibble p of a word holds element AWQ_ORDER[p] of its group of eight

def quantise(w, bits, gs, rng, sym=False, perm=None):
    """w [out, in] -> integer codes q [out, in], zero and scale [groups, out] (scale exactly f16), group of each column."""
    out, inn = w.shape; maxq = 2 ** bits - 1
    gidx = np.arange(inn) // gs if perm is None else perm   # perm: a shuffled column -> group map (GPTQ desc_act)
    ng = inn // gs
    q = np.zeros((out, inn), np.int64); zero = np.zeros((ng, out), np.int64); scale = np.zeros((ng, out), np.float32)
    for g in range(ng):
        cols = np.nonzero(gidx == g)[0]; x = w[:, cols]
        if sym:
            s = np.float16(np.abs(x).max(1) / (maxq // 2) * 1.02).astype(np.float32); z = np.full(out, (maxq + 1) // 2)
        else:
            lo = np.minimum(x.min(1), 0); hi = np.maximum(x.max(1), 0)
            s = np.float16((hi - lo) / maxq * 1.02).astype(np.float32); z = np.clip(np.rint(-lo / s), 1, maxq).astype(np.int64)   # z >= 1: see pack_gptq
        q[:, cols] = np.clip(np.rint(x / s[:, None]) + z[:, None], 0, maxq); zero[g] = z; scale[g] = s
    return q, zero, scale, gidx.astype(np.int32)

def dequant(q, zero, scale, gidx): return ((q - zero[gidx].T).astype(np.float32) * scale[gidx].T).astype(np.float32)   # [out, in]

def pack_words(m, bits, order=None):
    """m [rows, n] codes -> int32 [rows, n * bits / 32]: code j of a word at bits [bits * p, ...), p its position."""
    per = 32 // bits; rows, n = m.shape; w = np.zeros((rows, n // per), np.uint32)
    for p in range(per):
        src = order[p] if order else p
        w |= (m[:, src::per].astype(np.int64) & 0xFFFFFFFF).astype(np.uint32) << np.uint32(bits * p)
    return w.view(np.int32)

def pack_awq(q, zero, scale):       # qweight [in, out/8], qzeros [groups, out/8], scales [groups, out] (f16)
    return {"qweight": pack_words(q.T.copy(), 4, AWQ_ORDER), "qzeros": pack_words(zero, 4, AWQ_ORDER), "scales": scale.astype(np.float16)}

def pack_gptq(q, zero, scale, gidx, bits, v2=False):
    # qweight [in*bits/32, out]: the transposed codes packed down the input axis; qzeros [groups, out*bits/32].
    # The original format stores zero - 1 (as AutoGPTQ does; a zero of 0 would wrap to all ones there and spill into
    # the neighbouring fields, which is why quantise() keeps z >= 1).
    qz = zero if v2 else zero - 1
    return {"qweight": np.ascontiguousarray(pack_words(q.copy(), bits).T), "qzeros": pack_words(qz, bits), "scales": scale.astype(np.float16), "g_idx": gidx}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--forge", required=True); ap.add_argument("--work", default="build/golden_hf_quant")
    a = ap.parse_args(); run = tl.run
    from safetensors.numpy import save_file
    os.makedirs(a.work, exist_ok=True)
    base = os.path.join(a.work, "base"); W = tl.make_model(base)
    W = {k: v.astype(np.float16).astype(np.float32) for k, v in W.items()}   # these checkpoints are fp16
    fails = 0
    def check(ok, msg):
        nonlocal fails
        print(f"[{'ok' if ok else 'FAIL'}] {msg}"); fails += not ok
    def forge(*args, expect_fail=False):
        runner = os.environ.get("FORGE_RUNNER", "").split()
        r = subprocess.run(runner + [a.forge, *args], capture_output=True, text=True, errors="replace")
        if (r.returncode != 0) != expect_fail: print("UNEXPECTED exit", r.returncode, ":", " ".join(args), "\n", r.stdout, r.stderr); sys.exit(1)
        return r
    def dump(fp, name):
        pf = os.path.join(a.work, "dump.f32"); run([a.forge, "dump", fp, name, pf]); return np.fromfile(pf, dtype=np.float32)
    def codecs(fp):
        out = {}
        for ln in run([a.forge, "inspect", fp, "--tensors"]).splitlines():
            p = ln.split()
            if len(p) >= 2 and (p[0].startswith("layer.") or p[0] in ("tok_emb", "lm_head", "out_norm")): out[p[0]] = p[1]
        return out
    def manifest(fp):
        txt = run([a.forge, "inspect", fp]); return json.JSONDecoder().raw_decode(txt[txt.index("{"):])[0]
    prompt = [290, 72, 101, 108, 108, 111, 44, 32, 116, 104, 101, 32, 119]
    def logits(fp):
        lf = fp + ".logits.f32"; run([a.forge, "run", fp, "--tokens", ",".join(map(str, prompt)), "-n", "1", "--ids", "--logits", lf])
        return np.fromfile(lf, dtype=np.float32)
    ref32 = tl.forward({k: v.astype(np.float64) for k, v in W.items()}, prompt)[-1]

    def make(name, method, bits=4, gs=32, sym=False, shuffle=False, qcfg=None, v2=False):
        """A checkpoint directory with every decoder Linear quantised. Returns (dir, dequantised model, {forge name: W})."""
        d = os.path.join(a.work, name); os.makedirs(d, exist_ok=True)
        rng = np.random.default_rng(len(name)); T = {}; Wd = {}; lin = {}
        for k, v in W.items():
            if ".layers." not in k or "layernorm" in k: T[k] = v.astype(np.float16); Wd[k] = v; continue
            inn = v.shape[1]
            perm = rng.permutation(np.arange(inn) // gs) if shuffle else None
            q, z, s, gi = quantise(v, bits, gs, rng, sym, perm)
            p = pack_awq(q, z, s) if method == "awq" else pack_gptq(q, z, s, gi, bits, v2)
            for kk, vv in p.items(): T[k[:-len("weight")] + kk] = vv
            Wd[k] = dequant(q, z, s, gi)
            leaf = {"q_proj": "wq", "k_proj": "wk", "v_proj": "wv", "o_proj": "wo", "gate_proj": "w_gate", "up_proj": "w_up", "down_proj": "w_down"}[k.split(".")[-2]]
            lin[f"layer.{k.split('.')[2]}.{leaf}"] = Wd[k]
        save_file(T, os.path.join(d, "model.safetensors"))
        cfg = dict(tl.CFG); cfg["quantization_config"] = dict(quant_method=method, bits=bits, group_size=gs, **(qcfg or {}))
        json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
        import shutil; shutil.copy(os.path.join(base, "tokenizer.json"), d)
        return d, {k: v.astype(np.float64) for k, v in Wd.items()}, lin

    #        name            method  bits gs  sym    shuffle  extra config                                       stored as   exact
    cases = [("awq_g32",     "awq",  4,   32, False, False, dict(zero_point=True, version="gemm"),               "q4_g32",   False),
             ("awq_g64",     "awq",  4,   64, False, False, dict(zero_point=True, version="GEMM", backend="autoawq"), "q4_g32", False),
             ("gptq_sym",    "gptq", 4,   32, True,  False, dict(sym=True, desc_act=False),                      "q4_g32",   True),
             ("gptq_asym",   "gptq", 4,   64, False, False, dict(sym=False, desc_act=False),                     "q4_g32",   False),
             ("gptq_v2",     "gptq", 4,   32, False, False, dict(sym=False, checkpoint_format="gptq_v2"),        "q4_g32",   False),
             ("gptq_8bit",   "gptq", 8,   32, False, False, dict(sym=False),                                     "q8_g32",   False),
             ("gptq_desc",   "gptq", 4,   32, False, True,  dict(sym=False, desc_act=True),                      "f16",      False),
             ("gptq_2bit",   "gptq", 2,   32, False, False, dict(sym=False),                                     "f16",      False)]
    for name, method, bits, gs, sym, shuffle, qcfg, want_codec, exact in cases:
        d, Wd, lin = make(name, method, bits, gs, sym, shuffle, qcfg, v2=qcfg.get("checkpoint_format") == "gptq_v2")
        fp = os.path.join(a.work, name + ".fpk"); r = forge("import", d, "-o", fp)
        cd = codecs(fp); man = manifest(fp); sq = man.get("source_quantisation", {})
        ref = tl.forward(Wd, prompt)[-1]; got = logits(fp)
        err = np.abs(got - ref).max(); cos = got @ ref32 / np.linalg.norm(got) / np.linalg.norm(ref32)
        worst = 0.0; same = True
        for n, v in lin.items():
            g = dump(fp, n); same = same and np.array_equal(g, v.reshape(-1))
            worst = max(worst, float(np.abs(g - v.reshape(-1)).max() / np.abs(v).max()))
        ok = all(cd[n] == want_codec for n in lin) and cd["tok_emb"] == "f16" and sq.get("method") == method and sq.get("linears") == len(lin)
        ok = ok and sq.get("stored_as_grouped_codec") == (len(lin) if want_codec != "f16" else 0) and err < 5e-3 and worst < 2e-3 and (same or not exact)
        check(ok, f"{name:10s} {method} {bits}-bit group {gs}: Linears stored as {cd['layer.0.wq']}, weights vs scale*(q-zero): "
              f"{'identical' if same else f'max rel diff {worst:.1e}'}; logits vs dequantised reference max|err| {err:.1e}; cosine to fp16 model {cos:.4f}")
    # --codec: decode and quantise again, with a warning
    d, Wd, lin = make("awq_requant", "awq", 4, 32, False, False, dict(zero_point=True, version="gemm"))
    ref = tl.forward(Wd, prompt)[-1]
    for codec, tol in (("f32", 5e-3), ("q8_0", None)):
        fp = os.path.join(a.work, f"awq_requant_{codec}.fpk"); r = forge("import", d, "-o", fp, "--codec", codec)
        cd = codecs(fp); got = logits(fp); err = np.abs(got - ref).max(); cos = got @ ref / np.linalg.norm(got) / np.linalg.norm(ref)
        exact = all(np.array_equal(dump(fp, n), v.reshape(-1)) for n, v in lin.items()) if codec == "f32" else None
        ok = cd["layer.0.wq"] == codec and "warning" in r.stderr and (("double quantisation" in r.stderr) == (codec != "f32"))
        ok = ok and (err < tol and exact if tol else cos > 0.999) and manifest(fp)["source_quantisation"]["mode"] == "requantised"
        check(ok, f"awq --codec {codec}: Linears decoded and stored as {cd['layer.0.wq']} (f32 == scale*(q-zero): {exact}); logits max|err| {err:.1e} cosine {cos:.5f}")
    # refused variants
    for name, method, qcfg, word in (("awq_gemv", "awq", dict(version="gemv"), "gemv"), ("gptq_3bit", "gptq", dict(bits=3), "3 bits")):
        d = os.path.join(a.work, name); os.makedirs(d, exist_ok=True)
        src = os.path.join(a.work, "awq_g32")
        import shutil
        for f in ("model.safetensors", "tokenizer.json"): shutil.copy(os.path.join(src, f), d)
        cfg = dict(tl.CFG); cfg["quantization_config"] = dict(dict(quant_method=method, bits=4, group_size=32), **qcfg)
        json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
        r = forge("import", d, "-o", os.path.join(a.work, name + ".fpk"), expect_fail=True)
        check(word in r.stderr + r.stdout and "not supported" in r.stderr + r.stdout, f"{name} is refused: {(r.stderr + r.stdout).strip().splitlines()[-1][:140]}")
    print("golden hf quant:", "PASS" if fails == 0 else f"{fails} FAILURE(S)")
    sys.exit(1 if fails else 0)

if __name__ == "__main__": main()

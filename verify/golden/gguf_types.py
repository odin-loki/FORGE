#!/usr/bin/env python3
"""GGUF tensor types: every type the importer loads, against gguf-py.

  1. probes   random blocks of every block type: FORGE's decoder (forge dump) against gguf.quants.dequantize, bit for
              bit; plain numeric types (F64, I8..I64) against numpy; Q8_K against d * q computed here (gguf-py has no
              Q8_K dequantiser)
  2. models   a tiny random model with its weight matrices in each type (packed by gguf.quants.quantize where gguf-py
              has a quantiser, else by the numpy packers below, written from the block layouts): imported tensors
              equal gguf-py's dequantisation exactly, and the prefill logits match a float64 forward pass over the
              dequantised weights
  3. --codec  re-quantising a GGUF at import (keep / another codec), the manifest record and the warning
  4. types FORGE does not load (IQ2_*, IQ3_*, IQ1_*, Q8_1, MXFP4) fail with a message that names the type

Needs numpy, safetensors and gguf (pip install gguf).
"""
import argparse, json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny_llama as tl

CFG = dict(architectures=["Qwen2ForCausalLM"], model_type="qwen2", hidden_size=256, num_hidden_layers=1,
           num_attention_heads=4, num_key_value_heads=2, intermediate_size=512, vocab_size=300,
           rms_norm_eps=1e-5, rope_theta=10000.0, max_position_embeddings=128, bos_token_id=290,
           eos_token_id=291, tie_word_embeddings=False)
IQ4_K = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)

def f16(x): return np.asarray(x, np.float32).astype(np.float16)
def f16b(x): return f16(x).view(np.uint8).reshape(*np.shape(x), 2)
def safe_div(a, b): return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape, np.float32), where=b != 0)

# ---- numpy packers for the types gguf-py cannot quantise (layouts: docs/modelfile.md) ----
def pack2(q):  # [nb, 256] 2-bit codes -> qs[64]: element e of half h is bits 2*(e//32).. of byte 32*h + e%32
    q = q.reshape(-1, 2, 4, 32).astype(np.uint8)
    return (q << (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 4, 1)).sum(2).astype(np.uint8).reshape(-1, 64)

def pack_q2_k(x):
    x = x.reshape(-1, 16, 16).astype(np.float32)
    lo = np.minimum(x.min(-1), 0); hi = np.maximum(x.max(-1), 0)
    d = f16((hi - lo).max(-1) / 3 / 15).astype(np.float32)[:, None]; dmin = f16((-lo).max(-1) / 15).astype(np.float32)[:, None]
    ls = np.clip(np.ceil(safe_div((hi - lo) / 3, d)), 0, 15); lm = np.clip(np.rint(safe_div(-lo, dmin)), 0, 15)
    q = np.clip(np.rint(safe_div(x + (dmin * lm)[..., None], (d * ls)[..., None])), 0, 3)
    scales = (ls.astype(np.uint8) | (lm.astype(np.uint8) << 4))
    return np.concatenate([scales, pack2(q), f16b(d[:, 0]), f16b(dmin[:, 0])], axis=1)

def pack_q3_k(x):
    x = x.reshape(-1, 16, 16).astype(np.float32)
    mx = np.take_along_axis(x, np.abs(x).argmax(-1)[..., None], -1)[..., 0]  # signed largest magnitude per sub-block
    s = mx / -4
    big = np.take_along_axis(s, np.abs(s).argmax(-1)[:, None], -1)
    d = f16(big / -32).astype(np.float32)
    ls = np.clip(np.rint(safe_div(s, d)), -32, 31)
    q = np.clip(np.rint(safe_div(x, (d * ls)[..., None])), -4, 3).astype(np.int8).reshape(-1, 256)
    hbit = (q >= 0).reshape(-1, 2, 4, 32).astype(np.uint8)
    hmask = (hbit << np.arange(8, dtype=np.uint8).reshape(1, 2, 4, 1)).sum((1, 2)).astype(np.uint8)   # bit 4h + j of byte r
    u = (ls + 32).astype(np.uint8)
    sc = np.zeros((len(u), 12), np.uint8)
    sc[:, :8] = (u[:, :8] & 15) | ((u[:, 8:] & 15) << 4)
    for i in range(4): sc[:, 8 + i] = sum(((u[:, 4 * g + i] >> 4) << (2 * g)) for g in range(4))
    return np.concatenate([hmask, pack2(q & 3), sc, f16b(d[:, 0])], axis=1)

def iq4_codes(x, d):  # nearest codebook index of x / d, [.., 32] -> 16 bytes (element j low nibble, j + 16 high nibble)
    idx = np.abs(safe_div(x, d)[..., None] - IQ4_K).argmin(-1).astype(np.uint8)
    return idx[..., :16] | (idx[..., 16:] << 4)

def pack_iq4_nl(x):
    x = x.reshape(-1, 32).astype(np.float32)
    d = f16(np.abs(x).max(-1) / 120).astype(np.float32)[:, None]
    return np.concatenate([f16b(d[:, 0]), iq4_codes(x, d)], axis=1)

def pack_iq4_xs(x):
    x = x.reshape(-1, 8, 32).astype(np.float32)
    dl = np.abs(x).max(-1) / 120                                   # per sub-block step
    d = f16(dl.max(-1) / 31).astype(np.float32)[:, None]           # ls - 32 in [1, 31]
    l = np.clip(np.ceil(safe_div(dl, d)), 1, 31)
    u = (l + 32).astype(np.uint16)
    sl = ((u[:, 0::2] & 15) | ((u[:, 1::2] & 15) << 4)).astype(np.uint8)
    sh = sum(((u[:, s] >> 4) << (2 * s)) for s in range(8)).astype(np.uint16)
    return np.concatenate([f16b(d[:, 0]), sh.view(np.uint8).reshape(-1, 2), sl, iq4_codes(x, (d * l)[..., None]).reshape(-1, 128)], axis=1)

PACKERS = {"Q2_K": pack_q2_k, "Q3_K": pack_q3_k, "IQ4_NL": pack_iq4_nl, "IQ4_XS": pack_iq4_xs}
#           type       packer / quantiser                     model check: max |logit err| against the dequantised reference
MODEL_TYPES = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q2_K", "Q3_K", "IQ4_NL", "IQ4_XS", "TQ1_0", "TQ2_0", "F64"]
BLOCK_TYPES = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS", "TQ1_0", "TQ2_0"]
REFUSED = ["IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M", "Q8_1", "MXFP4"]

def quantise(v, qt):
    """v [rows, k] float32 -> (raw block bytes [rows, bytes], dequantised float32 [rows, k], oracle of the packer)"""
    import gguf
    t = getattr(gguf.GGMLQuantizationType, qt)
    if qt in PACKERS: raw = PACKERS[qt](v).reshape(v.shape[0], -1); who = "numpy packer"
    else: raw = gguf.quants.quantize(v, t); who = "gguf-py quantiser"
    return raw, gguf.quants.dequantize(raw, t).reshape(v.shape), who

def make_model(d):
    saved = dict(tl.CFG); tl.CFG.clear(); tl.CFG.update(CFG)
    try: return tl.make_model(d, seed=99)
    finally: tl.CFG.clear(); tl.CFG.update(saved)

def forward(W, ids):
    saved = dict(tl.CFG); tl.CFG.clear(); tl.CFG.update(CFG)
    try: return tl.forward(W, ids)
    finally: tl.CFG.clear(); tl.CFG.update(saved)

HF2G = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm", "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
TOP = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight", "lm_head.weight": "output.weight"}
FORGE_NAME = {"token_embd.weight": "tok_emb", "output.weight": "lm_head", "attn_q": "wq", "attn_k": "wk", "attn_v": "wv", "attn_output": "wo",
              "ffn_gate": "w_gate", "ffn_up": "w_up", "ffn_down": "w_down"}

def writer(path, hf_dir):
    import gguf
    wr = gguf.GGUFWriter(path, "qwen2")   # the llama template without llama.cpp's Q/K permutation (NeoX RoPE)
    wr.add_context_length(CFG["max_position_embeddings"]); wr.add_embedding_length(CFG["hidden_size"])
    wr.add_block_count(CFG["num_hidden_layers"]); wr.add_feed_forward_length(CFG["intermediate_size"])
    wr.add_head_count(CFG["num_attention_heads"]); wr.add_head_count_kv(CFG["num_key_value_heads"])
    wr.add_layer_norm_rms_eps(CFG["rms_norm_eps"]); wr.add_rope_freq_base(CFG["rope_theta"])
    tok = json.load(open(os.path.join(hf_dir, "tokenizer.json")))
    inv = {i: t for t, i in tok["model"]["vocab"].items()}
    for a in tok["added_tokens"]: inv[a["id"]] = a["content"]
    toks = [inv.get(i, f"<unused{i}>") for i in range(CFG["vocab_size"])]
    wr.add_tokenizer_model("gpt2"); wr.add_token_list(toks); wr.add_token_types([3 if t in ("<s>", "</s>") else 1 for t in toks])
    wr.add_token_merges(tok["model"]["merges"]); wr.add_bos_token_id(CFG["bos_token_id"]); wr.add_eos_token_id(CFG["eos_token_id"])
    return wr

def finish(wr): wr.write_header_to_file(); wr.write_kv_data_to_file(); wr.write_tensors_to_file(); wr.close()

def write_model(path, W, hf_dir, qt, extra=None):
    """Weight matrices in type qt. Returns (the model as FORGE decodes it, {forge tensor name: dequantised}, packer oracle)."""
    import gguf
    T = gguf.GGMLQuantizationType
    wr = writer(path, hf_dir); Wd = {}; deq = {}; who = "numpy"
    for k, v in W.items():
        v = v.astype(np.float32)
        if k in TOP: g = TOP[k]; fname = FORGE_NAME.get(g)
        else:
            mid = ".".join(k.split(".")[3:-1]); g = f"blk.{k.split('.')[2]}.{HF2G[mid]}.weight"
            fname = f"layer.{k.split('.')[2]}.{FORGE_NAME[HF2G[mid]]}" if HF2G[mid] in FORGE_NAME else None
        if v.ndim == 2 and qt == "F64": wr.add_tensor(g, v.astype(np.float64)); Wd[k] = v
        elif v.ndim == 2 and qt != "F32":
            raw, dq, who = quantise(v, qt)
            wr.add_tensor(g, raw, raw_shape=raw.shape, raw_dtype=getattr(T, qt)); Wd[k] = dq; deq[fname] = dq
        else: wr.add_tensor(g, v); Wd[k] = v
    for name, (arr, kw) in (extra or {}).items(): wr.add_tensor(name, arr, **kw)
    finish(wr)
    return {k: v.astype(np.float64) for k, v in Wd.items()}, deq, who

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--forge", required=True); ap.add_argument("--work", default="build/golden_gguf_types")
    a = ap.parse_args(); run = tl.run
    try: import gguf
    except ImportError: print("[skip] gguf python package not installed"); sys.exit(77)
    T = gguf.GGMLQuantizationType
    os.makedirs(a.work, exist_ok=True)
    hf = os.path.join(a.work, "hf"); W = make_model(hf)
    fails = 0
    def check(ok, msg):
        nonlocal fails
        print(f"[{'ok' if ok else 'FAIL'}] {msg}"); fails += not ok
    def forge(*args, expect_fail=False):
        cmd = [a.forge, *args]
        runner = os.environ.get("FORGE_RUNNER", "").split()
        r = subprocess.run(runner + cmd, capture_output=True, text=True, errors="replace")
        if (r.returncode != 0) != expect_fail: print("UNEXPECTED exit", r.returncode, ":", " ".join(cmd), "\n", r.stdout, r.stderr); sys.exit(1)
        return r
    def dump(fp, name):
        pf = os.path.join(a.work, "dump.f32"); run([a.forge, "dump", fp, name, pf]); return np.fromfile(pf, dtype=np.float32)
    def codecs(fp):  # {tensor: codec} from the tensor index
        out = {}
        for ln in run([a.forge, "inspect", fp, "--tensors"]).splitlines():
            p = ln.split()
            if len(p) >= 2 and (p[0].startswith(("layer.", "extra.")) or p[0] in ("tok_emb", "lm_head", "out_norm")): out[p[0]] = p[1]
        return out
    def manifest(fp):
        txt = run([a.forge, "inspect", fp]); i = txt.index("{"); return json.JSONDecoder().raw_decode(txt[i:])[0]
    prompt = [290, 72, 101, 108, 108, 111, 44, 32, 116, 104, 101, 32, 119]
    def logits(fp):
        lf = fp + ".logits.f32"; run([a.forge, "run", fp, "--tokens", ",".join(map(str, prompt)), "-n", "1", "--ids", "--logits", lf])
        return np.fromfile(lf, dtype=np.float32)
    ref32 = forward(W, prompt)[-1]

    # ---- 1. probes ----
    rng = np.random.default_rng(7); probes = {}; want = {}
    for qt in BLOCK_TYPES:   # random bytes exercise every bit pattern, including scales that decode to inf / nan
        t = getattr(T, qt); bs, bb = gguf.GGML_QUANT_SIZES[t]
        q = rng.integers(0, 256, size=(4, 1024 // bs * bb), dtype=np.uint8)
        probes[f"probe_{qt}"] = (q, dict(raw_shape=q.shape, raw_dtype=t))
        with np.errstate(all="ignore"): want[f"probe_{qt}"] = (gguf.quants.dequantize(q, t).reshape(-1), "gguf-py dequantize, random blocks")
    for dt in (np.float64, np.int8, np.int16, np.int32, np.int64):
        v = rng.normal(0, 3e4 if dt != np.int8 else 40, size=(3, 40)).astype(dt)
        if dt in (np.int32, np.int64): v[0, :4] = [2**24 + 1, -(2**24) - 3, 2**30 + 7, -5]
        if dt == np.int64: v[0, 4:6] = [2**40 + 12345, -(2**53) - 1]
        n = f"probe_{np.dtype(dt).name}"; probes[n] = (v, {}); want[n] = (v.astype(np.float32).reshape(-1), "numpy cast to float32")
    q8k = np.zeros((2, 292), np.uint8); qs = rng.integers(-128, 128, size=(2, 256), dtype=np.int8); dk = np.array([0.0123, -3.5e-4], np.float32)
    q8k[:, :4] = dk.view(np.uint8).reshape(2, 4); q8k[:, 4:260] = qs.view(np.uint8); q8k[:, 260:] = rng.integers(0, 256, size=(2, 32), dtype=np.uint8)
    probes["probe_Q8_K"] = (q8k, dict(raw_shape=(2, 292), raw_dtype=T.Q8_K)); want["probe_Q8_K"] = ((dk[:, None] * qs.astype(np.float32)).reshape(-1), "d * q computed here (no gguf-py dequantiser)")
    gp = os.path.join(a.work, "probes.gguf"); fp = os.path.join(a.work, "probes.fpk")
    write_model(gp, W, hf, "F32", probes)
    r = forge("import", gp, "-o", fp, "--allow-unused")   # the probes are leftovers: kept as extra.<name>
    cd = codecs(fp); man = manifest(fp)
    for n, (w_, who) in want.items():
        got = dump(fp, "extra." + n)
        same = got.shape == w_.shape and (np.array_equal(got, w_, equal_nan=True))
        check(same, f"{n[6:]:7s} decode == {who}  (stored as {cd.get('extra.' + n)})")
    native = {"Q4_0": "q4_0", "Q4_1": "q4_1", "Q5_0": "q5_0", "Q5_1": "q5_1", "Q8_0": "q8_0", "Q2_K": "q2_k", "Q3_K": "q3_k", "Q4_K": "q4_k", "Q5_K": "q5_k",
              "Q6_K": "q6_k", "IQ4_NL": "iq4_nl", "IQ4_XS": "iq4_xs", "TQ1_0": "tq1_0", "TQ2_0": "tq2_0"}
    check(all(cd.get(f"extra.probe_{k}") == v for k, v in native.items()), "block types are stored verbatim under the matching FORGE codec")
    tr = {e["from"]: e for e in man.get("transcoded", [])}
    check(set(tr) == {"F64", "I8", "I16", "I32", "I64", "Q8_K"} and all(e["to"] == "f32" for e in tr.values()) and "transcoded" in r.stderr,
          f"transcoded types are recorded in the manifest and printed: {sorted(tr)}")
    check(man.get("gguf_tensor_types", {}).get("Q3_K") == 1 and "requantised" not in man, "manifest lists the GGUF tensor types")

    # ---- 2. models ----
    for qt in MODEL_TYPES:
        gp = os.path.join(a.work, f"m_{qt}.gguf"); fp = os.path.join(a.work, f"m_{qt}.fpk")
        Wd, deq, who = write_model(gp, W, hf, qt)
        forge("import", gp, "-o", fp)
        exact = all(np.array_equal(dump(fp, n), v.reshape(-1), equal_nan=True) for n, v in deq.items())
        ref = forward(Wd, prompt)[-1]; got = logits(fp)
        err = np.abs(got - ref).max(); cos = got @ ref32 / np.linalg.norm(got) / np.linalg.norm(ref32)
        check((exact or qt == "F64") and err < 5e-3, f"model {qt:7s} ({who}): {len(deq)} tensors == gguf-py dequantize: {exact}; logits vs dequantised reference "
              f"max|err| {err:.1e}; cosine to the f32 model {cos:.4f}")

    # ---- 3. --codec ----
    gp = os.path.join(a.work, "m_Q4_0.gguf"); Wd, _, _ = write_model(gp, W, hf, "Q4_0"); ref = forward(Wd, prompt)[-1]
    keep = os.path.join(a.work, "rq_keep.fpk"); r = forge("import", gp, "-o", keep, "--codec", "keep")
    cd = codecs(keep)
    check(cd["layer.0.wq"] == "q4_0" and cd["tok_emb"] == "q4_0" and "warning" not in r.stderr and "requantised" not in manifest(keep), "--codec keep stores the GGUF types")
    for codec, emb, thr in (("q8_0", "q8_0", 0.9995), ("q4_k", "q8_0", 0.99), ("f32", "f32", None), ("q5_0", "q8_0", 0.995)):
        fp = os.path.join(a.work, f"rq_{codec}.fpk"); r = forge("import", gp, "-o", fp, "--codec", codec)
        cd = codecs(fp); man = manifest(fp); got = logits(fp)
        err = np.abs(got - ref).max(); cos = got @ ref / np.linalg.norm(got) / np.linalg.norm(ref)
        rq = man.get("requantised", {})
        ok = cd["layer.0.wq"] == codec and cd["lm_head"] == codec and cd["tok_emb"] == emb and cd["layer.0.attn_norm"] == "f32"
        ok = ok and rq.get("codec") == codec and rq.get("from", {}).get("Q4_0") == rq.get("tensors") == 9 and man.get("codec") == codec
        ok = ok and (("double" in r.stderr or "second time" in r.stderr) == (codec != "f32")) and (err < 5e-3 if thr is None else cos > thr)
        check(ok, f"--codec {codec}: Q4_0 re-encoded ({rq.get('tensors')} tensors, double quantised {rq.get('double_quantised')}), warning printed: "
              f"{'warning' in r.stderr}; logits vs the Q4_0 model max|err| {err:.1e} cosine {cos:.5f}")
    gp32 = os.path.join(a.work, "m_F32.gguf"); write_model(gp32, W, hf, "F32"); fp = os.path.join(a.work, "rq32_q4_k.fpk")
    r = forge("import", gp32, "-o", fp, "--codec", "q4_k", "--emb-codec", "q4_k"); cd = codecs(fp); got = logits(fp)
    cos = got @ ref32 / np.linalg.norm(got) / np.linalg.norm(ref32)
    check(cd["layer.0.w_up"] == "q4_k" and cd["tok_emb"] == "q4_k" and "warning" not in r.stderr and cos > 0.98, f"--codec q4_k on an F32 GGUF: no double-quantisation warning, cosine {cos:.4f}")
    r = forge("import", gp32, "-o", fp, "--codec", "q4_k_m", expect_fail=True)
    check("Hugging Face" in r.stderr + r.stdout, "--codec q4_k_m on a GGUF is refused with a reason")

    # ---- 4. refused types, malformed sizes ----
    for qt in REFUSED:
        t = getattr(T, qt, None)
        if t is None: print(f"[skip] gguf-py has no {qt}"); continue
        bs, bb = gguf.GGML_QUANT_SIZES[t]
        gp = os.path.join(a.work, "refused.gguf"); wr = writer(gp, hf)
        q = np.zeros((CFG["vocab_size"] * CFG["hidden_size"] // bs, bb), np.uint8)
        wr.add_tensor("token_embd.weight", q, raw_shape=(CFG["vocab_size"], CFG["hidden_size"] // bs * bb), raw_dtype=t); finish(wr)
        r = forge("import", gp, "-o", os.path.join(a.work, "refused.fpk"), expect_fail=True)
        check(qt in r.stderr + r.stdout and "not supported" in r.stderr + r.stdout, f"{qt} is refused by name: {(r.stderr + r.stdout).strip().splitlines()[-1][:150]}")
    print("golden gguf types:", "PASS" if fails == 0 else f"{fails} FAILURE(S)")
    sys.exit(1 if fails else 0)

if __name__ == "__main__": main()

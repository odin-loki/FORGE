"""Shared pieces of the tiny-model golden tests (tiny_gpt2 / tiny_gemma / tiny_phi3 / tiny_bert / tiny_vit /
tiny_whisper): writing a random HF checkpoint, float64 numpy building blocks, and the FORGE-vs-reference checks."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny_llama as tl

run = tl.run
rms = tl.rms
rope = tl.rope
_erf = np.vectorize(math.erf)

def gelu(x): return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))
def gelu_tanh(x): return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
def quick_gelu(x): return x / (1.0 + np.exp(-1.702 * x))
def silu(x): return x / (1.0 + np.exp(-x))
def layer_norm(x, w, b, eps):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b

def causal_mask(s, window=0):
    """Additive [s, s] mask: -inf above the diagonal, and (window > 0) where the key is `window` or more behind."""
    i, j = np.arange(s)[:, None], np.arange(s)[None, :]
    bad = j > i
    if window > 0: bad |= (i - j) >= window
    return np.where(bad, -np.inf, 0.0)

def attend(q, k, v, mask=None, scale=None, softcap=0.0):
    """q [s, h, d], k [t, hk, d], v [t, hk, dv] (hk divides h: grouped-query) -> [s, h, dv]; mask adds to [h, s, t]."""
    g = q.shape[1] // k.shape[1]
    k = np.repeat(k, g, axis=1); v = np.repeat(v, g, axis=1)
    sc = np.einsum("shd,thd->hst", q, k) * (scale if scale is not None else 1.0 / math.sqrt(q.shape[-1]))
    if softcap > 0: sc = softcap * np.tanh(sc / softcap)
    if mask is not None: sc = sc + mask
    sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
    return np.einsum("hst,thd->shd", sc, v)

def write_tokenizer(d, kind="BPE"):
    """tiny_llama's byte-level BPE tokenizer.json (ids < 300), or a WordPiece-typed file the runtime tokenizer skips."""
    os.makedirs(d, exist_ok=True)
    if kind != "BPE":
        vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + [f"tok{i}" for i in range(296)])}
        json.dump({"version": "1.0", "model": {"type": kind, "vocab": vocab, "unk_token": "[UNK]"}}, open(os.path.join(d, "tokenizer.json"), "w"))
        return
    b2u = tl.bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    merges = [("t", "h"), ("th", "e"), ("Ġ", "t"), ("Ġt", "he"), ("i", "n"), ("Ġ", "a"), ("e", "r"), ("o", "n"),
              ("Ġ", "s"), ("r", "e"), ("a", "n"), ("in", "g"), ("Ġ", "w"), ("o", "r"), ("Ġ", "f"), ("Ġ", "o")]
    for a, b in merges: vocab[a + b] = len(vocab)
    added = [{"id": 290, "content": "<s>", "special": True}, {"id": 291, "content": "</s>", "special": True}]
    json.dump({"version": "1.0", "added_tokens": added, "model": {"type": "BPE", "vocab": vocab, "merges": [f"{a} {b}" for a, b in merges]}},
              open(os.path.join(d, "tokenizer.json"), "w"))

def save_model(d, W, cfg, tokenizer="BPE"):
    """Write config.json + model.safetensors (+ tokenizer.json); returns the weights as float64 after the f32 rounding."""
    from safetensors.numpy import save_file
    os.makedirs(d, exist_ok=True)
    W = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in W.items()}
    save_file(W, os.path.join(d, "model.safetensors"))
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    if tokenizer: write_tokenizer(d, tokenizer)
    return {k: v.astype(np.float64) for k, v in W.items()}

def write_gguf(path, arch, meta, tensors):
    """A GGUF with `<arch>.<key>` metadata (int -> u32, float -> f32, str, bool) and F32 tensors. None if gguf-py is missing."""
    try:
        import gguf
    except ImportError:
        return None
    wr = gguf.GGUFWriter(path, arch)
    for k, v in meta.items():
        key = k if k.startswith("tokenizer.") or k.startswith("general.") else f"{arch}.{k}"
        if isinstance(v, bool): wr.add_bool(key, v)
        elif isinstance(v, int): wr.add_uint32(key, v)
        elif isinstance(v, float): wr.add_float32(key, v)
        else: wr.add_string(key, v)
    for k, v in tensors.items(): wr.add_tensor(k, np.ascontiguousarray(v, dtype=np.float32))
    wr.write_header_to_file(); wr.write_kv_data_to_file(); wr.write_tensors_to_file(); wr.close()
    return path

class Checker:
    def __init__(self, forge, work):
        self.forge, self.work, self.fails = forge, work, 0
        os.makedirs(work, exist_ok=True)

    def ok(self, cond, msg):
        print(f"[{'ok' if cond else 'FAIL'}] {msg}"); self.fails += not cond
        return cond

    def imp(self, src, name, codec="f32", extra=()):
        out = os.path.join(self.work, name)
        run([self.forge, "import", src, "-o", out] + (["--codec", codec] if codec else []) + list(extra))
        run([self.forge, "inspect", out, "--verify"])
        return out

    def logits(self, fpk, prompt, ref, label, tol=2e-3, cosmin=None, gen_n=1):
        """Prefill logits of `forge run` against the reference's last row; returns the generated ids."""
        lf = fpk + ".logits.f32"
        gen = run([self.forge, "run", fpk, "--tokens", ",".join(map(str, prompt)), "-n", str(gen_n), "--ids", "--logits", lf])
        got = np.fromfile(lf, dtype=np.float32).astype(np.float64)
        if got.shape != ref.shape: return self.ok(False, f"{label}: {got.shape} logits, expected {ref.shape}") and []
        err = np.abs(got - ref).max(); cos = got @ ref / np.linalg.norm(got) / np.linalg.norm(ref)
        self.ok(cos > cosmin if cosmin else err < tol, f"{label}: prefill logits max|err| {err:.2e}, cosine {cos:.6f}")
        return [int(t) for t in gen.split()]

    def decoder(self, hf, name, forward, prompt, eos, q8=True, gen=16, q8_cos=0.999):
        """HF import in f32 (logits + KV-cached greedy decode vs full recompute) and q8_0 (logits cosine)."""
        ref = forward(prompt)[-1]
        fp = self.imp(hf, f"{name}_f32.fpk")
        fg = self.logits(fp, prompt, ref, f"{name} hf f32", gen_n=gen)
        ids, ref_gen = list(prompt), []
        for _ in range(gen):
            t = int(np.argmax(forward(ids)[-1])); ref_gen.append(t); ids.append(t)
            if t == eos: break
        same = fg == ref_gen
        if not same and fg:   # a near-tie between the top two logits may legitimately flip in f32
            i = next((i for i in range(min(len(fg), len(ref_gen))) if fg[i] != ref_gen[i]), None)
            if i is not None: lg = np.sort(forward(prompt + ref_gen[:i])[-1]); same = lg[-1] - lg[-2] < 1e-3
        self.ok(same, f"{name}: greedy decode matches the reference: forge {fg[:8]}... ref {ref_gen[:8]}...")
        if q8: self.logits(self.imp(hf, f"{name}_q8_0.fpk", "q8_0"), prompt, ref, f"{name} hf q8_0", cosmin=q8_cos)
        return fp

    def embed(self, fpk, args, ref, label, tol=1e-3, cosmin=None):
        """`forge embed <fpk> <args> --out` against the reference vector (any shape, compared flat)."""
        of = fpk + "." + label.replace(" ", "_").replace("/", "_") + ".f32"
        run([self.forge, "embed", fpk] + list(args) + ["--out", of])
        got = np.fromfile(of, dtype=np.float32).astype(np.float64); ref = np.asarray(ref, dtype=np.float64).reshape(-1)
        if got.shape != ref.shape: return self.ok(False, f"{label}: {got.shape[0]} values, expected {ref.shape[0]}")
        err = np.abs(got - ref).max(); cos = got @ ref / np.linalg.norm(got) / np.linalg.norm(ref)
        return self.ok(cos > cosmin if cosmin else err < tol, f"{label}: max|err| {err:.2e}, cosine {cos:.6f}")

    def done(self, name):
        print(f"golden {name}:", "PASS" if self.fails == 0 else f"{self.fails} FAILURE(S)")
        sys.exit(1 if self.fails else 0)

def args():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--forge", required=True); ap.add_argument("--work", required=True)
    return ap.parse_args()

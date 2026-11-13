#!/usr/bin/env python3
"""Regenerate the binary seed corpus (tiny .fpk / .safetensors / .gguf / tokenizer.json). Usage: make_corpus.py <forge>
(make_corpus.py --gguf-types-only rewrites only the hand-written GGUF tensor-type seeds, which need neither forge nor gguf-py)."""
import json, os, subprocess, sys, tempfile
import numpy as np
from safetensors.numpy import save_file
forge = sys.argv[1]; here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
rng = np.random.default_rng(0)
d = tempfile.mkdtemp()
cfg = dict(architectures=["LlamaForCausalLM"], hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
           intermediate_size=8, vocab_size=20, rms_norm_eps=1e-5, max_position_embeddings=16, bos_token_id=1, eos_token_id=2)
W = {"model.embed_tokens.weight": rng.normal(size=(20, 8)), "model.norm.weight": np.ones(8), "lm_head.weight": rng.normal(size=(20, 8))}
p = "model.layers.0."
for n, s in [("input_layernorm", (8,)), ("post_attention_layernorm", (8,)), ("self_attn.q_proj", (8, 8)), ("self_attn.k_proj", (4, 8)),
             ("self_attn.v_proj", (4, 8)), ("self_attn.o_proj", (8, 8)), ("mlp.gate_proj", (8, 8)), ("mlp.up_proj", (8, 8)), ("mlp.down_proj", (8, 8))]:
    W[p + n + ".weight"] = rng.normal(size=s)
W = {k: v.astype(np.float32) for k, v in W.items()}
save_file(W, os.path.join(d, "model.safetensors")); json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
tok = {"model": {"type": "BPE", "vocab": {c: i for i, c in enumerate("abcdefghijklmnopqrst")}, "merges": ["a b", "c d"]},
       "added_tokens": [{"id": 1, "content": "<s>", "special": True}]}
json.dump(tok, open(os.path.join(d, "tokenizer.json"), "w"))
open(os.path.join(here, "tokenizer", "tok.json"), "w").write(json.dumps(tok) + "\x01abcd ab")
# The other tokenizer seeds (one per model type and pipeline stage, a SentencePiece model behind a \x02 byte) and the
# chat_template seeds are copies of the test fixtures: see tests/data/tokenizer and tests/data/chat_template.
save_file({"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.ones(4, dtype=np.float16)}, os.path.join(here, "safetensors", "small.safetensors"))
if forge != "--gguf-types-only":   # the seeds below need the forge binary and the gguf package
    subprocess.run([forge, "import", d, "-o", os.path.join(here, "fpk", "tiny.fpk"), "--codec", "q8_0"], check=True, capture_output=True)
    subprocess.run([forge, "keygen", "-o", os.path.join(d, "k")], check=True, capture_output=True)
    subprocess.run([forge, "sign", os.path.join(here, "fpk", "tiny.fpk"), "--key", os.path.join(d, "k.key"), "-o", os.path.join(here, "fpk", "tiny_signed.fpk")], check=True, capture_output=True)
    import gguf
    w = gguf.GGUFWriter(os.path.join(here, "gguf", "small.gguf"), "llama")
    w.add_block_count(1); w.add_token_list(["a", "b"]); w.add_tensor("t", np.ones((2, 32), np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

# GGUF seeds for the tensor-type paths (written by hand, so malformed tables are possible): one valid block or two of
# every loadable ggml type, then a block count that is not whole, a truncated data section, and dimensions / offsets
# that overflow.
import struct
def raw_gguf(path, tensors, cut=None):
    """tensors: [(name, ggml type, dims, offset or None, data bytes)]"""
    b = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 0); off = 0; blob = b""
    for name, ty, dims, o, data in tensors:
        b += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        b += struct.pack("<IQ", ty, off if o is None else o)
        blob += data + b"\0" * (-len(data) % 32); off = len(blob)
    b += b"\0" * (-len(b) % 32) + blob
    open(path, "wb").write(b if cut is None else b[:cut])
BLOCKS = {0: (1, 4), 1: (1, 2), 30: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 10: (256, 84), 11: (256, 110),
          12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292), 20: (32, 18), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4),
          27: (1, 8), 28: (1, 8), 34: (256, 54), 35: (256, 66)}
r2 = np.random.default_rng(3)
def blocks(ty, n): return r2.integers(0, 256, size=n // BLOCKS[ty][0] * BLOCKS[ty][1], dtype=np.uint8).tobytes()
G = os.path.join(here, "gguf")
raw_gguf(os.path.join(G, "types.gguf"), [(f"t{ty}", ty, [max(bs, 32), 2], None, blocks(ty, 2 * max(bs, 32))) for ty, (bs, bb) in BLOCKS.items()])
raw_gguf(os.path.join(G, "types_cut.gguf"), [(f"t{ty}", ty, [256], None, blocks(ty, 256)) for ty in (11, 23, 34, 2)], cut=700)
raw_gguf(os.path.join(G, "bad_blocks.gguf"), [("a", 12, [300], None, blocks(12, 512)), ("b", 2, [31, 3], None, blocks(2, 96)), ("c", 10, [256, 3], None, blocks(10, 256))])
raw_gguf(os.path.join(G, "overflow.gguf"), [("a", 2, [1 << 40, 1 << 40, 1 << 40], None, blocks(2, 32)), ("b", 14, [256], (1 << 64) - 64, b""),
                                            ("c", 27, [(1 << 63) + 5], None, b""), ("d", 16, [256], None, bytes(66)), ("e", 0, [1 << 61, 8], 1 << 62, b"")])

# fuzz_aot: structure-aware plan-blob seeds derived from corpus/aot/seed.fplan (see aot_seeds.py, standard library only;
# `python3 fuzz/aot_seeds.py` runs it alone). They are many; keep the ones libFuzzer's -merge=1 says add coverage.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aot_seeds
aot_seeds.main(["aot_seeds.py"])
for k in os.listdir(here): print(k, os.listdir(os.path.join(here, k)))

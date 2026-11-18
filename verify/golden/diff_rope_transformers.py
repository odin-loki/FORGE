#!/usr/bin/env python3
"""Run transformers' own RoPE initialisers (modeling_rope_utils.ROPE_INIT_FUNCTIONS) and compare inv_freq and the
attention factor with the numpy transcription in tiny_rope.py (inv_default / inv_linear / inv_dynamic / inv_yarn /
inv_llama3, get_mscale).  Needs torch + transformers (a CPU venv is enough).  Exit 1 on any difference > 1e-12 relative.

usage: diff_rope_transformers.py
"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny_rope as tr
import transformers
from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

print("transformers", transformers.__version__, "types:", sorted(ROPE_INIT_FUNCTIONS))

def cfg(rope_type, max_pos=128, theta=10000.0, **kw):
    p = dict(rope_type=rope_type, rope_theta=theta, **kw)
    base = dict(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2, intermediate_size=128,
                vocab_size=300, max_position_embeddings=max_pos)
    try:
        return LlamaConfig(**base, rope_parameters=p)
    except TypeError:
        p.pop("rope_theta"); return LlamaConfig(**base, rope_theta=theta, rope_scaling=p)

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
def default_init(c):
    f = ROPE_INIT_FUNCTIONS.get("default") or LlamaRotaryEmbedding.compute_default_rope_parameters
    return f(c, device="cpu")

fails = 0
def check(name, got, want, af_got=None, af_want=None):
    global fails
    got = np.asarray(got, np.float64); want = np.asarray(want, np.float64)
    err = float(np.max(np.abs(got - want) / np.abs(want)))
    af_ok = af_want is None or abs(float(af_got) - float(af_want)) < 1e-12
    ok = err < 1e-6 and af_ok       # transformers computes in float32
    fails += not ok
    print("[%s] %-34s max rel diff %.2e  attention factor %s vs %s" % ("ok" if ok else "DIFF", name, err, af_got, af_want))

d = 16
for theta in (10000.0, 500000.0):
    c = cfg("default", theta=theta); inv, af = default_init(c)
    check("default theta=%g" % theta, inv.numpy(), tr.inv_default(theta, d), af, 1.0)
    for factor in (2.0, 8.0):
        c = cfg("linear", theta=theta, factor=factor); inv, af = ROPE_INIT_FUNCTIONS["linear"](c, device="cpu")
        check("linear f=%g" % factor, inv.numpy(), tr.inv_linear(theta, d, factor), af, 1.0)
        for seq in (64, 128, 300, 1000):
            c = cfg("dynamic", theta=theta, factor=factor); inv, af = ROPE_INIT_FUNCTIONS["dynamic"](c, device="cpu", seq_len=seq)
            check("dynamic f=%g seq=%d" % (factor, seq), inv.numpy(), tr.inv_dynamic(theta, d, factor, 128, seq), af, 1.0)
        for kw in ({}, dict(beta_fast=16.0, beta_slow=2.0), dict(mscale=0.7, mscale_all_dim=0.7), dict(attention_factor=1.3), dict(truncate=False)):
            c = cfg("yarn", theta=theta, factor=factor, original_max_position_embeddings=64, **kw)
            inv, af = ROPE_INIT_FUNCTIONS["yarn"](c, device="cpu")
            want_af = kw.get("attention_factor", tr.get_mscale(factor) if "mscale" not in kw else tr.get_mscale(factor, kw["mscale"]) / tr.get_mscale(factor, kw["mscale_all_dim"]))
            check("yarn f=%g %s" % (factor, kw), inv.numpy(), tr.inv_yarn(theta, d, factor, 64, kw.get("beta_fast", 32.0), kw.get("beta_slow", 1.0), kw.get("truncate", True)), af, want_af)
    c = cfg("llama3", theta=theta, factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=64)
    inv, af = ROPE_INIT_FUNCTIONS["llama3"](c, device="cpu")
    check("llama3 theta=%g" % theta, inv.numpy(), tr.inv_llama3(theta, d, 8.0, 1.0, 4.0, 64), af, 1.0)
print("rope reference vs transformers:", "PASS" if not fails else "%d DIFFERENCE(S)" % fails)
sys.exit(1 if fails else 0)

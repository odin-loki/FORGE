#!/usr/bin/env python3
"""Diff the GGUF tensor names import/gguf_arch.hpp reads (gemma, gemma2, gemma3, phi3, gpt2, bert) against
llama.cpp's gguf-py/gguf/constants.py (TENSOR_NAMES and MODEL_TENSORS), and the metadata keys it reads against Keys.

usage: diff_gguf_tensor_names.py constants.py gguf_arch.hpp
Reports: names FORGE reads that llama.cpp's list for that architecture does not contain (mismatch), and llama.cpp tensors
of that architecture FORGE never reads (these must be in the leave-out list or the import reports them as unused).
"""
import re, sys
c = open(sys.argv[1], encoding="utf-8").read()
h = open(sys.argv[2], encoding="utf-8").read()

# TENSOR_NAMES: MODEL_TENSOR.X: "name"
tn = dict(re.findall(r"MODEL_TENSOR\.(\w+):\s+\"([^\"]+)\"", c[c.index("TENSOR_NAMES: dict[MODEL_TENSOR, str] = {"):]))
# MODEL_TENSORS: MODEL_ARCH.X: [ MODEL_TENSOR.A, ... ]
ms = c[c.index("MODEL_TENSORS: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {"):]
arch_t = {}
for m in re.finditer(r"MODEL_ARCH\.(\w+):\s*\[(.*?)\]", ms, re.S):
    arch_t.setdefault(m.group(1), re.findall(r"MODEL_TENSOR\.(\w+)", m.group(2)))
def names(arch):
    return {tn[t].replace("{bid}", "{}") for t in arch_t[arch] if t in tn}

# FORGE: split the importer into per-architecture source-name sets
body = h[h.index("put(\"tok_emb\""):h.index("// Every source tensor is accounted for")]
def srcs(txt):
    out = set()
    for pm in re.finditer(r'(P \+ )?"([a-z_0-9.]+)"', txt):
        n = pm.group(2)
        if "." not in n and not pm.group(1): continue
        out.add(("blk.{}." if pm.group(1) else "") + n)
    return out
def with_wb(s):   # pair() names are written without .weight/.bias
    r = set()
    for n in s:
        if n.endswith(".weight") or n.endswith(".bias") or n in ("position_embd.weight",): r.add(n)
        else: r.add(n + ".weight"); r.add(n + ".bias")
    return r
loop = body[body.index("for (i64 l = 0"):body.index('if (arch == "gpt2") { put("pos_emb"')]
blk = {
  "gemma": srcs(loop[loop.index("if (gemma) {"):loop.index('} else if (arch == "phi3")')]),
  "phi3":  srcs(loop[loop.index('} else if (arch == "phi3")'):loop.index('} else if (arch == "gpt2")')]),
  "gpt2":  with_wb(srcs(loop[loop.index('} else if (arch == "gpt2")'):loop.index("} else {\n            pair")])),
  "bert":  with_wb(srcs(loop[loop.index("} else {\n            pair"):])),
}
tail = body[body.index('if (arch == "gpt2") { put("pos_emb"'):]
glob = {"gpt2": {"position_embd.weight", "output_norm.weight", "output_norm.bias", "token_embd.weight", "output.weight"},
        "bert": {"position_embd.weight", "token_types.weight", "token_embd_norm.weight", "token_embd_norm.bias", "token_embd.weight"},
        "other": {"output_norm.weight", "token_embd.weight", "output.weight"}}
rc = 0
def cmp(arch, up_arch, forge_names):
    global rc
    up = names(up_arch)
    # upstream lists tensors without suffix: expand .weight/.bias
    up_full = set()
    for n in up: up_full |= {n + ".weight", n + ".bias"} if not n.endswith((".weight", ".bias")) else {n}
    bad = sorted(n for n in forge_names if n not in up_full and n.rsplit(".", 1)[0] not in up_full and n.rsplit(".", 1)[0] not in up)
    unread = sorted(n for n in up if not any(f == n or f.rsplit(".", 1)[0] == n or f == n + ".weight" for f in forge_names))
    print("%-7s (llama.cpp %s): FORGE reads %d names; not in llama.cpp's list: %s" % (arch, up_arch, len(forge_names), bad or "none"))
    print("        llama.cpp tensors of this arch FORGE never reads: %s" % (unread or "none"))
    if bad: rc = 1
for arch, up_arch in (("gemma", "GEMMA"), ("gemma2", "GEMMA2"), ("gemma3", "GEMMA3"), ("phi3", "PHI3"), ("gpt2", "GPT2"), ("bert", "BERT")):
    f = set(blk["gemma" if arch.startswith("gemma") else arch]) | glob.get(arch, glob["other"])
    if arch == "gemma": f -= {"blk.{}.post_attention_norm.weight", "blk.{}.post_ffw_norm.weight", "blk.{}.attn_q_norm.weight", "blk.{}.attn_k_norm.weight"}
    if arch == "gemma2": f -= {"blk.{}.attn_q_norm.weight", "blk.{}.attn_k_norm.weight"}
    cmp(arch, up_arch, f)

# metadata keys read via K("...") / Kor("...", ...)
keys = set(re.findall(r'\bKor?\("([a-z_.]+)"', h))
upkeys = set(re.findall(r'"\{arch\}\.([a-z_.]+)"', c))
print("metadata keys read by FORGE not defined in llama.cpp Keys: %s" % (sorted(k for k in keys if k not in upkeys) or "none"))
sys.exit(rc)

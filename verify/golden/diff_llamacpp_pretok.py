#!/usr/bin/env python3
"""Mechanical diff of FORGE's gguf_pre_regexes() table against llama.cpp's src/llama-vocab.cpp.

usage: diff_llamacpp_pretok.py UPSTREAM_llama-vocab.cpp FORGE_DUMP.json
FORGE_DUMP.json comes from verify/golden/dump_pretok.cpp (a program that prints
gguf_pre_regexes() for every name).  Prints one line per mismatch and exits 1 if any.
"""
import json, re, sys

up = open(sys.argv[1], encoding="utf-8").read()
forge = json.load(open(sys.argv[2], encoding="utf-8"))

STR = re.compile(r'"((?:[^"\\]|\\.)*)"')

def cunescape(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            m = {"\\": "\\", '"': '"', "n": "\n", "r": "\r", "t": "\t", "'": "'"}
            if n in m:
                out.append(m[n]); i += 2; continue
            raise SystemExit("unhandled C escape \\" + n)
        out.append(c); i += 1
    return "".join(out)

def norm(r):
    # \x{HEX} and raw non-ASCII characters both -> \x{lowerhex}
    r = re.sub(r"\\x\{([0-9a-fA-F]+)\}", lambda m: "\\x{%x}" % int(m.group(1), 16), r)
    r = r.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    return "".join(c if ord(c) < 128 else "\\x{%x}" % ord(c) for c in r)

# FORGE's empty list means "the ByteLevel pre-tokenizer's own regex", which is exactly llama.cpp's GPT2 expression.
BYTELEVEL = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"

def firstdiff(a, b):
    n = min(len(a), len(b))
    i = next((k for k in range(n) if a[k] != b[k]), n)
    return i, a[max(0, i - 25):i + 40], b[max(0, i - 25):i + 40]

# ---- upstream: pre_type -> regex list
bpe = up[up.index("struct llm_tokenizer_bpe : llm_tokenizer"):]
bpe = bpe[:bpe.index("std::vector<std::string> regex_exprs;")] if "std::vector<std::string> regex_exprs;" in bpe else bpe
sw = bpe[bpe.index("switch (vocab.get_pre_type())"):]
sw = sw[:sw.index("default:")] if False else sw
type_regex = {}
cases = re.finditer(r"((?:\s*case LLAMA_VOCAB_PRE_TYPE_\w+:\s*)+)(.*?)break;", sw, re.S)
for m in cases:
    labels = re.findall(r"LLAMA_VOCAB_PRE_TYPE_(\w+)", m.group(1))
    body = m.group(2)
    if "regex_exprs" not in body:
        for l in labels: type_regex.setdefault(l, None)
        continue
    body = re.sub(r"//[^\n]*", "", body)
    inner = body[body.index("{") + 1: body.rindex("}")]
    exprs, cur, pos = [], None, 0
    for tm in re.finditer(r'"((?:[^"\\]|\\.)*)"|(,)', inner):
        if tm.group(2):
            if cur is not None: exprs.append(cur); cur = None
        else:
            cur = (cur or "") + cunescape(tm.group(1))
    if cur is not None: exprs.append(cur)
    for l in labels: type_regex[l] = [norm(e) for e in exprs]

# ---- upstream: pre name -> (pre_type, ignore_merges)
start = up.index('tokenizer_pre == "default"')
chain = up[start - 400: up.index("add_space_prefix = true;", start)]
name_info = {}
for blk in re.split(r"\}\s*else\s+if\s*\(", chain):
    names = re.findall(r'tokenizer_pre\s*==\s*"([^"]+)"', blk.split("{")[0])
    pt = re.search(r"pre_type\s*=\s*LLAMA_VOCAB_PRE_TYPE_(\w+)", blk)
    if not names or not pt: continue
    ig = bool(re.search(r"ignore_merges\s*=\s*true", blk))
    for n in names: name_info[n] = (pt.group(1), ig)

mism, ok, missing, extra = [], 0, [], []
for name, f in sorted(forge.items()):
    if not f["found"]:
        continue
    if name not in name_info:
        extra.append(name); continue
    pt, ig = name_info[name]
    want = type_regex.get(pt)
    if want is None:
        want = []        # no regex_exprs case -> the ByteLevel default
    fr = [norm(r) for r in f["re"]]
    if not fr: fr = [BYTELEVEL]
    if not want: want = [BYTELEVEL]
    bad = []
    if fr != want:
        if len(fr) != len(want):
            bad.append("regex list length: forge=%d upstream(%s)=%d" % (len(fr), pt, len(want)))
        for k, (a, b) in enumerate(zip(fr, want)):
            if a != b:
                i, x, y = firstdiff(a, b)
                bad.append("regex[%d] differs at char %d: forge ...%s... | upstream(%s) ...%s..." % (k, i, x, pt, y))
    if f["ignore_merges"] != ig:
        bad.append("ignore_merges: forge=%s upstream=%s (%s)" % (f["ignore_merges"], ig, pt))
    if bad:
        mism.append((name, bad))
    else:
        ok += 1
for name in sorted(name_info):
    if not forge.get(name, {}).get("found"):
        missing.append((name, name_info[name]))

print("forge names found: %d, identical to upstream: %d" % (sum(1 for f in forge.values() if f["found"]), ok))
for name, bad in mism:
    print("MISMATCH", name)
    for b in bad: print("   ", b)
for n in extra: print("IN FORGE, NOT UPSTREAM:", n)
for n, v in missing: print("IN UPSTREAM, NOT FORGE: %s -> %s" % (n, v))
if __import__("os").environ.get("DUMP"): json.dump({"type_regex":type_regex,"name_info":name_info},open("upstream_tables.json","w"),indent=1)
sys.exit(1 if mism or extra else 0)

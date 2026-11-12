#!/usr/bin/env python3
"""Builds ops/intrinsics/x86.json from the compiler's own <immintrin.h>. Run by hand:

    python3 ops/intrinsics/build_x86_db.py          # needs clang++-23 (FORGE_RAW_CLANG); g++ and $FORGE_RAW_CLANG_OLD (an older clang++) are tried if present

Functions come from clang's AST (exact signature + target attribute). Function-like macros (the intrinsics
with immediate operands) get their parameter types from the casts in the expanded body; an `int` parameter
that the compiler only accepts as a constant becomes "const int" (a template parameter in the wrapper). The
return type is deduced by the compiler, the CPU features by compiling with a minimal feature set. Every
entry is then compiled with exactly its features; g++ and an older clang are tried too and recorded in "no"."""
import itertools, os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intrin_db as D

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "#include <immintrin.h>\n"

# LLVM feature -> (caps.hpp tag, namespace, -m flag). Features not listed here are outside Tier R (system
# instructions, MMX/3DNow!/XOP, key locker, ...): their intrinsics are dropped with that reason.
F = collections.OrderedDict()
def feat(llvm, tag=None, ns=None, flag=None):
    n = re.sub(r"[.\-]", "_", llvm)
    F[llvm] = {"name": n, "tag": tag or n, "ns": ns or tag or n, "flag": flag or "-m" + llvm}
# order = specificity: the last feature of an intrinsic's set names its namespace
feat("sse", "sse2"); feat("sse2"); feat("sse3", "sse4_2"); feat("ssse3", "sse4_2"); feat("sse4.1", "sse4_2")
feat("sse4.2", "sse4_2"); feat("crc32", "sse4_2"); feat("popcnt"); feat("lzcnt"); feat("bmi", "bmi1"); feat("bmi2")
feat("avx"); feat("f16c"); feat("fma", "fma3", "fma"); feat("avx2")
feat("avx512f"); feat("avx512vl"); feat("avx512dq"); feat("avx512bw"); feat("avx512cd")
for f in ["avx512vnni", "avx512bf16", "avx512fp16", "avx512ifma", "avx512vbmi", "avx512vbmi2", "avx512bitalg",
          "avx512vpopcntdq", "avx512vp2intersect"]: feat(f)
feat("avxvnni"); feat("avxvnniint8"); feat("avxvnniint16"); feat("avxifma"); feat("avxneconvert"); feat("avx10.2")
feat("aes", "aesni"); feat("pclmul", "pclmulqdq"); feat("sha", "shani"); feat("sha512"); feat("sm3"); feat("sm4")
feat("gfni"); feat("vaes"); feat("vpclmulqdq")
feat("amx-tile"); feat("amx-int8"); feat("amx-bf16"); feat("amx-fp16"); feat("amx-complex"); feat("amx-fp8"); feat("amx-avx512")
ORDER = {k: i for i, k in enumerate(F)}
IMPLIED = {"evex512", "no-evex512", "avx10.1", "avx10.1-512", "avx10.2-512", "64bit"}

INT_TYPES = {"int", "unsigned int", "unsigned", "char", "unsigned char", "signed char", "short", "unsigned short",
             "long long", "unsigned long long", "__int64", "unsigned __int64"}
INTERNAL = {"__v4sf": "__m128", "__v2df": "__m128d", "__v8sf": "__m256", "__v4df": "__m256d", "__v16sf": "__m512",
            "__v8df": "__m512d", "__v8hf": "__m128h", "__v16hf": "__m256h", "__v32hf": "__m512h",
            "__v8bf": "__m128bh", "__v16bf": "__m256bh", "__v32bf": "__m512bh"}


def public(t):
    t = " ".join(t.split())
    if t in INTERNAL: return INTERNAL[t]
    m = re.match(r"__v(\d+)(qi|hi|si|di|qu|hu|su|du|qs)$", t)
    if m:
        bits = int(m.group(1)) * {"q": 8, "h": 16, "s": 32, "d": 64}[m.group(2)[0]]
        return {128: "__m128i", 256: "__m256i", 512: "__m512i"}.get(bits, t)
    # east-const pointer spellings -> the documented form
    m = re.match(r"(.*?) const \*$", t)
    if m: t = f"const {m.group(1)} *"
    return t


def norm_type(t):
    """Unaligned typedefs are a clang/GCC detail; the documented parameter type is the aligned name."""
    return re.sub(r"\b(__m(?:128|256|512)[idh]?)_u\b", r"\1", t).replace("const int", "int") if t != "const int" else t


def feats_of(attr):
    return [f.strip() for f in attr.split(",") if f.strip() and f.strip() not in IMPLIED]


def flags_for(features):
    inv = {v["name"]: k for k, v in F.items()}
    return [F[inv[f]]["flag"] for f in features]


def parse_need(msgs):
    """Features named by clang's target-feature diagnostics (first alternative of an a|b requirement)."""
    need = []
    for m in msgs:
        r = re.search(r"needs target feature (.+)$", m)
        if r:
            alt = r.group(1).split("|")[0].strip("() ")
            need += [x.strip("() ") for x in alt.split(",")]
        r = re.search(r"requires target feature '([^']+)'", m)
        if r: need.append(r.group(1))
    return [n for n in need if n not in IMPLIED]


def main():
    dropped = []
    name_re = re.compile(r"^_{1,2}[a-z]\w*$")
    # ---- functions
    fns = [f for f in D.ast_functions([], SRC, name_re=name_re) if not f["name"].startswith("__builtin")]
    seen = set(); cands = []
    for f in fns:
        if f["name"] in seen: continue
        seen.add(f["name"])
        cands.append({"name": f["name"], "ret": norm_type(f["ret"]), "params": [norm_type(p) for p in f["params"]],
                      "llvm": feats_of(f["attrs"]["TargetAttr"]) if "TargetAttr" in f["attrs"] else None})
    print(f"functions: {len(cands)}")
    # ---- macros
    ms = {k: v for k, v in D.macros([], SRC, name_re=name_re).items()
          if v[0] and k not in seen and not k.startswith(("__builtin", "__has", "__cpuid", "__glibc", "__nonnull", "__attribute"))}
    names = sorted(ms)
    exp = D.expand([], SRC, [f"{n}({', '.join(f'__FP{i}__' for i in range(len(ms[n][0])))})" for n in names])
    fn_params = {c["name"]: c["params"] for c in cands}
    mac = []
    for n, body in zip(names, exp):
        ptypes = []
        for i in range(len(ms[n][0])):
            m = re.search(r"\(\s*([A-Za-z_][\w\s\*]*?)\s*\)\s*\(*\s*__FP%d__\b" % i, body)
            ptypes.append(public(m.group(1)) if m else None)
        # a macro that only forwards to a function: take the function's parameter types
        m = re.match(r"^\(?\s*(_\w+)\s*\((.*)\)\s*\)?$", body)
        if m and m.group(1) in fn_params:
            args, cur, d = [], "", 0
            for ch in m.group(2):
                if ch == "(": d += 1
                elif ch == ")": d -= 1
                if ch == "," and d == 0: args.append(cur); cur = ""
                else: cur += ch
            args.append(cur)
            for pos, a in enumerate(args):
                r = re.match(r"^[\s(]*__FP(\d+)__[\s)]*$", a)
                if r and pos < len(fn_params[m.group(1)]) and ptypes[int(r.group(1))] is None:
                    ptypes[int(r.group(1))] = fn_params[m.group(1)][pos]
        mac.append((n, ["int" if t == "const int" else t for t in ptypes]))
    print(f"macros: {len(mac)}")
    # an untyped parameter is an immediate; an int one is passed at run time unless the compiler insists on a constant
    allflags = [v["flag"] for v in F.values()]
    work = []
    for n, pt in mac:
        ps = ["const int" if t is None else t for t in pt]
        work.append({"name": n, "ret": "auto", "params": ps, "imm": [0] * ps.count("const int")})
    failed = D.resolve_immediates(allflags, SRC, work, lambda t: t in INT_TYPES)
    for k, r in failed.items(): dropped.append((work[k]["name"], "macro: does not compile with the parameter types read from its casts: " + r))
    live = [w for k, w in enumerate(work) if k not in failed]
    rets = D.deduce_returns(allflags, SRC, {f"forge_p_{i}": D.call_snippet(w, i) for i, w in enumerate(live)})
    for i, w in enumerate(live):
        r = rets.get(f"forge_p_{i}")
        if r is None: dropped.append((w["name"], "macro: return type not deduced")); continue
        c = {"name": w["name"], "ret": norm_type(public(r)), "params": [t if D.is_imm(t) else norm_type(t) for t in w["params"]], "llvm": None}
        if w.get("imm"): c["imm"] = w["imm"]
        cands.append(c)
    # ---- features for whatever has no target attribute: compile bare, read the diagnostic, repeat
    for c in cands:
        if c["llvm"] is None: c["llvm"] = []
    todo = list(cands)       # the attributed functions are checked here too: an attribute can under-state (macros inside)
    for _ in range(8):
        groups = collections.defaultdict(list)
        for c in todo: groups[tuple(c["llvm"])].append(c)
        nxt = []
        for fs, cs in groups.items():
            if any(f not in F for f in fs):
                continue
            bad = D.probe([F[f]["flag"] for f in fs], SRC, [D.call_snippet(c, i) for i, c in enumerate(cs)], codegen=True)
            for i, msgs in bad.items():
                need = [x for x in parse_need(msgs) if x not in cs[i]["llvm"]]
                if need: cs[i]["llvm"] = cs[i]["llvm"] + need; nxt.append(cs[i])
                else: cs[i]["fail"] = msgs[0]
        todo = nxt
        if not todo: break
    # ---- filter to the Tier R feature set, final validation with exactly the claimed features
    entries = []
    for c in cands:
        if "fail" in c: dropped.append((c["name"], "does not compile: " + c["fail"])); continue
        fs = sorted(set(c["llvm"]), key=lambda f: ORDER.get(f, -1))
        unk = [f for f in fs if f not in F]
        if unk: dropped.append((c["name"], "feature outside Tier R: " + ",".join(unk))); continue
        if not fs:
            if not c["name"].startswith("_mm"): dropped.append((c["name"], "no target feature: generic helper, not a SIMD intrinsic")); continue
            fs = ["sse2"]
        e = {"name": c["name"], "ret": c["ret"], "params": c["params"], "features": [F[f]["name"] for f in fs]}
        if c.get("imm"): e["imm"] = c["imm"]
        entries.append(e)
    groups = collections.defaultdict(list)
    for i, e in enumerate(entries): groups[tuple(e["features"])].append(i)
    keep = [True] * len(entries)
    retry = []
    for fs, ids in groups.items():
        bad = D.probe(flags_for(fs), SRC, [D.call_snippet(entries[i], i) for i in ids], codegen=True)
        for k, msgs in bad.items():
            if "crash" in msgs[0] and "avx512vl" not in fs and any(f.startswith("avx512") for f in fs): retry.append(ids[k])
            else: keep[ids[k]] = False; dropped.append((entries[ids[k]]["name"], "does not compile with its features: " + msgs[0]))
    # a 512-bit intrinsic whose lowering goes through 128/256-bit steps cannot be selected without AVX512VL
    for i in retry:
        fs = sorted(set(entries[i]["features"]) | {"avx512vl"}, key=lambda n: [v["name"] for v in F.values()].index(n))
        if D.probe(flags_for(fs), SRC, [D.call_snippet(entries[i], i)], codegen=True):
            keep[i] = False; dropped.append((entries[i]["name"], "does not compile with its features (backend cannot select)"))
        else: entries[i]["features"] = fs
    entries = [e for e, k in zip(entries, keep) if k]
    hdr = D.header_index([e["name"] for e in entries], only=lambda f: "intrin" in f)
    for e in entries: e["header"] = hdr.get(e["name"], "immintrin.h")
    # ---- other compilers
    def gcc_flags(fs):
        fl = []
        for f in flags_for(fs):
            if f in ("-mcrc32",): continue
            fl.append(f)
        return SRC, fl
    D.classify(entries, lambda e, i: D.call_snippet(e, i),
               [("clang_old", D.CLANG_OLD, lambda fs: (SRC, flags_for(fs))), ("gcc13", "g++", gcc_flags)])
    meta = collections.OrderedDict()
    meta["isa"] = "x86"
    meta["source"] = f"derived from <immintrin.h> of {D.cc_version(D.CLANG)} by ops/intrinsics/build_x86_db.py"
    meta["checked_with"] = [D.cc_version(c) for c in (D.CLANG, D.CLANG_OLD, "g++") if D.have(c)]
    meta["features"] = {v["name"]: {"tag": v["tag"], "ns": v["ns"], "flag": v["flag"]} for v in F.values()}
    D.write_db(os.path.join(HERE, "x86.json"), meta, entries, dropped)
    c = collections.Counter(",".join(e.get("no", [])) for e in entries); print(dict(c))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Builds the Tier R databases of the smaller SIMD ISAs from clang's headers. Run by hand:

    python3 ops/intrinsics/build_misc_db.py [loongarch] [wasm] [mips] [ppc] [s390x]      # default: all

loongarch  <lsxintrin.h>, <lasxintrin.h>: functions from the AST; the immediate-operand macros from the
           `/*type*/` comments of their parameter lists (ui3, si12, ... are immediates).
wasm       <wasm_simd128.h>: functions from the AST (target attribute = simd128 / relaxed-simd / fp16); the
           parameters marked __REQUIRE_CONSTANT become immediates; the four shuffle macros.
mips       <msa.h> is a list of `#define __msa_x __builtin_msa_x`: the signature of each builtin is recovered
           from the compiler's diagnostics (argument count, then one parameter type per round).
ppc        <altivec.h>: every overload of every vec_* function, taken at five levels (AltiVec, VSX, POWER8,
           POWER9, POWER10); an overload belongs to the lowest level that declares it.
s390x      <vecintrin.h>: as ppc, for z13 / z14 / z15 / z17.

Everything is compiled (parse + code generation) with exactly its level before it is written."""
import os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intrin_db as D

HERE = os.path.dirname(os.path.abspath(__file__))
INT = re.compile(r"^(int|unsigned int|unsigned|long|unsigned long|long long|unsigned long long|long int|u?int(8|16|32|64)_t|size_t|signed char|unsigned char|char|short|unsigned short)$")
def int_like(t): return bool(INT.match(t))


def finish(isa, entries, dropped, feats, tgt, prelude, flags_of, source, gcc=None, old=True, key=lambda e: e["name"]):
    """Final check with exactly each entry's features, the other compilers, and the files."""
    groups = collections.defaultdict(list)
    for i, e in enumerate(entries): groups[tuple(e["features"])].append(i)
    keep = [True] * len(entries)
    for fs, ids in groups.items():
        bad = D.probe(tgt + flags_of(fs), prelude, [D.call_snippet(entries[i], i) for i in ids], codegen=True)
        for k, msgs in bad.items(): keep[ids[k]] = False; dropped.append((entries[ids[k]]["name"], "does not compile with its features: " + msgs[0]))
    entries = [e for e, k in zip(entries, keep) if k]
    comp = []
    if old: comp.append(("clang_old", D.CLANG_OLD, lambda fs: (prelude, tgt + flags_of(fs))))
    if gcc: comp.append(("gcc13", gcc[0], lambda fs: (prelude, gcc[1](fs))))
    D.classify(entries, lambda e, i: D.call_snippet(e, i), comp)
    meta = collections.OrderedDict()
    meta["isa"] = isa
    meta["source"] = source + f" of {D.cc_version(D.CLANG)} by ops/intrinsics/build_misc_db.py"
    meta["checked_with"] = [D.cc_version(c) for c in [D.CLANG] + [c[1] for c in comp] if D.have(c)]
    meta["features"] = feats
    D.write_db(os.path.join(HERE, isa + ".json"), meta, entries, dropped)
    print(" ", dict(collections.Counter(",".join(e.get("no", [])) for e in entries)))


def settle(tgt, prelude, cands, dropped, flags_of):
    """Immediates, then (for ret == auto) the deduced return type."""
    out = []
    groups = collections.defaultdict(list)
    for c in cands: groups[tuple(c["features"])].append(c)
    for fs, cs in groups.items():
        failed = D.resolve_immediates(tgt + flags_of(fs), prelude, cs, int_like)
        for k, r in failed.items(): dropped.append((cs[k]["name"], "does not compile: " + r))
        live = [c for k, c in enumerate(cs) if k not in failed]
        autos = [c for c in live if c["ret"] == "auto"]
        if autos:
            rets = D.deduce_returns(tgt + flags_of(fs), prelude, {f"forge_p_{i}": D.call_snippet(c, i) for i, c in enumerate(autos)})
            for i, c in enumerate(autos): c["ret"] = rets.get(f"forge_p_{i}", "auto")
        for c in live:
            if c["ret"] == "auto": dropped.append((c["name"], "return type not deduced"))
            else: out.append(c)
    return out


# ------------------------------------------------------------------------------------------------ LoongArch
def loongarch():
    tgt = ["--target=loongarch64-linux-gnu"]; pre = "#include <lsxintrin.h>\n#include <lasxintrin.h>\n"
    flags_of = lambda fs: ["-mlasx" if "lasx" in fs else "-mlsx"]
    dropped, cands, seen = [], [], set()
    feat = lambda n: ["lasx"] if n.startswith("__lasx") else ["lsx"]
    for f in D.ast_functions(tgt + ["-mlasx"], pre, name_re=re.compile(r"^__la?sx_")):
        if f["name"] in seen: continue
        seen.add(f["name"]); cands.append({"name": f["name"], "ret": f["ret"], "params": f["params"], "features": feat(f["name"])})
    inc = D.resource_dir()
    for h in ("lsxintrin.h", "lasxintrin.h"):
        text = open(os.path.join(inc, h)).read()
        for m in re.finditer(r"#define (__la?sx_\w+)\(([^)]*)\)", text):
            if m.group(1) in seen: continue
            seen.add(m.group(1)); ps = []
            for p in m.group(2).split(","):
                r = re.search(r"/\*\s*(.*?)\s*\*/", p)
                t = r.group(1) if r else "int"
                ps.append("int" if re.match(r"^([us]?i\d+|idx)$", t) else " ".join(t.split()))
            cands.append({"name": m.group(1), "ret": "auto", "params": ps, "features": feat(m.group(1))})
    print("loongarch candidates:", len(cands))
    entries = settle(tgt, pre, cands, dropped, flags_of)
    feats = {"lsx": {"tag": "lsx", "ns": "lsx", "flag": "-mlsx", "if": "defined(__loongarch_sx)"},
             "lasx": {"tag": "lasx", "ns": "lasx", "flag": "-mlasx", "if": "defined(__loongarch_asx)"}}
    finish("loongarch", entries, dropped, feats, tgt, pre, flags_of, "derived from lsxintrin.h and lasxintrin.h")


# ------------------------------------------------------------------------------------------------ WebAssembly
def wasm():
    tgt = ["--target=wasm32-unknown-unknown"]; pre = "#include <wasm_simd128.h>\n"
    FL = {"simd128": "-msimd128", "relaxed_simd": "-mrelaxed-simd", "fp16": "-mfp16"}
    flags_of = lambda fs: [FL[f] for f in fs]
    dropped, cands, seen = [], [], set()
    text = open(os.path.join(D.resource_dir(), "wasm_simd128.h")).read()
    for f in D.ast_functions(tgt + list(FL.values()), pre, name_re=re.compile(r"^wasm_")):
        if f["name"] in seen: continue
        seen.add(f["name"])
        if "DeprecatedAttr" in f["attrs"]: dropped.append((f["name"], "deprecated alias")); continue
        fs = [x.strip().replace("-", "_") for x in f["attrs"].get("TargetAttr", "simd128").split(",")]
        if "simd128" not in fs: fs = ["simd128"] + fs
        c = {"name": f["name"], "ret": f["ret"], "params": list(f["params"]), "features": fs}
        # parameters the header marks __REQUIRE_CONSTANT are immediates
        m = re.search(r"\b%s\(([^)]*)\)([^{;]*)[{;]" % re.escape(f["name"]), text)
        if m:
            names = [p.split()[-1].lstrip("*") for p in m.group(1).split(",") if p.strip() and p.strip() != "void"]
            const = set(re.findall(r"__REQUIRE_CONSTANT\((\w+)\)", m.group(2)))
            if const and len(names) == len(c["params"]):
                c["imm"] = []
                for j, n in enumerate(names):
                    if n in const: c["params"][j] = "const int"; c["imm"].append(0)
        cands.append(c)
    for n, (ps, body) in sorted(D.macros(tgt + list(FL.values()), pre, name_re=re.compile(r"^wasm_")).items()):
        if ps is None or n in seen: continue
        if "DEPRECATED" in body: dropped.append((n, "deprecated alias")); continue
        types = ["v128_t" if re.search(r"\(__\w+\)\(%s\)" % re.escape(p), body) else "const int" for p in ps]      # lane indices
        cands.append({"name": n, "ret": "auto", "params": types, "features": ["simd128"], "imm": [0] * types.count("const int")})
    print("wasm candidates:", len(cands))
    entries = settle(tgt, pre, cands, dropped, flags_of)
    feats = {"simd128": {"tag": "wasm_simd128", "ns": "simd128", "flag": "-msimd128", "if": "defined(__wasm_simd128__)"},
             "relaxed_simd": {"tag": "wasm_relaxed_simd", "ns": "relaxed_simd", "flag": "-mrelaxed-simd"},
             "fp16": {"tag": "wasm_fp16", "ns": "fp16", "flag": "-mfp16"}}
    finish("wasm", entries, dropped, feats, tgt, pre, flags_of, "derived from wasm_simd128.h")


# ------------------------------------------------------------------------------------------------ MIPS MSA
MSA_VEC = {(16, "char"): "v16i8", (16, "signed char"): "v16i8", (16, "unsigned char"): "v16u8", (8, "short"): "v8i16",
           (8, "unsigned short"): "v8u16", (4, "int"): "v4i32", (4, "unsigned int"): "v4u32", (2, "long long"): "v2i64",
           (2, "unsigned long long"): "v2u64", (4, "float"): "v4f32", (2, "double"): "v2f64"}


def msa_type(t):
    m = re.search(r"__vector_size__\((\d+) \* sizeof\(([^)]+)\)\)\)\) ", t)
    return MSA_VEC.get((int(m.group(1)), m.group(2)), t) if m else t


def mips():
    tgt = ["--target=mips64el-linux-gnuabi64", "-mmsa"]; pre = "#include <msa.h>\nstruct forge_X {};\n"
    dropped = []
    names = sorted(k for k, v in D.macros(tgt, pre, name_re=re.compile(r"^__msa_")).items() if v[0] is None)
    bad = D.probe(tgt, pre, [f"void forge_p_{i}() {{ {n}(); }}" for i, n in enumerate(names)])
    state = {}
    for i, n in enumerate(names):
        msg = bad.get(i, [""])[0]
        m = re.search(r"expected (\d+), have 0", msg)
        if i in bad and not m: dropped.append((n, "argument count not found: " + msg)); continue
        state[n] = [None] * (int(m.group(1)) if m else 0)       # per parameter: None | ("rt", type) | ("imm", value)
    live = list(state)
    for _ in range(12):
        if not live: break
        sn = []
        for n in live:
            args, call = [], []
            for j, s in enumerate(state[n]):
                if s is None: call.append("forge_X()")
                elif s[0] == "rt": args.append(f"{s[1]} a{j}"); call.append(f"a{j}")
                else: call.append(str(s[1]))
            head = f"auto forge_p_{len(sn)}({', '.join(args)}) {{ return {n}("
            cols, c = [], len(head) + 1
            for a in call: cols.append((c, c + len(a))); c += len(a) + 2
            sn.append((head + ", ".join(call) + "); }", cols))
        bad = D.probe(tgt, pre, [t for t, _ in sn], cols=True)
        nxt = []
        for k, errs in bad.items():
            n = live[k]; acted = False
            for col, msg in errs:
                j = next((a for a, (lo, hi) in enumerate(sn[k][1]) if lo <= col < hi), None)
                if j is None: continue
                r = re.search(r"from 'forge_X' to '([^']+)'", msg)
                if r and state[n][j] is None: state[n][j] = ("rt", msa_type(r.group(1) + " ").strip()); acted = True
                if acted: break
            if acted: nxt.append(n)
            elif any(s is None for s in state[n]): dropped.append((n, "signature not recovered: " + errs[0][1])); del state[n]
            # else: every type is known; what is left is an immediate, settled below
        live = nxt
    for n in live: dropped.append((n, "signature did not converge")); del state[n]
    cands = [{"name": n, "ret": "auto", "params": [s[1] for s in st], "features": ["msa"]} for n, st in state.items()]
    entries = settle(tgt, pre, cands, dropped, lambda fs: [])
    for c in entries: c["ret"] = msa_type(c["ret"] + " ").strip()
    print("mips candidates:", len(entries))
    feats = {"msa": {"tag": "msa", "ns": "msa", "flag": "-mmsa", "if": "defined(__mips_msa)"}}
    finish("mips", entries, dropped, feats, tgt[:1], "#include <msa.h>\n", lambda fs: ["-mmsa"], "derived from msa.h (signatures recovered from the __builtin_msa_* diagnostics)")


# ------------------------------------------------------------------------------------------------ levels (ppc, s390x)
def levelled(isa, tgt, pre, levels, source, gcc):
    """levels: [(feature, flags, guard)] from lowest to highest."""
    dropped, cands, seen = [], [], set()
    for feat, flags, _ in levels:
        for f in D.ast_functions(tgt + flags, pre, name_re=re.compile(r"^vec_\w+$")):
            key = (f["name"], tuple(f["params"]))
            if key in seen: continue
            seen.add(key)
            if "DeprecatedAttr" in f["attrs"]: dropped.append((f["name"], "deprecated overload")); continue
            fix = lambda t: t.replace("__bool unsigned ", "__bool ")       # the AST's spelling of bool vectors is not valid source
            cands.append({"name": f["name"], "ret": fix(f["ret"]), "params": [fix(p) for p in f["params"]], "features": [feat]})
    macs = D.macros(tgt + levels[-1][1], pre, name_re=re.compile(r"^vec_\w+$"))
    fn_names = {c["name"] for c in cands}
    for n, (ps, body) in sorted(macs.items()):
        if n not in fn_names: dropped.append((n, "type-generic or alias macro without a typed declaration"))
    print(isa, "candidates:", len(cands))
    flag = {f: fl for f, fl, _ in levels}
    flags_of = lambda fs: flag[fs[0]]
    entries = settle(tgt, pre, cands, dropped, flags_of)
    feats = collections.OrderedDict((f, {"tag": f, "ns": f, "flag": " ".join(fl), "if": g}) for f, fl, g in levels)
    finish(isa, entries, dropped, feats, tgt, pre, flags_of, source, gcc=(gcc[0], lambda fs: gcc[1](fs[0])))


def ppc():
    lv = [("altivec", ["-mcpu=power7", "-mno-vsx", "-maltivec"], "defined(__ALTIVEC__)"),
          ("vsx", ["-mcpu=power7"], "defined(__VSX__)"),
          ("power8_vector", ["-mcpu=power8"], "defined(__POWER8_VECTOR__)"),
          ("power9_vector", ["-mcpu=power9"], "defined(__POWER9_VECTOR__)"),
          ("power10_vector", ["-mcpu=power10"], "defined(__POWER10_VECTOR__)")]
    g = {"altivec": "power8", "vsx": "power8", "power8_vector": "power8", "power9_vector": "power9", "power10_vector": "power10"}
    levelled("ppc", ["--target=powerpc64le-linux-gnu"], "#include <altivec.h>\n", lv, "derived from altivec.h",
             ("powerpc64le-linux-gnu-g++", lambda f: ["-mcpu=" + g[f]]))


def s390x():
    lv = [("zvector", ["-march=z13", "-mzvector"], "defined(__VEC__) && __ARCH__ >= 11"),
          ("zvector_e1", ["-march=z14", "-mzvector"], "defined(__VEC__) && __ARCH__ >= 12"),
          ("zvector_e2", ["-march=z15", "-mzvector"], "defined(__VEC__) && __ARCH__ >= 13"),
          ("zvector_e3", ["-march=z17", "-mzvector"], "defined(__VEC__) && __ARCH__ >= 15")]
    g = {"zvector": "z13", "zvector_e1": "z14", "zvector_e2": "z15", "zvector_e3": "arch15"}
    levelled("s390x", ["--target=s390x-linux-gnu"], "#include <vecintrin.h>\n", lv, "derived from vecintrin.h",
             ("s390x-linux-gnu-g++", lambda f: ["-march=" + g[f], "-mzvector"]))


if __name__ == "__main__":
    todo = sys.argv[1:] or ["loongarch", "wasm", "mips", "ppc", "s390x"]
    for t in todo: globals()[t]()

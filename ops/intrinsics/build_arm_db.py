#!/usr/bin/env python3
"""Builds ops/intrinsics/arm.json from clang's <arm_neon.h>, <arm_acle.h>, <arm_sve.h>, <arm_neon_sve_bridge.h>
and <arm_sme.h>. Run by hand:

    python3 ops/intrinsics/build_arm_db.py     # needs clang++-23 (FORGE_RAW_CLANG); aarch64-linux-gnu-g++ and $FORGE_RAW_CLANG_OLD (an older clang++) are tried if present

NEON/ACLE  functions from the AST (signature + target attribute); the lane/shift macros from their
           `T __sN = __pN;` bodies, with the immediates found by compiling.
SVE        every non-overloaded and overloaded declaration from the AST. arm_sve.h carries no feature
           information (the declarations alias compiler builtins), so the features are read from clang's
           "needs target feature" diagnostic, and the constant operands from "must be a constant integer".
SME        as SVE, plus the calling context: each intrinsic is compiled from a non-streaming, a streaming and a
           streaming-compatible caller, with and without ZA / ZT0 state; the wrapper gets the keyword
           attributes of the least demanding context that compiles ("attr": s | sc, "za": za | zt0 | za,zt0)."""
import os, re, sys, collections, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intrin_db as D

HERE = os.path.dirname(os.path.abspath(__file__))
TGT = ["--target=aarch64-linux-gnu"]
NEON = "#include <arm_neon.h>\n#include <arm_acle.h>\n"
SVE = "#include <arm_sve.h>\n#if __has_include(<arm_neon_sve_bridge.h>)\n#include <arm_neon_sve_bridge.h>\n#endif\n"
SME = "#include <arm_sme.h>\n"

RENAME = {"v8.1a": "rdm", "v8.3a": "complex", "v8.5a": "frintts", "fullfp16": "fp16"}
ARCH = {"rdm": "armv8.1-a", "complex": "armv8.3-a", "frintts": "armv8.5-a"}
TAGS = {"neon": "neon", "dotprod": "dotprod", "i8mm": "i8mm"}
LEGACY = {"sve_aes": "sve2-aes", "sve_bitperm": "sve2-bitperm", "sve_sha3": "sve2-sha3", "sve_sm4": "sve2-sm4"}
DROP_FEATURES = {"dsp", "gcs", "ls64", "rand"}      # ACLE system / DSP helpers, not SIMD
RANK = {"neon": 0, "sve": 1, "sve2": 2, "sme": 3, "sme2": 4}
SEEN = {}      # db feature name -> llvm name


def fname(llvm):
    n = RENAME.get(llvm, llvm).replace("-", "_").replace(".", "_")
    SEEN.setdefault(n, llvm)
    return n


def tag(n):
    if n in TAGS: return n
    return n if n.startswith(("sve", "sme", "ssve")) else "arm_" + n


def march(features, legacy=False):
    arch = "armv8-a"
    ext = []
    for f in features:
        if f in ARCH: arch = max(arch, ARCH[f])
        elif f == "neon": pass
        elif legacy: ext.append(LEGACY.get(f, SEEN.get(f, f)).replace("fullfp16", "fp16"))
        else: ext.append(SEEN.get(f, f).replace("fullfp16", "fp16"))
    return ["-march=" + arch + "".join("+" + e for e in ext)]


def order(fs):
    fs = list(dict.fromkeys(fs))
    return sorted(fs, key=lambda f: (RANK.get(f, 10), f))


INT_RE = re.compile(r"^(u?int(8|16|32|64)_t|int|unsigned int|long|unsigned long|enum \w+|sv(pattern|prfop))$")
def int_like(t): return bool(INT_RE.match(t))
def enum_like(t): return t.startswith("enum ") or t in ("svpattern", "svprfop")


def attr_text(e, plain_sc=False):
    a = {"s": "__arm_streaming", "sc": "" if plain_sc else "__arm_streaming_compatible"}.get(e.get("attr", ""), "")
    if e.get("za"): a += " __arm_inout(" + ", ".join(f'"{z}"' for z in e["za"].split(",")) + ")"
    return a.strip()


def parse_need(msgs, want_sme, have=()):
    need = []
    for m in msgs:
        r = re.search(r"needs target feature (.+)$", m)
        if r:
            alts = D.feature_alternatives(r.group(1))
            def is_sme(a): return any(x.startswith(("sme", "ssve")) for x in a)
            def cost(a): return (is_sme(a) != want_sme, len([x for x in a if fname(x) not in have]))
            need += min(alts, key=cost)
        r = re.search(r"requires target feature '([^']+)'", m)
        if r: need.append(r.group(1))
        r = re.search(r"state requires '([^']+)'", m) or re.search(r"mode requires '([^']+)'", m)
        if r: need.append(r.group(1))
    return [fname(n) for n in need]


def infer_features(entries, prelude, dropped):
    """Grow each entry's feature list from the compiler's diagnostics until it compiles."""
    todo = list(range(len(entries)))
    for _ in range(8):
        groups = collections.defaultdict(list)
        for i in todo: groups[tuple(entries[i]["features"])].append(i)
        nxt = []
        for fs, ids in groups.items():
            try:
                bad = D.probe(TGT + march(fs), prelude, [D.call_snippet(entries[i], i, attrs=attr_text(entries[i], entries[i]["sec"] != "sme")) for i in ids], codegen=True)
            except RuntimeError as ex:
                for i in ids: entries[i]["fail"] = str(ex)
                continue
            for k, msgs in bad.items():
                e = entries[ids[k]]
                need = [x for x in parse_need(msgs, e["sec"] == "sme", e["features"]) if x not in e["features"]]
                if need: e["features"] = order(e["features"] + need); nxt.append(ids[k])
                else: e["fail"] = msgs[0]
        todo = nxt
        if not todo: break
    for i in todo: entries[i]["fail"] = "feature set did not converge"
    out = []
    for e in entries:
        if "fail" in e: dropped.append((e["name"], "does not compile: " + e["fail"]))
        else: out.append(e)
    return out


def neon(dropped):
    name_re = re.compile(r"^(v[a-z0-9]\w*|__crc32\w+|__rint\w+|__cls\w*|__jcvt)$")
    fns = D.ast_functions(TGT, NEON, name_re=name_re)
    entries, seen = [], set()
    for f in fns:
        if f["name"] in seen or "TargetAttr" not in f["attrs"]: continue
        seen.add(f["name"])
        raw = [x.strip() for x in f["attrs"]["TargetAttr"].split(",")]
        if any(x in DROP_FEATURES for x in raw): dropped.append((f["name"], "feature outside Tier R: " + ",".join(raw))); continue
        entries.append({"name": f["name"], "ret": f["ret"], "params": f["params"], "features": order(fname(x) for x in raw), "sec": "neon"})
    print("neon functions:", len(entries))
    # the immediate-operand intrinsics are statement-expression macros
    ms = {k: v for k, v in D.macros(TGT, NEON, name_re=re.compile(r"^v[a-z0-9]\w*$")).items() if v[0] is not None and k not in seen}
    cands, opts = [], []
    for n in sorted(ms):
        ps, body = ms[n]
        m = re.match(r"__extension__ \(\{ (?:(\w+) __ret(?:_\d+)?; )?", body)
        if not m: dropped.append((n, "macro: body form not understood")); continue
        ret = m.group(1) or "void"
        types, vecs = [], []
        for p in ps:
            r = re.search(r"(\w+) __s\d+(?:_\d+)? = %s;" % re.escape(p), body)
            types.append(r.group(1) if r else None)
            if r: vecs.append(r.group(1))
        # untyped parameters are immediates or pointers to the element type
        elems = []
        for v in vecs + ([ret] if ret != "void" else []):
            r = re.match(r"(u?int|float|poly|bfloat)(\d+)x", v)
            if r:
                el = f"{r.group(1)}{r.group(2)}_t"
                if el not in elems: elems.append(el)
        o = []
        for t in types:
            if t is not None: o.append([t])
            else: o.append(["int"] + [f"const {el} *" for el in elems] + [f"{el} *" for el in elems])
        cands.append({"name": n, "ret": ret}); opts.append(o)
    # features of a macro: compile bare and read the diagnostic
    sn, own = [], []
    for ci, (c, o) in enumerate(zip(cands, opts)):
        for combo in itertools.islice(itertools.product(*o), 64):
            own.append((ci, list(combo)))
    work = [dict(cands[ci], params=combo, features=["neon"], sec="neon") for ci, combo in own]
    allf = ["-march=armv9.5-a+aes+sha3+sm4+fp16+bf16+i8mm+dotprod+rdm+fp16fml+lut+fp8+fp8dot2+fp8dot4+fp8fma+faminmax"]
    failed = D.resolve_immediates(TGT + allf, NEON, work, lambda t: t == "int")
    done = set(); nm = 0
    for wi, (ci, combo) in enumerate(own):
        if wi in failed or ci in done: continue
        done.add(ci); entries.append(work[wi]); nm += 1
    for ci, c in enumerate(cands):
        if ci not in done: dropped.append((c["name"], "macro: no parameter typing compiles"))
    print("neon macros:", nm)
    return infer_features(entries, NEON, dropped)


def scalable(dropped):
    name_re = re.compile(r"^(sv\w+|__arm_in_streaming_mode)$")
    fns = D.ast_functions(TGT + ["-march=armv9-a+sve+sme"], SVE + SME, name_re=name_re)
    entries, seen = [], set()
    for f in fns:
        key = (f["name"], tuple(f["params"]))
        if key in seen: continue
        seen.add(key)
        alias = f["attrs"].get("ArmBuiltinAliasAttr", "")
        sme = alias.startswith("__builtin_sme") or "sme" in f["attrs"].get("TargetAttr", "")
        entries.append({"name": f["name"], "ret": f["ret"], "params": [p.replace("enum ", "") for p in f["params"]],
                        "features": ["sme"] if sme else ["sve"], "sec": "sme" if sme else "sve"})
    print("sve/sme declarations:", len(entries))
    full = TGT + ["-march=armv9-a+sve+sve2+sme+sme2"]
    warn = ("-Wno-everything", "-Werror=undefined-arm-za", "-Werror=undefined-arm-zt0")
    pre = SVE + SME
    # constants: resolve in each calling context until one accepts the call
    ctxs = ["", "__arm_streaming __arm_inout(\"za\", \"zt0\")", "__arm_inout(\"za\", \"zt0\")",
            "__arm_streaming_compatible __arm_inout(\"za\", \"zt0\")"]
    pending = list(range(len(entries))); reason = {}
    for ctx in ctxs:
        if not pending: break
        work = [dict(entries[i]) for i in pending]
        failed = D.resolve_immediates(full, pre, work, int_like, attrs_of=lambda e: ctx, warn=warn, enum_like=enum_like)
        nxt = []
        for k, i in enumerate(pending):
            if k in failed: reason[i] = failed[k]; nxt.append(i)
            else: entries[i] = work[k]
        pending = nxt
    for i in pending: dropped.append((entries[i]["name"], "does not compile in any calling context: " + reason[i]))
    entries = [e for i, e in enumerate(entries) if i not in set(pending)]
    # calling context: which of {non-streaming, streaming, streaming-compatible} x {ZA, ZT0 state} are accepted
    ok = [set() for _ in entries]
    zas = ["", "za", "zt0", "za,zt0"]
    for kind in ("", "s", "sc"):
        for za in zas:
            ids = [i for i, e in enumerate(entries) if (e["sec"] == "sme" or za == "") and not any(k == kind for k, _ in ok[i])]
            if not ids: continue
            bad = D.probe(full, pre, [D.call_snippet(entries[i], i, attrs=attr_text({"attr": kind, "za": za})) for i in ids], warn=warn)
            for k, i in enumerate(ids):
                if k not in bad: ok[i].add((kind, za))
    out = []
    for e, o in zip(entries, ok):
        kinds = {k: z for k, z in sorted(o, key=lambda kz: -zas.index(kz[1]))}      # least ZA state per kind
        if "sc" in kinds: e["attr"] = "sc"; za = kinds["sc"]
        elif "s" in kinds and "" not in kinds: e["attr"] = "s"; za = kinds["s"]; e["sec"] = "sme"
        elif "" in kinds: za = kinds[""]
        else: dropped.append((e["name"], "no calling context accepted")); continue
        if "sme" not in e["features"] and e["sec"] == "sme": e["features"] = ["sme"]
        if za: e["za"] = za
        if "zt0" in za and "sme2" not in e["features"]: e["features"] = order(e["features"] + ["sme2"])
        out.append(e)
    c = collections.Counter((e["sec"], e.get("attr", "-"), e.get("za", "-")) for e in out); print("contexts:", dict(c))
    return infer_features(out, pre, dropped)


def demote_sc(entries):
    """An SVE intrinsic is only streaming-compatible unconditionally if it compiles from a streaming-compatible
    caller with just its own features plus SME; the ones that need a streaming-mode twin feature (ssve-aes, ...)
    keep a plain wrapper."""
    groups = collections.defaultdict(list)
    for i, e in enumerate(entries):
        if e.get("attr") == "sc" and not any(f.startswith(("sme", "ssve")) for f in e["features"]): groups[tuple(e["features"])].append(i)
    n = 0
    for fs, ids in groups.items():
        bad = D.probe(TGT + march(list(fs) + ["sme"]), SVE + SME, [D.call_snippet(entries[i], i, attrs="__arm_streaming_compatible") for i in ids], codegen=True)
        for k in bad: entries[ids[k]].pop("attr"); n += 1
    print("streaming-compatible only with a streaming-mode feature (plain wrapper):", n)


def main():
    if "--demote-only" in sys.argv:      # re-apply demote_sc to an existing database
        import json
        db = json.load(open(os.path.join(HERE, "arm.json")))
        for f, v in db["features"].items(): SEEN[f] = v["flag"].lstrip("+")
        demote_sc(db["intrinsics"])
        D.write_db(os.path.join(HERE, "arm.json"), {k: v for k, v in db.items() if k != "intrinsics"}, db["intrinsics"], None)
        return
    dropped = []
    entries = neon(dropped) + scalable(dropped)
    # final check with exactly the recorded features
    groups = collections.defaultdict(list)
    for i, e in enumerate(entries): groups[(e["sec"], tuple(e["features"]))].append(i)
    keep = [True] * len(entries)
    for (sec, fs), ids in groups.items():
        pre = NEON if sec == "neon" else SVE + (SME if sec == "sme" else "")
        bad = D.probe(TGT + march(fs), pre, [D.call_snippet(entries[i], i, attrs=attr_text(entries[i], sec != "sme")) for i in ids], codegen=True)
        for k, msgs in bad.items(): keep[ids[k]] = False; dropped.append((entries[ids[k]]["name"], "does not compile with its features: " + msgs[0]))
    entries = [e for e, k in zip(entries, keep) if k]
    # other compilers / 32-bit Arm
    def pre_of(e): return NEON if e["sec"] == "neon" else SVE + (SME if e["sec"] == "sme" else "")
    for key, cc, tflags, legacy in [("clang_old", D.CLANG_OLD, TGT, True), ("gcc13", "aarch64-linux-gnu-g++", [], True),
                                    ("a32", D.CLANG, ["--target=armv7a-linux-gnueabihf", "-mfpu=neon-vfpv4", "-mfloat-abi=hard"], None)]:
        if not D.have(cc): print(f"  ({cc} not installed: '{key}' not recorded)"); continue
        groups = collections.defaultdict(list)
        for i, e in enumerate(entries): groups[(e["sec"], tuple(e["features"]))].append(i)
        for (sec, fs), ids in groups.items():
            if key == "a32":
                if sec != "neon" or fs != ("neon",):
                    for i in ids: entries[i].setdefault("no", []).append(key)
                    continue
                fl = tflags
            else: fl = tflags + march(fs, legacy=True)
            try: bad = D.probe(fl, pre_of(entries[ids[0]]), [D.call_snippet(entries[i], i, attrs=attr_text(entries[i], sec != "sme")) for i in ids], cc=cc)
            except RuntimeError as ex:
                print(f"  ({key}: {ex}: {len(ids)} entries marked unavailable)")
                bad = {k: 1 for k in range(len(ids))}
            for k in bad: entries[ids[k]].setdefault("no", []).append(key)
    demote_sc(entries)
    feats = collections.OrderedDict()
    for e in entries:
        e.pop("sec")
        for f in e["features"]:
            feats.setdefault(f, {"tag": tag(f), "ns": f, "flag": ARCH.get(f) or "+" + SEEN.get(f, f).replace("fullfp16", "fp16")})
            if f.startswith("sve"): feats[f]["if"] = "FORGE_RAW_ARM_SVE"
            if f.startswith(("sme", "ssve")): feats[f]["if"] = "FORGE_RAW_ARM_SME"
    meta = collections.OrderedDict()
    meta["isa"] = "arm"
    meta["source"] = f"derived from arm_neon.h, arm_acle.h, arm_sve.h, arm_neon_sve_bridge.h and arm_sme.h of {D.cc_version(D.CLANG)} by ops/intrinsics/build_arm_db.py"
    meta["checked_with"] = [D.cc_version(c) for c in (D.CLANG, D.CLANG_OLD, "aarch64-linux-gnu-g++") if D.have(c)]
    meta["features"] = dict(sorted(feats.items(), key=lambda kv: (RANK.get(kv[0], 10), kv[0])))
    D.write_db(os.path.join(HERE, "arm.json"), meta, entries, dropped)
    print(dict(collections.Counter((e["features"][0] if e["features"][0] in ("sve", "sme") else "neon", ",".join(e.get("no", []))) for e in entries)))


if __name__ == "__main__":
    main()

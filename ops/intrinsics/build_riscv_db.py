#!/usr/bin/env python3
"""Builds ops/intrinsics/riscv.json (and riscv_policy.json). Run by hand:

    python3 ops/intrinsics/build_riscv_db.py --rvv-doc <dir with the auto-generated lists of rvv-intrinsic-doc>

clang's <riscv_vector.h> declares nothing: `#pragma clang riscv intrinsic vector` makes the compiler create
each `__riscv_v*` builtin on first use, so there is no header to read. The candidate list is therefore the
official one (riscv-non-isa/rvv-intrinsic-doc, auto-generated/**/intrinsic_funcs.adoc: the v1.0 explicit,
non-overloaded API, vector crypto and bfloat16 included), and the compiler is the judge: every prototype
is compiled against clang's builtins with its documented signature. Which operands must be constants
(vxrm, frm, vget/vset indices) and which extensions an intrinsic needs (Zvfh, Zvbb, Zvk*, ...) are read
from the diagnostics. The tail/mask policy variants (_tu, _tum, _tumu, _mu) go to riscv_policy.json.

The scalar helpers come from the AST of <riscv_bitmanip.h> and <riscv_crypto.h>, compiled once per extension."""
import argparse, os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intrin_db as D

HERE = os.path.dirname(os.path.abspath(__file__))
TGT = ["--target=riscv64-linux-gnu"]
VEC = "#include <riscv_vector.h>\n"
SCALAR = "#include <riscv_bitmanip.h>\n#include <riscv_crypto.h>\n"
SCALAR_EXT = ["zba", "zbb", "zbc", "zbs", "zbkb", "zbkc", "zbkx", "zknd", "zkne", "zknh", "zksed", "zksh"]
RANK = {"v": 0}


def march(fs):
    return ["-march=rv64gc" + ("v" if "v" in fs else "") + "".join("_" + f for f in fs if f != "v")]


def order(fs):
    fs = list(dict.fromkeys(fs))
    if "zvfh" in fs and "zvfhmin" in fs: fs.remove("zvfhmin")
    return sorted(fs, key=lambda f: (RANK.get(f, 5), f))


def prototypes(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    out = []
    for m in re.finditer(r"^([A-Za-z_][\w \*]*?)\s*\b(__riscv_\w+)\s*\(([^;{}]*?)\)\s*;", text, re.M | re.S):
        ret, name, ps = " ".join(m.group(1).split()), m.group(2), " ".join(m.group(3).split())
        params = []
        for p in [x.strip() for x in ps.split(",") if x.strip()]:
            if p == "void": continue
            r = re.match(r"(.*?)(\w+)$", p)        # drop the parameter name
            params.append(r.group(1).strip() if r and r.group(1).strip() else p)
        out.append({"name": name, "ret": ret, "params": params})
    return out


def need(msgs):
    out = []
    for m in msgs:
        r = re.search(r"requires the '([^']+)' extension", m) or re.search(r"following extensions: (.+)$", m)
        if r:
            alts = [a.strip() for a in re.split(r" or |,", r.group(1)) if a.strip()]
            out.append(alts[-1] if "type" in m else alts[0])
    return [o for o in out if not o.startswith("zve")]


def validate(cands, dropped):
    INT = re.compile(r"^(size_t|unsigned int|unsigned|int|u?int(8|16|32|64)_t)$")
    for c in cands: c["features"] = ["v"]
    # features first (type and builtin requirements), with every integer operand passed at run time
    todo = list(range(len(cands)))
    for _ in range(6):
        groups = collections.defaultdict(list)
        for i in todo: groups[tuple(cands[i]["features"])].append(i)
        nxt = []
        for fs, ids in groups.items():
            try: bad = D.probe(TGT + march(fs), VEC, [D.call_snippet(cands[i], i) for i in ids])
            except RuntimeError:       # an extension clang only accepts as experimental
                for i in ids: cands[i]["fail"] = "needs an experimental extension: " + ",".join(fs)
                continue
            for k, msgs in bad.items():
                e = cands[ids[k]]
                n = [x for x in need(msgs) if x not in e["features"]]
                if n: e["features"] = order(e["features"] + n); nxt.append(ids[k])
        todo = nxt
        if not todo: break
    out = []
    groups = collections.defaultdict(list)
    for i, c in enumerate(cands):
        if "fail" in c: dropped.append((c["name"], c["fail"]))
        else: groups[tuple(c["features"])].append(i)
    for fs, ids in groups.items():
        work = [cands[i] for i in ids]
        try: failed = D.resolve_immediates(TGT + march(fs), VEC, work, lambda t: bool(INT.match(t)))
        except RuntimeError as ex: failed = {k: str(ex) for k in range(len(work))}
        live = [w for k, w in enumerate(work) if k not in failed]
        for k, r in failed.items(): dropped.append((work[k]["name"], "does not compile: " + r))
        bad = D.probe(TGT + march(fs), VEC, [D.call_snippet(e, i) for i, e in enumerate(live)], codegen=True)
        for k, msgs in bad.items(): dropped.append((live[k]["name"], "does not compile with its features: " + msgs[0]))
        out += [e for k, e in enumerate(live) if k not in bad]
    return out


def scalar(dropped):
    base = {f["name"] for f in D.ast_functions(TGT + ["-march=rv64gc"], SCALAR, name_re=re.compile(r"^__riscv_"))}
    entries, seen = [], set(base)
    for ext in SCALAR_EXT:
        for f in D.ast_functions(TGT + ["-march=rv64gc_" + ext], SCALAR, name_re=re.compile(r"^__riscv_")):
            if f["name"] in seen: continue
            seen.add(f["name"])
            entries.append({"name": f["name"], "ret": f["ret"], "params": f["params"], "features": [ext]})
    keep = []
    groups = collections.defaultdict(list)
    for e in entries: groups[tuple(e["features"])].append(e)
    for fs, es in groups.items():
        failed = D.resolve_immediates(TGT + march(fs), SCALAR, es, lambda t: t in ("int", "unsigned", "unsigned int", "uint32_t"))
        es2 = [e for k, e in enumerate(es) if k not in failed]
        for k, r in failed.items(): dropped.append((es[k]["name"], "does not compile: " + r))
        bad = D.probe(TGT + march(fs), SCALAR, [D.call_snippet(e, i) for i, e in enumerate(es2)], codegen=True)
        for k, msgs in bad.items(): dropped.append((es2[k]["name"], "does not compile with its features: " + msgs[0]))
        keep += [e for k, e in enumerate(es2) if k not in bad]
    for n in base: dropped.append((n, "no extension required: generic helper"))
    return keep


def finish(name, entries, dropped, source, inherit=None):
    def flags_of(prefix):
        def f(fs):
            pre = VEC if "v" in fs else SCALAR
            return pre, prefix + march(fs)
        return f
    if inherit is None:
        D.classify(entries, lambda e, i: D.call_snippet(e, i),
                   [("clang_old", D.CLANG_OLD, flags_of(TGT)), ("gcc13", "riscv64-linux-gnu-g++", flags_of([]))])
    else:
        # A policy variant exists in another compiler exactly when its base intrinsic does (g++ needs minutes per
        # thousand RVV calls, so the 40k variants are not compiled one by one there).
        for e in entries:
            b = inherit.get(re.sub(r"_(tu|tum|tumu|mu)$", "", e["name"])) or inherit.get(re.sub(r"_(tu|tum|tumu|mu)$", "_m", e["name"]))
            if b is None: e["no"] = ["clang_old", "gcc13"]
            elif b.get("no"): e["no"] = list(b["no"])
    feats = collections.OrderedDict()
    for e in entries:
        for f in e["features"]:
            feats.setdefault(f, {"tag": "rvv" if f == "v" else f, "ns": "rvv" if f == "v" else f, "flag": f,
                                 "if": "defined(__riscv_vector)" if f == "v" else f"defined(__riscv_{f})"})
    meta = collections.OrderedDict()
    meta["isa"] = "riscv"
    meta["source"] = source
    meta["checked_with"] = [D.cc_version(c) for c in (D.CLANG, D.CLANG_OLD, "riscv64-linux-gnu-g++") if D.have(c)]
    meta["features"] = dict(sorted(feats.items(), key=lambda kv: (RANK.get(kv[0], 5), kv[0])))
    D.write_db(os.path.join(HERE, name), meta, entries, dropped)
    print(dict(collections.Counter(",".join(e.get("no", [])) for e in entries)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rvv-doc", required=True)
    ap.add_argument("--policy-only", action="store_true", help="keep riscv.json, rebuild riscv_policy.json")
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.rvv_doc) if f.endswith(".adoc") and "overloaded" not in f)
    base, policy, seen = [], [], set()
    for f in files:
        for p in prototypes(os.path.join(a.rvv_doc, f)):
            if p["name"] in seen: continue
            seen.add(p["name"])
            (policy if re.search(r"_(tu|tum|tumu|mu)$", p["name"]) else base).append(p)
    print(f"prototypes: {len(base)} + {len(policy)} policy variants")
    src = f"prototype list of riscv-non-isa/rvv-intrinsic-doc (auto-generated/), validated against {D.cc_version(D.CLANG)}; " \
          f"scalar helpers from its riscv_bitmanip.h / riscv_crypto.h; by ops/intrinsics/build_riscv_db.py"
    if not a.policy_only:
        dropped = []
        entries = validate(base, dropped) + scalar(dropped)
        finish("riscv.json", entries, dropped, src)
    import json
    done = {e["name"]: e for e in json.load(open(os.path.join(HERE, "riscv.json")))["intrinsics"]}
    dropped = []
    finish("riscv_policy.json", validate(policy, dropped), dropped, src, inherit=done)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tier R generator: intrinsic database -> strongly-typed, feature-gated C++ wrappers.

Each wrapper is a template over a lane Target with a `requires Has<T, feature>` clause, so using an
intrinsic the target lacks is a compile error naming the feature, not a SIGILL in the field.

    intrin_gen.py --db ops/intrinsics/<isa>.json --out <dir>/forge/simd/raw/<isa>.gen.hpp [--check <isa>.check.cpp]

The database (ops/intrinsics/*.json, built by ops/intrinsics/build_*_db.py) has one entry per intrinsic:

    name, ret, params     the signature; a parameter spelled "const T" (by value) is an immediate operand and
                          becomes a template parameter of the wrapper
    features              the ISA features it needs (database "features" maps each to a caps.hpp tag, the
                          namespace the wrapper lives in and, where the compiler needs the feature enabled to
                          even name the types, the preprocessor condition "if")
    imm                   a valid value per immediate (used by the --check translation unit)
    no                    compilers that reject it: clang_old (an older clang the database was also tried with), gcc13 (-> gated on the compiler), a32
    attr, za              Arm streaming context: s = __arm_streaming, sc = streaming-compatible; ZA / ZT0 state

--check writes a translation unit that calls every wrapper once (tools/check_raw_intrinsics.py compiles it
for every target)."""
import argparse, json, os, re

KEYWORDS = set("and or xor not bitand bitor compl and_eq or_eq xor_eq not_eq new delete union switch case default float "
               "double int long short signed unsigned char bool void auto register static extern inline const volatile "
               "if else for while do return break continue goto true false this class struct enum namespace template "
               "typename using operator public private protected virtual friend try catch throw typedef sizeof".split())

# MSVC ships <immintrin.h> with the documented Intel set for these features only (and no GNU-style target
# attributes); everything else is left out there. Unverified in this repository's CI: see docs/simd.md.
MSVC_X86 = {"sse2", "sse4_2", "popcnt", "lzcnt", "bmi1", "bmi2", "avx", "avx2", "fma", "f16c", "avx512f", "avx512vl", "avx512dq",
            "avx512bw", "avx512cd", "avx512vnni", "avx512bf16", "avx512ifma", "avx512vbmi", "avx512vbmi2", "avx512bitalg",
            "avx512vpopcntdq", "avxvnni", "aesni", "pclmulqdq", "shani", "gfni", "vaes", "vpclmulqdq"}


def strip_prefix(*prefixes, repl=""):
    def f(n):
        for p in prefixes:
            if n.startswith(p): return repl + n[len(p):]
        return n
    return f


ISA = {
    "x86": dict(guard="defined(__x86_64__) || defined(_M_X64)", name=lambda n: n.lstrip("_"),
                includes=["#include <immintrin.h>"], msvc=MSVC_X86),
    "arm": dict(guard="defined(__aarch64__) || defined(_M_ARM64) || (defined(__arm__) && defined(__ARM_NEON))",
                name=strip_prefix("sv", "__", "v"), msvc={"neon"},
                includes=["#if FORGE_RAW_MSVC", "#include <arm64_neon.h>", "#else", "#include <arm_neon.h>",
                          "#if __has_include(<arm_acle.h>)", "#include <arm_acle.h>", "#endif", "#endif",
                          "// SVE and SME types can only be named where the feature is enabled for the translation unit.",
                          "#if defined(__ARM_FEATURE_SVE) && __has_include(<arm_sve.h>)", "#include <arm_sve.h>",
                          "#if __has_include(<arm_neon_sve_bridge.h>)", "#include <arm_neon_sve_bridge.h>", "#endif",
                          "#define FORGE_RAW_ARM_SVE 1", "#else", "#define FORGE_RAW_ARM_SVE 0", "#endif",
                          "#if defined(__ARM_FEATURE_SME) && __has_include(<arm_sme.h>) && (FORGE_RAW_CLANG || (defined(__GNUC__) && __GNUC__ >= 14))",
                          "#include <arm_sme.h>", "#define FORGE_RAW_ARM_SME 1",
                          "#define FORGE_RAW_ARM_SC __arm_streaming_compatible   // SVE wrappers stay callable from streaming code",
                          "#else", "#define FORGE_RAW_ARM_SME 0", "#define FORGE_RAW_ARM_SC", "#endif"]),
    # GCC 13 predates the v1.0 intrinsic API (no vxrm / frm operands, ...); GCC 14 is the first with it.
    "riscv": dict(guard="defined(__riscv)", name=strip_prefix("__riscv_"), msvc=None, gcc_min=14,
                  includes=["#if defined(__riscv_vector)", "#include <riscv_vector.h>", "#endif",
                            "#if __has_include(<riscv_bitmanip.h>)", "#include <riscv_bitmanip.h>", "#endif",
                            "#if __has_include(<riscv_crypto.h>)", "#include <riscv_crypto.h>", "#endif"]),
    "loongarch": dict(guard="defined(__loongarch__)", name=strip_prefix("__lsx_", "__lasx_"), msvc=None,
                      includes=["#if defined(__loongarch_sx)", "#include <lsxintrin.h>", "#endif",
                                "#if defined(__loongarch_asx)", "#include <lasxintrin.h>", "#endif"]),
    "ppc": dict(guard="defined(__powerpc__) || defined(__powerpc64__)", name=strip_prefix("vec_", repl="v_"), msvc=None,
                includes=["#if defined(__ALTIVEC__)", "#include <altivec.h>", "#endif"]),
    "wasm": dict(guard="defined(__wasm__)", name=strip_prefix("wasm_"), msvc=None,
                 includes=["#if defined(__wasm_simd128__)", "#include <wasm_simd128.h>", "#endif"]),
    "s390x": dict(guard="defined(__s390x__)", name=strip_prefix("vec_", repl="v_"), msvc=None,
                  includes=["#if defined(__VEC__)", "#include <vecintrin.h>", "#endif"]),
    "mips": dict(guard="defined(__mips__)", name=strip_prefix("__msa_"), msvc=None,
                 includes=["#if defined(__mips_msa)", "#include <msa.h>", "#endif"]),
}


def is_imm(t):
    return t.startswith("const ") and "*" not in t and "&" not in t


def wrapper_name(cfg, n):
    w = cfg["name"](n)
    if w in KEYWORDS or not re.match(r"^[A-Za-z_]\w*$", w): w += "_"
    return w


def guard_of(e, feats, cfg, ns, gcc_checked):
    """The preprocessor condition under which this entry exists (empty = always)."""
    g = []
    for f in e["features"]:
        c = feats[f].get("if")
        if c and c not in g: g.append(c)
    no = set(e.get("no", []))
    c18, gcc = "clang_old" in no, "gcc13" in no
    if "a32" in no: g.append("FORGE_RAW_A64")
    gmin = cfg.get("gcc_min")       # the first GCC that implements the same API (not verified here), if any
    if c18 and gcc: g.append(f"(FORGE_RAW_CLANG_GE(23) || FORGE_RAW_GCC_GE({gmin}))" if gmin else "FORGE_RAW_CLANG_GE(23)")
    elif gcc: g.append(f"(FORGE_RAW_CLANG || FORGE_RAW_GCC_GE({gmin}))" if gmin else "FORGE_RAW_CLANG")
    elif c18: g.append("(FORGE_RAW_CLANG_GE(23) || FORGE_RAW_GCC)" if gcc_checked else "FORGE_RAW_CLANG_GE(23)")
    elif cfg["msvc"] is not None and ns not in cfg["msvc"]: g.append("!FORGE_RAW_MSVC")
    return " && ".join(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--check"); ap.add_argument("--count-name", default="intrinsic_count")
    a = ap.parse_args()
    db = json.load(open(a.db, encoding="utf-8"))
    isa = db["isa"]; cfg = ISA[isa]
    feats = db.get("features") or {}
    legacy = {"sse": "sse2", "ssse3": "sse4_2", "sse4_1": "sse4_2", "fma": "fma3"}
    gcc_checked = any("g++" in c or "gcc" in c.lower() for c in db.get("checked_with", []))
    base = os.path.basename(a.db)
    stem = os.path.basename(a.out)

    def info(f):
        if f not in feats: feats[f] = {"tag": legacy.get(f, f), "ns": "sse4_2" if legacy.get(f) == "sse4_2" else f}
        return feats[f]

    groups = {}      # namespace -> guard -> [entry]
    for e in db["intrinsics"]:
        if "features" not in e: e["features"] = [e["feature"]]
        for f in e["features"]: info(f)
        ns = info(e["features"][-1])["ns"]
        groups.setdefault(ns, {}).setdefault(guard_of(e, feats, cfg, ns, gcc_checked), []).append(e)

    L = [f"// GENERATED by ops/gen/intrin_gen.py from ops/intrinsics/{base} — do not edit.", "#pragma once",
         "#include <forge/simd/caps.hpp>", "#include <forge/simd/raw/config.hpp>", f"#if {cfg['guard']}"] + cfg["includes"] + \
        ["#if FORGE_RAW_GCC   // FORGE_RAW_ARG names vector types as template arguments; their alignment / may_alias attributes do not matter for a value cast",
         "#pragma GCC diagnostic push", "#pragma GCC diagnostic ignored \"-Wignored-attributes\"", "#endif",
         "", f"namespace forge::simd::raw::{isa} {{", "", f"inline constexpr int {a.count_name} = {len(db['intrinsics'])};", ""]
    C = [f"// GENERATED by ops/gen/intrin_gen.py from ops/intrinsics/{base}: calls every wrapper once — do not edit.",
         f"#include <forge/simd/raw/{stem}>", f"#if {cfg['guard']}", "namespace {",
         "struct forge_all : ::forge::simd::caps<" + ", ".join(sorted({"::forge::simd::" + v["tag"] for v in feats.values()})) + "> {};",
         f"namespace r = ::forge::simd::raw::{isa};"]
    n = 0
    for ns in sorted(groups):
        L.append(f"namespace {ns} {{")
        for guard in sorted(groups[ns]):
            if guard: L.append(f"#if {guard}"); C.append(f"#if {guard}")
            for e in groups[ns][guard]:
                name = e["name"]; w = wrapper_name(cfg, name)
                tparams, args, call, targs, cargs, k = [], [], [], [], [], 0
                imm = e.get("imm", [])
                for i, t in enumerate(e["params"]):
                    if is_imm(t):
                        ty = t[6:]
                        tparams.append(f"{ty} I{i}"); call.append(f"I{i}")
                        v = imm[k] if k < len(imm) else 0; k += 1
                        targs.append(str(v) if ty == "int" else f"static_cast<{ty}>({v})")
                    else:
                        args.append(f"{t} a{i}"); call.append(f"a{i}"); cargs.append(f"a{i}")
                # One argument goes through a cast to a T-dependent spelling of its own type. That makes the call a
                # dependent expression: the intrinsic is looked up and checked when the wrapper is instantiated, not
                # when this header is parsed, so a compiler that lacks one intrinsic only fails where it is used
                # (and intrinsics that test their operand with __builtin_constant_p see the template argument's value).
                for i, t in enumerate(e["params"]):
                    if not is_imm(t): call[i] = f"FORGE_RAW_ARG({t}, a{i})"; break
                else:
                    if e["params"]: call[0] = f"FORGE_RAW_ARG({e['params'][0][6:]}, I0)"
                req = " && ".join(f"fs::Has<T, fs::{t}>" for t in dict.fromkeys(feats[f]["tag"] for f in e["features"]))
                sme = any(f.startswith(("sme", "ssve")) for f in e["features"])
                attr = {"s": "__arm_streaming ", "sc": "__arm_streaming_compatible " if sme else "FORGE_RAW_ARM_SC "}.get(e.get("attr", ""), "")
                if e.get("za"): attr += "__arm_inout(" + ", ".join(f'"{z}"' for z in e["za"].split(",")) + ") "
                ret = "" if e["ret"] == "void" else "return "
                L.append(f"template <{', '.join(tparams + ['fs::Target T'])}> requires {req}")
                L.append(f"FORGE_RAW_INLINE {e['ret']} {w}({', '.join(args)}) {attr}{{ {ret}{name}({', '.join(call)}); }}")
                C.append(f"{e['ret']} forge_c_{n}({', '.join(args)}) {attr}{{ {ret}r::{ns}::{w}<{', '.join(targs + ['forge_all'])}>({', '.join(cargs)}); }}")
                n += 1
            if guard: L.append("#endif"); C.append("#endif")
        L.append(f"}} // namespace {ns}")
        L.append("")
    L += [f"}} // namespace forge::simd::raw::{isa}", "#if FORGE_RAW_GCC", "#pragma GCC diagnostic pop", "#endif", "#endif", ""]
    C += ["} // namespace", "#endif", ""]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w", encoding="utf-8", newline="\n").write("\n".join(L))
    if a.check: open(a.check, "w", encoding="utf-8", newline="\n").write("\n".join(C))


if __name__ == "__main__":
    main()

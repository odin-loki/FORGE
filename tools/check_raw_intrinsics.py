#!/usr/bin/env python3
"""Syntax-checks every Tier R wrapper of every architecture against the real target headers.

    python3 tools/check_raw_intrinsics.py [--clang clang++-23] [--out DIR] [--isa x86,arm,...] [-j N] [--codegen]

For each database in ops/intrinsics it runs ops/gen/intrin_gen.py, which also writes a translation unit that
calls every wrapper once, and compiles that for the architecture's targets with
`clang++ --target=<triple> <feature flags> -fsyntax-only -ffreestanding -nostdinc++`. Configurations with
features switched off check the guards (a header must stay valid when SVE, VSX, ... are not enabled), and one
translation unit that includes every generated header is compiled for every target (each header must be
includable, and empty, off its architecture).

No C++ standard library for the foreign targets is needed: caps.hpp only uses <type_traits>, which
tools/raw_check_shim provides for this check. Exit status 77 (ctest: skipped) if the compiler is missing or
does not carry a target. ctest runs this as `raw_intrinsics_cross` (label raw_cross)."""
import argparse, concurrent.futures, json, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DBS = ["x86", "arm", "riscv", "riscv_policy", "loongarch", "ppc", "wasm", "s390x", "mips"]


def configs(isa, db):
    """[(label, [flags])]: the first enables everything the database uses, the rest switch features off."""
    f = db["features"]
    if isa == "x86":
        t = ["--target=x86_64-linux-gnu"]
        return [("all", t + [v["flag"] for v in f.values()]), ("baseline", t)]
    if isa == "arm":
        t = ["--target=aarch64-linux-gnu"]
        arch = max([v["flag"] for v in f.values() if v["flag"].startswith("armv")] + ["armv8-a"])
        ext = [v["flag"] for v in f.values() if v["flag"].startswith("+") and v["flag"] != "+neon"]
        nosme = [e for e in ext if not e.startswith(("+sme", "+ssve"))]
        neon = [e for e in nosme if not e.startswith("+sve") and e not in ("+f32mm", "+f64mm")]
        return [("all", t + ["-march=" + arch + "".join(ext)]), ("no-sme", t + ["-march=" + arch + "".join(nosme)]),
                ("neon", t + ["-march=" + arch + "".join(neon)]), ("armv8.0", t + ["-march=armv8-a"]),
                ("armv7", ["--target=armv7a-linux-gnueabihf", "-mfpu=neon-vfpv4", "-mfloat-abi=hard"])]
    if isa == "riscv":
        t = ["--target=riscv64-linux-gnu"]
        z = "".join("_" + v["flag"] for v in f.values() if v["flag"] != "v")
        return [("all", t + ["-march=rv64gcv" + z]), ("v", t + ["-march=rv64gcv"]), ("rv64gc", t + ["-march=rv64gc"])]
    if isa == "loongarch":
        t = ["--target=loongarch64-linux-gnu"]
        return [("lasx", t + ["-mlasx"]), ("lsx", t + ["-mlsx"]), ("base", t)]
    if isa == "ppc":
        t = ["--target=powerpc64le-linux-gnu"]
        return [("power10", t + ["-mcpu=power10"]), ("power9", t + ["-mcpu=power9"]), ("power8", t + ["-mcpu=power8"]),
                ("vsx", t + ["-mcpu=power7"]), ("altivec", t + ["-mcpu=power7", "-mno-vsx", "-maltivec"]),
                ("none", t + ["-mcpu=power7", "-mno-vsx", "-mno-altivec"])]
    if isa == "wasm":
        t = ["--target=wasm32-unknown-unknown"]
        return [("all", t + [v["flag"] for v in f.values()]), ("simd128", t + ["-msimd128"]), ("none", t)]
    if isa == "s390x":
        t = ["--target=s390x-linux-gnu"]
        return [("z17", t + ["-march=z17", "-mzvector"]), ("z15", t + ["-march=z15", "-mzvector"]), ("z14", t + ["-march=z14", "-mzvector"]),
                ("z13", t + ["-march=z13", "-mzvector"]), ("no-zvector", t + ["-march=z13"])]
    if isa == "mips":
        t = ["--target=mips64el-linux-gnuabi64"]
        return [("msa", t + ["-mmsa"]), ("none", t)]
    raise SystemExit("unknown isa " + isa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clang", default="clang++-23"); ap.add_argument("--out", default=os.path.join(ROOT, "build", "raw_check"))
    ap.add_argument("--isa", default=",".join(DBS)); ap.add_argument("-j", type=int, default=2)
    ap.add_argument("--codegen", action="store_true", help="also lower the all-features configuration to assembly (slow)")
    a = ap.parse_args()
    if shutil.which(a.clang) is None:
        print(f"SKIP: {a.clang} not found"); return 77
    gen = os.path.join(a.out, "generated"); rawdir = os.path.join(gen, "forge", "simd", "raw")
    os.makedirs(rawdir, exist_ok=True)
    base = [a.clang, "-std=c++20", "-ffreestanding", "-nostdinc++", "-fno-caret-diagnostics", "-ferror-limit=20", "-Werror=return-type",
            "-I", os.path.join(ROOT, "tools", "raw_check_shim"), "-I", os.path.join(ROOT, "simd", "include"), "-I", gen]
    jobs, counts = [], {}
    everything = os.path.join(a.out, "all_headers.cpp")
    with open(everything, "w") as f:
        f.write("".join(f"#include <forge/simd/raw/{d}.gen.hpp>\n" for d in DBS) + "#include <forge/simd/raw.hpp>\nint forge_raw_all_headers;\n")
    for d in DBS:      # every header is generated: all_headers.cpp includes them all
        db_path = os.path.join(ROOT, "ops", "intrinsics", d + ".json")
        check = os.path.join(a.out, d + ".check.cpp")
        cmd = [sys.executable, os.path.join(ROOT, "ops", "gen", "intrin_gen.py"), "--db", db_path, "--out", os.path.join(rawdir, d + ".gen.hpp"), "--check", check]
        if d == "riscv_policy": cmd += ["--count-name", "policy_intrinsic_count"]
        subprocess.run(cmd, check=True)
        if d not in a.isa.split(","): continue
        db = json.load(open(db_path, encoding="utf-8")); isa = db["isa"]; counts[d] = len(db["intrinsics"])
        for i, (label, flags) in enumerate(configs(isa, db)):
            jobs.append((f"{d}/{label}", base + flags + ["-fsyntax-only", check]))
            if i == 0 and a.codegen: jobs.append((f"{d}/{label} (codegen)", base + flags + ["-S", "-o", os.devnull, check]))
            if d != "riscv_policy": jobs.append((f"all-headers/{isa}/{label}", base + flags + ["-fsyntax-only", everything]))
    # does this clang carry the targets at all?
    probe = subprocess.run([a.clang, "--print-targets"], capture_output=True, text=True).stdout
    need = {"x86": "x86-64", "arm": "aarch64", "riscv": "riscv64", "riscv_policy": "riscv64", "loongarch": "loongarch64", "ppc": "ppc64le",
            "wasm": "wasm32", "s390x": "systemz", "mips": "mips64el"}
    missing = [d for d in counts if need[d] not in probe]
    if missing:
        print("SKIP: " + a.clang + " has no backend for " + ", ".join(missing)); return 77

    def run(job):
        r = subprocess.run(job[1], capture_output=True, text=True)
        return job[0], r.returncode, r.stderr
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.j) as ex:
        for label, rc, err in ex.map(run, jobs):
            print(("ok   " if rc == 0 else "FAIL ") + label, flush=True)
            if rc != 0:
                failed += 1
                print("\n".join(err.splitlines()[:40]))
    print(f"{len(jobs) - failed}/{len(jobs)} configurations passed; wrappers: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

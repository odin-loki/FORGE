#!/usr/bin/env python3
"""Differential conformance test for `forge script`.

Every tests/script_conformance/*.cpp is an ordinary C++23 program. Each one is compiled with the host C++ compiler
and run; `forge script` then runs the same file on the tree-walker (--tree) and on the bytecode VM. Standard output
and the exit status of all three must be identical. The programs cover the subset docs/scripting.md says is
supported, so a hole in the interpreter (a missing member function, an operator form the parser refuses, a wrong
conversion) shows up as a diff against the real compiler rather than as a surprise in somebody's script.

    run.py --forge build/dev/forge [--cxx clang++] [--work DIR] [--jobs N] [--only NAME] [--keep] [files...]

Exit status: 0 all programs agree, 1 a divergence (or a program the host compiler rejects), 77 no usable host
compiler (ctest: skipped). A compiler is usable when it builds a C++23 program that includes <print>; the flags
tried are `-std=c++23` and then `-std=c++23 -stdlib=libc++` (clang with an older libstdc++), MSVC `/std:c++latest`.
"""
import argparse
import concurrent.futures
import difflib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = '#include <print>\n#include <format>\n#include <ranges>\n#include <span>\nint main() { std::println("{}", std::format("{:>3}", 1)); }\n'


def run(cmd, timeout, **kw):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, **kw)
        return p.returncode, p.stdout.decode("utf-8", "replace").replace("\r\n", "\n"), p.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return -999, "", "timed out after %d s" % timeout
    except OSError as e:
        return -998, "", str(e)


def find_compiler(cxx, work):
    """-> (argv prefix, output flag style) or None."""
    names = [cxx] if cxx else [os.environ.get("CXX", ""), "clang++-23", "clang++", "g++-14", "g++", "c++", "cl"]
    probe = os.path.join(work, "probe.cpp")
    with open(probe, "w") as f:
        f.write(PROBE)
    for name in names:
        if not name:
            continue
        exe = shutil.which(name.split()[0])
        if not exe:
            continue
        base = [exe] + name.split()[1:]
        if os.path.basename(exe).lower() in ("cl", "cl.exe"):
            variants = [(base + ["/nologo", "/std:c++latest", "/EHsc", "/utf-8"], "msvc")]
        else:
            variants = [(base + ["-std=c++23"], "gnu"), (base + ["-std=c++23", "-stdlib=libc++"], "gnu")]
        for argv, style in variants:
            out = os.path.join(work, "probe.exe")
            cmd = argv + ([probe, "/Fe:" + out, "/Fo:" + os.path.join(work, "probe.obj")] if style == "msvc" else [probe, "-o", out])
            rc, _, _ = run(cmd, 300)
            if rc == 0 and run([out], 60)[0] == 0:
                return argv, style
    return None


def one(path, forge, compiler, work, timeout):
    """-> (name, ok, report text)."""
    argv, style = compiler
    name = os.path.splitext(os.path.basename(path))[0]
    exe = os.path.join(work, name + ".exe")
    cmd = argv + ([path, "/Fe:" + exe, "/Fo:" + os.path.join(work, name + ".obj")] if style == "msvc" else ["-O1", "-w", path, "-o", exe])
    rc, _, err = run(cmd, 900)
    if rc != 0:
        return name, False, "the host compiler rejects %s (the program must be valid C++23):\n%s" % (os.path.basename(path), err[-3000:])
    want_rc, want, err = run([exe], timeout, cwd=work)
    if want_rc < 0 and want_rc > -990 or want_rc in (-999, -998):
        return name, False, "the natively compiled program failed to run (%s): %s" % (want_rc, err[-1000:])
    report = []
    for mode, flags in (("tree", ["--tree"]), ("vm", [])):
        got_rc, got, err = run([forge, "script"] + flags + [path], timeout, cwd=work)
        if got == want and got_rc == want_rc:
            continue
        diff = "".join(difflib.unified_diff(want.splitlines(True), got.splitlines(True), "native", "forge script (%s)" % mode, n=1))
        report.append("%s [%s]: exit %s (native %s)\n%s%s" % (name, mode, got_rc, want_rc, diff[:6000], ("stderr: " + err[-1500:]) if err.strip() else ""))
    return name, not report, "\n".join(report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forge", required=True)
    ap.add_argument("--cxx", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--only", default="")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    forge = os.path.abspath(a.forge)
    if not os.path.exists(forge):
        print("no forge binary at %s" % forge)
        return 1
    work = os.path.abspath(a.work) if a.work else tempfile.mkdtemp(prefix="forge-conformance-")
    os.makedirs(work, exist_ok=True)
    try:
        compiler = find_compiler(a.cxx, work)
        if not compiler:
            print("SKIP: no host C++23 compiler with <print> found (tried %s)" % (a.cxx or "$CXX, clang++, g++, c++, cl"))
            return 77
        files = [os.path.abspath(f) for f in a.files] or sorted(os.path.join(HERE, f) for f in os.listdir(HERE) if f.endswith(".cpp"))
        if a.only:
            files = [f for f in files if a.only in os.path.basename(f)]
        print("host compiler: %s" % " ".join(compiler[0]))
        bad = 0
        with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
            for name, ok, report in ex.map(lambda f: one(f, forge, compiler, work, a.timeout), files):
                print("[%s] %s" % (" ok " if ok else "FAIL", name))
                if not ok:
                    bad += 1
                    print(report)
        print("%d program(s), %d divergence(s)" % (len(files), bad))
        return 1 if bad else 0
    finally:
        if not a.keep and not a.work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

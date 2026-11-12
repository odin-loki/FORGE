#!/usr/bin/env python3
"""Performance baselines: run `forge bench` (model prefill/decode + GEMM per lane) and record or check them.
  tools/perf_baseline.py --forge build/forge --model build/golden/tiny_f32.fpk            record perf/baseline-<arch>.json
  tools/perf_baseline.py --forge ... --model ... --check [--tolerance 0.5]                 fail if any metric < (1-tol) x baseline
      [--allow-missing]      a baseline metric this machine cannot produce (e.g. no AVX-512 lane) is reported, not failed
      [--report out.json]    write the measured metrics, the baseline, each ratio and status as JSON
      [--markdown FILE]      append the result table as Markdown (CI: $GITHUB_STEP_SUMMARY)
Throughput metrics only (higher is better). Baselines are per machine class; regenerate when hardware changes."""
import argparse, json, os, platform, re, subprocess, sys

def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout

def measure(forge, model, gemm, reps):
    m = {}
    for _ in range(reps):   # best of N: least sensitive to noise from other load
        out = run([forge, "bench", model, "-p", "32", "-n", "32"])
        for k, v in re.findall(r"^(prefill|decode)\s+\d+ tokens:\s+([\d.]+) tok/s", out, re.M):
            m[f"{k}_tok_s"] = max(m.get(f"{k}_tok_s", 0), float(v))
        out = run([forge, "bench", "--gemm", str(gemm)])
        for k, v in re.findall(r"^(\S+)\s+gemm_nt \d+\^3:\s+([\d.]+) GFLOP/s", out, re.M):
            key = f"gemm{gemm}_{k.replace('::', '_')}_gflops"; m[key] = max(m.get(key, 0), float(v))
    return m

def cpu_name():
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name"): return ln.split(":", 1)[1].strip()
    except OSError: pass
    return platform.processor() or "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forge", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--gemm", type=int, default=256); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--baseline"); ap.add_argument("--check", action="store_true"); ap.add_argument("--tolerance", type=float, default=0.5)
    ap.add_argument("--allow-missing", action="store_true"); ap.add_argument("--report"); ap.add_argument("--markdown")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = a.baseline or os.path.join(root, "perf", f"baseline-{platform.machine()}.json")
    got = measure(a.forge, a.model, a.gemm, a.reps)
    if not got: sys.exit("no metrics parsed from forge bench output")
    if not a.check:
        doc = {"machine": platform.machine(), "cpu": cpu_name(), "threads": os.cpu_count(), "metrics": dict(sorted(got.items()))}
        open(path, "w").write(json.dumps(doc, indent=2) + "\n"); print("wrote", path); return 0
    bdoc = json.load(open(path)); base = bdoc["metrics"]; bad = 0; rows = []
    for k in sorted(set(base) | set(got)):
        v, g = base.get(k), got.get(k)
        if v is None: st = "new"                       # measured here, not in the baseline
        elif g is None: st = "n/a" if a.allow_missing else "MISSING"
        else: st = "ok" if g >= (1 - a.tolerance) * v else "SLOW"
        bad += st in ("SLOW", "MISSING")
        rows.append({"metric": k, "measured": g, "baseline": v, "ratio": round(g / v, 3) if g is not None and v else None, "status": st})
        num = lambda x: f"{x:10.2f}" if x is not None else "       n/a"
        print(f"{st:7s} {k:40s} {num(g)}  baseline {num(v)}")
    if a.report:
        doc = {"machine": platform.machine(), "cpu": cpu_name(), "threads": os.cpu_count(), "tolerance": a.tolerance,
               "baseline_file": os.path.relpath(path, root), "baseline_cpu": bdoc.get("cpu"), "baseline_threads": bdoc.get("threads"),
               "passed": not bad, "results": rows}
        open(a.report, "w").write(json.dumps(doc, indent=2) + "\n")
    if a.markdown:
        num = lambda x: f"{x:.2f}" if x is not None else "n/a"
        md = [f"### forge bench vs `{os.path.relpath(path, root)}`", "",
              f"This machine: {cpu_name()}, {os.cpu_count()} threads. Baseline: {bdoc.get('cpu')}, {bdoc.get('threads')} threads. "
              f"A metric fails below {1 - a.tolerance:.2f} x baseline.", "",
              "| metric | measured | baseline | ratio | status |", "|---|---:|---:|---:|---|"]
        md += [f"| `{r['metric']}` | {num(r['measured'])} | {num(r['baseline'])} | {num(r['ratio'])} | {r['status']} |" for r in rows]
        open(a.markdown, "a").write("\n".join(md) + "\n")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())

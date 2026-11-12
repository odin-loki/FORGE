#!/usr/bin/env python3
"""Generate docs/reference.md from the sources of truth: the op spec (ops/defs/*.fspec), the GPU kernel list
(ops/gen/gpu_gen.py), the lowering registrations (graph/include/forge/graph/{kernels,vulkan,plan,codecs}.hpp,
gpu/include/forge/gpu/cuda.hpp) and the C ABI header (inference/include/forge.h).
  tools/gen_docs.py                          write docs/reference.md
  tools/gen_docs.py --check                  exit 1 if docs/reference.md is stale (ctest: docs_up_to_date)
  tools/gen_docs.py --check-coverage FILE    exit 1 if the lowering table read from the sources differs from the
                                             one the built code reports (FILE = the forge_coverage_dump executable,
                                             or its output saved as *.json; ctest: coverage_table)"""
import glob, json, os, re, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root, "ops", "gen"))
import fspec_gen, gpu_gen  # noqa: E402

def gpu_names():
    return set(gpu_gen.UNARY) | set(gpu_gen.BINARY) | set(gpu_gen.OTHER)

def src(*rel):
    return open(os.path.join(root, *rel), encoding="utf-8").read()

def coverage():
    """Which lowerings exist per op and backend, read from the code that registers them.
    cpu: op -> lowering names (registry::make_cpu and the SIMD registrations in kernels.hpp);
    vulkan: ops with a generated GLSL kernel (one "vk.<op>" each) plus the lowerings vulkan_registry adds by name;
    gpu: ops in gpu::cuda_hip::kernel_sig, which graph::gpu_registry registers for CUDA, HIP, OpenCL, Level Zero
    and Metal alike; bcast: binaries the CUDA/HIP runtime also runs with broadcasting."""
    cpu = {}
    for m in re.finditer(r'\br\.add\(\{"([\w.]+)",\s*ops::op::(\w+)', src("graph", "include", "forge", "graph", "kernels.hpp")):
        cpu.setdefault(m.group(2), []).append(m.group(1))
    vsrc = src("graph", "include", "forge", "graph", "vulkan.hpp")
    vk = {n: ["vk." + n] for n in sorted(gpu_gen.UNARY) + sorted(gpu_gen.BINARY) + gpu_gen.OTHER}
    for m in re.finditer(r'r->add\(\{"(vk\.\w+)",\s*ops::op::(\w+)', vsrc): vk.setdefault(m.group(2), []).append(m.group(1))
    csrc = src("gpu", "include", "forge", "gpu", "cuda.hpp")
    body = re.search(r"inline bool kernel_sig\(.*?\n\}", csrc, re.S).group(0)
    gpu = set(re.findall(r'"(\w+)"', body))
    bc = re.search(r"inline int bcast_op\(.*?\n\}", csrc, re.S).group(0)
    bcast = set(re.findall(r'op == "(\w+)"', bc))
    return {"cpu": cpu, "vulkan": vk, "gpu": gpu, "bcast": bcast}

def codec_coverage():
    """Weight codecs (graph/codecs.hpp) and where matmul_t runs on the encoded weight."""
    names = re.findall(r'^\s*\{"(\w+)", dtype::\w+, (\d+), (\d+),', src("graph", "include", "forge", "graph", "codecs.hpp"), re.M)
    psrc = src("graph", "include", "forge", "graph", "plan.hpp")
    fmt = re.search(r"inline int fmt_of\(.*?\n\}", psrc, re.S).group(0)
    direct = set(re.findall(r'c->name == "(\w+)"', fmt))          # codecs with their own SIMD dot kernels
    vsrc = src("graph", "include", "forge", "graph", "vulkan.hpp")
    def lst(var): return set(re.findall(r'"(\w+)"', re.search(var + r" = \{(.*?)\};", vsrc).group(1)))
    return names, direct, lst("device_qgemm_codecs"), lst("device_qglu_codecs")

def render_coverage(ops):
    c = coverage(); names = [o["name"] for o in ops]
    for k in ("cpu", "vulkan"):
        unknown = sorted(set(c[k]) - set(names))
        if unknown: sys.exit(f"gen_docs: {k} lowerings for ops not in the spec: {unknown}")
    unknown = sorted(c["gpu"] - set(names))
    if unknown: sys.exit(f"gen_docs: kernel_sig names not in the spec: {unknown}")
    tick = lambda xs: " ".join(f"`{x}`" for x in xs)
    n_cpu = sum(n in c["cpu"] for n in names); n_vk = sum(n in c["vulkan"] for n in names); n_gpu = sum(n in c["gpu"] for n in names)
    out = ["", "## Lowering coverage (op × backend)", "",
           "Generated from the code that registers the lowerings: `registry::make_cpu` and the SIMD registrations "
           "(`graph/include/forge/graph/kernels.hpp`), the Vulkan kernel list (`ops/gen/gpu_gen.py`, `vulkan_registry`), "
           "and `gpu::cuda_hip::kernel_sig` (`gpu/include/forge/gpu/cuda.hpp`), the one table `graph::gpu_registry` registers "
           "from for CUDA, HIP, OpenCL, Level Zero and Metal. The `coverage_table` test compares this table with the registries "
           "of the built code (`forge_coverage_dump`).", "",
           "- Every op has the float64 scalar reference (`ops/`), which runs whenever no lowering applies and is what the oracle compares against.",
           "- A lowering is used only for the shapes and attributes it supports, and only after it passed the oracle on this machine; "
           "this table says which lowerings exist, not that a backend was tested on real hardware (see [STATUS.md](STATUS.md)).",
           "- `lane.*` lowerings run on the SIMD lane chosen at run time (AVX2, AVX-512, NEON, SVE, RVV, ... or the scalar lane); "
           "`cpu.*` lowerings are plain C++.",
           "- All lowerings in this table take f32 tensors; other dtypes enter as encoded weights (next table).",
           f"- Ops with a lowering: CPU {n_cpu}/{len(names)}, Vulkan {n_vk}/{len(names)}, CUDA/HIP/OpenCL/Level Zero/Metal {n_gpu}/{len(names)}. "
           "`+bcast` = the CUDA/HIP runtime also runs the broadcasting form.", "",
           "| op | scalar ref | CPU | Vulkan | CUDA / HIP | OpenCL / Level Zero | Metal |", "|---|---|---|---|---|---|---|"]
    for n in names:
        g = n in c["gpu"]
        ch = (f"`cuda.{n}` `hip.{n}`" + (" +bcast" if n in c["bcast"] else "")) if g else ""
        out.append(f"| `{n}` | yes | {tick(c['cpu'].get(n, []))} | {tick(c['vulkan'].get(n, []))} | {ch} | "
                   f"{f'`opencl.{n}` `level_zero.{n}`' if g else ''} | {f'`metal.{n}`' if g else ''} |")
    codecs, direct, qgemm, qglu = codec_coverage()
    out += ["", "### Weight codecs (`matmul_t` over an encoded weight)", "",
            "Generated from `graph/codecs.hpp` (codec list), `qweights::fmt_of` in `graph/plan.hpp` (codecs with their own SIMD dot "
            "kernels; the others are decoded block by block inside `cpu.qgemm_nt`) and the `device_qgemm_codecs` / "
            "`device_qglu_codecs` lists in `graph/vulkan.hpp` (weights that stay encoded on a CUDA or HIP device). "
            "Vulkan, OpenCL, Level Zero and Metal have no encoded-weight kernels: weights are decoded to f32 before upload.", "",
            "| codec | block (elements / bytes) | CPU `cpu.qgemm_nt` (weight kept encoded) | CUDA / HIP `qmatmul_t` | CUDA / HIP fused gate-up (`qglu`) |", "|---|---|---|---|---|"]
    for n, blk, by in codecs:
        out.append(f"| `{n}` | {blk} / {by} | {'SIMD dot on the encoded rows' if n in direct else 'decode + f32 dot'} | "
                   f"{'encoded on device' if n in qgemm else ''} | {'yes' if n in qglu else ''} |")
    return out

def check_coverage(path):
    """Compare the table read from the sources with forge_coverage_dump's view of the built code."""
    if path.endswith(".json"): built = json.load(open(path, encoding="utf-8"))
    else:   # the forge_coverage_dump executable itself
        import subprocess
        built = json.loads(subprocess.run([path], check=True, capture_output=True, text=True).stdout)
    c = coverage(); bad = []
    ops = []
    for p in sorted(glob.glob(os.path.join(root, "ops", "defs", "*.fspec"))): ops += fspec_gen.parse(p)
    names = [o["name"] for o in ops]
    if sorted(built["cpu"]) != sorted(names): bad.append(f"op list: spec {sorted(names)} vs built {sorted(built['cpu'])}")
    for n in names:
        a, b = sorted(c["cpu"].get(n, [])), sorted(built["cpu"].get(n, []))
        if a != b: bad.append(f"cpu {n}: sources {a} vs built {b}")
    if sorted(built["gpu"]) != sorted(c["gpu"]): bad.append(f"gpu kernel_sig: sources {sorted(c['gpu'])} vs built {sorted(built['gpu'])}")
    if built["vulkan"] is not None:
        gen = sorted(set(c["vulkan"]) - {"reshape"})   # vk.reshape reuses the scale kernel
        if sorted(built["vulkan"]) != gen: bad.append(f"vulkan kernels: sources {gen} vs built {sorted(built['vulkan'])}")
    if built["codecs"] != [n for n, _, _ in codec_coverage()[0]]: bad.append(f"codecs: built {built['codecs']}")
    for b in bad: print("coverage mismatch:", b)
    if not bad: print(f"coverage table matches the built code ({len(names)} ops; vulkan {'checked' if built['vulkan'] is not None else 'not built'})")
    return 1 if bad else 0

def capi():
    src = open(os.path.join(root, "inference", "include", "forge.h"), encoding="utf-8").read()
    abi = re.search(r"#define FORGE_ABI_VERSION (\d+)", src).group(1)
    fns = []
    for m in re.finditer(r"^([A-Za-z_][\w\s\*]*?\b(forge_\w+)\s*\([^;]*\);)[ \t]*(/\*(.*?)\*/)?", src, re.M | re.S):
        sig = " ".join(m.group(1).replace("FORGE_API ", "").split())   # export macro is not part of the signature
        fns.append((m.group(2), sig, (m.group(4) or "").strip()))
    return abi, fns

def render():
    ops = []
    for p in sorted(glob.glob(os.path.join(root, "ops", "defs", "*.fspec"))):
        ops += fspec_gen.parse(p)
    g = gpu_names()
    out = ["# FORGE reference (generated by tools/gen_docs.py — do not edit)", "",
           f"## Ops ({len(ops)}; {sum(o['name'] in g for o in ops)} with GPU kernels)", "",
           "Every op has a float64 scalar reference that each lowering is tested against.", "",
           "| op | class | inputs | shape rule | tolerance | fusion | attrs | GPU | semantics |", "|---|---|---|---|---|---|---|---|---|"]
    for o in ops:
        out.append(f"| `{o['name']}` | {o['class']} | {o['inputs']} | {o['shape']} | {o['tol']} | {o['fuse']} | {o['attrs']} | "
                   f"{'yes' if o['name'] in g else ''} | {o['doc'].replace('|', '/')} |")
    out += render_coverage(ops)
    abi, fns = capi()
    out += ["", f"## C ABI (version {abi}, `inference/include/forge.h`)", "", "| function | signature | notes |", "|---|---|---|"]
    for n, sig, note in fns:
        out.append(f"| `{n}` | `{sig}` | {note} |")
    return "\n".join(out) + "\n"

def main():
    if "--check-coverage" in sys.argv: return check_coverage(sys.argv[sys.argv.index("--check-coverage") + 1])
    path = os.path.join(root, "docs", "reference.md"); text = render()
    if "--check" in sys.argv:
        cur = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
        if cur != text:
            print("docs/reference.md is stale: run tools/gen_docs.py"); return 1
        print("docs/reference.md up to date"); return 0
    open(path, "w", encoding="utf-8", newline="\n").write(text); print("wrote", path); return 0

if __name__ == "__main__":
    sys.exit(main())

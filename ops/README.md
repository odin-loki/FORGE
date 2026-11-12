# ops: one definition per operation

## Concepts
Every tensor operation is declared once, in a `.fspec` file, with a float64 scalar reference implementation. The other layers are generated from that declaration or checked against it:
- The **graph** IR uses the generated `enum class op` and its info table.
- **Lowerings** (CPU SIMD, Vulkan, CUDA, HIP, OpenCL, Level Zero, Metal, RPC) must agree with the reference within the declared tolerance. The oracle in `verify/` checks each lowering before it is used.
- **GPU kernels** are generated from the same specs, and every generated kernel must name a declared op.

The generated op table is [docs/reference.md](../docs/reference.md) (87 ops, 36 with GPU kernels).

## Layout

| Path | Contents |
|------|----------|
| `ops/defs/*.fspec` | Op declarations: `elementwise` (40), `reduction` (11), `movement` (8), `contraction` (4), `sequence` (12: encoder / embedding-model ops), `spatial` (12: vision / audio / diffusion ops). The file name is only grouping; the class is declared per op |
| `ops/include/forge/ops/op.hpp` | Metadata types: `op_class`, `shape_rule`, `tolerance`, `fusion`, `op_info` |
| `ops/include/forge/ops/reference.hpp`, `reference_ext.hpp` | Scalar references, `forge::ops::ref::<name>` (`reference_ext.hpp`: the `sequence` and `spatial` families) |
| `ops/intrinsics/<isa>.json` | Intrinsic databases, one per ISA family: `x86`, `arm` (NEON, SVE/SVE2, SME/SME2), `riscv` and `riscv_policy` (RVV 1.0, vector crypto, scalar bitmanip), `loongarch` (LSX, LASX), `ppc` (AltiVec, VSX), `wasm` (SIMD128, relaxed SIMD), `s390x` (z/Vector), `mips` (MSA). One line per intrinsic: name, return and parameter types (`const T` = immediate operand), required features, the compilers that reject it. See [Intrinsic databases](#intrinsic-databases) |
| `ops/intrinsics/<isa>.dropped.txt` | Candidates the build script left out of the database, each with the reason (the compiler's diagnostic) |
| `ops/intrinsics/build_*_db.py`, `intrin_db.py` | Derive the databases from clang's own intrinsic headers and validate every entry by compiling it. Run by hand; the build only reads the JSON |
| `ops/gen/fspec_gen.py` | Generates `forge/ops/ops.gen.hpp` and `ops.md` in the build tree |
| `ops/gen/gpu_gen.py` | Generates the GPU kernels: one GLSL compute shader per kernel (Vulkan; the spec kernels and the `VK_PORT` ports of the portable kernels), `kernels.cu` (CUDA/HIP, plus CUDA/HIP-only quantised, attention and state-op kernels), `kernels.cl` (OpenCL, Level Zero), `kernels.metal` (Metal). Also embeds the compiled images into headers |
| `ops/gen/intrin_gen.py` | Generates `forge/simd/raw/<isa>.gen.hpp` from each database: C++ intrinsic wrappers gated by CPU feature. Using an intrinsic the target lacks is a compile error |
| `ops/gen/fit_math.py` | Fits the polynomials behind `simd` erf, erfc and lgamma. Run by hand: `python ops/gen/fit_math.py > simd/include/forge/simd/math_coeffs.hpp` |

## The .fspec format
One `op` line per operation. After the name come key/value pairs in any order. `#` starts a comment line.

```
op <name> class <elementwise|reduction|contraction|data_movement|stateful>
   inputs <n> shape <rule> tol <exact|ulp:N|rel:X|abs:X> fuse <yes|no|epilogue>
   cost <flops per output element> [attrs "<comma list>"] doc "<text>"
```

All keys except `attrs` are required. `shape` must name a `shape_rule` from `op.hpp`. Duplicate op names fail the generator.

Examples:

```
op add    class elementwise    inputs 2  shape broadcast  tol exact  fuse yes  cost 1  doc "y = a + b (numpy broadcasting)"
op slice  class data_movement  inputs 1  shape slice      tol exact  fuse no   cost 0  attrs "axis,start,end" doc "x[start:end] along axis"
```

## Generated outputs
CMake runs the generators at build time:

| Target | Generator | Output |
|--------|-----------|--------|
| `forge_ops_gen` | `fspec_gen.py` | `forge/ops/ops.gen.hpp`: `enum class op`, `op_count`, the info table (name, class, arity, shape rule, tolerance, fusion, cost, attrs, doc), `op_from_name`, and the binding of each op to `forge::ops::ref::<name>`. A missing reference is a compile error. Also `ops.md` |
| `forge_gpu_gen` | `gpu_gen.py` | Kernel sources in the build tree; SPIR-V, PTX and AMDGPU images when the tools exist ([gpu.md](../docs/gpu.md)) |
| `forge_raw_gen` | `intrin_gen.py` | `forge/simd/raw/{x86,arm,riscv,riscv_policy,loongarch,ppc,wasm,s390x,mips}.gen.hpp`: Tier R, one header per ISA family, each empty off its architecture. `forge/simd/raw.hpp` includes the current target's ([simd.md](../docs/simd.md)) |

`tools/gen_docs.py` builds `docs/reference.md` from the `.fspec` files, `gpu_gen.py`'s kernel list and the C ABI header. ctest `docs_up_to_date` runs it with `--check`.

## Op families
| Family (file) | Ops | Conventions |
|---------------|-----|-------------|
| `spatial` | `conv1d`, `conv2d`, `conv_transpose2d`, `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`, `global_avg_pool`, `upsample_nearest`, `upsample_bilinear`, `group_norm`, `batch_norm`, `pad` | One sample, no batch axis: `x[c, l]` or `x[c, h, w]` (as sequences are `[s, d]`). Weights use the PyTorch layouts (`conv`: `[o, c/groups, kh, kw]`; transposed: `[c, o/groups, kh, kw]`); the bias input may be `[1]`. A 2-D attr is one number or `[h, w]`. `pad` takes `[before, after]` pairs from the last axis backwards |
| `sequence` | `attention_bias`, `alibi_bias`, `sinusoid_embed`, `l2_normalize`, `masked_mean`, `topk`, `topk_indices`, `index_select`, `quick_gelu`, `mish`, `hard_swish`, `floor` | `attention_bias(q, k, v, bias)` is attention with an additive score bias broadcast to `[heads, s, t]` (`-inf` masks a key: padding masks, ALiBi, relative-position tables), optional soft-capping and a causal or two-sided window; `q` and `k`/`v` may come from different sequences (cross-attention), also with plain `attention` and `causal: 0` |

Already covered by older ops, so not duplicated: `where` is `select(a, b, cond)`; `cast` between float formats is `convert` (to an integer: `floor`); `split` is `slice`; `concat` takes two inputs (chain it); mean pooling over a sequence is `reduce_mean(axis: 0)` or `masked_mean`; `layer_norm`, `gelu`, `gelu_tanh`, `erf` and `argmax` exist.

New ops go at the end of a file (or in a new file that sorts after the existing ones): the generator numbers ops in file-name, then line, order, so appending keeps every existing `enum class op` value.

## Intrinsic databases
The databases are derived, not typed. Each build script reads the intrinsic headers of the clang named in the database's `source` line (clang 23), compiles every candidate against them and writes the JSON plus a `.dropped.txt` log:

| Script | Databases | Source |
|--------|-----------|--------|
| `build_x86_db.py` | `x86.json` | `<immintrin.h>`: functions from clang's AST (signature and target attribute); the immediate-operand macros from the casts in their expansion, return type deduced by the compiler |
| `build_arm_db.py` | `arm.json` | `<arm_neon.h>`, `<arm_acle.h>`, `<arm_sve.h>`, `<arm_neon_sve_bridge.h>`, `<arm_sme.h>`. SVE/SME declarations carry no feature or streaming information, so it is read from the compiler's diagnostics |
| `build_riscv_db.py --rvv-doc DIR` | `riscv.json`, `riscv_policy.json` | The compiler declares the RVV builtins itself (no header to read), so the candidates are the prototype lists of [rvv-intrinsic-doc](https://github.com/riscv-non-isa/rvv-intrinsic-doc) (`auto-generated/**/intrinsic_funcs.adoc`), each compiled against clang's builtins. Scalar helpers from `<riscv_bitmanip.h>`, `<riscv_crypto.h>` |
| `build_misc_db.py [isa…]` | `loongarch.json`, `ppc.json`, `wasm.json`, `s390x.json`, `mips.json` | `<lsxintrin.h>`, `<lasxintrin.h>`, `<altivec.h>`, `<wasm_simd128.h>`, `<vecintrin.h>`, `<msa.h>` |

What the scripts establish for every entry, by compiling it for its target (`clang++ --target=… -ffreestanding`, parse and code generation):
- **Features.** From the target attribute where the header has one, otherwise from the compiler's "needs target feature" diagnostic, starting from the baseline. The entry is then compiled with exactly those features.
- **Immediates.** A parameter the compiler only accepts as a constant becomes `const T`, with a valid sample value in `imm`.
- **Other compilers.** `no` lists the compilers that reject the entry: `gcc13` (the matching cross g++, where one is installed), `clang_old` (an older clang++, when `FORGE_RAW_CLANG_OLD` names one; the database header records which), `a32` (32-bit Arm). The generator turns that into a preprocessor guard; nothing is left out because an older compiler lacks it.
- **Arm streaming context.** `attr` (`s` = `__arm_streaming`, `sc` = streaming-compatible) and `za` (`za`, `zt0`): the least demanding calling context that compiles.

To regenerate after a compiler update: run the script(s), review the diff of the JSON and the `.dropped.txt`, add any new feature tag to `simd/include/forge/simd/caps.hpp` (the generator maps `features` → `tag`), then run `python3 tools/check_raw_intrinsics.py` (ctest `raw_intrinsics_cross`), which compiles every generated wrapper for every target.

## Adding an op
1. Add an `op` line to the right `.fspec` file.
2. Add `forge::ops::ref::<name>` to `reference.hpp`.
3. If the op needs a new shape rule, add it to `shape_rule` and `infer_shape` in `op.hpp` and a case generator to `verify::make_case` (`verify/include/forge/verify/oracle.hpp`): the oracle draws its inputs and attrs from there.
4. Optionally add a GPU kernel in `gpu_gen.py` and a CPU lowering in graph (`kernels.hpp` / `kernels_ext.hpp`). An op with neither runs the scalar reference; on a device backend a step without a device kernel runs on the CPU between device steps.
5. Rebuild, then run `python tools/gen_docs.py` to refresh `docs/reference.md`.
6. `test_graph` checks the new op: `oracle_matrix_all_ops` runs every op through the full compile pipeline against its reference, and `lowerings_pass_oracle` checks every CPU lowering. Device lowerings are checked by the backend tests (`test_vulkan`, `test_gpu_*`).

# simd: CPU features and kernel dispatch

## Concepts
`forge::simd` detects what the CPU supports and what the OS has enabled (XGETBV state for AVX/AVX-512, the Linux AMX tile permission, the SVE vector length, …). It then picks **one table of kernel function pointers** for the process. The choice is made once, on the first call to `active()`; every later call goes through that table.

Kernels are written once and compiled per target:

| Layer | Header | Contents |
|-------|--------|----------|
| Tier P | `vec.hpp` | Portable fixed-width vectors and masks over native registers |
| Tier A | `algo.hpp`, `qdot.hpp` | Algorithms written against Tier P (dot, softmax, GEMM, quantised-row decode and dot, …) |
| Tier R | `raw.hpp`, `forge/simd/raw/<isa>.gen.hpp` (generated) | Raw intrinsic wrappers for every architecture, generated from `ops/intrinsics/<isa>.json` by `ops/gen/intrin_gen.py`. Each wrapper requires its CPU feature, so using one the target lacks is a compile error. See [Tier R](#tier-r-raw-intrinsics) |
| Capabilities | `caps.hpp` | Feature tags; a target is a set of features |
| Maths | `math.hpp`, `math_coeffs.hpp` | Vectorised libm. The erf/erfc/lgamma coefficients come from `ops/gen/fit_math.py` (run by hand; the output is checked in) |

`simd/src/` builds one translation unit per target, each with its own flags and namespace, into the static library `forge_simd_mv`.

Targets per architecture:

| Arch | Targets (preferred first) | Requirements |
|------|---------------------------|--------------|
| x86_64 | `avx512`, `avx2`, `scalar` | `avx2`: the x86-64-v3 set (SSE2, SSE4.2, POPCNT, AVX, AVX2, FMA, F16C, BMI2). `avx512`: v3 plus AVX-512 F, BW, DQ, VL, CD |
| aarch64 | `sve`, `neon`, `scalar` | `sve`: a table built for exactly the hardware vector length (128, 256, 512 or 1024 bits) |
| riscv64 | `rvv`, `scalar` | `rvv`: a table built for exactly the hardware VLEN (128, 256, 512 or 1024 bits) |
| other | `scalar` | |

SVE and RVV tables are built only for the lengths the compiler accepts (CMake prints `SVE kernel tables:` / `RVV kernel tables:`).

Matrix tile units are separate (`tile.hpp`, namespace `forge::simd::tile`). They compute C = A·Bᵀ like `gemm_nt` and return `false` when the unit is absent, the OS refuses tile state, or the compiler could not build them. Callers then fall back to the vector GEMM.

| Function | Unit | Accuracy |
|----------|------|----------|
| `amx_gemm_bf16` | Intel AMX | Inputs rounded to bf16; f32 accumulate (relative error about 2⁻⁸·√k) |
| `amx_gemm_s8` | Intel AMX | int8 × int8 → int32, exact |
| `sme_gemm_f32` | Arm SME | f32 outer products, same class as `gemm_nt` |

`amx_available()`, `sme_available()` and `sme_vl_bytes()` report the units.

## Tier R: raw intrinsics
`forge/simd/raw/<isa>.gen.hpp` wraps the compiler's intrinsics one to one: 103,666 wrappers across eight ISA families. A wrapper has the intrinsic's exact signature, is a template over a Target, and requires the feature tags its intrinsic needs:

```cpp
#include <forge/simd/raw.hpp>                        // the current architecture's header; opt-in, nothing else includes it
namespace r = forge::simd::raw::native;              // raw::x86, raw::arm, raw::riscv, raw::loongarch, raw::ppc, raw::wasm, raw::s390x, raw::mips

__m256i s = r::avx2::mm256_add_epi32<x86_64_v3>(a, b);            // ok: x86_64_v3 has avx2
__m512  t = r::avx512f::mm512_add_ps<x86_64_v3>(x, y);            // does not compile: x86_64_v3 lacks avx512f
__m256i u = r::avx2::mm256_shuffle_epi32<0x1b, x86_64_v3>(a);     // immediates are template arguments
svfloat32_t v = r::sve::add_f32_x<neoverse2>(pg, p, q);           // sizeless types pass through
auto w = r::rvv::vadd_vv_i32m1<rvv1>(p, q, vl);
```

- **Names.** The namespace is the feature (the most specific one when an intrinsic needs several: `_mm256_mask_add_epi8` is `raw::x86::avx512bw::mm256_mask_add_epi8` and requires `avx512bw` and `avx512vl`). The wrapper drops the intrinsic's prefix (`_mm_` → `mm_`, `vaddq_f32` → `addq_f32`, `svadd_f32_x` → `add_f32_x`, `__riscv_vadd_vv_i32m1` → `vadd_vv_i32m1`, `__lsx_vadd_w` → `vadd_w`, `wasm_i32x4_add` → `i32x4_add`, `__msa_addv_w` → `addv_w`, `vec_add` → `v_add`), so it never collides with a compiler that defines the intrinsic as a macro. Overloaded intrinsics (SVE short forms, AltiVec and z/Vector `vec_*`) are overload sets with concrete types.
- **Gating.** `requires Has<T, tag>` per feature. SSE/SSE2 need `sse2`; SSE3, SSSE3, SSE4.1, SSE4.2 and CRC32 need `sse4_2`. Tags that only Tier R uses are the `FORGE_RAW_FEATURES` list in `caps.hpp`. The wrappers are `always_inline` and carry no target attribute: as with the intrinsic itself, the calling code must be compiled for the feature (`-m…` flags or a target attribute).
- **Optional ISAs.** SVE, SME, RVV, LSX/LASX, AltiVec/VSX, z/Vector, MSA and SIMD128 types can only be named in a translation unit compiled for that ISA, so those parts of a header exist only when the compiler's macro (`__ARM_FEATURE_SVE`, `__riscv_vector`, `__loongarch_sx`, `__VSX__`, `__VEC__`, `__mips_msa`, `__wasm_simd128__`, …) is defined. Every header is includable on every platform and is empty off its architecture.
- **Arm streaming mode.** SME wrappers carry the keyword attributes their intrinsic needs (`__arm_streaming`, `__arm_streaming_compatible`, `__arm_inout("za")`, `__arm_inout("zt0")`). SVE wrappers whose intrinsic is streaming-compatible are marked `__arm_streaming_compatible` when the translation unit has SME enabled, so they stay callable from streaming functions.
- **Compilers.** The databases are derived from clang 23. An intrinsic that GCC 13 or the older clang the database was tried with (`checked_with` in the JSON) rejects is guarded (`FORGE_RAW_CLANG`, `FORGE_RAW_CLANG_GE(23)` in `forge/simd/raw/config.hpp`), not removed. Not verified in this repository: MSVC (x64 gets the wrappers of the long-standing feature sets whose signatures GCC also accepts, ARM64 the baseline NEON ones), Apple clang, GCC 14+ for the entries GCC 13 lacks (they stay clang-only), GCC for LoongArch, MIPS and WebAssembly.
- **`compiled_t`** (`caps.hpp`) is the Target of the features the translation unit is compiled for, from the predefined macros: `r::avx2::mm256_add_epi32<compiled_t>(a, b)` compiles exactly when the file is built with AVX2.

| Architecture | Database | Intrinsics | Size | Namespace | Feature namespaces (most used first) | Regenerate with | Derived from |
|---|---|---|---|---|---|---|---|
| x86-64 | `x86.json` | 7,136 | 1.0 MB | `raw::x86` | `avx512f`, `avx512vl`, `avx512fp16`, `avx512bw`, `avx10_2`, `sse2`, `avx512dq`, `avx2`, `avx`, `avx512vbmi2`, `sse4_2`, `avx512cd`, `avx512bf16`, `avx512vnni`, `fma` (tag `fma3`), `avx512vbmi`, `gfni`, `avx512bitalg`, `amx_avx512`, `avx512vpopcntdq`, `avx512ifma`, `bmi1`, `avxneconvert`, `amx_tile`, `amx_int8`, `amx_fp8`, `avxvnniint8`, `avxvnniint16`, `bmi2`, `avxvnni`, `vaes`, `shani`, `aesni`, `f16c`, `amx_complex`, `avx512vp2intersect`, `sm4`, `avxifma`, `sha512`, `sm3`, `amx_bf16`, `amx_fp16`, `popcnt`, `vpclmulqdq`, `pclmulqdq` | `build_x86_db.py` | `immintrin.h` |
| AArch64 (NEON subset also ARMv7) | `arm.json` | 22,766 | 3.0 MB | `raw::arm` | `sve`, `neon`, `sve2`, `sve2p1`, `sme2`, `sme`, `sme_mop4`, `sve2p3`, `fp16` (tag `arm_fp16`), `f64mm` (tag `arm_f64mm`), `sme_i16i64`, `sve_b16b16`, `sme2p1`, `sve2p2`, `faminmax` (tag `arm_faminmax`), `fp8` (tag `arm_fp8`), `lut` (tag `arm_lut`), `fp8fma` (tag `arm_fp8fma`), `sme_lutv2`, `bf16` (tag `arm_bf16`), `sve_bitperm`, `sme_f16f16`, `rdm` (tag `arm_rdm`), `sme_b16b16`, `sme_f64f64`, `complex` (tag `arm_complex`), `sme_f8f32`, `sme_f8f16`, `i8mm`, `sve_bfscale`, `sme_tmop`, `fp16fml` (tag `arm_fp16fml`), `frintts` (tag `arm_frintts`), `sme2p2`, `sve_aes2`, `sha3` (tag `arm_sha3`), `sme2p3`, `rcpc3` (tag `arm_rcpc3`), `sve_aes`, `dotprod`, `fp8dot2` (tag `arm_fp8dot2`), `fp8dot4` (tag `arm_fp8dot4`), `sha2` (tag `arm_sha2`), `sm4` (tag `arm_sm4`), `crc` (tag `arm_crc`), `aes` (tag `arm_aes`), `f16f32dot` (tag `arm_f16f32dot`), `sve_sha3`, `sve_sm4`, `f8f16mm` (tag `arm_f8f16mm`), `f8f32mm` (tag `arm_f8f32mm`), `f32mm` (tag `arm_f32mm`), `sve_b16mm`, `sve_f16f32mm`, `f16f32mm` (tag `arm_f16f32mm`), `f16mm` (tag `arm_f16mm`) | `build_arm_db.py` | `arm_neon.h`, `arm_acle.h`, `arm_sve.h`, `arm_neon_sve_bridge.h`, `arm_sme.h` |
| RISC-V | `riscv.json` | 29,329 | 4.1 MB | `raw::riscv` | `rvv`, `zvfh`, `zvfbfmin`, `zvfhmin`, `zvkb`, `zvbb`, `zvkned`, `zvl256b`, `zvfbfwma`, `zvbc`, `zvksed`, `zbb`, `zvknha`, `zvkg`, `zvknhb`, `zknh`, `zvksh`, `zbc`, `zknd`, `zbkb`, `zbkx`, `zkne`, `zksh`, `zvl512b` | `build_riscv_db.py` | rvv-intrinsic-doc prototype lists, `riscv_bitmanip.h`, `riscv_crypto.h` |
| RISC-V (`_tu`/`_tum`/`_tumu`/`_mu` variants) | `riscv_policy.json` | 37,510 | 5.9 MB | `raw::riscv` | `rvv`, `zvfh`, `zvfbfmin`, `zvfhmin`, `zvkb`, `zvbb`, `zvfbfwma`, `zvkned`, `zvbc`, `zvl256b`, `zvksed`, `zvknha`, `zvkg`, `zvknhb`, `zvksh`, `zvl512b` | `build_riscv_db.py` | rvv-intrinsic-doc prototype lists |
| LoongArch | `loongarch.json` | 1,470 | 145 KB | `raw::loongarch` | `lasx`, `lsx` | `build_misc_db.py loongarch` | `lsxintrin.h`, `lasxintrin.h` |
| PowerPC | `ppc.json` | 2,739 | 342 KB | `raw::ppc` | `altivec`, `vsx`, `power8_vector`, `power10_vector`, `power9_vector` | `build_misc_db.py ppc` | `altivec.h` |
| WebAssembly | `wasm.json` | 328 | 35 KB | `raw::wasm` | `simd128` (tag `wasm_simd128`), `fp16` (tag `wasm_fp16`), `relaxed_simd` (tag `wasm_relaxed_simd`) | `build_misc_db.py wasm` | `wasm_simd128.h` |
| s390x | `s390x.json` | 1,857 | 247 KB | `raw::s390x` | `zvector`, `zvector_e1`, `zvector_e3`, `zvector_e2` | `build_misc_db.py s390x` | `vecintrin.h` |
| MIPS | `mips.json` | 531 | 48 KB | `raw::mips` | `msa` | `build_misc_db.py mips` | `msa.h` |

The databases and their build scripts are in `ops/intrinsics/` ([ops/README.md](../ops/README.md#intrinsic-databases)); each `<isa>.dropped.txt` lists what was left out and why. Coverage notes:
- x86: MMX, 3DNow!, XOP/FMA4/TBM and system instructions (XSAVE, RTM, key locker, CET, …) are outside Tier R. AVX10.2 intrinsics are tagged `avx10_2`; AVX10.1 has no intrinsics of its own (it is the AVX-512 set).
- RISC-V: the explicit (non-overloaded) v1.0 API. The policy variants are a second header, `forge/simd/raw/riscv_policy.gen.hpp` (same namespaces, `policy_intrinsic_count`), which `raw.hpp` does not include.
- PowerPC: every typed overload in `altivec.h`; the type-generic `_Generic` macros (`vec_ctf`, `vec_cts`, …) have no declaration to wrap and are listed in `ppc.dropped.txt`.

Checks:
- `test_simd_raw` (every architecture; a second build for the optional ISA where the default flags leave it off: `test_simd_raw_sve`, `_rvv`, `_lsx`, `_zvector`, `_msa`) runs a sample of wrappers and asserts the gating at compile time: a wrapper call is well-formed for a Target with the feature and ill-formed without it.
- ctest `raw_intrinsics_cross` (label `raw_cross`; `tools/check_raw_intrinsics.py`) instantiates every wrapper of every architecture and compiles it with `clang++-23 --target=<triple> … -fsyntax-only -ffreestanding -nostdinc++` against the real target headers, in several feature configurations per architecture (all features, baseline, in between), plus one translation unit that includes every header for every target. It is skipped when that clang is not installed (`-DFORGE_RAW_CHECK_CLANG=` to name another).
<!-- end Tier R -->

## API (`simd/include/forge/simd/dispatch.hpp`)

| Function | Returns |
|----------|---------|
| `const cpu_features& detect()` | Cached feature flags (`avx2`, `avx512f`, …, `amx_permitted`, `sve_bytes`, `sve_hw_bytes`, `rvv_bytes`, `hybrid`, `vendor`, `brand`). `.summary()` is the one-line description `forge info` prints |
| `const kernels& active()` | The table for this process. `FORGE_SIMD_TARGET` selects a table by name if it is usable here; otherwise the best usable target wins |
| `const kernels* by_name(const char*)` | A specific table, or `nullptr` if it is not built or not usable here. Names: `scalar`; `avx2`, `avx512` (x86); `neon`, `sve` (aarch64); `rvv` (riscv64) |

`kernels` fields:

| Field | Operation |
|-------|-----------|
| `target` | Table name |
| `saxpy`, `dot`, `sum` | y = a·x + y; dot product; sum |
| `softmax_row`, `rms_norm_row`, `logsumexp` | Row operations |
| `exp`, `log`, `tanh`, `sigmoid`, `silu`, `gelu` | Element-wise maths over arrays |
| `silu_mul`, `add`, `sub`, `mul` | Binary element-wise |
| `gemm_nt`, `gemm_nn` | C = A·Bᵀ and C = A·B, row-major |
| `math1(fn, …)`, `math2(fn, …)` | Extended maths by id: `fn1` = exp2, expm1, log2, log10, log1p, sqrt, rsqrt, cbrt, sin, cos, tan, asin, acos, atan, sinh, cosh, erf, erfc, gelu_erf, softplus, lgamma; `fn2` = pow, atan2, hypot |
| `qdecode`, `qdot` | Decode a quantised row (q8_0, q4_k, q6_k, q5_k; `simd::qfmt`) or dot it against f32 activations without materialising it. Other codecs use the scalar decoder passed in. Results are bit-identical to `dot(x, decode(row))` |

## Integer matvec (`qdot_i8.hpp`)
The CPU quantised matmul (`graph::qweights::gemm_nt`) quantises each activation row once to int8 (blocks of 256 with int16 sums of 16 and 32 for q4_k/q5_k/q6_k; blocks of 32 for q8_0) and forms exact int32 sums per weight block, the K-quant sub-scales folded in. Only one or two `fma`s per block are floating point, in the same order on every ISA, so the `scalar` reference, `avx2` (maddubs + madd), `avxvnni` and `avx512vnni` (vpdpwssd / vpdpbusd on ymm) tables give identical bits, and a row's result does not depend on how many rows ran with it (1..8 vectors share one pass over the weights). `qi8_active()` picks the table once (`FORGE_QI8_TARGET` overrides; `FORGE_SIMD_TARGET=scalar|avx2` also caps it); `FORGE_CPU_INT8=0` returns to the f32 path. Tests: `qdot_i8_matches_reference_on_every_isa` (memcmp of every table against the reference, batch vs single vector, error vs f32 below 1% of sum |x·w|). `avxvnni` is compiled but not run here (no hardware). `forge bench --kernels <MB>` prints GB/s per format.

## Example
`docs/examples/simd_dispatch.cpp` (ctest `doc_example_simd_dispatch`):

```cpp
std::printf("cpu: %s\n", forge::simd::detect().summary().c_str());
const forge::simd::kernels& k = forge::simd::active();   // FORGE_SIMD_TARGET overrides
std::vector<float> x(1000, 1.f), y(1000, 2.f);
k.saxpy(3.f, x.data(), y.data(), x.size());               // y = 3*x + y
float d = k.dot(x.data(), y.data(), x.size());            // 5000
```

## Environment

| Variable | Effect |
|----------|--------|
| `FORGE_SIMD_TARGET` | Force a table (`scalar`, `avx2`, …). Ignored if that table is not usable on this CPU |
| `FORGE_LANE_TARGET` | Old name of `FORGE_SIMD_TARGET`, still read (after it) |
| `FORGE_SVE_KEEP_VL=1` | aarch64 Linux: do not lower the SVE vector length at start-up (see below) |

## Platform notes and limits
- Link `forge_simd_mv` whenever you use simd, directly or through gpu, graph or inference.
- aarch64 Linux: if the hardware SVE length has no table (not a power of two, or above 1024 bits), a start-up constructor lowers the process VL with `prctl(PR_SVE_SET_VL)` to the largest length that has one. With `FORGE_SVE_KEEP_VL=1` the length is kept, `sve` is unusable, and NEON is used.
- Run so far: x86_64 scalar/AVX2/AVX-512 natively. SVE (128–512 bits) and RVV (128–1024 bits) only under qemu-user. AMX natively on a Xeon and under Intel SDE. SME only under qemu. See [STATUS.md](STATUS.md).
- AMX on the Xeon VM failed intermittently. The cause was the VM losing tile registers across context switches, not FORGE (`tools/amx_state_probe.cpp` shows it with no FORGE code). `test_tile` retries before failing.

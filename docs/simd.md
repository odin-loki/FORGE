# simd: CPU features and kernel dispatch

## Concepts
`forge::simd` detects what the CPU supports and what the OS has enabled (XGETBV state for AVX/AVX-512, the Linux AMX tile permission, the SVE vector length, …). It then picks **one table of kernel function pointers** for the process. The choice is made once, on the first call to `active()`; every later call goes through that table.

Kernels are written once and compiled per target:

| Layer | Header | Contents |
|-------|--------|----------|
| Tier P | `vec.hpp`, `vec_gn.hpp`, `vec_types.hpp` | Portable fixed-width vectors and masks over native registers. `vec.hpp` is `vec<float>` with the AVX2, AVX-512, AArch64 NEON, SVE and RVV branches; `vec_gn.hpp` has the 128/256-bit backends listed under "Other architectures"; `vec_types.hpp` (included on demand) adds `vec<T>` for double and the 8/16/32/64-bit integers on every backend: see [Element types](#element-types-vec_typeshpp) |
| Tier A | `algo.hpp`, `qdot.hpp`, `algo_num.hpp`, `algo_search.hpp`, `algo_sort.hpp`, `algo_text.hpp` | Algorithms written against Tier P (dot, softmax, GEMM, quantised-row decode and dot, search, sort, …) |
| Interop | `interop.hpp` | `std::experimental::simd` conversions and `std::mdspan` overloads of the Tier A algorithms, where the standard library has them: see [Standard library interop](#standard-library-interop-interophpp) |
| Tier R | `raw.hpp`, `forge/simd/raw/<isa>.gen.hpp` (generated) | Raw intrinsic wrappers for every architecture, generated from `ops/intrinsics/<isa>.json` by `ops/gen/intrin_gen.py`. Each wrapper requires its CPU feature, so using one the target lacks is a compile error. See [Tier R](#tier-r-raw-intrinsics) |
| Capabilities | `caps.hpp` | Feature tags; a target is a set of features |
| Maths | `math.hpp`, `math_coeffs.hpp` | Vectorised libm. The erf/erfc/lgamma coefficients come from `ops/gen/fit_math.py` (run by hand; the output is checked in) |

`simd/src/` builds one translation unit per target, each with its own flags and namespace, into the static library `forge_simd_mv`.

Every kernel table has three companion tables for the same ISA, each with its own dispatcher that follows the kernel table's name and usability rule:

| Table | Header | Dispatch | Built from |
|-------|--------|----------|------------|
| Tier A algorithms | `tier_a.hpp` | `tier_a_active()`, `tier_a_by_name()` | `tiera_impl.inc` over `algo_*.hpp` |
| Attention and delta rule | `seq.hpp` | `seq_active()`, `seq_by_name()` | `seq_impl.hpp` |
| int8 matvec | `qdot_i8.hpp` | `qi8_active()`, `qi8_by_name()`, `qi8_target_names()` | `qi8_x86.inc` (AVX), `qi8_arm.inc` (NEON, dotprod, SVE), `qi8_gn.inc` (every other ISA) |

`scalar`, `avx2` and `avx512` have their own source files. For every other table (`sse2`, `sse4.2`, `neon`, `sve`, `rvv`, `lasx`, `lsx`, `vsx`, `vxe`, `simd128`, `msa`) one file, `simd/src/isa_extra.cpp`, is compiled next to the kernel table with the same flags (plus `-ffp-contract=off`) and instantiates the same templates for that backend; `isa_target.inc` reads the target from the compile flags. Nothing is forked per ISA. The 32-bit Arm `neon` table (opt-in) has no companions.

Targets per architecture:

| Arch | Targets (preferred first) | Requirements |
|------|---------------------------|--------------|
| x86_64 | `avx512`, `avx2`, `sse4.2`, `sse2`, `scalar` | `avx2`: the x86-64-v3 set (SSE2, SSE4.2, POPCNT, AVX, AVX2, FMA, F16C, BMI2). `avx512`: v3 plus AVX-512 F, BW, DQ, VL, CD. `sse4.2`: SSE2, SSE4.2, POPCNT (the TU is built with `-msse4.2 -mpopcnt`). `sse2`: the x86-64 baseline |
| aarch64 | `sve`, `neon`, `scalar` | `sve`: a table built for exactly the hardware vector length (128, 256, 512 or 1024 bits) |
| riscv64 | `rvv`, `scalar` | `rvv`: a table built for exactly the hardware VLEN (128, 256, 512 or 1024 bits) |
| loongarch64 | `lasx`, `lsx`, `scalar` | `AT_HWCAP` bits LSX (1<<4) and LASX (1<<5) |
| ppc64le | `vsx`, `scalar` | `AT_HWCAP` VSX and `AT_HWCAP2` ISA 2.07 (the TU is built for POWER8) |
| s390x | `vxe`, `scalar` | `AT_HWCAP` VXRS and VXRS_EXT (z14: binary32 vectors) |
| wasm32 | `simd128`, `scalar` | Chosen when the module is built (`-msimd128`); nothing is detected at run time |
| mips64 | `msa`, `scalar` | `AT_HWCAP` MSA |
| arm (32-bit) | `scalar`; `neon` only with `FORGE_SIMD_ARM32_NEON=1` | `AT_HWCAP` NEON and VFPv4. Never chosen automatically (see "Other architectures") |
| x86 (32-bit) | `sse4.2`, `sse2`, `scalar` | As on x86_64. No AVX tables |
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
| `const kernels* by_name(const char*)` | A specific table, or `nullptr` if it is not built or not usable here. Names: `scalar`; `avx512`, `avx2`, `sse4.2` (also `sse42`), `sse2` (x86); `neon`, `sve` (aarch64); `rvv` (riscv64); `lasx`, `lsx`; `vsx`; `vxe`; `simd128`; `msa` |
| `std::span<const char* const> target_names()` | Every table name, preferred first, `scalar` last. `active()` takes the first one `by_name` accepts; `forge info` prints the usable ones as `targets` |

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
The CPU quantised matmul (`graph::qweights::gemm_nt`) quantises each activation row once to int8 (blocks of 256 with int16 sums of 16 and 32 for q4_k/q5_k/q6_k; blocks of 32 for q8_0) and forms exact int32 sums per weight block, the K-quant sub-scales folded in. Only one or two `fma`s per block are floating point, in the same order on every ISA, so the `scalar` reference, the x86 tables `avx2` (maddubs + madd), `avxvnni` and `avx512vnni` (vpdpwssd / vpdpbusd on ymm) and the AArch64 tables `neon` (ARMv8.0: smull/smlal into int16, widening multiply by the sub-scale), `dotprod` (ARMv8.2 sdot) and `sve` (svdot, vector-length agnostic: one object for every VL) give identical bits, and a row's result does not depend on how many rows ran with it (1..8 vectors share one pass over the weights). `qi8_active()` picks the table once: avx512vnni > avxvnni > avx2 on x86; on AArch64 `sve` when the vector length is 256 bits or more, else `dotprod`, else `neon`; `scalar` on every other architecture (`FORGE_QI8_TARGET` overrides; `FORGE_SIMD_TARGET=scalar|avx2|neon` also caps it). `FORGE_CPU_INT8=0` returns to the f32 path. The integer path is the default on x86 only: on AArch64 the tables are built and tested but `gemm_nt` stays on the f32 path unless `FORGE_CPU_INT8=1`, because they have run only under qemu-user and have never been timed on hardware (`qi8_default_on()` is the one place to change). AArch64 detection (`cpu_features::dotprod`, `i8mm`): `getauxval(AT_HWCAP)` on Linux/Android, `hw.optional.arm.FEAT_DotProd` on Apple (always on for macOS), `elf_aux_info` on FreeBSD, `IsProcessorFeaturePresent` on Windows; only the Linux path has been executed. `i8mm` (usdot) is detected but not used: every unsigned K-quant value fits in 7 bits, so sdot is already exact. Tests: `qdot_i8_matches_reference_on_every_isa` (memcmp of every table against the reference, batch vs single vector, error vs f32 below 1% of sum |x·w|). The test also multiplies raw blocks at the ends of the integer ranges (0xFF / 0x80 / 0x7F weight bytes, activations saturated at ±127, a NaN activation). `avxvnni` ran on the 2026-11-12 build machine (a Xeon with AVX-VNNI); `neon`, `dotprod` and `sve` have run under qemu-user only (ctest `qi8_aarch64_*`, label `emu`). `forge bench --kernels <MB>` prints GB/s per format.

Every other Tier P backend has an integer table too: `sse4.2`, `sse2`, `rvv`, `lasx`, `lsx`, `vsx`, `vxe`, `simd128`, `msa`. They are one source, `simd/src/qi8_gn.inc`, written with GNU vector types: 16 int8 lanes are multiplied into int16 lanes and widened to int32 sums with `__builtin_convertvector`, which the compiler lowers to the widening multiplies of the ISA the file is compiled for (the RVV object is vector-length agnostic in effect: 128-bit operations, any VLEN). The integer sums are exact, the float steps are the reference's, and activation quantisation rounds with the add-and-subtract of 1.5·2²³ (the value is at most 127 in magnitude), which gives `nearbyint`'s result; NaN becomes −127 as in the reference. These tables are **not** hand-tuned (no `pmaddubsw`-style instruction selection is forced) and none has been timed: `qi8_default_on()` stays false for them, including `sse4.2` / `sse2` on x86 (only the `avx*` tables are on by default), so they are used only with `FORGE_CPU_INT8=1`. MSVC builds do not have them (no GNU vector types). The block formats are little-endian byte layouts read with `memcpy`; on a big-endian host (s390x) the tables agree with that host's scalar reference, but the fixed-block reference hash that ties every little-endian platform to x86 is not compared there. Test: `test_simd_tables` (`qi8_tables_match_scalar_bit_for_bit`: quantisation of saturated, tied, zero, NaN and infinite activations; products over all-ones / 0x80 / 0x7F / random blocks with scales of either sign; 1..13 vectors per pass) for every table usable on the CPU; it needs only the simd library, so it is the test the emulated architectures run (ctest `tables_*`).

## Tier A algorithms (`tier_a.hpp`)
General-purpose algorithms beside the kernel table, with their own dispatch: `tier_a_active()` returns a `tier_a_kernels` table (`tier_a_by_name(name)` a specific one; the names and usability rules are the kernel tables', `target_names()` lists them). There is a table for every kernel table: `scalar`, `avx512`, `avx2`, `sse4.2`, `sse2`, `neon`, `sve`, `rvv` (one per vector length, like the kernels), `lasx`, `lsx`, `vsx`, `vxe`, `simd128`, `msa`. All are instantiations of the same templates; what differs per table is the row "Vectorisation" below: the `vec<float, Tg>` parts use the table's registers, the hand-written byte search, i32 prefix sum and sort partition exist for AVX2 and AVX-512 only, and everything else is a loop the compiler vectorises with the table's flags. The templates live in `algo_num.hpp`, `algo_search.hpp`, `algo_sort.hpp` and `algo_text.hpp` (namespace `forge::simd::algo`, each a template on the target). Test: `test_simd_tiera` checks every table usable on the CPU against scalar references over random inputs and sizes 0, 1, vector width ± 1, unaligned starts and up to 2²⁰ elements (ctest `tiera_*` runs it per emulated CPU model, and natively with `sse2`, `sse4.2` and `avx2` forced as the active table).

"Explicit" below means hand-written vector code; "compiler" means a plain loop the compiler vectorises with the table's flags; "scalar" means neither.

| Group | Functions | Vectorisation |
|-------|-----------|---------------|
| f32 reduce / compare | `min_f32`, `max_f32` (NaN skipped), `argmin_f32`, `argmax_f32` (first index), `count_f32`, `find_f32`, `any_f32`, `all_f32` (`cmp`: eq, ne, lt, le, gt, ge against a threshold, C++ NaN semantics), `clamp_f32`; generic `find_if`, `count_if`, `any_of`, `all_of` over a `vec -> mask` predicate (templates only) | Explicit (`vec<float>` / `strip`) |
| Prefix sum | `scan_add_f32`, `scan_add_f64` (inclusive or exclusive; defined as the left-to-right sum, so every table gives the same bits), `scan_add_i32` (wraps mod 2³²) | f32/f64 scalar by definition; i32 explicit on AVX2 and AVX-512 |
| Histogram | `histogram_u8` (256 bins), `histogram_f32` (equal-width bins over `[lo, hi)`) | u8 scalar (four interleaved tables); f32 bin index by the compiler, increments scalar |
| Transpose | `transpose_f32/f64` (out of place, 32×32 blocks), `transpose_square_f32/f64` (in place) | Cache-blocked scalar loops |
| f64 / integer BLAS-1 | `dot_f64`, `sum_f64`, `axpy_f64`, `dot_i32`, `dot_i16`, `dot_i8` (exact, 64-bit result), `axpy_i32`, `sum_i32`, `min_i32`, `max_i32`, `find_i32`, `count_eq_i32` | Compiler. `dot_f64` / `sum_f64` use eight fixed partial sums; tables can differ in the last bits (fused multiply-add) |
| Bitsets | `popcount`, `popcount_and`, `bitset_op` (and, or, xor, and-not), `bitset_find_first` | Compiler |
| Byte search | `find_byte` (memchr), `count_byte`, `mismatch`, `find_bytes` (memmem: first/last-byte filter then `memcmp`; worst case O(n·m)) | Explicit on AVX2 (32 bytes) and AVX-512BW (64 bytes); 8 bytes at a time with integer tricks elsewhere (little-endian; byte by byte on s390x) |
| Sorting | `sort_network<N>` and `sort_small` (optimal networks for N ≤ 8, Batcher merge-exchange above), `sort_i32`, `sort_u32`, `sort_f32` (IEEE total order: −NaN < −∞ < … < −0 < +0 < … < +∞ < +NaN, bits preserved), `partial_sort_f32`, `select_f32`, `topk_f32` (indices and values of the k largest, ties by lower index, NaN lowest) | Quicksort with an explicit partition on AVX-512 (compress-store) and AVX2 (permute table), branch-free scalar elsewhere; ≤ 32 keys by network; heap sort takes over past a depth limit. The sorts allocate O(n) scratch |
| Text | `utf8_error` / `utf8_valid` (strict: overlongs, surrogates, > U+10FFFF and truncation rejected), `hex_encode`, `hex_decode`, `base64_encode`, `base64_decode` (RFC 4648 standard padded and URL-safe unpadded; strict) | UTF-8: explicit ASCII-run skip, other sequences scalar (not the table-lookup validator). Hex: compiler. Base64: scalar tables |

`topk_f32` is not used by the sampler (`inference/sampler.hpp` keeps its own `nth_element` + sort): the tie order there is part of the engine's reproducibility rules and was not changed.

## Attention and delta-rule kernels (`seq.hpp`)
`seq_kernels` (chunked attention over an f32 or int8 KV cache, the int16-query integer path, the gated delta rule; used by `graph::attn_cpu` and `state_ops`) is written once in `seq_impl.hpp` against a 16-lane vector type `V` whose operations are all exactly rounded and whose horizontal sums use one fixed tree, so every table returns the same bits. `scalar`, `avx2` and `avx512` implement `V` directly; every other table implements it with Tier P: 16 / N registers of `vec<float, Tg>` (N = 4, 8, 16), or the low 16 lanes of one register when N = 32 (1024-bit SVE / RVV: loads and stores are masked). Details that matter for bit identity:

- SSE and WebAssembly have no fused multiply-add instruction and Tier P's `fma` there rounds twice, so those two backends compute `fma` (and the exponential, which is built from it) per lane with `forge::fmaf_exact`. Their tables are correct and bit-identical but have no speed advantage in the fused steps over `scalar`; they were not timed.
- `exp` scales by 2ⁿ with Tier P's `ldexp_int` (a multiply by a constructed power of two, `svscale` on SVE): exact for the range the kernel clamps to.
- Maximum uses Tier P's `max` (the second operand on a tie or NaN), which is what the AVX2 table does.
- The int16 × int8 integer dot of the q16 path is a plain loop the compiler vectorises; integer sums are exact in any order.

Test: `test_simd_tables` (`seq_tables_match_scalar_bit_for_bit`): `exp1` on a grid of 18,001 arguments, `attn_chunk` and `attn_chunk_q16` over f32 and int8 keys and values, d = 16 … 600 (16- and 32-lane tails, the lowered q16 code range), 1 to 9 slots, full, tail, short offset and empty-slot chunks, `attn_merge`, `attn_finish`, `l2norm_scale`, `delta_cols` for 1 … 32 columns: `memcmp` against the scalar table. `test_graph` (`attn_cpu_*`) repeats the comparison through the graph for every usable table.

## Element types (`vec_types.hpp`)
`#include <forge/simd/vec_types.hpp>` adds `vec<T, Tg>` and `mask<T, Tg>` for `double`, `int8_t` … `int64_t` and `uint8_t` … `uint64_t`, and half-precision / bfloat16 loads and stores of `vec<float, Tg>`. `vec.hpp` is unchanged (float only) and does not include it. A vector is one native register of the target: `vec<T, Tg>::bytes / sizeof(T)` lanes, 64 bytes for AVX-512, 32 for AVX2 and LASX, 16 for SSE, NEON, LSX, VSX, VXE, SIMD128 and MSA, the table's fixed length for SVE and RVV; 16 bytes for `scalar_t` and for any target the translation unit is not compiled for.

```cpp
#include <forge/simd/vec_types.hpp>
using T = forge::simd::x86_64_v3;                       // in a translation unit built with -mavx2 -mfma
using VI = forge::simd::vec<std::int16_t, T>;           // 16 lanes
auto m  = forge::simd::mask<std::int16_t, T>::first_n(n - i);   // tail mask
VI x = VI::load(a + i, m), y = VI::load(b + i, m);      // masked loads never touch the inactive lanes' memory
store(o + i, add_sat(x, y), m);                         // saturating add, masked store
auto wide = widen_lo(x) * widen_lo(y);                  // vec<int32_t, T>: the low 8 lanes as int32 products
```

| Group | Operations |
|-------|------------|
| Construct | `vec::zero()`, `vec::splat(x)`, `vec::iota(start, step)`, `v[i]`, `v.to_array(p)` |
| Load / store | `vec::load(p)` and `store(p, v)` (any alignment), `vec::load_aligned(p)` / `store_aligned(p, v)` (aligned to the register size), `vec::load(p, mask)` / `store(p, v, mask)` (inactive lanes read as 0, are not written, and their memory is never touched), `stream(p, v)` + `stream_fence()` (non-temporal) |
| Arithmetic | `+ - *` (integers wrap), `/` (double), unary `-`, `min`, `max`, `abs`, `add_sat`, `sub_sat` (integers), `sqrt`, `fma` (double) |
| Bits | `& \| ^ ~`, `bit_andnot`, `shl` / `shr` and `<< >>` by one count or per-lane counts (arithmetic right shift for signed types), `popcount` |
| Compare / select | `eq ne lt le gt ge` → `mask`; `select(m, a, b)`; mask `& \| ^ !`, `first_n`, `all`, `from_bits`, `bits()`, `count()`, `first()`, `any()`, `all_set()`, `none()`, `m[i]` |
| Convert | `convert<U>(v)` (same lane width: int32 ↔ uint32, int64 ↔ double), `bitcast<U>(v)`, `to_int` / `to_float` / `bitcast` / `bitcast_f32` (with `vec<float>`), `widen_lo` / `widen_hi` (to twice the width), `narrow(lo, hi)` (truncating) and `narrow_sat<U>(lo, hi)` (saturating, to the half-width type), `promote_lo` / `promote_hi` (float → double), `demote(lo, hi)` |
| Gather / scatter | `vec::gather(base, idx)`, `scatter(base, idx, v)`: N `int32_t` indices in [0, 2²⁸); a scatter with repeated indices keeps the highest lane |
| Reduce | `reduce_add`, `reduce_min`, `reduce_max`, `reduce_and`, `reduce_or`, `reduce_xor` (double `reduce_add`: the fixed tree `(v[i] + v[i + N/2])` halved down to one lane) |
| f16 / bf16 | `load_f16<Tg>(const uint16_t*)`, `store_f16`, `load_bf16`, `store_bf16`: N half or bfloat16 values ↔ `vec<float, Tg>` |

The exact semantics of each operation are at the top of the header.

**How each backend implements it.** On GCC and Clang the register is a GNU vector type and the element-wise operations are vector operators and builtins, which the compiler lowers to the instructions of the ISA the translation unit is built for: the same source is the SSE2, AVX2, AVX-512, NEON, SVE, RVV, LSX, LASX, VSX, VXE, SIMD128 and MSA backend. Intrinsics are used where no generic form exists:

| Operation | Native | Elsewhere |
|-----------|--------|-----------|
| Masked load / store | AVX-512 (`vmovdqu8/16/32/64` with a mask register), AVX2 for 32/64-bit lanes (`vpmaskmov`), SVE (`svld1` / `svst1` with a predicate), RVV (`vle` / `vse` with a mask) | Lane by lane, active lanes only |
| Gather | AVX2 and AVX-512 (32/64-bit lanes), SVE (32/64-bit lanes), RVV (every lane width, `vluxei32`) | Lane by lane |
| Scatter | AVX-512, SVE (32/64-bit lanes), RVV (`vsoxei32`, ordered) | Lane by lane |
| Streaming store | x86 `movntdq` / `vmovntdq`; Clang's non-temporal store hint on other ISAs (meant for AArch64 `stnp`; that form has not been compiled here, the AArch64 build used GCC) | An aligned store (GCC off x86) |
| Saturating add / sub, popcount | Clang: `__builtin_elementwise_add_sat` / `_sub_sat` / `_popcount` (native `padds`, `sqadd`, `vsadd`, `cnt`, …) | GCC: compare-and-select and bit-trick formulas on vectors |
| f16 | F16C (`vcvtph2ps`), AVX-512, AArch64 NEON (`fcvtl` / `fcvtn`) | `forge::f16` per lane (SSE, SVE, RVV, LSX/LASX, VSX, VXE, SIMD128, MSA) |
| bf16 | — | Integer shifts and the `forge::bf16` rounding on every ISA (AVX512-BF16's `vcvtneps2bf16` is not used: the kernel tables are not built with it) |

Where a translation unit is not compiled for any vector ISA (baseline mips64, riscv64 without V, …) and on MSVC (which has no GNU vector types), every target uses a portable array implementation with the same semantics; `emulated_t` selects it on any compiler. So on MSVC `vec<T, Tg>` is always 16 bytes of plain loops: the intrinsics above are not used there.

**Tests** (`tests/simd_types_check.hpp`; FORGE_PLAN A.10): every operation against a scalar reference written with plain integer arithmetic. Exhaustive: all 2¹⁶ input pairs for every binary int8 and uint8 operation (add, sub, mul, min, max, saturating add / sub, and, or, xor, and-not, per-lane shifts, six compares, select) and all 2¹⁶ inputs for every unary int16 and uint16 operation and conversion (abs, negate, not, popcount, shifts by every count, widen, narrow, saturating narrow to either signedness, sign conversion); 32/64-bit lanes and double: 23 edge values crossed with each other and 120 random values; masked loads and stores for every length against exactly-sized heap blocks; random masks; gathers and scatters including repeated indices; reductions; int32 ↔ float, int64 ↔ double, float ↔ double; all 65,536 half-precision and bfloat16 patterns in, 200,000 floats out. About 25.5 million checks per target. `test_simd_types` runs it at the baseline flags (`scalar_t` and `emulated_t`); `test_simd_types_<table>` (ctest `types_<table>`) is the same check compiled with one kernel table's flags, with a baseline `main` that exits 77 (skipped) when the CPU cannot run that table.

Compiler defects found by these tests under zig 0.17's clang (reproduced without FORGE code in `tools/repro_llvm_vector_codegen.cpp`):
- mips64 MSA: the generic vector `fma` on `double` is emitted as `fmadd.d` with the operands rotated (a + b·c), and a sign-mask `abs` feeding a vector square root is dropped. `vec_types.hpp` uses the MSA intrinsics for `fma`, `sqrt` and `abs` on `double`. Baseline mips64 (no MSA) mis-legalises int32 vector compares, which is why builds without a vector ISA use the portable arrays.
- loongarch64: the half-precision conversion loop is miscompiled when vectorised with LASX (it is kept out of line there), and a 16-byte struct of int8 passed by value comes back with one lane wrong at `-O2`. The second one only affects the portable arrays (`emulated_t`), which no LoongArch build uses; `test_simd_types` reports those checks on loongarch64 without counting them.
- Building *libc* with `-mmsa` (zig rebuilding musl for the flag) did not terminate; the toolchain file passes the flag per translation unit, which is unaffected.

## Standard library interop (`interop.hpp`)
`#include <forge/simd/interop.hpp>` (FORGE_PLAN A.8). Each half exists only when the standard library has the type; `FORGE_SIMD_HAVE_STD_SIMD` and `FORGE_SIMD_HAVE_MDSPAN` say which.

| Library feature | Macro tested | What you get |
|-----------------|--------------|--------------|
| `std::experimental::simd` (`<experimental/simd>`, libstdc++ 11+) | `__cpp_lib_experimental_parallel_simd` | `to_std(v)` → `std_simd<T, Tg>` (the `simd` with `vec<T, Tg>::N` lanes: the native ABI for that count when there is one, else `fixed_size`), `from_std<Tg>(s)`, and the same pair for `mask<T, Tg>` ↔ `simd_mask`. `T` is float or any `vec_types.hpp` type. The conversion is a store and a load of one register |
| `std::mdspan` (`<mdspan>`, libc++ 18+, libstdc++ 15+) | `__cpp_lib_mdspan` | `forge::simd::md::` overloads of the Tier A table entries taking `std::mdspan` with the default accessor: `min`, `max`, `argmin`, `argmax`, `count`, `find`, `any`, `all`, `clamp`, `scan_add`, `sum`, `dot`, `find_byte`, `count_byte`, `histogram`, `sort`, `partial_sort`, `select`, `topk`, `argmax_rows` (rank 2), `transpose` (rank 2). A view whose elements are contiguous in row-major order is used in place; any other layout (`layout_stride` with gaps, `layout_left` of rank > 1; up to rank 3) is copied to a temporary and, for the functions that write, copied back |
| C++26 `std::simd` (`<simd>`) | `__cpp_lib_simd` | **Nothing yet.** No standard library available to this repository ships it, so no conversion could be compiled |

Test: `test_simd_std` (the half the default library has; libstdc++ 13: the `simd` half) and `test_simd_std_libcxx` (the same source built with `-stdlib=libc++` when Clang has a libc++ with `<mdspan>`; it compiles the scalar Tier A table from source so that nothing in it mixes two standard libraries). With clang 23 on the build machine: libstdc++ 13 has `<experimental/simd>` and no `<mdspan>`; libc++ 23 has `<mdspan>` and no feature-tested `<experimental/simd>`. So each half has been compiled and run against exactly one library; the mdspan half has not been compiled against libstdc++ 15, and the conversions have run for `scalar_t` and `emulated_t` registers only.

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
| `FORGE_SIMD_TARGET` | Force a table (`scalar`, `avx2`, `sse2`, `lsx`, …: any `by_name` name). Ignored if that table is not usable on this CPU |
| `FORGE_LANE_TARGET` | Old name of `FORGE_SIMD_TARGET`, still read (after it) |
| `FORGE_SVE_KEEP_VL=1` | aarch64 Linux: do not lower the SVE vector length at start-up (see below) |
| `FORGE_SIMD_ARM32_NEON=1` | 32-bit Arm: make the `neon` table selectable with `FORGE_SIMD_TARGET=neon` (it flushes subnormals to zero) |

## Other architectures (`vec_gn.hpp`)
`vec_gn.hpp` is one more Tier P backend ("kind 6") with one implementation per ISA, selected by the compile flags of the translation unit. `simd/src/kernels_gn.cpp` is compiled once per table with that ISA's flags and exports the same `kernels` table as every other target (same Tier A templates, nothing forked). `vec_gn.hpp` itself has `float` vectors only; the other element types of these ISAs come from `vec_types.hpp`.

| ISA | Tag / preset | Lanes | Header, flags | Notes |
|-----|--------------|-------|---------------|-------|
| x86 SSE2 | `sse2` / `x86_64_v1` | 4 | `emmintrin.h`, none on x86-64 | Intrinsics only (also MSVC). No FMA instruction: `fma` goes through double (exact product, one extra rounding; within 2⁻²⁹ ulp of correctly rounded). Rounding and select are emulated |
| x86 SSE4.2 | `sse2` / `x86_64_v2` | 4 | `smmintrin.h`, `-msse4.2 -mpopcnt` | As SSE2 with `roundps` and `blendvps` |
| LoongArch LSX | `lsx` / `loongarch_lsx` | 4 | `lsxintrin.h`, `-mlsx` | |
| LoongArch LASX | `lasx` / `loongarch_lasx` | 8 | `lasxintrin.h`, `-mlasx` | |
| Power VSX | `vsx` / `power8` | 4 | `altivec.h`, `-mcpu=power8 -mvsx` | ppc64le only (big-endian ppc64 gets `scalar`) |
| s390x vector | `s390_vxe` / `s390x_z14` | 4 | `vecintrin.h`, `-march=z14 -mzvector` | Needs z14 (binary32 vectors); z13 gets `scalar`. Big-endian |
| WebAssembly SIMD128 | `wasm_simd128` / `wasm_simd` | 4 | `wasm_simd128.h`, `-msimd128` | `fma` through f64 as for SSE. relaxed-simd is not used: its `madd` may or may not fuse, so results would differ between engines |
| MIPS MSA | `msa` / `mips_msa` | 4 | `msa.h`, `-mmsa` | |
| Armv7 NEON + VFPv4 | `neon` / `armv7_neon` | 4 | `arm_neon.h`, `-mfpu=neon-vfpv4` | Opt-in (`FORGE_SIMD_ARM32_NEON=1`): Advanced SIMD on AArch32 flushes subnormals to zero, so `exp` below −87, `cbrt` and `erfc` of values with subnormal results miss the tolerance |

Apart from SSE, the backends use GNU vector types for `+ - * /`, compares, bit operations and shifts, and the ISA's intrinsics for fused multiply-add, square root, rounding and mask extraction. Those TUs are built with `-ffp-contract=off`. All `vec_gn.hpp` functions have internal linkage, so the SSE2 and SSE4.2 tables (or LSX and LASX) never share a function body.

Detection (`dispatch.cpp`) reads `AT_HWCAP` / `AT_HWCAP2` with `getauxval` (Linux) or `elf_aux_info` (FreeBSD; not run). A feature the whole build already requires (`__loongarch_sx`, `__VSX__` with POWER8) counts as present.

What has run, all under emulation except SSE (2026-11-12; details in [STATUS.md](STATUS.md)):

| Table | Compiled with | Run | `test_simd` |
|-------|---------------|-----|-------------|
| `sse2`, `sse4.2` (x86_64) | clang 23, g++ 13 (compile only) | Natively (forced with `FORGE_SIMD_TARGET`), and `qemu-x86_64 -cpu Nehalem` / `-cpu core2duo` (no AVX; core2duo has no SSE4.2) | 7/7 |
| `sse2`, `sse4.2` (32-bit x86) | zig 0.17 (clang 22), by hand | `qemu-i386 -cpu Nehalem`, `-cpu core2duo` | 7/7. The 32-bit CMake branch has not been configured |
| `lasx`, `lsx` | zig 0.17, CMake + toolchain file | `qemu-loongarch64 -cpu la464` | 7/7, each table also forced |
| `vsx` | zig 0.17, CMake + toolchain file; g++ 13 by hand | `qemu-ppc64le -cpu power9` and `-cpu power8` | 7/7 |
| `vxe` | zig 0.17, CMake + toolchain file; g++ 13 by hand | `qemu-s390x -cpu max` | 7/7. Not run on a z14-only CPU model (qemu-user rejects `-cpu z14`) |
| `simd128` | zig 0.17 `wasm32-wasi`, `tests/wasm_simd.sh` | node 22 (V8) | 7/7 |
| `msa` | zig 0.17, CMake + toolchain file | `qemu-mips64el -cpu Loongson-3A4000` | 7/7 |
| `neon` (32-bit Arm) | zig 0.17 `-mcpu=cortex_a7`, by hand | `qemu-arm -cpu cortex-a7` | 5/7: the three subnormal sweeps fail, hence opt-in. CMake branch not configured |

The companion tables (Tier A, `seq`, int8) and `vec_types.hpp`, per ISA (2026-11-12). "Tables" is `test_simd_tables` (3 tests: companions present and active, `seq` bit-identical to scalar, int8 bit-identical to scalar); "Tier A" is `test_simd_tiera` (14 tests, every usable table against scalar references); "types" is the `vec_types.hpp` check of that register width (about 25.5 million checks). Everything outside x86_64 ran under an emulator only, and none of it on real hardware of its architecture.

| ISA / table | Compiled with | Run on | Tables | Tier A | Types |
|-------------|---------------|--------|--------|--------|-------|
| x86_64 `sse2`, `sse4.2`, `avx2`, `avx512`; int8 also `avxvnni`, `avx512vnni` | clang 23; also g++ 13 by hand for the `sse2` / `sse4.2` companion tables (3/3) and the `sse4.2`, `avx2`, `avx512` types checks (0 failed) | the build machine (a Xeon with AVX-512 and AVX-VNNI), all tables in one run and each lower one forced | 3/3 | 14/14 | `scalar`, `emulated`, `sse2`, `sse4.2`, `avx2`, `avx512`: 0 failed |
| AArch64 `neon` | g++ 13 cross | `qemu-aarch64 -cpu cortex-a57` and `-cpu max,sve=off` | 3/3 each | 14/14 each | 0 failed (GCC: the formula forms of saturation and popcount) |
| AArch64 `sve` (tables for 128, 256, 512, 1024 bits) | g++ 13 cross | `qemu-aarch64 -cpu max,sve<N>=on` at 128, 256, 384, 512, 640, 1024, 1536, 2048 bits (384 and 640 are lowered to the next table; above 1024 to 1024) | 3/3 at all 8 | 14/14 at all 8 | 128, 256, 512, 1024: 0 failed (predicated loads / stores, gathers, scatters through the SVE intrinsics) |
| riscv64 `rvv` (tables for VLEN 128, 256, 512, 1024) | zig 0.17 (clang 22) | `qemu-riscv64 -cpu rv64,v=true,vlen=<N>` at all four | 3/3 at all 4 | 14/14 at all 4 | 0 failed at all 4 (masked loads / stores, `vluxei32` gathers, `vsoxei32` scatters) |
| loongarch64 `lsx`, `lasx` | zig 0.17 | `qemu-loongarch64 -cpu la464` | 3/3 each | 14/14 each | `lsx`, `lasx`: 0 failed. `emulated_t` (not used on LoongArch): 46,976 of the 25.5 million checks differ, a compiler defect (see [Element types](#element-types-vec_typeshpp)) |
| ppc64le `vsx` | zig 0.17 | `qemu-ppc64le -cpu power9` | 3/3 | 14/14 | 0 failed |
| s390x `vxe` (big-endian) | zig 0.17 | `qemu-s390x -cpu max` | 3/3 (the little-endian reference hash is not compared) | 14/14 | 0 failed |
| mips64el `msa` | zig 0.17 | `qemu-mips64el -cpu Loongson-3A4000` | 3/3 | 14/14 | 0 failed after the MSA workarounds; baseline mips64 uses the portable arrays |
| wasm32 `simd128` | zig 0.17 `wasm32-wasi`, `tests/wasm_simd.sh` | node 22 | 3/3 | 14/14, one test per process (node itself segfaults part-way when all 14 run in one process; each passes alone, also with `--liftoff-only`) | 0 failed |
| 32-bit x86, 32-bit Arm | — | — | not built, not run | not built | not built |

Not run for any of these: MSVC (which gets the portable arrays for `vec_types.hpp` and no GNU-vector int8 tables), Apple clang, the NDK, GCC 14, FreeBSD; GCC for any architecture but AArch64; any timing. The int8 and `seq` tables of the non-x86 architectures are bit-identical to scalar under emulation and have never been timed, so none is enabled by default beyond what was before (`seq` and Tier A dispatch do pick them: they give the same bits as scalar by construction and by test).

"By hand" means `kernels_gn.cpp`, `kernels_scalar.cpp`, `dispatch.cpp`, `qi8_dispatch.cpp`, `seq_scalar.cpp`, `seq_dispatch.cpp` and `tests/test_simd.cpp` compiled and linked directly with the flags the CMake branch uses. None of these tables has run on real hardware of its architecture, and no test other than `test_simd` has run on these architectures.

Cross-build and test (zig wrappers as in [building.md](building.md); GNU cross compilers work the same way without the two `FORGE_CROSS_*` options):

```sh
cmake -S . -B build/loongarch64 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/loongarch64-lasx-qemu.cmake \
  -DFORGE_CROSS_CC=/path/zig-cc-loongarch64 -DFORGE_CROSS_CXX=/path/zig-c++-loongarch64 \
  -DCMAKE_EXE_LINKER_FLAGS=-static -DFORGE_BUILD_EXAMPLES=OFF
ninja -C build/loongarch64 test_simd
qemu-loongarch64 -cpu la464 build/loongarch64/test_simd      # or: ctest -R 'test_simd|lane_'
```

The same with `ppc64le-vsx-qemu.cmake` (`zig c++ -target powerpc64le-linux-musl`, `qemu-ppc64le -cpu power9`), `s390x-vxe-qemu.cmake` (`s390x-linux-musl`, `qemu-s390x -cpu max`) and `mips64el-msa-qemu.cmake` (`mips64el-linux-muslabi64`, `qemu-mips64el -cpu Loongson-3A4000`). Under an emulator ctest adds `lane_<table>`: `test_simd` with that table forced, and for the companion tables `tables_<table>`, `tiera_<table>` and `types_<table>` (`…_vl<N>` per vector length for SVE and RVV, `…_neon_a57` / `…_neon_nosve` for NEON; label `emu`). Natively on x86_64, `lane_sse2` does the same, `tiera_sse2` / `tiera_sse4.2` / `tiera_avx2` force each lower table, and `types_sse2`, `types_sse42`, `types_avx2`, `types_avx512` run where the CPU has the ISA (skipped otherwise).

### WebAssembly
FORGE as a whole is not ported to WebAssembly and there is no toolchain file. What works on `wasm32-wasi` is the simd layer: the headers under `forge/simd/` (with `forge/core/dtypes.hpp` and `fma.hpp`), the `scalar` and `simd128` kernel tables, `dispatch.cpp` (`detect()`, `active()`, `by_name()`), the scalar `seq` and `qi8` tables, and `tests/test_simd.cpp` including its oracle (`forge/verify/oracle.hpp`, `forge/ops`), which compile unchanged. `tests/wasm_simd.sh` builds exactly that, with the `simd128` companion tables (`isa_extra.cpp`), `test_simd_tables`, `test_simd_tiera`, `test_simd_types` and the SIMD128 `vec_types.hpp` check, and runs them:

```sh
cmake --preset dev                                  # generated headers
CXX="zig c++ -target wasm32-wasi" tests/wasm_simd.sh      # runs under node; WASM_RUN="wasmtime run --dir=." for wasmtime
```

A module is validated as a whole, so SIMD128 is a build-time choice: with `-msimd128` the `simd128` table is compiled in and active, with `WASM_SIMD=0` (CMake: `-DFORGE_WASM_SIMD=OFF`) only `scalar` exists. zig's wasi C++ runtime has no exception support; the script links `tests/wasm_no_exceptions.cpp`, which aborts on a throw. The `WASM_SIMD=0` build (scalar tables only) ran the same tests on 2026-11-12. Not run: wasmtime, a browser, Emscripten, and the `wasm` branch of `CMakeLists.txt` (written, never configured).

## Platform notes and limits
- Link `forge_simd_mv` whenever you use simd, directly or through gpu, graph or inference.
- aarch64 Linux: if the hardware SVE length has no table (not a power of two, or above 1024 bits), a start-up constructor lowers the process VL with `prctl(PR_SVE_SET_VL)` to the largest length that has one. With `FORGE_SVE_KEEP_VL=1` the length is kept, `sve` is unusable, and NEON is used.
- MSVC (`cl`) builds have no CPUID detection in `dispatch.cpp`, so only `scalar` is usable there; the SSE tables are compiled but never selected. `vec_types.hpp` uses its portable arrays for every target on MSVC, and the GNU-vector int8 tables are not built.
- Run so far: x86_64 scalar/SSE2/SSE4.2/AVX2/AVX-512 natively; the `vec_gn.hpp` tables for other architectures only under emulation (table above). SVE (128–512 bits) and RVV (128–1024 bits) only under qemu-user. AMX natively on a Xeon and under Intel SDE. SME only under qemu. See [STATUS.md](STATUS.md).
- AMX on the Xeon VM failed intermittently. The cause was the VM losing tile registers across context switches, not FORGE (`tools/amx_state_probe.cpp` shows it with no FORGE code). `test_tile` retries before failing.

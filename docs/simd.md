# simd: CPU features and kernel dispatch

## Concepts
`forge::simd` detects what the CPU supports and what the OS has enabled (XGETBV state for AVX/AVX-512, the Linux AMX tile permission, the SVE vector length, …). It then picks **one table of kernel function pointers** for the process. The choice is made once, on the first call to `active()`; every later call goes through that table.

Kernels are written once and compiled per target:

| Layer | Header | Contents |
|-------|--------|----------|
| Tier P | `vec.hpp` | Portable fixed-width vectors and masks over native registers |
| Tier A | `algo.hpp`, `qdot.hpp` | Algorithms written against Tier P (dot, softmax, GEMM, quantised-row decode and dot, …) |
| Tier R | `forge/simd/raw/x86.gen.hpp` (generated) | Raw x86 intrinsic wrappers from `ops/intrinsics/x86.json` (`ops/gen/intrin_gen.py`). Each wrapper requires its CPU feature, so using one the target lacks is a compile error |
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
The CPU quantised matmul (`graph::qweights::gemm_nt`) quantises each activation row once to int8 (blocks of 256 with int16 sums of 16 and 32 for q4_k/q5_k/q6_k; blocks of 32 for q8_0) and forms exact int32 sums per weight block, the K-quant sub-scales folded in. Only one or two `fma`s per block are floating point, in the same order on every ISA, so the `scalar` reference, the x86 tables `avx2` (maddubs + madd), `avxvnni` and `avx512vnni` (vpdpwssd / vpdpbusd on ymm) and the AArch64 tables `neon` (ARMv8.0: smull/smlal into int16, widening multiply by the sub-scale), `dotprod` (ARMv8.2 sdot) and `sve` (svdot, vector-length agnostic: one object for every VL) give identical bits, and a row's result does not depend on how many rows ran with it (1..8 vectors share one pass over the weights). `qi8_active()` picks the table once: avx512vnni > avxvnni > avx2 on x86; on AArch64 `sve` when the vector length is 256 bits or more, else `dotprod`, else `neon`; `scalar` on every other architecture (`FORGE_QI8_TARGET` overrides; `FORGE_SIMD_TARGET=scalar|avx2|neon` also caps it). `FORGE_CPU_INT8=0` returns to the f32 path. The integer path is the default on x86 only: on AArch64 the tables are built and tested but `gemm_nt` stays on the f32 path unless `FORGE_CPU_INT8=1`, because they have run only under qemu-user and have never been timed on hardware (`qi8_default_on()` is the one place to change). AArch64 detection (`cpu_features::dotprod`, `i8mm`): `getauxval(AT_HWCAP)` on Linux/Android, `hw.optional.arm.FEAT_DotProd` on Apple (always on for macOS), `elf_aux_info` on FreeBSD, `IsProcessorFeaturePresent` on Windows; only the Linux path has been executed. `i8mm` (usdot) is detected but not used: every unsigned K-quant value fits in 7 bits, so sdot is already exact. Tests: `qdot_i8_matches_reference_on_every_isa` (memcmp of every table against the reference, batch vs single vector, error vs f32 below 1% of sum |x·w|). The test also multiplies raw blocks at the ends of the integer ranges (0xFF / 0x80 / 0x7F weight bytes, activations saturated at ±127, a NaN activation). `avxvnni` is compiled but not run here (no hardware); `neon`, `dotprod` and `sve` have run under qemu-user only (ctest `qi8_aarch64_*`, label `emu`). `forge bench --kernels <MB>` prints GB/s per format.

## Tier A algorithms (`tier_a.hpp`)
General-purpose algorithms beside the kernel table, with their own dispatch: `tier_a_active()` returns a `tier_a_kernels` table (`tier_a_by_name("scalar" | "avx2" | "avx512")` a specific one). Tables exist for `scalar` everywhere and `avx2` / `avx512` on x86_64; there are no NEON, SVE or RVV tables (the `scalar` table is compiled with the baseline flags, so the compiler may still vectorise its loops). The templates live in `algo_num.hpp`, `algo_search.hpp`, `algo_sort.hpp` and `algo_text.hpp` (namespace `forge::simd::algo`, each a template on the target). Test: `test_simd_tiera` checks every table against scalar references over random inputs and sizes 0, 1, vector width ± 1, unaligned starts and up to 2²⁰ elements.

"Explicit" below means hand-written vector code; "compiler" means a plain loop the compiler vectorises with the table's flags; "scalar" means neither.

| Group | Functions | Vectorisation |
|-------|-----------|---------------|
| f32 reduce / compare | `min_f32`, `max_f32` (NaN skipped), `argmin_f32`, `argmax_f32` (first index), `count_f32`, `find_f32`, `any_f32`, `all_f32` (`cmp`: eq, ne, lt, le, gt, ge against a threshold, C++ NaN semantics), `clamp_f32`; generic `find_if`, `count_if`, `any_of`, `all_of` over a `vec -> mask` predicate (templates only) | Explicit (`vec<float>` / `strip`) |
| Prefix sum | `scan_add_f32`, `scan_add_f64` (inclusive or exclusive; defined as the left-to-right sum, so every table gives the same bits), `scan_add_i32` (wraps mod 2³²) | f32/f64 scalar by definition; i32 explicit on AVX2 and AVX-512 |
| Histogram | `histogram_u8` (256 bins), `histogram_f32` (equal-width bins over `[lo, hi)`) | u8 scalar (four interleaved tables); f32 bin index by the compiler, increments scalar |
| Transpose | `transpose_f32/f64` (out of place, 32×32 blocks), `transpose_square_f32/f64` (in place) | Cache-blocked scalar loops |
| f64 / integer BLAS-1 | `dot_f64`, `sum_f64`, `axpy_f64`, `dot_i32`, `dot_i16`, `dot_i8` (exact, 64-bit result), `axpy_i32`, `sum_i32`, `min_i32`, `max_i32`, `find_i32`, `count_eq_i32` | Compiler. `dot_f64` / `sum_f64` use eight fixed partial sums; tables can differ in the last bits (fused multiply-add) |
| Bitsets | `popcount`, `popcount_and`, `bitset_op` (and, or, xor, and-not), `bitset_find_first` | Compiler |
| Byte search | `find_byte` (memchr), `count_byte`, `mismatch`, `find_bytes` (memmem: first/last-byte filter then `memcmp`; worst case O(n·m)) | Explicit on AVX2 (32 bytes) and AVX-512BW (64 bytes); 8 bytes at a time with integer tricks elsewhere |
| Sorting | `sort_network<N>` and `sort_small` (optimal networks for N ≤ 8, Batcher merge-exchange above), `sort_i32`, `sort_u32`, `sort_f32` (IEEE total order: −NaN < −∞ < … < −0 < +0 < … < +∞ < +NaN, bits preserved), `partial_sort_f32`, `select_f32`, `topk_f32` (indices and values of the k largest, ties by lower index, NaN lowest) | Quicksort with an explicit partition on AVX-512 (compress-store) and AVX2 (permute table), branch-free scalar elsewhere; ≤ 32 keys by network; heap sort takes over past a depth limit. The sorts allocate O(n) scratch |
| Text | `utf8_error` / `utf8_valid` (strict: overlongs, surrogates, > U+10FFFF and truncation rejected), `hex_encode`, `hex_decode`, `base64_encode`, `base64_decode` (RFC 4648 standard padded and URL-safe unpadded; strict) | UTF-8: explicit ASCII-run skip, other sequences scalar (not the table-lookup validator). Hex: compiler. Base64: scalar tables |

`topk_f32` is not used by the sampler (`inference/sampler.hpp` keeps its own `nth_element` + sort): the tie order there is part of the engine's reproducibility rules and was not changed.

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

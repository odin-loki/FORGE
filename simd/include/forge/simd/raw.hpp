#pragma once
// Tier R umbrella: the generated raw intrinsic wrappers of the architecture being compiled for.
//
//   #include <forge/simd/raw.hpp>
//   namespace r = forge::simd::raw::native;                  // raw::x86, raw::arm, raw::riscv, ...
//   auto v = r::avx2::mm256_add_epi32<forge::simd::x86_64_v3>(a, b);
//
// Each wrapper requires its feature tags (caps.hpp), so calling one the Target lacks does not compile.
// The headers are large (tens of thousands of templates on Arm and RISC-V): include this, or one
// forge/simd/raw/<isa>.gen.hpp, only in the translation units that use raw intrinsics. Nothing else in FORGE
// includes it. Every <isa>.gen.hpp is includable on every platform and is empty off its architecture.
// The RISC-V tail/mask policy variants (_tu, _tum, _tumu, _mu) are a separate opt-in header,
// forge/simd/raw/riscv_policy.gen.hpp.
#include <forge/simd/caps.hpp>

#if defined(__x86_64__) || defined(_M_X64)
#include <forge/simd/raw/x86.gen.hpp>
namespace forge::simd::raw { namespace native = x86; inline constexpr const char* isa = "x86"; }
#elif defined(__aarch64__) || defined(_M_ARM64) || (defined(__arm__) && defined(__ARM_NEON))
#include <forge/simd/raw/arm.gen.hpp>
namespace forge::simd::raw { namespace native = arm; inline constexpr const char* isa = "arm"; }
#elif defined(__riscv)
#include <forge/simd/raw/riscv.gen.hpp>
namespace forge::simd::raw { namespace native = riscv; inline constexpr const char* isa = "riscv"; }
#elif defined(__loongarch__)
#include <forge/simd/raw/loongarch.gen.hpp>
namespace forge::simd::raw { namespace native = loongarch; inline constexpr const char* isa = "loongarch"; }
#elif defined(__powerpc__) || defined(__powerpc64__)
#include <forge/simd/raw/ppc.gen.hpp>
namespace forge::simd::raw { namespace native = ppc; inline constexpr const char* isa = "ppc"; }
#elif defined(__wasm__)
#include <forge/simd/raw/wasm.gen.hpp>
namespace forge::simd::raw { namespace native = wasm; inline constexpr const char* isa = "wasm"; }
#elif defined(__s390x__)
#include <forge/simd/raw/s390x.gen.hpp>
namespace forge::simd::raw { namespace native = s390x; inline constexpr const char* isa = "s390x"; }
#elif defined(__mips__)
#include <forge/simd/raw/mips.gen.hpp>
namespace forge::simd::raw { namespace native = mips; inline constexpr const char* isa = "mips"; }
#else
namespace forge::simd::raw { namespace none { inline constexpr int intrinsic_count = 0; } namespace native = none; inline constexpr const char* isa = "none"; }
#endif

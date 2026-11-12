#pragma once
// Tier R support macros shared by the generated forge/simd/raw/<isa>.gen.hpp headers.
// The intrinsic databases are derived from one compiler's headers (see docs/simd.md, Tier R); an intrinsic
// that an older or different compiler does not provide is gated here rather than left out.

#if defined(__clang__)
#define FORGE_RAW_CLANG 1
#else
#define FORGE_RAW_CLANG 0
#endif
// Upstream LLVM version checks. Apple's clang uses its own version numbers, so it never passes one.
#if defined(__clang__) && !defined(__apple_build_version__)
#define FORGE_RAW_CLANG_GE(major) (__clang_major__ >= (major))
#else
#define FORGE_RAW_CLANG_GE(major) 0
#endif
#if defined(__GNUC__) && !defined(__clang__)
#define FORGE_RAW_GCC 1
#else
#define FORGE_RAW_GCC 0
#endif
#define FORGE_RAW_GCC_GE(major) (FORGE_RAW_GCC && __GNUC__ >= (major))
#if defined(_MSC_VER) && !defined(__clang__)
#define FORGE_RAW_MSVC 1
#else
#define FORGE_RAW_MSVC 0
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#define FORGE_RAW_A64 1
#else
#define FORGE_RAW_A64 0
#endif

#if FORGE_RAW_MSVC
#define FORGE_RAW_INLINE __forceinline
#else
#define FORGE_RAW_INLINE [[gnu::always_inline]] inline
#endif

namespace forge::simd::raw {
namespace fs = ::forge::simd;   // the generated requires-clauses name the feature tags through this alias
// dep<T>::id<U> is U, spelled so that it depends on the wrapper's Target parameter (see FORGE_RAW_ARG).
template <class> struct dep { template <class U> using id = U; };
}
// An argument cast to its own type through dep<T>: a no-op that makes the intrinsic call a dependent expression,
// so it is resolved when the wrapper is instantiated rather than when the header is parsed. No comma survives the
// expansion (the intrinsic may be a macro that forwards its arguments to another macro).
#define FORGE_RAW_ARG(type, x) static_cast<typename ::forge::simd::raw::dep<T>::template id<type>>(x)

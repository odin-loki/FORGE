#pragma once
// Private to simd/src: the tables one ISA translation unit (isa_extra.cpp) exports beside its kernel table,
// and the lookup dispatch.cpp provides for the Tier A, seq and int8 dispatchers.
#include <forge/simd/qdot_i8.hpp>
#include <forge/simd/seq.hpp>
#include <forge/simd/tier_a.hpp>

namespace forge::simd {

struct isa_tables {
    const char* target;
    const tier_a_kernels* tiera;
    const seq_kernels* seq;
    const qi8_kernels* qi8;     // nullptr: the ISA has its own integer kernels (AArch64) or none (MSVC builds)
};

// The extra tables of the kernel table by_name(t) returns; nullptr when that table is unusable here or has
// none (scalar, avx2, avx512: those three have their own sources).
const isa_tables* isa_extra_by_name(const char* t);

} // namespace forge::simd

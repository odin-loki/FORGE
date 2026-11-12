// Driver of one per-table vec_types.hpp test executable (tests/simd_types_isa.cpp): compiled with the baseline
// flags and free of templates shared with the ISA translation unit, so nothing built for the ISA runs before
// the dispatcher has confirmed that this CPU has it. Exit code 77 (ctest: skipped) when it does not.
#include <forge/simd/dispatch.hpp>
#include <cstdio>
#include <cstring>

extern "C" const char* forge_types_table();
extern "C" int forge_types_bits();
extern "C" int forge_types_run();

int main() {
    const char* t = forge_types_table();
    const auto& f = forge::simd::detect();
    bool usable = forge::simd::by_name(t) != nullptr;
    // the scalable tables are per vector length: only the one built for this CPU's length can run
    if (std::strcmp(t, "sve") == 0) usable = usable && f.sve_bytes * 8 == forge_types_bits();
    if (std::strcmp(t, "rvv") == 0) usable = usable && f.rvv_bytes * 8 == forge_types_bits();
    if (!usable) { std::printf("vec_types %s (%d-bit registers): not usable on this CPU, skipped\n", t, forge_types_bits()); return 77; }
    const int fails = forge_types_run();
    std::printf("%d failure(s)\n", fails);
    return fails ? 1 : 0;
}

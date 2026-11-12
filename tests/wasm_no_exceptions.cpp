// tests/wasm_simd.sh only: wasm32-wasi C++ runtimes are usually built without exception support, so the
// two entry points a `throw` needs are missing at link time. These weak fallbacks turn a throw into an abort
// with a message (the simd tests never throw on a passing run); a runtime that has the real ones wins.
#include <cstdio>
#include <cstdlib>

extern "C" {
__attribute__((weak)) void* __cxa_allocate_exception(std::size_t n) noexcept { return std::malloc(n); }
__attribute__((weak)) void __cxa_throw(void*, void*, void (*)(void*)) {
    std::fputs("wasm test build: C++ exception thrown, aborting (no exception support in this runtime)\n", stderr);
    std::abort();
}
}

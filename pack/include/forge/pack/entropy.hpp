#pragma once
// Portable OS entropy: getentropy needs Android API 28 / macOS 10.12; arc4random_buf is everywhere on BSD/Apple/Bionic.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#if defined(_WIN32)
#include <windows.h>
#include <bcrypt.h>
#elif !defined(__ANDROID__) && !defined(__APPLE__)
#include <sys/random.h>
#endif

namespace forge::pack {
inline int os_entropy(void* p, std::size_t n) {
#if defined(_WIN32)
    return BCryptGenRandom(nullptr, static_cast<PUCHAR>(p), static_cast<ULONG>(n), BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0 ? 0 : -1;
#elif defined(__ANDROID__) || defined(__APPLE__)
    ::arc4random_buf(p, n);
    return 0;
#else
    auto* b = static_cast<std::uint8_t*>(p);
    while (n) { auto r = ::getrandom(b, n, 0); if (r <= 0) return -1; b += r; n -= std::size_t(r); }
    return 0;
#endif
}
}  // namespace forge::pack

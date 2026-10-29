#pragma once
// Counter-based Philox4x32-10: identical streams on every backend (pure function of key+counter).
#include <array>
#include <cstdint>

namespace forge::rand {
struct philox4x32 {
    std::array<std::uint32_t, 2> key;
    static constexpr std::array<std::uint32_t, 4> block(std::array<std::uint32_t, 4> c, std::array<std::uint32_t, 2> k) {
        for (int r = 0; r < 10; ++r) {
            std::uint64_t p0 = std::uint64_t(0xD2511F53u) * c[0], p1 = std::uint64_t(0xCD9E8D57u) * c[2];
            c = {std::uint32_t(p1 >> 32) ^ c[1] ^ k[0], std::uint32_t(p1), std::uint32_t(p0 >> 32) ^ c[3] ^ k[1], std::uint32_t(p0)};
            k[0] += 0x9E3779B9u; k[1] += 0xBB67AE85u;
        }
        return c;
    }
    constexpr std::array<std::uint32_t, 4> operator()(std::uint64_t counter, std::uint64_t stream = 0) const {
        return block({std::uint32_t(counter), std::uint32_t(counter >> 32), std::uint32_t(stream), std::uint32_t(stream >> 32)}, key);
    }
    static constexpr float to_unit(std::uint32_t x) { return float(x >> 8) * (1.0f / 16777216.0f); }
};
} // namespace forge::rand

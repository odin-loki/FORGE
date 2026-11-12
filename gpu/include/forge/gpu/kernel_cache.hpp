#pragma once
// On-disk cache of driver-compiled kernel programs (OpenCL program binaries, Level Zero native module binaries), so a
// runtime's open() does not make the driver compile the generated kernels again. Stored next to the Vulkan pipeline
// cache: $FORGE_CACHE_DIR, else $XDG_CACHE_HOME/forge, else ~/.cache/forge (%LOCALAPPDATA%\forge on Windows).
//
//   * Key: backend kind, device name, vendor, device and driver version strings, a hash of the kernel source / IL and
//     the build options. The file name carries a hash of the key; the full key is stored in the file and compared, so
//     a hash collision or a file copied from another machine is never used.
//   * File: magic, key, payload size, FNV-1a of the payload, payload. Anything that does not check out (truncated,
//     edited, other key) is ignored and the program is built from source again; so is a binary the driver refuses.
//   * Written atomically (temporary file + rename) with owner-only permissions (0600; the directory is created 0700).
//     A file that is a symlink, or (POSIX) not owned by the current user, is not read.
//   * FORGE_GPU_KERNEL_CACHE=0 turns reading and writing off.
// A program binary is code the driver will run: the cache directory must not be writable by other users. The
// permissions above are what this file enforces; pointing FORGE_CACHE_DIR at a shared directory is the caller's risk.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>
#if !defined(_WIN32)
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace forge::gpu::kernel_cache {

inline bool enabled() { const char* e = std::getenv("FORGE_GPU_KERNEL_CACHE"); return !(e && std::string(e) == "0"); }
inline std::uint64_t fnv(std::string_view s, std::uint64_t h = 1469598103934665603ull) {
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
    return h;
}
inline std::string dir() {
    if (const char* e = std::getenv("FORGE_CACHE_DIR"); e && *e) return e;
    if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) return std::string(x) + "/forge";
    if (const char* h = std::getenv("HOME"); h && *h) return std::string(h) + "/.cache/forge";
    if (const char* l = std::getenv("LOCALAPPDATA"); l && *l) return std::string(l) + "/forge";
    return {};
}
// "<dir>/<kind>-<hash of key>.bin"; empty when there is no cache directory.
inline std::string path(const std::string& kind, const std::string& key) {
    std::string d = dir(); if (d.empty()) return {};
    char buf[32]; std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(fnv(key)));
    return d + "/" + kind + "-" + buf + ".bin";
}
inline constexpr char magic[8] = {'F', 'G', 'K', 'C', '1', 0, 0, 0};

// The cached payload for (kind, key); empty when absent, disabled or invalid in any way.
inline std::vector<unsigned char> load(const std::string& kind, const std::string& key) {
    std::vector<unsigned char> out;
    if (!enabled()) return out;
    const std::string p = path(kind, key); if (p.empty()) return out;
    std::error_code ec;
    auto st = std::filesystem::symlink_status(p, ec);
    if (ec || !std::filesystem::is_regular_file(st)) return out;   // absent, or a symlink / special file: never followed
#if !defined(_WIN32)
    struct stat sb{};
    if (::lstat(p.c_str(), &sb) != 0 || sb.st_uid != ::geteuid() || (sb.st_mode & (S_IWGRP | S_IWOTH))) return out;
#endif
    std::ifstream in(p, std::ios::binary); if (!in) return out;
    std::vector<char> d((std::istreambuf_iterator<char>(in)), {});
    std::size_t o = 0;
    auto u64 = [&](std::uint64_t& v) { if (d.size() - o < 8) return false; std::memcpy(&v, d.data() + o, 8); o += 8; return true; };
    if (d.size() < sizeof magic || std::memcmp(d.data(), magic, sizeof magic) != 0) return out;
    o = sizeof magic;
    std::uint64_t klen = 0, n = 0, sum = 0;
    if (!u64(klen) || klen != key.size() || d.size() - o < klen || std::memcmp(d.data() + o, key.data(), klen) != 0) return out;
    o += klen;
    if (!u64(n) || !u64(sum) || d.size() - o != n || !n) return out;
    if (fnv({d.data() + o, std::size_t(n)}) != sum) return out;
    out.assign(d.begin() + std::ptrdiff_t(o), d.end());
    return out;
}
// Store payload for (kind, key); false when disabled or the file could not be written (never an error for the caller).
inline bool store(const std::string& kind, const std::string& key, const unsigned char* data, std::size_t n) {
    if (!enabled() || !n) return false;
    const std::string p = path(kind, key); if (p.empty()) return false;
    std::error_code ec;
    const auto parent = std::filesystem::path(p).parent_path();
    if (!std::filesystem::exists(parent, ec)) {
        std::filesystem::create_directories(parent, ec); if (ec) return false;
        std::filesystem::permissions(parent, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace, ec);
    }
    static unsigned seq = 0;
    const std::string tmp = p + ".tmp" + std::to_string(std::uint64_t(reinterpret_cast<std::uintptr_t>(&seq)) ^ fnv(key)) + "." + std::to_string(++seq);
    std::filesystem::remove(tmp, ec);
    {
        std::ofstream o(tmp, std::ios::binary); if (!o) return false;
        o.close();   // owner-only before any content is written
        std::filesystem::permissions(tmp, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write, std::filesystem::perm_options::replace, ec);
        if (ec) { std::filesystem::remove(tmp, ec); return false; }
        o.open(tmp, std::ios::binary | std::ios::app); if (!o) { std::filesystem::remove(tmp, ec); return false; }
        const std::uint64_t klen = key.size(), len = n, sum = fnv({reinterpret_cast<const char*>(data), n});
        o.write(magic, sizeof magic); o.write(reinterpret_cast<const char*>(&klen), 8); o.write(key.data(), std::streamsize(klen));
        o.write(reinterpret_cast<const char*>(&len), 8); o.write(reinterpret_cast<const char*>(&sum), 8);
        o.write(reinterpret_cast<const char*>(data), std::streamsize(n));
        if (!o) { o.close(); std::filesystem::remove(tmp, ec); return false; }
    }
    std::filesystem::rename(tmp, p, ec);
    if (ec) { std::filesystem::remove(tmp, ec); return false; }
    return true;
}

} // namespace forge::gpu::kernel_cache

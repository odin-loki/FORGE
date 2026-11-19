#pragma once
// forge::cache: a content-addressed on-disk store keyed by BLAKE3.
//
//   cache::store c(dir, {.max_bytes = 1 << 30});
//   auto k = c.put("blob");                         // content address: key = blake3(content)
//   auto m = cache::key_of({"solver", query, opts}); // memo key from several inputs (length-prefixed)
//   c.put(m, result); if (auto v = c.get(m)) ...
//
// Layout: <dir>/<first 2 hex>/<remaining 62 hex>. Each file is a 32-byte BLAKE3 of the value followed by
// the value; get() verifies it and deletes a corrupt entry. Writes go through os::fs::write_file_atomic, so
// readers (also other processes) see a whole entry or none. get() refreshes the entry's mtime; with
// max_bytes set, put() evicts the least recently used entries until the store fits. Thread-safe.
//
// Raw files (no header): put_file(key, src) keeps a whole file as it is under <dir>/raw/..., get_path(key) returns its path (and
// refreshes its LRU time), so a file that other code must open itself (a SQLite shard, a model blob) can live in the store and be
// shared by every user of the same key. They share the byte budget and the LRU order with the headered entries. With
// verify = true get_path() hashes the file and drops it when BLAKE3 differs from the key (use it when the key is the content hash).
#include <forge/hash/blake3.hpp>
#include <forge/os/fs.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace forge::cache {
using key = hash::blake3::digest;

inline key key_of(std::string_view data) { hash::blake3 h; h.update(data.data(), data.size()); return h.final(); }
// Several inputs, each prefixed by its length so ("ab","c") and ("a","bc") give different keys.
inline key key_of(std::initializer_list<std::string_view> parts) {
    hash::blake3 h;
    for (auto p : parts) { std::uint64_t n = p.size(); std::uint8_t b[8]; for (int i = 0; i < 8; ++i) b[i] = std::uint8_t(n >> (8 * i)); h.update(b, 8); h.update(p.data(), p.size()); }
    return h.final();
}
inline std::string to_hex(const key& k) { return hash::blake3::hex(k); }
inline std::optional<key> from_hex(std::string_view s) {
    if (s.size() != 64) return std::nullopt;
    key k{}; auto v = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
    for (std::size_t i = 0; i < 32; ++i) { int a = v(s[2 * i]), b = v(s[2 * i + 1]); if (a < 0 || b < 0) return std::nullopt; k[i] = std::uint8_t(a * 16 + b); }
    return k;
}

struct options { std::uint64_t max_bytes = 0; };   // 0 = unbounded
struct store_stats { std::uint64_t entries = 0, bytes = 0, hits = 0, misses = 0, evictions = 0, corrupt = 0; };

class store {
public:
    explicit store(std::filesystem::path dir, options o = {}) : dir_(std::move(dir)), opt_(o) {
        std::filesystem::create_directories(dir_);
        std::lock_guard g(m_); rescan_();
    }
    const std::filesystem::path& dir() const { return dir_; }
    std::filesystem::path path_of(const key& k) const { auto h = to_hex(k); return dir_ / h.substr(0, 2) / h.substr(2); }

    key put(std::string_view value) { auto k = key_of(value); put(k, value); return k; }
    // Returns false if the write failed (the store is unchanged then).
    bool put(const key& k, std::string_view value) {
        auto v = key_of(value);
        std::string blob(reinterpret_cast<const char*>(v.data()), v.size()); blob.append(value);
        auto p = path_of(k);
        std::error_code ec; std::filesystem::create_directories(p.parent_path(), ec);
        std::lock_guard g(m_);
        std::uint64_t old = file_size_(p);
        if (auto st = os::fs::write_file_atomic(p.string(), blob); st.err) return false;
        if (old) { st_.bytes -= std::min(st_.bytes, old); } else ++st_.entries;
        st_.bytes += blob.size();
        if (opt_.max_bytes && st_.bytes > opt_.max_bytes) evict_to_(opt_.max_bytes, &p);
        return true;
    }
    std::optional<std::string> get(const key& k) {
        auto p = path_of(k);
        std::ifstream f(p, std::ios::binary);
        std::lock_guard g(m_);
        if (!f) { ++st_.misses; return std::nullopt; }
        std::string blob((std::istreambuf_iterator<char>(f)), {}); f.close();
        if (blob.size() < 32 || [&] { auto v = key_of(std::string_view(blob).substr(32)); return !std::equal(v.begin(), v.end(), reinterpret_cast<const std::uint8_t*>(blob.data())); }()) {
            ++st_.corrupt; ++st_.misses; remove_(p); return std::nullopt;
        }
        std::error_code ec; std::filesystem::last_write_time(p, std::filesystem::file_time_type::clock::now(), ec);
        ++st_.hits;
        return blob.substr(32);
    }

    // ---- raw files: kept as they are, no header ------------------------------------------------------------------
    std::filesystem::path raw_path_of(const key& k) const { auto h = to_hex(k); return dir_ / "raw" / h.substr(0, 2) / h.substr(2); }
    // Keeps a copy of the file `src` under `k` (a hard link when the store is on the same volume and `link` is true, else a copy
    // written next to the entry and renamed into place, so readers see a whole file or none). An existing entry is only touched.
    // Returns false if the file could not be stored.
    bool put_file(const key& k, const std::filesystem::path& src, bool link = true) {
        auto p = raw_path_of(k);
        std::error_code ec; std::filesystem::create_directories(p.parent_path(), ec);
        std::lock_guard g(m_);
        if (std::filesystem::is_regular_file(p, ec)) { std::filesystem::last_write_time(p, std::filesystem::file_time_type::clock::now(), ec); return true; }
        const auto tmp = p.parent_path() / (".tmp-" + p.filename().string() + "-" + std::to_string(++tmp_seq_));
        std::filesystem::remove(tmp, ec);
        bool ok = false;
        if (link) { std::filesystem::create_hard_link(src, tmp, ec); ok = !ec; }
        if (!ok) { ec.clear(); ok = std::filesystem::copy_file(src, tmp, std::filesystem::copy_options::overwrite_existing, ec) && !ec; }
        if (!ok) { std::filesystem::remove(tmp, ec); return false; }
        std::filesystem::rename(tmp, p, ec);
        if (ec) { std::filesystem::remove(tmp, ec); return false; }
        ++st_.entries; st_.bytes += file_size_(p);
        if (opt_.max_bytes && st_.bytes > opt_.max_bytes) evict_to_(opt_.max_bytes, &p);
        return true;
    }
    // The path of the raw file stored under `k`, or nothing. Refreshes the entry's LRU time. verify: BLAKE3 of the file must equal
    // `k` (for keys that are content hashes); a mismatch removes the entry and counts as corrupt.
    std::optional<std::filesystem::path> get_path(const key& k, bool verify = false) {
        auto p = raw_path_of(k);
        std::lock_guard g(m_);
        std::error_code ec;
        if (!std::filesystem::is_regular_file(p, ec)) { ++st_.misses; return std::nullopt; }
        if (verify) {
            hash::blake3 h; std::string buf(1 << 20, '0');
            { std::ifstream f(p, std::ios::binary);   // closed before a possible delete (Windows)
              while (f) { f.read(buf.data(), std::streamsize(buf.size())); auto n = std::size_t(f.gcount()); if (!n) break; h.update(buf.data(), n); } }
            if (h.final() != k) { ++st_.corrupt; ++st_.misses; remove_(p); return std::nullopt; }
        }
        std::filesystem::last_write_time(p, std::filesystem::file_time_type::clock::now(), ec);
        ++st_.hits;
        return p;
    }
    bool contains_file(const key& k) const { std::error_code ec; return std::filesystem::is_regular_file(raw_path_of(k), ec); }
    bool erase_file(const key& k) { std::lock_guard g(m_); return remove_(raw_path_of(k)); }
    bool contains(const key& k) const { std::error_code ec; return std::filesystem::is_regular_file(path_of(k), ec); }
    bool erase(const key& k) { std::lock_guard g(m_); return remove_(path_of(k)); }
    // Evict least recently used entries until the store holds at most `bytes`. Returns bytes freed.
    std::uint64_t prune(std::uint64_t bytes) { std::lock_guard g(m_); rescan_(); return evict_to_(bytes, nullptr); }
    void clear() { prune(0); }
    store_stats stats() const { std::lock_guard g(m_); return st_; }
    void refresh() { std::lock_guard g(m_); rescan_(); }   // re-count after other processes wrote

private:
    static std::uint64_t file_size_(const std::filesystem::path& p) { std::error_code ec; auto n = std::filesystem::file_size(p, ec); return ec ? 0 : n; }
    bool remove_(const std::filesystem::path& p) {
        auto n = file_size_(p); std::error_code ec;
        if (!std::filesystem::remove(p, ec)) return false;
        st_.bytes -= std::min(st_.bytes, n); if (st_.entries) --st_.entries;
        return true;
    }
    struct ent { std::filesystem::path p; std::filesystem::file_time_type t; std::uint64_t n; };
    std::vector<ent> list_() const {
        std::vector<ent> v; std::error_code ec;
        for (auto it = std::filesystem::recursive_directory_iterator(dir_, ec); !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
            if (!it->is_regular_file(ec)) continue;
            auto name = it->path().filename().string();
            if (name.size() != 62 || it->path().parent_path().filename().string().size() != 2) continue;   // skip temp files and strangers
            v.push_back({it->path(), it->last_write_time(ec), it->file_size(ec)});
        }
        return v;
    }
    void rescan_() { st_.entries = 0; st_.bytes = 0; for (auto& e : list_()) { ++st_.entries; st_.bytes += e.n; } }
    std::uint64_t evict_to_(std::uint64_t target, const std::filesystem::path* keep) {
        auto v = list_();
        std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.t < b.t; });
        std::uint64_t freed = 0;
        for (auto& e : v) {
            if (st_.bytes <= target) break;
            if (keep && e.p == *keep) continue;
            if (remove_(e.p)) { freed += e.n; ++st_.evictions; }
        }
        return freed;
    }
    std::filesystem::path dir_;
    options opt_;
    mutable std::mutex m_;
    store_stats st_;
    std::uint64_t tmp_seq_ = 0;
};
} // namespace forge::cache

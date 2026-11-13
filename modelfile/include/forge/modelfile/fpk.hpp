#pragma once
#include <forge/os/os.hpp>
#include <forge/core/mapped.hpp>
#include "forge/modelfile/entropy.hpp"
// .fpk container: one file holding manifest, blueprint, tokenizer, tensor index and tensor data.
//
//   [header 64B] [sections ... each aligned to 64B] [section table]
//   header : "FORGEPK1" | u32 version | u32 n_sections | u64 table_offset | u64 file_size | 32B reserved
//   table  : n x { char name[32]; u64 offset; u64 size; u8 blake3[32]; u8 pad[16] }  (96 B each)
//   "tensors" section is JSON: [{name, codec, shape, offset, bytes, blake3}], offsets relative to "data".
//   Every section and every tensor carries a BLAKE3 hash; readers verify on demand (`verify()`).
//   Optional encryption hook: the "data" section may be sealed with XChaCha20-Poly1305 (Monocypher; the
//   plan's AES-GCM was swapped for an audited, constant-time, dependency-free AEAD). An "encryption"
//   section records {alg, nonce, mac, key_id}; the tensor index is bound as associated data. Metadata stays
//   readable, so signing, inspection and verification of section hashes work without the key.
//   Two schemes exist (docs/format.md "Encryption"):
//     alg "xchacha20-poly1305"          one AEAD over the whole data section (decrypted whole at open; written only on request)
//     alg "xchacha20-poly1305-chunked"  what `encrypt_file` writes: the data section is sealed in fixed-size chunks,
//       ciphertext in place (same offsets as the plaintext), one 16-byte tag per chunk in the "enc_macs" section.
//       Chunk i uses the file nonce with i XORed into its last 8 bytes, and authenticates
//       "forge-fpk-enc-v2\0" | file nonce | u64 i | u64 n_chunks | u64 data_bytes | u64 chunk_bytes | BLAKE3(tensors),
//       so a chunk cannot be moved, dropped, or taken from another file, and the index cannot be edited.
//       A reader decrypts only the chunks a tensor needs, when the tensor is first used.
//   Sharded files (docs/format.md "Shards"; modelfile/shard.hpp writes them): the tensor bytes live outside the
//   container, in content-addressed blob files "<blob dir>/<hh>/<blake3 hex>.fpkb". A "shards" section lists the
//   blobs {blake3, bytes}; every "tensors" entry then carries "shard" (an index into that list) and its offset is
//   within the blob; "data" is empty. Blobs are looked up in $FORGE_BLOB_DIR and in "blobs" next to the file, mapped
//   when a tensor in them is first used, and each tensor is hashed against the index before its bytes are handed out.
#include <forge/core/error.hpp>
#include <forge/core/tensor.hpp>
#include <forge/graph/codecs.hpp>
#include <forge/hash/blake3.hpp>
#include <forge/json/json.hpp>
#include <forge/security/enforce.hpp>
#include <forge/security/files.hpp>
#include <forge/security/path.hpp>
#include <filesystem>
#include <set>
#include "monocypher.h"
#include <forge/os/locked_memory.hpp>
#include <cmath>
#include <memory>
#include <mutex>
#include <new>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <map>

namespace forge::modelfile {

inline constexpr char magic[8] = {'F', 'O', 'R', 'G', 'E', 'P', 'K', '1'};
inline constexpr std::uint32_t format_version = 1;
// Minor revision of version 1 (manifest "forge_format_minor"): 1 adds the optional "experts" section (an index of
// stacked MoE expert tensors whose per-expert slices start on expert_alignment file offsets). Readers that predate it
// ignore the section; the tensors themselves are ordinary entries of "tensors".
// 2 marks a file whose "blueprint" section is in the binary form (modelfile/blueprint_bin.hpp), which readers before
// that form cannot parse (they stop with a blueprint parse error); files with a text blueprint stay at their minor.
inline constexpr std::uint32_t format_minor = 1;
inline constexpr std::uint32_t format_minor_binary_blueprint = 2;
inline constexpr std::size_t alignment = 64;            // sections, and tensors of files that record no "tensor_alignment"
inline constexpr std::size_t expert_alignment = 4096;
// Tensor alignment (manifest "tensor_alignment"): every tensor starts on a file offset that is a multiple of it, so a
// tensor can be mapped, read with direct I/O or handed to a device upload on its own. Writers take 64 (compact), 4096
// (page) or 65536 (large pages / DMA); any power of two from 64 to 1 MiB is a valid value in a file.
inline constexpr std::size_t page_alignment = 4096, large_alignment = 65536;
inline constexpr std::uint64_t auto_alignment_threshold = std::uint64_t(64) << 20;
inline bool valid_tensor_alignment(std::uint64_t a) { return a >= alignment && a <= (std::uint64_t(1) << 20) && (a & (a - 1)) == 0; }
// The alignment used when none is asked for: page alignment once the weights are large enough that the padding
// (half a page per tensor on average) is noise, 64 bytes below that, where it would dominate small files.
inline std::size_t default_tensor_alignment(std::uint64_t weight_bytes) { return weight_bytes >= auto_alignment_threshold ? page_alignment : alignment; }
namespace detail {
// manifest JSON with "tensor_alignment" set (unchanged when it is not a JSON object)
inline std::string stamp_alignment(const std::string& manifest, std::size_t a) {
    json::value m;
    try { m = json::parse(manifest); } catch (const forge_exception&) { return manifest; }
    if (!m.is_obj()) return manifest;
    auto& o = m.obj_mut(); bool have = false;
    for (auto& [k, v] : o) if (k == "tensor_alignment") { v = json::value(double(a)); have = true; }
    if (!have) o.emplace_back("tensor_alignment", json::value(double(a)));
    return json::dump(m);
}
}

struct tensor_entry {
    std::string name, codec;
    shape_t shape;
    std::uint64_t offset = 0, bytes = 0;
    std::string blake3;
    int shard = -1;   // sharded files: index into the "shards" list (offset is then within that blob); -1: in "data"
};
inline constexpr char shards_section[] = "shards";
inline constexpr std::string_view blob_ext = ".fpkb";
inline constexpr std::uint32_t format_minor_sharded = 3;
struct shard_info { std::string blake3; std::uint64_t bytes = 0; };
inline bool is_blake3_hex(std::string_view h) {
    if (h.size() != 64) return false;
    for (char c : h) if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}
// Path of blob `hash` under `dir`: "<dir>/<first two hex digits>/<hash>.fpkb". The hash comes from a model file, so
// it is checked to be 64 lowercase hex digits and the join goes through security::safe_join (FSS CAP-3): nothing a
// file can put in its index names a path outside the blob directory.
inline std::string blob_path(const std::string& dir, std::string_view hash) {
    if (!is_blake3_hex(hash)) raise(errc::format_error, "blob name '{}' is not a BLAKE3 hash", std::string(hash).substr(0, 80));
    return security::safe_join(dir, std::string(hash.substr(0, 2)) + "/" + std::string(hash) + std::string(blob_ext));
}
// Where the blobs of the model file at `path` are looked for, in order: $FORGE_BLOB_DIR, then "blobs" next to the file.
inline std::vector<std::string> blob_search_dirs(const std::string& path) {
    std::vector<std::string> d;
    if (auto* e = std::getenv("FORGE_BLOB_DIR"); e && *e) d.emplace_back(e);
    if (!path.empty()) {
        auto parent = os::fs::u8path(path).parent_path();
        d.push_back(os::fs::u8str((parent.empty() ? std::filesystem::path(".") : parent) / "blobs"));
    }
    return d;
}

class writer {
public:
    void set_section(std::string name, std::string bytes) { sections_[std::move(name)] = std::move(bytes); }
    void set_manifest(const json::value& m) { set_section("manifest", json::dump(m)); }
    // Encodes f32 data with the named codec (any of graph::codecs::all()).
    void add_tensor(std::string name, const shape_t& shape, std::span<const float> data, std::string_view codec = "f32") {
        auto* c = graph::codecs::find(codec);
        if (!c) raise(errc::unsupported, "unknown codec '{}'", codec);
        if (std::size_t(numel(shape)) != data.size()) raise(errc::shape_mismatch, "tensor {}: shape/data mismatch", name);
        if (c->block > 1 && data.size() % std::size_t(c->block)) c = graph::codecs::find("f16");  // fallback for odd sizes
        data_.resize((data_.size() + align_ - 1) / align_ * align_, 0);
        tensor_entry e{std::move(name), std::string(c->name), shape, data_.size(), c->bytes(data.size()), {}};
        data_.resize(data_.size() + e.bytes);
        c->encode(data.data(), data.size(), data_.data() + e.offset);
        e.blake3 = hash::blake3::hex(hash::blake3::of({data_.data() + e.offset, e.bytes}));
        tensors_.push_back(std::move(e));
    }
    // Stores already-encoded blocks (e.g. Q8_0 / Q4_K straight from a GGUF file) without re-quantising.
    void add_raw(std::string name, const shape_t& shape, std::string_view codec, std::span<const std::uint8_t> bytes) {
        auto* c = graph::codecs::find(codec);
        if (!c) raise(errc::unsupported, "unknown codec '{}'", codec);
        if (c->bytes(std::size_t(numel(shape))) != bytes.size()) raise(errc::format_error, "tensor {}: {} bytes do not match {} {}", name, bytes.size(), codec, shape_str(shape));
        data_.resize((data_.size() + align_ - 1) / align_ * align_, 0);
        tensor_entry e{std::move(name), std::string(c->name), shape, data_.size(), bytes.size(), {}};
        data_.insert(data_.end(), bytes.begin(), bytes.end());
        e.blake3 = hash::blake3::hex(hash::blake3::of(bytes));
        tensors_.push_back(std::move(e));
    }
    // Tensor alignment of the file (64, 4096, 65536; see valid_tensor_alignment). Set it before the first tensor.
    void set_alignment(std::size_t a) {
        if (!valid_tensor_alignment(a)) raise(errc::invalid_argument, "tensor alignment {} is not a power of two from 64 to 1 MiB", a);
        if (!tensors_.empty()) raise(errc::invalid_argument, "set_alignment: tensors were already added");
        align_ = a;
    }
    std::size_t tensor_alignment() const { return align_; }
    void write(const std::string& path) {
        if (auto it = sections_.find("manifest"); it != sections_.end()) it->second = detail::stamp_alignment(it->second, align_);
        json::array idx;
        for (auto& t : tensors_) {
            json::array sh; for (auto d : t.shape) sh.push_back(double(d));
            idx.push_back(json::object{{"name", t.name}, {"codec", t.codec}, {"shape", sh}, {"offset", double(t.offset)},
                                       {"bytes", double(t.bytes)}, {"blake3", t.blake3}});
        }
        sections_["tensors"] = json::dump(idx);
        // Lay out sections first, then stream them to disk: the tensor data is never copied into a second buffer,
        // so peak memory stays at roughly one copy of the encoded weights (matters for 8B+ models).
        struct ent { std::string name; const std::uint8_t* p; std::uint64_t off, size; hash::blake3::digest h; };
        std::vector<ent> table;
        std::uint64_t pos = 64;
        auto place = [&](const std::string& nm, const std::uint8_t* p, std::size_t n, std::size_t al = alignment) {
            pos = (pos + al - 1) / al * al;
            table.push_back({nm, p, pos, n, hash::blake3::of({p, n})});
            pos += n;
        };
        for (auto& [nm, s] : sections_) place(nm, reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
        place("data", data_.data(), data_.size(), align_);   // tensor offsets are multiples of align_ within data
        std::uint64_t toff = (pos + alignment - 1) / alignment * alignment;
        std::vector<std::uint8_t> tail;
        for (auto& e : table) {
            std::uint8_t rec[96] = {};
            if (e.name.size() >= 32) raise(errc::invalid_argument, "section name too long: {}", e.name);
            std::memcpy(rec, e.name.data(), e.name.size());
            std::memcpy(rec + 32, &e.off, 8); std::memcpy(rec + 40, &e.size, 8); std::memcpy(rec + 48, e.h.data(), 32);
            tail.insert(tail.end(), rec, rec + 96);
        }
        std::uint32_t n = std::uint32_t(table.size());
        std::uint64_t fsz = toff + tail.size();
        std::uint8_t head[64] = {};
        std::memcpy(head, magic, 8); std::memcpy(head + 8, &format_version, 4); std::memcpy(head + 12, &n, 4);
        std::memcpy(head + 16, &toff, 8); std::memcpy(head + 24, &fsz, 8);
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) raise(errc::io_error, "cannot write {}", path);
        std::uint64_t at = 0;
        static const std::uint8_t zeros[alignment] = {};
        auto emit = [&](const std::uint8_t* p, std::size_t k) {
            if (k && std::fwrite(p, 1, k, f) != k) { std::fclose(f); raise(errc::io_error, "short write {}", path); }
            at += k;
        };
        auto pad_to = [&](std::uint64_t off) { while (at < off) emit(zeros, std::size_t(std::min<std::uint64_t>(alignment, off - at))); };
        emit(head, 64);
        for (auto& e : table) { pad_to(e.off); emit(e.p, e.size); }
        pad_to(toff);
        emit(tail.data(), tail.size());
        std::fclose(f);
    }
    std::size_t tensor_bytes() const { return data_.size(); }
    const std::vector<tensor_entry>& entries() const { return tensors_; }   // every tensor added so far, in order
    // Pre-size the data buffer so growth never holds two copies of the weights at once.
    void reserve(std::size_t bytes) { data_.reserve(bytes); }
private:
    std::map<std::string, std::string> sections_;
    std::vector<tensor_entry> tensors_;
    std::vector<std::uint8_t> data_;
    std::size_t align_ = alignment;
};

// Streaming writer: tensors go to disk as they are added (the caller frees each one), so peak memory is one
// tensor, not the whole model. Layout: [header][data section][other sections][table]; the header is patched last.
class stream_writer {
public:
    explicit stream_writer(std::string path) : path_(std::move(path)), tmp_(path_ + ".part") {
        f_ = std::fopen(tmp_.c_str(), "wb");
        if (!f_) raise(errc::io_error, "cannot write {}", tmp_);
        std::uint8_t z[64] = {}; emit_(z, 64);
    }
    stream_writer(const stream_writer&) = delete;
    ~stream_writer() { if (f_) { std::fclose(f_); std::remove(tmp_.c_str()); } }
    void set_section(std::string name, std::string bytes) { sections_[std::move(name)] = std::move(bytes); }
    void set_manifest(const json::value& m) { set_section("manifest", json::dump(m)); }
    // Tensor alignment of the file: every tensor's file offset is a multiple of it. Set it before the first tensor.
    void set_alignment(std::size_t a) {
        if (!valid_tensor_alignment(a)) raise(errc::invalid_argument, "tensor alignment {} is not a power of two from 64 to 1 MiB", a);
        if (!tensors_.empty()) raise(errc::invalid_argument, "set_alignment: tensors were already added");
        align_ = a;
    }
    std::size_t tensor_alignment() const { return align_; }
    std::uint64_t data_offset() const { return 64; }   // file offset of the data section
    void add_tensor(std::string name, const shape_t& shape, std::span<const float> data, std::string_view codec = "f32", std::size_t align = alignment) {
        auto* c = graph::codecs::find(codec);
        if (!c) raise(errc::unsupported, "unknown codec '{}'", codec);
        if (std::size_t(numel(shape)) != data.size()) raise(errc::shape_mismatch, "tensor {}: shape/data mismatch", name);
        if (c->block > 1 && data.size() % std::size_t(c->block)) c = graph::codecs::find("f16");
        std::vector<std::uint8_t> buf(c->bytes(data.size()));
        c->encode(data.data(), data.size(), buf.data());
        put_(std::move(name), std::string(c->name), shape, buf, align);
    }
    // `align` (a multiple of 64) applies to the tensor's file offset: expert_alignment for stacked MoE experts.
    void add_raw(std::string name, const shape_t& shape, std::string_view codec, std::span<const std::uint8_t> bytes, std::size_t align = alignment) {
        auto* c = graph::codecs::find(codec);
        if (!c) raise(errc::unsupported, "unknown codec '{}'", codec);
        if (c->bytes(std::size_t(numel(shape))) != bytes.size()) raise(errc::format_error, "tensor {}: {} bytes do not match {} {}", name, bytes.size(), codec, shape_str(shape));
        put_(std::move(name), std::string(c->name), shape, bytes, align);
    }
    std::size_t tensor_bytes() const { return dlen_; }
    // the most recently added tensor (offset relative to the data section, which starts at file offset 64)
    const tensor_entry& last() const { if (tensors_.empty()) raise(errc::invalid_argument, "no tensor written"); return tensors_.back(); }
    const std::vector<tensor_entry>& entries() const { return tensors_; }   // every tensor written so far, in order
    void finish() {
        if (auto it = sections_.find("manifest"); it != sections_.end()) it->second = detail::stamp_alignment(it->second, align_);
        json::array idx;
        for (auto& t : tensors_) {
            json::array sh; for (auto d : t.shape) sh.push_back(double(d));
            idx.push_back(json::object{{"name", t.name}, {"codec", t.codec}, {"shape", sh}, {"offset", double(t.offset)},
                                       {"bytes", double(t.bytes)}, {"blake3", t.blake3}});
        }
        sections_["tensors"] = json::dump(idx);
        struct ent { std::string name; std::uint64_t off, size; hash::blake3::digest h; };
        std::vector<ent> table{{"data", 64, dlen_, dhash_.final()}};
        for (auto& [nm, s] : sections_) {
            pad_();
            table.push_back({nm, at_, s.size(), hash::blake3::of({reinterpret_cast<const std::uint8_t*>(s.data()), s.size()})});
            emit_(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
        }
        pad_();
        std::uint64_t toff = at_;
        for (auto& e : table) {
            std::uint8_t rec[96] = {};
            if (e.name.size() >= 32) raise(errc::invalid_argument, "section name too long: {}", e.name);
            std::memcpy(rec, e.name.data(), e.name.size());
            std::memcpy(rec + 32, &e.off, 8); std::memcpy(rec + 40, &e.size, 8); std::memcpy(rec + 48, e.h.data(), 32);
            emit_(rec, 96);
        }
        std::uint32_t n = std::uint32_t(table.size()); std::uint64_t fsz = at_;
        std::uint8_t head[64] = {};
        std::memcpy(head, magic, 8); std::memcpy(head + 8, &format_version, 4); std::memcpy(head + 12, &n, 4);
        std::memcpy(head + 16, &toff, 8); std::memcpy(head + 24, &fsz, 8);
        if (std::fseek(f_, 0, SEEK_SET) || std::fwrite(head, 1, 64, f_) != 64 || std::fclose(f_)) { f_ = nullptr; raise(errc::io_error, "cannot finish {}", tmp_); }
        f_ = nullptr;
        if (auto rs = forge::os::rename(tmp_, path_); !rs) { forge::os::remove(tmp_); raise(errc::io_error, "cannot replace {}: {}", path_, rs.err.what); }
    }
private:
    void emit_(const std::uint8_t* p, std::size_t k) {
        if (k && std::fwrite(p, 1, k, f_) != k) raise(errc::io_error, "short write {}", tmp_);
        at_ += k;
    }
    void pad_() { static const std::uint8_t z[alignment] = {}; emit_(z, std::size_t((alignment - at_ % alignment) % alignment)); }
    void put_(std::string name, std::string codec, const shape_t& shape, std::span<const std::uint8_t> b, std::size_t align = alignment) {
        static const std::uint8_t z[alignment] = {};
        if (align < alignment || align % alignment) raise(errc::invalid_argument, "tensor alignment {} is not a multiple of {}", align, alignment);
        align = std::max(align, align_);   // both are powers of two (or multiples of 64 below the file's alignment)
        std::size_t padn = (align - (64 + dlen_) % align) % align;   // data starts at file offset 64: align the file offset
        for (std::size_t left = padn; left;) { std::size_t n = std::min(left, alignment); emit_(z, n); dhash_.update(z, n); left -= n; }
        dlen_ += padn;
        tensor_entry e{std::move(name), std::move(codec), shape, dlen_, b.size(), hash::blake3::hex(hash::blake3::of(b))};
        emit_(b.data(), b.size()); dhash_.update(b.data(), b.size()); dlen_ += b.size();
        tensors_.push_back(std::move(e));
    }
    std::string path_, tmp_;
    std::FILE* f_ = nullptr;
    std::size_t align_ = alignment;
    std::uint64_t at_ = 0, dlen_ = 0;
    hash::blake3 dhash_;
    std::map<std::string, std::string> sections_;
    std::vector<tensor_entry> tensors_;
};

namespace detail {
inline std::string hex(const std::uint8_t* p, std::size_t n) { static const char* d = "0123456789abcdef"; std::string s; for (std::size_t i = 0; i < n; ++i) { s += d[p[i] >> 4]; s += d[p[i] & 15]; } return s; }
inline std::vector<std::uint8_t> unhex(std::string_view s) {
    auto v = [](char c) -> int { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
    if (s.size() % 2) raise(errc::invalid_argument, "bad hex");
    std::vector<std::uint8_t> o; for (std::size_t i = 0; i < s.size(); i += 2) { int a = v(s[i]), b = v(s[i + 1]); if (a < 0 || b < 0) raise(errc::invalid_argument, "bad hex"); o.push_back(std::uint8_t(a * 16 + b)); }
    return o;
}
inline std::string enc_key_id(const std::array<std::uint8_t, 32>& k) { auto d = hash::blake3::of({k.data(), k.size()}); return hex(d.data(), 8); }
inline std::string enc_ad(std::string_view tensors_json) { return std::string("forge-fpk-enc-v1\0", 17) + std::string(tensors_json); }
// ---- chunked encryption ("forge-fpk-enc-v2") ----
inline constexpr std::string_view enc2_alg = "xchacha20-poly1305-chunked";
inline constexpr std::string_view enc2_macs = "enc_macs";
inline void put_le64(std::uint8_t* p, std::uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = std::uint8_t(v >> (8 * i)); }
struct enc2_header {
    std::array<std::uint8_t, 24> nonce{};
    std::uint64_t chunk = 0, data_bytes = 0, n_chunks = 0;
    hash::blake3::digest index{};   // BLAKE3 of the "tensors" section
    std::array<std::uint8_t, 24> chunk_nonce(std::uint64_t i) const { auto n = nonce; for (int b = 0; b < 8; ++b) n[std::size_t(16 + b)] ^= std::uint8_t(i >> (8 * b)); return n; }
    std::array<std::uint8_t, 17 + 24 + 32 + 32> ad(std::uint64_t i) const {
        std::array<std::uint8_t, 17 + 24 + 32 + 32> a{};
        std::memcpy(a.data(), "forge-fpk-enc-v2", 17); std::memcpy(a.data() + 17, nonce.data(), 24);
        put_le64(a.data() + 41, i); put_le64(a.data() + 49, n_chunks); put_le64(a.data() + 57, data_bytes); put_le64(a.data() + 65, chunk);
        std::memcpy(a.data() + 73, index.data(), 32);
        return a;
    }
    std::uint64_t chunk_len(std::uint64_t i) const { return std::min<std::uint64_t>(chunk, data_bytes - i * chunk); }
};
// Plaintext of a sealed tensor: 64-byte aligned, wiped before release; in locked, dump-excluded memory on request.
class plain_buffer {
public:
    plain_buffer() = default;
    plain_buffer(std::size_t n, bool lock) : n_(n) {
        if (lock) lb_ = os::locked_buffer(n);
        else if (n) p_.reset(static_cast<std::uint8_t*>(::operator new(n, std::align_val_t(64))));
    }
    plain_buffer(plain_buffer&&) noexcept = default;
    plain_buffer& operator=(plain_buffer&& o) noexcept { if (this != &o) { wipe_(); p_ = std::move(o.p_); lb_ = std::move(o.lb_); n_ = std::exchange(o.n_, 0); } return *this; }
    ~plain_buffer() { wipe_(); }
    std::uint8_t* data() { return lb_.empty() ? p_.get() : lb_.data(); }
    const std::uint8_t* data() const { return lb_.empty() ? p_.get() : lb_.data(); }
    std::size_t size() const { return n_; }
    bool locked() const { return lb_.locked(); }
private:
    void wipe_() { if (p_ && n_) crypto_wipe(p_.get(), n_); }
    struct del { void operator()(std::uint8_t* p) const { ::operator delete(p, std::align_val_t(64)); } };
    std::unique_ptr<std::uint8_t, del> p_; os::locked_buffer lb_; std::size_t n_ = 0;
};
// Where a rewritten container must put its data section so that tensors keep their file alignment: at an offset
// with the same remainder modulo the file's tensor alignment as before. {64, 0} for files that predate the option.
struct data_layout { std::uint64_t align = alignment, phase = 0; };
// Raw container write with sections copied verbatim.
inline void write_sections(const std::string& out, std::vector<std::pair<std::string, std::string>> secs, data_layout dl = {}) {
    std::sort(secs.begin(), secs.end());
    std::vector<std::uint8_t> o(64, 0);
    struct ent { std::string name; std::uint64_t off, size; hash::blake3::digest h; };
    std::vector<ent> table;
    for (auto& [nm, sec] : secs) {
        if (nm.size() >= 32) raise(errc::invalid_argument, "section name too long: {}", nm);
        while (o.size() % alignment) o.push_back(0);
        if (nm == "data" && dl.align > alignment) o.resize(o.size() + std::size_t((dl.phase + dl.align - o.size() % dl.align) % dl.align), 0);
        table.push_back({nm, o.size(), sec.size(), hash::blake3::of({reinterpret_cast<const std::uint8_t*>(sec.data()), sec.size()})});
        o.insert(o.end(), sec.begin(), sec.end());
    }
    while (o.size() % alignment) o.push_back(0);
    std::uint64_t toff = o.size();
    for (auto& e : table) {
        std::uint8_t rec[96] = {};
        std::memcpy(rec, e.name.data(), e.name.size());
        std::memcpy(rec + 32, &e.off, 8); std::memcpy(rec + 40, &e.size, 8); std::memcpy(rec + 48, e.h.data(), 32);
        o.insert(o.end(), rec, rec + 96);
    }
    std::uint32_t n = std::uint32_t(table.size()); std::uint64_t fsz = o.size();
    std::memcpy(o.data(), magic, 8); std::memcpy(o.data() + 8, &format_version, 4); std::memcpy(o.data() + 12, &n, 4);
    std::memcpy(o.data() + 16, &toff, 8); std::memcpy(o.data() + 24, &fsz, 8);
    auto tmp = out + ".tmp";
    { std::ofstream f(tmp, std::ios::binary); f.write(reinterpret_cast<const char*>(o.data()), std::streamsize(o.size())); if (!f) raise(errc::io_error, "cannot write {}", tmp); }
    if (auto rs = forge::os::rename(tmp, out); !rs) { forge::os::remove(tmp); raise(errc::io_error, "cannot replace {}: {}", out, rs.err.what); }
}
}
using model_key = std::array<std::uint8_t, 32>;
// Key from FORGE_MODEL_KEY (64 hex chars) or the file named by FORGE_MODEL_KEY_FILE (hex, whitespace ok).
inline std::optional<model_key> model_key_from_env() {
    std::string h;
    if (auto* e = std::getenv("FORGE_MODEL_KEY")) h = e;
    else if (auto* f = std::getenv("FORGE_MODEL_KEY_FILE")) {
        security::require_private_file("FORGE_MODEL_KEY_FILE", f, security::file_kind::secret);   // FSS-FIO-2
        std::ifstream in(f); std::string w; while (in >> w) h += w;
        if (!w.empty()) crypto_wipe(w.data(), w.size());
    }
    else return std::nullopt;
    auto b = detail::unhex(h); if (!h.empty()) crypto_wipe(h.data(), h.size());
    if (b.size() != 32) raise(errc::invalid_argument, "model key must be 32 bytes (64 hex chars)");
    model_key k; std::memcpy(k.data(), b.data(), 32); crypto_wipe(b.data(), b.size()); return k;
}

// Memory-mapped reader. Tensors are decoded to f32 on request (and cached by the caller).
class reader {
public:
    explicit reader(const std::string& path, std::optional<model_key> key = std::nullopt) : path_(path) {
        // FSS CAP-1: every .fpk opened (run, serve, inspect, dump, verify, sign, encrypt) passes enforce(fs_read)
        // under an active policy; no-op with profile none.
        security::enforce(security::capability::fs_read, path);
        map_ = forge::map_readonly(path, size_);
        blob_dirs_ = blob_search_dirs(path);
        try { init_(path, key); } catch (...) { forge::unmap_readonly(map_, size_); map_ = nullptr; throw; }   // the dtor does not run on a throwing ctor
    }
    // A .fpk that is already in memory (embedded in a compiled model, docs/aot-models.md). The bytes are borrowed and
    // must outlive the reader; `label` stands in for the path in messages.
    reader(std::span<const std::uint8_t> mem, const std::string& label, std::optional<model_key> key = std::nullopt) : path_(label), map_(mem.data()), size_(mem.size()), borrowed_(true) {
        blob_dirs_ = blob_search_dirs({});   // no file to look next to: a sharded image finds its blobs in $FORGE_BLOB_DIR only
        init_(label, key);
    }
private:
    void init_(const std::string& path, std::optional<model_key> key) {
        if (size_ < 64) raise(errc::format_error, "{}: too small to be .fpk", path);
        if (std::memcmp(map_, magic, 8)) raise(errc::format_error, "{}: bad magic", path);
        std::uint32_t ver, n; std::uint64_t toff, fsz;
        std::memcpy(&ver, map_ + 8, 4); std::memcpy(&n, map_ + 12, 4); std::memcpy(&toff, map_ + 16, 8); std::memcpy(&fsz, map_ + 24, 8);
        if (ver != format_version) raise(errc::unsupported, "{}: format version {} not supported", path, ver);
        if (fsz != size_ || toff > size_ || std::uint64_t(n) * 96 > size_ - toff) raise(errc::format_error, "{}: truncated or corrupt", path);
        for (std::uint32_t i = 0; i < n; ++i) {
            const auto* r = map_ + toff + 96 * i;
            section s; s.name.assign(reinterpret_cast<const char*>(r), strnlen(reinterpret_cast<const char*>(r), 32));
            std::memcpy(&s.offset, r + 32, 8); std::memcpy(&s.size, r + 40, 8); std::memcpy(s.hash.data(), r + 48, 32);
            if (s.offset > size_ || s.size > size_ - s.offset) raise(errc::format_error, "{}: section {} out of bounds", path, s.name);
            sections_[s.name] = s;
        }
        if (!has_section("tensors") || !has_section("data")) raise(errc::format_error, "{}: no tensor index or data section", path);
        if (has_section(shards_section)) {
            if (has_section("encryption")) raise(errc::unsupported, "{}: a sharded file cannot be encrypted", path);
            json::value sv = json::parse(section_text(shards_section));
            if (!sv.is_obj() || sv.num_or("version", 0) != 1) raise(errc::unsupported, "{}: shards section version {} is not supported (this build reads version 1)", path, sv.num_or("version", 0));
            auto* arr = sv.find("shards");
            if (!arr || !arr->is_arr() || arr->arr().size() > (std::size_t(1) << 20)) raise(errc::format_error, "{}: bad shards section", path);
            std::set<std::string> seen;
            for (auto& e : arr->arr()) {
                shard_info si; si.blake3 = e.str_or("blake3", ""); double nb = e.num_or("bytes", -1);
                if (!is_blake3_hex(si.blake3) || !(nb >= 0 && nb <= 9e15) || nb != std::floor(nb) || !seen.insert(si.blake3).second) raise(errc::format_error, "{}: bad shards section", path);
                si.bytes = std::uint64_t(nb); shards_.push_back(std::move(si));
            }
            sharded_ = true; shard_map_.assign(shards_.size(), nullptr); shard_file_.assign(shards_.size(), std::string());
        }
        if (has_section("encryption")) { auto k = key ? key : model_key_from_env(); if (k) decrypt(k); else locked_ = true; }
        auto idx = json::parse(section_text("tensors"));
        std::uint64_t dsize = sections_.at("data").size;
        for (auto& t : idx.arr()) {
            tensor_entry e; e.name = t["name"].str(); e.codec = t["codec"].str();
            for (auto& d : t["shape"].arr()) e.shape.push_back(d.i());
            double off = t["offset"].num(), nb = t["bytes"].num();
            if (!(off >= 0 && off <= 9e15 && nb >= 0 && nb <= 9e15)) raise(errc::format_error, "tensor {} out of bounds", e.name);
            e.offset = std::uint64_t(off); e.bytes = std::uint64_t(nb); e.blake3 = t["blake3"].str();
            if (sharded_) {   // every tensor of a sharded file lies in one of its blobs
                double si = t.num_or("shard", -1);
                if (!(si >= 0 && si < double(shards_.size())) || si != std::floor(si)) raise(errc::format_error, "tensor {} names no shard of this file", e.name);
                e.shard = int(si); dsize = shards_[std::size_t(e.shard)].bytes;
                if (!is_blake3_hex(e.blake3)) raise(errc::format_error, "tensor {} has no content hash", e.name);
            } else if (t.find("shard")) raise(errc::format_error, "tensor {} names a shard, but the file has no shards section", e.name);
            if (e.offset > dsize || e.bytes > dsize - e.offset) raise(errc::format_error, "tensor {} out of bounds", e.name);
            // the element count must be what the stored bytes hold: decoders and kernels size their reads from the shape
            std::uint64_t cnt = 1;
            for (auto d : e.shape) { if (d < 0 || (d && cnt > (std::uint64_t(1) << 40) / std::uint64_t(d))) raise(errc::format_error, "tensor {} has a bad shape", e.name); cnt *= std::uint64_t(d); }
            if (auto* c = graph::codecs::find(e.codec); c && c->bytes(std::size_t(cnt)) != e.bytes)
                raise(errc::format_error, "tensor {}: {} bytes do not match {} {}", e.name, e.bytes, e.codec, shape_str(e.shape));
            tensors_[e.name] = e;
        }
    }
public:
    reader(const reader&) = delete;
    reader& operator=(const reader&) = delete;
    ~reader() {
        for (std::size_t i = 0; i < shard_map_.size(); ++i) if (shard_map_[i]) forge::unmap_readonly(shard_map_[i], std::size_t(shards_[i].bytes));
        if (!borrowed_) forge::unmap_readonly(map_, size_);
    }

    bool has_section(const std::string& n) const { return sections_.count(n) != 0; }
    // Plaintext bytes of a section. For "data" prefer tensor_bytes(): on a chunk-encrypted file this call has to
    // decrypt the whole section into memory (kept until the reader closes), which is what the chunked scheme avoids.
    std::span<const std::uint8_t> section_bytes(const std::string& n) const {
        auto it = sections_.find(n);
        if (it == sections_.end()) raise(errc::not_found, "{}: no section '{}'", path_, n);
        if (n == "data" && locked_) raise_locked_();
        if (n == "data" && enc2_) {
            std::lock_guard lk(pmu_);
            if (!plain_) { plain_.emplace(it->second.size); try { decrypt_range_(0, it->second.size, plain_->data()); } catch (...) { plain_.reset(); throw; } }
        }
        if (n == "data" && plain_) return {plain_->data(), plain_->size()};
        return {map_ + it->second.offset, it->second.size};
    }
    // Section bytes exactly as stored (ciphertext for an encrypted data section).
    std::span<const std::uint8_t> raw_section_bytes(const std::string& n) const {
        auto it = sections_.find(n);
        if (it == sections_.end()) raise(errc::not_found, "{}: no section '{}'", path_, n);
        return {map_ + it->second.offset, it->second.size};
    }
    bool locked() const { return locked_; }                                    // encrypted and opened without the key
    bool encrypted() const { return !locked_ && (plain_.has_value() || enc2_.has_value()); }   // encrypted and opened with the key
    bool chunked_encryption() const { return enc2_.has_value(); }              // the on-demand scheme (alg ...-chunked)
    // size of the tensor data, without reading it (a sharded file: the sum of its blobs)
    std::uint64_t data_size() const {
        if (!sharded_) return sections_.at("data").size;
        std::uint64_t n = 0; for (auto& s : shards_) n += s.bytes; return n;
    }
    // ---- sharded files ----
    bool sharded() const { return sharded_; }
    const std::vector<shard_info>& shards() const { return shards_; }
    const std::vector<std::string>& blob_dirs() const { return blob_dirs_; }
    // Directories searched for blobs, replacing the default ($FORGE_BLOB_DIR, then "blobs" next to the file).
    void set_blob_dirs(std::vector<std::string> d) { std::lock_guard lk(pmu_); blob_dirs_ = std::move(d); }
    // The file that holds shard i, or "" when it is in none of the blob directories (not downloaded yet). Only the
    // presence and the size are looked at here; the content is checked when a tensor is read (and by verify()).
    std::string find_shard(std::size_t i) const {
        const auto& s = shards_.at(i);
        for (auto& d : blob_dirs_) {
            std::string p;
            try { p = blob_path(d, s.blake3); } catch (const std::exception&) { continue; }
            std::error_code ec;
            if (std::filesystem::is_regular_file(os::fs::u8path(p), ec) && std::filesystem::file_size(os::fs::u8path(p), ec) == s.bytes && !ec) return p;
        }
        return {};
    }
    std::vector<std::size_t> missing_shards() const { std::vector<std::size_t> m; for (std::size_t i = 0; i < shards_.size(); ++i) if (find_shard(i).empty()) m.push_back(i); return m; }
    // The file's tensor alignment: manifest "tensor_alignment", 64 when it is absent (files written before the option)
    // or not a valid value. Informational: readers find tensors by offset and never rely on it.
    std::size_t tensor_alignment() const {
        try { double a = manifest().num_or("tensor_alignment", double(alignment)); if (a >= 64 && a <= 1048576 && valid_tensor_alignment(std::uint64_t(a)) && double(std::uint64_t(a)) == a) return std::size_t(a); }
        catch (const forge_exception&) {}
        return alignment;
    }
    // Tensors whose file offset is not a multiple of `a` (0 = the file's own alignment): empty for a file as written.
    std::vector<std::string> misaligned_tensors(std::size_t a = 0) const {
        if (!a) a = tensor_alignment();
        std::vector<std::string> bad; const std::uint64_t d = sharded_ ? 0 : sections_.at("data").offset;   // a blob starts at offset 0 of its file
        for (auto& [n, e] : tensors_) if ((d + e.offset) % a) bad.push_back(n);
        return bad;
    }
    // Layout a rewrite of this file must keep (detail::write_sections): the data section's offset modulo the alignment
    // in force (the recorded one, 4096 for files with an experts index, else 64).
    detail::data_layout layout() const {
        std::uint64_t a = tensor_alignment();
        if (a < expert_alignment && has_section("experts")) a = expert_alignment;
        return {a, sections_.at("data").offset % a};
    }
    std::string_view section_text(const std::string& n) const {
        auto b = section_bytes(n); return {reinterpret_cast<const char*>(b.data()), b.size()};
    }
    json::value manifest() const { return json::parse(section_text("manifest")); }
    const std::map<std::string, tensor_entry>& tensors() const { return tensors_; }
    const tensor_entry& entry(const std::string& n) const {
        auto it = tensors_.find(n);
        if (it == tensors_.end()) raise(errc::not_found, "{}: no tensor '{}'", path_, n);
        return it->second;
    }
    // The encoded bytes of one tensor, valid until the reader closes (or release_plaintext()). A plain file returns a
    // view of the mapping. A chunk-encrypted file decrypts the chunks the tensor lies in on first use and keeps that
    // tensor's plaintext (in locked, dump-excluded memory with lock_plaintext(true) or FORGE_LOCK_WEIGHTS=1), so the
    // plaintext resident in memory is the tensors that were asked for, not the file.
    std::span<const std::uint8_t> tensor_bytes(const std::string& n) const {
        const auto& e = entry(n);
        if (sharded_) {
            // lazy, per tensor: the blob is mapped on first use and this tensor's bytes are hashed against the index
            // (which the signature covers) before anyone sees them
            std::lock_guard lk(pmu_);
            auto b = shard_bytes_(std::size_t(e.shard)).subspan(std::size_t(e.offset), std::size_t(e.bytes));
            if (!shard_verified_.count(n)) {
                if (hash::blake3::hex(hash::blake3::of(b)) != e.blake3)
                    raise(errc::integrity_error, "{}: tensor {} in blob {} does not match its hash (corrupt or substituted blob)", path_, n, shards_[std::size_t(e.shard)].blake3.substr(0, 16));
                shard_verified_.insert(n);
            }
            return b;
        }
        if (!enc2_) return section_bytes("data").subspan(std::size_t(e.offset), std::size_t(e.bytes));
        if (locked_) raise_locked_();
        std::lock_guard lk(pmu_);
        if (plain_) return {plain_->data() + e.offset, std::size_t(e.bytes)};   // someone already asked for the whole section
        auto it = ptensors_.find(n);
        if (it == ptensors_.end()) {
            detail::plain_buffer b(std::size_t(e.bytes), lock_plain_);
            decrypt_range_(e.offset, e.bytes, b.data());
            president_ += e.bytes;
            it = ptensors_.emplace(n, std::move(b)).first;
        }
        return {it->second.data(), it->second.size()};
    }
    // Decrypted tensors are allocated in locked memory from now on (os/locked_memory.hpp; best effort, see
    // plaintext_locked()). Default: the FORGE_LOCK_WEIGHTS environment variable ("1" = on).
    void lock_plaintext(bool on) { std::lock_guard lk(pmu_); lock_plain_ = on; }
    // Bytes of decrypted tensor data this reader holds (chunk-encrypted files; 0 for plain files, which are mapped).
    std::uint64_t plaintext_resident() const { std::lock_guard lk(pmu_); return president_ + (enc2_ && plain_ ? plain_->size() : 0); }
    bool plaintext_locked() const { std::lock_guard lk(pmu_); for (auto& [n, b] : ptensors_) if (!b.locked()) return false; return !ptensors_.empty(); }
    // Wipes and frees every decrypted tensor of a chunk-encrypted file. Spans returned by tensor_bytes() before the
    // call dangle: only call it when nothing (no plan, no session) still uses them.
    void release_plaintext() const {
        std::lock_guard lk(pmu_);
        ptensors_.clear(); president_ = 0;
        if (enc2_ && plain_) { crypto_wipe(plain_->data(), plain_->size()); plain_.reset(); }
    }
    tensor load(const std::string& n) const {
        const auto& e = entry(n);
        auto* c = graph::codecs::find(e.codec);
        if (!c) raise(errc::unsupported, "tensor {}: unknown codec {}", n, e.codec);
        tensor t(e.shape);
        if (enc2_ && !cached_(n)) {   // decode from a scratch copy that is wiped: nothing of this tensor stays decrypted here
            if (locked_) raise_locked_();
            detail::plain_buffer b(std::size_t(e.bytes), false);
            { std::lock_guard lk(pmu_); decrypt_range_(e.offset, e.bytes, b.data()); }
            c->decode(b.data(), t.size(), t.data.data());
        } else c->decode(tensor_bytes(n).data(), t.size(), t.data.data());
        return t;
    }
    // Verifies every section hash and every tensor hash. Returns list of problems (empty == ok).
    std::vector<std::string> verify() const {
        std::vector<std::string> bad;
        for (auto& [n, s] : sections_)
            if (hash::blake3::of({map_ + s.offset, s.size}) != s.hash) bad.push_back("section " + n);
        if (locked_) return bad;                       // tensor hashes are over plaintext: need the key
        if (sharded_) {   // every blob must be present and hash to its name; then every tensor must hash
            std::lock_guard lk(pmu_);
            std::vector<char> present(shards_.size(), 0);
            for (std::size_t i = 0; i < shards_.size(); ++i) {
                try { present[i] = 1; if (hash::blake3::hex(hash::blake3::of(shard_bytes_(i))) != shards_[i].blake3) bad.push_back("shard " + shards_[i].blake3.substr(0, 16)); }
                catch (const forge_exception&) { present[i] = 0; bad.push_back("shard " + shards_[i].blake3.substr(0, 16) + " (missing)"); }
            }
            for (auto& [n, e] : tensors_)
                if (present[std::size_t(e.shard)] && hash::blake3::hex(hash::blake3::of(shard_bytes_(std::size_t(e.shard)).subspan(std::size_t(e.offset), std::size_t(e.bytes)))) != e.blake3) bad.push_back("tensor " + n);
            return bad;
        }
        if (enc2_) {   // every chunk must authenticate (also the ones no tensor lies in), then every tensor must hash
            std::lock_guard lk(pmu_);
            detail::plain_buffer scratch(std::size_t(enc2_->chunk), false);
            for (std::uint64_t i = 0; i < enc2_->n_chunks; ++i)
                if (!unlock_chunk_(i, scratch.data())) { bad.push_back(std::format("encrypted chunk {}", i)); return bad; }
            for (auto& [n, e] : tensors_) {
                detail::plain_buffer b(std::size_t(e.bytes), false);
                decrypt_range_(e.offset, e.bytes, b.data());
                if (hash::blake3::hex(hash::blake3::of({b.data(), std::size_t(e.bytes)})) != e.blake3) bad.push_back("tensor " + n);
            }
            return bad;
        }
        auto d = section_bytes("data");
        for (auto& [n, e] : tensors_)
            if (hash::blake3::hex(hash::blake3::of({d.data() + e.offset, e.bytes})) != e.blake3) bad.push_back("tensor " + n);
        return bad;
    }
    std::size_t file_size() const { return size_; }
    struct section { std::string name; std::uint64_t offset = 0, size = 0; hash::blake3::digest hash{}; };
    const std::map<std::string, section>& sections() const { return sections_; }
private:
    [[noreturn]] void raise_locked_() const {
        raise(errc::invalid_argument, "{}: model is encrypted (key id {}); set FORGE_MODEL_KEY or FORGE_MODEL_KEY_FILE",
              path_, json::parse(section_text("encryption")).str_or("key_id", "?"));
    }
    // The bytes of shard i (callers hold pmu_): found in the blob directories, mapped once, size checked.
    std::span<const std::uint8_t> shard_bytes_(std::size_t i) const {
        const auto& s = shards_[i];
        if (!s.bytes) return {};
        if (!shard_map_[i]) {
            auto p = find_shard(i);
            if (p.empty()) {
                std::string dirs; for (auto& d : blob_dirs_) dirs += (dirs.empty() ? "" : ", ") + d;
                raise(errc::not_found, "{}: blob {}{} ({} bytes) is not in {}; fetch it or set FORGE_BLOB_DIR", path_, s.blake3, std::string(blob_ext), s.bytes, dirs.empty() ? "any blob directory (none is set)" : dirs);
            }
            security::enforce(security::capability::fs_read, p);
            std::size_t len = 0; auto* m = forge::map_readonly(p, len);
            if (len != s.bytes) { forge::unmap_readonly(m, len); raise(errc::integrity_error, "{}: blob {} changed size while it was opened", path_, s.blake3.substr(0, 16)); }
            shard_map_[i] = m; shard_file_[i] = p;
        }
        return {shard_map_[i], std::size_t(s.bytes)};
    }
    bool cached_(const std::string& n) const { std::lock_guard lk(pmu_); return plain_.has_value() || ptensors_.count(n) != 0; }
    void decrypt(const std::optional<model_key>& key) {
        json::value e;
        try { e = json::parse(section_text("encryption")); } catch (const forge_exception&) { raise(errc::format_error, "{}: bad encryption header", path_); }
        if (!e.is_obj()) raise(errc::format_error, "{}: bad encryption header", path_);
        const std::string alg = e.str_or("alg", "");
        if (alg != "xchacha20-poly1305" && alg != detail::enc2_alg) raise(errc::unsupported, "{}: unknown encryption {}", path_, alg.empty() ? "?" : alg);
        if (detail::enc_key_id(*key) != e.str_or("key_id", "")) raise(errc::integrity_error, "{}: wrong key (file wants key id {})", path_, e.str_or("key_id", "?"));
        std::vector<std::uint8_t> nonce;
        try { nonce = detail::unhex(e.str_or("nonce", "")); } catch (const forge_exception&) {}
        if (nonce.size() != 24) raise(errc::format_error, "{}: bad encryption header", path_);
        auto& d = sections_.at("data");
        if (alg == detail::enc2_alg) {
            detail::enc2_header h; std::memcpy(h.nonce.data(), nonce.data(), 24);
            double chunk = e.num_or("chunk_bytes", 0), total = e.num_or("data_bytes", -1), nch = e.num_or("chunks", -1);
            if (!(chunk >= 64 && chunk <= double(1u << 30)) || chunk != std::floor(chunk) || !(total >= 0) || !(nch >= 0) || total != std::floor(total) || nch != std::floor(nch) || total > 9e15 || nch > 9e15)
                raise(errc::format_error, "{}: bad encryption header", path_);
            h.chunk = std::uint64_t(chunk); h.data_bytes = std::uint64_t(total); h.n_chunks = std::uint64_t(nch);
            auto mi = sections_.find(std::string(detail::enc2_macs));
            // the header is authenticated through every chunk; these checks catch a cut or padded file before any read
            if (h.data_bytes != d.size || h.n_chunks != (h.data_bytes + h.chunk - 1) / h.chunk || mi == sections_.end() || mi->second.size != h.n_chunks * 16)
                raise(errc::integrity_error, "{}: encrypted data is truncated or its header was altered", path_);
            h.index = hash::blake3::of(raw_section_bytes("tensors"));
            enc2_ = h; key_ = os::locked_buffer(key->data(), 32);
            if (auto* lw = std::getenv("FORGE_LOCK_WEIGHTS")) lock_plain_ = std::string(lw) == "1";
            return;
        }
        std::vector<std::uint8_t> mac;
        try { mac = detail::unhex(e.str_or("mac", "")); } catch (const forge_exception&) {}
        if (mac.size() != 16) raise(errc::format_error, "{}: bad encryption header", path_);
        auto ad = detail::enc_ad(section_text("tensors"));
        plain_.emplace(d.size);
        if (crypto_aead_unlock(plain_->data(), mac.data(), key->data(), nonce.data(), reinterpret_cast<const std::uint8_t*>(ad.data()), ad.size(), map_ + d.offset, d.size))
            { plain_.reset(); raise(errc::integrity_error, "{}: decryption failed (tampered data or index)", path_); }
    }
    // Chunk i of a chunk-encrypted data section into dst (chunk_len(i) bytes). False when it does not authenticate.
    bool unlock_chunk_(std::uint64_t i, std::uint8_t* dst) const {
        const auto& h = *enc2_; auto nonce = h.chunk_nonce(i); auto ad = h.ad(i);
        const auto* ct = map_ + sections_.at("data").offset + i * h.chunk;
        const auto* mac = map_ + sections_.at(std::string(detail::enc2_macs)).offset + i * 16;
        return crypto_aead_unlock(dst, mac, key_.data(), nonce.data(), ad.data(), ad.size(), ct, std::size_t(h.chunk_len(i))) == 0;
    }
    // Plaintext of data[off, off + len) into dst (callers hold pmu_). Whole chunks decrypt straight into dst; a chunk
    // that is only partly wanted goes through a scratch buffer, which is wiped.
    void decrypt_range_(std::uint64_t off, std::uint64_t len, std::uint8_t* dst) const {
        if (!len) return;
        const auto& h = *enc2_;
        detail::plain_buffer scratch;
        for (std::uint64_t i = off / h.chunk, last = (off + len - 1) / h.chunk; i <= last; ++i) {
            std::uint64_t cs = i * h.chunk, cl = h.chunk_len(i);
            bool ok;
            if (cs >= off && cs + cl <= off + len) ok = unlock_chunk_(i, dst + (cs - off));
            else {
                if (!scratch.size()) scratch = detail::plain_buffer(std::size_t(h.chunk), false);
                ok = unlock_chunk_(i, scratch.data());
                std::uint64_t a = std::max(cs, off), b = std::min(cs + cl, off + len);
                if (ok) std::memcpy(dst + (a - off), scratch.data() + (a - cs), std::size_t(b - a));
            }
            if (!ok) { crypto_wipe(dst, std::size_t(len)); raise(errc::integrity_error, "{}: decryption failed in chunk {} (tampered data, tag, header or index)", path_, i); }
        }
    }
    mutable std::optional<std::vector<std::uint8_t>> plain_;   // whole data section in plaintext (single-AEAD scheme; chunked: only after section_bytes("data"))
    std::optional<detail::enc2_header> enc2_;
    os::locked_buffer key_;                                     // chunked scheme: the key stays for on-demand decryption
    mutable std::mutex pmu_;
    mutable std::map<std::string, detail::plain_buffer> ptensors_;
    mutable std::uint64_t president_ = 0;
    bool sharded_ = false;
    std::vector<shard_info> shards_;
    std::vector<std::string> blob_dirs_;
    mutable std::vector<const std::uint8_t*> shard_map_;
    mutable std::vector<std::string> shard_file_;
    mutable std::set<std::string> shard_verified_;
    bool lock_plain_ = false;
    bool locked_ = false;
    std::string path_;
    const std::uint8_t* map_ = nullptr;
    std::size_t size_ = 0;
    std::map<std::string, section> sections_;
    std::map<std::string, tensor_entry> tensors_;
    bool borrowed_ = false;   // map_ is the caller's memory, not a mapping
};

struct encrypt_options {
    std::size_t chunk_bytes = std::size_t(1) << 20;   // plaintext bytes per sealed chunk (64 .. 2^30)
    bool single_aead = false;                        // write the old one-AEAD scheme (for readers that predate the chunked one)
};

// Seal the data section of `in` with `key`, writing `out`. Existing signatures are dropped (re-sign after).
inline void encrypt_file(const std::string& in, const std::string& out, const model_key& key, const encrypt_options& opt = {}) {
    std::vector<std::pair<std::string, std::string>> secs; std::string tj; std::vector<std::uint8_t> plain; detail::data_layout dl;
    {
        reader r(in);
        if (r.has_section("encryption")) raise(errc::invalid_argument, "{}: already encrypted", in);
        if (r.sharded()) raise(errc::unsupported, "{}: a sharded file cannot be encrypted (its tensors are plain content-addressed blobs); join it first", in);
        if (auto bad = r.verify(); !bad.empty()) raise(errc::format_error, "{}: refusing to encrypt a corrupt file ({})", in, bad.front());
        dl = r.layout();
        for (auto& [name, s] : r.sections()) {
            if (name == "signature") continue;
            auto b = r.section_bytes(name);
            if (name == "data") plain.assign(b.begin(), b.end()); else secs.emplace_back(name, std::string(reinterpret_cast<const char*>(b.data()), b.size()));
            if (name == "tensors") tj.assign(reinterpret_cast<const char*>(b.data()), b.size());
        }
    }
    std::array<std::uint8_t, 24> nonce{};
    if (forge::modelfile::os_entropy(nonce.data(), nonce.size()) != 0) raise(errc::io_error, "getentropy failed");
    std::string ct(plain.size(), '\0');
    if (opt.single_aead) {
        std::array<std::uint8_t, 16> mac{};
        auto ad = detail::enc_ad(tj);
        crypto_aead_lock(reinterpret_cast<std::uint8_t*>(ct.data()), mac.data(), key.data(), nonce.data(), reinterpret_cast<const std::uint8_t*>(ad.data()), ad.size(), plain.data(), plain.size());
        secs.emplace_back("encryption", json::dump(json::object{{"alg", "xchacha20-poly1305"}, {"nonce", detail::hex(nonce.data(), 24)}, {"mac", detail::hex(mac.data(), 16)}, {"key_id", detail::enc_key_id(key)}}));
    } else {
        if (opt.chunk_bytes < 64 || opt.chunk_bytes > (std::size_t(1) << 30)) raise(errc::invalid_argument, "encrypt: chunk size {} is outside 64 .. 2^30", opt.chunk_bytes);
        detail::enc2_header h; h.nonce = nonce; h.chunk = opt.chunk_bytes; h.data_bytes = plain.size(); h.n_chunks = (h.data_bytes + h.chunk - 1) / h.chunk;
        h.index = hash::blake3::of({reinterpret_cast<const std::uint8_t*>(tj.data()), tj.size()});
        std::string macs(std::size_t(h.n_chunks) * 16, '\0');
        for (std::uint64_t i = 0; i < h.n_chunks; ++i) {
            auto cn = h.chunk_nonce(i); auto ad = h.ad(i);
            crypto_aead_lock(reinterpret_cast<std::uint8_t*>(ct.data()) + i * h.chunk, reinterpret_cast<std::uint8_t*>(macs.data()) + i * 16, key.data(), cn.data(),
                             ad.data(), ad.size(), plain.data() + i * h.chunk, std::size_t(h.chunk_len(i)));
        }
        secs.emplace_back(std::string(detail::enc2_macs), std::move(macs));
        secs.emplace_back("encryption", json::dump(json::object{{"alg", std::string(detail::enc2_alg)}, {"scheme", "forge-fpk-enc-v2"}, {"nonce", detail::hex(nonce.data(), 24)},
                                                               {"chunk_bytes", double(h.chunk)}, {"chunks", double(h.n_chunks)}, {"data_bytes", double(h.data_bytes)},
                                                               {"key_id", detail::enc_key_id(key)}}));
    }
    if (!plain.empty()) crypto_wipe(plain.data(), plain.size());
    secs.emplace_back("data", std::move(ct));
    detail::write_sections(out, std::move(secs), dl);
}
inline model_key generate_model_key() { model_key k; if (forge::modelfile::os_entropy(k.data(), k.size()) != 0) raise(errc::io_error, "getentropy failed"); return k; }

} // namespace forge::modelfile

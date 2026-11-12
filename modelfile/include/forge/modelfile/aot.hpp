#pragma once
// Precompiled plans in a model file (docs/aot-models.md): the "aot" section of a .fpk, or a sidecar file next to it.
//
// The section holds a graph::aot blob (graph/aot.hpp). It is an ordinary section: the table carries its BLAKE3 and a
// signature made after it was added covers it like every other section. Adding or changing it on a signed file
// changes the signed digest, so the signature no longer verifies; strict loading then refuses the whole file.
#include <forge/graph/aot.hpp>
#include <forge/modelfile/sign.hpp>
#include <cstdio>

namespace forge::modelfile::aot {

inline constexpr char section[] = "aot";
inline constexpr char sidecar_ext[] = ".fplan";   // <model path> + ".fplan": the blob alone, for files that must not change

// Key of the model in `r` for precompiled plans (graph::aot::model_key over its blueprint and tensor index).
inline graph::aot::digest model_key(const reader& r) {
    return graph::aot::model_key(r.section_text("blueprint"), r.section_text("tensors"));
}

// Whether a signed file's signature covers its "aot" section as it is on disk. The digest is rebuilt from the stored
// table hashes, except for the "aot" section, whose hash is recomputed from its bytes: it equals the signed digest
// only if those bytes are the ones that were signed, so the weights do not have to be hashed again. At least one
// signature over that digest must verify (strict loading has checked the whole file and the trusted keys before).
inline bool covered_by_signature(const reader& r, std::string* why = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (!r.has_section("signature")) return fail("unsigned");
    if (!r.has_section(section)) return fail("no aot section");
    json::value s;
    try { s = json::parse(r.section_text("signature")); } catch (const std::exception&) { return fail("bad signature section"); }
    if (s.str_or("alg", "") != "ed25519" || s.str_or("scheme", "") != sign_scheme) return fail("unsupported signature scheme");
    hash::blake3 h;
    auto put = [&](const void* p, std::size_t n) { h.update(static_cast<const std::uint8_t*>(p), n); };
    put(sign_scheme.data(), sign_scheme.size());
    for (auto& [name, sec] : r.sections()) {
        if (name == "signature") continue;
        auto d = name == section ? hash::blake3::of(r.raw_section_bytes(name)) : sec.hash;
        std::uint64_t ln = name.size(), sz = sec.size;
        put(&ln, 8); put(name.data(), name.size()); put(&sz, 8); put(d.data(), d.size());
    }
    auto d = h.final();
    if (s.str_or("digest", "") != detail::to_hex(d)) return fail("the signature does not cover the aot section (added or changed after signing)");
    auto* arr = s.find("signatures");
    if (!arr || !arr->is_arr()) return fail("no signatures");
    for (auto& e : arr->arr()) {
        std::vector<std::uint8_t> pub, sig;
        try { pub = detail::from_hex(e["public_key"].str()); sig = detail::from_hex(e["sig"].str()); } catch (...) { continue; }
        if (pub.size() == 32 && sig.size() == 64 && crypto_ed25519_check(sig.data(), pub.data(), d.data(), d.size()) == 0) return true;
    }
    return fail("no valid signature");
}

// Rewrites `in` to `out` (may be the same path) with the "aot" section set to `blob`, or removed when `blob` is
// empty. Other sections are copied as stored, straight from the mapping (the weights are never held in memory twice).
// A signature section is dropped: it would no longer match; sign again afterwards. Returns whether one was dropped.
inline bool write_section(const std::string& in, const std::string& out, std::span<const std::uint8_t> blob) {
    bool dropped = false;
    auto tmp = out + ".aot.part";
    {
        reader r(in);
        // Existing sections keep their file offsets (the experts index records offsets into the data section, and
        // importers align tensors to the file); the new section goes after them. A section that is dropped from the
        // middle of the file leaves a zero-filled gap.
        struct ent { std::string name; std::span<const std::uint8_t> bytes; hash::blake3::digest h; std::uint64_t off = 0; };
        std::vector<ent> secs;
        for (auto& [name, s] : r.sections()) {
            if (name == "signature") { dropped = true; continue; }
            if (name == section) continue;
            secs.push_back({name, r.raw_section_bytes(name), s.hash, s.offset});
        }
        std::sort(secs.begin(), secs.end(), [](const ent& a, const ent& b) { return a.off < b.off; });
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) raise(errc::io_error, "cannot write {}", tmp);
        std::uint64_t at = 0; bool ok = true;
        static const std::uint8_t zeros[alignment] = {};
        auto emit = [&](const std::uint8_t* p, std::size_t k) { if (k && std::fwrite(p, 1, k, f) != k) ok = false; at += k; };
        auto pad_to = [&](std::uint64_t off) { while (at < off) emit(zeros, std::size_t(std::min<std::uint64_t>(alignment, off - at))); };
        auto pad = [&] { pad_to((at + alignment - 1) / alignment * alignment); };
        std::uint8_t head[64] = {};
        emit(head, 64);
        for (auto& e : secs) {
            if (e.off < at) { std::fclose(f); forge::os::remove(tmp); raise(errc::format_error, "{}: overlapping sections", in); }
            pad_to(e.off); emit(e.bytes.data(), e.bytes.size());
        }
        if (!blob.empty()) { pad(); secs.push_back({section, blob, hash::blake3::of(blob), at}); emit(blob.data(), blob.size()); }
        pad();
        std::uint64_t toff = at;
        for (auto& e : secs) {
            std::uint8_t rec[96] = {};
            std::memcpy(rec, e.name.data(), e.name.size());
            std::uint64_t sz = e.bytes.size();
            std::memcpy(rec + 32, &e.off, 8); std::memcpy(rec + 40, &sz, 8); std::memcpy(rec + 48, e.h.data(), 32);
            emit(rec, 96);
        }
        std::uint32_t n = std::uint32_t(secs.size()); std::uint64_t fsz = at;
        std::memcpy(head, magic, 8); std::memcpy(head + 8, &format_version, 4); std::memcpy(head + 12, &n, 4);
        std::memcpy(head + 16, &toff, 8); std::memcpy(head + 24, &fsz, 8);
        ok = ok && std::fseek(f, 0, SEEK_SET) == 0 && std::fwrite(head, 1, 64, f) == 64;
        if (std::fclose(f) != 0 || !ok) { forge::os::remove(tmp); raise(errc::io_error, "cannot write {}", tmp); }
    }   // the mapping of `in` is released before it may be replaced
    if (auto rs = forge::os::rename(tmp, out); !rs) { forge::os::remove(tmp); raise(errc::io_error, "cannot replace {}: {}", out, rs.err.what); }
    return dropped;
}

inline void write_sidecar(const std::string& path, std::span<const std::uint8_t> blob) {
    auto tmp = path + ".part";
    { std::ofstream f(tmp, std::ios::binary | std::ios::trunc); f.write(reinterpret_cast<const char*>(blob.data()), std::streamsize(blob.size())); if (!f) raise(errc::io_error, "cannot write {}", tmp); }
    if (auto rs = forge::os::rename(tmp, path); !rs) { forge::os::remove(tmp); raise(errc::io_error, "cannot replace {}: {}", path, rs.err.what); }
}

} // namespace forge::modelfile::aot

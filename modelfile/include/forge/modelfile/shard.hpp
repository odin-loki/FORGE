#pragma once
// Sharded model files and the content-addressed blob store (FORGE_PLAN D.3 "Sharding: one manifest, many blob
// files; partial downloads" and "Content-addressed tensors ... deduplicated across files").
//
// A sharded model is a small .fpk (manifest, blueprint, tokenizer, tensor index, every other section) whose tensor
// bytes live in blob files named by their BLAKE3 hash:
//
//     model.fpk                                   "shards": [{blake3, bytes}], tensors[i] = {..., shard, offset}
//     blobs/ab/ab12...ef.fpkb                     raw bytes; BLAKE3 of the file is its name
//
//   per tensor (default)   one blob per tensor, named by the tensor's own hash. Two models written to the same blob
//                          directory share every tensor they have in common: a fine-tune that changes 3 of 300
//                          tensors adds 3 blobs.
//   packed (shard_bytes)   tensors in file order are packed into blobs of at most that size (a tensor larger than it
//                          gets a blob to itself), each tensor on the file's tensor alignment. Fewer, larger files
//                          for transfer; two models share a blob only when a whole pack is identical.
//
// The index is the root of trust: the main file's signature covers the "tensors" and "shards" sections, i.e. every
// blob hash and every tensor hash, so blobs carry no signature of their own and can come from anywhere. A reader maps
// a blob when a tensor in it is first used and hashes that tensor against the index before handing it out
// (reader::tensor_bytes); reader::verify() also hashes every blob whole. A blob that is absent only matters when one
// of its tensors is asked for.
//
// Blob names come from model files, which are untrusted: a name is used only if it is 64 lowercase hex digits, and
// the path is built with security::safe_join (modelfile::blob_path).
#include <forge/modelfile/fpk.hpp>
#include <atomic>
#include <tuple>

namespace forge::modelfile {

struct shard_options {
    std::uint64_t shard_bytes = 0;   // 0: one blob per tensor; else pack tensors into blobs of at most this many bytes
    std::string blob_dir;            // where blobs are written; "" = "blobs" next to the output file
};
struct shard_report {
    std::size_t tensors = 0, shards = 0, written = 0, reused = 0;   // reused: blobs that were already in the store
    std::uint64_t bytes = 0, bytes_written = 0, bytes_reused = 0;
    std::string blob_dir;
    std::vector<std::string> notes;
};

namespace detail {
// Writes `parts` (spans with zero padding between them, as laid out by `at`) to the blob store under their hash.
// Returns {hash, bytes, reused}. The file is written under a temporary name and renamed, so a reader never sees a
// partial blob; an existing blob of the same name and size is left alone.
struct blob_part { std::span<const std::uint8_t> bytes; std::uint64_t at = 0; };
inline std::tuple<std::string, std::uint64_t, bool> store_blob(const std::string& dir, const std::vector<blob_part>& parts, const std::string* known_hash = nullptr) {
    namespace sfs = std::filesystem;
    std::uint64_t total = 0; for (auto& p : parts) total = std::max<std::uint64_t>(total, p.at + p.bytes.size());
    auto exists = [&](const std::string& hash) { std::error_code ec; auto p = os::fs::u8path(blob_path(dir, hash)); return sfs::is_regular_file(p, ec) && sfs::file_size(p, ec) == total && !ec; };
    if (known_hash && exists(*known_hash)) return {*known_hash, total, true};   // per-tensor blobs: the hash is the tensor's, no need to write to find out
    std::error_code ec; sfs::create_directories(os::fs::u8path(dir), ec);
    if (ec) raise(errc::io_error, "cannot create blob directory {}: {}", dir, ec.message());
    security::enforce(security::capability::fs_write, dir);
    static std::atomic<unsigned> seq{0};
    auto tmp = (os::fs::u8path(dir) / (".part-" + std::to_string(os::pid()) + "-" + std::to_string(seq++))).string();
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) raise(errc::io_error, "cannot write {}", tmp);
    hash::blake3 h; std::uint64_t at = 0; bool ok = true;
    static const std::uint8_t zeros[4096] = {};
    auto emit = [&](const std::uint8_t* p, std::size_t n) { if (n && std::fwrite(p, 1, n, f) != n) ok = false; h.update(p, n); at += n; };
    for (auto& p : parts) {
        while (at < p.at) emit(zeros, std::size_t(std::min<std::uint64_t>(sizeof zeros, p.at - at)));
        emit(p.bytes.data(), p.bytes.size());
    }
    if (std::fclose(f) != 0 || !ok) { os::remove(tmp); raise(errc::io_error, "cannot write {}", tmp); }
    auto hash = hash::blake3::hex(h.final());
    if (exists(hash)) { os::remove(tmp); return {hash, total, true}; }
    auto dst = blob_path(dir, hash);
    sfs::create_directories(os::fs::u8path(dst).parent_path(), ec);
    if (auto rs = os::rename(tmp, dst); !rs) { os::remove(tmp); raise(errc::io_error, "cannot store blob {}: {}", dst, rs.err.what); }
    return {hash, total, false};
}
inline std::string default_blob_dir(const std::string& out) {
    if (auto* e = std::getenv("FORGE_BLOB_DIR"); e && *e) return e;
    auto parent = os::fs::u8path(out).parent_path();
    return os::fs::u8str((parent.empty() ? std::filesystem::path(".") : parent) / "blobs");
}
} // namespace detail

// Splits the single-file model `in` into the sharded model `out` plus blobs in the blob directory.
// Dropped: a signature and precompiled plans (the tensor index changes: sign / run forge aot on the result), and an
// "experts" index (it records offsets into the data section, which no longer exists). Refused: encrypted input (its
// tensors are ciphertext: nothing to address by content), input that is already sharded, out == in.
inline shard_report shard_split(const std::string& in, const std::string& out, const shard_options& opt = {}) {
    shard_report rep;
    rep.blob_dir = opt.blob_dir.empty() ? detail::default_blob_dir(out) : opt.blob_dir;
    { std::error_code ec; if (std::filesystem::exists(os::fs::u8path(out), ec) && std::filesystem::equivalent(os::fs::u8path(in), os::fs::u8path(out), ec)) raise(errc::invalid_argument, "shard split: the output must be another file than the input"); }
    std::vector<std::pair<std::string, std::string>> secs; json::array idx, shards;
    {
        reader r(in);
        if (r.has_section("encryption")) raise(errc::unsupported, "{}: an encrypted file cannot be sharded (its tensors are ciphertext)", in);
        if (r.sharded()) raise(errc::invalid_argument, "{}: already sharded", in);
        if (auto bad = r.verify(); !bad.empty()) raise(errc::format_error, "{}: refusing to shard a corrupt file ({})", in, bad.front());
        const std::uint64_t align = r.tensor_alignment();
        std::vector<const tensor_entry*> order;
        for (auto& [n, e] : r.tensors()) order.push_back(&e);
        std::sort(order.begin(), order.end(), [](auto* a, auto* b) { return a->offset < b->offset; });
        std::map<std::string, std::pair<int, std::uint64_t>> where;   // tensor -> (shard index, offset)
        std::map<std::string, int> shard_of_hash;
        auto add_shard = [&](const std::string& hash, std::uint64_t bytes, bool reused) {
            auto [it, fresh] = shard_of_hash.emplace(hash, int(shards.size()));
            if (fresh) {
                shards.push_back(json::object{{"blake3", hash}, {"bytes", double(bytes)}});
                ++rep.shards; rep.bytes += bytes;
                if (reused) { ++rep.reused; rep.bytes_reused += bytes; } else { ++rep.written; rep.bytes_written += bytes; }
            }
            return it->second;
        };
        if (!opt.shard_bytes) {
            for (auto* e : order) {
                auto [hash, bytes, reused] = detail::store_blob(rep.blob_dir, {{r.tensor_bytes(e->name), 0}}, &e->blake3);
                if (hash != e->blake3) raise(errc::integrity_error, "{}: tensor {} does not match its hash", in, e->name);
                where[e->name] = {add_shard(hash, bytes, reused), 0};
            }
        } else {
            std::vector<detail::blob_part> parts; std::vector<std::pair<const tensor_entry*, std::uint64_t>> members; std::uint64_t at = 0;
            auto flush = [&] {
                if (members.empty()) return;
                auto [hash, bytes, reused] = detail::store_blob(rep.blob_dir, parts);
                int si = add_shard(hash, bytes, reused);
                for (auto& [e, off] : members) where[e->name] = {si, off};
                parts.clear(); members.clear(); at = 0;
            };
            for (auto* e : order) {
                std::uint64_t off = (at + align - 1) / align * align;
                if (!members.empty() && off + e->bytes > opt.shard_bytes) { flush(); off = 0; }
                parts.push_back({r.tensor_bytes(e->name), off}); members.emplace_back(e, off); at = off + e->bytes;
            }
            flush();
        }
        for (auto& t : json::parse(r.section_text("tensors")).arr()) {
            json::object o;
            const auto& w = where.at(t["name"].str());
            for (auto& [k, v] : t.obj()) if (k != "offset" && k != "shard") o.emplace_back(k, v);
            o.emplace_back("offset", double(w.second)); o.emplace_back("shard", double(w.first));
            idx.push_back(std::move(o)); ++rep.tensors;
        }
        for (auto& [name, s] : r.sections()) {
            if (name == "signature" || name == "aot" || name == "experts" || name == "tensors" || name == "data") continue;
            auto b = r.raw_section_bytes(name);
            std::string bytes(reinterpret_cast<const char*>(b.data()), b.size());
            if (name == "manifest") {
                try {
                    auto m = json::parse(bytes);
                    if (m.is_obj()) {
                        auto& o = m.obj_mut();
                        auto set = [&](const char* k, json::value v) { for (auto& [kk, vv] : o) if (kk == k) { vv = std::move(v); return; } o.emplace_back(k, std::move(v)); };
                        set("sharded", true);
                        if (m.num_or("forge_format_minor", 0) < double(format_minor_sharded)) set("forge_format_minor", double(format_minor_sharded));
                        bytes = json::dump(m);
                    }
                } catch (const forge_exception&) {}
            }
            secs.emplace_back(name, std::move(bytes));
        }
        if (r.has_section("signature")) rep.notes.push_back("the signature was dropped (the tensor index changed): sign the sharded file");
        if (r.has_section("aot")) rep.notes.push_back("precompiled plans were dropped (they are keyed on the tensor index): run forge aot on the sharded file");
        if (r.has_section("experts")) rep.notes.push_back("the experts index was dropped (it records offsets into the data section)");
    }
    secs.emplace_back("data", std::string());
    secs.emplace_back("tensors", json::dump(json::value(idx)));
    secs.emplace_back(shards_section, json::dump(json::value(json::object{{"version", 1.0}, {"layout", opt.shard_bytes ? "packed" : "per-tensor"}, {"shards", shards}})));
    detail::write_sections(out, std::move(secs));
    return rep;
}

// Joins the sharded model `in` (all of its blobs must be present; each tensor is checked against the index as it is
// copied) into the single file `out`. The tensor alignment is kept. A signature and precompiled plans are dropped.
inline shard_report shard_join(const std::string& in, const std::string& out) {
    shard_report rep;
    { std::error_code ec; if (std::filesystem::exists(os::fs::u8path(out), ec) && std::filesystem::equivalent(os::fs::u8path(in), os::fs::u8path(out), ec)) raise(errc::invalid_argument, "shard join: the output must be another file than the input"); }
    reader r(in);
    if (!r.sharded()) raise(errc::invalid_argument, "{}: not a sharded file", in);
    if (auto miss = r.missing_shards(); !miss.empty())
        raise(errc::not_found, "{}: {} of {} blobs are missing (first: {}{}); fetch them or set FORGE_BLOB_DIR", in, miss.size(), r.shards().size(), r.shards()[miss.front()].blake3, std::string(blob_ext));
    stream_writer w(out);
    w.set_alignment(r.tensor_alignment());
    std::vector<const tensor_entry*> order;
    for (auto& [n, e] : r.tensors()) order.push_back(&e);
    std::sort(order.begin(), order.end(), [](auto* a, auto* b) { return std::pair(a->shard, a->offset) < std::pair(b->shard, b->offset); });
    for (auto* e : order) {
        auto b = r.tensor_bytes(e->name);   // hashed against the index here
        if (auto* c = graph::codecs::find(e->codec)) w.add_raw(e->name, e->shape, c->name, b);
        else raise(errc::unsupported, "tensor {}: unknown codec {}", e->name, e->codec);
        ++rep.tensors; rep.bytes += e->bytes;
    }
    rep.shards = r.shards().size();
    for (auto& [name, s] : r.sections()) {
        if (name == "signature" || name == "aot" || name == "tensors" || name == "data" || name == shards_section) continue;
        std::string bytes(r.section_text(name));
        if (name == "manifest") {
            try { auto m = json::parse(bytes); if (m.is_obj()) { std::erase_if(m.obj_mut(), [](auto& kv) { return kv.first == "sharded"; }); bytes = json::dump(m); } }
            catch (const forge_exception&) {}
        }
        w.set_section(name, std::move(bytes));
    }
    if (r.has_section("signature")) rep.notes.push_back("the signature was dropped (the tensor index changed): sign the joined file");
    if (r.has_section("aot")) rep.notes.push_back("precompiled plans were dropped: run forge aot on the joined file");
    w.finish();
    return rep;
}

} // namespace forge::modelfile

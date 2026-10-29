#pragma once
// .fpk container: one file holding manifest, blueprint, tokenizer, tensor index and tensor data.
//
//   [header 64B] [sections ... each aligned to 64B] [section table]
//   header : "FORGEPK1" | u32 version | u32 n_sections | u64 table_offset | u64 file_size | 32B reserved
//   table  : n x { char name[32]; u64 offset; u64 size; u8 blake3[32]; u8 pad[16] }  (96 B each)
//   "tensors" section is JSON: [{name, codec, shape, offset, bytes, blake3}], offsets relative to "data".
//   Every section and every tensor carries a BLAKE3 hash; readers verify on demand (`verify()`).
#include <forge/core/error.hpp>
#include <forge/core/tensor.hpp>
#include <forge/flow/codecs.hpp>
#include <forge/hash/blake3.hpp>
#include <forge/json/json.hpp>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace forge::pack {

inline constexpr char magic[8] = {'F', 'O', 'R', 'G', 'E', 'P', 'K', '1'};
inline constexpr std::uint32_t format_version = 1;
inline constexpr std::size_t alignment = 64;

struct tensor_entry {
    std::string name, codec;
    shape_t shape;
    std::uint64_t offset = 0, bytes = 0;
    std::string blake3;
};

class writer {
public:
    void set_section(std::string name, std::string bytes) { sections_[std::move(name)] = std::move(bytes); }
    void set_manifest(const json::value& m) { set_section("manifest", json::dump(m)); }
    // Encodes f32 data with the named codec (f32,f16,bf16,fp8_e4m3,q8_0,q4_k).
    void add_tensor(std::string name, const shape_t& shape, std::span<const float> data, std::string_view codec = "f32") {
        auto* c = flow::codecs::find(codec);
        if (!c) raise(errc::unsupported, "unknown codec '{}'", codec);
        if (std::size_t(numel(shape)) != data.size()) raise(errc::shape_mismatch, "tensor {}: shape/data mismatch", name);
        if (c->block > 1 && data.size() % std::size_t(c->block)) c = flow::codecs::find("f16");  // fallback for odd sizes
        while (data_.size() % alignment) data_.push_back(0);
        tensor_entry e{std::move(name), std::string(c->name), shape, data_.size(), c->bytes(data.size()), {}};
        data_.resize(data_.size() + e.bytes);
        c->encode(data.data(), data.size(), data_.data() + e.offset);
        e.blake3 = hash::blake3::hex(hash::blake3::of({data_.data() + e.offset, e.bytes}));
        tensors_.push_back(std::move(e));
    }
    // Stores already-encoded blocks (e.g. Q8_0 / Q4_K straight from a GGUF file) without re-quantising.
    void add_raw(std::string name, const shape_t& shape, std::string_view codec, std::span<const std::uint8_t> bytes) {
        auto* c = flow::codecs::find(codec);
        if (!c) raise(errc::unsupported, "unknown codec '{}'", codec);
        if (c->bytes(std::size_t(numel(shape))) != bytes.size()) raise(errc::format_error, "tensor {}: {} bytes do not match {} {}", name, bytes.size(), codec, shape_str(shape));
        while (data_.size() % alignment) data_.push_back(0);
        tensor_entry e{std::move(name), std::string(c->name), shape, data_.size(), bytes.size(), {}};
        data_.insert(data_.end(), bytes.begin(), bytes.end());
        e.blake3 = hash::blake3::hex(hash::blake3::of(bytes));
        tensors_.push_back(std::move(e));
    }
    void write(const std::string& path) {
        json::array idx;
        for (auto& t : tensors_) {
            json::array sh; for (auto d : t.shape) sh.push_back(double(d));
            idx.push_back(json::object{{"name", t.name}, {"codec", t.codec}, {"shape", sh}, {"offset", double(t.offset)},
                                       {"bytes", double(t.bytes)}, {"blake3", t.blake3}});
        }
        sections_["tensors"] = json::dump(idx);
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) raise(errc::io_error, "cannot write {}", path);
        std::vector<std::uint8_t> out(64, 0);
        struct ent { std::string name; std::uint64_t off, size; hash::blake3::digest h; };
        std::vector<ent> table;
        auto put = [&](const std::string& nm, const std::uint8_t* p, std::size_t n) {
            while (out.size() % alignment) out.push_back(0);
            table.push_back({nm, out.size(), n, hash::blake3::of({p, n})});
            out.insert(out.end(), p, p + n);
        };
        for (auto& [nm, s] : sections_) put(nm, reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
        put("data", data_.data(), data_.size());
        while (out.size() % alignment) out.push_back(0);
        std::uint64_t toff = out.size();
        for (auto& e : table) {
            std::uint8_t rec[96] = {};
            if (e.name.size() >= 32) raise(errc::invalid_argument, "section name too long: {}", e.name);
            std::memcpy(rec, e.name.data(), e.name.size());
            std::memcpy(rec + 32, &e.off, 8); std::memcpy(rec + 40, &e.size, 8); std::memcpy(rec + 48, e.h.data(), 32);
            out.insert(out.end(), rec, rec + 96);
        }
        std::uint32_t n = std::uint32_t(table.size());
        std::uint64_t fsz = out.size();
        std::memcpy(out.data(), magic, 8); std::memcpy(out.data() + 8, &format_version, 4); std::memcpy(out.data() + 12, &n, 4);
        std::memcpy(out.data() + 16, &toff, 8); std::memcpy(out.data() + 24, &fsz, 8);
        if (std::fwrite(out.data(), 1, out.size(), f) != out.size()) { std::fclose(f); raise(errc::io_error, "short write {}", path); }
        std::fclose(f);
    }
    std::size_t tensor_bytes() const { return data_.size(); }
private:
    std::map<std::string, std::string> sections_;
    std::vector<tensor_entry> tensors_;
    std::vector<std::uint8_t> data_;
};

// Memory-mapped reader. Tensors are decoded to f32 on request (and cached by the caller).
class reader {
public:
    explicit reader(const std::string& path) : path_(path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) raise(errc::io_error, "cannot open {}", path);
        struct stat st{}; ::fstat(fd, &st); size_ = std::size_t(st.st_size);
        if (size_ < 64) { ::close(fd); raise(errc::format_error, "{}: too small to be .fpk", path); }
        map_ = static_cast<const std::uint8_t*>(::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0));
        ::close(fd);
        if (map_ == MAP_FAILED) raise(errc::io_error, "mmap failed for {}", path);
        if (std::memcmp(map_, magic, 8)) raise(errc::format_error, "{}: bad magic", path);
        std::uint32_t ver, n; std::uint64_t toff, fsz;
        std::memcpy(&ver, map_ + 8, 4); std::memcpy(&n, map_ + 12, 4); std::memcpy(&toff, map_ + 16, 8); std::memcpy(&fsz, map_ + 24, 8);
        if (ver != format_version) raise(errc::unsupported, "{}: format version {} not supported", path, ver);
        if (fsz != size_ || toff + std::uint64_t(n) * 96 > size_) raise(errc::format_error, "{}: truncated or corrupt", path);
        for (std::uint32_t i = 0; i < n; ++i) {
            const auto* r = map_ + toff + 96 * i;
            section s; s.name.assign(reinterpret_cast<const char*>(r), strnlen(reinterpret_cast<const char*>(r), 32));
            std::memcpy(&s.offset, r + 32, 8); std::memcpy(&s.size, r + 40, 8); std::memcpy(s.hash.data(), r + 48, 32);
            if (s.offset + s.size > size_) raise(errc::format_error, "{}: section {} out of bounds", path, s.name);
            sections_[s.name] = s;
        }
        auto idx = json::parse(section_text("tensors"));
        for (auto& t : idx.arr()) {
            tensor_entry e; e.name = t["name"].str(); e.codec = t["codec"].str();
            for (auto& d : t["shape"].arr()) e.shape.push_back(d.i());
            e.offset = std::uint64_t(t["offset"].num()); e.bytes = std::uint64_t(t["bytes"].num()); e.blake3 = t["blake3"].str();
            if (e.offset + e.bytes > sections_.at("data").size) raise(errc::format_error, "tensor {} out of bounds", e.name);
            tensors_[e.name] = e;
        }
    }
    reader(const reader&) = delete;
    reader& operator=(const reader&) = delete;
    ~reader() { if (map_ && map_ != MAP_FAILED) ::munmap(const_cast<std::uint8_t*>(map_), size_); }

    bool has_section(const std::string& n) const { return sections_.count(n) != 0; }
    std::span<const std::uint8_t> section_bytes(const std::string& n) const {
        auto it = sections_.find(n);
        if (it == sections_.end()) raise(errc::not_found, "{}: no section '{}'", path_, n);
        return {map_ + it->second.offset, it->second.size};
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
    tensor load(const std::string& n) const {
        const auto& e = entry(n);
        auto* c = flow::codecs::find(e.codec);
        if (!c) raise(errc::unsupported, "tensor {}: unknown codec {}", n, e.codec);
        tensor t(e.shape);
        c->decode(section_bytes("data").data() + e.offset, t.size(), t.data.data());
        return t;
    }
    // Verifies every section hash and every tensor hash. Returns list of problems (empty == ok).
    std::vector<std::string> verify() const {
        std::vector<std::string> bad;
        for (auto& [n, s] : sections_)
            if (hash::blake3::of({map_ + s.offset, s.size}) != s.hash) bad.push_back("section " + n);
        auto d = section_bytes("data");
        for (auto& [n, e] : tensors_)
            if (hash::blake3::hex(hash::blake3::of({d.data() + e.offset, e.bytes})) != e.blake3) bad.push_back("tensor " + n);
        return bad;
    }
    std::size_t file_size() const { return size_; }
    struct section { std::string name; std::uint64_t offset = 0, size = 0; hash::blake3::digest hash{}; };
    const std::map<std::string, section>& sections() const { return sections_; }
private:
    std::string path_;
    const std::uint8_t* map_ = nullptr;
    std::size_t size_ = 0;
    std::map<std::string, section> sections_;
    std::map<std::string, tensor_entry> tensors_;
};

} // namespace forge::pack

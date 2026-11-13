#pragma once
// The .fpk "adapters" section (FORGE_PLAN D.2 "[ADAPTERS] optional LoRA sets"): LoRA adapters carried inside the
// model file, selected by name.
//
//   {"version":1,"adapters":[{"name":"style","alpha":16,"rank":4,"targets":["layer.0.wq","layer.0.wv"]}]}
//
// The factors of adapter N for weight W are ordinary tensors of the file, named "adapter.N.W.lora_a" ([rank, in]) and
// "adapter.N.W.lora_b" ([out, rank]) (PEFT layout; W' = W + alpha / rank * B @ A). Being ordinary tensors they are
// hashed, aligned, encrypted and signed with the rest; a reader that predates the section sees unused tensors and an
// unknown section, and runs the base model.
//
// embed_adapter() adds one adapter (from a LoRA .fpk as model::add_adapter_file reads it: "<W>.lora_a" / "<W>.lora_b"
// tensors, manifest "lora_alpha") to a model file; remove_adapter() takes one out. Both rewrite the container, which
// changes the tensor index: a signature and precompiled plans ("aot") are dropped, and encrypted files are refused
// (embed first, then encrypt, then sign).
#include <forge/modelfile/fpk.hpp>
#include <set>

namespace forge::modelfile {

inline constexpr char adapters_section[] = "adapters";
inline constexpr std::string_view adapter_prefix = "adapter.";

struct adapter_info {
    std::string name;
    double alpha = 1;
    i64 rank = 0;
    std::vector<std::string> targets;   // weight names, sorted
    static std::string tensor_a(std::string_view adapter, std::string_view target) { return std::string(adapter_prefix) + std::string(adapter) + "." + std::string(target) + ".lora_a"; }
    static std::string tensor_b(std::string_view adapter, std::string_view target) { return std::string(adapter_prefix) + std::string(adapter) + "." + std::string(target) + ".lora_b"; }
};

// Adapter names are identifiers chosen by whoever built the file: letters, digits, '_' and '-', at most 64.
inline bool valid_adapter_name(std::string_view n) {
    if (n.empty() || n.size() > 64) return false;
    for (char c : n) if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) return false;
    return true;
}

inline std::string dump_adapters(const std::vector<adapter_info>& as) {
    json::array a;
    for (auto& x : as) {
        json::array t; for (auto& n : x.targets) t.push_back(n);
        a.push_back(json::object{{"name", x.name}, {"alpha", x.alpha}, {"rank", double(x.rank)}, {"targets", t}});
    }
    return json::dump(json::value(json::object{{"version", 1.0}, {"adapters", a}}));
}

// The adapters a file carries (empty without the section). The section is untrusted: names, counts and ranks are
// bounded, and every target must have both factor tensors in the index with shapes that fit the target weight.
inline std::vector<adapter_info> read_adapters(const reader& r) {
    std::vector<adapter_info> out;
    if (!r.has_section(adapters_section)) return out;
    auto bad = [&](const std::string& why) { raise(errc::format_error, "adapters section: {}", why); };
    json::value v = json::parse(r.section_text(adapters_section));
    if (!v.is_obj()) bad("not a JSON object");
    if (v.num_or("version", 0) != 1) raise(errc::unsupported, "adapters section: version {} is not supported (this build reads version 1)", v.num_or("version", 0));
    auto* arr = v.find("adapters");
    if (!arr || !arr->is_arr() || arr->arr().size() > 4096) bad("no adapter list");
    std::set<std::string> seen;
    for (auto& e : arr->arr()) {
        adapter_info a; a.name = e.str_or("name", "");
        if (!valid_adapter_name(a.name)) bad("bad adapter name");
        if (!seen.insert(a.name).second) bad("adapter '" + a.name + "' is listed twice");
        a.alpha = e.num_or("alpha", 1);
        double rk = e.num_or("rank", 0);
        if (!std::isfinite(a.alpha) || !(rk >= 1 && rk <= 65536) || rk != std::floor(rk)) bad("adapter '" + a.name + "' has a bad alpha or rank");
        a.rank = i64(rk);
        auto* t = e.find("targets");
        if (!t || !t->is_arr() || t->arr().empty() || t->arr().size() > 65536) bad("adapter '" + a.name + "' has no targets");
        for (auto& n : t->arr()) {
            if (!n.is_str() || n.str().empty() || n.str().size() > 256) bad("adapter '" + a.name + "' has a bad target name");
            const auto& w = n.str();
            auto& ts = r.tensors();
            auto wi = ts.find(w), ai = ts.find(adapter_info::tensor_a(a.name, w)), bi = ts.find(adapter_info::tensor_b(a.name, w));
            if (wi == ts.end() || ai == ts.end() || bi == ts.end()) bad("adapter '" + a.name + "': target " + w + " or its factors are not in the tensor index");
            const auto& W = wi->second.shape; const auto& A = ai->second.shape; const auto& B = bi->second.shape;
            if (W.size() != 2 || A.size() != 2 || B.size() != 2 || A[0] != a.rank || B[1] != a.rank || B[0] != W[0] || A[1] != W[1])
                bad("adapter '" + a.name + "': factors of " + w + " do not fit the weight");
            a.targets.push_back(w);
        }
        std::sort(a.targets.begin(), a.targets.end());
        if (std::adjacent_find(a.targets.begin(), a.targets.end()) != a.targets.end()) bad("adapter '" + a.name + "' lists a target twice");
        out.push_back(std::move(a));
    }
    return out;
}

namespace detail {
// Rewrites `in` to `out` with `drop` tensors removed from the index and `add` (name, shape, codec, bytes) appended to
// the data section, and `adapters` as the adapters section (removed when empty). Dropped tensors' bytes stay in the
// data section as dead padding: offsets of the others do not move.
struct raw_tensor { std::string name; shape_t shape; std::string codec; std::vector<std::uint8_t> bytes; };
inline void rewrite_tensors(const std::string& in, const std::string& out, const std::set<std::string>& drop, const std::vector<raw_tensor>& add,
                            const std::vector<adapter_info>& adapters) {
    std::vector<std::pair<std::string, std::string>> secs; data_layout dl; std::string data; json::array idx;
    {
        reader r(in);
        if (r.has_section("encryption")) raise(errc::unsupported, "{}: the file is encrypted; change adapters on the plain file, then encrypt it", in);
        if (r.sharded()) raise(errc::unsupported, "{}: the file is sharded; change adapters on the joined file (forge shard join), then split it again", in);
        if (auto bad = r.verify(); !bad.empty()) raise(errc::format_error, "{}: refusing to rewrite a corrupt file ({})", in, bad.front());
        dl = r.layout();
        for (auto& [name, s] : r.sections()) {
            if (name == "signature" || name == "aot" || name == "tensors" || name == adapters_section) continue;   // signature and plans no longer match the index
            auto b = r.raw_section_bytes(name);
            if (name == "data") data.assign(reinterpret_cast<const char*>(b.data()), b.size());
            else secs.emplace_back(name, std::string(reinterpret_cast<const char*>(b.data()), b.size()));
        }
        for (auto& t : json::parse(r.section_text("tensors")).arr()) if (!drop.count(t["name"].str())) idx.push_back(t);
        for (auto& t : add) if (r.tensors().count(t.name) && !drop.count(t.name)) raise(errc::invalid_argument, "{}: tensor {} already exists", in, t.name);
    }
    for (auto& t : add) {
        auto* c = graph::codecs::find(t.codec);
        if (!c || c->bytes(std::size_t(numel(t.shape))) != t.bytes.size()) raise(errc::format_error, "tensor {}: bytes do not match {} {}", t.name, t.codec, shape_str(t.shape));
        while ((dl.phase + data.size()) % dl.align) data.push_back('\0');   // the new tensor's file offset keeps the file's alignment
        json::array sh; for (auto d : t.shape) sh.push_back(double(d));
        idx.push_back(json::object{{"name", t.name}, {"codec", t.codec}, {"shape", sh}, {"offset", double(data.size())}, {"bytes", double(t.bytes.size())},
                                   {"blake3", hash::blake3::hex(hash::blake3::of(t.bytes))}});
        data.append(reinterpret_cast<const char*>(t.bytes.data()), t.bytes.size());
    }
    secs.emplace_back("data", std::move(data));
    secs.emplace_back("tensors", json::dump(json::value(idx)));
    if (!adapters.empty()) secs.emplace_back(adapters_section, dump_adapters(adapters));
    write_sections(out, std::move(secs), dl);
}
} // namespace detail

// Adds the LoRA adapter stored in `lora_fpk` to the model `in` under `name`, writing `out` (may equal `in`).
inline adapter_info embed_adapter(const std::string& in, const std::string& out, const std::string& name, const std::string& lora_fpk) {
    if (!valid_adapter_name(name)) raise(errc::invalid_argument, "adapter name '{}': use letters, digits, '_' and '-' (at most 64)", name);
    std::vector<adapter_info> all; adapter_info a; a.name = name;
    std::vector<detail::raw_tensor> add;
    {
        reader m(in), l(lora_fpk);
        all = read_adapters(m);
        for (auto& x : all) if (x.name == name) raise(errc::invalid_argument, "{}: already has an adapter named '{}'", in, name);
        a.alpha = l.manifest().num_or("lora_alpha", 1);
        for (auto& [n, e] : l.tensors()) {
            auto cut = n.rfind(".lora_");
            if (cut == std::string::npos || n.substr(cut) != ".lora_a") continue;
            auto w = n.substr(0, cut);
            auto bi = l.tensors().find(w + ".lora_b"); auto wi = m.tensors().find(w);
            if (bi == l.tensors().end()) raise(errc::format_error, "{}: {} has no matching lora_b", lora_fpk, n);
            if (wi == m.tensors().end()) raise(errc::not_found, "{}: adapter targets {}, which the model does not have", in, w);
            const auto& A = e.shape; const auto& B = bi->second.shape; const auto& W = wi->second.shape;
            if (A.size() != 2 || B.size() != 2 || W.size() != 2 || B[1] != A[0] || B[0] != W[0] || A[1] != W[1])
                raise(errc::shape_mismatch, "adapter {}: factors of {} ({} x {}) do not fit the weight {}", name, w, shape_str(B), shape_str(A), shape_str(W));
            if (a.rank && a.rank != A[0]) raise(errc::unsupported, "adapter {}: targets use different ranks ({} and {})", name, a.rank, A[0]);
            a.rank = A[0]; a.targets.push_back(w);
            for (auto* src : {&e, &bi->second}) {
                auto bytes = l.tensor_bytes(src->name);
                add.push_back({src == &e ? adapter_info::tensor_a(name, w) : adapter_info::tensor_b(name, w), src->shape, src->codec, {bytes.begin(), bytes.end()}});
            }
        }
        if (a.targets.empty()) raise(errc::format_error, "{}: no <weight>.lora_a / <weight>.lora_b tensors", lora_fpk);
        std::sort(a.targets.begin(), a.targets.end());
    }
    all.push_back(a);
    detail::rewrite_tensors(in, out, {}, add, all);
    return a;
}

// Removes adapter `name` from `in`, writing `out`. Its tensors leave the index; their bytes stay as padding.
inline void remove_adapter(const std::string& in, const std::string& out, const std::string& name) {
    std::vector<adapter_info> all, keep; std::set<std::string> drop;
    { reader m(in); all = read_adapters(m); }
    for (auto& x : all) {
        if (x.name != name) { keep.push_back(x); continue; }
        for (auto& w : x.targets) { drop.insert(adapter_info::tensor_a(name, w)); drop.insert(adapter_info::tensor_b(name, w)); }
    }
    if (keep.size() == all.size()) raise(errc::not_found, "{}: no adapter named '{}'", in, name);
    detail::rewrite_tensors(in, out, drop, {}, keep);
}

} // namespace forge::modelfile

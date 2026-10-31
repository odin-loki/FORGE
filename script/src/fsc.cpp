// ForgeScript .fsc cache encoder/decoder (see fsc.hpp).
#include "fsc.hpp"
#include <forge/hash/blake3.hpp>
#include "monocypher.h"
#include "monocypher-ed25519.h"
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>

namespace forge::script::fsc {
namespace {
constexpr char magic[4] = {'F', 'S', 'C', '1'};
constexpr int max_depth = 4096;

struct W {
    std::vector<std::uint8_t> b;
    std::unordered_map<const void*, std::uint32_t> ids;
    void u8(std::uint8_t v) { b.push_back(v); }
    void u32(std::uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back(std::uint8_t(v >> (8 * i))); }
    void u64(std::uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back(std::uint8_t(v >> (8 * i))); }
    void i32(int v) { u32(std::uint32_t(v)); }
    void f64(double d) { std::uint64_t v; std::memcpy(&v, &d, 8); u64(v); }
    void str(const std::string& s) { u32(std::uint32_t(s.size())); b.insert(b.end(), s.begin(), s.end()); }
    void strs(const std::vector<std::string>& v) { u32(std::uint32_t(v.size())); for (auto& s : v) str(s); }
    // 0 = null, 1 = new object follows, 2 + id = back-reference
    bool ref(const void* p) {
        if (!p) { u8(0); return false; }
        if (auto it = ids.find(p); it != ids.end()) { u8(2); u32(it->second); return false; }
        ids.emplace(p, std::uint32_t(ids.size() + 1)); u8(1); return true;
    }
    void var(const VarRef& v) { str(v.name); i32(v.depth); i32(v.slot); }
    void ann_val(const TypeAnn& t) { str(t.name); u8(t.nullable); u32(std::uint32_t(t.args.size())); for (auto& a : t.args) ann_val(a); }
    void ann(const std::shared_ptr<TypeAnn>& t) { if (ref(t.get())) ann_val(*t); }
    void node(const NodeP& n) {
        if (!ref(n.get())) return;
        u32(std::uint32_t(n->k)); i32(n->line); str(n->s); u64(std::uint64_t(n->i)); f64(n->d); u8(n->flag); u8(n->wb);
        var(n->var); ann(n->type);
        u32(std::uint32_t(n->kids.size())); for (auto& k : n->kids) node(k);
        fn(n->fn); cls(n->cls); node(n->extra);
        u32(std::uint32_t(n->locals.size())); for (int l : n->locals) i32(l);
    }
    void fn(const std::shared_ptr<const FnDecl>& f) { if (ref(f.get())) fn_val(*f); }
    void fn_val(const FnDecl& f) {
        str(f.name); i32(f.line);
        u32(std::uint32_t(f.params.size()));
        for (auto& p : f.params) { str(p.name); i32(p.slot); ann(p.type); node(p.dflt); u8(p.by_ref); u8(p.out); u8(std::uint8_t(p.conv)); str(p.tname); u8(p.cnst); }
        ann(f.ret); node(f.body); i32(f.nslots);
        u8(f.is_method); u8(f.has_out); u8(f.is_top); u8(f.has_main); str(f.file); str(f.ret_name);
        u8(f.is_const); u8(f.is_static); u8(f.is_virtual); strs(f.attrs); str(f.access);
    }
    void meta(const std::shared_ptr<ClassMeta>& m) {
        if (!ref(m.get())) return;
        strs(m->attrs); u32(std::uint32_t(m->fields.size()));
        for (auto& f : m->fields) { str(f.name); str(f.type); ann(f.ann); u8(f.is_const); u8(f.is_static); str(f.access); strs(f.attrs); i32(f.line); }
        str(m->file); i32(m->line);
    }
    void cls(const std::shared_ptr<const ClassDecl>& c) {
        if (!ref(c.get())) return;
        str(c->name); str(c->base); strs(c->interfaces); u8(c->is_struct); u8(c->is_dynamic); u8(c->is_interface);
        u32(std::uint32_t(c->fields.size())); for (auto& [n, d] : c->fields) { str(n); node(d); }
        u32(std::uint32_t(c->methods.size())); for (auto& m : c->methods) fn(m);
        strs(c->required); strs(c->statics); var(c->var); meta(c->meta);
    }
};

struct bad : std::runtime_error { using std::runtime_error::runtime_error; };

struct R {
    std::span<const std::uint8_t> b; std::size_t p = 0; int depth = 0;
    std::vector<std::shared_ptr<void>> objs;   // id - 1 -> object (type-checked by tag below)
    std::vector<int> kinds;                    // 1 TypeAnn 2 Node 3 FnDecl 4 ClassDecl 5 ClassMeta
    void need(std::size_t n) { if (b.size() - p < n) throw bad("truncated"); }
    std::uint8_t u8() { need(1); return b[p++]; }
    bool bl() { auto v = u8(); if (v > 1) throw bad("bad bool"); return v; }
    std::uint32_t u32() { need(4); std::uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= std::uint32_t(b[p++]) << (8 * i); return v; }
    std::uint64_t u64() { need(8); std::uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= std::uint64_t(b[p++]) << (8 * i); return v; }
    int i32() { return int(u32()); }
    double f64() { std::uint64_t v = u64(); double d; std::memcpy(&d, &v, 8); return d; }
    std::uint32_t count(std::size_t min_each = 1) { auto n = u32(); if (std::size_t(n) * min_each > b.size() - p) throw bad("count exceeds payload"); return n; }
    std::string str() { auto n = count(); std::string s(reinterpret_cast<const char*>(b.data() + p), n); p += n; return s; }
    std::vector<std::string> strs() { std::vector<std::string> v(count(4)); for (auto& s : v) s = str(); return v; }
    struct guard { R& r; guard(R& x) : r(x) { if (++r.depth > max_depth) throw bad("nesting too deep"); } ~guard() { --r.depth; } };
    // returns: 0 null, 1 new (caller registers), 2 back-reference (out set)
    template <class T> int ref(int kind, std::shared_ptr<T>& out) {
        auto t = u8();
        if (t == 0) return 0;
        if (t == 1) return 1;
        if (t != 2) throw bad("bad ref tag");
        auto id = u32(); if (id == 0 || id > objs.size() || kinds[id - 1] != kind) throw bad("bad back-reference");
        out = std::static_pointer_cast<T>(objs[id - 1]); return 2;
    }
    template <class T> void reg(int kind, const std::shared_ptr<T>& o) { objs.push_back(o); kinds.push_back(kind); }
    VarRef var() { VarRef v; v.name = str(); v.depth = i32(); v.slot = i32(); return v; }
    TypeAnn ann_val() { guard g(*this); TypeAnn t; t.name = str(); t.nullable = bl(); t.args.resize(count()); for (auto& a : t.args) a = ann_val(); return t; }
    std::shared_ptr<TypeAnn> ann() {
        std::shared_ptr<TypeAnn> t; if (ref(1, t) != 1) return t;
        t = std::make_shared<TypeAnn>(); reg(1, t); *t = ann_val(); return t;
    }
    NodeP node() {
        guard g(*this);
        std::shared_ptr<Node> n; if (ref(2, n) != 1) return n;
        n = std::make_shared<Node>(); reg(2, n);
        auto k = u32(); if (k > std::uint32_t(NK::Import)) throw bad("bad node kind"); n->k = NK(k);
        n->line = i32(); n->s = str(); n->i = std::int64_t(u64()); n->d = f64(); n->flag = bl(); n->wb = bl();
        n->var = var(); n->type = ann();
        n->kids.resize(count()); for (auto& c : n->kids) c = node();
        n->fn = fn(); n->cls = cls(); n->extra = node();
        n->locals.resize(count(4)); for (auto& l : n->locals) l = i32();
        return n;
    }
    std::shared_ptr<const FnDecl> fn() {
        guard g(*this);
        std::shared_ptr<FnDecl> f; if (ref(3, f) != 1) return f;
        f = std::make_shared<FnDecl>(); reg(3, f);
        f->name = str(); f->line = i32();
        f->params.resize(count());
        for (auto& q : f->params) { q.name = str(); q.slot = i32(); q.type = ann(); q.dflt = node(); q.by_ref = bl(); q.out = bl(); q.conv = char(u8()); q.tname = str(); q.cnst = bl(); }
        f->ret = ann(); f->body = node(); f->nslots = i32();
        f->is_method = bl(); f->has_out = bl(); f->is_top = bl(); f->has_main = bl(); f->file = str(); f->ret_name = str();
        f->is_const = bl(); f->is_static = bl(); f->is_virtual = bl(); f->attrs = strs(); f->access = str();
        if (f->nslots < 0 || f->nslots > (1 << 20)) throw bad("bad slot count");
        return f;
    }
    std::shared_ptr<ClassMeta> meta() {
        std::shared_ptr<ClassMeta> m; if (ref(5, m) != 1) return m;
        m = std::make_shared<ClassMeta>(); reg(5, m);
        m->attrs = strs(); m->fields.resize(count());
        for (auto& f : m->fields) { f.name = str(); f.type = str(); f.ann = ann(); f.is_const = bl(); f.is_static = bl(); f.access = str(); f.attrs = strs(); f.line = i32(); }
        m->file = str(); m->line = i32(); return m;
    }
    std::shared_ptr<const ClassDecl> cls() {
        guard g(*this);
        std::shared_ptr<ClassDecl> c; if (ref(4, c) != 1) return c;
        c = std::make_shared<ClassDecl>(); reg(4, c);
        c->name = str(); c->base = str(); c->interfaces = strs(); c->is_struct = bl(); c->is_dynamic = bl(); c->is_interface = bl();
        c->fields.resize(count()); for (auto& [n, d] : c->fields) { n = str(); d = node(); }
        c->methods.resize(count()); for (auto& m : c->methods) m = fn();
        c->required = strs(); c->statics = strs(); c->var = var(); c->meta = meta();
        if (!c->meta) c->meta = std::make_shared<ClassMeta>();
        return c;
    }
};

std::array<std::uint8_t, 32> sig_digest(const key_t& key, std::span<const std::uint8_t> payload, std::uint8_t out[44 + 32]) {
    auto pd = hash::blake3::of(payload);
    std::memcpy(out, "forge-fsc-v1", 12); std::memcpy(out + 12, key.data(), 32); std::memcpy(out + 44, pd.data(), 32);
    return pd;
}
} // namespace

key_t key_of(std::string_view src, const std::string& file) {
    hash::blake3 h; h.update("forge-fsc\0", 10);
    std::uint8_t v[4]; for (int i = 0; i < 4; ++i) v[i] = std::uint8_t(vm_version >> (8 * i));
    h.update(v, 4); h.update(file.data(), file.size()); h.update("\0", 1); h.update(src.data(), src.size());
    auto d = h.final(); key_t k; std::memcpy(k.data(), d.data(), 32); return k;
}
std::string key_hex(const key_t& k) { hash::blake3::digest d; std::memcpy(d.data(), k.data(), 32); return hash::blake3::hex(d); }
std::string path_for(const std::string& dir, const key_t& k) { return (std::filesystem::path(dir) / (key_hex(k) + ".fsc")).string(); }

std::vector<std::uint8_t> encode(const FnDecl& top, const key_t& key, const std::array<std::uint8_t, 32>* seed) {
    W pw; pw.u8(1); pw.ids.emplace(&top, 1u); pw.fn_val(top);
    W w; w.b.insert(w.b.end(), magic, magic + 4); w.u32(vm_version); w.b.insert(w.b.end(), key.begin(), key.end());
    w.u64(pw.b.size()); w.b.insert(w.b.end(), pw.b.begin(), pw.b.end());
    if (!seed) { w.u8(0); return w.b; }
    std::uint8_t msg[76]; sig_digest(key, pw.b, msg);
    std::uint8_t sk[64], pub[32], sig[64], s2[32]; std::memcpy(s2, seed->data(), 32);
    crypto_ed25519_key_pair(sk, pub, s2); crypto_ed25519_sign(sig, sk, msg, sizeof msg); crypto_wipe(sk, 64);
    w.u8(1); w.b.insert(w.b.end(), pub, pub + 32); w.b.insert(w.b.end(), sig, sig + 64);
    return w.b;
}

std::optional<std::shared_ptr<const FnDecl>> decode(std::span<const std::uint8_t> bytes, const key_t& expect, const trust& t, std::string* why) {
    auto fail = [&](const std::string& m) -> std::optional<std::shared_ptr<const FnDecl>> { if (why) *why = m; return std::nullopt; };
    try {
        R h{bytes};
        h.need(4); if (std::memcmp(bytes.data(), magic, 4)) return fail("not an .fsc file"); h.p = 4;
        if (h.u32() != vm_version) return fail("VM version mismatch");
        h.need(32); if (std::memcmp(bytes.data() + h.p, expect.data(), 32)) return fail("stale (source or name changed)"); h.p += 32;
        auto n = h.u64(); if (n > bytes.size() - h.p) return fail("truncated payload");
        auto payload = bytes.subspan(h.p, std::size_t(n)); h.p += std::size_t(n);
        bool is_signed = h.bl();
        if (is_signed) {
            h.need(96); const std::uint8_t* pub = bytes.data() + h.p; const std::uint8_t* sig = pub + 32; h.p += 96;
            bool known = false; for (auto& k : t.pubkeys) if (!std::memcmp(k.data(), pub, 32)) known = true;
            if (!known) return fail("signed by an untrusted key");
            std::uint8_t msg[76]; sig_digest(expect, payload, msg);
            if (crypto_ed25519_check(sig, pub, msg, sizeof msg) != 0) return fail("bad signature");
        } else if (t.require_signed) return fail("unsigned cache rejected (signatures required)");
        if (h.p != bytes.size()) return fail("trailing bytes");
        R r{payload}; auto f = r.fn();
        if (!f || r.p != payload.size()) return fail("malformed payload");
        return f;
    } catch (const std::exception& e) { return fail(std::string("malformed: ") + e.what()); }
}

} // namespace forge::script::fsc

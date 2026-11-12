#pragma once
// forge::os::dbus: a small D-Bus wire client, so desktop features on Linux and FreeBSD need no helper program
// and no libdbus / libsystemd. Guide: docs/os-wrapper.md.
//
//   Codec (every OS): the D-Bus type system as a `value` tree, marshalling (little endian out, either order in)
//   and message framing. decode() takes bytes from an untrusted peer: every length, offset, alignment gap,
//   signature and nesting depth is checked, sizes are capped (limits), and nothing is read past the buffer.
//   fuzz/fuzz_dbus.cpp drives it.
//   Connection (POSIX): AF_UNIX transports (`unix:path=`, `unix:abstract=` on Linux) from
//   DBUS_SESSION_BUS_ADDRESS / DBUS_SYSTEM_BUS_ADDRESS or the standard socket paths, SASL EXTERNAL
//   authentication, optional unix-fd passing (receiving), Hello, method calls with a timeout, match rules and a
//   signal queue. Blocking, one thread at a time per connection. Windows, iOS and Android have no bus to talk
//   to: open() returns unsupported there.
#include <forge/os/os.hpp>
#include <deque>
#include <memory>
#if defined(FORGE_OS_POSIX)
#include <sys/un.h>
#endif

namespace forge::os::dbus {

struct limits {
    std::size_t max_message = 16u << 20;   // bytes of one message (the protocol allows 128 MiB)
    std::size_t max_nodes = 1u << 20;      // values in one decoded message
    unsigned max_depth = 64;               // container nesting (the protocol's limit)
};

// One D-Bus value. type: y b n q i u x t d (numbers), s o g (text), h (fd index), a (array), r (struct),
// e (dict entry), v (variant).
struct value {
    char type = 0;
    std::uint64_t u = 0;          // integers and booleans (signed ones sign-extended)
    double d = 0;
    std::string s;                // s / o / g text; for arrays the element signature
    std::string bytes;            // array of bytes ("ay") content
    std::vector<value> items;     // array elements, struct fields, {key, value}, or the one value in a variant

    static value byte(std::uint8_t v) { value x; x.type = 'y'; x.u = v; return x; }
    static value boolean(bool v) { value x; x.type = 'b'; x.u = v; return x; }
    static value i16(std::int16_t v) { value x; x.type = 'n'; x.u = std::uint64_t(std::int64_t(v)); return x; }
    static value u16(std::uint16_t v) { value x; x.type = 'q'; x.u = v; return x; }
    static value i32(std::int32_t v) { value x; x.type = 'i'; x.u = std::uint64_t(std::int64_t(v)); return x; }
    static value u32(std::uint32_t v) { value x; x.type = 'u'; x.u = v; return x; }
    static value i64(std::int64_t v) { value x; x.type = 'x'; x.u = std::uint64_t(v); return x; }
    static value u64(std::uint64_t v) { value x; x.type = 't'; x.u = v; return x; }
    static value f64(double v) { value x; x.type = 'd'; x.d = v; return x; }
    static value str(std::string v) { value x; x.type = 's'; x.s = std::move(v); return x; }
    static value path(std::string v) { value x; x.type = 'o'; x.s = std::move(v); return x; }
    static value sig(std::string v) { value x; x.type = 'g'; x.s = std::move(v); return x; }
    static value array(std::string element_signature, std::vector<value> v = {}) { value x; x.type = 'a'; x.s = std::move(element_signature); x.items = std::move(v); return x; }
    static value byte_array(std::string b) { value x; x.type = 'a'; x.s = "y"; x.bytes = std::move(b); return x; }
    static value strukt(std::vector<value> v) { value x; x.type = 'r'; x.items = std::move(v); return x; }
    static value entry(value k, value v) { value x; x.type = 'e'; x.items.push_back(std::move(k)); x.items.push_back(std::move(v)); return x; }
    static value variant(value v) { value x; x.type = 'v'; x.items.push_back(std::move(v)); return x; }
    // a{sv}: the usual options / hints dictionary
    static value dict(std::vector<std::pair<std::string, value>> kv = {}) {
        value x; x.type = 'a'; x.s = "{sv}";
        for (auto& [k, v] : kv) x.items.push_back(entry(str(k), variant(std::move(v))));
        return x;
    }
    std::int64_t as_int() const { return std::int64_t(u); }
    bool is_text() const { return type == 's' || type == 'o' || type == 'g'; }
    // The value inside a variant (or the value itself).
    const value& unwrap() const { const value* p = this; while (p->type == 'v' && p->items.size() == 1) p = &p->items[0]; return *p; }
    // Looks up a string key in an array of dict entries (a{s?}); nullptr when absent.
    const value* find(std::string_view key) const {
        if (type != 'a') return nullptr;
        for (auto& e : items) if (e.type == 'e' && e.items.size() == 2 && e.items[0].is_text() && e.items[0].s == key) return &e.items[1];
        return nullptr;
    }
    std::string signature() const {
        switch (type) {
        case 'a': return "a" + s;
        case 'r': { std::string o = "("; for (auto& i : items) o += i.signature(); return o + ")"; }
        case 'e': { std::string o = "{"; for (auto& i : items) o += i.signature(); return o + "}"; }
        default: return type ? std::string(1, type) : std::string();
        }
    }
};

namespace detail {
inline std::size_t align_of(char c) {
    switch (c) { case 'y': case 'g': case 'v': return 1; case 'n': case 'q': return 2; case 'x': case 't': case 'd': case '(': case '{': case 'r': case 'e': return 8; default: return 4; }
}
inline bool basic(char c) { return c && std::strchr("ybnqiuxtdsogh", c) != nullptr; }
// Length of the single complete type at sig[pos], 0 if it is malformed (depth: 32 arrays, 32 structs).
inline std::size_t type_len(std::string_view sig, std::size_t pos, unsigned arrays = 0, unsigned structs = 0) {
    if (pos >= sig.size()) return 0;
    char c = sig[pos];
    if (basic(c) || c == 'v') return 1;
    if (c == 'a') { if (arrays >= 32) return 0; std::size_t n = type_len(sig, pos + 1, arrays + 1, structs); return n ? n + 1 : 0; }
    if (c == '(') {
        if (structs >= 32) return 0;
        std::size_t i = pos + 1; if (i < sig.size() && sig[i] == ')') return 0;   // empty structs do not exist
        while (i < sig.size() && sig[i] != ')') { std::size_t n = type_len(sig, i, arrays, structs + 1); if (!n) return 0; i += n; }
        return i < sig.size() ? i + 1 - pos : 0;
    }
    if (c == '{') {   // only as an array element; key must be basic
        if (structs >= 32 || arrays == 0 || pos == 0 || sig[pos - 1] != 'a') return 0;
        if (pos + 1 >= sig.size() || !basic(sig[pos + 1])) return 0;
        std::size_t n = type_len(sig, pos + 2, arrays, structs + 1); if (!n) return 0;
        std::size_t end = pos + 2 + n; return end < sig.size() && sig[end] == '}' ? end + 1 - pos : 0;
    }
    return 0;
}
}
// A sequence of complete types (a message body signature), at most 255 bytes.
inline bool valid_signature(std::string_view sig) {
    if (sig.size() > 255) return false;
    for (std::size_t i = 0; i < sig.size();) { std::size_t n = detail::type_len(sig, i); if (!n) return false; i += n; }
    return true;
}
inline bool valid_single_type(std::string_view sig) { return !sig.empty() && sig.size() <= 255 && detail::type_len(sig, 0) == sig.size(); }
inline bool valid_object_path(std::string_view p) {
    if (p.empty() || p[0] != '/') return false;
    if (p.size() == 1) return true;
    for (std::size_t i = 1; i < p.size(); ++i) {
        char c = p[i];
        if (c == '/') { if (p[i - 1] == '/' || i + 1 == p.size()) return false; continue; }
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) return false;
    }
    return true;
}

namespace detail {
// ---- marshalling (little endian)
struct writer {
    std::string b;
    void pad(std::size_t a) { while (b.size() % a) b.push_back('\0'); }
    template <class T> void num(T v) { pad(sizeof(T)); for (std::size_t i = 0; i < sizeof(T); ++i) b.push_back(char((std::uint64_t(v) >> (8 * i)) & 0xff)); }
    void put_u32_at(std::size_t at, std::uint32_t v) { for (int i = 0; i < 4; ++i) b[at + std::size_t(i)] = char((v >> (8 * i)) & 0xff); }
    bool put(const value& v, unsigned depth = 0) {
        if (depth > 64) return false;
        switch (v.type) {
        case 'y': b.push_back(char(v.u)); return true;
        case 'b': num<std::uint32_t>(v.u ? 1u : 0u); return true;
        case 'n': case 'q': num<std::uint16_t>(std::uint16_t(v.u)); return true;
        case 'i': case 'u': case 'h': num<std::uint32_t>(std::uint32_t(v.u)); return true;
        case 'x': case 't': num<std::uint64_t>(v.u); return true;
        case 'd': { std::uint64_t r; std::memcpy(&r, &v.d, 8); num<std::uint64_t>(r); return true; }
        case 's': case 'o':
            if (v.s.size() > 0x7fffffffu || v.s.find('\0') != std::string::npos || (v.type == 'o' && !valid_object_path(v.s))) return false;
            num<std::uint32_t>(std::uint32_t(v.s.size())); b += v.s; b.push_back('\0'); return true;
        case 'g':
            if (!valid_signature(v.s)) return false;
            b.push_back(char(v.s.size())); b += v.s; b.push_back('\0'); return true;
        case 'a': {
            if (!valid_single_type("a" + v.s)) return false;
            num<std::uint32_t>(0); std::size_t len_at = b.size() - 4;
            pad(align_of(v.s[0])); std::size_t start = b.size();
            if (v.s == "y" && v.items.empty()) b += v.bytes;
            else for (auto& i : v.items) { if (i.signature() != v.s || !put(i, depth + 1)) return false; }
            if (b.size() - start > (1u << 26)) return false;
            put_u32_at(len_at, std::uint32_t(b.size() - start)); return true;
        }
        case 'r': case 'e':
            if (v.items.empty() || (v.type == 'e' && (v.items.size() != 2 || !basic(v.items[0].type)))) return false;
            pad(8); for (auto& i : v.items) if (!put(i, depth + 1)) return false;
            return true;
        case 'v': {
            if (v.items.size() != 1) return false;
            std::string sg = v.items[0].signature(); if (!valid_single_type(sg)) return false;
            b.push_back(char(sg.size())); b += sg; b.push_back('\0'); return put(v.items[0], depth + 1);
        }
        default: return false;
        }
    }
};
// ---- unmarshalling: every access is bounds-checked against [0, n)
struct reader {
    const std::uint8_t* p; std::size_t n; bool big; const limits& lim; std::size_t pos = 0, nodes = 0;
    bool pad(std::size_t a) {   // padding must exist (its content is not required to be zero by every sender)
        std::size_t r = pos % a; if (!r) return true;
        if (a - r > n - pos) return false;
        pos += a - r; return true;
    }
    template <class T> bool num(T& out) {
        if (!pad(sizeof(T)) || n - pos < sizeof(T)) return false;
        std::uint64_t v = 0;
        for (std::size_t i = 0; i < sizeof(T); ++i) v |= std::uint64_t(p[pos + i]) << (8 * (big ? sizeof(T) - 1 - i : i));
        out = T(v); pos += sizeof(T); return true;
    }
    bool text(std::size_t len, std::string& out) {   // len bytes then a NUL
        if (len >= n - pos) return false;             // needs len + 1 bytes
        if (p[pos + len] != 0) return false;
        out.assign(reinterpret_cast<const char*>(p + pos), len); pos += len + 1;
        return out.find('\0') == std::string::npos;
    }
    // Reads the one complete type `sig` (already validated).
    bool get(std::string_view sig, value& v, unsigned depth) {
        if (depth > lim.max_depth || ++nodes > lim.max_nodes || sig.empty()) return false;
        char c = sig[0];
        switch (c) {
        case 'y': if (pos >= n) return false; v.type = c; v.u = p[pos++]; return true;
        case 'b': { std::uint32_t x; if (!num(x) || x > 1) return false; v.type = c; v.u = x; return true; }
        case 'n': { std::uint16_t x; if (!num(x)) return false; v.type = c; v.u = std::uint64_t(std::int64_t(std::int16_t(x))); return true; }
        case 'q': { std::uint16_t x; if (!num(x)) return false; v.type = c; v.u = x; return true; }
        case 'i': { std::uint32_t x; if (!num(x)) return false; v.type = c; v.u = std::uint64_t(std::int64_t(std::int32_t(x))); return true; }
        case 'u': case 'h': { std::uint32_t x; if (!num(x)) return false; v.type = c; v.u = x; return true; }
        case 'x': case 't': { std::uint64_t x; if (!num(x)) return false; v.type = c; v.u = x; return true; }
        case 'd': { std::uint64_t x; if (!num(x)) return false; v.type = c; std::memcpy(&v.d, &x, 8); return true; }
        case 's': case 'o': {
            std::uint32_t len; if (!num(len) || !text(len, v.s)) return false;
            v.type = c; return c == 's' || valid_object_path(v.s);
        }
        case 'g': {
            if (pos >= n) return false;
            std::size_t len = p[pos++]; if (!text(len, v.s)) return false;
            v.type = c; return valid_signature(v.s);
        }
        case 'a': {
            std::uint32_t len; if (!num(len) || len > (1u << 26)) return false;
            std::string_view el = sig.substr(1);
            if (!pad(align_of(el[0])) || len > n - pos) return false;   // the padding before the first element is not counted in len
            const std::size_t end = pos + len;
            v.type = 'a'; v.s.assign(el);
            if (el == "y") { v.bytes.assign(reinterpret_cast<const char*>(p + pos), len); pos = end; return true; }
            reader sub{p, end, big, lim, pos, nodes};   // elements may not run past the array's own end
            while (sub.pos < end) { v.items.emplace_back(); if (!sub.get(el, v.items.back(), depth + 1)) return false; }
            pos = sub.pos; nodes = sub.nodes; return pos == end;
        }
        case '(': case '{': {
            if (!pad(8)) return false;
            v.type = c == '(' ? 'r' : 'e';
            for (std::size_t i = 1; i + 1 < sig.size();) {
                std::size_t len = type_len(sig, i, 1); if (!len) return false;
                v.items.emplace_back(); if (!get(sig.substr(i, len), v.items.back(), depth + 1)) return false;
                i += len;
            }
            return !v.items.empty();
        }
        case 'v': {
            if (pos >= n) return false;
            std::size_t len = p[pos++]; std::string sg; if (!text(len, sg) || !valid_single_type(sg)) return false;
            v.type = 'v'; v.items.emplace_back(); return get(sg, v.items.back(), depth + 1);
        }
        default: return false;
        }
    }
    // A sequence of complete types.
    bool get_all(std::string_view sig, std::vector<value>& out) {
        for (std::size_t i = 0; i < sig.size();) {
            std::size_t len = type_len(sig, i); if (!len) return false;
            out.emplace_back(); if (!get(sig.substr(i, len), out.back(), 0)) return false;
            i += len;
        }
        return true;
    }
};
}

// File descriptors that arrived with a message; whatever is not taken is closed with the last copy.
struct fd_bag {
    std::vector<int> fds;
    ~fd_bag() {
#if defined(FORGE_OS_POSIX)
        for (int f : fds) if (f >= 0) ::close(f);
#endif
    }
};
enum class msg_type : std::uint8_t { invalid = 0, method_call = 1, method_return = 2, error = 3, signal = 4 };
struct message {
    msg_type type = msg_type::invalid; std::uint8_t flags = 0;   // flags: 1 = no reply expected, 2 = no auto start
    std::uint32_t serial = 0, reply_serial = 0, unix_fds = 0;
    std::string path, iface, member, error_name, destination, sender, signature;
    std::vector<value> body;
    std::shared_ptr<fd_bag> fds;
    // The descriptor a value of type 'h' refers to; the caller owns it afterwards (-1 if absent / already taken).
    int take_fd(const value& h) const {
        if (h.type != 'h' || !fds || h.u >= fds->fds.size()) return -1;
        return std::exchange(fds->fds[std::size_t(h.u)], -1);
    }
    // First body value as text ("" if there is none): the human-readable part of an error reply.
    std::string text() const { return !body.empty() && body[0].is_text() ? body[0].s : std::string(); }
};

// Serializes m (m.serial must be non-zero; the signature is derived from the body). Empty error = ok.
inline result<std::string> encode(const message& m) {
    detail::writer body; std::string sig;
    for (auto& v : m.body) { sig += v.signature(); if (!body.put(v)) return {{}, errno_error(EINVAL, "dbus: value cannot be marshalled (type '" + std::string(1, v.type ? v.type : '?') + "')")}; }
    if (!valid_signature(sig)) return {{}, errno_error(EINVAL, "dbus: invalid body signature")};
    if (!m.serial || m.type == msg_type::invalid) return {{}, errno_error(EINVAL, "dbus: message without serial or type")};
    if (!m.path.empty() && !valid_object_path(m.path)) return {{}, errno_error(EINVAL, "dbus: invalid object path " + m.path)};
    std::vector<value> fields;
    auto field = [&](std::uint8_t code, value v) { fields.push_back(value::strukt({value::byte(code), value::variant(std::move(v))})); };
    if (!m.path.empty()) field(1, value::path(m.path));
    if (!m.iface.empty()) field(2, value::str(m.iface));
    if (!m.member.empty()) field(3, value::str(m.member));
    if (!m.error_name.empty()) field(4, value::str(m.error_name));
    if (m.reply_serial) field(5, value::u32(m.reply_serial));
    if (!m.destination.empty()) field(6, value::str(m.destination));
    if (!m.sender.empty()) field(7, value::str(m.sender));
    if (!sig.empty()) field(8, value::sig(sig));
    if (m.unix_fds) field(9, value::u32(m.unix_fds));
    detail::writer w;
    w.b.push_back('l'); w.b.push_back(char(m.type)); w.b.push_back(char(m.flags)); w.b.push_back(1);
    w.num<std::uint32_t>(std::uint32_t(body.b.size())); w.num<std::uint32_t>(m.serial);
    if (!w.put(value::array("(yv)", std::move(fields)))) return {{}, errno_error(EINVAL, "dbus: header cannot be marshalled")};
    w.pad(8); w.b += body.b;
    if (w.b.size() > (1u << 27)) return {{}, errno_error(EMSGSIZE, "dbus: message too large")};
    return {std::move(w.b), {}};
}
// How many bytes the message starting at p occupies, from its first 16 bytes: 0 = need more data, error = not a
// D-Bus message or over the limit.
inline result<std::size_t> message_size(const std::uint8_t* p, std::size_t n, const limits& lim = {}) {
    if (n < 16) return {0, {}};
    if ((p[0] != 'l' && p[0] != 'B') || p[3] != 1) return {0, errno_error(EPROTO, "dbus: bad message header")};
    const bool big = p[0] == 'B';
    auto u32 = [&](std::size_t at) { std::uint32_t v = 0; for (std::size_t i = 0; i < 4; ++i) v |= std::uint32_t(p[at + i]) << (8 * (big ? 3 - i : i)); return v; };
    std::uint64_t body = u32(4), fields = u32(12);
    if (fields > (1u << 26) || body > (1u << 27)) return {0, errno_error(EPROTO, "dbus: message lengths out of range")};
    std::uint64_t total = 16 + ((fields + 7) & ~std::uint64_t(7)) + body;
    if (total > lim.max_message) return {0, errno_error(EMSGSIZE, "dbus: message larger than the limit")};
    return {std::size_t(total), {}};
}
// Decodes exactly one message occupying [p, p + n).
inline result<message> decode(const std::uint8_t* p, std::size_t n, const limits& lim = {}) {
    auto bad = [](const char* why) { return result<message>{{}, errno_error(EPROTO, std::string("dbus: ") + why)}; };
    auto sz = message_size(p, n, lim); if (!sz) return {{}, sz.err};
    if (sz.value == 0 || sz.value != n) return bad("truncated or over-long message");
    const bool big = p[0] == 'B';
    message m; m.type = msg_type(p[1]); m.flags = p[2];
    if (p[1] == 0) return bad("invalid message type");
    detail::reader h{p, n, big, lim}; h.pos = 4;
    std::uint32_t body_len = 0; if (!h.num(body_len) || !h.num(m.serial) || !m.serial) return bad("bad serial");
    std::vector<value> fa;
    if (!h.get_all("a(yv)", fa) || fa.size() != 1) return bad("malformed header fields");
    if (!h.pad(8) || n - h.pos != body_len) return bad("header and body lengths disagree");
    for (auto& f : fa[0].items) {
        if (f.items.size() != 2 || f.items[1].items.size() != 1) return bad("malformed header field");
        const value& v = f.items[1].items[0]; const std::uint64_t code = f.items[0].u;
        auto want = [&](char t) { return v.type == t; };
        switch (code) {
        case 1: if (!want('o')) return bad("PATH is not an object path"); m.path = v.s; break;
        case 2: if (!want('s')) return bad("INTERFACE is not a string"); m.iface = v.s; break;
        case 3: if (!want('s')) return bad("MEMBER is not a string"); m.member = v.s; break;
        case 4: if (!want('s')) return bad("ERROR_NAME is not a string"); m.error_name = v.s; break;
        case 5: if (!want('u')) return bad("REPLY_SERIAL is not a uint32"); m.reply_serial = std::uint32_t(v.u); break;
        case 6: if (!want('s')) return bad("DESTINATION is not a string"); m.destination = v.s; break;
        case 7: if (!want('s')) return bad("SENDER is not a string"); m.sender = v.s; break;
        case 8: if (!want('g')) return bad("SIGNATURE is not a signature"); m.signature = v.s; break;
        case 9: if (!want('u')) return bad("UNIX_FDS is not a uint32"); m.unix_fds = std::uint32_t(v.u); break;
        default: break;   // unknown fields are ignored
        }
    }
    switch (m.type) {
    case msg_type::method_call: if (m.path.empty() || m.member.empty()) return bad("method call without path or member"); break;
    case msg_type::signal: if (m.path.empty() || m.member.empty() || m.iface.empty()) return bad("signal without path, interface or member"); break;
    case msg_type::error: if (m.error_name.empty() || !m.reply_serial) return bad("error without name or reply serial"); break;
    case msg_type::method_return: if (!m.reply_serial) return bad("return without reply serial"); break;
    default: break;   // unknown types must be ignored by the caller
    }
    detail::reader b{p + h.pos, body_len, big, lim}; b.nodes = h.nodes;
    if (!b.get_all(m.signature, m.body) || b.pos != body_len) return bad("body does not match its signature");
    return {std::move(m), {}};
}

// ---------------------------------------------------------------- addresses
struct transport { std::string kind, path; bool abstract = false; };
// "unix:path=/run/user/1000/bus,guid=..;unix:abstract=/tmp/x" -> entries in order (values %-unescaped).
inline std::vector<transport> parse_address(std::string_view addr) {
    std::vector<transport> out;
    auto unescape = [](std::string_view s) {
        std::string o; auto hex = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '%' && i + 2 < s.size() && hex(s[i + 1]) >= 0 && hex(s[i + 2]) >= 0) { o.push_back(char(hex(s[i + 1]) * 16 + hex(s[i + 2]))); i += 2; }
            else o.push_back(s[i]);
        }
        return o;
    };
    while (!addr.empty()) {
        auto semi = addr.find(';'); std::string_view one = addr.substr(0, semi);
        addr = semi == std::string_view::npos ? std::string_view() : addr.substr(semi + 1);
        auto colon = one.find(':'); if (colon == std::string_view::npos) continue;
        transport t; t.kind = std::string(one.substr(0, colon)); one.remove_prefix(colon + 1);
        while (!one.empty()) {
            auto comma = one.find(','); std::string_view kv = one.substr(0, comma);
            one = comma == std::string_view::npos ? std::string_view() : one.substr(comma + 1);
            auto eq = kv.find('='); if (eq == std::string_view::npos) continue;
            auto k = kv.substr(0, eq); auto v = unescape(kv.substr(eq + 1));
            if (k == "path") t.path = v; else if (k == "abstract") { t.path = v; t.abstract = true; }
        }
        out.push_back(std::move(t));
    }
    return out;
}

enum class bus { session, system };
// The address open(bus) would use; "" when there is no such bus on this machine.
inline std::string default_address(bus b) {
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS) && !defined(FORGE_OS_ANDROID)
    if (b == bus::session) {
        if (auto a = getenv("DBUS_SESSION_BUS_ADDRESS"); !a.empty()) return a;
        std::string rt = getenv("XDG_RUNTIME_DIR"); if (rt.empty()) rt = "/run/user/" + std::to_string(::getuid());
        struct stat st{}; if (::stat((rt + "/bus").c_str(), &st) == 0 && S_ISSOCK(st.st_mode)) return "unix:path=" + rt + "/bus";
        return {};
    }
    if (auto a = getenv("DBUS_SYSTEM_BUS_ADDRESS"); !a.empty()) return a;
    for (const char* p : {"/run/dbus/system_bus_socket", "/var/run/dbus/system_bus_socket"}) { struct stat st{}; if (::stat(p, &st) == 0 && S_ISSOCK(st.st_mode)) return std::string("unix:path=") + p; }
    return {};
#else
    (void)b; return {};
#endif
}

// ---------------------------------------------------------------- connection
class connection {
public:
    connection() = default;
    connection(const connection&) = delete; connection& operator=(const connection&) = delete;
    connection(connection&& o) noexcept { swap_(o); }
    connection& operator=(connection&& o) noexcept { if (this != &o) { close(); swap_(o); } return *this; }
    ~connection() { close(); }

    static result<connection> open(bus b) {
        std::string a = default_address(b);
        if (a.empty()) return {{}, unsupported(b == bus::session ? "D-Bus session bus (none running)" : "D-Bus system bus (none running)")};
        return open_address(a);
    }
    // Connects to the first usable transport of `address`, authenticates and says Hello.
    static result<connection> open_address(const std::string& address, int timeout_ms = 5000) {
#if defined(FORGE_OS_POSIX)
        error last = unsupported("D-Bus address without a unix transport (" + address + ")");
        for (auto& t : parse_address(address)) {
            if (t.kind != "unix" || t.path.empty()) continue;
            connection c; c.timeout_ms_ = timeout_ms;
            if (auto e = c.connect_(t)) { last = e; continue; }
            if (auto e = c.auth_()) { last = e; continue; }
            message hello; hello.type = msg_type::method_call; hello.destination = "org.freedesktop.DBus"; hello.path = "/org/freedesktop/DBus";
            hello.iface = "org.freedesktop.DBus"; hello.member = "Hello";
            auto r = c.call(hello, timeout_ms); if (!r) { last = r.err; continue; }
            c.name_ = r->text();
            return {std::move(c), {}};
        }
        return {{}, last};
#else
        (void)address; (void)timeout_ms; return {{}, unsupported("D-Bus")};
#endif
    }
    bool valid() const { return fd_ >= 0; }
    // The unique name the bus gave this connection (":1.42").
    const std::string& unique_name() const { return name_; }
    // True when the peer agreed to pass file descriptors (needed for replies of type 'h').
    bool passes_fds() const { return fd_passing_; }
    void set_limits(const limits& l) { lim_ = l; }
    void close() {
#if defined(FORGE_OS_POSIX)
        if (fd_ >= 0) ::close(fd_);
        for (int f : pending_fds_) ::close(f);
#endif
        fd_ = -1; pending_fds_.clear(); rx_.clear(); signals_.clear();
    }

    // Sends a method call and waits for its reply. An error reply becomes an error whose text is
    // "<error name>: <message>"; a service that is not there (ServiceUnknown, NameHasNoOwner, a failed
    // activation) reads as unsupported. last_error_name() has the D-Bus error name.
    result<message> call(message m, int timeout_ms = 25000) {
        last_error_name_.clear();
        if (fd_ < 0) return {{}, errno_error(ENOTCONN, "dbus: not connected")};
        m.type = msg_type::method_call; m.serial = next_serial_(); m.flags = std::uint8_t(m.flags & ~1u);
        if (auto e = send_(m)) return {{}, e};
        const std::uint64_t end = monotonic_ns() + std::uint64_t(timeout_ms < 0 ? 0 : timeout_ms) * 1000000ull;
        for (;;) {
            auto now = monotonic_ns(); if (now >= end) return {{}, errno_error(ETIMEDOUT, "dbus: no reply to " + m.iface + "." + m.member)};
            auto r = read_message_(int((end - now + 999999ull) / 1000000ull)); if (!r) return {{}, r.err};
            if (r->type == msg_type::signal) { queue_signal_(std::move(r.value)); continue; }
            if (r->reply_serial != m.serial) continue;   // not ours (a late reply to a call that timed out, a method call to us)
            if (r->type == msg_type::method_return) return r;
            if (r->type == msg_type::error) {
                last_error_name_ = r->error_name; std::string what = r->error_name + ": " + r->text();
                if (r->error_name == "org.freedesktop.DBus.Error.ServiceUnknown" || r->error_name == "org.freedesktop.DBus.Error.NameHasNoOwner" ||
                    r->error_name.rfind("org.freedesktop.DBus.Error.Spawn.", 0) == 0)
                    return {{}, error{unsupported_code, "dbus: " + what}};
                return {{}, errno_error(EIO, "dbus: " + what)};
            }
        }
    }
    result<message> call(const std::string& destination, const std::string& path, const std::string& iface, const std::string& member,
                         std::vector<value> args = {}, int timeout_ms = 25000) {
        message m; m.destination = destination; m.path = path; m.iface = iface; m.member = member; m.body = std::move(args);
        return call(std::move(m), timeout_ms);
    }
    // org.freedesktop.DBus.Properties.Get; the variant is unwrapped.
    result<value> get_property(const std::string& destination, const std::string& path, const std::string& iface, const std::string& name, int timeout_ms = 5000) {
        auto r = call(destination, path, "org.freedesktop.DBus.Properties", "Get", {value::str(iface), value::str(name)}, timeout_ms);
        if (!r) return {{}, r.err};
        if (r->body.empty()) return {{}, errno_error(EPROTO, "dbus: empty property reply")};
        return {r->body[0].unwrap(), {}};
    }
    // Sends without waiting (a signal, or a call whose reply is not wanted).
    status send(message m) {
        if (fd_ < 0) return {{}, errno_error(ENOTCONN, "dbus: not connected")};
        if (m.type == msg_type::invalid) m.type = msg_type::method_call;
        if (m.type == msg_type::method_call) m.flags |= 1;
        m.serial = next_serial_();
        if (auto e = send_(m)) return {{}, e};
        return {};
    }
    // Subscribes to signals: "type='signal',interface='org.freedesktop.portal.Settings'".
    status add_match(const std::string& rule) {
        auto r = call("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "AddMatch", {value::str(rule)}, timeout_ms_);
        if (!r) return {{}, r.err};
        return {};
    }
    // The next signal, waiting up to timeout_ms; ETIMEDOUT when none arrived.
    result<message> next_signal(int timeout_ms) {
        if (!signals_.empty()) { message m = std::move(signals_.front()); signals_.pop_front(); return {std::move(m), {}}; }
        if (fd_ < 0) return {{}, errno_error(ENOTCONN, "dbus: not connected")};
        const std::uint64_t end = monotonic_ns() + std::uint64_t(timeout_ms < 0 ? 0 : timeout_ms) * 1000000ull;
        for (;;) {
            auto now = monotonic_ns(); int left = now >= end ? 0 : int((end - now + 999999ull) / 1000000ull);
            auto r = read_message_(left); if (!r) return {{}, r.err};
            if (r->type == msg_type::signal) return r;
            if (now >= end) return {{}, errno_error(ETIMEDOUT, "dbus: no signal")};
        }
    }
    const std::string& last_error_name() const { return last_error_name_; }
    // For an event loop: readable when a message may be waiting.
    int native() const { return fd_; }

private:
    void swap_(connection& o) noexcept {
        std::swap(fd_, o.fd_); std::swap(serial_, o.serial_); std::swap(name_, o.name_); std::swap(rx_, o.rx_); std::swap(signals_, o.signals_);
        std::swap(pending_fds_, o.pending_fds_); std::swap(fd_passing_, o.fd_passing_); std::swap(lim_, o.lim_); std::swap(timeout_ms_, o.timeout_ms_);
        std::swap(last_error_name_, o.last_error_name_);
    }
    std::uint32_t next_serial_() { if (++serial_ == 0) ++serial_; return serial_; }
    void queue_signal_(message m) { if (signals_.size() >= 1024) signals_.pop_front(); signals_.push_back(std::move(m)); }
#if defined(FORGE_OS_POSIX)
    error connect_(const transport& t) {
        sockaddr_un sa{}; sa.sun_family = AF_UNIX; socklen_t len = 0;
        if (t.abstract) {
#if defined(__linux__)
            if (t.path.size() + 1 > sizeof sa.sun_path) return errno_error(ENAMETOOLONG, "dbus: socket name too long");
            std::memcpy(sa.sun_path + 1, t.path.data(), t.path.size()); len = socklen_t(offsetof(sockaddr_un, sun_path) + 1 + t.path.size());
#else
            return unsupported("dbus: abstract sockets");
#endif
        } else {
            if (t.path.size() >= sizeof sa.sun_path) return errno_error(ENAMETOOLONG, "dbus: socket path too long");
            std::memcpy(sa.sun_path, t.path.data(), t.path.size()); len = socklen_t(offsetof(sockaddr_un, sun_path) + t.path.size() + 1);
        }
        int fd = cloexec_socket(AF_UNIX, SOCK_STREAM); if (fd < 0) return last_error("socket");
#if defined(__APPLE__)
        { int one = 1; ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one); }
#endif
        int r; do r = ::connect(fd, reinterpret_cast<sockaddr*>(&sa), len); while (r != 0 && errno == EINTR);
        if (r != 0) { int e = errno; ::close(fd); if (e == ENOENT || e == ECONNREFUSED) return unsupported("D-Bus (" + t.path + ": " + error_text(e) + ")"); return {e, "dbus: connect " + t.path + ": " + error_text(e)}; }
        fd_ = fd; return {};
    }
    error write_all_(const char* p, std::size_t n) {
        while (n) {
#if defined(MSG_NOSIGNAL)
            auto r = ::send(fd_, p, n, MSG_NOSIGNAL);
#else
            auto r = ::send(fd_, p, n, 0);
#endif
            if (r < 0) { if (errno == EINTR) continue; return last_error("dbus: send"); }
            p += r; n -= std::size_t(r);
        }
        return {};
    }
    // Reads more bytes (and any descriptors) into rx_. ETIMEDOUT when nothing came; ECONNRESET at end of stream.
    error fill_(int timeout_ms) {
        struct pollfd pf{fd_, POLLIN, 0}; int pr;
        do pr = ::poll(&pf, 1, timeout_ms); while (pr < 0 && errno == EINTR);
        if (pr < 0) return last_error("dbus: poll");
        if (pr == 0) return errno_error(ETIMEDOUT, "dbus: timed out");
        char buf[4096]; alignas(cmsghdr) char ctl[CMSG_SPACE(16 * sizeof(int))];
        iovec iov{buf, sizeof buf}; msghdr mh{}; mh.msg_iov = &iov; mh.msg_iovlen = 1; mh.msg_control = ctl; mh.msg_controllen = sizeof ctl;
        int flags = 0;
#if defined(MSG_CMSG_CLOEXEC)
        flags = MSG_CMSG_CLOEXEC;
#endif
        ssize_t r; do r = ::recvmsg(fd_, &mh, flags); while (r < 0 && errno == EINTR);
        if (r < 0) return last_error("dbus: recv");
        for (cmsghdr* c = CMSG_FIRSTHDR(&mh); c; c = CMSG_NXTHDR(&mh, c))
            if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS && c->cmsg_len >= CMSG_LEN(0)) {
                std::size_t cnt = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
                for (std::size_t i = 0; i < cnt; ++i) { int f; std::memcpy(&f, CMSG_DATA(c) + i * sizeof(int), sizeof f); ::fcntl(f, F_SETFD, FD_CLOEXEC); if (pending_fds_.size() < 64) pending_fds_.push_back(f); else ::close(f); }
            }
        if (r == 0) return errno_error(ECONNRESET, "dbus: connection closed by the peer");
        rx_.append(buf, std::size_t(r)); return {};
    }
    error auth_() {
        std::string uid = std::to_string(::getuid()), hex;
        for (char c : uid) { static const char* d = "0123456789abcdef"; hex.push_back(d[(c >> 4) & 15]); hex.push_back(d[c & 15]); }
        std::string hello = std::string(1, '\0') + "AUTH EXTERNAL " + hex + "\r\n";
        if (auto e = write_all_(hello.data(), hello.size())) return e;
        auto line = [&](std::string& out) -> error {   // one CRLF line, bounded
            for (;;) {
                auto nl = rx_.find("\r\n"); if (nl != std::string::npos) { out = rx_.substr(0, nl); rx_.erase(0, nl + 2); return {}; }
                if (rx_.size() > 4096) return errno_error(EPROTO, "dbus: authentication line too long");
                if (auto e = fill_(timeout_ms_)) return e;
            }
        };
        std::string l; if (auto e = line(l)) return e;
        if (l.rfind("OK", 0) != 0) return errno_error(EACCES, "dbus: authentication rejected (" + l.substr(0, 80) + ")");
        if (auto e = write_all_("NEGOTIATE_UNIX_FD\r\n", 19)) return e;
        if (auto e = line(l)) return e;
        fd_passing_ = l.rfind("AGREE_UNIX_FD", 0) == 0;
        return write_all_("BEGIN\r\n", 7);
    }
    error send_(const message& m) {
        auto b = encode(m); if (!b) return b.err;
        return write_all_(b->data(), b->size());
    }
    result<message> read_message_(int timeout_ms) {
        const std::uint64_t end = monotonic_ns() + std::uint64_t(timeout_ms < 0 ? 0 : timeout_ms) * 1000000ull;
        for (;;) {
            auto sz = message_size(reinterpret_cast<const std::uint8_t*>(rx_.data()), rx_.size(), lim_);
            if (!sz) { close(); return {{}, sz.err}; }   // the stream cannot be resynchronised
            if (sz.value && rx_.size() >= sz.value) {
                auto m = decode(reinterpret_cast<const std::uint8_t*>(rx_.data()), sz.value, lim_);
                rx_.erase(0, sz.value);
                if (!m) { close(); return {{}, m.err}; }
                if (m->unix_fds) {
                    if (m->unix_fds > pending_fds_.size()) { close(); return {{}, errno_error(EPROTO, "dbus: message announces descriptors that did not arrive")}; }
                    m->fds = std::make_shared<fd_bag>();
                    m->fds->fds.assign(pending_fds_.begin(), pending_fds_.begin() + std::ptrdiff_t(m->unix_fds));
                    pending_fds_.erase(pending_fds_.begin(), pending_fds_.begin() + std::ptrdiff_t(m->unix_fds));
                }
                return m;
            }
            auto now = monotonic_ns();
            if (auto e = fill_(now >= end ? 0 : int((end - now + 999999ull) / 1000000ull))) return {{}, e};
        }
    }
#else
    error send_(const message&) { return unsupported("D-Bus"); }
    result<message> read_message_(int) { return {{}, unsupported("D-Bus")}; }
#endif
    int fd_ = -1; std::uint32_t serial_ = 0; std::string name_, rx_, last_error_name_; std::deque<message> signals_;
    std::vector<int> pending_fds_; bool fd_passing_ = false; limits lim_; int timeout_ms_ = 5000;
};

} // namespace forge::os::dbus

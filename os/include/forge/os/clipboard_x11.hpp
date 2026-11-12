#pragma once
// forge::os::clipboard::x11: the X11 CLIPBOARD selection spoken directly (libxcb loaded at run time), so Linux
// and FreeBSD need no xclip / xsel. clipboard.hpp uses it whenever DISPLAY is set.
//   get(): asks the selection owner for UTF8_STRING and reads the answer, including INCR (chunked) transfers.
//   set(): X11 has no clipboard store: the owner answers every paste. A background thread of this process owns
//   the selection and serves requests (TARGETS, UTF8_STRING, text/plain;charset=utf-8, STRING, TEXT; INCR for
//   large texts) until another program copies something or the process exits. A clipboard manager, which most
//   desktops run, keeps the text beyond that; without one it is gone when the process ends.
// Data from the X server and from other clients is treated as untrusted: lengths are bounded (max_bytes).
#include <forge/os/detail.hpp>
#include <type_traits>

#if (defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)) && !defined(FORGE_OS_ANDROID)
namespace forge::os::clipboard::x11 {
inline constexpr std::size_t max_bytes = 256u << 20;   // refuse to buffer more than this from a selection owner

namespace detail_x {
struct cookie { unsigned seq; };
struct screen_iter { std::uint32_t* data; int rem; int index; };   // xcb_screen_iterator_t; a screen starts with its root window id
struct atom_reply { std::uint8_t type, pad; std::uint16_t seq; std::uint32_t length, atom; };
struct prop_reply { std::uint8_t type, format; std::uint16_t seq; std::uint32_t length, ptype, bytes_after, value_len; std::uint8_t pad[12]; };
struct owner_reply { std::uint8_t type, pad; std::uint16_t seq; std::uint32_t length, owner; };
struct event { std::uint8_t type, detail; std::uint16_t seq; std::uint32_t f[7]; };   // 32 bytes on the wire
inline unsigned property_state(const event* e) { return reinterpret_cast<const std::uint8_t*>(&e->f[3])[0]; }   // PropertyNotify: 0 new value, 1 deleted
struct api {
    void* (*connect)(const char*, int*) = nullptr; int (*has_error)(void*) = nullptr; void (*disconnect)(void*) = nullptr;
    const void* (*get_setup)(void*) = nullptr; screen_iter (*roots)(const void*) = nullptr; std::uint32_t (*generate_id)(void*) = nullptr;
    cookie (*create_window)(void*, std::uint8_t, std::uint32_t, std::uint32_t, std::int16_t, std::int16_t, std::uint16_t, std::uint16_t, std::uint16_t, std::uint16_t, std::uint32_t, std::uint32_t, const void*) = nullptr;
    cookie (*destroy_window)(void*, std::uint32_t) = nullptr;
    cookie (*intern_atom)(void*, std::uint8_t, std::uint16_t, const char*) = nullptr; atom_reply* (*intern_atom_reply)(void*, cookie, void**) = nullptr;
    cookie (*convert_selection)(void*, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) = nullptr;
    int (*flush)(void*) = nullptr; event* (*poll_for_event)(void*) = nullptr; int (*get_fd)(void*) = nullptr;
    cookie (*get_property)(void*, std::uint8_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) = nullptr;
    prop_reply* (*get_property_reply)(void*, cookie, void**) = nullptr; void* (*property_value)(const prop_reply*) = nullptr; int (*property_value_length)(const prop_reply*) = nullptr;
    cookie (*change_property)(void*, std::uint8_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint8_t, std::uint32_t, const void*) = nullptr;
    cookie (*delete_property)(void*, std::uint32_t, std::uint32_t) = nullptr;
    cookie (*set_selection_owner)(void*, std::uint32_t, std::uint32_t, std::uint32_t) = nullptr;
    cookie (*get_selection_owner)(void*, std::uint32_t) = nullptr; owner_reply* (*get_selection_owner_reply)(void*, cookie, void**) = nullptr;
    cookie (*send_event)(void*, std::uint8_t, std::uint32_t, std::uint32_t, const char*) = nullptr;
    cookie (*change_window_attributes)(void*, std::uint32_t, std::uint32_t, const void*) = nullptr;
    std::uint32_t (*max_request_length)(void*) = nullptr;
    bool ok = false;
    static const api& get() {
        static api a = [] {
            api x; void* h = ::dlopen("libxcb.so.1", RTLD_NOW | RTLD_LOCAL); if (!h) h = ::dlopen("libxcb.so", RTLD_NOW | RTLD_LOCAL);
            if (!h) return x;
            bool all = true;
            auto sym = [&](auto& fn, const char* name) { fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(::dlsym(h, name)); all = all && fn != nullptr; };
            sym(x.connect, "xcb_connect"); sym(x.has_error, "xcb_connection_has_error"); sym(x.disconnect, "xcb_disconnect"); sym(x.get_setup, "xcb_get_setup");
            sym(x.roots, "xcb_setup_roots_iterator"); sym(x.generate_id, "xcb_generate_id"); sym(x.create_window, "xcb_create_window"); sym(x.destroy_window, "xcb_destroy_window");
            sym(x.intern_atom, "xcb_intern_atom"); sym(x.intern_atom_reply, "xcb_intern_atom_reply"); sym(x.convert_selection, "xcb_convert_selection"); sym(x.flush, "xcb_flush");
            sym(x.poll_for_event, "xcb_poll_for_event"); sym(x.get_fd, "xcb_get_file_descriptor"); sym(x.get_property, "xcb_get_property"); sym(x.get_property_reply, "xcb_get_property_reply");
            sym(x.property_value, "xcb_get_property_value"); sym(x.property_value_length, "xcb_get_property_value_length"); sym(x.change_property, "xcb_change_property");
            sym(x.delete_property, "xcb_delete_property"); sym(x.set_selection_owner, "xcb_set_selection_owner"); sym(x.get_selection_owner, "xcb_get_selection_owner");
            sym(x.get_selection_owner_reply, "xcb_get_selection_owner_reply"); sym(x.send_event, "xcb_send_event"); sym(x.change_window_attributes, "xcb_change_window_attributes");
            sym(x.max_request_length, "xcb_get_maximum_request_length");
            x.ok = all; return x;
        }();
        return a;
    }
};
// One connection with a small unmapped window that receives property events.
struct session {
    const api& x = api::get(); void* c = nullptr; std::uint32_t win = 0, root = 0;
    std::uint32_t a_clipboard = 0, a_utf8 = 0, a_incr = 0, a_prop = 0, a_targets = 0, a_text = 0, a_mime = 0;
    error open() {
        if (!x.ok) return unsupported("clipboard (libxcb is not installed)");
        if (getenv("DISPLAY").empty()) return unsupported("clipboard (no X11 display)");
        int scr = 0; c = x.connect(nullptr, &scr);
        if (!c || x.has_error(c)) { if (c) x.disconnect(c); c = nullptr; return unsupported("clipboard (cannot connect to the X server)"); }
        auto it = x.roots(x.get_setup(c)); if (!it.data || it.rem <= 0) { close(); return unsupported("clipboard (the X server has no screen)"); }
        root = *it.data; win = x.generate_id(c);
        const std::uint32_t mask = 0x00400000;   // PropertyChange
        x.create_window(c, 0 /*copy depth*/, win, root, -10, -10, 1, 1, 0, 1 /*InputOutput*/, 0 /*copy visual*/, 0x0800 /*CW_EVENT_MASK*/, &mask);
        a_clipboard = atom("CLIPBOARD"); a_utf8 = atom("UTF8_STRING"); a_incr = atom("INCR"); a_prop = atom("FORGE_SELECTION"); a_targets = atom("TARGETS");
        a_text = atom("TEXT"); a_mime = atom("text/plain;charset=utf-8");
        if (!a_clipboard || !a_utf8 || !a_incr || !a_prop || !a_targets) { close(); return error{EIO, "clipboard: the X server refused to intern atoms"}; }
        return {};
    }
    std::uint32_t atom(const char* name) {
        auto* r = x.intern_atom_reply(c, x.intern_atom(c, 0, std::uint16_t(std::strlen(name)), name), nullptr);
        std::uint32_t a = r ? r->atom : 0; std::free(r); return a;
    }
    void close() { if (c) { if (win) x.destroy_window(c, win); x.flush(c); x.disconnect(c); } c = nullptr; win = 0; }
    // The next event, waiting until `deadline_ns`; nullptr on timeout or a broken connection. Caller frees.
    event* next(std::uint64_t deadline_ns) {
        for (;;) {
            if (event* e = x.poll_for_event(c)) return e;
            if (x.has_error(c)) return nullptr;
            auto now = monotonic_ns(); if (now >= deadline_ns) return nullptr;
            struct pollfd p{x.get_fd(c), POLLIN, 0}; (void)::poll(&p, 1, int(std::min<std::uint64_t>((deadline_ns - now + 999999ull) / 1000000ull, 100)));
        }
    }
    // Reads the whole property (and deletes it). False when it is over max_bytes or the server did not answer.
    bool read_property(std::uint32_t w, std::uint32_t prop, std::string& out, std::uint32_t& type) {
        std::uint32_t off = 0; type = 0;
        for (;;) {
            auto* r = x.get_property_reply(c, x.get_property(c, 0, w, prop, 0 /*any type*/, off, 1u << 18 /*1 MiB per round*/), nullptr);
            if (!r) return false;
            type = r->ptype; int len = x.property_value_length(r); std::uint32_t after = r->bytes_after;
            if (len > 0 && r->format == 8) { if (out.size() + std::size_t(len) > max_bytes) { std::free(r); return false; } out.append(static_cast<const char*>(x.property_value(r)), std::size_t(len)); }
            else if (len > 0 && r->ptype == a_incr) { /* the INCR size hint: not text */ }
            std::free(r);
            if (!after || len <= 0) break;
            off += std::uint32_t(len) / 4;
        }
        x.delete_property(c, w, prop); x.flush(c);
        return true;
    }
};
// The process-wide selection owner.
struct owner {
    std::mutex m; std::shared_ptr<const std::string> text; std::unique_ptr<session> s; std::thread th; std::atomic<bool> stop{false}, owning{false};
    struct transfer { std::uint32_t requestor, prop, target; std::shared_ptr<const std::string> data; std::size_t off = 0; };
    std::vector<transfer> incr;
    static owner& instance() { static owner o; return o; }
    ~owner() { shutdown(); }
    void shutdown() { stop = true; if (th.joinable()) th.join(); if (s) s->close(); s.reset(); owning = false; }
    status take(const std::string& t) {
        std::lock_guard<std::mutex> g(m);
        if (!s || (s->c && s->x.has_error(s->c))) {   // first use, or the server went away
            stop = true; if (th.joinable()) th.join(); if (s) s->close();
            s = std::make_unique<session>(); if (auto e = s->open()) { s.reset(); return {{}, e}; }
            stop = false; th = std::thread([this] { serve(); });
        }
        { std::lock_guard<std::mutex> tg(tm_); text = std::make_shared<const std::string>(t); }
        auto& x = s->x;
        x.set_selection_owner(s->c, s->win, s->a_clipboard, 0 /*CurrentTime*/);
        auto* r = x.get_selection_owner_reply(s->c, x.get_selection_owner(s->c, s->a_clipboard), nullptr);
        bool mine = r && r->owner == s->win; std::free(r);
        if (!mine) return {{}, error{EBUSY, "clipboard: could not take ownership of the X11 selection"}};
        owning = true; return {};
    }
    void notify(std::uint32_t requestor, std::uint32_t selection, std::uint32_t target, std::uint32_t prop, std::uint32_t time) {
        event e{}; e.type = 31 /*SelectionNotify*/; e.f[0] = time; e.f[1] = requestor; e.f[2] = selection; e.f[3] = target; e.f[4] = prop;
        s->x.send_event(s->c, 0, requestor, 0, reinterpret_cast<const char*>(&e));
    }
    void serve() {
        auto& x = s->x; void* c = s->c;
        while (!stop) {
            event* e = s->next(monotonic_ns() + 50'000'000ull);
            if (!e) { if (x.has_error(c)) { owning = false; return; } continue; }
            const int type = e->type & 0x7f;
            if (type == 29 /*SelectionClear*/) { owning = false; std::lock_guard<std::mutex> tg(tm_); text.reset(); incr.clear(); }
            else if (type == 30 /*SelectionRequest*/) {
                const std::uint32_t time = e->f[0], requestor = e->f[2], selection = e->f[3], target = e->f[4]; std::uint32_t prop = e->f[5] ? e->f[5] : target;
                std::shared_ptr<const std::string> t; { std::lock_guard<std::mutex> tg(tm_); t = text; }
                if (!t || selection != s->a_clipboard) notify(requestor, selection, target, 0, time);
                else if (target == s->a_targets) {
                    std::uint32_t list[5] = {s->a_targets, s->a_utf8, s->a_mime, 31 /*STRING*/, s->a_text}; std::uint32_t n = 0; std::uint32_t out[5];
                    for (auto a : list) if (a) out[n++] = a;
                    x.change_property(c, 0 /*replace*/, requestor, prop, 4 /*ATOM*/, 32, n, out); notify(requestor, selection, target, prop, time);
                } else if (target == s->a_utf8 || target == 31 /*STRING*/ || (s->a_text && target == s->a_text) || (s->a_mime && target == s->a_mime)) {
                    const std::uint32_t reply_type = target == s->a_text ? s->a_utf8 : target;
                    const std::size_t limit = std::min<std::size_t>(std::size_t(x.max_request_length(c)) * 4 - 64, 1u << 20);
                    if (t->size() <= limit) x.change_property(c, 0, requestor, prop, reply_type, 8, std::uint32_t(t->size()), t->data());
                    else {   // INCR: announce, then one chunk each time the requestor deletes the property
                        const std::uint32_t ev_mask = 0x00400000; x.change_window_attributes(c, requestor, 0x0800, &ev_mask);
                        std::uint32_t size = std::uint32_t(std::min<std::size_t>(t->size(), 0x7fffffffu));
                        x.change_property(c, 0, requestor, prop, s->a_incr, 32, 1, &size);
                        if (incr.size() < 64) incr.push_back({requestor, prop, reply_type, t, 0});
                    }
                    notify(requestor, selection, target, prop, time);
                } else notify(requestor, selection, target, 0, time);   // a target we do not offer
                x.flush(c);
            } else if (type == 28 /*PropertyNotify*/ && property_state(e) == 1 /*Delete*/) {
                const std::uint32_t w = e->f[0], atom = e->f[1];
                for (std::size_t i = 0; i < incr.size(); ++i) if (incr[i].requestor == w && incr[i].prop == atom) {
                    auto& tr = incr[i]; const std::size_t chunk = std::min<std::size_t>(tr.data->size() - tr.off, 256u << 10);
                    x.change_property(c, 0, w, atom, tr.target, 8, std::uint32_t(chunk), tr.data->data() + tr.off);   // a zero-length chunk ends the transfer
                    tr.off += chunk; if (chunk == 0) incr.erase(incr.begin() + std::ptrdiff_t(i));
                    x.flush(c); break;
                }
            }
            std::free(e);
        }
    }
    std::mutex tm_;
};
}

// True when an X server is reachable and libxcb is installed.
inline bool available() {
    if (getenv("DISPLAY").empty() || !detail_x::api::get().ok) return false;
    detail_x::session s; if (s.open()) return false;
    s.close(); return true;
}
// True while this process owns the selection (what set() stored is what others paste).
inline bool owned() { return detail_x::owner::instance().owning.load(); }
// The clipboard text; "" when nothing is copied or the owner offers no text.
inline result<std::string> get(int timeout_ms = 3000) {
    detail_x::session s; if (auto e = s.open()) return {{}, e};
    struct closer { detail_x::session& s; ~closer() { s.close(); } } guard{s};
    auto& x = s.x; const std::uint64_t deadline = monotonic_ns() + std::uint64_t(timeout_ms < 0 ? 0 : timeout_ms) * 1000000ull;
    x.convert_selection(s.c, s.win, s.a_clipboard, s.a_utf8, s.a_prop, 0 /*CurrentTime*/); x.flush(s.c);
    std::string out; bool notified = false;
    while (!notified) {
        detail_x::event* e = s.next(deadline);
        if (!e) return {{}, errno_error(ETIMEDOUT, "clipboard: the selection owner did not answer")};
        if ((e->type & 0x7f) == 31 /*SelectionNotify*/ && e->f[1] == s.win) { notified = true; if (e->f[4] == 0) { std::free(e); return {std::string(), {}}; } }   // property None: nothing to paste
        std::free(e);
    }
    std::uint32_t type = 0;
    if (!s.read_property(s.win, s.a_prop, out, type)) return {{}, error{EIO, "clipboard: could not read the selection"}};
    if (type != s.a_incr) return {std::move(out), {}};
    out.clear();   // INCR: deleting the property (done above) started the transfer; each new value is one chunk
    for (;;) {
        detail_x::event* e = s.next(deadline);
        if (!e) return {{}, errno_error(ETIMEDOUT, "clipboard: the incremental transfer stalled")};
        const bool fresh = (e->type & 0x7f) == 28 /*PropertyNotify*/ && e->f[0] == s.win && e->f[1] == s.a_prop && detail_x::property_state(e) == 0 /*NewValue*/;
        std::free(e); if (!fresh) continue;
        std::string chunk; if (!s.read_property(s.win, s.a_prop, chunk, type)) return {{}, error{EFBIG, "clipboard: the selection is too large or unreadable"}};
        if (chunk.empty()) return {std::move(out), {}};
        if (out.size() + chunk.size() > max_bytes) return {{}, error{EFBIG, "clipboard: the selection is too large"}};
        out += chunk;
    }
}
// Offers `text` as the clipboard from this process (see the top of this file for how long it lasts).
inline status set(const std::string& text) {
    if (text.size() > max_bytes) return {{}, errno_error(EFBIG, "clipboard: text too large")};
    return detail_x::owner::instance().take(text);
}
// Stops serving the selection (the clipboard becomes empty unless a clipboard manager kept a copy).
inline void release() { detail_x::owner::instance().shutdown(); }

} // namespace forge::os::clipboard::x11
#endif

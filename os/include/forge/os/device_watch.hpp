#pragma once
// forge::os::devices::monitor: device hot-plug notifications (USB devices, disks, network adapters, ...).
//   Linux/Android: the kernel's uevent netlink socket (NETLINK_KOBJECT_UEVENT, multicast group 1; only messages
//                  sent by the kernel are accepted). Needs no udev. Refused in some containers and for Android
//                  apps: polling then.
//   FreeBSD:       devd's seqpacket socket /var/run/devd.seqpacket.pipe (attach "+", detach "-", notify "!").
//   macOS:         IOKit matching notifications (IOServiceAddMatchingNotification for IOUSBHostDevice and IOMedia
//                  on a private dispatch queue), late-bound like the rest of the layer.
//   Windows:       CM_Register_Notification for every device interface class (cfgmgr32, Windows 8+).
//   Fallback (and force_polling): devices::usb() + devices::block() compared every poll_interval_ms.
//   iOS:           unsupported (no device registry access).
// poll() returns the events since the last call. Subsystems are spelled the Linux way ("usb", "block", "net",
// "pci", "input", ...; "other" when the OS does not say).
#include <forge/os/devices.hpp>
#if defined(__linux__)
#include <linux/netlink.h>
#elif defined(FORGE_OS_FREEBSD)
#include <sys/un.h>
#endif

namespace forge::os::devices {
enum class device_action { added, removed, changed };
inline const char* device_action_name(device_action a) { return a == device_action::added ? "added" : a == device_action::removed ? "removed" : "changed"; }
struct device_event { device_action action = device_action::changed; std::string subsystem, name; };
struct monitor_options { bool force_polling = false; int poll_interval_ms = 2000; };

namespace detail_dev {
// A kernel uevent datagram: "add@/devices/..\0ACTION=add\0DEVPATH=..\0SUBSYSTEM=usb\0DEVNAME=bus/usb/001/004\0...".
// false for udev's own "libudev" messages and anything without ACTION / DEVPATH.
inline bool parse_uevent(std::string_view msg, device_event& ev) {
    if (msg.size() >= 8 && msg.compare(0, 8, std::string_view("libudev\0", 8)) == 0) return false;
    std::string action, devpath, subsystem, devname; std::size_t i = msg.find('\0'); if (i == std::string_view::npos) return false;
    for (++i; i < msg.size();) {
        std::size_t e = msg.find('\0', i); if (e == std::string_view::npos) e = msg.size();
        std::string_view kv = msg.substr(i, e - i); i = e + 1;
        auto eq = kv.find('='); if (eq == std::string_view::npos) continue;
        auto k = kv.substr(0, eq), v = kv.substr(eq + 1);
        if (k == "ACTION") action = v; else if (k == "DEVPATH") devpath = v; else if (k == "SUBSYSTEM") subsystem = v; else if (k == "DEVNAME") devname = v;
    }
    if (action.empty() || devpath.empty()) return false;
    ev.action = action == "add" ? device_action::added : action == "remove" ? device_action::removed : device_action::changed;
    if (action == "bind" || action == "unbind") return false;   // driver binding: the add / remove already told the story
    ev.subsystem = subsystem.empty() ? "other" : subsystem; ev.name = !devname.empty() ? devname : devpath.substr(devpath.rfind('/') + 1);
    return true;
}
// One devd message: "+ugen0.2 at bus=0 ..." (attach), "-ugen0.2 at ..." (detach), "!system=USB subsystem=DEVICE
// type=ATTACH ugen=ugen0.2 ..." (notify). Only attach / detach lines and DEVFS create / destroy notifications
// become events.
inline bool parse_devd(std::string_view msg, device_event& ev) {
    while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\0')) msg.remove_suffix(1);
    if (msg.size() < 2) return false;
    auto field = [&](std::string_view key) -> std::string {
        std::string k = std::string(key) + "="; std::size_t p = 0;
        while ((p = msg.find(k, p)) != std::string_view::npos) { if (p == 1 || msg[p - 1] == ' ') { auto e = msg.find(' ', p); return std::string(msg.substr(p + k.size(), (e == std::string_view::npos ? msg.size() : e) - p - k.size())); } ++p; }
        return {};
    };
    auto subsystem_of = [](const std::string& n) -> std::string {
        if (n.rfind("ugen", 0) == 0 || n.rfind("uhub", 0) == 0 || n.rfind("umass", 0) == 0 || n.rfind("usbus", 0) == 0) return "usb";
        if (n.rfind("da", 0) == 0 || n.rfind("ada", 0) == 0 || n.rfind("nvd", 0) == 0 || n.rfind("nda", 0) == 0 || n.rfind("cd", 0) == 0 || n.rfind("mmcsd", 0) == 0) return "block";
        return "other";
    };
    if (msg[0] == '+' || msg[0] == '-') {
        auto sp = msg.find(' '); ev.name = std::string(msg.substr(1, sp == std::string_view::npos ? std::string_view::npos : sp - 1));
        if (ev.name.empty()) return false;
        ev.action = msg[0] == '+' ? device_action::added : device_action::removed; ev.subsystem = subsystem_of(ev.name); return true;
    }
    if (msg[0] == '!') {
        std::string sys = field("system"), type = field("type");
        if (sys == "DEVFS" && (type == "CREATE" || type == "DESTROY")) {
            ev.name = field("cdev"); if (ev.name.empty()) return false;
            ev.action = type == "CREATE" ? device_action::added : device_action::removed; ev.subsystem = subsystem_of(ev.name); return true;
        }
        if (sys == "IFNET" && (type == "ATTACH" || type == "DETACH")) { ev.name = field("subsystem"); ev.subsystem = "net"; ev.action = type == "ATTACH" ? device_action::added : device_action::removed; return !ev.name.empty(); }
    }
    return false;
}
// What changed between two enumerations (names only).
inline void diff_names(const std::vector<std::string>& a, const std::vector<std::string>& b, const char* subsystem, std::vector<device_event>& out) {
    for (auto& n : b) if (std::find(a.begin(), a.end(), n) == a.end()) out.push_back({device_action::added, subsystem, n});
    for (auto& n : a) if (std::find(b.begin(), b.end(), n) == b.end()) out.push_back({device_action::removed, subsystem, n});
}
struct queue {   // events pushed from OS callback threads
    std::mutex m; std::condition_variable cv; std::vector<device_event> q;
    void push(device_event e) { { std::lock_guard<std::mutex> g(m); if (q.size() < 4096) q.push_back(std::move(e)); } cv.notify_all(); }
    std::vector<device_event> take(int timeout_ms) {
        std::unique_lock<std::mutex> l(m);
        if (q.empty() && timeout_ms != 0) { if (timeout_ms < 0) cv.wait(l, [&] { return !q.empty(); }); else cv.wait_for(l, std::chrono::milliseconds(timeout_ms), [&] { return !q.empty(); }); }
        std::vector<device_event> out; out.swap(q); return out;
    }
};
#if defined(FORGE_OS_WINDOWS)
struct cm_filter { DWORD cbSize, Flags; int FilterType; DWORD Reserved; union { GUID ClassGuid; HANDLE hTarget; wchar_t InstanceId[200]; } u; };
struct cm_event_data { int FilterType; DWORD Reserved; GUID ClassGuid; wchar_t SymbolicLink[1]; };
inline const char* class_subsystem(const GUID& g) {
    static const GUID usb{0xA5DCBF10, 0x6530, 0x11D2, {0x90, 0x1F, 0x00, 0xC0, 0x4F, 0xB9, 0x51, 0xED}}, disk{0x53F56307, 0xB6BF, 0x11D0, {0x94, 0xF2, 0x00, 0xA0, 0xC9, 0x1E, 0xFB, 0x8B}},
        volume{0x53F5630D, 0xB6BF, 0x11D0, {0x94, 0xF2, 0x00, 0xA0, 0xC9, 0x1E, 0xFB, 0x8B}}, netif{0xCAC88484, 0x7515, 0x4C03, {0x82, 0xE6, 0x71, 0xA8, 0x7A, 0xBA, 0xC3, 0x61}},
        hid{0x4D1E55B2, 0xF16F, 0x11CF, {0x88, 0xCB, 0x00, 0x11, 0x11, 0x00, 0x00, 0x30}};
    if (IsEqualGUID(g, usb)) return "usb";
    if (IsEqualGUID(g, disk) || IsEqualGUID(g, volume)) return "block";
    if (IsEqualGUID(g, netif)) return "net";
    if (IsEqualGUID(g, hid)) return "input";
    return "other";
}
inline DWORD CALLBACK cm_callback(void*, void* ctx, int action, cm_event_data* data, DWORD size) {
    if (!ctx || !data || data->FilterType != 0 || (action != 0 && action != 1) || size < sizeof(cm_event_data)) return ERROR_SUCCESS;
    device_event e; e.action = action == 0 ? device_action::added : device_action::removed; e.subsystem = class_subsystem(data->ClassGuid);
    std::size_t max = (size - offsetof(cm_event_data, SymbolicLink)) / sizeof(wchar_t), n = 0; while (n < max && data->SymbolicLink[n]) ++n;
    e.name = narrow(std::wstring_view(data->SymbolicLink, n));
    static_cast<queue*>(ctx)->push(std::move(e)); return ERROR_SUCCESS;
}
#endif
#if defined(FORGE_OS_MACOS)
struct iokit_notify {
    void* (*port_create)(unsigned) = nullptr; void (*port_destroy)(void*) = nullptr; void (*port_queue)(void*, void*) = nullptr;
    int (*add)(void*, const char*, const void*, void (*)(void*, unsigned), void*, unsigned*) = nullptr;
    const void* (*matching)(const char*) = nullptr; unsigned (*next)(unsigned) = nullptr; int (*release)(unsigned) = nullptr; int (*name)(unsigned, char*) = nullptr;
    void* (*queue_create)(const char*, void*) = nullptr; void (*queue_release)(void*) = nullptr; void (*sync_f)(void*, void*, void (*)(void*)) = nullptr;
    bool ok = false;
    static const iokit_notify& api() {
        static const iokit_notify a = [] {
            iokit_notify x; void* h = ::dlopen("/System/Library/Frameworks/IOKit.framework/IOKit", RTLD_LAZY); if (!h) return x;
            auto f = [&](auto& p, const char* n, void* lib) { p = reinterpret_cast<std::remove_reference_t<decltype(p)>>(::dlsym(lib, n)); };
            f(x.port_create, "IONotificationPortCreate", h); f(x.port_destroy, "IONotificationPortDestroy", h); f(x.port_queue, "IONotificationPortSetDispatchQueue", h);
            f(x.add, "IOServiceAddMatchingNotification", h); f(x.matching, "IOServiceMatching", h); f(x.next, "IOIteratorNext", h); f(x.release, "IOObjectRelease", h); f(x.name, "IORegistryEntryGetName", h);
            f(x.queue_create, "dispatch_queue_create", RTLD_DEFAULT); f(x.queue_release, "dispatch_release", RTLD_DEFAULT); f(x.sync_f, "dispatch_sync_f", RTLD_DEFAULT);
            x.ok = x.port_create && x.port_destroy && x.port_queue && x.add && x.matching && x.next && x.release && x.name && x.queue_create && x.queue_release && x.sync_f;
            return x;
        }();
        return a;
    }
};
struct mac_sub { queue* q = nullptr; device_action action = device_action::added; const char* subsystem = "other"; std::atomic<bool> armed{false}; unsigned iter = 0; };
inline void mac_drain(void* ctx, unsigned iter) {   // also called once at registration: that pass arms the notification
    auto* s = static_cast<mac_sub*>(ctx); auto& io = iokit_notify::api();
    while (unsigned obj = io.next(iter)) {
        if (s->armed) { char nm[128] = {}; io.name(obj, nm); s->q->push({s->action, s->subsystem, nm}); }
        io.release(obj);
    }
}
#endif
}

class monitor {
public:
    monitor(const monitor&) = delete; monitor& operator=(const monitor&) = delete;
    static result<std::unique_ptr<monitor>> open(monitor_options o = {}) {
        std::unique_ptr<monitor> m(new monitor); m->opt_ = o; if (m->opt_.poll_interval_ms < 10) m->opt_.poll_interval_ms = 10;
#if defined(FORGE_OS_IOS)
        return {{}, unsupported("device monitor")};
#else
        if (!o.force_polling) m->open_native_();
        if (!m->native_) { m->snapshot_(m->usb_, m->blk_); m->last_ = monotonic_ns(); }
        return {std::move(m), {}};
#endif
    }
    // uevent, devd, iokit, cfgmgr or polling.
    const char* backend() const { return native_ ? native_name_ : "polling"; }
    std::vector<device_event> poll(int timeout_ms) {
        std::vector<device_event> out;
        if (!native_) { poll_snapshot_(timeout_ms, out); return out; }
#if defined(__linux__) || defined(FORGE_OS_FREEBSD)
        pollfd pf{fd_, POLLIN, 0}; int wait = timeout_ms;
        while (::poll(&pf, 1, wait) > 0 && (pf.revents & POLLIN)) {
            char buf[8192];
#if defined(__linux__)
            sockaddr_nl from{}; iovec iov{buf, sizeof buf}; msghdr mh{}; mh.msg_name = &from; mh.msg_namelen = sizeof from; mh.msg_iov = &iov; mh.msg_iovlen = 1;
            auto n = ::recvmsg(fd_, &mh, MSG_DONTWAIT); if (n <= 0) break;
            device_event ev; if (from.nl_pid == 0 && detail_dev::parse_uevent(std::string_view(buf, std::size_t(n)), ev)) out.push_back(std::move(ev));   // nl_pid 0: from the kernel, not another process
#else
            auto n = ::recv(fd_, buf, sizeof buf, MSG_DONTWAIT); if (n <= 0) break;
            device_event ev; if (detail_dev::parse_devd(std::string_view(buf, std::size_t(n)), ev)) out.push_back(std::move(ev));
#endif
            wait = out.empty() ? wait : 20; if (out.size() > 4096) break;
        }
#elif defined(FORGE_OS_WINDOWS) || defined(FORGE_OS_MACOS)
        out = q_->take(timeout_ms);
#endif
        return out;
    }
    ~monitor() {
#if defined(__linux__) || defined(FORGE_OS_FREEBSD)
        if (fd_ >= 0) ::close(fd_);
#elif defined(FORGE_OS_WINDOWS)
        if (reg_) {
            static HMODULE cm = LoadLibraryExW(L"cfgmgr32.dll", nullptr, 0x800);
            if (auto unreg = cm ? reinterpret_cast<DWORD(WINAPI*)(void*)>(reinterpret_cast<void*>(GetProcAddress(cm, "CM_Unregister_Notification"))) : nullptr) unreg(reg_);   // waits for callbacks in flight
        }
#elif defined(FORGE_OS_MACOS)
        if (port_) {
            auto& io = detail_dev::iokit_notify::api();
            io.sync_f(dq_, nullptr, [](void*) {});   // a callback in flight finishes first
            io.port_destroy(port_); for (auto& s : subs_) if (s->iter) io.release(s->iter);
            io.sync_f(dq_, nullptr, [](void*) {}); io.queue_release(dq_);
        }
#endif
    }
private:
    monitor() = default;
    static void snapshot_(std::vector<std::string>& usb_names, std::vector<std::string>& blk_names) {
        usb_names.clear(); blk_names.clear();
        if (auto u = usb()) for (auto& d : u.value) usb_names.push_back(d.path + " " + d.vendor_id + ":" + d.product_id);
        if (auto b = block()) for (auto& d : b.value) blk_names.push_back(d.name);
    }
    void poll_snapshot_(int timeout_ms, std::vector<device_event>& out) {
        const std::uint64_t interval = std::uint64_t(opt_.poll_interval_ms) * 1000000ull;
        const std::uint64_t end = timeout_ms < 0 ? ~std::uint64_t(0) : monotonic_ns() + std::uint64_t(timeout_ms) * 1000000ull;
        for (;;) {
            std::uint64_t now = monotonic_ns();
            if (now - last_ >= interval || timeout_ms == 0) {
                std::vector<std::string> u, b; snapshot_(u, b);
                detail_dev::diff_names(usb_, u, "usb", out); detail_dev::diff_names(blk_, b, "block", out);
                usb_ = std::move(u); blk_ = std::move(b); last_ = now;
                if (!out.empty()) return;
            }
            if (now >= end) return;
            std::uint64_t next = last_ + interval; sleep_ns(std::min(next > now ? next - now : interval, end - now));
        }
    }
    void open_native_() {
#if defined(__linux__)
        fd_ = ::socket(AF_NETLINK, SOCK_DGRAM | SOCK_CLOEXEC, NETLINK_KOBJECT_UEVENT); if (fd_ < 0) return;
        sockaddr_nl sa{}; sa.nl_family = AF_NETLINK; sa.nl_groups = 1;   // group 1: kernel uevents (2 is udev's rebroadcast)
        if (::bind(fd_, reinterpret_cast<sockaddr*>(&sa), sizeof sa)) { ::close(fd_); fd_ = -1; return; }
        native_ = true; native_name_ = "uevent";
#elif defined(FORGE_OS_FREEBSD)
        fd_ = cloexec_socket(AF_UNIX, SOCK_SEQPACKET); if (fd_ < 0) return;
        sockaddr_un a{}; a.sun_family = AF_UNIX; std::snprintf(a.sun_path, sizeof a.sun_path, "%s", "/var/run/devd.seqpacket.pipe");
        if (::connect(fd_, reinterpret_cast<sockaddr*>(&a), sizeof a)) { ::close(fd_); fd_ = -1; return; }   // devd not running
        native_ = true; native_name_ = "devd";
#elif defined(FORGE_OS_WINDOWS)
        static HMODULE cm = LoadLibraryExW(L"cfgmgr32.dll", nullptr, 0x800 /*LOAD_LIBRARY_SEARCH_SYSTEM32*/);
        using cb_fn = DWORD(CALLBACK*)(void*, void*, int, detail_dev::cm_event_data*, DWORD);
        auto reg = cm ? reinterpret_cast<DWORD(WINAPI*)(detail_dev::cm_filter*, void*, cb_fn, void**)>(reinterpret_cast<void*>(GetProcAddress(cm, "CM_Register_Notification"))) : nullptr;
        if (!reg || !GetProcAddress(cm, "CM_Unregister_Notification")) return;
        q_ = std::make_unique<detail_dev::queue>();
        detail_dev::cm_filter f{}; f.cbSize = sizeof f; f.Flags = 0x1 /*CM_NOTIFY_FILTER_FLAG_ALL_INTERFACE_CLASSES*/; f.FilterType = 0 /*DEVICEINTERFACE*/;
        if (reg(&f, q_.get(), &detail_dev::cm_callback, &reg_) != 0 /*CR_SUCCESS*/ || !reg_) { reg_ = nullptr; q_.reset(); return; }
        native_ = true; native_name_ = "cfgmgr";
#elif defined(FORGE_OS_MACOS)
        auto& io = detail_dev::iokit_notify::api(); if (!io.ok) return;
        port_ = io.port_create(0); if (!port_) return;
        dq_ = io.queue_create("forge.os.device_watch", nullptr); if (!dq_) { io.port_destroy(port_); port_ = nullptr; return; }
        q_ = std::make_unique<detail_dev::queue>(); io.port_queue(port_, dq_);
        struct what { const char* cls; const char* note; device_action action; const char* subsystem; };
        static const what subs[] = {{"IOUSBHostDevice", "IOServiceFirstMatch", device_action::added, "usb"}, {"IOUSBHostDevice", "IOServiceTerminate", device_action::removed, "usb"},
                                    {"IOMedia", "IOServiceFirstMatch", device_action::added, "block"}, {"IOMedia", "IOServiceTerminate", device_action::removed, "block"}};
        for (auto& w : subs) {
            auto s = std::make_unique<detail_dev::mac_sub>(); s->q = q_.get(); s->action = w.action; s->subsystem = w.subsystem;
            const void* match = io.matching(w.cls); if (!match) continue;   // the dictionary is consumed by the call below
            if (io.add(port_, w.note, match, &detail_dev::mac_drain, s.get(), &s->iter) != 0) continue;
            detail_dev::mac_drain(s.get(), s->iter); s->armed = true;   // the first pass lists what is already there
            subs_.push_back(std::move(s));
        }
        if (subs_.empty()) { io.port_destroy(port_); port_ = nullptr; io.queue_release(dq_); dq_ = nullptr; q_.reset(); return; }
        native_ = true; native_name_ = "iokit";
#endif
    }
    monitor_options opt_; bool native_ = false; const char* native_name_ = "polling";
    std::vector<std::string> usb_, blk_; std::uint64_t last_ = 0;
#if defined(__linux__) || defined(FORGE_OS_FREEBSD)
    int fd_ = -1;
#elif defined(FORGE_OS_WINDOWS)
    std::unique_ptr<detail_dev::queue> q_; void* reg_ = nullptr;
#elif defined(FORGE_OS_MACOS)
    std::unique_ptr<detail_dev::queue> q_; void* port_ = nullptr; void* dq_ = nullptr; std::vector<std::unique_ptr<detail_dev::mac_sub>> subs_;
#endif
};
} // namespace forge::os::devices

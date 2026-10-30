// forge_mobile Linux backend (postmarketOS, Mobian, PinePhone, Librem 5, any IIO/evdev device):
//   sensors   IIO sysfs channels under <sysfs>/bus/iio/devices, sampled at the requested rate;
//             battery from <sysfs>/class/power_supply. Location: not_supported (needs GeoClue over D-Bus).
//   touch     evdev nodes in <input> whose <sysfs>/class/input/<node>/device/capabilities/abs has
//             ABS_MT_POSITION_X (multitouch protocol B), or ABS_X + BTN_TOUCH (single touch).
//   haptics   evdev force feedback (FF_RUMBLE) — the vibra motor on most phones.
//   display   DRM connector modes, else fbdev virtual_size; orientation from the accelerometer.
//   cameras   <sysfs>/class/video4linux.  notify: notify-send.  permissions: not_required.
#include "internal.hpp"
#if defined(__linux__) && !defined(__ANDROID__)
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <linux/input.h>
#include <poll.h>
#include <spawn.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fstream>
#include <map>

extern char** environ;

namespace forge::os::mobile::detail {
namespace {
std::string slurp(const std::string& p) { std::ifstream f(p); if (!f) return {}; std::string s((std::istreambuf_iterator<char>(f)), {}); while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back(); return s; }
bool exists(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0; }
std::vector<std::string> ls(const std::string& d) { std::vector<std::string> v; if (auto l = list_dir(d)) for (auto& e : l.value) v.push_back(e.name); std::sort(v.begin(), v.end()); return v; }
bool num(const std::string& p, double& out) { auto s = slurp(p); if (s.empty()) return false; char* e = nullptr; out = std::strtod(s.c_str(), &e); return e != s.c_str(); }

// Bit test on a sysfs capability mask ("hex words, most significant first").
bool cap_bit(const std::string& mask, int bit) {
    std::vector<unsigned long long> w; std::size_t i = 0;
    while (i < mask.size()) { while (i < mask.size() && mask[i] == ' ') ++i; std::size_t j = i; while (j < mask.size() && mask[j] != ' ') ++j; if (j > i) w.push_back(std::stoull(mask.substr(i, j - i), nullptr, 16)); i = j; }
    std::size_t word = std::size_t(bit / 64); if (word >= w.size()) return false;
    return (w[w.size() - 1 - word] >> (bit % 64)) & 1ull;
}

struct channel { std::string dev; std::string base; int axes; double mul; };   // value = (raw+offset)*scale*mul
struct iio_sensor { sensor_info info; std::vector<channel> ch; std::string input_file; double input_mul = 1; };

double read_chan(const channel& c, int axis, bool& ok) {
    static const char* ax[] = {"_x", "_y", "_z"};
    std::string a = c.axes > 1 ? ax[axis] : "";
    double v; ok = true;
    if (num(c.dev + "/" + c.base + a + "_input", v)) return v * c.mul;
    double raw; if (!num(c.dev + "/" + c.base + a + "_raw", raw)) { ok = false; return 0; }
    double off = 0, sc = 1; if (!num(c.dev + "/" + c.base + a + "_offset", off)) num(c.dev + "/" + c.base + "_offset", off);
    if (!num(c.dev + "/" + c.base + a + "_scale", sc)) num(c.dev + "/" + c.base + "_scale", sc);
    return (raw + off) * sc * c.mul;
}

struct linux_backend final : backend, std::enable_shared_from_this<linux_backend> {
    std::string sys, input;
    linux_backend(std::string s, std::string i) : sys(std::move(s)), input(std::move(i)) {}
    std::string name() const override { return "linux"; }

    std::vector<iio_sensor> scan() const {
        std::vector<iio_sensor> out; std::string root = sys + "/bus/iio/devices";
        struct kind_def { sensor_kind k; const char* base; int axes; double mul; };
        static const kind_def defs[] = {
            {sensor_kind::accelerometer, "in_accel", 3, 1.0}, {sensor_kind::gyroscope, "in_anglvel", 3, 1.0},
            {sensor_kind::magnetometer, "in_magn", 3, 100.0 /* gauss -> uT */}, {sensor_kind::light, "in_illuminance", 1, 1.0},
            {sensor_kind::proximity, "in_proximity", 1, 1.0}, {sensor_kind::pressure, "in_pressure", 1, 10.0 /* kPa -> hPa */}};
        for (auto& d : ls(root)) {
            std::string dev = root + "/" + d; auto nm = slurp(dev + "/name");
            for (auto& k : defs) {
                std::string probe = dev + "/" + k.base + (k.axes > 1 ? "_x" : "");
                if (!exists(probe + "_raw") && !exists(probe + "_input")) continue;
                iio_sensor s; s.info = {k.k, nm.empty() ? d : nm, "iio", 0, 0, 0};
                double sc; if (num(dev + "/" + k.base + "_scale", sc)) s.info.resolution = sc * k.mul;
                if (auto f = slurp(dev + "/sampling_frequency"); !f.empty()) s.info.max_rate_hz = std::strtod(f.c_str(), nullptr);
                s.ch.push_back({dev, k.base, k.axes, k.mul}); out.push_back(std::move(s));
            }
        }
        if (!battery_dir().empty()) { iio_sensor b; b.info = {sensor_kind::battery, "power_supply", "linux", 1, 100, 1}; out.push_back(b); }
        return out;
    }
    std::string battery_dir() const {
        std::string r = sys + "/class/power_supply";
        for (auto& e : ls(r)) if (slurp(r + "/" + e + "/type") == "Battery") return r + "/" + e;
        return {};
    }
    result<std::vector<sensor_info>> sensors() override { std::vector<sensor_info> v; for (auto& s : scan()) v.push_back(s.info); return {v, {}}; }

    bool sample(const iio_sensor& s, sensor_event& e) const {
        e = {}; e.kind = s.info.kind; e.t_ns = now_ns();
        if (s.info.kind == sensor_kind::battery) {
            auto d = battery_dir(); double cap; if (d.empty() || !num(d + "/capacity", cap)) return false;
            auto st = slurp(d + "/status"); e.v = {cap, (st == "Charging" || st == "Full") ? 1.0 : 0.0}; e.n = 2; return true;
        }
        auto& c = s.ch[0];
        for (int a = 0; a < c.axes; ++a) { bool ok; e.v[a] = read_chan(c, a, ok); if (!ok) return false; }
        e.n = c.axes; return true;
    }
    result<sensor_stream> open_sensor(sensor_kind k, double hz) override {
        if (k == sensor_kind::location) return {{}, unsupported("location on Linux (use GeoClue)")};
        for (auto& s : scan()) if (s.info.kind == k) {
            auto q = std::make_shared<queue>(); bridge::get().add_sensor(k, q);
            auto w = std::make_shared<worker>(); auto period = std::uint64_t(1e9 / hz);
            w->start([this_s = s, q, wp = w.get(), period, self = shared_from_this()] {
                for (;;) {
                    sensor_event e; if (self->sample(this_s, e)) q->push({false, e, {}});
                    if (!wp->sleep_ns(period)) return;
                }
            });
            return {make_sensor(k, q, w), {}};
        }
        return {{}, unsupported(std::string("sensor ") + sensor_name(k) + " (no IIO device)")};
    }

    // ---- touch
    std::vector<std::string> touch_nodes() const {
        std::vector<std::string> v;
        for (auto& n : ls(input)) {
            if (n.rfind("event", 0)) continue;
            std::string abs = slurp(sys + "/class/input/" + n + "/device/capabilities/abs"), key = slurp(sys + "/class/input/" + n + "/device/capabilities/key");
            if (abs.empty()) continue;
            if (cap_bit(abs, ABS_MT_POSITION_X) || (cap_bit(abs, ABS_X) && !key.empty() && cap_bit(key, BTN_TOUCH))) v.push_back(input + "/" + n);
        }
        return v;
    }
    result<touch_stream> open_touch() override {
        auto nodes = touch_nodes(); if (nodes.empty()) return {{}, unsupported("touch input (no evdev touchscreen)")};
        int fd = ::open(nodes[0].c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
        if (fd < 0) return {{}, last_error("open " + nodes[0])};
        auto q = std::make_shared<queue>(); bridge::get().add_touch(q);
        auto w = std::make_shared<worker>();
        w->start([fd, q, wp = w.get()] { evdev_loop(fd, *q, *wp); ::close(fd); });
        return {make_touch(q, w), {}};
    }
    static void evdev_loop(int fd, queue& q, worker& w) {
        struct slot { int id = -1; double x = 0, y = 0, p = 1; bool changed = false, down = false, up = false; };
        std::vector<slot> slots(16); int cur = 0; bool st_touch = false, st_changed = false;
        auto active = [&] { int n = 0; for (auto& s : slots) n += s.id >= 0; return n; };
        input_event buf[64];
        while (!w.stop) {
            pollfd pf{fd, POLLIN, 0}; int r = ::poll(&pf, 1, 50);
            if (r < 0 && errno != EINTR) break; if (r <= 0) continue;
            ssize_t n = ::read(fd, buf, sizeof buf);
            if (n == 0) break; if (n < 0) { if (errno == EAGAIN || errno == EINTR) continue; break; }
            for (std::size_t i = 0; i < std::size_t(n) / sizeof(input_event); ++i) {
                auto& ev = buf[i]; auto& s = slots[std::size_t(cur)];
                if (ev.type == EV_ABS) {
                    switch (ev.code) {
                    case ABS_MT_SLOT: cur = std::clamp(ev.value, 0, 15); break;
                    case ABS_MT_TRACKING_ID: if (ev.value >= 0) { s.id = ev.value; s.down = true; } else { s.up = true; } s.changed = true; break;
                    case ABS_MT_POSITION_X: s.x = ev.value; s.changed = true; break;
                    case ABS_MT_POSITION_Y: s.y = ev.value; s.changed = true; break;
                    case ABS_MT_PRESSURE: s.p = ev.value; s.changed = true; break;
                    case ABS_X: if (s.id < 0 || cur == 0) { slots[0].x = ev.value; st_changed = true; } break;
                    case ABS_Y: if (s.id < 0 || cur == 0) { slots[0].y = ev.value; st_changed = true; } break;
                    }
                } else if (ev.type == EV_KEY && ev.code == BTN_TOUCH) {
                    st_touch = true; if (ev.value) { if (slots[0].id < 0) { slots[0].id = 0; slots[0].down = true; } } else slots[0].up = true;
                    slots[0].changed = true;
                } else if (ev.type == EV_SYN && ev.code == SYN_REPORT) {
                    std::uint64_t t = std::uint64_t(ev.input_event_sec) * 1000000000ull + std::uint64_t(ev.input_event_usec) * 1000ull;
                    if (st_touch && st_changed) slots[0].changed = true;
                    for (auto& sl : slots) {
                        if (!sl.changed) continue;
                        touch_phase ph = sl.up ? touch_phase::up : sl.down ? touch_phase::down : touch_phase::move;
                        int id = sl.id; int ptrs = active();
                        q.push({true, {}, {t, ph, id, sl.x, sl.y, sl.p, ptrs}});
                        if (sl.up) sl.id = -1;
                        sl.changed = sl.down = sl.up = false;
                    }
                    st_changed = false;
                }
            }
        }
        q.finish();
    }

    // ---- haptics via evdev force feedback
    status vibrate(int ms, double amp) override {
        for (auto& n : ls(input)) {
            if (n.rfind("event", 0)) continue;
            auto ff = slurp(sys + "/class/input/" + n + "/device/capabilities/ff");
            if (ff.empty() || !cap_bit(ff, FF_RUMBLE)) continue;
            int fd = ::open((input + "/" + n).c_str(), O_RDWR | O_CLOEXEC); if (fd < 0) return {{}, last_error("open " + n)};
            ff_effect e{}; e.type = FF_RUMBLE; e.id = -1; e.replay.length = std::uint16_t(ms);
            e.u.rumble.strong_magnitude = std::uint16_t(std::clamp(amp, 0.0, 1.0) * 0xFFFF); e.u.rumble.weak_magnitude = e.u.rumble.strong_magnitude;
            if (::ioctl(fd, EVIOCSFF, &e) < 0) { auto er = last_error("EVIOCSFF"); ::close(fd); return {{}, er}; }
            input_event play{}; play.type = EV_FF; play.code = std::uint16_t(e.id); play.value = 1;
            bool ok = ::write(fd, &play, sizeof play) == ssize_t(sizeof play);
            ::usleep(useconds_t(ms) * 1000); ::ioctl(fd, EVIOCRMFF, e.id); ::close(fd);
            return ok ? status{} : status{{}, last_error("play rumble")};
        }
        return backend::vibrate(ms, amp);
    }

    result<display_info> display() override {
        if (auto b = backend::display()) return b;
        display_info d; std::string drm = sys + "/class/drm";
        for (auto& c : ls(drm)) {
            if (c.find('-') == std::string::npos || slurp(drm + "/" + c + "/status") != "connected") continue;
            auto modes = slurp(drm + "/" + c + "/modes"); auto nl = modes.find('\n'); auto m = modes.substr(0, nl);
            if (std::sscanf(m.c_str(), "%dx%d", &d.width_px, &d.height_px) == 2) break;
        }
        if (!d.width_px) { auto v = slurp(sys + "/class/graphics/fb0/virtual_size"); std::sscanf(v.c_str(), "%d,%d", &d.width_px, &d.height_px); }
        if (!d.width_px) return {{}, unsupported("display metrics (no DRM/fbdev)")};
        d.orient = d.width_px > d.height_px ? orientation::landscape_left : orientation::portrait;
        if (auto o = accel_orientation(); o != orientation::unknown) d.orient = o;
        return {d, {}};
    }
    orientation accel_orientation() const {
        for (auto& s : scan()) if (s.info.kind == sensor_kind::accelerometer) {
            sensor_event e; if (!sample(s, e)) break;
            double x = e.v[0], y = e.v[1]; if (std::hypot(x, y) < 3.0) return orientation::unknown;   // lying flat
            if (std::fabs(y) >= std::fabs(x)) return y >= 0 ? orientation::portrait : orientation::portrait_upside_down;
            return x >= 0 ? orientation::landscape_left : orientation::landscape_right;
        }
        return orientation::unknown;
    }
    result<orientation> screen_orientation() override {
        if (auto o = accel_orientation(); o != orientation::unknown) return {o, {}};
        auto d = display(); if (!d) return {{}, d.err}; return {d.value.orient, {}};
    }
    result<std::vector<camera_info>> cameras() override {
        std::vector<camera_info> v; std::string r = sys + "/class/video4linux";
        for (auto& c : ls(r)) { auto nm = slurp(r + "/" + c + "/name"); auto lc = nm; for (auto& ch : lc) ch = char(std::tolower(ch)); v.push_back({c, nm, lc.find("front") != std::string::npos}); }
        return {v, {}};
    }
    result<permission_state> permission_status(permission p) override { if (auto b = backend::permission_status(p)) return b; return {permission_state::not_required, {}}; }
    status request_permission(permission) override { return {}; }
    status notify(std::string_view t, std::string_view b) override {
        if (auto h = backend::notify(t, b); !is_unsupported(h.err)) return h;
        std::string ts(t), bs(b); char* argv[] = {const_cast<char*>("notify-send"), const_cast<char*>("--app-name=forge"), ts.data(), bs.data(), nullptr};
        pid_t pid; if (posix_spawnp(&pid, "notify-send", nullptr, nullptr, argv, environ) != 0) return {{}, unsupported("notifications (no notify-send)")};
        int st = 0; ::waitpid(pid, &st, 0);
        return WIFEXITED(st) && WEXITSTATUS(st) == 0 ? status{} : status{{}, {-1, "notify-send failed"}};
    }
};
} // namespace
std::shared_ptr<backend> make_linux(std::string sysfs, std::string input) { return std::make_shared<linux_backend>(std::move(sysfs), std::move(input)); }
} // namespace forge::os::mobile::detail
#endif

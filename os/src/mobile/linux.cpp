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
#include "evdev.hpp"
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
#include <linux/videodev2.h>
#include <map>
#include <forge/os/process.hpp>

extern char** environ;

namespace forge::os::mobile::detail {
namespace {
std::string slurp(const std::string& p) { std::ifstream f(p); if (!f) return {}; std::string s((std::istreambuf_iterator<char>(f)), {}); while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back(); return s; }
bool exists(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0; }
std::vector<std::string> ls(const std::string& d) { std::vector<std::string> v; if (auto l = list_dir(d)) for (auto& e : l.value) v.push_back(e.name); std::sort(v.begin(), v.end()); return v; }
// sysfs attribute write; the value only lands when write() succeeds (an ofstream reports failure at flush, if ever).
status write_attr(const std::string& p, const std::string& v) {
    int fd = ::open(p.c_str(), O_WRONLY | O_CLOEXEC); if (fd < 0) return {{}, last_error("open " + p)};
    ssize_t n = ::write(fd, v.data(), v.size()); auto e = n == ssize_t(v.size()) ? error{} : last_error("write " + p);
    ::close(fd); return {{}, e};
}
// nmcli -t -e yes: fields separated by ':', with ':' and '\' inside values escaped by a backslash.
std::vector<std::string> nm_fields(const std::string& line) {
    std::vector<std::string> f(1);
    for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '\\' && i + 1 < line.size()) f.back() += line[++i];
        else if (line[i] == ':') f.emplace_back();
        else f.back() += line[i];
    }
    return f;
}
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
            {sensor_kind::proximity, "in_proximity", 1, 1.0}, {sensor_kind::pressure, "in_pressure", 1, 10.0 /* kPa -> hPa */},
            {sensor_kind::temperature, "in_temp", 1, 0.001 /* milli degC */}, {sensor_kind::humidity, "in_humidityrelative", 1, 0.001 /* milli %RH */},
            {sensor_kind::step_counter, "in_steps", 1, 1.0}, {sensor_kind::gravity, "in_gravity", 3, 1.0}};
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

    // ---- touch: every direct-touch node (touchscreens, not touchpads), positions in display pixels
    std::vector<std::string> touch_nodes() const {
        std::vector<std::string> v;
        for (auto& n : ls(input)) {
            if (n.rfind("event", 0)) continue;
            std::string dev = sys + "/class/input/" + n + "/device";
            std::string abs = slurp(dev + "/capabilities/abs"), key = slurp(dev + "/capabilities/key");
            if (abs.empty()) continue;
            if (!cap_bit(abs, ABS_MT_POSITION_X) && !(cap_bit(abs, ABS_X) && !key.empty() && cap_bit(key, BTN_TOUCH))) continue;
            if (auto props = slurp(dev + "/properties"); !props.empty()) { if (!cap_bit(props, INPUT_PROP_DIRECT)) continue; }
            else { int fd = ::open((input + "/" + n).c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC); bool direct = fd >= 0 && evdev_is_direct(fd); if (fd >= 0) ::close(fd); if (!direct) continue; }
            v.push_back(input + "/" + n);
        }
        return v;
    }
    result<touch_stream> open_touch() override {
        auto paths = touch_nodes(); if (paths.empty()) return {{}, unsupported("touch input (no evdev touchscreen)")};
        int w_px = 0, h_px = 0; if (auto d = display()) { w_px = d.value.width_px; h_px = d.value.height_px; }
        std::vector<evdev_node> nodes; error first_err{};
        for (auto& p : paths) {
            int fd = ::open(p.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
            if (fd < 0) { if (!first_err) first_err = last_error("open " + p); continue; }
            nodes.push_back(evdev_setup(fd, w_px, h_px));
        }
        if (nodes.empty()) return {{}, first_err};
        auto q = std::make_shared<queue>(); bridge::get().add_touch(q);
        auto w = std::make_shared<worker>();
        w->start([nodes = std::move(nodes), q, wp = w.get()]() mutable { evdev_loop(std::move(nodes), *q, *wp); });
        return {make_touch(q, w), {}};
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
    // Capture nodes only: UVC also registers a metadata node per camera, codecs register m2m nodes. VIDIOC_QUERYCAP on
    // the device node decides; without access to it (fake trees, permissions) the sysfs "index" (0 = the capture
    // interface of a multi-node device) does.
    result<std::vector<camera_info>> cameras() override {
        std::vector<camera_info> v; std::string r = sys + "/class/video4linux";
        std::string devdir = input.substr(0, input.rfind('/'));   // /dev/input -> /dev
        for (auto& c : ls(r)) {
            int capture = -1;
            int fd = ::open((devdir + "/" + c).c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
            if (fd >= 0) {
                v4l2_capability cap{};
                if (::ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
                    std::uint32_t caps = (cap.capabilities & V4L2_CAP_DEVICE_CAPS) ? cap.device_caps : cap.capabilities;
                    capture = (caps & (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_VIDEO_CAPTURE_MPLANE)) && !(caps & (V4L2_CAP_VIDEO_M2M | V4L2_CAP_VIDEO_M2M_MPLANE)) ? 1 : 0;
                }
                ::close(fd);
            }
            if (capture < 0) { auto idx = slurp(r + "/" + c + "/index"); capture = idx.empty() || idx == "0"; }
            if (!capture) continue;
            auto nm = slurp(r + "/" + c + "/name"); auto lc = nm; for (auto& ch : lc) ch = char(std::tolower(ch));
            v.push_back({c, nm, lc.find("front") != std::string::npos});
        }
        return {v, {}};
    }
    // ---- services
    std::string first_class(const char* cls, const char* contains = nullptr) const {
        std::string r = sys + "/class/" + cls;
        for (auto& e : ls(r)) { if (!contains) return r + "/" + e; auto lc = e; for (auto& c : lc) c = char(std::tolower(c)); if (lc.find(contains) != std::string::npos) return r + "/" + e; }
        return {};
    }
    result<double> brightness() override {
        auto d = first_class("backlight"); double b, m;
        if (d.empty() || !num(d + "/brightness", b) || !num(d + "/max_brightness", m) || m <= 0) return {{}, unsupported("screen brightness (no backlight)")};
        return {b / m, {}};
    }
    status set_brightness(double l) override {
        auto d = first_class("backlight"); double m; if (d.empty() || !num(d + "/max_brightness", m)) return {{}, unsupported("screen brightness (no backlight)")};
        return write_attr(d + "/brightness", std::to_string(std::lround(l * m)) + "\n");
    }
    status torch(bool on) override {
        auto d = first_class("leds", "flash"); if (d.empty()) d = first_class("leds", "torch");
        if (d.empty()) return {{}, unsupported("torch (no flash LED)")};
        double m = 1; num(d + "/max_brightness", m);
        return write_attr(d + "/brightness", std::to_string(on ? long(m) : 0L) + "\n");
    }
    result<bool> nfc_available() override { return {!ls(sys + "/class/nfc").empty(), {}}; }
    result<connectivity_info> connectivity() override {
        auto c = backend::connectivity(); if (!c) return c;
        std::string out;   // NetworkManager over its CLI (D-Bus underneath); optional
        if (run_tool({"nmcli", "-t", "-e", "yes", "-f", "ACTIVE,SSID,SIGNAL", "dev", "wifi"}, &out, 5000) == 0) {
            std::size_t i = 0;
            while (i < out.size()) { auto j = out.find('\n', i); auto line = out.substr(i, j - i); i = j == std::string::npos ? out.size() : j + 1;
                auto f = nm_fields(line);
                if (f.size() == 3 && f[0] == "yes") { c.value.ssid = f[1]; c.value.signal_percent = std::atoi(f[2].c_str()); break; } }
        }
        if (c.value.transport == "cellular" || run_tool({"mmcli", "-L"}, &out, 5000) == 0 && out.find("/Modem/") != std::string::npos) {
            if (c.value.transport == "unknown") c.value.transport = "cellular";
        }
        return c;
    }
    // BlueZ through bluetoothctl (D-Bus underneath). Only devices heard during this scan are returned ("[NEW] Device"
    // / "[CHG] Device ... RSSI:" lines), not BlueZ's cache of paired or earlier devices; rssi is 0 when the
    // advertisement carried none.
    result<std::vector<ble_device>> bluetooth_scan(int ms) override {
        std::string out;
        int rc = run_tool({"bluetoothctl", "--timeout", std::to_string(std::max(1, ms / 1000)), "scan", "on"}, &out, ms + 5000);
        if (rc < 0) return {{}, unsupported("bluetooth scan (no bluetoothctl/BlueZ)")};
        if (rc != 0) return {{}, {rc, "bluetoothctl scan failed"}};
        std::vector<ble_device> v; std::size_t i = 0;
        auto find = [&](const std::string& a) -> ble_device& { for (auto& d : v) if (d.address == a) return d; v.push_back({a, "", 0}); return v.back(); };
        while (i < out.size()) {
            auto j = out.find('\n', i); auto line = out.substr(i, j - i); i = j == std::string::npos ? out.size() : j + 1;
            for (std::size_t k; (k = line.find('\x1b')) != std::string::npos;) { auto e = line.find('m', k); line.erase(k, e == std::string::npos ? std::string::npos : e - k + 1); }   // colours
            auto at = line.find("Device "); if (at == std::string::npos || line.size() < at + 24) continue;
            bool is_new = line.find("[NEW]") != std::string::npos, is_chg = line.find("[CHG]") != std::string::npos;
            if (!is_new && !is_chg) continue;
            auto addr = line.substr(at + 7, 17); auto rest = line.size() > at + 25 ? line.substr(at + 25) : std::string();
            if (is_new) { auto& d = find(addr); if (rest != addr && rest.find(':') == std::string::npos) d.name = rest; }
            else if (auto r = rest.find("RSSI: "); r != std::string::npos) {
                auto val = rest.substr(r + 6); auto paren = val.find('(');   // "-60" or "0xffffffc4 (-60)"
                find(addr).rssi = std::atoi((paren != std::string::npos ? val.substr(paren + 1) : val).c_str());
            }
        }
        return {v, {}};
    }
    result<biometric_info> biometrics() override {
        auto user = os::getenv("USER"); if (user.empty() || user[0] == '-') return {biometric_info{false, false, ""}, {}};
        std::string out; int rc = run_tool({"fprintd-list", user}, &out, 5000);
        // Not installed, no reader ("No devices available", exit 1) or fprintd not running: nothing usable.
        if (rc != 0 || out.find("No devices") != std::string::npos) return {biometric_info{false, false, ""}, {}};
        bool enrolled = out.find("Fingerprints for user") != std::string::npos && out.find("has no fingers enrolled") == std::string::npos;
        return {biometric_info{true, enrolled, "fingerprint"}, {}};
    }
    result<device_info> info() override {
        auto d = backend::info(); if (!d) return d;
        // Device tree (ARM boards, phones): model, and the form factor from chassis-type where the DT has it. A DT
        // alone does not make a phone (Raspberry Pi, routers): without chassis-type only a battery-powered device with
        // a touchscreen is taken for a phone, a battery alone for a laptop.
        auto dt = sys + "/firmware/devicetree/base";
        auto strip = [](std::string v) { while (!v.empty() && v.back() == '\0') v.pop_back(); return v; };
        std::string model = strip(slurp(dt + "/model"));
        if (!model.empty()) {
            d.value.model = model; auto ct = strip(slurp(dt + "/chassis-type"));
            d.value.form = ct == "handset" ? form_factor::phone : ct == "tablet" ? form_factor::tablet : ct == "watch" ? form_factor::watch
                         : ct == "laptop" || ct == "convertible" ? form_factor::laptop : ct == "desktop" || ct == "all-in-one" ? form_factor::desktop
                         : ct == "television" ? form_factor::tv : ct == "embedded" ? form_factor::unknown
                         : !battery_dir().empty() ? (touch_nodes().empty() ? form_factor::laptop : form_factor::phone) : form_factor::unknown;
        }
        auto dmi = sys + "/class/dmi/id";
        if (auto v = slurp(dmi + "/sys_vendor"); !v.empty()) d.value.manufacturer = v;
        if (auto v = slurp(dmi + "/product_name"); !v.empty() && model.empty()) d.value.model = v;
        if (auto ct = slurp(dmi + "/chassis_type"); !ct.empty()) {
            int t = std::atoi(ct.c_str());
            d.value.form = (t == 8 || t == 9 || t == 10 || t == 14 || t == 31) ? form_factor::laptop : (t == 30 || t == 32) ? form_factor::tablet : form_factor::desktop;
        }
        return d;
    }
    result<audio_clip> record_audio(int ms, int rate) override {   // ALSA through arecord
        std::string out;
        int rc = run_tool({"arecord", "-q", "-t", "raw", "-f", "S16_LE", "-c", "1", "-r", std::to_string(rate), "-d", std::to_string(std::max(1, (ms + 999) / 1000))}, &out, ms + 10000);
        if (rc < 0) return {{}, unsupported("audio recording (no arecord)")};
        if (rc != 0) return {{}, {rc, "arecord failed"}};
        audio_clip c; c.sample_rate = rate; c.samples.resize(std::min<std::size_t>(out.size() / 2, std::size_t(rate) * std::size_t(ms) / 1000));
        std::memcpy(c.samples.data(), out.data(), c.samples.size() * 2); return {c, {}};
    }
    status play_audio(const audio_clip& c) override {
        spawn_spec sp; sp.argv = {"aplay", "-q", "-t", "raw", "-f", "S16_LE", "-c", std::to_string(c.channels), "-r", std::to_string(c.sample_rate)};
        sp.stdin_data = std::string(reinterpret_cast<const char*>(c.samples.data()), c.samples.size() * 2);
        auto r = spawn_wait(sp); if (!r || r.value.exit_code == 127) return {{}, unsupported("audio playback (no aplay)")};
        return r.value.exit_code == 0 ? status{} : status{{}, {r.value.exit_code, "aplay failed"}};
    }
    result<permission_state> permission_status(permission p) override { if (auto b = backend::permission_status(p)) return b; return {permission_state::not_required, {}}; }
    status request_permission(permission) override { return {}; }
    status notify(std::string_view t, std::string_view b) override {
        if (auto h = backend::notify(t, b); !is_unsupported(h.err)) return h;
        std::string ts(t), bs(b); char* argv[] = {const_cast<char*>("notify-send"), const_cast<char*>("--app-name=forge"), const_cast<char*>("--"), ts.data(), bs.data(), nullptr};
        pid_t pid; if (posix_spawnp(&pid, "notify-send", nullptr, nullptr, argv, environ) != 0) return {{}, unsupported("notifications (no notify-send)")};
        int st = 0; ::waitpid(pid, &st, 0);
        return WIFEXITED(st) && WEXITSTATUS(st) == 0 ? status{} : status{{}, {-1, "notify-send failed"}};
    }
};
} // namespace
std::shared_ptr<backend> make_linux(std::string sysfs, std::string input) { return std::make_shared<linux_backend>(std::move(sysfs), std::move(input)); }
} // namespace forge::os::mobile::detail
#endif

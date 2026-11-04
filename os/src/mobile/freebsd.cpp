// forge_mobile FreeBSD backend.
//   sensors   temperature: sysctl dev.cpu.0.temperature / hw.acpi.thermal.tz0.temperature (deci-kelvin);
//             battery: hw.acpi.battery.life / hw.acpi.acline (via forge::os::power).
//   touch     evdev (/dev/input/event*, `kldload evdev` + hw.usb.usbhid.enable) multitouch protocol B and
//             single touch, every node with INPUT_PROP_DIRECT (touchscreens; touchpads are skipped).
//   display   not exposed without X/Wayland: app glue calls report_display().  Hot-plug arrives via devd(8);
//             apps can forward it with feed_lifecycle/feed_sensor.
#include "internal.hpp"
#if defined(__FreeBSD__)
#include "evdev.hpp"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/sysctl.h>

namespace forge::os::mobile::detail {
namespace {
bool temp_c(double& out) {
    for (const char* n : {"dev.cpu.0.temperature", "hw.acpi.thermal.tz0.temperature"}) {
        int dk = 0; std::size_t len = sizeof dk;
        if (::sysctlbyname(n, &dk, &len, nullptr, 0) == 0 && dk > 0) { out = dk / 10.0 - 273.15; return true; }
    }
    return false;
}
struct freebsd_backend final : backend {
    std::string name() const override { return "freebsd"; }
    result<std::vector<sensor_info>> sensors() override {
        std::vector<sensor_info> v; double t;
        if (temp_c(t)) v.push_back({sensor_kind::temperature, "sysctl thermal", "freebsd", 10, 0, 0.1});
        if (power_status().has_battery) v.push_back({sensor_kind::battery, "acpi battery", "freebsd", 1, 100, 1});
        return {v, {}};
    }
    result<sensor_stream> open_sensor(sensor_kind k, double hz) override {
        if (k != sensor_kind::temperature && k != sensor_kind::battery) return {{}, unsupported(std::string("sensor ") + sensor_name(k) + " on FreeBSD")};
        auto q = std::make_shared<queue>(); bridge::get().add_sensor(k, q);
        auto w = std::make_shared<worker>(); auto period = std::uint64_t(1e9 / hz);
        w->start([q, k, wp = w.get(), period] {
            do {
                sensor_event e; e.kind = k; e.t_ns = now_ns();
                if (k == sensor_kind::temperature) { if (temp_c(e.v[0])) { e.n = 1; q->push({false, e, {}}); } }
                else { auto p = power_status(); if (p.battery_percent >= 0) { e.v = {double(p.battery_percent), p.charging ? 1.0 : 0.0}; e.n = 2; q->push({false, e, {}}); } }
            } while (wp->sleep_ns(period));
        });
        return {make_sensor(k, q, w), {}};
    }
    // Every direct-touch evdev node (INPUT_PROP_DIRECT: touchscreens, not touchpads); positions in display pixels
    // once the app has reported the display size (report_display), else in device units.
    result<touch_stream> open_touch() override {
        int w_px = 0, h_px = 0; if (auto d = display()) { w_px = d.value.width_px; h_px = d.value.height_px; }
        std::vector<evdev_node> nodes;
        for (int i = 0; i < 64; ++i) {
            std::string p = "/dev/input/event" + std::to_string(i);
            int fd = ::open(p.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC); if (fd < 0) continue;
            constexpr int B = int(8 * sizeof(unsigned long));
            unsigned long bits[(ABS_CNT + B - 1) / B] = {}, keys[(KEY_CNT + B - 1) / B] = {};
            bool abs = ::ioctl(fd, EVIOCGBIT(EV_ABS, sizeof bits), bits) >= 0, key = ::ioctl(fd, EVIOCGBIT(EV_KEY, sizeof keys), keys) >= 0;
            auto bit = [](const unsigned long* b, int n) { return (b[n / B] >> (n % B)) & 1ul; };
            bool touch = abs && (bit(bits, ABS_MT_POSITION_X) || (bit(bits, ABS_X) && key && bit(keys, BTN_TOUCH)));
            if (touch && evdev_is_direct(fd)) nodes.push_back(evdev_setup(fd, w_px, h_px)); else ::close(fd);
        }
        if (nodes.empty()) return {{}, unsupported("touch input (no evdev touchscreen)")};
        auto q = std::make_shared<queue>(); bridge::get().add_touch(q); auto w = std::make_shared<worker>();
        w->start([nodes = std::move(nodes), q, wp = w.get()]() mutable { evdev_loop(std::move(nodes), *q, *wp); });
        return {make_touch(q, w), {}};
    }
    result<permission_state> permission_status(permission p) override { if (auto b = backend::permission_status(p)) return b; return {permission_state::not_required, {}}; }
    status request_permission(permission) override { return {}; }
};
} // namespace
std::shared_ptr<backend> make_freebsd() { return std::make_shared<freebsd_backend>(); }
} // namespace forge::os::mobile::detail
#endif

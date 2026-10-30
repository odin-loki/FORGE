// forge::os::mobile: simulated backend, gesture recognizer, Linux IIO/evdev against fake trees.
#include <forge/os/mobile.hpp>
#include "check.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#if defined(__linux__) && !defined(__ANDROID__)
#include <fcntl.h>
#include <linux/input.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
using namespace forge::os;
namespace m = forge::os::mobile;
namespace fs = std::filesystem;

static const char* k_trace = R"({
 "display": {"width": 1080, "height": 2400, "dpi": 420, "refresh_hz": 120, "orientation": "portrait"},
 "sensors": ["accelerometer", {"kind": "gyroscope", "name": "sim-gyro"}],
 "cameras": [{"id": "0", "name": "back"}, {"id": "1", "name": "selfie", "front": true}],
 "permissions": {"camera": "denied", "notifications": "not_determined"},
 "events": [
  {"t_ms": 0, "type": "sensor", "kind": "accelerometer", "v": [0.1, 9.8, 0.2]},
  {"t_ms": 10, "type": "sensor", "kind": "accelerometer", "v": [0.2, 9.7, 0.1]},
  {"t_ms": 12, "type": "sensor", "kind": "gyroscope", "v": [0.01, 0.02, 0.03]},
  {"t_ms": 20, "type": "touch", "phase": "down", "id": 0, "x": 100, "y": 100},
  {"t_ms": 80, "type": "touch", "phase": "up", "id": 0, "x": 102, "y": 101},
  {"t_ms": 200, "type": "touch", "phase": "down", "id": 0, "x": 100, "y": 500},
  {"t_ms": 300, "type": "touch", "phase": "up", "id": 0, "x": 400, "y": 510},
  {"t_ms": 400, "type": "orientation", "value": "landscape"}
 ]})";

TEST(sim_device_basics) {
    auto d = m::device::simulated(k_trace); CHECK(d); if (!d) return;
    CHECK(d->backend_name() == "sim");
    CHECK(d->has_sensor(m::sensor_kind::accelerometer)); CHECK(!d->has_sensor(m::sensor_kind::light));
    auto disp = d->display(); CHECK(disp); CHECK(disp->orient == m::orientation::landscape_left); CHECK(disp->width_px == 2400);
    auto cams = d->cameras(); CHECK(cams && cams->size() == 2 && cams->at(1).front);
    CHECK(d->permission_status(m::permission::camera).value == m::permission_state::denied);
    CHECK(d->notify("t", "b"));
    CHECK(d->request_permission(m::permission::notifications));
    CHECK(d->permission_status(m::permission::notifications).value == m::permission_state::granted);
    CHECK(d->vibrate(40, 0.5)); CHECK(!d->vibrate(0));
    auto log = d->sim_log(); CHECK(log.size() == 3);
    auto light = d->open_sensor(m::sensor_kind::light); CHECK(!light && is_unsupported(light.err));
}
TEST(sim_sensor_stream) {
    auto d = m::device::simulated(k_trace); if (!d) { CHECK(false); return; }
    auto s = d->open_sensor(m::sensor_kind::accelerometer, 100); CHECK(s); if (!s) return;
    std::vector<m::sensor_event> all;
    for (int i = 0; i < 50 && !s->finished(); ++i) for (auto& e : s->poll(20)) all.push_back(e);
    CHECK(all.size() == 2); if (all.size() == 2) { CHECK(all[0].n == 3); CHECK(all[1].v[1] == 9.7); }
}
TEST(sim_touch_and_gestures) {
    auto d = m::device::simulated(k_trace); if (!d) { CHECK(false); return; }
    auto t = d->open_touch(); CHECK(t); if (!t) return;
    std::vector<m::gesture> g;
    for (int i = 0; i < 50 && !t->finished(); ++i) for (auto& x : t->gestures(20)) g.push_back(x);
    CHECK(g.size() == 2);
    if (g.size() == 2) { CHECK(g[0].kind == m::gesture_kind::tap); CHECK(g[1].kind == m::gesture_kind::swipe); CHECK(g[1].dx == 300); }
}
TEST(gesture_pinch_double_long) {
    m::gesture_recognizer r; std::vector<m::gesture> g;
    auto f = [&](std::uint64_t ms, m::touch_phase p, int id, double x, double y) { for (auto& e : r.feed({ms * 1000000, p, id, x, y})) g.push_back(e); };
    f(0, m::touch_phase::down, 0, 10, 10); f(50, m::touch_phase::up, 0, 10, 10);
    f(150, m::touch_phase::down, 0, 12, 10); f(200, m::touch_phase::up, 0, 12, 10);
    CHECK(g.size() == 3 && g[2].kind == m::gesture_kind::double_tap); g.clear();
    f(1000, m::touch_phase::down, 0, 50, 50); f(1700, m::touch_phase::move, 0, 52, 50); f(1800, m::touch_phase::up, 0, 52, 50);
    CHECK(g.size() == 1 && g[0].kind == m::gesture_kind::long_press); g.clear();
    f(3000, m::touch_phase::down, 0, 100, 100); f(3001, m::touch_phase::down, 1, 200, 100);
    f(3050, m::touch_phase::move, 1, 300, 100);
    f(3100, m::touch_phase::up, 0, 100, 100); f(3100, m::touch_phase::up, 1, 300, 100);
    CHECK(!g.empty() && g.back().kind == m::gesture_kind::pinch && g.back().scale == 2.0);
}
TEST(sim_bad_trace) {
    CHECK(!m::device::simulated("{\"events\":[{\"type\":\"warp\"}]}"));
    CHECK(!m::device::simulated("not json"));
}
TEST(bridge_feed) {
    auto d = m::device::simulated("{}"); if (!d) { CHECK(false); return; }
    auto t = d->open_touch(); if (!t) { CHECK(false); return; }
    for (int i = 0; i < 20 && !t->finished(); ++i) t->poll(5);
    m::feed_touch({1, m::touch_phase::down, 3, 1, 2});
    auto v = t->poll(0); CHECK(v.size() == 1 && v[0].id == 3);
}
TEST(null_device_unsupported) {
    auto d = m::device::null_device(); CHECK(d);
    auto s = d->open_sensor(m::sensor_kind::accelerometer); CHECK(!s && is_unsupported(s.err));
    CHECK(is_unsupported(d->open_touch().err));
    CHECK(is_unsupported(d->cameras().err));
}
TEST(platform_open) { auto d = m::device::open(); CHECK(d); if (d) CHECK(!d->backend_name().empty()); }

#if defined(__linux__) && !defined(__ANDROID__)
static void put(const fs::path& p, const std::string& s) { fs::create_directories(p.parent_path()); std::ofstream(p) << s << "\n"; }
TEST(linux_iio_fake_sysfs) {
    fs::path root = fs::temp_directory_path() / ("forge-mobile-" + std::to_string(getpid()));
    fs::remove_all(root);
    auto dev = root / "sys/bus/iio/devices/iio:device0";
    put(dev / "name", "bmi260"); put(dev / "in_accel_x_raw", "100"); put(dev / "in_accel_y_raw", "-2000"); put(dev / "in_accel_z_raw", "50");
    put(dev / "in_accel_scale", "0.0047");
    auto dev1 = root / "sys/bus/iio/devices/iio:device1"; put(dev1 / "name", "stk3310"); put(dev1 / "in_illuminance_input", "321.5");
    put(dev1 / "in_proximity_raw", "7");
    auto bat = root / "sys/class/power_supply/battery"; put(bat / "type", "Battery"); put(bat / "capacity", "77"); put(bat / "status", "Charging");
    put(root / "sys/class/drm/card0-DSI-1/status", "connected"); put(root / "sys/class/drm/card0-DSI-1/modes", "720x1440");
    put(root / "sys/class/video4linux/video0/name", "ov5640 front");
    put(root / "sys/class/input/event0/device/capabilities/abs", "260800000000000");   // ABS_MT_POSITION_X/Y, TRACKING_ID, SLOT
    fs::create_directories(root / "input");
    auto d = m::device::linux_with((root / "sys").string(), (root / "input").string()); CHECK(d); if (!d) return;
    auto ss = d->sensors(); CHECK(ss && ss->size() == 4);
    auto a = d->open_sensor(m::sensor_kind::accelerometer, 200); CHECK(a);
    if (a) { auto v = a->poll(500, 1); CHECK(v.size() == 1); if (!v.empty()) { CHECK(std::abs(v[0].v[1] + 9.4) < 1e-9); } }
    auto l = d->open_sensor(m::sensor_kind::light, 100); if (l) { auto v = l->poll(500, 1); CHECK(!v.empty() && v[0].v[0] == 321.5); } else CHECK(false);
    auto b = d->open_sensor(m::sensor_kind::battery, 100); if (b) { auto v = b->poll(500, 1); CHECK(!v.empty() && v[0].v[0] == 77 && v[0].v[1] == 1); } else CHECK(false);
    auto disp = d->display(); CHECK(disp && disp->width_px == 720 && disp->orient == m::orientation::portrait_upside_down);
    auto cams = d->cameras(); CHECK(cams && cams->size() == 1 && cams->at(0).front);
    CHECK(d->permission_status(m::permission::camera).value == m::permission_state::not_required);
    CHECK(is_unsupported(d->open_sensor(m::sensor_kind::location).err));

    // evdev multitouch protocol B through a FIFO standing in for /dev/input/event0
    auto node = root / "input/event0"; CHECK(::mkfifo(node.c_str(), 0600) == 0);
    auto t = d->open_touch(); CHECK(t);
    if (t) {
        int w = -1; for (int i = 0; i < 100 && w < 0; ++i) { w = ::open(node.c_str(), O_WRONLY | O_NONBLOCK); if (w < 0) ::usleep(10000); }
        CHECK(w >= 0);
        auto ev = [&](int type, int code, int val, int ms) { input_event e{}; e.input_event_sec = 0; e.input_event_usec = ms * 1000; e.type = std::uint16_t(type); e.code = std::uint16_t(code); e.value = val; (void)!::write(w, &e, sizeof e); };
        ev(EV_ABS, ABS_MT_SLOT, 0, 0); ev(EV_ABS, ABS_MT_TRACKING_ID, 5, 0); ev(EV_ABS, ABS_MT_POSITION_X, 10, 0); ev(EV_ABS, ABS_MT_POSITION_Y, 20, 0); ev(EV_SYN, SYN_REPORT, 0, 0);
        ev(EV_ABS, ABS_MT_SLOT, 1, 10); ev(EV_ABS, ABS_MT_TRACKING_ID, 6, 10); ev(EV_ABS, ABS_MT_POSITION_X, 30, 10); ev(EV_ABS, ABS_MT_POSITION_Y, 40, 10); ev(EV_SYN, SYN_REPORT, 0, 10);
        ev(EV_ABS, ABS_MT_SLOT, 0, 20); ev(EV_ABS, ABS_MT_TRACKING_ID, -1, 20); ev(EV_SYN, SYN_REPORT, 0, 20);
        ::close(w);
        std::vector<m::touch_event> all; for (int i = 0; i < 100 && !t->finished(); ++i) for (auto& e : t->poll(20)) all.push_back(e);
        CHECK(all.size() == 3);
        if (all.size() == 3) {
            CHECK(all[0].phase == m::touch_phase::down && all[0].id == 5 && all[0].x == 10 && all[0].y == 20);
            CHECK(all[1].phase == m::touch_phase::down && all[1].id == 6 && all[1].pointers == 2);
            CHECK(all[2].phase == m::touch_phase::up && all[2].id == 5 && all[2].t_ns == 20000000);
        }
    }
    fs::remove_all(root);
}
#endif

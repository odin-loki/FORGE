// forge_mobile: shared pieces (names, queues, streams, gestures, device, bridge), null and simulated backends.
#include "internal.hpp"
#include <forge/json/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace forge::os::mobile {
using namespace detail;

// ------------------------------------------------------------------ names
static constexpr const char* k_sensor[] = {"accelerometer", "gyroscope", "magnetometer", "light", "proximity", "pressure", "location", "battery",
    "step_counter", "step_detector", "rotation_vector", "gravity", "linear_acceleration", "heart_rate", "temperature", "humidity"};
const char* sensor_name(sensor_kind k) { return k_sensor[int(k)]; }
std::optional<sensor_kind> parse_sensor(std::string_view s) { for (int i = 0; i < sensor_kind_count; ++i) if (s == k_sensor[i]) return sensor_kind(i); return std::nullopt; }
static constexpr const char* k_phase[] = {"down", "move", "up", "cancel"};
const char* touch_phase_name(touch_phase p) { return k_phase[int(p)]; }
static std::optional<touch_phase> parse_phase(std::string_view s) { for (int i = 0; i < 4; ++i) if (s == k_phase[i]) return touch_phase(i); return std::nullopt; }
static constexpr const char* k_gesture[] = {"tap", "double_tap", "long_press", "swipe", "pinch", "rotate"};
const char* gesture_name(gesture_kind g) { return k_gesture[int(g)]; }
static constexpr const char* k_orient[] = {"unknown", "portrait", "portrait_upside_down", "landscape_left", "landscape_right"};
const char* orientation_name(orientation o) { return k_orient[int(o)]; }
std::optional<orientation> parse_orientation(std::string_view s) {
    if (s == "landscape") return orientation::landscape_left;
    for (int i = 0; i < 5; ++i) if (s == k_orient[i]) return orientation(i); return std::nullopt;
}
static constexpr const char* k_perm[] = {"sensors", "location", "camera", "notifications", "vibrate"};
const char* permission_name(permission p) { return k_perm[int(p)]; }
std::optional<permission> parse_permission(std::string_view s) { for (int i = 0; i < 5; ++i) if (s == k_perm[i]) return permission(i); return std::nullopt; }
static constexpr const char* k_pstate[] = {"granted", "denied", "not_determined", "not_required"};
const char* permission_state_name(permission_state s) { return k_pstate[int(s)]; }
static std::optional<permission_state> parse_pstate(std::string_view s) { for (int i = 0; i < 4; ++i) if (s == k_pstate[i]) return permission_state(i); return std::nullopt; }

namespace detail {
std::pmr::memory_resource* mobile_resource() {
    static forge::mem::budget_resource r(forge::mem::host_budget(), "mobile");
    return &r;
}
bridge& bridge::get() { static bridge b; return b; }

status backend::vibrate(int ms, double amp) {
    auto& br = bridge::get(); std::function<bool(int, double)> f; { std::lock_guard g(br.m); f = br.hooks.vibrate; }
    if (f) return f(ms, amp) ? status{} : status{{}, {-1, "vibrate: app hook failed"}};
    return {{}, unsupported("vibration")};
}
result<display_info> backend::display() {
    auto& br = bridge::get(); std::lock_guard g(br.m);
    if (br.disp) return {*br.disp, {}};
    return {{}, unsupported("display metrics")};
}
result<permission_state> backend::permission_status(permission p) {
    auto& br = bridge::get(); std::lock_guard g(br.m);
    if (auto& s = br.perms[int(p)]) return {*s, {}};
    return {{}, unsupported(std::string("permission ") + permission_name(p))};
}
status backend::request_permission(permission p) {
    auto& br = bridge::get(); std::function<bool(permission)> f; { std::lock_guard g(br.m); f = br.hooks.request_permission; }
    if (f) return f(p) ? status{} : status{{}, {-1, "request_permission: app hook failed"}};
    return {{}, unsupported("permission requests")};
}
status backend::notify(std::string_view t, std::string_view b) {
    auto& br = bridge::get(); std::function<bool(std::string_view, std::string_view)> f; { std::lock_guard g(br.m); f = br.hooks.notify; }
    if (f) return f(t, b) ? status{} : status{{}, {-1, "notify: app hook failed"}};
    return {{}, unsupported("notifications")};
}
} // namespace detail

// ------------------------------------------------------------------ bridge entry points
void feed_touch(const touch_event& e) {
    auto& br = bridge::get(); std::lock_guard g(br.m);
    std::erase_if(br.touch, [&](auto& w) { auto q = w.lock(); if (!q) return true; q->push({true, {}, e}); return false; });
}
void feed_sensor(const sensor_event& e) {
    auto& br = bridge::get(); std::lock_guard g(br.m);
    std::erase_if(br.sensor, [&](auto& p) { auto q = p.second.lock(); if (!q) return true; if (p.first == e.kind) q->push({false, e, {}}); return false; });
}
void report_display(const display_info& d) { auto& br = bridge::get(); std::lock_guard g(br.m); br.disp = d; }
void report_permission(permission p, permission_state s) { auto& br = bridge::get(); std::lock_guard g(br.m); br.perms[int(p)] = s; }
void set_action_hooks(action_hooks h) { auto& br = bridge::get(); std::lock_guard g(br.m); br.hooks = std::move(h); }

// ------------------------------------------------------------------ streams
sensor_stream::sensor_stream(sensor_stream&&) noexcept = default;
sensor_stream& sensor_stream::operator=(sensor_stream&&) noexcept = default;
sensor_stream::~sensor_stream() { keep_.reset(); }   // stop the producer before the queue goes
std::vector<sensor_event> sensor_stream::poll(int timeout_ms, std::size_t max) {
    std::vector<sensor_event> v; if (!q_) return v;
    q_->drain(timeout_ms, max, [&](item& it) { if (!it.touch) v.push_back(it.s); }); return v;
}
std::uint64_t sensor_stream::dropped() const { return q_ ? q_->dropped.load() : 0; }
bool sensor_stream::finished() const { if (!q_) return true; std::lock_guard g(q_->m); return q_->finished && q_->q.empty(); }

touch_stream::touch_stream(touch_stream&&) noexcept = default;
touch_stream& touch_stream::operator=(touch_stream&&) noexcept = default;
touch_stream::~touch_stream() { keep_.reset(); }
std::vector<touch_event> touch_stream::poll(int timeout_ms, std::size_t max) {
    std::vector<touch_event> v; if (!q_) return v;
    q_->drain(timeout_ms, max, [&](item& it) { if (it.touch) v.push_back(it.t); }); return v;
}
std::vector<gesture> touch_stream::gestures(int timeout_ms) {
    std::vector<gesture> out;
    for (auto& e : poll(timeout_ms)) for (auto& g : rec_.feed(e)) out.push_back(g);
    return out;
}
std::uint64_t touch_stream::dropped() const { return q_ ? q_->dropped.load() : 0; }
bool touch_stream::finished() const { if (!q_) return true; std::lock_guard g(q_->m); return q_->finished && q_->q.empty(); }

// ------------------------------------------------------------------ gestures
std::vector<gesture> gesture_recognizer::feed(const touch_event& e) {
    std::vector<gesture> out;
    auto find = [&](int id) { return std::find_if(live_.begin(), live_.end(), [&](const pt& p) { return p.id == id; }); };
    auto dist2 = [&] { double dx = live_[0].x - live_[1].x, dy = live_[0].y - live_[1].y; return std::sqrt(dx * dx + dy * dy); };
    switch (e.phase) {
    case touch_phase::down:
        if (auto it = find(e.id); it != live_.end()) live_.erase(it);
        live_.push_back({e.id, e.x, e.y, e.x, e.y, e.t_ns, false});
        if (live_.size() == 2) { multi_ = true; pinch0_ = dist2(); pinch_last_ = 1; angle0_ = std::atan2(live_[1].y - live_[0].y, live_[1].x - live_[0].x); angle_last_ = 0; }
        break;
    case touch_phase::move: {
        auto it = find(e.id); if (it == live_.end()) break;
        it->x = e.x; it->y = e.y;
        if (live_.size() == 2 && pinch0_ > 0) {
            double s = dist2() / pinch0_;
            double cx = (live_[0].x + live_[1].x) / 2, cy = (live_[0].y + live_[1].y) / 2;
            if (std::fabs(s - pinch_last_) >= 0.05) { pinch_last_ = s; out.push_back({gesture_kind::pinch, e.t_ns, cx, cy, 0, 0, s}); }
            constexpr double pi = 3.14159265358979323846;
            double a = std::atan2(live_[1].y - live_[0].y, live_[1].x - live_[0].x) - angle0_;
            while (a > pi) a -= 2 * pi; while (a < -pi) a += 2 * pi;
            if (std::fabs(a - angle_last_) >= 0.1) { angle_last_ = a; out.push_back({gesture_kind::rotate, e.t_ns, cx, cy, 0, 0, s, a}); }
        } else if (!multi_ && !it->long_fired && std::hypot(it->x - it->x0, it->y - it->y0) <= c_.tap_slop_px && e.t_ns - it->t0 >= c_.long_press_ns) {
            it->long_fired = true; out.push_back({gesture_kind::long_press, e.t_ns, it->x, it->y});
        }
        break;
    }
    case touch_phase::up: {
        auto it = find(e.id); if (it == live_.end()) break;
        it->x = e.x; it->y = e.y;
        double dx = it->x - it->x0, dy = it->y - it->y0, d = std::hypot(dx, dy); std::uint64_t dt = e.t_ns - it->t0;
        if (!multi_ && !it->long_fired) {
            if (d >= c_.swipe_min_px) out.push_back({gesture_kind::swipe, e.t_ns, it->x0, it->y0, dx, dy});
            else if (d <= c_.tap_slop_px && dt >= c_.long_press_ns) out.push_back({gesture_kind::long_press, e.t_ns, it->x, it->y});
            else if (d <= c_.tap_slop_px && dt <= c_.tap_max_ns) {
                out.push_back({gesture_kind::tap, e.t_ns, it->x, it->y});
                if (have_tap_ && e.t_ns - last_tap_t_ <= c_.double_tap_ns && std::hypot(it->x - last_tap_x_, it->y - last_tap_y_) <= c_.tap_slop_px * 2) {
                    out.push_back({gesture_kind::double_tap, e.t_ns, it->x, it->y}); have_tap_ = false;
                } else { have_tap_ = true; last_tap_t_ = e.t_ns; last_tap_x_ = it->x; last_tap_y_ = it->y; }
            }
        }
        live_.erase(it);
        if (live_.size() < 2) pinch0_ = 0;
        if (live_.empty()) multi_ = false;
        break;
    }
    case touch_phase::cancel: live_.clear(); multi_ = false; pinch0_ = 0; break;
    }
    return out;
}

// ------------------------------------------------------------------ null backend
namespace detail {
struct null_backend final : backend { std::string name() const override { return "null"; } };
std::shared_ptr<backend> make_null() { return std::make_shared<null_backend>(); }

// ------------------------------------------------------------------ simulated backend (JSON trace)
struct sim_backend final : backend, std::enable_shared_from_this<sim_backend> {
    struct ev { std::uint64_t t_ns; item it; std::optional<orientation> orient; std::optional<app_event> life; };
    json::value services = json::object{}; double bright = 0.5; std::string clip; std::shared_ptr<worker> life_w; bool life_started = false;
    bool realtime = false;
    std::vector<sensor_info> sens; std::vector<camera_info> cams; std::optional<display_info> disp;
    std::array<permission_state, 5> perms{permission_state::granted, permission_state::granted, permission_state::granted, permission_state::granted, permission_state::granted};
    std::vector<ev> events; std::uint64_t t0 = now_ns();
    mutable std::mutex m; std::vector<sim_record> log;

    std::string name() const override { return "sim"; }
    result<std::vector<sensor_info>> sensors() override { return {sens, {}}; }
    bool has(sensor_kind k) const { return std::any_of(sens.begin(), sens.end(), [&](auto& s) { return s.kind == k; }); }
    template <class Pred> std::shared_ptr<worker> replay(std::shared_ptr<queue> q, Pred pred) {
        auto w = std::make_shared<worker>(); auto self = shared_from_this();
        w->start([self, q, pred, wp = w.get()] {
            std::uint64_t start = now_ns(), base = self->events.empty() ? 0 : self->events.front().t_ns;
            for (auto& e : self->events) {
                if (wp->stop) return;
                if (!pred(e.it) || e.orient || e.life) continue;
                if (self->realtime) { auto due = start + (e.t_ns - base), n = now_ns(); if (due > n && !wp->sleep_ns(due - n)) return; }
                q->push(e.it);
            }
            q->finish();
        });
        return w;
    }
    result<sensor_stream> open_sensor(sensor_kind k, double) override {
        if (!has(k)) return {{}, unsupported(std::string("sensor ") + sensor_name(k) + " (not in trace)")};
        { std::lock_guard g(m); if (perms[k == sensor_kind::location ? 1 : 0] == permission_state::denied) return {{}, {13, std::string("sensor ") + sensor_name(k) + ": permission denied"}}; }
        auto q = std::make_shared<queue>(); bridge::get().add_sensor(k, q);
        auto w = replay(q, [k](const item& it) { return !it.touch && it.s.kind == k; });
        return {make_sensor(k, q, w), {}};
    }
    result<touch_stream> open_touch() override {
        auto q = std::make_shared<queue>(); bridge::get().add_touch(q);
        auto w = replay(q, [](const item& it) { return it.touch; });
        return {make_touch(q, w), {}};
    }
    status vibrate(int ms, double amp) override {
        std::lock_guard g(m);
        if (perms[int(permission::vibrate)] == permission_state::denied) return {{}, {13, "vibrate: permission denied"}};
        log.push_back({"vibrate", std::to_string(ms) + "ms@" + std::to_string(amp)}); return {};
    }
    result<display_info> display() override {
        if (!disp) return {{}, unsupported("display metrics (not in trace)")};
        display_info d = *disp; std::uint64_t el = now_ns() - t0, base = events.empty() ? 0 : events.front().t_ns;
        for (auto& e : events) if (e.orient && (!realtime || e.t_ns - base <= el)) d.orient = *e.orient;
        if ((d.orient == orientation::landscape_left || d.orient == orientation::landscape_right) != (d.width_px > d.height_px)) std::swap(d.width_px, d.height_px);
        return {d, {}};
    }
    result<std::vector<camera_info>> cameras() override { return {cams, {}}; }
    result<permission_state> permission_status(permission p) override { std::lock_guard g(m); return {perms[int(p)], {}}; }
    status request_permission(permission p) override {
        std::lock_guard g(m); log.push_back({"request_permission", permission_name(p)});
        if (perms[int(p)] == permission_state::not_determined) perms[int(p)] = permission_state::granted;
        return perms[int(p)] == permission_state::denied ? status{{}, {13, std::string(permission_name(p)) + ": permission denied"}} : status{};
    }
    status notify(std::string_view t, std::string_view b) override {
        std::lock_guard g(m);
        if (perms[int(permission::notifications)] == permission_state::denied) return {{}, {13, "notify: permission denied"}};
        log.push_back({"notify", std::string(t) + ": " + std::string(b)}); return {};
    }
    std::vector<sim_record> sim_log() const override { std::lock_guard g(m); return log; }
    ~sim_backend() override { life_w.reset(); }
    std::optional<std::string> svc(std::string_view op, std::string_view args) override {
        std::lock_guard g(m);
        auto deny = [&](permission p) { return perms[int(p)] == permission_state::denied; };
        auto act = [&](std::optional<permission> p = std::nullopt) -> std::optional<std::string> {
            if (p && deny(*p)) return std::string("{\"error\":\"") + std::string(op) + ": permission denied\",\"code\":13}";
            log.push_back({std::string(op), std::string(args)}); return std::string("true");
        };
        if (op == "clipboard_get") return json::dump(json::value(clip));
        if (op == "clipboard_set") { clip = json::parse(args)["text"].str(); return act(); }
        if (op == "brightness") return std::to_string(bright);
        if (op == "set_brightness") { bright = json::parse(args)["level"].num(); return act(); }
        if (op == "torch" || op == "capture_photo") {
            if (cams.empty()) return std::string("{\"unsupported\":\"camera (none in trace)\"}");
            if (op == "torch") return act(permission::camera);
            if (deny(permission::camera)) return act(permission::camera);
            log.push_back({"capture_photo", std::string(args)});
            if (auto* p = services.find("photo")) return json::dump(*p);
            return std::string("{\"bytes\":[255,216,255,224,0,16,74,70,73,70,0,1,255,217]}");   // minimal JFIF SOI..EOI
        }
        if (op == "record_audio") { if (auto r = act(); *r != "true") return r; if (auto* a = services.find("audio")) return json::dump(*a); return std::string("{}"); }
        if (op == "keep_awake" || op == "lock_orientation" || op == "show_keyboard" || op == "open_url" || op == "share" || op == "play_audio") return act();
        if (op == "haptic_pattern") return act(permission::vibrate);
        if (op == "bluetooth_scan" || op == "gnss") { if (deny(permission::location)) return act(permission::location); }
        if (auto* v = services.find(op)) return json::dump(*v);
        return std::string("{\"unsupported\":\"") + std::string(op) + " (not in trace)\"}";
    }
    void lifecycle_started() override {
        std::lock_guard g(m); if (life_started) return; life_started = true;
        life_w = std::make_shared<worker>();
        life_w->start([this, wp = life_w.get()] {
            std::uint64_t start = now_ns(), base = events.empty() ? 0 : events.front().t_ns;
            for (auto& e : events) {
                if (!e.life) continue;
                if (realtime) { auto due = start + (e.t_ns - base), n = now_ns(); if (due > n && !wp->sleep_ns(due - n)) return; }
                if (wp->stop) return;
                dispatch_lifecycle(*e.life);
            }
        });
    }
};

std::shared_ptr<backend> make_sim(std::string_view text, bool realtime, error& err) {
    auto b = std::make_shared<sim_backend>(); b->realtime = realtime;
    try {
        auto j = json::parse(text);
        if (auto* d = j.find("display")) {
            display_info di; di.width_px = int(d->num_or("width", 0)); di.height_px = int(d->num_or("height", 0));
            di.dpi = d->num_or("dpi", 0); di.scale = d->num_or("scale", di.dpi > 0 ? di.dpi / 160.0 : 1); di.refresh_hz = d->num_or("refresh_hz", 60);
            di.orient = parse_orientation(d->str_or("orientation", "portrait")).value_or(orientation::unknown);
            b->disp = di;
        }
        if (auto* s = j.find("sensors")) for (auto& x : s->arr()) {
            sensor_info si; auto k = parse_sensor(x.is_str() ? x.str() : x["kind"].str());
            if (!k) { err = {22, "trace: unknown sensor kind"}; return nullptr; }
            si.kind = *k; si.name = x.is_obj() ? x.str_or("name", std::string("sim-") + sensor_name(*k)) : std::string("sim-") + sensor_name(*k);
            si.vendor = "forge-sim"; si.max_rate_hz = x.is_obj() ? x.num_or("max_rate_hz", 200) : 200; b->sens.push_back(si);
        }
        if (auto* c = j.find("cameras")) for (auto& x : c->arr()) b->cams.push_back({x.str_or("id", "0"), x.str_or("name", "sim-camera"), x.find("front") && x["front"].b()});
        if (auto* p = j.find("permissions")) for (auto& [k, v] : p->obj()) {
            auto pk = parse_permission(k); auto ps = parse_pstate(v.str());
            if (!pk || !ps) { err = {22, "trace: bad permission entry '" + k + "'"}; return nullptr; }
            b->perms[int(*pk)] = *ps;
        }
        if (auto* sv = j.find("services")) { if (!sv->is_obj()) { err = {22, "trace: services must be an object"}; return nullptr; } b->services = *sv;
            if (auto* c = sv->find("clipboard")) b->clip = c->str(); if (auto* br = sv->find("brightness")) b->bright = br->num(); }
        if (auto* e = j.find("events")) for (auto& x : e->arr()) {
            sim_backend::ev ev{}; ev.t_ns = std::uint64_t(x.num_or("t_ms", 0) * 1e6);
            auto type = x.str_or("type", "");
            if (type == "sensor") {
                auto k = parse_sensor(x["kind"].str()); if (!k) { err = {22, "trace: unknown sensor '" + x["kind"].str() + "'"}; return nullptr; }
                ev.it.s.kind = *k; ev.it.s.t_ns = ev.t_ns; auto& v = x["v"].arr(); ev.it.s.n = int(std::min<std::size_t>(v.size(), 4));
                for (int i = 0; i < ev.it.s.n; ++i) ev.it.s.v[i] = v[i].num();
                if (!b->has(*k)) b->sens.push_back({*k, std::string("sim-") + sensor_name(*k), "forge-sim", 200, 0, 0});
            } else if (type == "touch") {
                auto ph = parse_phase(x.str_or("phase", "")); if (!ph) { err = {22, "trace: bad touch phase"}; return nullptr; }
                ev.it.touch = true; ev.it.t = {ev.t_ns, *ph, int(x.num_or("id", 0)), x.num_or("x", 0), x.num_or("y", 0), x.num_or("pressure", 1), int(x.num_or("pointers", 1))};
            } else if (type == "orientation") {
                auto o = parse_orientation(x.str_or("value", "")); if (!o) { err = {22, "trace: bad orientation"}; return nullptr; }
                ev.orient = *o;
            } else if (type == "lifecycle") {
                auto l = parse_app_event(x.str_or("value", "")); if (!l) { err = {22, "trace: bad lifecycle value"}; return nullptr; }
                ev.life = *l;
            } else { err = {22, "trace: unknown event type '" + type + "'"}; return nullptr; }
            b->events.push_back(ev);
        }
        std::stable_sort(b->events.begin(), b->events.end(), [](auto& a, auto& c) { return a.t_ns < c.t_ns; });
    } catch (const std::exception& ex) { err = {22, std::string("trace: ") + ex.what()}; return nullptr; }
    return b;
}
} // namespace detail

// ------------------------------------------------------------------ device
device::device(device&&) noexcept = default;
device& device::operator=(device&&) noexcept = default;
device::~device() = default;

result<device> device::null_device() { device d; d.b_ = make_null(); return {std::move(d), {}}; }
result<device> device::simulated(std::string_view json, bool realtime) {
    error e; auto b = make_sim(json, realtime, e); if (!b) return {{}, e};
    device d; d.b_ = std::move(b); return {std::move(d), {}};
}
result<device> device::simulated_file(const std::string& path, bool realtime) {
    std::ifstream f(path, std::ios::binary); if (!f) return {{}, {2, "trace: cannot open " + path}};
    std::stringstream ss; ss << f.rdbuf(); return simulated(ss.str(), realtime);
}
#if defined(__linux__)
result<device> device::linux_with(std::string sysfs, std::string input) {
#if defined(__ANDROID__)
    (void)sysfs; (void)input; return {{}, unsupported("linux_with on Android")};
#else
    device d; d.b_ = make_linux(std::move(sysfs), std::move(input)); return {std::move(d), {}};
#endif
}
#endif
result<device> device::open() {
    if (const char* t = std::getenv("FORGE_MOBILE_TRACE"); t && *t) return simulated_file(t, true);
    device d;
#if defined(__ANDROID__)
    d.b_ = make_android();
#elif defined(__APPLE__)
    d.b_ = make_apple();
#elif defined(_WIN32)
    d.b_ = make_windows();
#elif defined(__FreeBSD__)
    d.b_ = make_freebsd();
#elif defined(__linux__)
    const char* s = std::getenv("FORGE_MOBILE_SYSFS"); const char* i = std::getenv("FORGE_MOBILE_INPUT");
    d.b_ = make_linux(s && *s ? s : "/sys", i && *i ? i : "/dev/input");
#else
    d.b_ = make_null();
#endif
    return {std::move(d), {}};
}

#define FORGE_MOBILE_NEED(ret) if (!b_) return {{}, {22, "mobile: empty device"}}
std::string device::backend_name() const { return b_ ? b_->name() : "none"; }
result<std::vector<sensor_info>> device::sensors() { FORGE_MOBILE_NEED(); return b_->sensors(); }
bool device::has_sensor(sensor_kind k) { auto s = sensors(); return s && std::any_of(s.value.begin(), s.value.end(), [&](auto& x) { return x.kind == k; }); }
result<sensor_stream> device::open_sensor(sensor_kind k, double hz) { FORGE_MOBILE_NEED(); if (!(hz > 0) || hz > 10000) return {{}, {22, "open_sensor: rate_hz must be in (0, 10000]"}}; return b_->open_sensor(k, hz); }
result<touch_stream> device::open_touch() { FORGE_MOBILE_NEED(); return b_->open_touch(); }
status device::vibrate(int ms, double amp) { FORGE_MOBILE_NEED(); if (ms <= 0 || ms > 60000 || amp < 0 || amp > 1) return {{}, {22, "vibrate: ms in (0, 60000], amplitude in [0, 1]"}}; return b_->vibrate(ms, amp); }
result<display_info> device::display() { FORGE_MOBILE_NEED(); return b_->display(); }
result<orientation> device::screen_orientation() { FORGE_MOBILE_NEED(); return b_->screen_orientation(); }
result<std::vector<camera_info>> device::cameras() { FORGE_MOBILE_NEED(); return b_->cameras(); }
result<permission_state> device::permission_status(permission p) { FORGE_MOBILE_NEED(); return b_->permission_status(p); }
status device::request_permission(permission p) { FORGE_MOBILE_NEED(); return b_->request_permission(p); }
status device::notify(std::string_view t, std::string_view body) { FORGE_MOBILE_NEED(); return b_->notify(t, body); }
std::vector<sim_record> device::sim_log() const { return b_ ? b_->sim_log() : std::vector<sim_record>{}; }

} // namespace forge::os::mobile

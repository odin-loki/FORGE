#pragma once
// forge::os wall-clock time: Unix time, local UTC offset, ISO-8601 / RFC-3339 formatting and parsing, broken-down
// civil time, and a timer_queue that runs callbacks after a delay or periodically on one background thread.
// (Monotonic time and sleep live in os.hpp.) Guide: docs/os-wrapper.md.
#include <forge/os/os.hpp>
#include <condition_variable>
#include <ctime>
#include <map>
#include <mutex>
#include <thread>

namespace forge::os {
inline std::int64_t unix_time_ns() { return std::int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count()); }
inline std::int64_t unix_time() { return unix_time_ns() / 1000000000; }

struct civil_time { int year = 1970, month = 1, day = 1, hour = 0, minute = 0, second = 0, nanos = 0, weekday = 4, yearday = 0; int offset_s = 0; };
// Days since 1970-01-01 <-> civil date (proleptic Gregorian; Howard Hinnant's algorithms).
inline std::int64_t days_from_civil(int y, int m, int d) {
    y -= m <= 2; std::int64_t era = (y >= 0 ? y : y - 399) / 400; unsigned yoe = unsigned(y - era * 400);
    unsigned doy = unsigned((153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1); unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + std::int64_t(doe) - 719468;
}
inline civil_time civil_from_unix_ns(std::int64_t ns, int offset_s = 0) {
    std::int64_t s = (ns >= 0 ? ns / 1000000000 : (ns - 999999999) / 1000000000) + offset_s;
    civil_time c; c.nanos = int(ns - (s - offset_s) * 1000000000); c.offset_s = offset_s;
    std::int64_t z = (s >= 0 ? s / 86400 : (s - 86399) / 86400), sod = s - z * 86400;
    c.hour = int(sod / 3600); c.minute = int(sod % 3600 / 60); c.second = int(sod % 60);
    c.weekday = int(((z % 7) + 11) % 7);   // 1970-01-01 was a Thursday (4)
    std::int64_t zz = z + 719468, era = (zz >= 0 ? zz : zz - 146096) / 146097; unsigned doe = unsigned(zz - era * 146097);
    unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; std::int64_t y = std::int64_t(yoe) + era * 400;
    unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100), mp = (5 * doy + 2) / 153;
    c.day = int(doy - (153 * mp + 2) / 5 + 1); c.month = int(mp < 10 ? mp + 3 : mp - 9); c.year = int(y + (c.month <= 2));
    c.yearday = int(z - days_from_civil(c.year, 1, 1));
    return c;
}
inline std::int64_t unix_ns_from_civil(const civil_time& c) {
    return ((days_from_civil(c.year, c.month, c.day) * 86400 + c.hour * 3600 + c.minute * 60 + c.second) - c.offset_s) * 1000000000 + c.nanos;
}
// The local zone's offset from UTC (seconds east) at `unix_s` (DST-aware).
inline int utc_offset_seconds(std::int64_t unix_s) {
    std::time_t t = std::time_t(unix_s); std::tm l{}, g{};
#if defined(FORGE_OS_WINDOWS)
    localtime_s(&l, &t); gmtime_s(&g, &t);
#else
    ::localtime_r(&t, &l); ::gmtime_r(&t, &g);
#endif
    std::int64_t a = days_from_civil(l.tm_year + 1900, l.tm_mon + 1, l.tm_mday) * 86400 + l.tm_hour * 3600 + l.tm_min * 60 + l.tm_sec;
    std::int64_t b = days_from_civil(g.tm_year + 1900, g.tm_mon + 1, g.tm_mday) * 86400 + g.tm_hour * 3600 + g.tm_min * 60 + g.tm_sec;
    return int(a - b);
}
inline int utc_offset_seconds() { return utc_offset_seconds(unix_time()); }
inline civil_time local_time(std::int64_t ns) { return civil_from_unix_ns(ns, utc_offset_seconds(ns / 1000000000)); }
inline std::string timezone_name() {
    auto tz = getenv("TZ"); if (!tz.empty()) return tz[0] == ':' ? tz.substr(1) : tz;
#if defined(FORGE_OS_POSIX)
    char b[512]; auto n = ::readlink("/etc/localtime", b, sizeof b - 1);
    if (n > 0) { std::string s(b, std::size_t(n)); auto k = s.find("zoneinfo/"); if (k != std::string::npos) return s.substr(k + 9); }
    if (auto f = read_file("/etc/timezone")) { auto s = f.value; while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back(); if (!s.empty()) return s; }
#endif
    return "";
}

// "2026-10-30T09:15:00.123Z" (UTC) or "2026-10-30T20:15:00.123+11:00". digits = fractional-second digits (0..9).
inline std::string format_iso8601(std::int64_t ns, int offset_s = 0, int digits = 3) {
    auto c = civil_from_unix_ns(ns, offset_s); char b[64];
    std::snprintf(b, sizeof b, "%04d-%02d-%02dT%02d:%02d:%02d", c.year, c.month, c.day, c.hour, c.minute, c.second);
    std::string s = b;
    if (digits > 0) { char f[16]; std::snprintf(f, sizeof f, "%09d", c.nanos); s += '.'; s.append(f, std::size_t(digits > 9 ? 9 : digits)); }
    if (offset_s == 0) s += 'Z';
    else { int o = offset_s < 0 ? -offset_s : offset_s; std::snprintf(b, sizeof b, "%c%02d:%02d", offset_s < 0 ? '-' : '+', o / 3600, o % 3600 / 60); s += b; }
    return s;
}
inline std::string format_local_iso8601(std::int64_t ns, int digits = 3) { return format_iso8601(ns, utc_offset_seconds(ns / 1000000000), digits); }
// Parses YYYY-MM-DD[(T| )HH:MM[:SS[.frac]]][Z|±HH[:MM]]. No zone means UTC. Returns Unix ns.
inline result<std::int64_t> parse_iso8601(std::string_view s) {
    civil_time c; std::size_t i = 0;
    auto num = [&](int n, int& out) { if (i + std::size_t(n) > s.size()) return false; int v = 0; for (int k = 0; k < n; ++k) { char ch = s[i + std::size_t(k)]; if (ch < '0' || ch > '9') return false; v = v * 10 + (ch - '0'); } out = v; i += std::size_t(n); return true; };
    auto lit = [&](char ch) { if (i < s.size() && s[i] == ch) { ++i; return true; } return false; };
    auto bad = [&] { return result<std::int64_t>{0, error{EINVAL, "parse_iso8601: bad timestamp '" + std::string(s) + "'"}}; };
    if (!num(4, c.year) || !lit('-') || !num(2, c.month) || !lit('-') || !num(2, c.day)) return bad();
    if (i < s.size() && (s[i] == 'T' || s[i] == 't' || s[i] == ' ')) {
        ++i; if (!num(2, c.hour) || !lit(':') || !num(2, c.minute)) return bad();
        if (lit(':')) { if (!num(2, c.second)) return bad();
            if (lit('.') || lit(',')) { int k = 0; long long f = 0; while (i < s.size() && s[i] >= '0' && s[i] <= '9') { if (k < 9) { f = f * 10 + (s[i] - '0'); ++k; } ++i; } if (!k) return bad(); while (k++ < 9) f *= 10; c.nanos = int(f); } }
        if (lit('Z') || lit('z')) {}
        else if (i < s.size() && (s[i] == '+' || s[i] == '-')) { int sg = s[i++] == '-' ? -1 : 1, h = 0, m = 0; if (!num(2, h)) return bad(); lit(':'); if (i < s.size()) num(2, m); c.offset_s = sg * (h * 3600 + m * 60); }
    }
    if (i != s.size() || c.month < 1 || c.month > 12 || c.day < 1 || c.day > 31 || c.hour > 23 || c.minute > 59 || c.second > 60) return bad();
    return {unix_ns_from_civil(c), {}};
}

// ---------------------------------------------------------------- timer queue
// One worker thread; callbacks run on it in deadline order. after()/every() return an id for cancel().
class timer_queue {
public:
    using id = std::uint64_t;
    timer_queue() : th_([this] { run(); }) {}
    ~timer_queue() { { std::lock_guard<std::mutex> g(m_); stop_ = true; } cv_.notify_all(); th_.join(); }
    timer_queue(const timer_queue&) = delete; timer_queue& operator=(const timer_queue&) = delete;
    id after(std::uint64_t delay_ms, std::function<void()> fn) { return add(delay_ms, 0, std::move(fn)); }
    id every(std::uint64_t period_ms, std::function<void()> fn, bool fire_now = false) { return add(fire_now ? 0 : period_ms, period_ms ? period_ms : 1, std::move(fn)); }
    bool cancel(id t) { std::lock_guard<std::mutex> g(m_); for (auto it = q_.begin(); it != q_.end(); ++it) if (it->second.id == t) { q_.erase(it); return true; } if (running_ == t) { running_cancelled_ = true; return true; } return false; }
    std::size_t pending() { std::lock_guard<std::mutex> g(m_); return q_.size(); }
private:
    struct item { id id; std::uint64_t period_ns; std::function<void()> fn; };
    id add(std::uint64_t delay_ms, std::uint64_t period_ms, std::function<void()> fn) {
        std::lock_guard<std::mutex> g(m_); id t = ++next_;
        q_.emplace(monotonic_ns() + delay_ms * 1000000ull, item{t, period_ms * 1000000ull, std::move(fn)});
        cv_.notify_all(); return t;
    }
    void run() {
        std::unique_lock<std::mutex> g(m_);
        while (!stop_) {
            if (q_.empty()) { cv_.wait(g); continue; }
            auto now = monotonic_ns(), due = q_.begin()->first;
            if (due > now) { cv_.wait_for(g, std::chrono::nanoseconds(due - now)); continue; }
            auto it = q_.begin(); item x = std::move(it->second); q_.erase(it);
            running_ = x.id; running_cancelled_ = false;
            g.unlock(); x.fn(); g.lock();
            running_ = 0;
            if (x.period_ns && !running_cancelled_ && !stop_) { auto nxt = due + x.period_ns; auto n2 = monotonic_ns(); if (nxt < n2) nxt = n2; q_.emplace(nxt, std::move(x)); }
        }
    }
    std::mutex m_; std::condition_variable cv_; std::multimap<std::uint64_t, item> q_;
    id next_ = 0, running_ = 0; bool running_cancelled_ = false, stop_ = false;
    std::thread th_;
};
} // namespace forge::os

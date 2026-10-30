// forge::os::locale: time zones. std::chrono::tzdb where the standard library ships one (MSVC), otherwise the
// system zoneinfo (TZif v1-v4 plus the POSIX TZ footer for times past the last transition).
#include "locale_internal.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#if defined(__ANDROID__)
#  include <sys/system_properties.h>
#endif
#if defined(_MSC_VER) && defined(__cpp_lib_chrono) && __cpp_lib_chrono >= 201907L
#  define FORGE_TZ_CHRONO 1
#endif

namespace forge::os::locale {
using namespace detail;
namespace fs = std::filesystem;

namespace {
std::int64_t days_civil(int y, int m, int d) { return days_from_civil(y, m, d); }

// ---- POSIX TZ strings ("AEST-10AEDT,M10.1.0,M4.1.0/3")
struct posix_rule { char kind = 0; int month = 0, week = 0, wday = 0, day = 0; int time = 7200; };
struct posix_tz { std::string std_name, dst_name; int std_off = 0, dst_off = 0; bool has_dst = false; posix_rule start, end; bool ok = false; };
bool parse_name(std::string_view& s, std::string& out) {
    if (!s.empty() && s[0] == '<') { auto e = s.find('>'); if (e == std::string_view::npos) return false; out = s.substr(1, e - 1); s.remove_prefix(e + 1); return true; }
    std::size_t i = 0; while (i < s.size() && std::isalpha((unsigned char)s[i])) ++i;
    if (i < 3) return false; out = s.substr(0, i); s.remove_prefix(i); return true;
}
bool parse_hms(std::string_view& s, int& out, bool allow_sign) {
    int sign = 1;
    if (allow_sign && !s.empty() && (s[0] == '+' || s[0] == '-')) { if (s[0] == '-') sign = -1; s.remove_prefix(1); }
    int parts[3] = {0, 0, 0}, n = 0;
    while (n < 3) {
        std::size_t i = 0; int v = 0; while (i < s.size() && std::isdigit((unsigned char)s[i])) v = v * 10 + (s[i++] - '0');
        if (!i) return n > 0;
        parts[n++] = v; s.remove_prefix(i);
        if (!s.empty() && s[0] == ':') s.remove_prefix(1); else break;
    }
    out = sign * (parts[0] * 3600 + parts[1] * 60 + parts[2]); return true;
}
bool parse_rule(std::string_view& s, posix_rule& r) {
    if (s.empty()) return false;
    if (s[0] == 'M') {
        s.remove_prefix(1); r.kind = 'M';
        if (std::sscanf(std::string(s).c_str(), "%d.%d.%d", &r.month, &r.week, &r.wday) != 3) return false;
        while (!s.empty() && s[0] != '/' && s[0] != ',') s.remove_prefix(1);
    } else {
        r.kind = s[0] == 'J' ? 'J' : 'n'; if (s[0] == 'J') s.remove_prefix(1);
        std::size_t i = 0; while (i < s.size() && std::isdigit((unsigned char)s[i])) r.day = r.day * 10 + (s[i++] - '0'); s.remove_prefix(i);
    }
    if (!s.empty() && s[0] == '/') { s.remove_prefix(1); if (!parse_hms(s, r.time, true)) return false; }
    return true;
}
posix_tz parse_posix(std::string_view s) {
    posix_tz z;
    if (!parse_name(s, z.std_name) || !parse_hms(s, z.std_off, true)) return z;
    z.std_off = -z.std_off;
    if (s.empty()) { z.ok = true; return z; }
    if (!parse_name(s, z.dst_name)) return z;
    z.has_dst = true; z.dst_off = z.std_off + 3600;
    if (!s.empty() && s[0] != ',') { int o; if (!parse_hms(s, o, true)) return z; z.dst_off = -o; }
    if (s.empty()) { z.start = {'M', 3, 2, 0, 0, 7200}; z.end = {'M', 11, 1, 0, 0, 7200}; z.ok = true; return z; }   // US default
    if (s[0] != ',') return z; s.remove_prefix(1);
    if (!parse_rule(s, z.start)) return z;
    if (s.empty() || s[0] != ',') return z; s.remove_prefix(1);
    if (!parse_rule(s, z.end)) return z;
    z.ok = true; return z;
}
// Local seconds (in the offset in force before the transition) of a rule in year y.
std::int64_t rule_time(const posix_rule& r, int y) {
    std::int64_t day;
    if (r.kind == 'M') {
        std::int64_t first = days_civil(y, r.month, 1);
        int wd_first = int(((first % 7) + 11) % 7);   // 1970-01-01 was Thursday (4)
        int d = 1 + (r.wday - wd_first + 7) % 7 + (r.week - 1) * 7;
        static const int mdays[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        int dim = mdays[r.month - 1] + (r.month == 2 && ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0));
        while (d > dim) d -= 7;
        day = first + d - 1;
    } else if (r.kind == 'J') {
        bool leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        day = days_civil(y, 1, 1) + r.day - 1 + (leap && r.day >= 60 ? 1 : 0);
    } else day = days_civil(y, 1, 1) + r.day;
    return day * 86400 + r.time;
}
zone_info posix_at(const posix_tz& z, std::int64_t t) {
    if (!z.has_dst) return {std::chrono::seconds(z.std_off), false, z.std_name};
    auto ymd = civil_from_days((t + z.std_off) / 86400 - ((t + z.std_off) % 86400 < 0));
    int y = ymd[0];
    for (int yy : {y - 1, y, y + 1}) (void)yy;
    std::int64_t s = rule_time(z.start, y) - z.std_off, e = rule_time(z.end, y) - z.dst_off;
    bool dst = s < e ? (t >= s && t < e) : !(t >= e && t < s);   // southern hemisphere wraps the year
    return dst ? zone_info{std::chrono::seconds(z.dst_off), true, z.dst_name} : zone_info{std::chrono::seconds(z.std_off), false, z.std_name};
}

// ---- TZif
struct tzif {
    std::vector<std::int64_t> trans; std::vector<std::uint8_t> idx;
    struct ttype { int off; bool dst; std::string abbr; };
    std::vector<ttype> types; posix_tz footer; bool has_footer = false;
    zone_info at(std::int64_t t) const {
        if (trans.empty() || t < trans.front()) {
            if (trans.empty() && has_footer) return posix_at(footer, t);
            std::size_t k = 0; for (std::size_t i = 0; i < types.size(); ++i) if (!types[i].dst) { k = i; break; }
            if (types.empty()) return {};
            return {std::chrono::seconds(types[k].off), types[k].dst, types[k].abbr};
        }
        if (t >= trans.back() && has_footer) return posix_at(footer, t);
        auto it = std::upper_bound(trans.begin(), trans.end(), t); std::size_t i = std::size_t(it - trans.begin()) - 1;
        auto& ty = types[idx[i]]; return {std::chrono::seconds(ty.off), ty.dst, ty.abbr};
    }
};
std::uint32_t be32(const unsigned char* p) { return std::uint32_t(p[0]) << 24 | std::uint32_t(p[1]) << 16 | std::uint32_t(p[2]) << 8 | p[3]; }
std::int64_t be64(const unsigned char* p) { return std::int64_t(std::uint64_t(be32(p)) << 32 | be32(p + 4)); }
std::optional<tzif> parse_tzif(const std::string& d) {
    auto u = reinterpret_cast<const unsigned char*>(d.data());
    if (d.size() < 44 || d.compare(0, 4, "TZif") != 0) return std::nullopt;
    auto counts = [&](std::size_t at, std::uint32_t c[6]) { for (int i = 0; i < 6; ++i) c[i] = be32(u + at + 20 + 4 * i); };
    std::uint32_t c[6]; counts(0, c);   // isutcnt isstdcnt leapcnt timecnt typecnt charcnt
    std::size_t pos = 44; int tsize = 4; char ver = d[4];
    if (ver >= '2') {
        pos += c[3] * 4 + c[3] + c[4] * 6 + c[5] + c[2] * 8 + c[1] + c[0];
        if (d.size() < pos + 44) return std::nullopt;
        counts(pos, c); pos += 44; tsize = 8;
    }
    tzif z;
    std::size_t need = c[3] * tsize + c[3] + c[4] * 6 + c[5];
    if (d.size() < pos + need) return std::nullopt;
    for (std::uint32_t i = 0; i < c[3]; ++i) z.trans.push_back(tsize == 8 ? be64(u + pos + i * 8) : std::int64_t(std::int32_t(be32(u + pos + i * 4))));
    pos += c[3] * tsize;
    for (std::uint32_t i = 0; i < c[3]; ++i) z.idx.push_back(u[pos + i]);
    pos += c[3];
    std::size_t types_at = pos, abbr_at = pos + c[4] * 6;
    for (std::uint32_t i = 0; i < c[4]; ++i) {
        auto p = u + types_at + i * 6; std::size_t ai = p[5];
        std::string ab; for (std::size_t k = abbr_at + ai; k < abbr_at + c[5] && d[k]; ++k) ab += d[k];
        z.types.push_back({std::int32_t(be32(p)), p[4] != 0, ab});
    }
    for (auto i : z.idx) if (i >= z.types.size()) return std::nullopt;
    pos = abbr_at + c[5] + c[2] * (tsize + 4) + c[1] + c[0];
    if (ver >= '2' && pos < d.size() && d[pos] == '\n') {
        auto e = d.find('\n', pos + 1);
        if (e != std::string::npos) { z.footer = parse_posix(std::string_view(d).substr(pos + 1, e - pos - 1)); z.has_footer = z.footer.ok; }
    }
    return z;
}
std::vector<std::string> tz_dirs() {
    std::vector<std::string> r;
    if (const char* d = std::getenv("TZDIR")) r.push_back(d);
    for (auto d : {"/usr/share/zoneinfo", "/usr/lib/zoneinfo", "/usr/share/lib/zoneinfo", "/var/db/timezone/zoneinfo", "/etc/zoneinfo"}) r.push_back(d);
    return r;
}
std::string tz_dir() {
    static std::string d = [] { for (auto& x : tz_dirs()) { std::error_code ec; if (fs::exists(fs::path(x) / "UTC", ec) || fs::exists(fs::path(x) / "Etc" / "UTC", ec)) return x; } return std::string(); }();
    return d;
}
std::mutex z_mu; std::map<std::string, std::shared_ptr<const tzif>, std::less<>> z_cache;
std::shared_ptr<const tzif> load_zone(std::string_view name) {
    std::lock_guard lk(z_mu);
    if (auto it = z_cache.find(name); it != z_cache.end()) return it->second;
    std::shared_ptr<const tzif> res;
    if (name.find("..") == std::string_view::npos && !name.empty()) {
        std::vector<fs::path> cands;
        if (name.front() == '/') cands.push_back(fs::path(name)); else if (!tz_dir().empty()) cands.push_back(fs::path(tz_dir()) / name);
        for (auto& p : cands) {
            std::ifstream f(p, std::ios::binary); if (!f) continue;
            std::stringstream ss; ss << f.rdbuf();
            if (auto z = parse_tzif(ss.str())) { res = std::make_shared<const tzif>(std::move(*z)); break; }
        }
        if (!res) {   // a bare POSIX TZ string ("EST5EDT,M3.2.0,M11.1.0")
            auto p = parse_posix(name);
            if (p.ok) { tzif z; z.footer = p; z.has_footer = true; res = std::make_shared<const tzif>(std::move(z)); }
        }
    }
    z_cache.emplace(std::string(name), res);
    return res;
}
std::string read_first_line(const char* path) { std::ifstream f(path); std::string l; std::getline(f, l); while (!l.empty() && (l.back() == ' ' || l.back() == '\r')) l.pop_back(); return l; }
}  // namespace

std::string tz_backend() {
#if FORGE_TZ_CHRONO
    return "chrono-tzdb";
#else
    auto d = tz_dir(); return d.empty() ? "utc-only" : "tzif:" + d;
#endif
}
std::string local_zone() {
    if (const char* tz = std::getenv("TZ"); tz && *tz) {
        std::string s = tz[0] == ':' ? tz + 1 : tz;
        if (auto p = s.find("zoneinfo/"); p != std::string::npos) s = s.substr(p + 9);
        return s;
    }
#if FORGE_TZ_CHRONO
    try { return std::string(std::chrono::current_zone()->name()); } catch (...) { return "UTC"; }
#elif defined(__APPLE__)
    if (auto z = apple_local_zone(); !z.empty()) return z;
#elif defined(__ANDROID__)
    char v[PROP_VALUE_MAX] = {}; if (__system_property_get("persist.sys.timezone", v) > 0) return v;
#endif
#if !defined(_WIN32)
    std::error_code ec;
    auto target = fs::read_symlink("/etc/localtime", ec);
    if (!ec) { auto s = target.string(); if (auto p = s.find("zoneinfo/"); p != std::string::npos) return s.substr(p + 9); }
    for (auto f : {"/etc/timezone", "/var/db/zoneinfo"}) { auto l = read_first_line(f); if (!l.empty()) return l; }
    if (auto l = read_first_line("/etc/sysconfig/clock"); l.find("ZONE=") != std::string::npos) { auto v = l.substr(l.find("ZONE=") + 5); std::erase(v, '"'); return v; }
#endif
    return "UTC";
}
std::vector<std::string> zone_names() {
    std::vector<std::string> r;
#if FORGE_TZ_CHRONO
    try { for (auto& z : std::chrono::get_tzdb().zones) r.emplace_back(z.name()); } catch (...) {}
#else
    auto d = tz_dir(); if (d.empty()) return {"UTC"};
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(d, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        auto rel = fs::relative(it->path(), d, ec).generic_string();
        if (it->is_directory(ec)) { if (rel == "posix" || rel == "right" || rel == "SystemV") it.disable_recursion_pending(); continue; }
        if (rel.empty() || !std::isupper((unsigned char)rel[0]) || rel.find('.') != std::string::npos) continue;
        std::ifstream f(it->path(), std::ios::binary); char m[4] = {}; f.read(m, 4);
        if (std::memcmp(m, "TZif", 4) == 0) r.push_back(rel);
    }
    std::sort(r.begin(), r.end());
#endif
    return r;
}
result<zone_info> zone_at(std::string_view zone, sys_seconds t) {
    result<zone_info> r; std::string z = zone.empty() ? local_zone() : std::string(zone);
    if (z == "UTC" || z == "Etc/UTC" || z == "Z" || z == "GMT" || z == "Etc/GMT") { r.value = {std::chrono::seconds(0), false, z == "GMT" ? "GMT" : "UTC"}; return r; }
#if FORGE_TZ_CHRONO
    try {
        auto info = std::chrono::locate_zone(z)->get_info(t);
        r.value = {std::chrono::duration_cast<std::chrono::seconds>(info.offset), info.save != std::chrono::minutes(0), info.abbrev};
        return r;
    } catch (...) {}
#endif
    auto tz = load_zone(z);
    if (!tz) { r.err = {ENOENT, "time zone '" + z + "' not found (" + tz_backend() + ")"}; return r; }
    r.value = tz->at(t.time_since_epoch().count());
    return r;
}
result<sys_seconds> from_local(std::string_view zone, std::int64_t local, bool earlier) {
    result<sys_seconds> r;
    auto off = [&](std::int64_t t) -> std::optional<std::int64_t> { auto z = zone_at(zone, sys_seconds(std::chrono::seconds(t))); if (!z) return std::nullopt; return z.value.offset.count(); };
    auto a = off(local - 86400), b = off(local + 86400);
    if (!a || !b) { auto z = zone_at(zone, sys_seconds{}); r.err = z.err; return r; }
    std::vector<std::int64_t> valid;
    for (auto o : {*a, *b}) { std::int64_t t = local - o; if (off(t) == o && std::find(valid.begin(), valid.end(), t) == valid.end()) valid.push_back(t); }
    std::sort(valid.begin(), valid.end());
    if (valid.empty()) r.value = sys_seconds(std::chrono::seconds(local - *a));   // in a gap: move forward by the gap
    else r.value = sys_seconds(std::chrono::seconds(earlier ? valid.front() : valid.back()));
    return r;
}
}  // namespace forge::os::locale

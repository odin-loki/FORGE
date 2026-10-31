#pragma once
// forge::os::settings: one documented table for every environment variable a program reads.
//
//   inline const auto& kTimeout = settings::define<std::int64_t>("PRISM_TIMEOUT_MS", 30000, "Per-query timeout");
//   auto t = kTimeout.get();                       // parsed, defaulted; re-parsed when the variable changes
//   settings::markdown();                          // the table, for docs
//   settings::undocumented("PRISM_");              // set in the environment but never defined (typos)
//
// Reading a name that was never defined through settings::get_raw() throws, so ad-hoc getenv() calls can
// be replaced one by one and the table stays complete. Kinds: string, int, bool (1/0/true/false/yes/no/
// on/off), size (bytes with K/M/G/T suffixes, base 1024) and real. A value that does not parse falls back
// to the default and is reported by problems().
#include <forge/os/os.hpp>
#include <cctype>
#include <cerrno>
#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#if !defined(_WIN32)
extern char** environ;
#endif

namespace forge::os::settings {
enum class kind { string, integer, boolean, size, real };
struct info { std::string name, help, default_text; kind k; };

namespace detail {
inline std::string lower(std::string s) { for (auto& c : s) c = char(std::tolower((unsigned char)c)); return s; }
inline std::optional<bool> parse_bool(std::string s) {
    s = lower(s);
    if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
    if (s == "0" || s == "false" || s == "no" || s == "off" || s.empty()) return false;
    return std::nullopt;
}
inline std::optional<std::int64_t> parse_int(const std::string& s) {
    if (s.empty()) return std::nullopt; char* e = nullptr; errno = 0; long long v = std::strtoll(s.c_str(), &e, 0);
    if (errno || *e) return std::nullopt; return v;
}
inline std::optional<std::uint64_t> parse_size(const std::string& s) {
    if (s.empty()) return std::nullopt; char* e = nullptr; double v = std::strtod(s.c_str(), &e);
    if (e == s.c_str() || v < 0) return std::nullopt;
    std::string u = lower(e); while (!u.empty() && u.front() == ' ') u.erase(0, 1);
    if (!u.empty() && u.back() == 'b') u.pop_back();
    if (!u.empty() && u.back() == 'i') u.pop_back();
    double m = 1;
    if (u.empty()) m = 1; else if (u == "k") m = 1024.0; else if (u == "m") m = 1048576.0; else if (u == "g") m = 1073741824.0; else if (u == "t") m = 1099511627776.0; else return std::nullopt;
    return std::uint64_t(v * m);
}
inline std::optional<double> parse_real(const std::string& s) { if (s.empty()) return std::nullopt; char* e = nullptr; double v = std::strtod(s.c_str(), &e); if (*e) return std::nullopt; return v; }

struct table {
    std::mutex m;
    std::map<std::string, info> vars;
    std::vector<std::string> problems;
    unsigned generation = 1;
};
inline table& tab() { static table t; return t; }
template <class T> constexpr kind kind_of() {
    if constexpr (std::is_same_v<T, bool>) return kind::boolean;
    else if constexpr (std::is_same_v<T, std::uint64_t>) return kind::size;
    else if constexpr (std::is_integral_v<T>) return kind::integer;
    else if constexpr (std::is_floating_point_v<T>) return kind::real;
    else return kind::string;
}
template <class T> std::string to_text(const T& v) {
    if constexpr (std::is_same_v<T, bool>) return v ? "true" : "false";
    else if constexpr (std::is_arithmetic_v<T>) return std::to_string(v);
    else return std::string(v);
}
} // namespace detail

inline const char* kind_name(kind k) { static const char* n[] = {"string", "int", "bool", "size", "real"}; return n[int(k)]; }

// Register a name (idempotent for identical definitions; a conflicting redefinition throws).
inline void declare(std::string name, kind k, std::string default_text, std::string help) {
    auto& t = detail::tab(); std::lock_guard g(t.m);
    auto [it, fresh] = t.vars.try_emplace(name, info{name, help, default_text, k});
    if (!fresh && (it->second.k != k || it->second.default_text != default_text))
        throw std::logic_error("settings: " + name + " defined twice with different kind/default");
}
inline bool defined(std::string_view name) { auto& t = detail::tab(); std::lock_guard g(t.m); return t.vars.count(std::string(name)) > 0; }
// The raw environment value of a defined name (nullopt if unset). Throws std::logic_error if undefined.
inline std::optional<std::string> get_raw(std::string_view name) {
    if (!defined(name)) throw std::logic_error("settings: " + std::string(name) + " is read but was never defined");
    if (const char* v = std::getenv(std::string(name).c_str())) return std::string(v);
    return std::nullopt;
}
inline void reload() { auto& t = detail::tab(); std::lock_guard g(t.m); ++t.generation; t.problems.clear(); }
inline std::vector<std::string> problems() { auto& t = detail::tab(); std::lock_guard g(t.m); return t.problems; }
inline std::vector<info> all() { auto& t = detail::tab(); std::lock_guard g(t.m); std::vector<info> v; for (auto& [_, i] : t.vars) v.push_back(i); return v; }

// Typed handle. T: std::string, bool, std::int64_t (or any integral), std::uint64_t (= size), double.
template <class T> class setting {
public:
    setting(std::string name, T dflt, std::string help) : name_(std::move(name)), dflt_(std::move(dflt)) {
        declare(name_, detail::kind_of<T>(), detail::to_text(dflt_), std::move(help));
    }
    const std::string& name() const { return name_; }
    const T& default_value() const { return dflt_; }
    bool is_set() const { return std::getenv(name_.c_str()) != nullptr; }
    // The parsed value is cached per raw value: a setenv/unsetenv (a test's
    // environment guard, a parent setting a child's knob) is seen by the next
    // get(), and reload() forces a re-parse (and clears problems()).
    T get() const {
        auto& t = detail::tab();
        const char* e = std::getenv(name_.c_str());
        std::optional<std::string> raw = e ? std::optional<std::string>(e) : std::nullopt;
        { std::lock_guard g(t.m); if (gen_ == t.generation && raw == raw_) return val_; }
        T v = dflt_;
        if (raw) {
            std::string s(*raw); bool ok = true;
            if constexpr (std::is_same_v<T, bool>) { auto p = detail::parse_bool(s); ok = bool(p); if (p) v = *p; }
            else if constexpr (std::is_same_v<T, std::uint64_t>) { auto p = detail::parse_size(s); ok = bool(p); if (p) v = *p; }
            else if constexpr (std::is_integral_v<T>) { auto p = detail::parse_int(s); ok = bool(p); if (p) v = T(*p); }
            else if constexpr (std::is_floating_point_v<T>) { auto p = detail::parse_real(s); ok = bool(p); if (p) v = T(*p); }
            else v = T(s);
            if (!ok) { std::lock_guard g(t.m); t.problems.push_back(name_ + "=" + s + ": not a valid " + kind_name(detail::kind_of<T>()) + ", using " + detail::to_text(dflt_)); }
        }
        std::lock_guard g(t.m); val_ = v; gen_ = t.generation; raw_ = std::move(raw); return v;
    }
    operator T() const { return get(); }
private:
    std::string name_; T dflt_;
    mutable T val_{}; mutable unsigned gen_ = 0; mutable std::optional<std::string> raw_;
};
template <class T> setting<T> define(std::string name, T dflt, std::string help) { return setting<T>(std::move(name), std::move(dflt), std::move(help)); }
inline setting<std::string> define(std::string name, const char* dflt, std::string help) { return setting<std::string>(std::move(name), dflt, std::move(help)); }

// Environment variables that start with `prefix` but were never defined (likely typos or dead settings).
inline std::vector<std::string> undocumented(std::string_view prefix) {
    std::vector<std::string> out;
#if defined(_WIN32)
    char** env = _environ;
#else
    char** env = environ;
#endif
    for (char** e = env; e && *e; ++e) {
        std::string_view kv(*e); auto eq = kv.find('='); auto k = kv.substr(0, eq);
        if (k.starts_with(prefix) && !defined(k)) out.emplace_back(k);
    }
    return out;
}
// Markdown table of every defined setting (optionally only those starting with prefix), sorted by name.
inline std::string markdown(std::string_view prefix = {}) {
    std::string s = "| Variable | Type | Default | Description |\n|---|---|---|---|\n";
    for (auto& i : all()) {
        if (!std::string_view(i.name).starts_with(prefix)) continue;
        s += "| `" + i.name + "` | " + kind_name(i.k) + " | `" + (i.default_text.empty() ? std::string("\"\"") : i.default_text) + "` | " + i.help + " |\n";
    }
    return s;
}
} // namespace forge::os::settings

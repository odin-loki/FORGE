#pragma once
// Leveled, thread-safe logging with module tags and pluggable sinks.
//
//   forge::log::info("loaded {} tensors", n);              // module "forge"
//   forge::log::logger L{"mind"}; L.debug("kv {}", b);     // module "mind"
//
// Threshold: FORGE_LOG="warn" or "info,mind=debug,flow=trace" (first letter of each level counts:
// t[race] d[ebug] i[nfo] w[arn] e[rror] o[ff]; default warn). set_level()/set_module_level() override it.
// Compile-time strip: define FORGE_LOG_MIN to 0..5 (trace..off) and calls below it compile to nothing.
// Sinks: stderr (default), append-to-file, or any callback. Each record is formatted first and written
// as one line under a lock, so concurrent lines never interleave. Format: text (default) or JSON lines
// (FORGE_LOG_FORMAT=json or set_format()).
// A sink must not log itself (the sink lock is held while it runs).
#include <forge/os/os.hpp>
#include <forge/os/time.hpp>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef FORGE_LOG_MIN
#define FORGE_LOG_MIN 0
#endif

namespace forge::log {
enum class level : int { trace, debug, info, warn, error, off };
enum class format { text, json };

struct record {
    level lvl;
    std::string_view module;
    std::string_view message;
    std::int64_t unix_ns;
    std::uint64_t thread;
};
using sink = std::function<void(const record&, std::string_view line)>;   // line = formatted, no newline

inline const char* level_name(level l) {
    static const char* n[] = {"trace", "debug", "info", "warn", "error", "off"};
    return n[int(l) < 0 || int(l) > 5 ? 5 : int(l)];
}
inline level parse_level(std::string_view s, level dflt = level::warn) {
    if (s.empty()) return dflt;
    switch (s[0] | 0x20) {
        case 't': return level::trace; case 'd': return level::debug; case 'i': return level::info;
        case 'w': return level::warn; case 'e': return level::error; case 'o': return level::off;
        default: return dflt;
    }
}

namespace detail {
struct state {
    std::mutex m;                                          // guards everything below except the atomics
    std::atomic<int> global{int(level::warn)};
    std::atomic<int> floor{int(level::warn)};              // min over global and module overrides (fast reject)
    std::vector<std::pair<std::string, level>> modules;
    std::vector<std::pair<int, sink>> sinks;
    int next_id = 1;
    bool use_stderr = true;
    std::atomic<int> fmt{int(format::text)};
    state() {
        if (const char* e = std::getenv("FORGE_LOG")) parse_spec(e);
        if (const char* f = std::getenv("FORGE_LOG_FORMAT"); f && (f[0] | 0x20) == 'j') fmt = int(format::json);
    }
    void recompute() {
        int f = global.load();
        for (auto& [_, l] : modules) f = std::min(f, int(l));
        floor.store(f);
    }
    void parse_spec(std::string_view s) {
        while (!s.empty()) {
            auto c = s.find(','); auto part = s.substr(0, c);
            s = c == std::string_view::npos ? std::string_view{} : s.substr(c + 1);
            auto eq = part.find('=');
            if (eq == std::string_view::npos) global = int(parse_level(part));
            else modules.emplace_back(std::string(part.substr(0, eq)), parse_level(part.substr(eq + 1)));
        }
        recompute();
    }
    level threshold_for(std::string_view mod) {
        std::lock_guard l(m);
        for (auto& [n, lv] : modules) if (n == mod) return lv;
        return level(global.load());
    }
};
inline state& st() { static state s; return s; }

inline void json_escape(std::string& o, std::string_view s) {
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break; case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break; case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
            default: if (c < 0x20) o += std::format("\\u{:04x}", c); else o += char(c);
        }
    }
}
inline std::string format_line(const record& r, format f) {
    static const char tag[] = {'T', 'D', 'I', 'W', 'E', 'O'};
    std::string o;
    if (f == format::json) {
        o = std::format("{{\"ts\":\"{}\",\"level\":\"{}\",\"module\":\"", os::format_iso8601(r.unix_ns), level_name(r.lvl));
        json_escape(o, r.module); o += "\",\"thread\":" + std::to_string(r.thread) + ",\"msg\":\"";
        json_escape(o, r.message); o += "\"}";
    } else {
        o = r.module == "forge" ? std::format("[forge {}] ", tag[int(r.lvl)]) : std::format("[forge:{} {}] ", r.module, tag[int(r.lvl)]);
        o += r.message;
    }
    return o;
}
} // namespace detail

// ---- configuration --------------------------------------------------------------------------------
inline void set_level(level l) { auto& s = detail::st(); std::lock_guard g(s.m); s.global = int(l); s.recompute(); }
inline void set_module_level(std::string_view mod, level l) {
    auto& s = detail::st(); std::lock_guard g(s.m);
    for (auto& [n, lv] : s.modules) if (n == mod) { lv = l; s.recompute(); return; }
    s.modules.emplace_back(std::string(mod), l); s.recompute();
}
inline void configure(std::string_view spec) {   // same syntax as FORGE_LOG; resets module overrides
    auto& s = detail::st(); std::lock_guard g(s.m); s.modules.clear(); s.parse_spec(spec);
}
inline level threshold(std::string_view mod = "forge") { return detail::st().threshold_for(mod); }
inline void set_format(format f) { detail::st().fmt = int(f); }
inline bool enabled(level l, std::string_view mod = "forge") {
    if (int(l) < FORGE_LOG_MIN || l == level::off) return false;
    auto& s = detail::st();
    if (int(l) < s.floor.load(std::memory_order_relaxed)) return false;
    return int(l) >= int(s.threshold_for(mod));
}

// Sinks return an id for remove_sink. The built-in stderr sink stays on unless set_stderr(false).
inline int add_sink(sink k) { auto& s = detail::st(); std::lock_guard g(s.m); int id = s.next_id++; s.sinks.emplace_back(id, std::move(k)); return id; }
inline void remove_sink(int id) { auto& s = detail::st(); std::lock_guard g(s.m); std::erase_if(s.sinks, [&](auto& p) { return p.first == id; }); }
inline void set_stderr(bool on) { auto& s = detail::st(); std::lock_guard g(s.m); s.use_stderr = on; }
// Appends each line to `path` (created if missing) and flushes it. Returns the sink id, or 0 if the file
// could not be opened.
inline int add_file_sink(const std::string& path) {
    std::shared_ptr<std::FILE> f(std::fopen(path.c_str(), "ab"), [](std::FILE* p) { if (p) std::fclose(p); });
    if (!f) return 0;
    return add_sink([f](const record&, std::string_view line) {
        std::fwrite(line.data(), 1, line.size(), f.get()); std::fputc('\n', f.get()); std::fflush(f.get());
    });
}

// ---- emitting -------------------------------------------------------------------------------------
inline void emit(level l, std::string_view mod, std::string_view msg) {
    auto& s = detail::st();
    record r{l, mod, msg, os::unix_time_ns(), os::thread_id()};
    std::string line = detail::format_line(r, format(s.fmt.load()));
    std::lock_guard g(s.m);   // one writer at a time: lines never interleave
    if (s.use_stderr) { std::string nl = line + '\n'; std::fwrite(nl.data(), 1, nl.size(), stderr); std::fflush(stderr); }
    for (auto& [_, k] : s.sinks) k(r, line);
}
template <class... A> void write_module(level l, std::string_view mod, std::format_string<A...> f, A&&... a) {
    if (!enabled(l, mod)) return;
    emit(l, mod, std::format(f, std::forward<A>(a)...));
}
template <class... A> void write(level l, std::format_string<A...> f, A&&... a) { write_module(l, "forge", f, std::forward<A>(a)...); }

#define FORGE_LOG_FN_(name, L)                                                                           \
    template <class... A> void name(std::format_string<A...> f, A&&... a) {                              \
        if constexpr (int(L) >= FORGE_LOG_MIN) write_module(L, "forge", f, std::forward<A>(a)...); \
    }
FORGE_LOG_FN_(trace, level::trace)
FORGE_LOG_FN_(debug, level::debug)
FORGE_LOG_FN_(info, level::info)
FORGE_LOG_FN_(warn, level::warn)
FORGE_LOG_FN_(error, level::error)
#undef FORGE_LOG_FN_

// A module-tagged logger: logger L{"mind"}; L.info("..."). Cheap to copy; holds only the tag.
struct logger {
    std::string_view module = "forge";
#define FORGE_LOG_M_(name, L)                                                                            \
    template <class... A> void name(std::format_string<A...> f, A&&... a) const {                        \
        if constexpr (int(L) >= FORGE_LOG_MIN) write_module(L, module, f, std::forward<A>(a)...);               \
    }
    FORGE_LOG_M_(trace, level::trace)
    FORGE_LOG_M_(debug, level::debug)
    FORGE_LOG_M_(info, level::info)
    FORGE_LOG_M_(warn, level::warn)
    FORGE_LOG_M_(error, level::error)
#undef FORGE_LOG_M_
    bool enabled(level l) const { return log::enabled(l, module); }
};
} // namespace forge::log

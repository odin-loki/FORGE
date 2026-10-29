#pragma once
#include <cstdio>
#include <cstdlib>
#include <format>

namespace forge::log {
enum class level { trace, debug, info, warn, error };
inline level& threshold() {
    static level l = [] {
        const char* e = std::getenv("FORGE_LOG");
        if (!e) return level::warn;
        switch (e[0]) { case 't': return level::trace; case 'd': return level::debug;
                        case 'i': return level::info; case 'e': return level::error; default: return level::warn; }
    }();
    return l;
}
template <class... A> void write(level l, std::format_string<A...> f, A&&... a) {
    if (l < threshold()) return;
    static const char* tag[] = {"T", "D", "I", "W", "E"};
    std::fprintf(stderr, "[forge %s] %s\n", tag[int(l)], std::format(f, std::forward<A>(a)...).c_str());
}
template <class... A> void info(std::format_string<A...> f, A&&... a) { write(level::info, f, std::forward<A>(a)...); }
template <class... A> void warn(std::format_string<A...> f, A&&... a) { write(level::warn, f, std::forward<A>(a)...); }
} // namespace forge::log

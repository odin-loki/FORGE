#pragma once
// forge::os console: is-a-terminal checks, terminal size, ANSI/VT colour enablement (Windows 10+ needs a mode
// flag), raw / no-echo input modes as RAII guards, password prompts, and colour helpers that respect NO_COLOR
// and FORCE_COLOR. Guide: docs/os-wrapper.md.
#include <forge/os/os.hpp>
#include <cctype>
#include <iostream>
#if defined(FORGE_OS_POSIX)
#include <sys/ioctl.h>
#include <termios.h>
#endif

namespace forge::os::console {
enum class stream { in = 0, out = 1, err = 2 };
inline bool is_tty(stream s) {
#if defined(FORGE_OS_POSIX)
    return ::isatty(int(s)) == 1;
#else
    return _isatty(int(s)) != 0;
#endif
}
struct size_t2 { int cols = 80, rows = 24; };
inline result<size_t2> size(stream s = stream::out) {
#if defined(FORGE_OS_POSIX)
    winsize w{}; if (::ioctl(int(s), TIOCGWINSZ, &w) || !w.ws_col) {
        size_t2 d; auto c = getenv("COLUMNS"), l = getenv("LINES");
        if (!c.empty()) d.cols = std::atoi(c.c_str()); if (!l.empty()) d.rows = std::atoi(l.c_str());
        return {d, is_tty(s) ? last_error("TIOCGWINSZ") : error{}};
    }
    return {{int(w.ws_col), int(w.ws_row)}, {}};
#else
    CONSOLE_SCREEN_BUFFER_INFO i{};
    if (!GetConsoleScreenBufferInfo(GetStdHandle(s == stream::err ? STD_ERROR_HANDLE : STD_OUTPUT_HANDLE), &i)) return {{}, last_error("GetConsoleScreenBufferInfo")};
    return {{i.srWindow.Right - i.srWindow.Left + 1, i.srWindow.Bottom - i.srWindow.Top + 1}, {}};
#endif
}
// Turn on ANSI escape processing (a no-op on POSIX terminals).
inline status enable_vt(stream s = stream::out) {
#if defined(FORGE_OS_WINDOWS)
    HANDLE h = GetStdHandle(s == stream::err ? STD_ERROR_HANDLE : STD_OUTPUT_HANDLE); DWORD m = 0;
    if (!GetConsoleMode(h, &m)) return {{}, last_error("GetConsoleMode")};
    if (!SetConsoleMode(h, m | 0x0004 /*ENABLE_VIRTUAL_TERMINAL_PROCESSING*/)) return {{}, last_error("SetConsoleMode")};
#else
    (void)s;
#endif
    return {};
}
// Should we emit colour on `s`? NO_COLOR disables, FORCE_COLOR / CLICOLOR_FORCE enable, TERM=dumb disables.
inline bool color_enabled(stream s = stream::out) {
    if (has_env("NO_COLOR")) return false;
    if (has_env("FORCE_COLOR") || getenv("CLICOLOR_FORCE") == "1") return true;
    if (getenv("TERM") == "dumb") return false;
    return is_tty(s);
}
enum class color { none = 0, red = 31, green = 32, yellow = 33, blue = 34, magenta = 35, cyan = 36, gray = 90, bold = 1 };
inline std::string paint(std::string_view text, color c, bool enabled) {
    if (!enabled || c == color::none) return std::string(text);
    return "\x1b[" + std::to_string(int(c)) + "m" + std::string(text) + "\x1b[0m";
}

// RAII: the terminal stays in the given mode until the guard is destroyed.
class mode_guard {
public:
    enum kind { raw, no_echo };
    explicit mode_guard(kind k) {
#if defined(FORGE_OS_POSIX)
        if (::tcgetattr(0, &old_) != 0) return;
        termios t = old_;
        if (k == raw) { t.c_lflag &= tcflag_t(~(ICANON | ECHO | ISIG | IEXTEN)); t.c_iflag &= tcflag_t(~(IXON | ICRNL)); t.c_cc[VMIN] = 1; t.c_cc[VTIME] = 0; }
        else t.c_lflag &= tcflag_t(~ECHO);
        active_ = ::tcsetattr(0, TCSAFLUSH, &t) == 0;
#else
        h_ = GetStdHandle(STD_INPUT_HANDLE); if (!GetConsoleMode(h_, &old_)) return;
        DWORD m = old_ & ~DWORD(ENABLE_ECHO_INPUT); if (k == raw) m &= ~DWORD(ENABLE_LINE_INPUT | ENABLE_PROCESSED_INPUT);
        active_ = SetConsoleMode(h_, m) != 0;
#endif
    }
    ~mode_guard() {
        if (!active_) return;
#if defined(FORGE_OS_POSIX)
        ::tcsetattr(0, TCSAFLUSH, &old_);
#else
        SetConsoleMode(h_, old_);
#endif
    }
    mode_guard(const mode_guard&) = delete; mode_guard& operator=(const mode_guard&) = delete;
    bool active() const { return active_; }
private:
    bool active_ = false;
#if defined(FORGE_OS_POSIX)
    termios old_{};
#else
    HANDLE h_ = nullptr; DWORD old_ = 0;
#endif
};
// Prompt on stderr and read a line from stdin without echo (falls back to a plain read when not a terminal).
inline std::string read_password(const std::string& prompt) {
    std::cerr << prompt << std::flush;
    std::string line;
    { mode_guard g(mode_guard::no_echo); std::getline(std::cin, line); }
    std::cerr << "\n";
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return line;
}
// Ask a yes/no question; anything but y/yes (case-insensitive) is "no". Non-interactive stdin answers `dflt`.
inline bool confirm(const std::string& question, bool dflt = false) {
    if (!is_tty(stream::in)) return dflt;
    std::cerr << question << (dflt ? " [Y/n] " : " [y/N] ") << std::flush;
    std::string a; if (!std::getline(std::cin, a)) return dflt;
    for (auto& c : a) c = char(std::tolower(static_cast<unsigned char>(c)));
    if (a.empty()) return dflt;
    return a == "y" || a == "yes";
}
} // namespace forge::os::console

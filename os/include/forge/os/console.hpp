#pragma once
// forge::os console: is-a-terminal checks, terminal size, ANSI/VT colour enablement (Windows 10+ needs a mode
// flag), raw / no-echo input modes as RAII guards, password prompts, and colour helpers that respect NO_COLOR
// and FORCE_COLOR. Guide: docs/os-wrapper.md.
#include <forge/os/os.hpp>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <optional>
#if defined(FORGE_OS_POSIX)
#include <sys/ioctl.h>
#include <termios.h>
#endif

namespace forge::os::console {
enum class stream { in = 0, out = 1, err = 2 };
#if defined(FORGE_OS_WINDOWS)
inline HANDLE std_handle(stream s) { return GetStdHandle(s == stream::in ? STD_INPUT_HANDLE : s == stream::err ? STD_ERROR_HANDLE : STD_OUTPUT_HANDLE); }
// mintty / MSYS2 / Cygwin terminals are not consoles: the program sees a named pipe called
// \msys-<hex>-ptyN-{from,to}-master or \cygwin-<hex>-ptyN-...
inline bool is_mintty_pipe(HANDLE h) {
    if (GetFileType(h) != FILE_TYPE_PIPE) return false;
    alignas(FILE_NAME_INFO) unsigned char b[sizeof(FILE_NAME_INFO) + MAX_PATH * sizeof(wchar_t)] = {};
    auto* fi = reinterpret_cast<FILE_NAME_INFO*>(b);
    if (!GetFileInformationByHandleEx(h, FileNameInfo, fi, sizeof b - sizeof(wchar_t))) return false;
    std::wstring n(fi->FileName, fi->FileNameLength / sizeof(wchar_t));
    return (n.find(L"\\msys-") == 0 || n.find(L"\\cygwin-") == 0) && n.find(L"-pty") != std::wstring::npos && n.find(L"-master") != std::wstring::npos;
}
#endif
// A terminal: a console on Windows (not merely a character device: NUL is one too) or a mintty/MSYS pty pipe.
inline bool is_tty(stream s) {
#if defined(FORGE_OS_POSIX)
    return ::isatty(int(s)) == 1;
#else
    HANDLE h = std_handle(s); if (!h || h == INVALID_HANDLE_VALUE) return false;
    DWORD m = 0; if (GetFileType(h) == FILE_TYPE_CHAR && GetConsoleMode(h, &m)) return true;
    return is_mintty_pipe(h);
#endif
}
struct size_t2 { int cols = 80, rows = 24; };
// Terminal size; when there is no terminal to ask, COLUMNS / LINES (else 80x24), with an error only if `s` is a
// terminal that would not say.
inline result<size_t2> size(stream s = stream::out) {
    auto fallback = [](size_t2 d = {}) {
        auto c = getenv("COLUMNS"), l = getenv("LINES"); int v = 0;
        if (!c.empty() && (v = std::atoi(c.c_str())) > 0) d.cols = v;
        if (!l.empty() && (v = std::atoi(l.c_str())) > 0) d.rows = v;
        return d;
    };
#if defined(FORGE_OS_POSIX)
    winsize w{}; if (::ioctl(int(s), TIOCGWINSZ, &w) || !w.ws_col) return {fallback(), is_tty(s) ? last_error("TIOCGWINSZ") : error{}};
    return {{int(w.ws_col), int(w.ws_row)}, {}};
#else
    CONSOLE_SCREEN_BUFFER_INFO i{};
    HANDLE h = std_handle(s == stream::in ? stream::out : s);
    // Redirected, or mintty (a pipe: only the pty knows the size, and it exports COLUMNS / LINES to shells).
    if (!GetConsoleScreenBufferInfo(h, &i)) return {fallback(), {}};
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
        // raw also turns on VT input (Windows 10+), so arrow and function keys arrive as the same escape sequences
        // as on a POSIX terminal; older consoles refuse that flag and get raw mode without it.
        h_ = GetStdHandle(STD_INPUT_HANDLE); if (!GetConsoleMode(h_, &old_)) return;
        DWORD m = old_ & ~DWORD(ENABLE_ECHO_INPUT); if (k == raw) m &= ~DWORD(ENABLE_LINE_INPUT | ENABLE_PROCESSED_INPUT);
        active_ = (k == raw && SetConsoleMode(h_, m | 0x0200 /*ENABLE_VIRTUAL_TERMINAL_INPUT*/)) || SetConsoleMode(h_, m);
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
// Windows consoles: read as UTF-16 (ReadConsoleW) and return UTF-8, so non-ASCII passwords survive any code page.
inline std::string read_password(const std::string& prompt) {
    std::cerr << prompt << std::flush;
    std::string line;
#if defined(FORGE_OS_WINDOWS)
    HANDLE in = GetStdHandle(STD_INPUT_HANDLE); DWORD m = 0;
    if (in && in != INVALID_HANDLE_VALUE && GetFileType(in) == FILE_TYPE_CHAR && GetConsoleMode(in, &m)) {
        mode_guard g(mode_guard::no_echo);
        std::wstring w; wchar_t b[256];
        for (;;) {
            DWORD n = 0; if (!ReadConsoleW(in, b, 256, &n, nullptr) || !n) break;
            w.append(b, n);
            if (w.find(L'\n') != std::wstring::npos || w.find(L'\r') != std::wstring::npos) break;
        }
        while (!w.empty() && (w.back() == L'\n' || w.back() == L'\r')) w.pop_back();
        std::cerr << "\n"; return narrow(w);
    }
#endif
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

// ---------------------------------------------------------------- mouse, bracketed paste, focus, alternate screen
// Terminal features every modern terminal switches with the same escape sequences (xterm's private modes), so
// one implementation serves POSIX terminals and the Windows console (Windows 10+: set_feature turns VT output on;
// to *receive* mouse, paste and focus reports as input, hold a mode_guard(raw), which enables VT input there).
//   mouse            SGR mouse reports: "\x1b[<b;x;yM" (press / move) and "...m" (release); any-motion tracking
//   bracketed_paste  pasted text arrives between "\x1b[200~" and "\x1b[201~"
//   focus_events     "\x1b[I" when the terminal gains focus, "\x1b[O" when it loses it
//   alternate_screen a second screen buffer without scrollback (full-screen programs); off restores the first
// Turn features off again before exiting (the terminal keeps the mode otherwise).
enum class feature { mouse, bracketed_paste, focus_events, alternate_screen };
inline std::string feature_sequence(feature f, bool on) {
    const char c = on ? 'h' : 'l';
    switch (f) {
        case feature::mouse: return on ? "\x1b[?1000h\x1b[?1003h\x1b[?1006h" : "\x1b[?1006l\x1b[?1003l\x1b[?1000l";
        case feature::bracketed_paste: return std::string("\x1b[?2004") + c;
        case feature::focus_events: return std::string("\x1b[?1004") + c;
        case feature::alternate_screen: return std::string("\x1b[?1049") + c;
    }
    return {};
}
// Write raw bytes to a terminal stream, past stdio's buffers (they must be flushed by the caller if it mixes both).
inline status write_raw(std::string_view bytes, stream s = stream::out) {
#if defined(FORGE_OS_POSIX)
    const int fd = int(s == stream::in ? stream::out : s);
    for (std::size_t off = 0; off < bytes.size();) {
        auto n = ::write(fd, bytes.data() + off, bytes.size() - off);
        if (n < 0) { if (errno == EINTR) continue; return {{}, last_error("console write")}; }
        off += std::size_t(n);
    }
#else
    HANDLE h = std_handle(s == stream::in ? stream::out : s);
    for (std::size_t off = 0; off < bytes.size();) {
        DWORD n = 0; if (!WriteFile(h, bytes.data() + off, DWORD(std::min<std::size_t>(bytes.size() - off, 1u << 20)), &n, nullptr)) return {{}, last_error("console write")};
        off += n;
    }
#endif
    return {};
}
// Switch a feature on a terminal. An error with ENOTTY when `s` is not a terminal (nothing is written, so
// redirected output never fills with escape sequences); on Windows the console must accept VT processing.
inline status set_feature(feature f, bool on, stream s = stream::out) {
    if (!is_tty(s)) return {{}, errno_error(ENOTTY, "console: not a terminal")};
#if defined(FORGE_OS_WINDOWS)
    if (!is_mintty_pipe(std_handle(s))) if (auto e = enable_vt(s); !e) return e;
#endif
    return write_raw(feature_sequence(f, on), s);
}
// Terminal resizes. POSIX terminals announce them with SIGWINCH and Windows consoles with a
// WINDOW_BUFFER_SIZE_EVENT input record; both only say "ask again", and taking either over would interfere with
// the program's own signal handling or input loop. poll() therefore just asks: it returns the new size when it
// differs from the last one seen, which is cheap enough to call once per frame or input wait.
class resize_watcher {
public:
    explicit resize_watcher(stream s = stream::out) : s_(s), last_(console::size(s).value) {}
    size_t2 current() const { return last_; }
    std::optional<size_t2> poll() {
        auto now = console::size(s_).value;
        if (now.cols == last_.cols && now.rows == last_.rows) return std::nullopt;
        last_ = now; return now;
    }
private:
    stream s_; size_t2 last_;
};
} // namespace forge::os::console

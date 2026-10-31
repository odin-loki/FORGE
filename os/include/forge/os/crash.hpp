#pragma once
// Crash handling: stack traces and crash dumps.
//   stack_trace(): _Unwind_Backtrace + dladdr on POSIX; CaptureStackBackTrace + DbgHelp SymFromAddr on Windows.
//   crash_handler::install({dir, app, minidump, after}): Windows writes a real minidump (<dir>/<app>-<pid>.dmp, MiniDumpWriteDump) plus a
//   text report; POSIX catches SIGSEGV/SIGBUS/SIGILL/SIGFPE/SIGABRT/SIGTRAP on an alternate stack and writes
//   <dir>/<app>-<pid>.crash (signal, fault address, backtrace, and on Linux/Android the module map), then
//   re-raises so the OS core dump / tombstone / ReportCrash still happens.
#include <forge/os/detail.hpp>
#if defined(FORGE_OS_POSIX)
#include <unwind.h>
#endif
#include <exception>

namespace forge::os {
struct stack_frame { void* addr = nullptr; std::string symbol, module; std::uintptr_t offset = 0; };
namespace detail {
#if defined(FORGE_OS_POSIX)
struct unwind_state { void** out; int n, max, skip; };
inline _Unwind_Reason_Code unwind_cb(struct _Unwind_Context* c, void* arg) {
    auto* s = static_cast<unwind_state*>(arg); auto ip = _Unwind_GetIP(c);
    if (!ip) return _URC_END_OF_STACK;
    if (s->skip > 0) { --s->skip; return _URC_NO_REASON; }
    if (s->n >= s->max) return _URC_END_OF_STACK;
    s->out[s->n++] = reinterpret_cast<void*>(ip); return _URC_NO_REASON;
}
#endif
inline int raw_backtrace(void** out, int max, int skip) {
#if defined(FORGE_OS_WINDOWS)
    return int(CaptureStackBackTrace(DWORD(skip + 1), DWORD(max), out, nullptr));
#else
    unwind_state s{out, 0, max, skip + 1}; _Unwind_Backtrace(unwind_cb, &s); return s.n;
#endif
}
#if defined(FORGE_OS_WINDOWS)
struct dbghelp {
    HMODULE h = nullptr; bool inited = false;
    BOOL(WINAPI* SymInitialize)(HANDLE, PCSTR, BOOL) = nullptr;
    BOOL(WINAPI* SymFromAddr)(HANDLE, DWORD64, PDWORD64, void*) = nullptr;
    DWORD(WINAPI* SymSetOptions)(DWORD) = nullptr;
    BOOL(WINAPI* MiniDumpWriteDump)(HANDLE, DWORD, HANDLE, int, void*, void*, void*) = nullptr;
    static dbghelp& get() {
        static dbghelp d = [] { dbghelp x; x.h = LoadLibraryW(L"dbghelp.dll"); if (!x.h) return x;
            x.SymInitialize = reinterpret_cast<decltype(x.SymInitialize)>(reinterpret_cast<void*>(GetProcAddress(x.h, "SymInitialize")));
            x.SymFromAddr = reinterpret_cast<decltype(x.SymFromAddr)>(reinterpret_cast<void*>(GetProcAddress(x.h, "SymFromAddr")));
            x.SymSetOptions = reinterpret_cast<decltype(x.SymSetOptions)>(reinterpret_cast<void*>(GetProcAddress(x.h, "SymSetOptions")));
            x.MiniDumpWriteDump = reinterpret_cast<decltype(x.MiniDumpWriteDump)>(reinterpret_cast<void*>(GetProcAddress(x.h, "MiniDumpWriteDump")));
            if (x.SymSetOptions) x.SymSetOptions(0x2 | 0x10 /*UNDNAME|LOAD_LINES*/);
            if (x.SymInitialize) x.inited = x.SymInitialize(GetCurrentProcess(), nullptr, TRUE) != 0;
            return x; }();
        return d;
    }
};
#endif
inline void symbolize(stack_frame& f) {
#if defined(FORGE_OS_WINDOWS)
    HMODULE m = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, static_cast<LPCWSTR>(f.addr), &m)) {
        wchar_t b[MAX_PATH]; DWORD n = GetModuleFileNameW(m, b, MAX_PATH); f.module = narrow({b, n}); f.offset = reinterpret_cast<std::uintptr_t>(f.addr) - reinterpret_cast<std::uintptr_t>(m);
    }
    auto& d = dbghelp::get();
    if (d.inited && d.SymFromAddr) {
        struct { ULONG SizeOfStruct, TypeIndex; ULONG64 Reserved[2]; ULONG Index, Size; ULONG64 ModBase; ULONG Flags; ULONG64 Value, Address; ULONG Register, Scope, Tag, NameLen, MaxNameLen; CHAR Name[256]; } si{};
        si.SizeOfStruct = 88; si.MaxNameLen = 255; DWORD64 disp = 0;
        if (d.SymFromAddr(GetCurrentProcess(), DWORD64(reinterpret_cast<std::uintptr_t>(f.addr)), &disp, &si)) f.symbol = si.Name;
    }
#else
    Dl_info i{};
    if (::dladdr(f.addr, &i)) {
        if (i.dli_fname) f.module = i.dli_fname; if (i.dli_sname) f.symbol = i.dli_sname;
        f.offset = reinterpret_cast<std::uintptr_t>(f.addr) - reinterpret_cast<std::uintptr_t>(i.dli_fbase);
    }
#endif
}
}
inline std::vector<stack_frame> stack_trace(int skip = 0, int max = 64) {
    std::vector<void*> raw(std::size_t(max > 0 ? max : 64)); int n = detail::raw_backtrace(raw.data(), int(raw.size()), skip + 1);
    std::vector<stack_frame> v; for (int i = 0; i < n; ++i) { stack_frame f; f.addr = raw[std::size_t(i)]; detail::symbolize(f); v.push_back(std::move(f)); }
    return v;
}
inline std::string format_stack_trace(const std::vector<stack_frame>& v) {
    std::string s; char b[64];
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::snprintf(b, sizeof b, "#%-2zu %p ", i, v[i].addr); s += b;
        s += v[i].symbol.empty() ? "??" : v[i].symbol;
        if (!v[i].module.empty()) { std::snprintf(b, sizeof b, "+0x%llx", (unsigned long long)v[i].offset); s += " (" + basename(v[i].module) + b + ")"; }
        s += '\n';
    }
    return s;
}

class crash_handler {
public:
    struct config { std::string dir, app = "forge"; bool minidump = true; std::function<void(const std::string& report_path)> after; };
    static status install(config c) {
        auto& s = st(); if (c.dir.empty()) c.dir = temp_dir();
        if (auto m = mkdirs(c.dir); !m) return m;
        s.cfg = std::move(c);
        std::string base = join(s.cfg.dir, s.cfg.app + "-" + std::to_string(pid()));
        std::snprintf(s.report, sizeof s.report, "%s.crash", base.c_str());
        std::snprintf(s.dump, sizeof s.dump, "%s.dmp", base.c_str());
#if defined(FORGE_OS_WINDOWS)
        detail::dbghelp::get();
        SetUnhandledExceptionFilter(&seh_);
#else
        static std::vector<std::uint8_t> alt(1 << 16);
        stack_t ss{}; ss.ss_sp = alt.data(); ss.ss_size = alt.size(); ::sigaltstack(&ss, nullptr);
        for (int sg : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT, SIGTRAP}) {
            struct sigaction sa{}; sa.sa_sigaction = &sig_; sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESETHAND; sigemptyset(&sa.sa_mask); ::sigaction(sg, &sa, nullptr);
        }
#endif
        std::set_terminate([] { write_report_("std::terminate", nullptr, -1); std::abort(); });
        s.installed = true; return {};
    }
    static const char* report_path() { return st().report; }
    static const char* dump_path() { return st().dump; }
    static bool installed() { return st().installed; }
    // Write a dump of the running process on demand (Windows minidump; elsewhere a text report).
    static status write_dump_now(const std::string& path) {
#if defined(FORGE_OS_WINDOWS)
        auto& d = detail::dbghelp::get(); if (!d.MiniDumpWriteDump) return {{}, unsupported("dbghelp MiniDumpWriteDump")};
        HANDLE f = CreateFileW(widen(path).c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        if (f == INVALID_HANDLE_VALUE) return {{}, last_error("create " + path)};
        BOOL ok = d.MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), f, 0x1 | 0x4 /*WithDataSegs|WithHandleData*/, nullptr, nullptr, nullptr);
        CloseHandle(f); if (!ok) return {{}, last_error("MiniDumpWriteDump")}; return {};
#else
        return write_file(path, "forge on-demand report\npid " + std::to_string(pid()) + "\n" + format_stack_trace(stack_trace(1)));
#endif
    }
private:
    struct state { config cfg; char report[1024] = {}; char dump[1024] = {}; bool installed = false; };
    static state& st() { static state s; return s; }
    static void write_report_(const char* what, const void* addr, int sig) {
        auto& s = st(); if (!s.report[0]) return;
        void* raw[64]; int n = detail::raw_backtrace(raw, 64, 1);
#if defined(FORGE_OS_WINDOWS)
        FILE* f = std::fopen(s.report, "wb"); if (!f) return;
        std::fprintf(f, "forge crash report\napp %s\npid %llu\nreason %s\naddress %p\nsignal %d\nbacktrace:\n", s.cfg.app.c_str(), (unsigned long long)pid(), what, addr, sig);
        for (int i = 0; i < n; ++i) { stack_frame fr; fr.addr = raw[i]; detail::symbolize(fr); std::fprintf(f, "#%-2d %p %s (%s)\n", i, fr.addr, fr.symbol.c_str(), basename(fr.module).c_str()); }
        std::fclose(f);
#else
        int fd = ::open(s.report, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600); if (fd < 0) return;
        auto put = [&](const char* t) { (void)!::write(fd, t, std::strlen(t)); };
        char b[512];
        std::snprintf(b, sizeof b, "forge crash report\napp %s\npid %llu\nreason %s\naddress %p\nsignal %d\nbacktrace:\n", s.cfg.app.c_str(), (unsigned long long)::getpid(), what, addr, sig); put(b);
        for (int i = 0; i < n; ++i) {
            Dl_info di{}; const char* sym = "??"; const char* mod = "?"; std::uintptr_t off = 0;
            if (::dladdr(raw[i], &di)) { if (di.dli_sname) sym = di.dli_sname; if (di.dli_fname) mod = di.dli_fname; off = reinterpret_cast<std::uintptr_t>(raw[i]) - reinterpret_cast<std::uintptr_t>(di.dli_fbase); }
            std::snprintf(b, sizeof b, "#%-2d %p %s (%s+0x%llx)\n", i, raw[i], sym, mod, (unsigned long long)off); put(b);
        }
#if defined(__linux__)
        put("maps:\n"); int m = ::open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
        if (m >= 0) { char mb[4096]; ssize_t r; while ((r = ::read(m, mb, sizeof mb)) > 0) (void)!::write(fd, mb, std::size_t(r)); ::close(m); }
#endif
        ::close(fd);
#endif
        if (s.cfg.after) s.cfg.after(s.report);
    }
#if defined(FORGE_OS_WINDOWS)
    static LONG WINAPI seh_(EXCEPTION_POINTERS* ep) {
        auto& s = st();
        if (s.cfg.minidump) if (auto& d = detail::dbghelp::get(); d.MiniDumpWriteDump) {
            HANDLE f = CreateFileA(s.dump, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
            if (f != INVALID_HANDLE_VALUE) {
                // MINIDUMP_EXCEPTION_INFORMATION is declared under #pragma pack(4) in dbghelp.h: the pointer sits at
                // offset 4, not 8. A naturally aligned copy makes MiniDumpWriteDump fail and leaves an empty .dmp.
#pragma pack(push, 4)
                struct mei_t { DWORD ThreadId; EXCEPTION_POINTERS* ExceptionPointers; BOOL ClientPointers; };
#pragma pack(pop)
                static_assert(sizeof(void*) != 8 || sizeof(mei_t) == 16, "MINIDUMP_EXCEPTION_INFORMATION layout");
                mei_t mei{GetCurrentThreadId(), ep, FALSE};
                d.MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), f, 0x1 | 0x4, &mei, nullptr, nullptr); CloseHandle(f);
            }
        }
        char what[64]; std::snprintf(what, sizeof what, "exception 0x%08lx", ep && ep->ExceptionRecord ? (unsigned long)ep->ExceptionRecord->ExceptionCode : 0ul);
        write_report_(what, ep && ep->ExceptionRecord ? ep->ExceptionRecord->ExceptionAddress : nullptr, -1);
        return EXCEPTION_EXECUTE_HANDLER;   // terminate with the exception code; no WER / JIT debugger hang
    }
#else
    static void sig_(int sg, siginfo_t* si, void*) {
        const char* w = sg == SIGSEGV ? "SIGSEGV" : sg == SIGBUS ? "SIGBUS" : sg == SIGILL ? "SIGILL" : sg == SIGFPE ? "SIGFPE" : sg == SIGABRT ? "SIGABRT" : "SIGTRAP";
        write_report_(w, si ? si->si_addr : nullptr, sg);
        ::raise(sg);   // SA_RESETHAND restored the default action
    }
#endif
};
} // namespace forge::os

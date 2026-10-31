#pragma once
// Typed OS errors: turns errno (POSIX: Linux/glibc, Android/bionic, FreeBSD, macOS/iOS) and Win32 error
// codes into a small portable taxonomy and a C++ exception, and turns synchronous fault signals into
// exceptions where that can be done safely.
//
//   auto k = os::classify_errno(EACCES);             // os_errc::permission_denied
//   os::throw_if(os::write_file(p, d));              // throws os::os_exception{kind, code, what}
//   os::fault_guard([&]{ risky_native_call(); });    // SIGSEGV/SIGBUS/SIGFPE/SIGILL/SIGSYS -> os_exception{signal}
//
// Sandbox violations: seccomp (Linux/Android) returns EPERM for denied syscalls; Capsicum (FreeBSD) returns
// ECAPMODE/ENOTCAPABLE; pledge (OpenBSD) and seccomp TRAP mode deliver SIGSYS/SIGABRT; macOS/iOS sandbox
// returns EPERM. When sandbox_status().sandboxed is true, EPERM/EACCES/ECAPMODE/ENOTCAPABLE classify as
// sandbox_violation.
//
// Fault recovery (fault_guard) uses sigsetjmp/siglongjmp with a per-thread jump target. It is only safe for
// code that holds no locks and owns no resources across the fault (C++ destructors in the skipped frames do
// not run). iOS/macOS: Mach exceptions are delivered as BSD signals unless a Mach exception port (e.g. a
// debugger or crash reporter) claims them first; then fault_guard cannot intercept. Windows: uses a vectored
// exception handler for access violations / illegal instructions / integer divide by zero, then longjmp.
#include <forge/os/os.hpp>
#include <cerrno>
#include <csetjmp>
#include <stdexcept>
#include <string>
#if defined(_WIN32)
#include <windows.h>
#else
#include <csignal>
#endif

namespace forge::os {

enum class os_errc {
    ok, not_found, already_exists, permission_denied, sandbox_violation, not_supported, invalid_argument,
    would_block, interrupted, timed_out, connection_refused, connection_reset, network_unreachable,
    address_in_use, no_space, too_many_files, busy, not_a_directory, is_a_directory, broken_pipe,
    out_of_memory, signal, io, other
};

inline const char* name(os_errc k) {
    switch (k) {
    case os_errc::ok: return "ok"; case os_errc::not_found: return "not_found"; case os_errc::already_exists: return "already_exists";
    case os_errc::permission_denied: return "permission_denied"; case os_errc::sandbox_violation: return "sandbox_violation";
    case os_errc::not_supported: return "not_supported"; case os_errc::invalid_argument: return "invalid_argument";
    case os_errc::would_block: return "would_block"; case os_errc::interrupted: return "interrupted"; case os_errc::timed_out: return "timed_out";
    case os_errc::connection_refused: return "connection_refused"; case os_errc::connection_reset: return "connection_reset";
    case os_errc::network_unreachable: return "network_unreachable"; case os_errc::address_in_use: return "address_in_use";
    case os_errc::no_space: return "no_space"; case os_errc::too_many_files: return "too_many_files"; case os_errc::busy: return "busy";
    case os_errc::not_a_directory: return "not_a_directory"; case os_errc::is_a_directory: return "is_a_directory";
    case os_errc::broken_pipe: return "broken_pipe"; case os_errc::out_of_memory: return "out_of_memory"; case os_errc::signal: return "signal";
    case os_errc::io: return "io"; default: return "other";
    }
}

// ForgeScript exception class for each kind (see script/src/builtins.cpp).
inline const char* script_error_type(os_errc k) {
    switch (k) {
    case os_errc::not_found: return "FileNotFoundError";
    case os_errc::already_exists: return "FileExistsError";
    case os_errc::permission_denied: return "PermissionError";
    case os_errc::sandbox_violation: return "SandboxViolationError";
    case os_errc::not_supported: return "NotSupportedError";
    case os_errc::timed_out: return "TimeoutError";
    case os_errc::interrupted: return "InterruptedError";
    case os_errc::connection_refused: case os_errc::connection_reset: case os_errc::network_unreachable: case os_errc::address_in_use:
    case os_errc::broken_pipe: return "ConnectionError";
    case os_errc::out_of_memory: return "OutOfMemoryError";
    case os_errc::signal: return "SignalError";
    case os_errc::invalid_argument: return "ValueError";
    default: return "OSError";
    }
}

namespace detail { inline bool& assume_sandboxed() { static bool b = false; return b; } }
// Set by whoever applies the sandbox (ForgeScript's engine) so EPERM is reported as a sandbox violation
// without re-reading /proc on every error.
inline void note_sandboxed(bool on = true) { detail::assume_sandboxed() = on; }

inline os_errc classify_errno(int e) {
    bool sbx = detail::assume_sandboxed();
    switch (e) {
    case 0: return os_errc::ok;
    case ENOENT: return os_errc::not_found;
    case EEXIST: return os_errc::already_exists;
    case EPERM: case EACCES: return sbx ? os_errc::sandbox_violation : os_errc::permission_denied;
#if defined(ECAPMODE)       // FreeBSD Capsicum
    case ECAPMODE: return os_errc::sandbox_violation;
#endif
#if defined(ENOTCAPABLE)
    case ENOTCAPABLE: return os_errc::sandbox_violation;
#endif
    case ENOSYS: return os_errc::not_supported;
#if defined(ENOTSUP)
    case ENOTSUP: return os_errc::not_supported;
#endif
#if defined(EOPNOTSUPP) && (!defined(ENOTSUP) || EOPNOTSUPP != ENOTSUP)
    case EOPNOTSUPP: return os_errc::not_supported;
#endif
    case EINVAL: return os_errc::invalid_argument;
    case EAGAIN: return os_errc::would_block;
#if defined(EWOULDBLOCK) && EWOULDBLOCK != EAGAIN
    case EWOULDBLOCK: return os_errc::would_block;
#endif
    case EINTR: return os_errc::interrupted;
    case ETIMEDOUT: return os_errc::timed_out;
    case ECONNREFUSED: return os_errc::connection_refused;
    case ECONNRESET: return os_errc::connection_reset;
    case ENETUNREACH: case EHOSTUNREACH: return os_errc::network_unreachable;
    case EADDRINUSE: return os_errc::address_in_use;
    case ENOSPC: return os_errc::no_space;
#if defined(EDQUOT)
    case EDQUOT: return os_errc::no_space;
#endif
    case EMFILE: case ENFILE: return os_errc::too_many_files;
    case EBUSY: return os_errc::busy;
    case ENOTDIR: return os_errc::not_a_directory;
    case EISDIR: return os_errc::is_a_directory;
    case EPIPE: return os_errc::broken_pipe;
    case ENOMEM: return os_errc::out_of_memory;
    case EIO: return os_errc::io;
    default: return os_errc::other;
    }
}

#if defined(_WIN32)
inline os_errc classify_win32(unsigned long e) {
    switch (e) {
    case ERROR_SUCCESS: return os_errc::ok;
    case ERROR_FILE_NOT_FOUND: case ERROR_PATH_NOT_FOUND: case ERROR_INVALID_DRIVE: case ERROR_MOD_NOT_FOUND: return os_errc::not_found;
    case ERROR_FILE_EXISTS: case ERROR_ALREADY_EXISTS: return os_errc::already_exists;
    case ERROR_ACCESS_DENIED: case ERROR_PRIVILEGE_NOT_HELD: return detail::assume_sandboxed() ? os_errc::sandbox_violation : os_errc::permission_denied;
    case ERROR_NOT_SUPPORTED: case ERROR_CALL_NOT_IMPLEMENTED: return os_errc::not_supported;
    case ERROR_INVALID_PARAMETER: case ERROR_INVALID_NAME: return os_errc::invalid_argument;
    case ERROR_SHARING_VIOLATION: case ERROR_LOCK_VIOLATION: case ERROR_BUSY: return os_errc::busy;
    case ERROR_TIMEOUT: case WAIT_TIMEOUT: return os_errc::timed_out;
    case ERROR_OPERATION_ABORTED: return os_errc::interrupted;
    case ERROR_DISK_FULL: case ERROR_HANDLE_DISK_FULL: return os_errc::no_space;
    case ERROR_TOO_MANY_OPEN_FILES: return os_errc::too_many_files;
    case ERROR_DIRECTORY: return os_errc::not_a_directory;
    case ERROR_BROKEN_PIPE: case ERROR_NO_DATA: return os_errc::broken_pipe;
    case ERROR_NOT_ENOUGH_MEMORY: case ERROR_OUTOFMEMORY: return os_errc::out_of_memory;
    case ERROR_CONNECTION_REFUSED: case 10061 /*WSAECONNREFUSED*/: return os_errc::connection_refused;
    case 10054 /*WSAECONNRESET*/: return os_errc::connection_reset;
    case 10051: case 10065: return os_errc::network_unreachable;
    case 10048 /*WSAEADDRINUSE*/: return os_errc::address_in_use;
    case 10060 /*WSAETIMEDOUT*/: return os_errc::timed_out;
    case 10035 /*WSAEWOULDBLOCK*/: return os_errc::would_block;
    default: return os_errc::other;
    }
}
#endif

// os::error codes are errno on POSIX and GetLastError()/WSAGetLastError() on Windows; unsupported_code
// maps to not_supported.
inline os_errc classify(const error& e) {
    if (!e) return os_errc::ok;
    if (is_unsupported(e)) return os_errc::not_supported;
#if defined(_WIN32)
    return classify_win32(static_cast<unsigned long>(e.code));
#else
    return classify_errno(e.code);
#endif
}

struct os_exception : std::runtime_error {
    os_errc kind; int code; int signo = 0;
    os_exception(os_errc k, int c, const std::string& w, int sig = 0) : std::runtime_error(w), kind(k), code(c), signo(sig) {}
    const char* script_type() const { return script_error_type(kind); }
};

inline void throw_if(const error& e) { if (e) throw os_exception(classify(e), e.code, e.what); }
template <class T> T throw_if(result<T> r) { throw_if(r.err); return std::move(r.value); }

// ---------------------------------------------------------------- fault signals -> exceptions
namespace detail {
struct fault_frame { std::jmp_buf* jb = nullptr; int sig = 0; };
inline thread_local fault_frame* t_fault = nullptr;
}

inline const char* signal_name(int s) {
    switch (s) {
#if !defined(_WIN32)
    case SIGSEGV: return "SIGSEGV"; case SIGBUS: return "SIGBUS"; case SIGFPE: return "SIGFPE"; case SIGILL: return "SIGILL";
#if defined(SIGSYS)
    case SIGSYS: return "SIGSYS (system call denied by the sandbox)";
#endif
#else
    case 1: return "EXCEPTION_ACCESS_VIOLATION"; case 2: return "EXCEPTION_ILLEGAL_INSTRUCTION"; case 3: return "EXCEPTION_INT_DIVIDE_BY_ZERO";
#endif
    default: return "signal";
    }
}

#if !defined(_WIN32)
namespace detail {
extern "C" inline void forge_fault_handler(int sig, siginfo_t*, void*) {
    if (auto* f = t_fault) { f->sig = sig; ::siglongjmp(*reinterpret_cast<sigjmp_buf*>(f->jb), 1); }
    ::signal(sig, SIG_DFL); ::raise(sig);   // not inside a guard: default action (crash/core) as before
}
inline void install_fault_handlers() {
    static bool done = [] {
        struct sigaction sa{}; sa.sa_sigaction = forge_fault_handler; sa.sa_flags = SA_SIGINFO | SA_NODEFER; sigemptyset(&sa.sa_mask);
        int sigs[] = {SIGSEGV, SIGBUS, SIGFPE, SIGILL,
#if defined(SIGSYS)
                      SIGSYS,
#endif
        };
        for (int s : sigs) ::sigaction(s, &sa, nullptr);
        return true; }();
    (void)done;
}
}
// Runs f; a synchronous fault inside it becomes os_exception{signal}. See the safety notes at the top.
template <class F> auto fault_guard(F&& f) -> decltype(f()) {
    detail::install_fault_handlers();
    sigjmp_buf jb; detail::fault_frame fr; fr.jb = reinterpret_cast<std::jmp_buf*>(&jb);
    auto* prev = detail::t_fault;
    if (sigsetjmp(jb, 1)) {
        detail::t_fault = prev;
        throw os_exception(fr.sig == SIGSYS ? os_errc::sandbox_violation : os_errc::signal, fr.sig, std::string("fault: ") + signal_name(fr.sig), fr.sig);
    }
    detail::t_fault = &fr;
    struct restore { detail::fault_frame* p; ~restore() { detail::t_fault = p; } } r{prev};
    return f();
}
#else
namespace detail {
inline LONG CALLBACK forge_veh(EXCEPTION_POINTERS* ep) {
    auto* f = t_fault; if (!f) return EXCEPTION_CONTINUE_SEARCH;
    switch (ep->ExceptionRecord->ExceptionCode) {
    case EXCEPTION_ACCESS_VIOLATION: f->sig = 1; break;
    case EXCEPTION_ILLEGAL_INSTRUCTION: f->sig = 2; break;
    case EXCEPTION_INT_DIVIDE_BY_ZERO: f->sig = 3; break;
    default: return EXCEPTION_CONTINUE_SEARCH;
    }
    std::longjmp(*f->jb, 1);
}
inline void install_fault_handlers() { static void* h = AddVectoredExceptionHandler(1, forge_veh); (void)h; }
}
template <class F> auto fault_guard(F&& f) -> decltype(f()) {
    detail::install_fault_handlers();
    std::jmp_buf jb; detail::fault_frame fr; fr.jb = &jb;
    auto* prev = detail::t_fault;
    if (setjmp(jb)) { detail::t_fault = prev; throw os_exception(os_errc::signal, fr.sig, std::string("fault: ") + signal_name(fr.sig), fr.sig); }
    detail::t_fault = &fr;
    struct restore { detail::fault_frame* p; ~restore() { detail::t_fault = p; } } r{prev};
    return f();
}
#endif

} // namespace forge::os

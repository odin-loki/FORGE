#pragma once
// forge::os waiting primitives that the kernel provides differently on every OS, behind one API each:
//   wait_on_address / wake_one / wake_all   futex-style: sleep while a 32-bit word holds a value.
//       Linux, Android: futex(2).  Windows: WaitOnAddress / WakeByAddress*.  FreeBSD: _umtx_op(2) UMTX_OP_WAIT_UINT.
//       macOS 14.4+ / iOS 17.4+: os_sync_wait_on_address (looked up at run time).  Older Apple systems (the only
//       public interface there is libc++'s) and anything else: a table of mutex + condition-variable buckets
//       keyed by the address. address_wait_backend() names the one in use.
//   wait_event     a flag any thread (or a signal handler, through set()) raises and others wait for; it stays
//       raised until reset(). native() can be registered with os::poller (POSIX) or waited on with
//       WaitForMultipleObjects (Windows). Linux, Android, FreeBSD 13+: eventfd.  macOS, iOS, older FreeBSD: a
//       non-blocking pipe (kqueue's EVFILT_USER wakes one waiter per trigger, a pipe wakes all).  Windows: a
//       manual-reset event object.
//   wait_timer     one-shot or periodic timer on the monotonic clock with nanosecond arguments.
//       Linux, Android: timerfd.  macOS, iOS, FreeBSD: kqueue EVFILT_TIMER (NOTE_NSECONDS).  Windows: a
//       high-resolution waitable timer (Windows 10 1803+; the 15.6 ms timer before that).
//   signal_listener   termination-style signals as a queue to wait on instead of a handler: a self-pipe written
//       from the handler on POSIX (works with any event loop; signalfd would need the signals blocked in every
//       thread, EVFILT_SIGNAL exists only on kqueue), console control events on Windows.
// Nothing here throws; unsupported paths return os::unsupported.
#include <forge/os/os.hpp>
#include <optional>
#if defined(FORGE_OS_LINUX) || defined(FORGE_OS_ANDROID)
#include <linux/futex.h>
#include <sys/eventfd.h>
#include <sys/timerfd.h>
#elif defined(FORGE_OS_FREEBSD)
#include <sys/umtx.h>
#if __FreeBSD_version >= 1300000
#include <sys/eventfd.h>
#define FORGE_OS_EVENTFD 1
#endif
#endif
#if defined(FORGE_OS_LINUX) || defined(FORGE_OS_ANDROID)
#define FORGE_OS_EVENTFD 1
#endif

namespace forge::os {

// ---------------------------------------------------------------- wait on address
enum class wait_result { woken, timed_out, mismatch };   // woken: by a wake or spuriously; re-check the word
namespace detail::aw {
struct bucket { std::mutex m; std::condition_variable cv; };
inline bucket& bucket_for(const void* p) {
    static bucket t[64];
    auto v = reinterpret_cast<std::uintptr_t>(p);
    return t[(v >> 2 ^ v >> 11) & 63];
}
inline wait_result table_wait(const std::atomic<std::uint32_t>& a, std::uint32_t expected, std::uint64_t timeout_ns) {
    auto& b = bucket_for(&a); std::unique_lock<std::mutex> lk(b.m);
    if (a.load(std::memory_order_seq_cst) != expected) return wait_result::mismatch;
    if (timeout_ns == ~0ull) { b.cv.wait(lk); return wait_result::woken; }
    return b.cv.wait_for(lk, std::chrono::nanoseconds(timeout_ns)) == std::cv_status::timeout ? wait_result::timed_out : wait_result::woken;
}
inline void table_wake(const void* p) { auto& b = bucket_for(p); { std::lock_guard<std::mutex> lk(b.m); } b.cv.notify_all(); }
#if defined(FORGE_OS_WINDOWS)
struct api {
    BOOL (WINAPI* wait)(volatile VOID*, PVOID, SIZE_T, DWORD) = nullptr;
    void (WINAPI* wake_one)(PVOID) = nullptr; void (WINAPI* wake_all)(PVOID) = nullptr;
    static const api& get() {
        static const api a = [] {
            api x; HMODULE m = GetModuleHandleW(L"kernelbase.dll");
            if (!m) m = LoadLibraryExW(L"api-ms-win-core-synch-l1-2-0.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            if (!m) return x;
            x.wait = reinterpret_cast<decltype(x.wait)>(reinterpret_cast<void*>(GetProcAddress(m, "WaitOnAddress")));
            x.wake_one = reinterpret_cast<decltype(x.wake_one)>(reinterpret_cast<void*>(GetProcAddress(m, "WakeByAddressSingle")));
            x.wake_all = reinterpret_cast<decltype(x.wake_all)>(reinterpret_cast<void*>(GetProcAddress(m, "WakeByAddressAll")));
            if (!x.wait || !x.wake_one || !x.wake_all) x = api{};
            return x;
        }();
        return a;
    }
};
#elif defined(__APPLE__)
struct api {   // <os/os_sync_wait_on_address.h>, macOS 14.4 / iOS 17.4
    int (*wait)(void*, std::uint64_t, std::size_t, std::uint32_t) = nullptr;
    int (*wait_timeout)(void*, std::uint64_t, std::size_t, std::uint32_t, std::uint32_t, std::uint64_t) = nullptr;
    int (*wake_one)(void*, std::size_t, std::uint32_t) = nullptr; int (*wake_all)(void*, std::size_t, std::uint32_t) = nullptr;
    static const api& get() {
        static const api a = [] {
            api x;
            x.wait = reinterpret_cast<decltype(x.wait)>(::dlsym(RTLD_DEFAULT, "os_sync_wait_on_address"));
            x.wait_timeout = reinterpret_cast<decltype(x.wait_timeout)>(::dlsym(RTLD_DEFAULT, "os_sync_wait_on_address_with_timeout"));
            x.wake_one = reinterpret_cast<decltype(x.wake_one)>(::dlsym(RTLD_DEFAULT, "os_sync_wake_by_address_any"));
            x.wake_all = reinterpret_cast<decltype(x.wake_all)>(::dlsym(RTLD_DEFAULT, "os_sync_wake_by_address_all"));
            if (!x.wait || !x.wait_timeout || !x.wake_one || !x.wake_all) x = api{};
            return x;
        }();
        return a;
    }
};
#endif
}
// Which mechanism wait_on_address uses here: "futex", "WaitOnAddress", "umtx", "os_sync" or "table".
inline const char* address_wait_backend() {
#if defined(FORGE_OS_LINUX) || defined(FORGE_OS_ANDROID)
    return "futex";
#elif defined(FORGE_OS_WINDOWS)
    return detail::aw::api::get().wait ? "WaitOnAddress" : "table";
#elif defined(FORGE_OS_FREEBSD)
    return "umtx";
#elif defined(__APPLE__)
    return detail::aw::api::get().wait ? "os_sync" : "table";
#else
    return "table";
#endif
}
// Sleep while `a` still holds `expected` (checked atomically with going to sleep), until a wake on `a`, the
// timeout (default: none) or a spurious wake-up. Callers loop:  while (a.load() == v) wait_on_address(a, v);
inline wait_result wait_on_address(const std::atomic<std::uint32_t>& a, std::uint32_t expected, std::uint64_t timeout_ns = ~0ull) {
    static_assert(sizeof(std::atomic<std::uint32_t>) == sizeof(std::uint32_t), "atomic<uint32_t> must be a plain word");
    void* addr = const_cast<void*>(static_cast<const void*>(&a));
#if defined(FORGE_OS_LINUX) || defined(FORGE_OS_ANDROID)
    struct timespec ts{}, *tp = nullptr;
    if (timeout_ns != ~0ull) { ts.tv_sec = time_t(timeout_ns / 1000000000ull); ts.tv_nsec = long(timeout_ns % 1000000000ull); tp = &ts; }
#if defined(SYS_futex)
    long r = ::syscall(SYS_futex, addr, FUTEX_WAIT | FUTEX_PRIVATE_FLAG, expected, tp, nullptr, 0);
#else   // 32-bit ABIs with only the 64-bit time_t call (riscv32)
    long r = ::syscall(SYS_futex_time64, addr, FUTEX_WAIT | FUTEX_PRIVATE_FLAG, expected, tp, nullptr, 0);
#endif
    if (r == 0 || errno == EINTR) return wait_result::woken;
    return errno == ETIMEDOUT ? wait_result::timed_out : errno == EAGAIN ? wait_result::mismatch : wait_result::woken;
#elif defined(FORGE_OS_WINDOWS)
    auto& w = detail::aw::api::get();
    if (!w.wait) return detail::aw::table_wait(a, expected, timeout_ns);
    if (a.load(std::memory_order_seq_cst) != expected) return wait_result::mismatch;
    DWORD ms = INFINITE;
    if (timeout_ns != ~0ull) { std::uint64_t m = (timeout_ns + 999999ull) / 1000000ull; ms = m >= INFINITE ? INFINITE - 1 : DWORD(m); }
    if (w.wait(addr, &expected, sizeof expected, ms)) return wait_result::woken;
    return GetLastError() == ERROR_TIMEOUT ? wait_result::timed_out : wait_result::woken;
#elif defined(FORGE_OS_FREEBSD)
    if (timeout_ns == ~0ull) {
        if (::_umtx_op(addr, UMTX_OP_WAIT_UINT_PRIVATE, expected, nullptr, nullptr) == 0 || errno == EINTR) return wait_result::woken;
    } else {
        struct _umtx_time t{}; t._timeout.tv_sec = time_t(timeout_ns / 1000000000ull); t._timeout.tv_nsec = long(timeout_ns % 1000000000ull);
        t._flags = 0; t._clockid = CLOCK_MONOTONIC;   // relative
        if (::_umtx_op(addr, UMTX_OP_WAIT_UINT_PRIVATE, expected, reinterpret_cast<void*>(sizeof t), &t) == 0 || errno == EINTR) return wait_result::woken;
    }
    if (errno == ETIMEDOUT) return wait_result::timed_out;
    return a.load(std::memory_order_seq_cst) != expected ? wait_result::mismatch : wait_result::woken;
#elif defined(__APPLE__)
    auto& w = detail::aw::api::get();
    if (!w.wait) return detail::aw::table_wait(a, expected, timeout_ns);
    if (a.load(std::memory_order_seq_cst) != expected) return wait_result::mismatch;
    int r = timeout_ns == ~0ull ? w.wait(addr, expected, 4, 0 /*OS_SYNC_WAIT_ON_ADDRESS_NONE*/)
                                : w.wait_timeout(addr, expected, 4, 0, 32 /*OS_CLOCK_MACH_ABSOLUTE_TIME*/, timeout_ns ? timeout_ns : 1);
    return r < 0 && errno == ETIMEDOUT ? wait_result::timed_out : wait_result::woken;
#else
    return detail::aw::table_wait(a, expected, timeout_ns);
#endif
}
namespace detail::aw {
inline void wake(const std::atomic<std::uint32_t>& a, bool all) {
    void* addr = const_cast<void*>(static_cast<const void*>(&a));
#if defined(FORGE_OS_LINUX) || defined(FORGE_OS_ANDROID)
#if defined(SYS_futex)
    ::syscall(SYS_futex, addr, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, all ? INT32_MAX : 1, nullptr, nullptr, 0);
#else
    ::syscall(SYS_futex_time64, addr, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, all ? INT32_MAX : 1, nullptr, nullptr, 0);
#endif
#elif defined(FORGE_OS_WINDOWS)
    auto& w = api::get();
    if (!w.wait) table_wake(addr); else if (all) w.wake_all(addr); else w.wake_one(addr);
#elif defined(FORGE_OS_FREEBSD)
    ::_umtx_op(addr, UMTX_OP_WAKE_PRIVATE, all ? INT32_MAX : 1, nullptr, nullptr);
#elif defined(__APPLE__)
    auto& w = api::get();
    if (!w.wait) table_wake(addr); else if (all) w.wake_all(addr, 4, 0); else w.wake_one(addr, 4, 0);
#else
    (void)all; table_wake(addr);
#endif
}
}
// Wake one / every thread waiting on `a` (the table backend always wakes all of them; that is a spurious
// wake-up for the others, which wait_on_address allows).
inline void wake_one(const std::atomic<std::uint32_t>& a) { detail::aw::wake(a, false); }
inline void wake_all(const std::atomic<std::uint32_t>& a) { detail::aw::wake(a, true); }

// ---------------------------------------------------------------- wait_event
class wait_event {
public:
#if defined(FORGE_OS_WINDOWS)
    using native_t = HANDLE;
#else
    using native_t = int;
#endif
    wait_event() = default;
    wait_event(const wait_event&) = delete; wait_event& operator=(const wait_event&) = delete;
    wait_event(wait_event&& o) noexcept { swap_(o); }
    wait_event& operator=(wait_event&& o) noexcept { if (this != &o) { close(); swap_(o); } return *this; }
    ~wait_event() { close(); }
    static result<wait_event> create(bool initially_set = false) {
        wait_event e;
#if defined(FORGE_OS_WINDOWS)
        e.h_ = CreateEventW(nullptr, TRUE, initially_set ? TRUE : FALSE, nullptr);
        if (!e.h_) return {{}, last_error("CreateEvent")};
#elif defined(FORGE_OS_EVENTFD)
        e.r_ = ::eventfd(initially_set ? 1 : 0, EFD_CLOEXEC | EFD_NONBLOCK);
        if (e.r_ < 0) return {{}, last_error("eventfd")};
#else
        int p[2]; if (cloexec_pipe(p)) return {{}, last_error("pipe")};
        for (int f : p) ::fcntl(f, F_SETFL, ::fcntl(f, F_GETFL) | O_NONBLOCK);
        e.r_ = p[0]; e.w_ = p[1];
        if (initially_set) e.set();
#endif
        return {std::move(e), {}};
    }
    bool valid() const {
#if defined(FORGE_OS_WINDOWS)
        return h_ != nullptr;
#else
        return r_ >= 0;
#endif
    }
    // Raise the flag. Async-signal-safe on POSIX (one write).
    void set() noexcept {
#if defined(FORGE_OS_WINDOWS)
        if (h_) SetEvent(h_);
#elif defined(FORGE_OS_EVENTFD)
        std::uint64_t one = 1; int e = errno; while (r_ >= 0 && ::write(r_, &one, sizeof one) < 0 && errno == EINTR) {} errno = e;
#else
        char c = 1; int e = errno; while (w_ >= 0 && ::write(w_, &c, 1) < 0 && errno == EINTR) {} errno = e;   // a full pipe is already set
#endif
    }
    void reset() noexcept {
#if defined(FORGE_OS_WINDOWS)
        if (h_) ResetEvent(h_);
#elif defined(FORGE_OS_EVENTFD)
        std::uint64_t v; while (r_ >= 0 && ::read(r_, &v, sizeof v) < 0 && errno == EINTR) {}
#else
        char b[64]; for (;;) { auto n = ::read(r_, b, sizeof b); if (n > 0) continue; if (n < 0 && errno == EINTR) continue; break; }
#endif
    }
    // True once the flag is raised (it stays raised: every waiter sees it until reset()); false on timeout
    // (timeout_ms < 0 waits without limit).
    bool wait(int timeout_ms = -1) const {
#if defined(FORGE_OS_WINDOWS)
        return h_ && WaitForSingleObject(h_, timeout_ms < 0 ? INFINITE : DWORD(timeout_ms)) == WAIT_OBJECT_0;
#else
        if (r_ < 0) return false;
        const std::uint64_t end = timeout_ms < 0 ? 0 : monotonic_ns() + std::uint64_t(timeout_ms) * 1000000ull;
        for (;;) {
            int ms = -1;
            if (timeout_ms >= 0) { auto now = monotonic_ns(); ms = now >= end ? 0 : int((end - now + 999999ull) / 1000000ull); }
            struct pollfd f{r_, POLLIN, 0}; int r = ::poll(&f, 1, ms);
            if (r > 0) return true;
            if (r == 0) return false;
            if (errno != EINTR) return false;
        }
#endif
    }
    bool is_set() const { return wait(0); }
    // What to poll for readability (POSIX) or wait on (Windows).
    native_t native() const {
#if defined(FORGE_OS_WINDOWS)
        return h_;
#else
        return r_;
#endif
    }
    void close() {
#if defined(FORGE_OS_WINDOWS)
        if (h_) CloseHandle(h_); h_ = nullptr;
#else
        if (r_ >= 0) ::close(r_); if (w_ >= 0) ::close(w_); r_ = w_ = -1;
#endif
    }
private:
    void swap_(wait_event& o) noexcept {
#if defined(FORGE_OS_WINDOWS)
        std::swap(h_, o.h_);
#else
        std::swap(r_, o.r_); std::swap(w_, o.w_);
#endif
    }
#if defined(FORGE_OS_WINDOWS)
    HANDLE h_ = nullptr;
#else
    int r_ = -1, w_ = -1;
#endif
};

// ---------------------------------------------------------------- wait_timer
class wait_timer {
public:
#if defined(FORGE_OS_WINDOWS)
    using native_t = HANDLE;
#else
    using native_t = int;
#endif
    wait_timer() = default;
    wait_timer(const wait_timer&) = delete; wait_timer& operator=(const wait_timer&) = delete;
    wait_timer(wait_timer&& o) noexcept { swap_(o); }
    wait_timer& operator=(wait_timer&& o) noexcept { if (this != &o) { close(); swap_(o); } return *this; }
    ~wait_timer() { close(); }
    static result<wait_timer> create() {
        wait_timer t;
#if defined(FORGE_OS_WINDOWS)
        t.h_ = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /*CREATE_WAITABLE_TIMER_HIGH_RESOLUTION*/, TIMER_ALL_ACCESS);
        if (!t.h_) t.h_ = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);   // before Windows 10 1803
        if (!t.h_) return {{}, last_error("CreateWaitableTimer")};
#elif defined(FORGE_OS_LINUX) || defined(FORGE_OS_ANDROID)
        t.fd_ = ::timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC | TFD_NONBLOCK);
        if (t.fd_ < 0) return {{}, last_error("timerfd_create")};
#elif defined(FORGE_OS_KQUEUE)
        t.fd_ = ::kqueue();
        if (t.fd_ < 0) return {{}, last_error("kqueue")};
        ::fcntl(t.fd_, F_SETFD, FD_CLOEXEC);
#else
        return {{}, unsupported("wait_timer")};
#endif
        return {std::move(t), {}};
    }
    bool valid() const {
#if defined(FORGE_OS_WINDOWS)
        return h_ != nullptr;
#else
        return fd_ >= 0;
#endif
    }
    // Fire once after first_ns, then every period_ns (0 = one shot). Re-arming replaces the previous schedule.
    status arm(std::uint64_t first_ns, std::uint64_t period_ns = 0) {
        if (!valid()) return {{}, errno_error(EBADF, "wait_timer: not created")};
        if (!first_ns) first_ns = 1;
        period_ = period_ns; due_ = monotonic_ns() + first_ns; armed_ = true;
#if defined(FORGE_OS_WINDOWS)
        return set_(first_ns);
#elif defined(FORGE_OS_LINUX) || defined(FORGE_OS_ANDROID)
        struct itimerspec s{};
        s.it_value.tv_sec = time_t(first_ns / 1000000000ull); s.it_value.tv_nsec = long(first_ns % 1000000000ull);
        s.it_interval.tv_sec = time_t(period_ns / 1000000000ull); s.it_interval.tv_nsec = long(period_ns % 1000000000ull);
        if (::timerfd_settime(fd_, 0, &s, nullptr)) return {{}, last_error("timerfd_settime")};
        return {};
#else
        return set_(first_ns, period_ns && period_ns == first_ns);
#endif
    }
    status disarm() {
        if (!valid()) return {};
        armed_ = false;
#if defined(FORGE_OS_WINDOWS)
        CancelWaitableTimer(h_);
#elif defined(FORGE_OS_LINUX) || defined(FORGE_OS_ANDROID)
        struct itimerspec s{}; ::timerfd_settime(fd_, 0, &s, nullptr);
        std::uint64_t n; (void)!::read(fd_, &n, sizeof n);
#else
        struct kevent k; EV_SET(&k, 1, EVFILT_TIMER, EV_DELETE, 0, 0, nullptr); ::kevent(fd_, &k, 1, nullptr, 0, nullptr);
#endif
        return {};
    }
    // Wait for the timer: the number of expirations since the last wait (periodic timers that fell behind report
    // more than one), or 0 when timeout_ms (< 0: no limit) passes first.
    result<std::uint64_t> wait(int timeout_ms = -1) {
        if (!valid()) return {0, errno_error(EBADF, "wait_timer: not created")};
#if defined(FORGE_OS_WINDOWS)
        DWORD w = WaitForSingleObject(h_, timeout_ms < 0 ? INFINITE : DWORD(timeout_ms));
        if (w == WAIT_TIMEOUT) return {0, {}};
        if (w != WAIT_OBJECT_0) return {0, last_error("wait_timer")};
        return {fired_(), {}};
#elif defined(FORGE_OS_LINUX) || defined(FORGE_OS_ANDROID)
        const std::uint64_t end = timeout_ms < 0 ? 0 : monotonic_ns() + std::uint64_t(timeout_ms) * 1000000ull;
        for (;;) {
            std::uint64_t n = 0; auto r = ::read(fd_, &n, sizeof n);
            if (r == ssize_t(sizeof n)) { if (!period_) armed_ = false; return {n, {}}; }
            if (r < 0 && errno != EAGAIN && errno != EINTR) return {0, last_error("timerfd read")};
            int ms = -1;
            if (timeout_ms >= 0) { auto now = monotonic_ns(); if (now >= end) return {0, {}}; ms = int((end - now + 999999ull) / 1000000ull); }
            struct pollfd f{fd_, POLLIN, 0}; if (::poll(&f, 1, ms) < 0 && errno != EINTR) return {0, last_error("poll")};
        }
#else
        struct timespec ts{}, *tp = nullptr;
        if (timeout_ms >= 0) { ts.tv_sec = timeout_ms / 1000; ts.tv_nsec = long(timeout_ms % 1000) * 1000000L; tp = &ts; }
        struct kevent ev; int n;
        do n = ::kevent(fd_, nullptr, 0, &ev, 1, tp); while (n < 0 && errno == EINTR);
        if (n < 0) return {0, last_error("kevent")};
        if (n == 0) return {0, {}};
        std::uint64_t count = ev.data > 0 ? std::uint64_t(ev.data) : 1;
        if (oneshot_) {   // the first interval differs from the period: continue with the period from here
            if (period_) { if (auto s = set_(period_, true); !s) return {count, s.err}; } else armed_ = false;
        }
        return {count, {}};
#endif
    }
    bool armed() const { return armed_; }
    native_t native() const {
#if defined(FORGE_OS_WINDOWS)
        return h_;
#else
        return fd_;
#endif
    }
    void close() {
#if defined(FORGE_OS_WINDOWS)
        if (h_) CloseHandle(h_); h_ = nullptr;
#else
        if (fd_ >= 0) ::close(fd_); fd_ = -1;
#endif
        armed_ = false;
    }
private:
    void swap_(wait_timer& o) noexcept {
        std::swap(period_, o.period_); std::swap(due_, o.due_); std::swap(armed_, o.armed_); std::swap(oneshot_, o.oneshot_);
#if defined(FORGE_OS_WINDOWS)
        std::swap(h_, o.h_);
#else
        std::swap(fd_, o.fd_);
#endif
    }
#if defined(FORGE_OS_WINDOWS)
    // One shot per SetWaitableTimer (its own period is in whole milliseconds); fired_() schedules the next one
    // from the due time, so a periodic timer does not drift.
    status set_(std::uint64_t ns) {
        LARGE_INTEGER d; d.QuadPart = -LONGLONG((ns + 99) / 100);
        if (!SetWaitableTimer(h_, &d, 0, nullptr, nullptr, FALSE)) return {{}, last_error("SetWaitableTimer")};
        return {};
    }
    std::uint64_t fired_() {
        if (!period_) { armed_ = false; return 1; }
        std::uint64_t now = monotonic_ns(), n = 1;
        if (now > due_) n += (now - due_) / period_;
        due_ += n * period_;
        (void)set_(due_ > now ? due_ - now : 1);
        return n;
    }
    HANDLE h_ = nullptr;
#else
#if defined(FORGE_OS_KQUEUE)
    status set_(std::uint64_t ns, bool periodic) {
        struct kevent k; oneshot_ = !periodic;
        EV_SET(&k, 1, EVFILT_TIMER, EV_ADD | EV_ENABLE | (periodic ? 0 : EV_ONESHOT), NOTE_NSECONDS, std::int64_t(ns), nullptr);
        if (::kevent(fd_, &k, 1, nullptr, 0, nullptr) < 0) return {{}, last_error("kevent EVFILT_TIMER")};
        return {};
    }
#endif
    int fd_ = -1;
#endif
    std::uint64_t period_ = 0, due_ = 0; bool armed_ = false, oneshot_ = false;
};

// ---------------------------------------------------------------- signal_listener
// Process-wide: listen(s) routes `s` into a queue (replacing any handler set with os::on_signal), wait() takes
// the next one. Signals that arrive several times before a wait() are reported once each time they were seen
// between waits. Windows delivers console control events (see os::on_signal); sig::pipe does not exist there.
class signal_listener {
public:
    static signal_listener& instance() { static signal_listener l; return l; }
    status listen(sig s) {
        std::lock_guard<std::mutex> g(m_);
        if (!ev_.valid()) { auto e = wait_event::create(); if (!e) return {{}, e.err}; ev_ = std::move(e.value); slot_().store(this); }
#if defined(FORGE_OS_WINDOWS)
        if (s == sig::pipe) return {{}, unsupported("SIGPIPE")};
#endif
        if (!on_signal(s, &signal_listener::handler_)) return {{}, last_error("on_signal")};
        return {};
    }
    // Back to the default action for `s`.
    void ignore(sig s) { on_signal(s, nullptr); }
    // The next signal, or nullopt after timeout_ms (< 0: no limit).
    std::optional<sig> wait(int timeout_ms = -1) {
        const std::uint64_t end = timeout_ms < 0 ? 0 : monotonic_ns() + std::uint64_t(timeout_ms) * 1000000ull;
        for (;;) {
            if (auto s = take_()) return s;
            int ms = -1;
            if (timeout_ms >= 0) { auto now = monotonic_ns(); if (now >= end) return std::nullopt; ms = int((end - now + 999999ull) / 1000000ull); }
            if (!ev_.valid() || !ev_.wait(ms)) { if (auto s = take_()) return s; if (timeout_ms >= 0 || !ev_.valid()) return std::nullopt; }
        }
    }
    // Readable (POSIX) / signalled (Windows) while a signal is queued: for an event loop.
    wait_event::native_t native() const { return ev_.native(); }
private:
    signal_listener() = default;
    static std::atomic<signal_listener*>& slot_() { static std::atomic<signal_listener*> s{nullptr}; return s; }
    static void handler_(int n) {   // signal context / console-control thread: atomics and one write only
        signal_listener* l = slot_().load(); if (!l) return;
        unsigned bit = 0;
#if defined(FORGE_OS_WINDOWS)
        bit = n == 2 ? 1u : n == 15 ? 2u : n == 1 ? 4u : 0u;
#else
        bit = n == SIGINT ? 1u : n == SIGTERM ? 2u : n == SIGHUP ? 4u : n == SIGPIPE ? 8u : 0u;
#endif
        l->pending_.fetch_or(bit); l->ev_.set();
    }
    std::optional<sig> take_() {
        ev_.reset();   // before reading the bits: a signal after this point raises the event again
        for (;;) {
            unsigned p = pending_.load(); if (!p) return std::nullopt;
            unsigned bit = p & (~p + 1u);
            if (!pending_.compare_exchange_weak(p, p & ~bit)) continue;
            if (p & ~bit) ev_.set();
            return bit == 1 ? sig::interrupt : bit == 2 ? sig::terminate : bit == 4 ? sig::hangup : sig::pipe;
        }
    }
    std::mutex m_; wait_event ev_; std::atomic<unsigned> pending_{0};
};

} // namespace forge::os

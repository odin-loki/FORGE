#pragma once
// forge::os::reactor: one event loop over sockets, pipes, local IPC streams, timers, posted functions, child
// exits, signals, wait_events, file-change notifications and file I/O. Guide: docs/os-wrapper.md.
//
//   The API is completion-shaped on every OS: start an operation, a callback (or a co_await) receives the result.
//   What waits underneath:
//     Linux / Android   epoll; eventfd wake-up; pidfd for exits (5.3+ / Android 12+, else a helper thread).
//     macOS / iOS / FreeBSD   kqueue (EVFILT_READ / WRITE / PROC); eventfd or pipe wake-up.
//     Windows           one I/O completion port as the queue. Sockets: WSAEventSelect + a thread-pool wait
//                       (RegisterWaitForSingleObject) that posts to the port, then non-blocking calls on the loop
//                       thread. Other handles (pipes, named pipes, consoles): ReadFile / WriteFile started on a
//                       helper thread; an overlapped handle is handed back to a thread-pool wait at once, a
//                       synchronous one keeps that helper thread until it finishes. Process handles and
//                       wait_events: thread-pool waits. Documented Win32 only.
//     File I/O          os::async_file_io (io_uring / IOCP / POSIX AIO / threads) on a service thread.
//     Signals           os::signal_listener.  File changes: os::file_watcher (its descriptor on POSIX; polled
//                       every 20 ms on Windows).
//   Timers are a deadline queue inside the loop: nanosecond deadlines, fired in deadline order (ties in creation
//   order), never early; the wait itself has millisecond granularity on epoll and Windows.
//
//   Threading: post(), stop(), hold() and unhold() may be called from any thread. Everything else belongs to
//   the thread that runs the loop (or to any one thread while the loop is not running). Callbacks run on the
//   loop thread, never inside the call that started the operation.
//   Every operation completes exactly once. cancel(id) completes it with is_cancelled(err). Buffers, sockets,
//   files and handles passed to an operation must stay valid until its callback has run; file and helper-thread
//   operations cannot be interrupted, so their callback (with the cancellation error) runs once the OS call has
//   returned. Destroying the reactor drops pending callbacks without calling them (after waiting for in-flight
//   OS calls); call cancel_all() and run() first when every callback must run.
//   Sockets and descriptors given to the reactor are switched to non-blocking mode and stay that way.
//   POSIX: writing to a pipe whose reader has gone raises SIGPIPE unless the process ignores it
//   (os::ignore_sigpipe()); sockets are written with MSG_NOSIGNAL / SO_NOSIGPIPE and only report EPIPE.
#include <forge/os/detail.hpp>
#include <forge/os/sync.hpp>
#include <forge/os/async_file.hpp>
#include <forge/os/process.hpp>   // wait_for_exit (exit fallback)
#include <concepts>
#include <coroutine>
#include <limits>
#include <functional>
#include <set>
#include <unordered_map>

namespace forge::os {

inline error cancelled_error(std::string_view what = "operation") { return errno_error(ECANCELED, std::string(what) + ": cancelled"); }
inline bool is_cancelled(const error& e) { return e.code == ECANCELED && e.domain == err_domain::posix; }

// What an I/O operation works on: a socket, or a stream descriptor / handle (pipe end, local_stream, tty).
// Built from os::socket, os::pipe_end, os::local_stream, os::wait_event ... (anything with native()), or from
// the raw value.
struct io_handle {
#if defined(FORGE_OS_WINDOWS)
    HANDLE h = INVALID_HANDLE_VALUE; SOCKET s = INVALID_SOCKET; bool is_socket = false;
    io_handle(SOCKET x) : s(x), is_socket(true) {}
    io_handle(HANDLE x) : h(x) {}
    std::uintptr_t key() const { return is_socket ? std::uintptr_t(s) : reinterpret_cast<std::uintptr_t>(h); }
#else
    int fd = -1; bool is_socket = false;
    io_handle(int x) : fd(x) {}
    std::uintptr_t key() const { return std::uintptr_t(unsigned(fd)); }
#endif
    template <class T> requires requires(const T& t) { t.native(); } io_handle(const T& t) : io_handle(t.native()) {}
};

class reactor {
public:
    struct op_id { std::uint64_t v = 0; explicit operator bool() const { return v != 0; } bool operator==(const op_id& o) const { return v == o.v; } };
    using io_fn = std::function<void(const error&, std::size_t)>;
    using done_fn = std::function<void(const error&)>;
    using accept_fn = std::function<void(const error&, socket)>;
    using signal_fn = std::function<void(sig)>;

    reactor() {
#if defined(FORGE_OS_WINDOWS)
        net_init();
        port_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
        if (!port_) { init_err_ = last_error("CreateIoCompletionPort"); return; }
#else
#if defined(FORGE_OS_EPOLL)
        q_ = ::epoll_create1(EPOLL_CLOEXEC);
#else
        q_ = ::kqueue();
        if (q_ >= 0) ::fcntl(q_, F_SETFD, FD_CLOEXEC);
#endif
        if (q_ < 0) { init_err_ = last_error("reactor queue"); return; }
        auto w = wait_event::create(); if (!w) { init_err_ = w.err; return; }
        wake_ = std::move(w.value);
        op* o = new_op_(kind::wait_r); o->internal = o->persistent = true; o->key = std::uintptr_t(unsigned(wake_.native()));
        o->tick = [this] { wake_.reset(); }; o->inline_tick = true;   // reset before woken_ is cleared (see run_once)
        enqueue_(o, true);
#endif
    }
    ~reactor() { shutdown_(); }
    reactor(const reactor&) = delete; reactor& operator=(const reactor&) = delete;

    bool valid() const { return !init_err_; }
    const error& init_error() const { return init_err_; }
    const char* backend() const {
#if defined(FORGE_OS_EPOLL)
        return "epoll";
#elif defined(FORGE_OS_KQUEUE)
        return "kqueue";
#else
        return "iocp";
#endif
    }

    // ---------------------------------------------------------------- loop control
    // Run fn on the loop thread (any thread may call this; fn runs in a later loop iteration, in posting order).
    void post(std::function<void()> fn) {
        { std::lock_guard<std::mutex> g(pm_); posted_.push_back(std::move(fn)); }
        wake_up_();
    }
    // Make run() return as soon as the current callback is done. restart() clears the request.
    void stop() { stopped_.store(true); wake_up_(); }
    void restart() { stopped_.store(false); }
    bool stopped() const { return stopped_.load(); }
    // While held, run() keeps waiting even when nothing is pending (for work other threads will post).
    void hold() { holds_.fetch_add(1); }
    void unhold() { holds_.fetch_sub(1); wake_up_(); }
    // Operations that have not completed yet (timers, I/O, watches).
    std::size_t pending() const { return work_; }
    // Runs until stop(), or until nothing is pending, queued or held. Returns the number of callbacks run.
    std::size_t run() {
        std::size_t n = 0;
        while (!stopped_.load() && has_work_()) n += run_once(-1);
        return n;
    }
    // Runs for at most ms milliseconds (or until stop()), also while idle.
    std::size_t run_for(int ms) {
        std::size_t n = 0; const std::uint64_t end = monotonic_ns() + std::uint64_t(ms < 0 ? 0 : ms) * 1000000ull;
        while (!stopped_.load()) {
            auto now = monotonic_ns(); if (now >= end) break;
            n += run_once(int((end - now + 999999ull) / 1000000ull));
        }
        return n;
    }
    // One iteration: waits up to timeout_ms (-1: until something happens) and runs what became ready.
    std::size_t run_once(int timeout_ms) {
        if (!valid()) return 0;
        take_posted_();
        std::uint64_t now = monotonic_ns();
        int ms = timeout_ms;
        if (!ready_.empty()) ms = 0;
        else if (!timers_.empty()) {
            std::uint64_t d = timers_.begin()->first;
            int t = d <= now ? 0 : int(std::min<std::uint64_t>((d - now + 999999ull) / 1000000ull, 0x7fffffffull));
            if (ms < 0 || t < ms) ms = t;
        }
#if defined(FORGE_OS_WINDOWS)
        if (!cancelling_.empty() && (ms < 0 || ms > 10)) ms = 10;   // re-issue CancelIoEx for calls that had not started
#endif
        poll_(ms);
        woken_.store(false);
        fire_timers_();
        take_posted_();
        std::size_t n = ready_.size(), ran = 0;   // callbacks queued by these callbacks wait for the next iteration
        for (; ran < n && !ready_.empty(); ++ran) { auto fn = std::move(ready_.front()); ready_.pop_front(); fn(); if (stopped_.load()) { ++ran; break; } }
        return ran;
    }

    // ---------------------------------------------------------------- timers
    // cb(no error) after delay_ns; with period_ns, again every period_ns until cancelled (then cb(cancelled)).
    op_id after(std::uint64_t delay_ns, done_fn cb) { return timer_(monotonic_ns() + delay_ns, 0, std::move(cb)); }
    op_id at(std::uint64_t monotonic_deadline_ns, done_fn cb) { return timer_(monotonic_deadline_ns, 0, std::move(cb)); }
    op_id every(std::uint64_t period_ns, done_fn cb) { if (!period_ns) period_ns = 1; return timer_(monotonic_ns() + period_ns, period_ns, std::move(cb)); }

    // ---------------------------------------------------------------- stream I/O
    // read: cb(err, n) with n >= 1 bytes, or n == 0 at end of stream. read_exact: all n bytes, or an error (EPIPE
    // at end of stream) with the count read so far. write: all n bytes, or an error with the count written.
    op_id read(io_handle h, void* buf, std::size_t n, io_fn cb) { return io_(kind::read, h, buf, n, false, std::move(cb)); }
    op_id read_exact(io_handle h, void* buf, std::size_t n, io_fn cb) { return io_(kind::read, h, buf, n, true, std::move(cb)); }
    op_id write(io_handle h, const void* buf, std::size_t n, io_fn cb) { return io_(kind::write, h, const_cast<void*>(buf), n, true, std::move(cb)); }
    op_id write_some(io_handle h, const void* buf, std::size_t n, io_fn cb) { return io_(kind::write, h, const_cast<void*>(buf), n, false, std::move(cb)); }
    // cb when the socket / descriptor is readable or writable (nothing is consumed). Windows: sockets only.
    op_id wait_readable(io_handle h, done_fn cb) { return wait_io_(kind::wait_r, h, std::move(cb)); }
    op_id wait_writable(io_handle h, done_fn cb) { return wait_io_(kind::wait_w, h, std::move(cb)); }
    // One connection from a listening socket (os::socket::listen, net::unix_listen ...).
    op_id accept(const socket& listener, accept_fn cb) {
        op* o = new_op_(kind::accept); const std::uint64_t oid = o->id; o->scb = std::move(cb); o->key = std::uintptr_t(listener.native()); o->is_socket = true;
        if (!valid()) return fail_(o, init_err_);
        if (!listener.valid()) return fail_(o, errno_error(EBADF, "accept: invalid socket"));
        start_sock_(o, true); return {oid};
    }
    // TCP connect to host:port: resolves on a helper thread, then tries each address in turn.
    op_id connect(const std::string& host, int port, accept_fn cb) {
        op* o = new_op_(kind::connect); const std::uint64_t oid = o->id; o->scb = std::move(cb); o->is_socket = true;
        if (!valid()) return fail_(o, init_err_);
        track_(o); auto st = std::make_shared<resolve_state>(); o->res = st; std::uint64_t id = o->id;
        o->what = "connect " + host + ":" + std::to_string(port);
        offload_([st, host, port] { auto r = socket::resolve(host.c_str(), port, false); if (r) st->ai = r.value; else st->err = r.err; },
                 [this, id, st] {
                     op* p = find_(id); if (!p) return;   // cancelled meanwhile
                     if (p->cancel_requested) { finish_(p, cancelled_error(p->what), 0); return; }
                     if (st->err) { finish_(p, st->err, 0); return; }
                     p->ai_next = st->ai; p->last_err = 0; connect_next_(p);
                 });
        return {oid};
    }

    // ---------------------------------------------------------------- files (through os::async_file_io)
    // cb(err, bytes); 0 bytes with no error is end of file. f must stay open until the callback has run.
    op_id read_file(file& f, void* buf, std::size_t n, std::uint64_t off, io_fn cb) { return file_(f, buf, n, off, false, std::move(cb)); }
    op_id write_file(file& f, const void* buf, std::size_t n, std::uint64_t off, io_fn cb) { return file_(f, const_cast<void*>(buf), n, off, true, std::move(cb)); }
    // Name of the async_file_io backend the file service uses ("" before the first file operation).
    std::string file_backend() const { return fsvc_ ? fsvc_->backend_name() : std::string(); }

    // ---------------------------------------------------------------- other waitables
    // cb once process `pid` has exited (a child is not reaped: call child::wait for its code). A pid that does not
    // exist completes at once.
    op_id on_exit(std::uint64_t pid, done_fn cb) {
        op* o = new_op_(kind::exit); const std::uint64_t oid = o->id; o->dcb = std::move(cb); o->what = "on_exit";
        if (!valid()) return fail_(o, init_err_);
        start_exit_(o, pid); return {oid};
    }
    // cb once `e` is set (it is not reset). e must outlive the operation.
    op_id on_event(const wait_event& e, done_fn cb) {
        op* o = new_op_(kind::wait_r); const std::uint64_t oid = o->id; o->dcb = std::move(cb); o->what = "on_event";
        if (!valid()) return fail_(o, init_err_);
        if (!e.valid()) return fail_(o, errno_error(EBADF, "on_event: invalid event"));
#if defined(FORGE_OS_WINDOWS)
        o->k = kind::hwait; if (!arm_wait_(o, e.native(), false)) return fail_(o, last_error("RegisterWaitForSingleObject"));
#else
        o->key = std::uintptr_t(unsigned(e.native())); enqueue_(o, true);
#endif
        return {oid};
    }
    // cb(s) each time signal `s` arrives (os::signal_listener: process-wide, replaces os::on_signal handlers),
    // until cancelled (no callback then). One reactor per process should watch signals. An empty id: the signal
    // cannot be watched here.
    op_id on_signal(sig s, signal_fn cb) {
        if (!valid() || !signal_listener::instance().listen(s)) return {};   // {}: not watching (e.g. SIGPIPE on Windows)
        op* o = new_op_(kind::signal); const std::uint64_t oid = o->id; o->what = "on_signal"; o->persistent = true; o->sig_cb = std::move(cb); o->sig_which = s;
        track_(o); sig_ops_.push_back(o->id);
        if (!sig_watch_) {
            op* w = new_op_(kind::wait_r); w->internal = w->persistent = true; w->what = "signals"; sig_watch_ = w->id;
            w->tick = [this] { drain_signals_(); };
#if defined(FORGE_OS_WINDOWS)
            w->k = kind::hwait; w->rearm = signal_listener::instance().native();
            if (!arm_wait_(w, w->rearm, false)) { drop_(w); sig_watch_ = 0; }
#else
            w->key = std::uintptr_t(unsigned(signal_listener::instance().native())); enqueue_(w, true);
#endif
        }
        return {oid};
    }
    // cb(events) whenever watcher `w` (os::file_watcher, os::tree_watcher: anything with poll(int)) has events,
    // until cancelled. POSIX: the watcher's descriptor is part of the loop; Windows: polled every 20 ms. The
    // watcher must outlive the operation and is only used from the loop thread.
    template <class W, class F> op_id watch(W& w, F cb) {
        auto pump = [&w, cb = std::move(cb)]() mutable { auto ev = w.poll(0); if (!ev.empty()) cb(std::move(ev)); };
#if defined(FORGE_OS_POSIX)
        if constexpr (requires { { w.native_handle() } -> std::convertible_to<int>; }) {
            int fd = w.native_handle();
            if (fd >= 0) {
                op* o = new_op_(kind::wait_r); const std::uint64_t oid = o->id; o->persistent = true; o->what = "watch"; o->tick = std::move(pump);
                if (!valid()) return fail_(o, init_err_);
                o->key = std::uintptr_t(unsigned(fd)); enqueue_(o, true); return {oid};
            }
        }
#endif
        return every(20'000'000ull, [pump = std::move(pump)](const error& e) mutable { if (!e) pump(); });
    }
    // Run work() on a helper thread, then cb on the loop thread (cb(cancelled) if cancelled; work still runs
    // unless it had not started).
    op_id offload(std::function<void()> work, done_fn cb) {
        op* o = new_op_(kind::offload); const std::uint64_t oid = o->id; o->dcb = std::move(cb); o->what = "offload";
        if (!valid()) return fail_(o, init_err_);
        track_(o); std::uint64_t id = o->id; auto skip = std::make_shared<std::atomic<bool>>(false); o->skip = skip;
        offload_([work = std::move(work), skip] { if (!skip->load()) work(); },
                 [this, id] { if (op* p = find_(id)) finish_(p, p->cancel_requested ? cancelled_error(p->what) : error{}, 0); });
        return {oid};
    }

    // Cancel one operation: true if it was pending. Its callback runs with is_cancelled(err) (at once for timers,
    // readiness waits and queued I/O; after the OS call returns for file and helper-thread operations).
    bool cancel(op_id id) {
        op* o = find_(id.v); if (!o || o->internal) return false;
        cancel_(o); return true;
    }
    // Cancel everything pending. run() then delivers the callbacks.
    void cancel_all() {
        std::vector<std::uint64_t> ids; for (auto& [id, o] : ops_) if (!o->internal) ids.push_back(id);
        std::sort(ids.begin(), ids.end());
        for (auto id : ids) if (op* o = find_(id)) cancel_(o);
    }
    // Forget what the reactor keeps for a socket that has no pending operations (Windows: the WSAEventSelect
    // association; the socket stays non-blocking). Call it before using such a socket with another event loop.
    void forget(io_handle h) {
#if defined(FORGE_OS_WINDOWS)
        if (!h.is_socket) return;
        auto it = fds_.find(h.key()); if (it == fds_.end() || !it->second.rd.empty() || !it->second.wr.empty()) return;
        release_sock_(it->second, h.s, true); fds_.erase(it);
#else
        (void)h;
#endif
    }

    // ---------------------------------------------------------------- coroutines
    struct io_result { error err; std::size_t n = 0; explicit operator bool() const { return !err; } };
    struct accept_result { error err; socket sock; explicit operator bool() const { return !err; } };
    template <class R> struct awaiter {
        std::function<void(std::function<void(R)>)> start; R value{};
        bool await_ready() const noexcept { return false; }
        void await_suspend(std::coroutine_handle<> h) { start([this, h](R r) { value = std::move(r); h.resume(); }); }
        R await_resume() { return std::move(value); }
    };
    // A coroutine that starts at once and owns itself: `reactor::task f(reactor& r) { co_await r.async_sleep(...); }`.
    struct task {
        struct promise_type {
            task get_return_object() noexcept { return {}; }
            std::suspend_never initial_suspend() noexcept { return {}; }
            std::suspend_never final_suspend() noexcept { return {}; }
            void return_void() noexcept {}
            void unhandled_exception() noexcept { std::terminate(); }
        };
    };
    awaiter<io_result> async_read(io_handle h, void* buf, std::size_t n) {
        return {[this, h, buf, n](std::function<void(io_result)> k) { read(h, buf, n, [k = std::move(k)](const error& e, std::size_t got) { k({e, got}); }); }};
    }
    awaiter<io_result> async_read_exact(io_handle h, void* buf, std::size_t n) {
        return {[this, h, buf, n](std::function<void(io_result)> k) { read_exact(h, buf, n, [k = std::move(k)](const error& e, std::size_t got) { k({e, got}); }); }};
    }
    awaiter<io_result> async_write(io_handle h, const void* buf, std::size_t n) {
        return {[this, h, buf, n](std::function<void(io_result)> k) { write(h, buf, n, [k = std::move(k)](const error& e, std::size_t got) { k({e, got}); }); }};
    }
    awaiter<io_result> async_read_file(file& f, void* buf, std::size_t n, std::uint64_t off) {
        return {[this, &f, buf, n, off](std::function<void(io_result)> k) { read_file(f, buf, n, off, [k = std::move(k)](const error& e, std::size_t got) { k({e, got}); }); }};
    }
    awaiter<io_result> async_write_file(file& f, const void* buf, std::size_t n, std::uint64_t off) {
        return {[this, &f, buf, n, off](std::function<void(io_result)> k) { write_file(f, buf, n, off, [k = std::move(k)](const error& e, std::size_t got) { k({e, got}); }); }};
    }
    awaiter<std::shared_ptr<accept_result>> async_accept(const socket& listener) {
        return {[this, &listener](std::function<void(std::shared_ptr<accept_result>)> k) {
            accept(listener, [k = std::move(k)](const error& e, socket s) { auto r = std::make_shared<accept_result>(); r->err = e; r->sock = std::move(s); k(std::move(r)); }); }};
    }
    awaiter<std::shared_ptr<accept_result>> async_connect(std::string host, int port) {
        return {[this, host = std::move(host), port](std::function<void(std::shared_ptr<accept_result>)> k) {
            connect(host, port, [k = std::move(k)](const error& e, socket s) { auto r = std::make_shared<accept_result>(); r->err = e; r->sock = std::move(s); k(std::move(r)); }); }};
    }
    awaiter<error> async_sleep(std::uint64_t ns) { return {[this, ns](std::function<void(error)> k) { after(ns, [k = std::move(k)](const error& e) { k(e); }); }}; }
    awaiter<error> async_exit(std::uint64_t pid) { return {[this, pid](std::function<void(error)> k) { on_exit(pid, [k = std::move(k)](const error& e) { k(e); }); }}; }
    awaiter<error> async_event(const wait_event& ev) { return {[this, &ev](std::function<void(error)> k) { on_event(ev, [k = std::move(k)](const error& e) { k(e); }); }}; }
    // Continue in a later loop iteration (also moves a coroutine onto the loop thread).
    awaiter<error> async_yield() { return {[this](std::function<void(error)> k) { post([k = std::move(k)] { k(error{}); }); }}; }

private:
    enum class kind : std::uint8_t { read, write, accept, connect, wait_r, wait_w, timer, exit, signal, offload, file, hio, hwait };
    struct resolve_state { addrinfo* ai = nullptr; error err; ~resolve_state() { if (ai) ::freeaddrinfo(ai); } };
#if defined(FORGE_OS_WINDOWS)
    struct wait_reg { HANDLE port = nullptr; std::uint64_t id = 0; HANDLE wait = nullptr; };
    struct hio_state {   // shared with the helper thread that starts the call
        OVERLAPPED ov{}; HANDLE h = INVALID_HANDLE_VALUE; char* buf = nullptr; DWORD len = 0; bool wr = false;
        std::atomic<int> phase{0};   // 0 queued, 1 in the call, 2 pending (overlapped), 3 done
        std::atomic<bool> cancel{false}; DWORD got = 0, err = 0;
        ~hio_state() { if (ov.hEvent) CloseHandle(ov.hEvent); }
    };
#endif
    struct op {
        std::uint64_t id = 0; kind k = kind::timer; bool internal = false, persistent = false, inline_tick = false, is_socket = false, all = false, cancel_requested = false, own_fd = false, notsock = false;
        std::uintptr_t key = 0; char* buf = nullptr; std::size_t len = 0, done = 0;
        io_fn cb; done_fn dcb; accept_fn scb; signal_fn sig_cb; sig sig_which = sig::interrupt; std::function<void()> tick;
        std::uint64_t deadline = 0, period = 0; std::string what;
        std::shared_ptr<resolve_state> res; addrinfo* ai_next = nullptr; sock_t csock = bad_sock; int last_err = 0;
        std::shared_ptr<std::atomic<bool>> skip;
#if defined(FORGE_OS_WINDOWS)
        wait_reg* reg = nullptr; std::uint64_t wid = 0; HANDLE own_handle = nullptr, hio_h = nullptr; bool hio_wr = false; HANDLE rearm = nullptr; std::shared_ptr<hio_state> hio;
#endif
    };
    struct fd_state {
        unsigned mask = 0; std::deque<op*> rd, wr;
#if defined(FORGE_OS_WINDOWS)
        HANDLE ev = nullptr; wait_reg* reg = nullptr; std::uint64_t wid = 0; int conn_err = 0;
#endif
    };

    // ---- bookkeeping
    op* new_op_(kind k) { op* o = new op; o->id = ++next_id_; o->k = k; return o; }
    op* find_(std::uint64_t id) { auto it = ops_.find(id); return it == ops_.end() ? nullptr : it->second; }
    void track_(op* o) { if (ops_.emplace(o->id, o).second && !o->internal) ++work_; }
    bool has_work_() {
        if (work_ || !ready_.empty() || holds_.load() > 0) return true;
        std::lock_guard<std::mutex> g(pm_); return !posted_.empty();
    }
    void take_posted_() {
        std::deque<std::function<void()>> q;
        { std::lock_guard<std::mutex> g(pm_); q.swap(posted_); }
        for (auto& f : q) ready_.push_back(std::move(f));
    }
    void wake_up_() {
        if (woken_.exchange(true)) return;
#if defined(FORGE_OS_WINDOWS)
        if (port_) PostQueuedCompletionStatus(port_, 0, key_wake, nullptr);
#else
        wake_.set();
#endif
    }
    // Queue the callback of a finished operation and free it. The operation is no longer findable.
    void finish_(op* o, const error& e, std::size_t n) {
        if (ops_.erase(o->id) && !o->internal) --work_;
        release_(o);
        if (o->k == kind::signal) { for (auto it = sig_ops_.begin(); it != sig_ops_.end(); ++it) if (*it == o->id) { sig_ops_.erase(it); break; } }
        if (o->scb) {
            auto sp = std::make_shared<socket>(o->csock); o->csock = bad_sock;
            if (e) sp->close();
            ready_.push_back([cb = std::move(o->scb), e, sp] { cb(e, std::move(*sp)); });
        } else if (o->cb) ready_.push_back([cb = std::move(o->cb), e, n] { cb(e, n); });
        else if (o->dcb) ready_.push_back([cb = std::move(o->dcb), e] { cb(e); });
        delete o;
    }
    op_id fail_(op* o, const error& e) { std::uint64_t id = o->id; track_(o); finish_(o, e, 0); return {id}; }
    void drop_(op* o) { if (ops_.erase(o->id) && !o->internal) --work_; release_(o); delete o; }
    // OS resources an operation owns.
    void release_(op* o) {
#if defined(FORGE_OS_WINDOWS)
        if (o->reg) { unregister_(o->reg); o->reg = nullptr; }
        if (o->wid) { waits_.erase(o->wid); o->wid = 0; }
        if (o->own_handle) { CloseHandle(o->own_handle); o->own_handle = nullptr; }
        if (o->csock != bad_sock && !o->scb) { ::closesocket(o->csock); o->csock = bad_sock; }
#else
        if (o->own_fd && int(o->key) >= 0) { ::close(int(o->key)); o->own_fd = false; }
        if (o->csock != bad_sock && !o->scb) { ::close(o->csock); o->csock = bad_sock; }
#endif
    }

    // ---- timers
    op_id timer_(std::uint64_t deadline, std::uint64_t period, done_fn cb) {
        op* o = new_op_(kind::timer); const std::uint64_t oid = o->id; o->dcb = std::move(cb); o->deadline = deadline; o->period = period; o->what = "timer";
        if (!valid()) return fail_(o, init_err_);
        track_(o); timers_.insert({deadline, o->id}); return {oid};
    }
    void fire_timers_() {
        if (timers_.empty()) return;
        const std::uint64_t now = monotonic_ns();
        while (!timers_.empty() && timers_.begin()->first <= now) {
            std::uint64_t id = timers_.begin()->second; timers_.erase(timers_.begin());
            op* o = find_(id); if (!o) continue;
            if (o->period) {
                o->deadline += o->period; if (o->deadline <= now) o->deadline = now + o->period;   // skip missed ticks
                timers_.insert({o->deadline, id});
                ready_.push_back([this, id] { if (op* p = find_(id)) { auto cb = p->dcb; cb(error{}); } });
            } else finish_(o, {}, 0);
        }
    }

    // ---- helper threads
    void offload_(std::function<void()> work, std::function<void()> then) {
        auto job = [this, work = std::move(work), then = std::move(then)]() mutable { work(); post(std::move(then)); };
        std::lock_guard<std::mutex> g(hm_);
        hq_.push_back(std::move(job));
        if (hidle_ == 0 && helpers_.size() < max_helpers_) helpers_.emplace_back([this] { helper_main_(); });
        hcv_.notify_one();
    }
    void helper_main_() {
        std::unique_lock<std::mutex> lk(hm_);
        for (;;) {
            ++hidle_; hcv_.wait(lk, [&] { return hstop_ || !hq_.empty(); }); --hidle_;
            if (hstop_) return;
            auto job = std::move(hq_.front()); hq_.pop_front();
            lk.unlock(); job(); lk.lock();
        }
    }

    // ---- file service: one thread owns the async_file_io
    struct file_req { file* f; void* buf; std::size_t n; std::uint64_t off, id; bool wr; };
    struct file_service {
        reactor& r; std::mutex m; std::condition_variable cv; std::deque<file_req> q; bool stop = false; std::thread th;
        std::atomic<const char*> name{""};
        explicit file_service(reactor& x) : r(x) { th = std::thread([this] { main_(); }); }
        ~file_service() { { std::lock_guard<std::mutex> g(m); stop = true; } cv.notify_all(); th.join(); }
        std::string backend_name() const { return name.load(); }
        void submit(const file_req& q2) { { std::lock_guard<std::mutex> g(m); q.push_back(q2); } cv.notify_all(); }
        void main_() {
            async_file_io aio; name.store(aio.backend_name());
            std::size_t inflight = 0;
            for (;;) {
                std::deque<file_req> take;
                {
                    std::unique_lock<std::mutex> lk(m);
                    if (!inflight) cv.wait(lk, [&] { return stop || !q.empty(); });
                    if (stop && q.empty() && !inflight) return;
                    take.swap(q);
                }
                for (auto& rq : take) {
                    auto st = rq.wr ? aio.write(*rq.f, rq.buf, rq.n, rq.off, rq.id) : aio.read(*rq.f, rq.buf, rq.n, rq.off, rq.id);
                    if (st) ++inflight; else { auto e = st.err; auto id = rq.id; r.post([&rr = r, id, e] { rr.file_done_(id, e, 0); }); }
                }
                if (!inflight) continue;
                for (auto& c : aio.wait(1, 2)) {   // 2 ms slices: new requests are picked up in between
                    --inflight; auto id = c.user; auto res = c.result;
                    r.post([&rr = r, id, res] {
                        if (res >= 0) rr.file_done_(id, {}, std::size_t(res));
                        else rr.file_done_(id, error{int(-res), "file I/O: " + error_text(int(-res))}, 0);
                    });
                }
            }
        }
    };
    op_id file_(file& f, void* buf, std::size_t n, std::uint64_t off, bool wr, io_fn cb) {
        op* o = new_op_(kind::file); const std::uint64_t oid = o->id; o->cb = std::move(cb); o->what = wr ? "write_file" : "read_file";
        if (!valid()) return fail_(o, init_err_);
        track_(o);
        if (!fsvc_) fsvc_ = std::make_unique<file_service>(*this);
        fsvc_->submit({&f, buf, n, off, o->id, wr}); return {oid};
    }
    void file_done_(std::uint64_t id, const error& e, std::size_t n) {
        op* o = find_(id); if (!o) return;
        if (o->cancel_requested) finish_(o, cancelled_error(o->what), n); else finish_(o, e, n);
    }

    // ---- readiness-driven operations (POSIX descriptors; Windows sockets)
    op_id io_(kind k, io_handle h, void* buf, std::size_t n, bool all, io_fn cb) {
        op* o = new_op_(k); const std::uint64_t oid = o->id; o->cb = std::move(cb); o->buf = static_cast<char*>(buf); o->len = n; o->all = all; o->key = h.key(); o->is_socket = h.is_socket;
        o->what = k == kind::read ? "read" : "write";
        if (!valid()) return fail_(o, init_err_);
#if defined(FORGE_OS_WINDOWS)
        if (!h.is_socket) { start_hio_(o, h.h); return {oid}; }
        if (h.s == INVALID_SOCKET) return fail_(o, errno_error(EBADF, o->what + ": invalid socket"));
#else
        if (h.fd < 0) return fail_(o, errno_error(EBADF, o->what + ": invalid descriptor"));
#endif
        if (n == 0) return fail_(o, {});
        start_sock_(o, k == kind::read); return {oid};
    }
    op_id wait_io_(kind k, io_handle h, done_fn cb) {
        op* o = new_op_(k); const std::uint64_t oid = o->id; o->dcb = std::move(cb); o->key = h.key(); o->is_socket = h.is_socket; o->what = "wait";
        if (!valid()) return fail_(o, init_err_);
#if defined(FORGE_OS_WINDOWS)
        if (!h.is_socket) return fail_(o, unsupported("reactor: readiness of a non-socket handle"));
        { fd_state& st = fds_[o->key]; std::string e; if (st.rd.empty() && st.wr.empty() && !select_sock_(st, o->key, e)) return fail_(o, error{sock_errno(), "wait: " + e}); }
#else
        if (h.fd < 0) return fail_(o, errno_error(EBADF, "wait: invalid descriptor"));
#endif
        enqueue_(o, k == kind::wait_r); return {oid};
    }
    static void set_nonblocking_(std::uintptr_t key) {
#if defined(FORGE_OS_WINDOWS)
        u_long v = 1; ::ioctlsocket(SOCKET(key), FIONBIO, &v);
#else
        int f = ::fcntl(int(key), F_GETFL); if (f >= 0 && !(f & O_NONBLOCK)) ::fcntl(int(key), F_SETFL, f | O_NONBLOCK);
#endif
    }
    // First attempt right away when nothing is queued ahead of it; otherwise (or when it would block) queue it.
    void start_sock_(op* o, bool rd) {
        track_(o); set_nonblocking_(o->key);
        auto it = fds_.find(o->key);
        bool empty = it == fds_.end() || (rd ? it->second.rd.empty() : it->second.wr.empty());
#if defined(FORGE_OS_WINDOWS)
        // Associate before trying: FD_WRITE is only recorded for a send that failed after the association.
        fd_state& st = fds_[o->key];
        if (st.rd.empty() && st.wr.empty()) { std::string e; if (!select_sock_(st, o->key, e)) { finish_(o, error{sock_errno(), o->what + ": " + e}, 0); return; } }
#endif
        if (empty && try_(o)) return;
        enqueue_(o, rd);
    }
    void enqueue_(op* o, bool rd) {
        track_(o);
        fd_state& st = fds_[o->key];
        (rd ? st.rd : st.wr).push_back(o);
        update_(o->key);
    }
    // Runs the operation once. True: finished (callback queued). False: it would block.
    bool try_(op* o) {
        switch (o->k) {
        case kind::read:
            for (;;) {
                long n = rd_(o, o->buf + o->done, o->len - o->done);
                if (n > 0) { o->done += std::size_t(n); if (!o->all || o->done == o->len) { finish_(o, {}, o->done); return true; } continue; }
                if (n == 0) { std::size_t d = o->done; finish_(o, o->all ? errno_error(EPIPE, "read: end of stream") : error{}, d); return true; }
                int e = sock_errno_(o);
                if (interrupted_(e)) continue;
                if (blocked_(e)) return false;
                std::size_t d = o->done; finish_(o, io_error_("read", e), d); return true;
            }
        case kind::write:
            for (;;) {
                long n = wr_(o, o->buf + o->done, o->len - o->done);
                if (n >= 0) { o->done += std::size_t(n); if (!o->all || o->done == o->len) { finish_(o, {}, o->done); return true; } if (n == 0) return false; continue; }
                int e = sock_errno_(o);
                if (interrupted_(e)) continue;
                if (blocked_(e)) return false;
                std::size_t d = o->done; finish_(o, io_error_("write", e), d); return true;
            }
        case kind::accept:
            for (;;) {
#if defined(FORGE_OS_WINDOWS)
                sock_t c = ::accept(SOCKET(o->key), nullptr, nullptr);
                if (c != bad_sock) { SetHandleInformation(reinterpret_cast<HANDLE>(c), HANDLE_FLAG_INHERIT, 0); ::WSAEventSelect(c, nullptr, 0); }   // an accepted socket inherits the listener's event association: drop it
#else
                sock_t c = cloexec_accept(int(o->key));
#endif
                if (c != bad_sock) {
#if defined(__APPLE__)
                    int one = 1; ::setsockopt(c, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
                    o->csock = c; finish_(o, {}, 0); return true;
                }
                int e = sock_errno();
                if (interrupted_(e)) continue;
#if defined(FORGE_OS_POSIX)
                if (e == ECONNABORTED) continue;
#endif
                if (blocked_(e)) return false;
                finish_(o, io_error_("accept", e), 0); return true;
            }
        default: return false;   // wait_r / wait_w / connect finish in drive_
        }
    }
    static bool interrupted_(int e) {
#if defined(FORGE_OS_WINDOWS)
        return e == WSAEINTR;
#else
        return e == EINTR;
#endif
    }
    static bool blocked_(int e) {
#if defined(FORGE_OS_WINDOWS)
        return e == WSAEWOULDBLOCK;
#else
        return e == EAGAIN || e == EWOULDBLOCK;
#endif
    }
    static error io_error_(const char* what, int e) { return error{e ? e : -1, std::string(what) + ": " + error_text(e)}; }
    static int sock_errno_(op*) { return sock_errno(); }
    static long rd_(op* o, char* p, std::size_t n) {
#if defined(FORGE_OS_WINDOWS)
        return ::recv(SOCKET(o->key), p, int(n < 0x40000000u ? n : 0x40000000u), 0);
#else
        return long(::read(int(o->key), p, n));
#endif
    }
    static long wr_(op* o, const char* p, std::size_t n) {
#if defined(FORGE_OS_WINDOWS)
        return ::send(SOCKET(o->key), p, int(n < 0x40000000u ? n : 0x40000000u), 0);
#else
#if defined(MSG_NOSIGNAL)
        if (!o->notsock) {   // a socket: no SIGPIPE when the peer is gone
            auto r = ::send(int(o->key), p, n, MSG_NOSIGNAL);
            if (r >= 0 || errno != ENOTSOCK) return long(r);
            o->notsock = true;
        }
#endif
        return long(::write(int(o->key), p, n));
#endif
    }
    // The descriptor / socket became readable and / or writable: run what is queued, in order.
    void drive_(std::uintptr_t key, bool r, bool w) {
        if (r) run_queue_(key, true);
        if (w) run_queue_(key, false);
        update_(key);
    }
    void run_queue_(std::uintptr_t key, bool rd) {
        for (;;) {
            auto it = fds_.find(key); if (it == fds_.end()) return;
            auto& q = rd ? it->second.rd : it->second.wr; if (q.empty()) return;
            op* o = q.front();
            if (o->persistent) {
                if (o->inline_tick) { o->tick(); return; }
                std::uint64_t id = o->id; ready_.push_back([this, id] { if (op* p = find_(id)) { auto t = p->tick; if (t) t(); } }); return;
            }
            if (o->k == kind::wait_r || o->k == kind::wait_w) { q.pop_front(); finish_(o, {}, 0); continue; }
            if (o->k == kind::connect) {
                q.pop_front();
                int so = 0; socklen_t l = sizeof so; ::getsockopt(o->csock, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&so), &l);
#if defined(FORGE_OS_WINDOWS)
                if (!so && it->second.conn_err) so = it->second.conn_err;
                it->second.conn_err = 0;
#endif
                if (so == 0) { int one = 1; ::setsockopt(o->csock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof one); finish_(o, {}, 0); continue; }
                o->last_err = so; update_(key); close_csock_(o); connect_next_(o); return;
            }
            if (!try_(o)) return;   // still blocked: keep it at the head
            // finished: try_ freed it; drop the queue entry (the queue may have been touched only by us)
            auto it2 = fds_.find(key); if (it2 == fds_.end()) return;
            auto& q2 = rd ? it2->second.rd : it2->second.wr; if (!q2.empty() && q2.front() == o) q2.pop_front();
        }
    }
    void remove_from_queue_(op* o) {
        auto it = fds_.find(o->key); if (it == fds_.end()) return;
        for (auto* q : {&it->second.rd, &it->second.wr})
            for (auto i = q->begin(); i != q->end(); ++i) if (*i == o) { q->erase(i); break; }
        update_(o->key);
    }
    void close_csock_(op* o) {
        if (o->csock == bad_sock) return;
#if defined(FORGE_OS_WINDOWS)
        auto it = fds_.find(std::uintptr_t(o->csock));
        if (it != fds_.end() && it->second.rd.empty() && it->second.wr.empty()) { release_sock_(it->second, o->csock, false); fds_.erase(it); }
        ::closesocket(o->csock);
#else
        ::close(o->csock);
#endif
        o->csock = bad_sock;
    }
    void connect_next_(op* o) {
        while (o->ai_next) {
            addrinfo* a = o->ai_next; o->ai_next = a->ai_next;
            socket s = socket::make(a->ai_family, SOCK_STREAM); if (!s.valid()) { o->last_err = sock_errno(); continue; }
            o->csock = s.release(); o->key = std::uintptr_t(o->csock); set_nonblocking_(o->key);
#if defined(FORGE_OS_WINDOWS)
            { fd_state& st = fds_[o->key]; std::string e; if (!select_sock_(st, o->key, e)) { o->last_err = sock_errno(); fds_.erase(o->key); ::closesocket(o->csock); o->csock = bad_sock; continue; } }
#endif
            if (::connect(o->csock, a->ai_addr, socklen_t(a->ai_addrlen)) == 0) { int one = 1; ::setsockopt(o->csock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof one); finish_(o, {}, 0); return; }
            int e = sock_errno();
            if (would_block(e) || interrupted_(e)) { fd_state& st = fds_[o->key]; st.wr.push_back(o); update_(o->key); return; }
            o->last_err = e; close_csock_(o);
        }
        int e = o->last_err;
        finish_(o, error{e ? e : -1, o->what + ": " + (e ? error_text(e) : std::string("no address"))}, 0);
    }

    // ---- cancellation
    void cancel_(op* o) {
        switch (o->k) {
        case kind::timer: timers_.erase({o->deadline, o->id}); finish_(o, cancelled_error(o->what), 0); break;
        case kind::read: case kind::write: case kind::accept: case kind::wait_r: case kind::wait_w: {
            std::size_t d = o->done; remove_from_queue_(o); finish_(o, cancelled_error(o->what), d); break; }
        case kind::connect:
            if (o->csock != bad_sock) { remove_from_queue_(o); close_csock_(o); finish_(o, cancelled_error(o->what), 0); }
            else o->cancel_requested = true;   // still resolving
            break;
        case kind::signal: finish_(o, {}, 0); break;   // no completion callback; the listener stays installed
        case kind::exit:
#if defined(FORGE_OS_KQUEUE)
            if (!o->skip) { struct kevent k; EV_SET(&k, o->key, EVFILT_PROC, EV_DELETE, 0, 0, nullptr); ::kevent(q_, &k, 1, nullptr, 0, nullptr); procs_.erase(o->id); finish_(o, cancelled_error(o->what), 0); break; }
#elif defined(FORGE_OS_EPOLL)
            if (!o->skip) { remove_from_queue_(o); finish_(o, cancelled_error(o->what), 0); break; }
#else
            if (!o->skip) { finish_(o, cancelled_error(o->what), 0); break; }
#endif
            o->cancel_requested = true; o->skip->store(true); break;
        case kind::offload: o->cancel_requested = true; if (o->skip) o->skip->store(true); break;
        case kind::file: o->cancel_requested = true; break;
        case kind::hwait: finish_(o, cancelled_error(o->what), 0); break;
        case kind::hio:
#if defined(FORGE_OS_WINDOWS)
            o->cancel_requested = true; o->hio->cancel.store(true);
            if (o->hio->phase.load() >= 1) CancelIoEx(o->hio->h, &o->hio->ov);
            cancelling_.push_back(o->id);
#endif
            break;
        }
    }

    // ---- child exit
    void start_exit_(op* o, std::uint64_t pid) {
#if defined(FORGE_OS_WINDOWS)
        if (pid == 0 || pid > 0xFFFFFFFFull) { fail_(o, error{ERROR_INVALID_PARAMETER, "on_exit: invalid pid"}); return; }
        HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, DWORD(pid));
        if (!h) { if (GetLastError() == ERROR_INVALID_PARAMETER) fail_(o, {}); else fail_(o, last_error("OpenProcess")); return; }
        o->k = kind::hwait; o->own_handle = h; track_(o);
        if (!arm_wait_(o, h, true)) finish_(o, last_error("RegisterWaitForSingleObject"), 0);
#elif defined(FORGE_OS_IOS)
        (void)pid; fail_(o, unsupported("on_exit"));
#else
        if (pid == 0 || pid > std::uint64_t(std::numeric_limits<pid_t>::max())) { fail_(o, errno_error(EINVAL, "on_exit: invalid pid")); return; }
#if defined(FORGE_OS_KQUEUE)
        struct kevent k; EV_SET(&k, pid, EVFILT_PROC, EV_ADD | EV_ONESHOT, NOTE_EXIT, 0, nullptr);
        if (::kevent(q_, &k, 1, nullptr, 0, nullptr) == 0) { o->key = std::uintptr_t(pid); track_(o); procs_[o->id] = pid; return; }
        if (errno == ESRCH) { fail_(o, {}); return; }
#else
        bool try_pidfd = true;
#if defined(FORGE_OS_ANDROID)
        try_pidfd = android_api_level() >= 31;   // older seccomp policies kill on unknown syscalls
#endif
#if defined(SYS_pidfd_open)
        if (try_pidfd) {
            int fd = int(::syscall(SYS_pidfd_open, pid_t(pid), 0u));
            if (fd >= 0) { ::fcntl(fd, F_SETFD, FD_CLOEXEC); o->k = kind::wait_r; o->own_fd = true; o->key = std::uintptr_t(unsigned(fd)); enqueue_(o, true); return; }
            if (errno == ESRCH) { fail_(o, {}); return; }
        }
#endif
        (void)try_pidfd;
#endif
        // No kernel event: a helper thread polls (100 ms slices, so cancel and shutdown stay prompt).
        track_(o); std::uint64_t id = o->id; auto skip = std::make_shared<std::atomic<bool>>(false); o->skip = skip;
        auto err = std::make_shared<error>();
        offload_([pid, skip, err] { while (!skip->load()) { auto r = wait_for_exit(pid, 100); if (!r) { *err = r.err; return; } if (r.value) return; } },
                 [this, id, err] { if (op* p = find_(id)) finish_(p, p->cancel_requested ? cancelled_error(p->what) : *err, 0); });
#endif
    }
    void drain_signals_() {
        auto& l = signal_listener::instance();
        while (auto s = l.wait(0)) {
            std::vector<std::uint64_t> ids = sig_ops_;
            for (auto id : ids) if (op* p = find_(id)) if (p->sig_which == *s) { auto cb = p->sig_cb; cb(*s); }
        }
#if defined(FORGE_OS_WINDOWS)
        if (op* w = find_(sig_watch_)) { if (w->reg) { unregister_(w->reg); w->reg = nullptr; } arm_wait_(w, w->rearm, false); }
#endif
    }

    // ---------------------------------------------------------------- backends
#if defined(FORGE_OS_POSIX)
    // Interest follows the queues: add / modify / delete as they fill and drain.
    void update_(std::uintptr_t key) {
        auto it = fds_.find(key); if (it == fds_.end()) return;
        fd_state& st = it->second; int fd = int(key);
        unsigned want = (st.rd.empty() ? 0u : 1u) | (st.wr.empty() ? 0u : 2u);
        if (want == st.mask) { if (!want) fds_.erase(it); return; }
        int err = 0;
#if defined(FORGE_OS_EPOLL)
        epoll_event e{}; e.events = ((want & 1) ? EPOLLIN : 0) | ((want & 2) ? EPOLLOUT : 0); e.data.fd = fd;
        if (::epoll_ctl(q_, !st.mask ? EPOLL_CTL_ADD : !want ? EPOLL_CTL_DEL : EPOLL_CTL_MOD, fd, &e) != 0 && want) err = errno;
#else
        struct kevent k[2]; int n = 0;
        if ((want ^ st.mask) & 1) EV_SET(&k[n++], fd, EVFILT_READ, (want & 1) ? EV_ADD : EV_DELETE, 0, 0, nullptr);
        if ((want ^ st.mask) & 2) EV_SET(&k[n++], fd, EVFILT_WRITE, (want & 2) ? EV_ADD : EV_DELETE, 0, 0, nullptr);
        for (int i = 0; i < n; ++i) if (::kevent(q_, &k[i], 1, nullptr, 0, nullptr) != 0 && (k[i].flags & EV_ADD)) err = errno;
#endif
        if (err) {   // e.g. a regular file (EPERM) or a closed descriptor: fail what is queued
            std::deque<op*> rd, wr; rd.swap(st.rd); wr.swap(st.wr);
#if defined(FORGE_OS_EPOLL)
            if (st.mask) ::epoll_ctl(q_, EPOLL_CTL_DEL, fd, nullptr);
#endif
            fds_.erase(it);
            for (auto* q : {&rd, &wr}) for (op* o : *q) { if (o->k == kind::connect) close_csock_(o); finish_(o, io_error_("reactor: cannot wait on this descriptor", err), o->done); }
            return;
        }
        st.mask = want; if (!want) fds_.erase(it);
    }
    void poll_(int ms) {
#if defined(FORGE_OS_EPOLL)
        epoll_event ev[64]; int n = ::epoll_wait(q_, ev, 64, ms);
        for (int i = 0; i < n; ++i) {
            bool bad = ev[i].events & (EPOLLHUP | EPOLLERR);
            drive_(std::uintptr_t(unsigned(ev[i].data.fd)), (ev[i].events & EPOLLIN) || bad, (ev[i].events & EPOLLOUT) || bad);
        }
#else
        struct kevent ev[64]; timespec ts{ms / 1000, (ms % 1000) * 1000000L};
        int n = ::kevent(q_, nullptr, 0, ev, 64, ms < 0 ? nullptr : &ts);
        for (int i = 0; i < n; ++i) {
            if (ev[i].filter == EVFILT_PROC) {
                for (auto it = procs_.begin(); it != procs_.end();) {
                    if (it->second == std::uint64_t(ev[i].ident)) { op* o = find_(it->first); it = procs_.erase(it); if (o) finish_(o, {}, 0); } else ++it;
                }
                continue;
            }
            if (ev[i].flags & EV_ERROR) continue;
            drive_(std::uintptr_t(unsigned(int(ev[i].ident))), ev[i].filter == EVFILT_READ, ev[i].filter == EVFILT_WRITE);
        }
#endif
    }
    void shutdown_() {
        stop_threads_();
        std::vector<op*> all; for (auto& [id, o] : ops_) all.push_back(o);
        ops_.clear(); fds_.clear(); timers_.clear(); ready_.clear();
        for (op* o : all) { o->scb = nullptr; release_(o); delete o; }
        if (q_ >= 0) ::close(q_);
        q_ = -1;
    }
#else
    // ---- Windows
    static constexpr ULONG_PTR key_wake = 1, key_wait = 2;
    static VOID CALLBACK wait_cb_(PVOID ctx, BOOLEAN) {
        auto* w = static_cast<wait_reg*>(ctx);
        PostQueuedCompletionStatus(w->port, 0, key_wait, reinterpret_cast<LPOVERLAPPED>(static_cast<std::uintptr_t>(w->id)));
    }
    wait_reg* register_(HANDLE h, std::uint64_t id, bool once) {
        auto* w = new wait_reg{port_, id, nullptr};
        if (!RegisterWaitForSingleObject(&w->wait, h, &reactor::wait_cb_, w, INFINITE, once ? WT_EXECUTEONLYONCE : WT_EXECUTEDEFAULT)) { delete w; return nullptr; }
        return w;
    }
    static void unregister_(wait_reg* w) { UnregisterWaitEx(w->wait, INVALID_HANDLE_VALUE); delete w; }   // waits for a running callback
    // One-shot wait on a handle for operation o (process, event). `tracked`: o is already in ops_.
    bool arm_wait_(op* o, HANDLE h, bool tracked) {
        if (!tracked) track_(o);
        if (o->wid) waits_.erase(o->wid);
        std::uint64_t wid = ++next_id_; waits_[wid] = {o->id, 0, false}; o->wid = wid;
        o->reg = register_(h, wid, true);
        if (!o->reg) { waits_.erase(wid); o->wid = 0; if (!tracked) { if (ops_.erase(o->id) && !o->internal) --work_; } return false; }
        return true;
    }
    bool select_sock_(fd_state& st, std::uintptr_t key, std::string& what) {
        if (!st.ev) {
            st.ev = CreateEventW(nullptr, FALSE, FALSE, nullptr); if (!st.ev) { what = "CreateEvent failed"; return false; }
            st.wid = ++next_id_; waits_[st.wid] = {0, key, true};
            st.reg = register_(st.ev, st.wid, false);
            if (!st.reg) { waits_.erase(st.wid); CloseHandle(st.ev); st.ev = nullptr; what = "RegisterWaitForSingleObject failed"; return false; }
        }
        if (::WSAEventSelect(SOCKET(key), st.ev, FD_READ | FD_WRITE | FD_ACCEPT | FD_CONNECT | FD_CLOSE) != 0) { what = error_text(sock_errno()); return false; }
        return true;
    }
    void release_sock_(fd_state& st, sock_t s, bool deselect) {
        if (deselect) ::WSAEventSelect(s, nullptr, 0);
        if (st.reg) { unregister_(st.reg); st.reg = nullptr; }
        if (st.wid) waits_.erase(st.wid);
        if (st.ev) { CloseHandle(st.ev); st.ev = nullptr; }
    }
    void update_(std::uintptr_t) {}   // the association stays while the socket is known (see forget())
    void poll_(int ms) {
        OVERLAPPED_ENTRY ent[64]; ULONG n = 0;
        if (!GetQueuedCompletionStatusEx(port_, ent, 64, &n, ms < 0 ? INFINITE : DWORD(ms), FALSE)) n = 0;
        for (ULONG i = 0; i < n; ++i) {
            if (ent[i].lpCompletionKey != key_wait) continue;   // wake-ups carry nothing
            std::uint64_t wid = std::uint64_t(reinterpret_cast<std::uintptr_t>(ent[i].lpOverlapped));
            auto it = waits_.find(wid); if (it == waits_.end()) continue;   // unregistered meanwhile
            if (it->second.is_sock) {
                std::uintptr_t key = it->second.key; auto f = fds_.find(key); if (f == fds_.end()) continue;
                WSANETWORKEVENTS ne{};
                if (::WSAEnumNetworkEvents(SOCKET(key), f->second.ev, &ne) != 0) { drive_(key, true, true); continue; }   // closed socket: let the calls fail
                if ((ne.lNetworkEvents & FD_CONNECT) && ne.iErrorCode[FD_CONNECT_BIT]) f->second.conn_err = ne.iErrorCode[FD_CONNECT_BIT];
                drive_(key, (ne.lNetworkEvents & (FD_READ | FD_ACCEPT | FD_CLOSE)) != 0, (ne.lNetworkEvents & (FD_WRITE | FD_CONNECT | FD_CLOSE)) != 0);
            } else {
                op* o = find_(it->second.op); waits_.erase(it); if (!o) continue;
                o->wid = 0;
                if (o->k == kind::hio) { hio_pending_done_(o); continue; }
                if (o->persistent) { std::uint64_t id = o->id; ready_.push_back([this, id] { if (op* p = find_(id)) { auto t = p->tick; if (t) t(); } }); continue; }
                finish_(o, {}, 0);
            }
        }
        // Calls that were cancelled before they reached the kernel: ask again until they have finished.
        for (std::size_t i = 0; i < cancelling_.size();) {
            op* o = find_(cancelling_[i]);
            if (!o) { cancelling_.erase(cancelling_.begin() + std::ptrdiff_t(i)); continue; }
            if (o->hio->phase.load() >= 1) CancelIoEx(o->hio->h, &o->hio->ov);
            ++i;
        }
    }
    // Handle I/O: the call starts on a helper thread (a synchronous handle would block the loop). Overlapped
    // handles return ERROR_IO_PENDING at once; the completion is then awaited through a thread-pool wait.
    void start_hio_(op* o, HANDLE h) {
        if (h == INVALID_HANDLE_VALUE || !h) { fail_(o, errno_error(EBADF, o->what + ": invalid handle")); return; }
        if (o->len == 0) { fail_(o, {}); return; }
        o->hio_wr = o->k == kind::write; o->hio_h = h; o->k = kind::hio; track_(o);
        launch_hio_(o);
    }
    void launch_hio_(op* o) {
        auto st = std::make_shared<hio_state>(); o->hio = st; std::uint64_t id = o->id;
        std::size_t left = o->len - o->done;
        st->h = o->hio_h; st->buf = o->buf + o->done; st->len = DWORD(left < 0x40000000u ? left : 0x40000000u); st->wr = o->hio_wr;
        st->ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!st->ov.hEvent) { finish_(o, last_error("CreateEvent"), o->done); return; }
        auto active = hio_active_;
        offload_([st, active] {
            if (st->cancel.load()) { st->err = ERROR_OPERATION_ABORTED; st->phase.store(3); return; }
            active->fetch_add(1); st->phase.store(1);
            BOOL ok = st->wr ? WriteFile(st->h, st->buf, st->len, nullptr, &st->ov) : ReadFile(st->h, st->buf, st->len, nullptr, &st->ov);
            DWORD e = ok ? 0 : GetLastError();
            if (!ok && e == ERROR_IO_PENDING) { st->phase.store(2); active->fetch_sub(1); if (st->cancel.load()) CancelIoEx(st->h, &st->ov); return; }
            DWORD got = 0; if (ok && !GetOverlappedResult(st->h, &st->ov, &got, TRUE)) e = GetLastError();
            st->got = got; st->err = e; st->phase.store(3); active->fetch_sub(1);
        }, [this, id, st] {
            op* p = find_(id); if (!p) return;
            if (st->phase.load() == 3) { hio_step_(p, st->got, st->err); return; }
            if (!arm_wait_(p, st->ov.hEvent, true)) {   // cannot wait in the pool: block for it (rare)
                DWORD got = 0, e = GetOverlappedResult(st->h, &st->ov, &got, TRUE) ? 0 : GetLastError(); hio_step_(p, got, e);
            }
        });
    }
    void hio_pending_done_(op* o) {
        if (o->reg) { unregister_(o->reg); o->reg = nullptr; }
        DWORD got = 0, e = GetOverlappedResult(o->hio->h, &o->hio->ov, &got, FALSE) ? 0 : GetLastError();
        hio_step_(o, got, e);
    }
    // One ReadFile / WriteFile finished: complete, or continue a write-all / read-exact.
    void hio_step_(op* o, DWORD got, DWORD e) {
        o->done += got; const bool wr = o->hio_wr;
        if (o->cancel_requested) { finish_(o, cancelled_error(o->what), o->done); return; }
        const bool eof = !wr && (e == ERROR_BROKEN_PIPE || e == ERROR_HANDLE_EOF || e == ERROR_PIPE_NOT_CONNECTED || (!e && got == 0));
        if (eof) { finish_(o, o->all && o->done < o->len ? errno_error(EPIPE, "read: end of stream") : error{}, o->done); return; }
        if (e) { finish_(o, error{int(e), o->what + ": " + error_text(int(e))}, o->done); return; }
        if (o->all && o->done < o->len) { launch_hio_(o); return; }
        finish_(o, {}, o->done);
    }
    void shutdown_() {
        // In-flight handle calls write into caller buffers and our OVERLAPPEDs: cancel them and wait.
        signal_stop_();
        auto cancel_hio = [&] { for (auto& [id, o] : ops_) if (o->k == kind::hio && o->hio) { o->hio->cancel.store(true); if (o->hio->phase.load() >= 1) CancelIoEx(o->hio->h, &o->hio->ov); } };
        cancel_hio();
        while (hio_active_->load() > 0) { cancel_hio(); Sleep(1); }   // a synchronous call that had not reached the kernel yet
        join_threads_();
        for (auto& [id, o] : ops_) if (o->k == kind::hio && o->hio && o->hio->phase.load() == 2) {
            CancelIoEx(o->hio->h, &o->hio->ov); DWORD got = 0; GetOverlappedResult(o->hio->h, &o->hio->ov, &got, TRUE);
        }
        std::vector<op*> all; for (auto& [id, o] : ops_) all.push_back(o);
        ops_.clear(); timers_.clear(); ready_.clear();
        for (op* o : all) { o->scb = nullptr; release_(o); delete o; }
        // Sockets keep their (now dead) event association: the values may have been closed and reused by now, so
        // they are not touched. forget() is the way to detach a live socket.
        for (auto& [key, st] : fds_) release_sock_(st, SOCKET(key), false);
        fds_.clear(); waits_.clear();
        if (port_) CloseHandle(port_);
        port_ = nullptr;
    }
#endif
    void signal_stop_() {
        for (auto& [id, o] : ops_) if (o->skip) o->skip->store(true);   // polling helpers stop at their next slice
        { std::lock_guard<std::mutex> g(hm_); hstop_ = true; }
        hcv_.notify_all();
    }
    void join_threads_() {
        fsvc_.reset();
        for (auto& t : helpers_) t.join();
        helpers_.clear();
        { std::lock_guard<std::mutex> g(hm_); hq_.clear(); }
        std::lock_guard<std::mutex> g(pm_); posted_.clear();
    }
    void stop_threads_() { signal_stop_(); join_threads_(); }

    error init_err_;
#if defined(FORGE_OS_WINDOWS)
    struct wait_target { std::uint64_t op = 0; std::uintptr_t key = 0; bool is_sock = false; };
    HANDLE port_ = nullptr; std::unordered_map<std::uint64_t, wait_target> waits_; std::vector<std::uint64_t> cancelling_;
    std::shared_ptr<std::atomic<int>> hio_active_ = std::make_shared<std::atomic<int>>(0);
#else
    int q_ = -1; wait_event wake_;
#if defined(FORGE_OS_KQUEUE)
    std::map<std::uint64_t, std::uint64_t> procs_;   // op id -> pid
#endif
#endif
    std::uint64_t next_id_ = 0; std::size_t work_ = 0;
    std::unordered_map<std::uint64_t, op*> ops_;
    std::unordered_map<std::uintptr_t, fd_state> fds_;
    std::set<std::pair<std::uint64_t, std::uint64_t>> timers_;   // (deadline, id)
    std::deque<std::function<void()>> ready_;
    std::vector<std::uint64_t> sig_ops_; std::uint64_t sig_watch_ = 0;
    std::mutex pm_; std::deque<std::function<void()>> posted_;
    std::atomic<bool> stopped_{false}, woken_{false}; std::atomic<int> holds_{0};
    std::mutex hm_; std::condition_variable hcv_; std::deque<std::function<void()>> hq_; std::vector<std::thread> helpers_; unsigned hidle_ = 0; bool hstop_ = false;
    unsigned max_helpers_ = 16;
    std::unique_ptr<file_service> fsvc_;
};

} // namespace forge::os

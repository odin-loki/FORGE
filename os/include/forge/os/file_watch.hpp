#pragma once
// File watching: inotify (Linux/Android), kqueue EVFILT_VNODE (macOS/iOS/FreeBSD), ReadDirectoryChangesW (Windows).
// Watching a file watches its directory and filters by name, so atomic replace-by-rename (editors, os::rename,
// forge pack writers) is seen as a modification of that file.  FSEvents is not used on macOS: it needs a
// CFRunLoop/dispatch queue per stream and is coalesced/latent; kqueue gives immediate per-file events.
#include <forge/os/detail.hpp>
#if defined(__linux__)
#include <sys/inotify.h>
#endif

namespace forge::os {
enum class watch_kind { created, modified, removed, renamed, other };
inline const char* watch_kind_name(watch_kind k) {
    switch (k) { case watch_kind::created: return "created"; case watch_kind::modified: return "modified"; case watch_kind::removed: return "removed"; case watch_kind::renamed: return "renamed"; default: return "other"; }
}
struct watch_event { std::string path; watch_kind kind = watch_kind::other; };

class file_watcher {
public:
    file_watcher() {
#if defined(__linux__)
        fd_ = ::inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
#elif defined(FORGE_OS_KQUEUE)
        fd_ = ::kqueue();
#endif
    }
    ~file_watcher() {
#if defined(FORGE_OS_WINDOWS)
        for (auto& d : dirs_) { CancelIo(d->h); CloseHandle(d->h); CloseHandle(d->ov.hEvent); }
#elif defined(FORGE_OS_KQUEUE)
        for (auto& w : kw_) ::close(w.fd);
        if (fd_ >= 0) ::close(fd_);
#else
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    file_watcher(const file_watcher&) = delete; file_watcher& operator=(const file_watcher&) = delete;
    const char* backend() const {
#if defined(__linux__)
        return "inotify";
#elif defined(FORGE_OS_KQUEUE)
        return "kqueue";
#elif defined(FORGE_OS_WINDOWS)
        return "ReadDirectoryChangesW";
#else
        return "none";
#endif
    }
    status add(const std::string& path) {
        auto st = stat(path); bool is_dir = st.is_dir;
        if (!st.exists && !stat(dirname(path)).is_dir) return {{}, {2, "watch: no such path " + path}};
        std::string dir = is_dir ? path : dirname(path), name = is_dir ? "" : basename(path);
#if defined(__linux__)
        if (fd_ < 0) return {{}, last_error("inotify_init1")};
        int wd = ::inotify_add_watch(fd_, dir.c_str(), IN_MODIFY | IN_CLOSE_WRITE | IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO | IN_ATTRIB | IN_DELETE_SELF | IN_MOVE_SELF);
        if (wd < 0) return {{}, last_error("inotify_add_watch " + dir)};
        targets_.push_back({wd, dir, name}); return {};
#elif defined(FORGE_OS_KQUEUE)
        if (fd_ < 0) return {{}, last_error("kqueue")};
        if (auto s = kq_add_(dir, true, dir, name); !s) return s;
        if (!is_dir && st.exists) kq_add_(path, false, dir, name);
        return {};
#elif defined(FORGE_OS_WINDOWS)
        for (auto& d : dirs_) if (d->dir == dir) { d->names.push_back(name); return {}; }
        auto d = std::make_unique<wdir>(); d->dir = dir; d->names.push_back(name);
        d->h = CreateFileW(widen(dir).c_str(), FILE_LIST_DIRECTORY, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, nullptr);
        if (d->h == INVALID_HANDLE_VALUE) return {{}, last_error("open dir " + dir)};
        d->ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!arm_(*d)) { auto e = last_error("ReadDirectoryChangesW"); CloseHandle(d->h); CloseHandle(d->ov.hEvent); return {{}, e}; }
        dirs_.push_back(std::move(d)); return {};
#else
        return {{}, unsupported("file watching")};
#endif
    }
    // Wait up to timeout_ms (-1 forever, 0 poll) and return whatever arrived.
    std::vector<watch_event> poll(int timeout_ms) {
        std::vector<watch_event> out;
#if defined(__linux__)
        pollfd p{fd_, POLLIN, 0}; if (::poll(&p, 1, timeout_ms) <= 0) return out;
        alignas(inotify_event) char buf[16384];
        for (;;) {
            auto n = ::read(fd_, buf, sizeof buf); if (n <= 0) break;
            for (char* q = buf; q < buf + n;) {
                auto* e = reinterpret_cast<inotify_event*>(q); q += sizeof(inotify_event) + e->len;
                std::string nm = e->len ? std::string(e->name) : "";
                for (auto& t : targets_) {
                    if (t.wd != e->wd || (!t.name.empty() && t.name != nm)) continue;
                    watch_kind k = e->mask & (IN_CREATE | IN_MOVED_TO) ? (t.name.empty() ? watch_kind::created : watch_kind::modified)
                                 : e->mask & (IN_DELETE | IN_DELETE_SELF) ? watch_kind::removed
                                 : e->mask & (IN_MOVED_FROM | IN_MOVE_SELF) ? watch_kind::renamed
                                 : e->mask & (IN_MODIFY | IN_CLOSE_WRITE | IN_ATTRIB) ? watch_kind::modified : watch_kind::other;
                    if (e->mask & IN_IGNORED) continue;
                    out.push_back({nm.empty() ? t.dir : join(t.dir, nm), k});
                }
            }
        }
#elif defined(FORGE_OS_KQUEUE)
        struct kevent ev[32]; timespec ts{timeout_ms / 1000, (timeout_ms % 1000) * 1000000L};
        int n = ::kevent(fd_, nullptr, 0, ev, 32, timeout_ms < 0 ? nullptr : &ts);
        for (int i = 0; i < n; ++i) {
            auto idx = std::size_t(reinterpret_cast<std::uintptr_t>(ev[i].udata)); if (idx >= kw_.size()) continue;
            auto w = kw_[idx];   // copy: kq_add_ may grow kw_
            if (w.is_dir) {
                if (w.name.empty()) { out.push_back({w.dir, watch_kind::modified}); continue; }
                // directory changed: did our file appear / get replaced?
                std::string p = join(w.dir, w.name); auto st = stat(p);
                bool tracked = false; for (auto& x : kw_) if (!x.is_dir && x.path == p && x.fd >= 0) tracked = true;
                if (st.exists && (!tracked || st.mtime_ns != w.mtime)) {
                    for (auto& x : kw_) if (!x.is_dir && x.path == p && x.fd >= 0) { ::close(x.fd); x.fd = -1; }
                    kq_add_(p, false, w.dir, w.name); kw_[idx].mtime = st.mtime_ns; out.push_back({p, tracked ? watch_kind::modified : watch_kind::created});
                } else if (!st.exists && tracked) {
                    for (auto& x : kw_) if (!x.is_dir && x.path == p && x.fd >= 0) { ::close(x.fd); x.fd = -1; }
                    out.push_back({p, watch_kind::removed});
                }
            } else {
                auto f = ev[i].fflags;
                out.push_back({w.path, f & NOTE_DELETE ? watch_kind::removed : f & NOTE_RENAME ? watch_kind::renamed : watch_kind::modified});
                if (f & (NOTE_DELETE | NOTE_RENAME)) { ::close(kw_[idx].fd); kw_[idx].fd = -1; }
            }
        }
#elif defined(FORGE_OS_WINDOWS)
        std::vector<HANDLE> hs; for (auto& d : dirs_) hs.push_back(d->ov.hEvent);
        if (hs.empty()) return out;
        DWORD r = WaitForMultipleObjects(DWORD(hs.size()), hs.data(), FALSE, timeout_ms < 0 ? INFINITE : DWORD(timeout_ms));
        if (r == WAIT_TIMEOUT || r == WAIT_FAILED) return out;
        for (auto& d : dirs_) {
            DWORD got = 0; if (!GetOverlappedResult(d->h, &d->ov, &got, FALSE)) continue;
            ResetEvent(d->ov.hEvent);
            for (DWORD off = 0; got;) {
                auto* fi = reinterpret_cast<FILE_NOTIFY_INFORMATION*>(d->buf + off);
                std::string nm = narrow({fi->FileName, fi->FileNameLength / sizeof(wchar_t)});
                for (auto& want : d->names) if (want.empty() || want == nm) {
                    watch_kind k = fi->Action == FILE_ACTION_ADDED ? (want.empty() ? watch_kind::created : watch_kind::modified) : fi->Action == FILE_ACTION_REMOVED ? watch_kind::removed
                                 : fi->Action == FILE_ACTION_RENAMED_OLD_NAME ? watch_kind::renamed : fi->Action == FILE_ACTION_RENAMED_NEW_NAME ? (want.empty() ? watch_kind::created : watch_kind::modified) : watch_kind::modified;
                    out.push_back({join(d->dir, nm), k}); break;
                }
                if (!fi->NextEntryOffset) break; off += fi->NextEntryOffset;
            }
            arm_(*d);
        }
#else
        (void)timeout_ms;
#endif
        return out;
    }
private:
#if defined(__linux__)
    struct target { int wd; std::string dir, name; };
    std::vector<target> targets_; int fd_ = -1;
#elif defined(FORGE_OS_KQUEUE)
    struct kwatch { int fd = -1; bool is_dir = false; std::string path, dir, name; std::int64_t mtime = 0; };
    status kq_add_(const std::string& p, bool is_dir, const std::string& dir, const std::string& name) {
#if defined(__APPLE__)
        int f = ::open(p.c_str(), O_EVTONLY | O_CLOEXEC);
#else
        int f = ::open(p.c_str(), O_RDONLY | O_CLOEXEC);
#endif
        if (f < 0) return {{}, last_error("open " + p)};
        kw_.push_back({f, is_dir, p, dir, name, is_dir && !name.empty() ? stat(join(dir, name)).mtime_ns : 0});
        struct kevent ev; EV_SET(&ev, f, EVFILT_VNODE, EV_ADD | EV_CLEAR, NOTE_WRITE | NOTE_DELETE | NOTE_RENAME | NOTE_EXTEND | NOTE_ATTRIB, 0, reinterpret_cast<void*>(std::uintptr_t(kw_.size() - 1)));
        if (::kevent(fd_, &ev, 1, nullptr, 0, nullptr)) return {{}, last_error("kevent")};
        return {};
    }
    std::vector<kwatch> kw_; int fd_ = -1;
#elif defined(FORGE_OS_WINDOWS)
    struct wdir { std::string dir; std::vector<std::string> names; HANDLE h = INVALID_HANDLE_VALUE; OVERLAPPED ov{}; alignas(DWORD) std::uint8_t buf[32768]; };
    static bool arm_(wdir& d) {
        return ReadDirectoryChangesW(d.h, d.buf, sizeof d.buf, FALSE, FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_DIR_NAME | FILE_NOTIFY_CHANGE_LAST_WRITE | FILE_NOTIFY_CHANGE_SIZE,
                                     nullptr, &d.ov, nullptr) != 0;
    }
    std::vector<std::unique_ptr<wdir>> dirs_;
#else
    int fd_ = -1;
#endif
};
} // namespace forge::os

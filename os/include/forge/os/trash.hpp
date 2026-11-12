#pragma once
// forge::os::trash(path): move a file or directory to the user's trash / recycle bin instead of deleting it.
//   Linux/FreeBSD: the freedesktop.org Trash specification 1.0, implemented here (no helper program): the home
//                  trash $XDG_DATA_HOME/Trash for files on the same filesystem, else $topdir/.Trash/$uid (when an
//                  administrator created a sticky $topdir/.Trash) or $topdir/.Trash-$uid on the file's own volume;
//                  a .trashinfo record (original path, deletion date) is written first, exclusively, so desktop
//                  file managers can list and restore the item.
//   macOS:         NSFileManager trashItemAtURL:resultingItemURL:error: through the Objective-C runtime.
//   Windows:       SHFileOperationW(FO_DELETE, FOF_ALLOWUNDO) without any UI, after SHQueryRecycleBinW confirms
//                  the drive has a recycle bin (network shares have none: unsupported rather than a silent
//                  permanent delete).
//   Android/iOS:   unsupported (no user-visible trash for app files).
// Returns the item's new location where the OS reports it ("" on Windows).
#include <forge/os/detail.hpp>
#if defined(FORGE_OS_POSIX)
#include <ctime>
#endif
#if defined(FORGE_OS_WINDOWS)
#include <shellapi.h>   // types only: shell32.dll is loaded at run time
#endif

namespace forge::os {
namespace detail_trash {
// Path= value of a .trashinfo file: RFC 2396 escaping, '/' kept.
inline std::string escape_path(const std::string& p) {
    std::string o; char b[4];
    for (unsigned char c : p) {
        bool keep = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '/' || c == '-' || c == '_' || c == '.' || c == '~';
        if (keep) o += char(c); else { std::snprintf(b, sizeof b, "%%%02X", unsigned(c)); o += b; }
    }
    return o;
}
// date: "YYYY-MM-DDThh:mm:ss" local time.
inline std::string trashinfo(const std::string& original_path, const std::string& date) {
    return "[Trash Info]\nPath=" + escape_path(original_path) + "\nDeletionDate=" + date + "\n";
}
inline std::string home_trash_dir() {
    std::string d = getenv("XDG_DATA_HOME");
    if (d.empty() || d[0] != '/') d = home_dir() + "/.local/share";
    return d + "/Trash";
}
#if defined(FORGE_OS_POSIX) && !defined(__APPLE__) && !defined(FORGE_OS_ANDROID)
inline bool mkdirs_private(const std::string& p) {
    if (p.empty()) return false;
    struct ::stat st{}; if (::stat(p.c_str(), &st) == 0) return S_ISDIR(st.st_mode);
    auto sl = p.rfind('/'); if (sl != std::string::npos && sl > 0 && !mkdirs_private(p.substr(0, sl))) return false;
    return ::mkdir(p.c_str(), 0700) == 0 || errno == EEXIST;
}
// The mount point of the filesystem holding `dir` (walk up while the device id stays the same).
inline std::string top_dir(std::string dir, dev_t dev) {
    for (;;) {
        auto sl = dir.rfind('/'); std::string parent = sl == 0 || sl == std::string::npos ? "/" : dir.substr(0, sl);
        struct ::stat st{}; if (dir == "/" || ::lstat(parent.c_str(), &st) != 0 || st.st_dev != dev) return dir;
        dir = parent;
    }
}
inline result<std::string> trash_xdg(const std::string& abs) {
    struct ::stat fst{}; if (::lstat(abs.c_str(), &fst) != 0) return {{}, last_error("trash " + abs)};
    auto sl = abs.rfind('/'); std::string parent = sl == 0 ? "/" : abs.substr(0, sl), name = abs.substr(sl + 1);
    if (name.empty() || name == "." || name == "..") return {{}, errno_error(EINVAL, "trash: not a file name: " + abs)};
    struct ::stat pst{}; if (::stat(parent.c_str(), &pst) != 0) return {{}, last_error("trash " + parent)};
    std::string tdir = home_trash_dir(), recorded = abs;
    struct ::stat tst{};
    bool home_ok = mkdirs_private(tdir + "/files") && mkdirs_private(tdir + "/info") && ::stat(tdir.c_str(), &tst) == 0;
    if (!home_ok || tst.st_dev != pst.st_dev) {   // another filesystem: its own trash directory, paths relative to the top
        std::string top = top_dir(parent, pst.st_dev), uid = std::to_string(::getuid()), shared = (top == "/" ? "" : top) + "/.Trash";
        struct ::stat sst{}; std::string cand;
        if (::lstat(shared.c_str(), &sst) == 0 && S_ISDIR(sst.st_mode) && (sst.st_mode & S_ISVTX)) cand = shared + "/" + uid;
        if (cand.empty() || !(mkdirs_private(cand + "/files") && mkdirs_private(cand + "/info"))) cand = (top == "/" ? "" : top) + "/.Trash-" + uid;
        if (!(mkdirs_private(cand + "/files") && mkdirs_private(cand + "/info")))
            return {{}, errno_error(EXDEV, "trash: no trash directory can be created on the volume of " + abs)};
        tdir = cand;
        if (top != "/" && abs.compare(0, top.size() + 1, top + "/") == 0) recorded = abs.substr(top.size() + 1);
    }
    std::time_t now = std::time(nullptr); std::tm tm{}; ::localtime_r(&now, &tm); char date[32] = {}; std::strftime(date, sizeof date, "%Y-%m-%dT%H:%M:%S", &tm);
    std::string body = trashinfo(recorded, date);
    for (int i = 0; i < 10000; ++i) {
        std::string cand = i ? name + "." + std::to_string(i) : name, info = tdir + "/info/" + cand + ".trashinfo";
        int fd = ::open(info.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) { if (errno == EEXIST) continue; return {{}, last_error("trash: " + info)}; }
        bool wrote = ::write(fd, body.data(), body.size()) == static_cast<ssize_t>(body.size()); ::close(fd);
        std::string dest = tdir + "/files/" + cand;
        struct ::stat dst{};
        if (!wrote || ::lstat(dest.c_str(), &dst) == 0) { ::unlink(info.c_str()); if (!wrote) return {{}, errno_error(EIO, "trash: cannot write " + info)}; continue; }
        if (::rename(abs.c_str(), dest.c_str()) != 0) { auto e = last_error("trash: rename " + abs); ::unlink(info.c_str()); return {{}, e}; }
        return {dest, {}};
    }
    return {{}, errno_error(EEXIST, "trash: too many items named " + name)};
}
#endif
}

inline bool trash_available() {
#if defined(FORGE_OS_ANDROID) || defined(FORGE_OS_IOS)
    return false;
#elif defined(FORGE_OS_MACOS)
    return detail::objc_api::get().ok();
#elif defined(FORGE_OS_WINDOWS)
    return true;
#elif defined(FORGE_OS_POSIX)
    return !home_dir().empty() || !getenv("XDG_DATA_HOME").empty();
#else
    return false;
#endif
}

inline result<std::string> trash(const std::string& path) {
    if (path.empty()) return {{}, errno_error(EINVAL, "trash: empty path")};
    std::string abs = absolute(path);
#if defined(FORGE_OS_ANDROID) || defined(FORGE_OS_IOS)
    return {{}, unsupported("trash")};
#elif defined(FORGE_OS_MACOS)
    auto& o = detail::objc_api::get();
    if (!o.ok()) return {{}, unsupported("trash (Objective-C runtime not loadable)")};
    if (!stat(abs).exists) return {{}, errno_error(ENOENT, "trash " + abs + ": no such file")};
    static void* foundation = ::dlopen("/System/Library/Frameworks/Foundation.framework/Foundation", RTLD_LAZY); (void)foundation;
    auto push = reinterpret_cast<void* (*)()>(::dlsym(o.h, "objc_autoreleasePoolPush")); auto pop = reinterpret_cast<void (*)(void*)>(::dlsym(o.h, "objc_autoreleasePoolPop"));
    void* pool = push ? push() : nullptr;
    result<std::string> r{{}, unsupported("trash (NSFileManager not available)")};
    void* str_cls = o.cls("NSString"); void* url_cls = o.cls("NSURL"); void* fm_cls = o.cls("NSFileManager");
    if (str_cls && url_cls && fm_cls) {
        void* s = o.send<void*, const char*>(str_cls, "stringWithUTF8String:", abs.c_str());
        void* u = s ? o.send<void*, void*>(url_cls, "fileURLWithPath:", s) : nullptr;
        void* fm = o.send(fm_cls, "defaultManager"); void* out = nullptr; void* err = nullptr;
        if (u && fm) {
            bool ok = o.send<signed char, void*, void**, void**>(fm, "trashItemAtURL:resultingItemURL:error:", u, &out, &err) != 0;
            if (ok) {
                std::string dest; if (out) if (void* p = o.send(out, "path")) if (auto c = o.send<const char*>(p, "UTF8String")) dest = c;
                r = {dest, {}};
            } else {
                std::string msg = "trash " + abs; long code = 0;
                if (err) { code = o.send<long>(err, "code"); if (void* d = o.send(err, "localizedDescription")) if (auto c = o.send<const char*>(d, "UTF8String")) msg += std::string(": ") + c; }
                r = {{}, {code ? int(code) : -1, msg}};
            }
        }
    }
    if (pool && pop) pop(pool);
    return r;
#elif defined(FORGE_OS_WINDOWS)
    static HMODULE sh = LoadLibraryExW(L"shell32.dll", nullptr, 0x800 /*LOAD_LIBRARY_SEARCH_SYSTEM32*/);
    using op_fn = int(WINAPI*)(SHFILEOPSTRUCTW*);
    using q_fn = HRESULT(WINAPI*)(const wchar_t*, SHQUERYRBINFO*);
    auto op = sh ? reinterpret_cast<op_fn>(reinterpret_cast<void*>(GetProcAddress(sh, "SHFileOperationW"))) : nullptr;
    auto query = sh ? reinterpret_cast<q_fn>(reinterpret_cast<void*>(GetProcAddress(sh, "SHQueryRecycleBinW"))) : nullptr;
    if (!op) return {{}, unsupported("trash (shell32 SHFileOperationW)")};
    std::wstring w = widen(abs); for (auto& c : w) if (c == L'/') c = L'\\';
    if (GetFileAttributesW(w.c_str()) == INVALID_FILE_ATTRIBUTES) return {{}, last_error("trash " + abs)};
    if (w.size() >= 2 && w[0] == L'\\' && w[1] == L'\\') return {{}, unsupported("trash (network shares have no recycle bin)")};
    if (query && w.size() >= 3 && w[1] == L':') {
        std::wstring root = w.substr(0, 3); SHQUERYRBINFO qi{}; qi.cbSize = sizeof qi;
        if (GetDriveTypeW(root.c_str()) == DRIVE_REMOTE || FAILED(query(root.c_str(), &qi))) return {{}, unsupported("trash (no recycle bin on " + narrow(root) + ")")};
    }
    w.push_back(L'\0');   // pFrom is a double-NUL-terminated list
    SHFILEOPSTRUCTW f{}; f.wFunc = FO_DELETE; f.pFrom = w.c_str();
    f.fFlags = FILEOP_FLAGS(FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI | FOF_NOCONFIRMMKDIR);
    int rc = op(&f);
    if (rc != 0) return {{}, error{rc, "trash " + abs + ": SHFileOperation error " + std::to_string(rc)}};
    if (f.fAnyOperationsAborted) return {{}, error{int(ERROR_CANCELLED), "trash " + abs + ": aborted"}};
    return {std::string(), {}};
#elif defined(FORGE_OS_POSIX)
    if (!trash_available()) return {{}, unsupported("trash (no home directory)")};
    return detail_trash::trash_xdg(abs);
#else
    return {{}, unsupported("trash")};
#endif
}
} // namespace forge::os

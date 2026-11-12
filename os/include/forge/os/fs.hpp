#pragma once
// forge::os::fs: file-system operations beyond os.hpp's file/stat/list_dir: copy (file / tree), links,
// permissions and ownership, timestamps, glob, recursive walk, advisory file locks, disk usage, mounts,
// temp files/dirs and atomic writes. std::filesystem underneath where it is portable; native calls for the
// rest. Nothing throws: every call returns os::result / os::status. Guide: docs/os-wrapper.md.
#include <forge/os/os.hpp>
#include <forge/os/private_files.hpp>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <system_error>
#if defined(FORGE_OS_POSIX)
#include <sys/file.h>
#include <sys/statvfs.h>
#include <sys/time.h>
#endif
#if defined(__APPLE__) || defined(__FreeBSD__)
#include <sys/param.h>
#include <sys/ucred.h>
#include <sys/mount.h>
#endif
#if defined(__linux__)
#include <sys/ioctl.h>
#include <sys/xattr.h>
#elif defined(__APPLE__)
#include <copyfile.h>
#include <sys/clonefile.h>
#if __has_include(<sys/xattr.h>)
#include <sys/xattr.h>
#else   // SDKs trimmed for cross-compiling leave the header out; the calls are in libSystem all the same
extern "C" {
ssize_t getxattr(const char* path, const char* name, void* value, size_t size, u_int32_t position, int options);
int setxattr(const char* path, const char* name, const void* value, size_t size, u_int32_t position, int options);
int removexattr(const char* path, const char* name, int options);
ssize_t listxattr(const char* path, char* namebuf, size_t size, int options);
}
#endif
#elif defined(__FreeBSD__)
#include <sys/extattr.h>
#elif defined(_WIN32)
#include <winioctl.h>
#endif

namespace forge::os::fs {
namespace stdfs = std::filesystem;

// std::filesystem reports errno values (generic_category: libstdc++, including mingw) or native codes
// (system_category: Win32 on MSVC); the error records which, so os::classify reads it right.
inline error from_ec(const std::error_code& ec, std::string_view what) {
    return {ec.value() ? ec.value() : -1, std::string(what) + ": " + ec.message(), ec.category() == std::generic_category() ? err_domain::posix : err_domain::native};
}
inline stdfs::path u8path(const std::string& s) { return stdfs::path(std::u8string(s.begin(), s.end())); }
inline std::string u8str(const stdfs::path& p) { auto u = p.u8string(); return std::string(u.begin(), u.end()); }

// ---------------------------------------------------------------- copy / move / remove
struct copy_opts { bool overwrite = false, recursive = true, follow_symlinks = false, skip_existing = false; };
inline status copy(const std::string& from, const std::string& to, copy_opts o = {}) {
    std::error_code ec;
    auto opt = stdfs::copy_options::none;
    if (o.overwrite) opt |= stdfs::copy_options::overwrite_existing;
    else if (o.skip_existing) opt |= stdfs::copy_options::skip_existing;
    if (o.recursive) opt |= stdfs::copy_options::recursive;
    opt |= o.follow_symlinks ? stdfs::copy_options::none : stdfs::copy_options::copy_symlinks;
    stdfs::copy(u8path(from), u8path(to), opt, ec);
    if (ec) return {{}, from_ec(ec, "copy " + from + " -> " + to)};
    return {};
}
inline status copy_tree(const std::string& from, const std::string& to, bool overwrite = false) { return copy(from, to, {overwrite, true, false, !overwrite}); }
inline status move(const std::string& from, const std::string& to) {
    std::error_code ec; stdfs::rename(u8path(from), u8path(to), ec);
    if (ec == std::errc::cross_device_link) {   // different file systems: copy then remove
        if (auto s = copy(from, to, {true, true, false, false}); !s) return s;
        stdfs::remove_all(u8path(from), ec);
    }
    if (ec) return {{}, from_ec(ec, "move " + from + " -> " + to)};
    return {};
}
inline result<std::uint64_t> remove_all(const std::string& p) {
    std::error_code ec; auto n = stdfs::remove_all(u8path(p), ec);
    if (ec) return {0, from_ec(ec, "remove " + p)};
    return {std::uint64_t(n), {}};
}
inline status make_dirs(const std::string& p) { std::error_code ec; stdfs::create_directories(u8path(p), ec); if (ec) return {{}, from_ec(ec, "mkdir " + p)}; return {}; }
inline bool exists(const std::string& p) { std::error_code ec; return stdfs::exists(u8path(p), ec); }
inline bool is_dir(const std::string& p) { std::error_code ec; return stdfs::is_directory(u8path(p), ec); }
inline bool is_file(const std::string& p) { std::error_code ec; return stdfs::is_regular_file(u8path(p), ec); }
inline bool is_symlink(const std::string& p) { std::error_code ec; return stdfs::is_symlink(u8path(p), ec); }
inline result<std::uint64_t> file_size(const std::string& p) { std::error_code ec; auto n = stdfs::file_size(u8path(p), ec); if (ec) return {0, from_ec(ec, "size " + p)}; return {std::uint64_t(n), {}}; }
inline result<std::string> absolute(const std::string& p) { std::error_code ec; auto a = stdfs::weakly_canonical(stdfs::absolute(u8path(p), ec), ec); if (ec) return {{}, from_ec(ec, "absolute " + p)}; return {u8str(a), {}}; }
inline std::string current_dir() { std::error_code ec; return u8str(stdfs::current_path(ec)); }
inline status set_current_dir(const std::string& p) { std::error_code ec; stdfs::current_path(u8path(p), ec); if (ec) return {{}, from_ec(ec, "cd " + p)}; return {}; }

// ---------------------------------------------------------------- links
#if defined(FORGE_OS_WINDOWS)
namespace detail {
// Reparse data, laid out by hand (REPARSE_DATA_BUFFER lives in the driver kit headers): tag, data length,
// reserved, then for mount points and symbolic links four USHORTs (substitute name offset / length, print name
// offset / length, in bytes into the path buffer); symbolic links add a ULONG of flags before the path buffer.
inline constexpr DWORD tag_mount_point = 0xA0000003u, tag_symlink = 0xA000000Cu;
inline HANDLE open_reparse(const std::string& p, DWORD access) {
    return CreateFileW(wpath(p).c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                       FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
}
}
// A directory junction: `link` becomes a directory that redirects to `target_dir` (made absolute; junctions hold
// absolute local paths). Needs no privilege, unlike symbolic links before Windows 10 1703 or without Developer Mode.
inline status junction(const std::string& target_dir, const std::string& link) {
    std::wstring full(32768, L'\0'); DWORD n = GetFullPathNameW(widen(target_dir).c_str(), DWORD(full.size()), full.data(), nullptr);
    if (!n || n >= full.size()) return {{}, last_error("junction " + link)};
    full.resize(n);
    for (auto& c : full) if (c == L'/') c = L'\\';
    std::wstring sub = L"\\??\\" + full;
    if (!CreateDirectoryW(wpath(link).c_str(), nullptr)) return {{}, last_error("junction " + link)};
    HANDLE h = detail::open_reparse(link, GENERIC_WRITE);
    auto fail = [&](const char* what) { auto e = last_error(std::string(what) + " " + link); if (h != INVALID_HANDLE_VALUE) CloseHandle(h); RemoveDirectoryW(wpath(link).c_str()); return status{{}, e}; };
    if (h == INVALID_HANDLE_VALUE) return fail("junction");
    const std::size_t sb = sub.size() * sizeof(wchar_t), pb = full.size() * sizeof(wchar_t);
    const std::size_t path_bytes = sb + sizeof(wchar_t) + pb + sizeof(wchar_t);
    std::vector<unsigned char> buf(8 + 8 + path_bytes, 0);
    auto put16 = [&](std::size_t off, std::size_t v) { buf[off] = static_cast<unsigned char>(v & 0xff); buf[off + 1] = static_cast<unsigned char>((v >> 8) & 0xff); };
    DWORD tag = detail::tag_mount_point; std::memcpy(buf.data(), &tag, 4);
    put16(4, 8 + path_bytes);                       // ReparseDataLength: everything after the 8-byte header
    put16(8, 0); put16(10, sb); put16(12, sb + sizeof(wchar_t)); put16(14, pb);
    std::memcpy(buf.data() + 16, sub.data(), sb); std::memcpy(buf.data() + 16 + sb + sizeof(wchar_t), full.data(), pb);
    DWORD got = 0;
    if (!DeviceIoControl(h, FSCTL_SET_REPARSE_POINT, buf.data(), DWORD(buf.size()), nullptr, 0, &got, nullptr)) return fail("FSCTL_SET_REPARSE_POINT");
    CloseHandle(h); return {};
}
// Symbolic link through CreateSymbolicLinkW (libstdc++'s std::filesystem on MinGW cannot create them). Unprivileged
// creation needs Windows 10 1703+ with Developer Mode; where the privilege is missing a directory target gets a
// junction instead (same effect for local directories), a file target fails with ERROR_PRIVILEGE_NOT_HELD.
inline status symlink(const std::string& target, const std::string& link) {
    // The target is stored as written (it may be relative to the link's directory), with backslashes.
    std::wstring wt = widen(target); for (auto& c : wt) if (c == L'/') c = L'\\';
    std::string probe = target;
    const bool rel = !(target.size() > 1 && (target[1] == ':' || is_sep(target[0])));
    if (rel) probe = os::join(os::dirname(link), target);
    DWORD attr = GetFileAttributesW(wpath(probe).c_str());
    const bool dir = attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY);
    const DWORD fl = dir ? 0x1u /*SYMBOLIC_LINK_FLAG_DIRECTORY*/ : 0u;
    auto wl = wpath(link);
    if (CreateSymbolicLinkW(wl.c_str(), wt.c_str(), fl | 0x2u /*SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE*/)) return {};
    DWORD e = GetLastError();
    if (e == ERROR_INVALID_PARAMETER) { if (CreateSymbolicLinkW(wl.c_str(), wt.c_str(), fl)) return {}; e = GetLastError(); }   // before 1703: flag unknown
    if (e == ERROR_PRIVILEGE_NOT_HELD && dir) return junction(probe, link);
    SetLastError(e); return {{}, last_error("symlink " + link)};
}
inline status hardlink(const std::string& target, const std::string& link) {
    if (!CreateHardLinkW(wpath(link).c_str(), wpath(target).c_str(), nullptr)) return {{}, last_error("hardlink " + link)};
    return {};
}
// The target of a symbolic link or junction (its print name: what the creator wrote, or the absolute path).
inline result<std::string> read_link(const std::string& p) {
    HANDLE h = detail::open_reparse(p, FILE_READ_ATTRIBUTES);
    if (h == INVALID_HANDLE_VALUE) return {{}, last_error("readlink " + p)};
    std::vector<unsigned char> buf(16 * 1024 + 64); DWORD got = 0;
    BOOL ok = DeviceIoControl(h, FSCTL_GET_REPARSE_POINT, nullptr, 0, buf.data(), DWORD(buf.size()), &got, nullptr);
    auto e = ok ? error{} : last_error("readlink " + p); CloseHandle(h);
    if (!ok) return {{}, e};
    DWORD tag = 0; std::memcpy(&tag, buf.data(), 4);
    if ((tag != detail::tag_symlink && tag != detail::tag_mount_point) || got < 16) return {{}, {int(ERROR_NOT_A_REPARSE_POINT), "readlink " + p + ": not a symbolic link or junction"}};
    auto get16 = [&](std::size_t off) { return std::size_t(buf[off]) | (std::size_t(buf[off + 1]) << 8); };
    const std::size_t so = get16(8), sl = get16(10), po = get16(12), pl = get16(14), base = tag == detail::tag_symlink ? 20 : 16;
    auto name = [&](std::size_t off, std::size_t len) { std::wstring w; if (base + off + len <= got) { w.resize(len / sizeof(wchar_t)); std::memcpy(w.data(), buf.data() + base + off, len); } return w; };
    std::wstring w = name(po, pl);
    if (w.empty()) { w = name(so, sl); if (w.rfind(L"\\??\\", 0) == 0) w = w.substr(4); }
    return {narrow(w), {}};
}
#else
// POSIX: `link` -> `target_dir` as a symbolic link to its absolute path (the closest thing to a junction).
inline status symlink(const std::string& target, const std::string& link);
inline status junction(const std::string& target_dir, const std::string& link) { return symlink(os::absolute(target_dir), link); }
inline status symlink(const std::string& target, const std::string& link) {
    std::error_code ec;
    if (is_dir(target)) stdfs::create_directory_symlink(u8path(target), u8path(link), ec);
    else stdfs::create_symlink(u8path(target), u8path(link), ec);
    if (ec) return {{}, from_ec(ec, "symlink " + link)};
    return {};
}
inline status hardlink(const std::string& target, const std::string& link) { std::error_code ec; stdfs::create_hard_link(u8path(target), u8path(link), ec); if (ec) return {{}, from_ec(ec, "hardlink " + link)}; return {}; }
inline result<std::string> read_link(const std::string& p) { std::error_code ec; auto t = stdfs::read_symlink(u8path(p), ec); if (ec) return {{}, from_ec(ec, "readlink " + p)}; return {u8str(t), {}}; }
#endif

// ---------------------------------------------------------------- permissions, ownership, times
inline result<unsigned> mode(const std::string& p) { std::error_code ec; auto s = stdfs::status(u8path(p), ec); if (ec) return {0, from_ec(ec, "stat " + p)}; return {unsigned(s.permissions()) & 07777u, {}}; }
inline status chmod(const std::string& p, unsigned mode) {
    std::error_code ec; stdfs::permissions(u8path(p), stdfs::perms(mode & 07777u), stdfs::perm_options::replace, ec);
    if (ec) return {{}, from_ec(ec, "chmod " + p)};
    return {};
}
inline status chown(const std::string& p, long uid, long gid) {
#if defined(FORGE_OS_POSIX)
    if (::lchown(p.c_str(), uid_t(uid), gid_t(gid))) return {{}, last_error("chown " + p)};
    return {};
#else   // NTFS owners are SIDs set through ACL APIs (and need SeRestorePrivilege for anyone but yourself); a
        // numeric uid/gid pair has no faithful mapping, so this stays unsupported.
    (void)p; (void)uid; (void)gid; return {{}, unsupported("chown")};
#endif
}
// Unix epoch minus file_clock epoch, in file_clock ticks (the epoch is implementation-defined; clock_cast is not
// available everywhere). Kept in the clock's own tick type: MSVC's 1601 FILETIME epoch is ~1.3e19 ns ago, which
// overflows int64 nanoseconds. Sampled once; good to well under a millisecond.
inline stdfs::file_time_type::duration file_clock_offset() {
    using fdur = stdfs::file_time_type::duration;
    static const fdur off = [] {
        auto f = stdfs::file_time_type::clock::now(); auto s = std::chrono::system_clock::now();
        fdur v = std::chrono::duration_cast<fdur>(s.time_since_epoch()) - f.time_since_epoch();
        // Every implementation's offset is a whole number of seconds (0, the 1601 FILETIME epoch, libstdc++'s
        // 2174 epoch), and some file clocks tick coarsely (mingw: ~15 ms), so round to the second.
        auto r = std::chrono::round<std::chrono::seconds>(v);
        return std::chrono::duration_cast<fdur>(r);
    }();
    return off;
}
#if defined(FORGE_OS_WINDOWS)
// Windows: FILETIME directly (100 ns ticks since 1601). mingw's libstdc++ last_write_time goes through
// _wstat64/_wutime64 and keeps whole seconds only; MSVC's is fine but this keeps both identical.
namespace detail {
inline constexpr std::int64_t filetime_unix_epoch = 116444736000000000LL;   // 1970-01-01 in FILETIME ticks
inline HANDLE open_for_times(const std::string& p, DWORD access) {
    return CreateFileW(widen(p).c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                       OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);   // BACKUP_SEMANTICS: directories too
}
}
inline result<std::int64_t> mtime_ns(const std::string& p) {
    HANDLE h = detail::open_for_times(p, FILE_READ_ATTRIBUTES);
    if (h == INVALID_HANDLE_VALUE) return {0, last_error("mtime " + p)};
    FILETIME w{}; BOOL ok = GetFileTime(h, nullptr, nullptr, &w);
    auto err = ok ? error{} : last_error("mtime " + p); CloseHandle(h);
    if (!ok) return {0, err};
    std::int64_t ticks = std::int64_t((std::uint64_t(w.dwHighDateTime) << 32) | w.dwLowDateTime);
    return {(ticks - detail::filetime_unix_epoch) * 100, {}};
}
inline status set_mtime_ns(const std::string& p, std::int64_t ns) {
    std::int64_t ticks = (ns >= 0 ? ns / 100 : -((-ns + 99) / 100)) + detail::filetime_unix_epoch;
    if (ticks < 0) return {{}, error{int(ERROR_INVALID_PARAMETER), "set mtime " + p + ": before 1601"}};
    HANDLE h = detail::open_for_times(p, FILE_WRITE_ATTRIBUTES);
    if (h == INVALID_HANDLE_VALUE) return {{}, last_error("set mtime " + p)};
    FILETIME w{DWORD(std::uint64_t(ticks) & 0xffffffffu), DWORD(std::uint64_t(ticks) >> 32)};
    BOOL ok = SetFileTime(h, nullptr, nullptr, &w);
    auto err = ok ? error{} : last_error("set mtime " + p); CloseHandle(h);
    if (!ok) return {{}, err};
    return {};
}
#else
// Modification time in ns since the Unix epoch.
inline result<std::int64_t> mtime_ns(const std::string& p) {
    std::error_code ec; auto t = stdfs::last_write_time(u8path(p), ec);
    if (ec) return {0, from_ec(ec, "mtime " + p)};
    return {std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch() + file_clock_offset()).count(), {}};
}
inline status set_mtime_ns(const std::string& p, std::int64_t ns) {
    std::error_code ec;
    auto d = std::chrono::duration_cast<stdfs::file_time_type::duration>(std::chrono::nanoseconds(ns)) - file_clock_offset();
    stdfs::last_write_time(u8path(p), stdfs::file_time_type(d), ec);
    if (ec) return {{}, from_ec(ec, "set mtime " + p)};
    return {};
}
#endif
inline status touch(const std::string& p) {
    if (!exists(p)) { std::ofstream f(u8path(p), std::ios::app); if (!f) return {{}, last_error("touch " + p)}; return {}; }
    return set_mtime_ns(p, std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

// ---------------------------------------------------------------- glob and walk
// Glob pattern match on '/'-separated paths: * and ? never cross '/', ** matches any number of whole
// segments, [abc] / [a-z] / [!x] classes, \ escapes. Case-sensitive.
inline bool glob_match(std::string_view pat, std::string_view s) {
    auto seg_match = [](std::string_view p, std::string_view t) {
        std::size_t pi = 0, ti = 0, star = std::string_view::npos, mark = 0;
        auto cls = [&](std::size_t& i, char c) {   // p[i] == '['
            std::size_t j = i + 1; bool neg = false, hit = false;
            if (j < p.size() && (p[j] == '!' || p[j] == '^')) { neg = true; ++j; }
            std::size_t start = j;
            for (; j < p.size() && (p[j] != ']' || j == start); ++j) {
                if (j + 2 < p.size() && p[j + 1] == '-' && p[j + 2] != ']') { if (c >= p[j] && c <= p[j + 2]) hit = true; j += 2; }
                else if (p[j] == c) hit = true;
            }
            if (j >= p.size()) return false;   // unterminated class: no match
            i = j + 1; return hit != neg;
        };
        while (ti < t.size()) {
            if (pi < p.size() && p[pi] == '*') { star = pi++; mark = ti; continue; }
            if (pi < p.size()) {
                if (p[pi] == '?') { ++pi; ++ti; continue; }
                if (p[pi] == '[') { std::size_t i = pi; if (cls(i, t[ti])) { pi = i; ++ti; continue; } }
                else if (p[pi] == '\\' && pi + 1 < p.size()) { if (p[pi + 1] == t[ti]) { pi += 2; ++ti; continue; } }
                else if (p[pi] == t[ti]) { ++pi; ++ti; continue; }
            }
            if (star == std::string_view::npos) return false;
            pi = star + 1; ti = ++mark;
        }
        while (pi < p.size() && p[pi] == '*') ++pi;
        return pi == p.size();
    };
    auto split = [](std::string_view x) { std::vector<std::string_view> v; std::size_t a = 0; for (std::size_t i = 0; i <= x.size(); ++i) if (i == x.size() || x[i] == '/') { v.push_back(x.substr(a, i - a)); a = i + 1; } return v; };
    auto P = split(pat), S = split(s);
    std::function<bool(std::size_t, std::size_t)> m = [&](std::size_t i, std::size_t j) -> bool {
        if (i == P.size()) return j == S.size();
        if (P[i] == "**") { for (std::size_t k = j; k <= S.size(); ++k) if (m(i + 1, k)) return true; return false; }
        if (j == S.size()) return false;
        return seg_match(P[i], S[j]) && m(i + 1, j + 1);
    };
    return m(0, 0);
}
struct walk_entry { std::string path; bool is_dir = false, is_file = false, is_link = false; std::uint64_t size = 0; int depth = 0; };
struct walk_opts { int max_depth = -1; bool follow_symlinks = false, include_dirs = true, include_hidden = true; };
// Recursive listing below `root` (not including root). Unreadable directories are skipped.
inline result<std::vector<walk_entry>> walk(const std::string& root, walk_opts o = {}) {
    std::error_code ec;
    auto opt = stdfs::directory_options::skip_permission_denied;
    if (o.follow_symlinks) opt |= stdfs::directory_options::follow_directory_symlink;
    stdfs::recursive_directory_iterator it(u8path(root), opt, ec), end;
    if (ec) return {{}, from_ec(ec, "walk " + root)};
    std::vector<walk_entry> out;
    for (; it != end; it.increment(ec)) {
        if (ec) { ec.clear(); continue; }
        auto& e = *it;
        auto name = u8str(e.path().filename());
        if (!o.include_hidden && !name.empty() && name[0] == '.') { if (e.is_directory(ec)) it.disable_recursion_pending(); continue; }
        if (o.max_depth >= 0 && it.depth() >= o.max_depth) it.disable_recursion_pending();
        walk_entry w; w.path = u8str(e.path()); w.depth = it.depth();
        w.is_link = e.is_symlink(ec); w.is_dir = e.is_directory(ec); w.is_file = e.is_regular_file(ec);
        if (w.is_file) { auto n = e.file_size(ec); w.size = ec ? 0 : std::uint64_t(n); ec.clear(); }
        if (w.is_dir && !o.include_dirs) continue;
        out.push_back(std::move(w));
    }
    return {std::move(out), {}};
}
// Paths matching `pattern` (relative patterns resolve against `base`; results keep the same form).
inline result<std::vector<std::string>> glob(const std::string& pattern, const std::string& base = ".") {
    std::string pat = pattern;
#if defined(FORGE_OS_WINDOWS)
    for (auto& c : pat) if (c == '\\') c = '/';
#endif
    // Longest literal directory prefix: walk only below it.
    std::size_t cut = 0;
    for (std::size_t i = 0; i < pat.size(); ++i) { if (pat[i] == '*' || pat[i] == '?' || pat[i] == '[') break; if (pat[i] == '/') cut = i + 1; }
    std::string prefix = pat.substr(0, cut);
    bool abs = !pat.empty() && (pat[0] == '/' || (pat.size() > 1 && pat[1] == ':'));
    std::string root = abs ? (prefix.empty() ? "/" : prefix) : (base + "/" + prefix);
    std::vector<std::string> out;
    if (cut == pat.size()) { if (exists(abs ? pat : base + "/" + pat)) out.push_back(pat); return {std::move(out), {}}; }
    bool deep = pat.find("**") != std::string::npos;
    int depth = 0; for (std::size_t i = cut; i < pat.size(); ++i) if (pat[i] == '/') ++depth;
    walk_opts wo; wo.max_depth = deep ? -1 : depth;
    auto w = walk(root, wo);
    if (!w) { if (!exists(root)) return {std::move(out), {}}; return {{}, w.err}; }
    std::string rootn = u8str(u8path(root).lexically_normal());
    for (auto& c : rootn) if (c == '\\') c = '/';   // Windows: lexically_normal gives '\\', entries below are compared with '/'
    if (!rootn.empty() && rootn.back() != '/') rootn += '/';
    for (auto& e : *w) {
        std::string p = u8str(u8path(e.path).lexically_normal());
        for (auto& c : p) if (c == '\\') c = '/';
        if (p.rfind(rootn, 0) != 0) continue;
        std::string rel = prefix + p.substr(rootn.size());
        if (glob_match(pat, rel)) out.push_back(rel);
    }
    std::sort(out.begin(), out.end());
    return {std::move(out), {}};
}

// ---------------------------------------------------------------- advisory locks
class file_lock {
public:
    file_lock() = default;
    file_lock(const file_lock&) = delete; file_lock& operator=(const file_lock&) = delete;
    file_lock(file_lock&& o) noexcept : h_(std::exchange(o.h_, invalid())) {}
    file_lock& operator=(file_lock&& o) noexcept { if (this != &o) { unlock(); h_ = std::exchange(o.h_, invalid()); } return *this; }
    ~file_lock() { unlock(); }
    // Lock `path` (created if missing). exclusive=false takes a shared lock. wait=false fails at once if held.
    static result<file_lock> acquire(const std::string& path, bool exclusive = true, bool wait = true) {
        file_lock l;
#if defined(FORGE_OS_POSIX)
        int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd < 0) return {{}, last_error("lock " + path)};
        int op = (exclusive ? LOCK_EX : LOCK_SH) | (wait ? 0 : LOCK_NB);
        while (::flock(fd, op) != 0) { if (errno == EINTR) continue; auto e = last_error("lock " + path); ::close(fd); return {{}, e}; }
        l.h_ = fd;
#else
        HANDLE h = CreateFileW(widen(path).c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_ALWAYS, 0, nullptr);
        if (h == INVALID_HANDLE_VALUE) return {{}, last_error("lock " + path)};
        OVERLAPPED ov{};
        DWORD fl = (exclusive ? LOCKFILE_EXCLUSIVE_LOCK : 0) | (wait ? 0 : LOCKFILE_FAIL_IMMEDIATELY);
        if (!LockFileEx(h, fl, 0, MAXDWORD, MAXDWORD, &ov)) { auto e = last_error("lock " + path); CloseHandle(h); return {{}, e}; }
        l.h_ = h;
#endif
        return {std::move(l), {}};
    }
    bool held() const { return h_ != invalid(); }
    void unlock() {
        if (!held()) return;
#if defined(FORGE_OS_POSIX)
        ::flock(h_, LOCK_UN); ::close(h_);
#else
        OVERLAPPED ov{}; UnlockFileEx(h_, 0, MAXDWORD, MAXDWORD, &ov); CloseHandle(h_);
#endif
        h_ = invalid();
    }
private:
#if defined(FORGE_OS_POSIX)
    using handle = int; static constexpr handle invalid() { return -1; }
#else
    using handle = HANDLE; static handle invalid() { return INVALID_HANDLE_VALUE; }
#endif
    handle h_ = invalid();
};

// ---------------------------------------------------------------- disk usage and mounts
struct disk_usage_t { std::uint64_t total = 0, free = 0, available = 0; };
inline result<disk_usage_t> disk_usage(const std::string& p) {
    std::error_code ec; auto s = stdfs::space(u8path(p), ec);
    if (ec) return {{}, from_ec(ec, "disk usage " + p)};
    return {{std::uint64_t(s.capacity), std::uint64_t(s.free), std::uint64_t(s.available)}, {}};
}
// /proc/self/mounts escapes space, tab, newline and backslash as \ooo; anything that is not three octal digits
// stays literal (never throws).
inline std::string mount_unescape(const std::string& s) {
    auto oct = [](char c) { return c >= '0' && c <= '7'; };
    std::string o;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 3 < s.size() && oct(s[i + 1]) && oct(s[i + 2]) && oct(s[i + 3])) {
            o += char(((s[i + 1] - '0') << 6) | ((s[i + 2] - '0') << 3) | (s[i + 3] - '0')); i += 3;
        } else o += s[i];
    }
    return o;
}
struct mount_info { std::string device, path, type, options; };
inline result<std::vector<mount_info>> mounts() {
    std::vector<mount_info> out;
#if defined(__linux__)
    std::ifstream f("/proc/self/mounts");
    if (!f) return {{}, last_error("/proc/self/mounts")};
    std::string dev, path, type, opts, a, b;
    while (f >> dev >> path >> type >> opts >> a >> b) out.push_back({mount_unescape(dev), mount_unescape(path), type, opts});
    return {std::move(out), {}};
#elif defined(FORGE_OS_WINDOWS)
    char buf[512]; DWORD n = GetLogicalDriveStringsA(sizeof buf, buf);
    for (char* p = buf; p < buf + n && *p; p += std::strlen(p) + 1) {
        char fsname[64] = {}; GetVolumeInformationA(p, nullptr, 0, nullptr, nullptr, nullptr, fsname, sizeof fsname);
        out.push_back({p, p, fsname, ""});
    }
    return {std::move(out), {}};
#elif defined(__APPLE__) || defined(__FreeBSD__)
    // getfsstat into our own buffer (getmntinfo's buffer is shared process state, so not thread-safe).
    int n = ::getfsstat(nullptr, 0, MNT_NOWAIT); if (n < 0) return {{}, last_error("getfsstat")};
    std::vector<struct statfs> v(std::size_t(n) + 8);
    n = ::getfsstat(v.data(), int(v.size() * sizeof(struct statfs)), MNT_NOWAIT); if (n < 0) return {{}, last_error("getfsstat")};
    for (int i = 0; i < n; ++i) {
        const auto& s = v[std::size_t(i)]; std::string o = (s.f_flags & MNT_RDONLY) ? "ro" : "rw";
        if (s.f_flags & MNT_NOSUID) o += ",nosuid";
        if (s.f_flags & MNT_NOEXEC) o += ",noexec";
        if (s.f_flags & MNT_SYNCHRONOUS) o += ",sync";
        if (s.f_flags & MNT_LOCAL) o += ",local";
        out.push_back({s.f_mntfromname, s.f_mntonname, s.f_fstypename, o});
    }
    return {std::move(out), {}};
#else
    return {{}, unsupported("mounts")};
#endif
}

// ---------------------------------------------------------------- private files and directories (FSS FIO-1 / FIO-2)
// check_private, ensure_private_dir, write_new_private_file, private_kind: <forge/os/private_files.hpp> (namespace
// forge::os, so headers below os::fs can use them), available here too.
using os::private_kind;
using os::check_private;
using os::ensure_private_dir;
using os::write_new_private_file;

// ---------------------------------------------------------------- temp files and atomic writes
// FIO-1. Names: prefix + 10 characters from std::random_device + suffix; a name that exists is never reused.
//   make_temp_dir   POSIX mkdir(2) 0700 (exclusive); Windows CreateDirectoryW with the private DACL (the user, SYSTEM,
//                   Administrators; not inherited), so the directory is private wherever temp_dir() points.
//   make_temp_file  POSIX O_CREAT|O_EXCL|O_NOFOLLOW, 0600; Windows CREATE_NEW with the private DACL. In a shared
//                   sticky /tmp the file itself is private and cannot be pre-created or redirected by a symlink;
//                   for a private directory as well, pass dir = a make_temp_dir() / ensure_private_dir() result.
// `dir` (default temp_dir()) is where they are created.
inline std::string random_suffix(std::size_t n = 10) {
    static const char a[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    std::random_device rd; std::string s; for (std::size_t i = 0; i < n; ++i) s += a[rd() % (sizeof a - 1)];
    return s;
}
inline result<std::string> make_temp_dir(const std::string& prefix = "forge-", const std::string& dir = "") {
    const std::string base = dir.empty() ? temp_dir() : dir;
    for (int i = 0; i < 16; ++i) {
        auto p = u8str(u8path(base) / u8path(prefix + random_suffix()));
#if defined(FORGE_OS_POSIX)
        if (::mkdir(p.c_str(), 0700) == 0) return {p, {}};
        if (errno != EEXIST) return {{}, last_error("make_temp_dir " + p)};
#else
        detail_priv::private_sa sa(true);
        if (CreateDirectoryW(widen(p).c_str(), sa.get())) return {p, {}};
        if (GetLastError() != ERROR_ALREADY_EXISTS) return {{}, last_error("make_temp_dir " + p)};
#endif
    }
    return {{}, error{-1, "make_temp_dir: could not create a unique directory"}};
}
inline result<std::string> make_temp_file(const std::string& prefix = "forge-", const std::string& suffix = "", const std::string& dir = "") {
    const std::string base = dir.empty() ? temp_dir() : dir;
    for (int i = 0; i < 16; ++i) {
        auto p = u8str(u8path(base) / u8path(prefix + random_suffix() + suffix));
#if defined(FORGE_OS_POSIX)
        int fd = ::open(p.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (fd >= 0) { ::close(fd); return {p, {}}; }
        if (errno != EEXIST) return {{}, last_error("make_temp_file")};
#else   // CREATE_NEW: exclusive like O_EXCL, so a name that appears between choosing and creating is never reused
        detail_priv::private_sa sa(false);
        HANDLE h = CreateFileW(widen(p).c_str(), GENERIC_READ | GENERIC_WRITE, 0, sa.get(), CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) { CloseHandle(h); return {p, {}}; }
        if (GetLastError() != ERROR_FILE_EXISTS && GetLastError() != ERROR_ALREADY_EXISTS) return {{}, last_error("make_temp_file")};
#endif
    }
    return {{}, error{-1, "make_temp_file: could not create a unique file"}};
}
// Write to a sibling temp file, flush it to disk and rename over `path`: readers see the old or the new
// content, never a torn file. mode: POSIX permission bits (less the umask); on Windows a mode without any write
// bit makes the file read-only, the rest has no NTFS equivalent.
inline status write_file_atomic(const std::string& path, std::string_view data, unsigned mode = 0644) {
    std::string tmp = path + ".tmp-" + random_suffix(8);
#if defined(FORGE_OS_POSIX)
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode);
    if (fd < 0) return {{}, last_error("write " + tmp)};
    std::size_t off = 0;
    while (off < data.size()) { auto n = ::write(fd, data.data() + off, data.size() - off); if (n < 0) { if (errno == EINTR) continue; auto e = last_error("write " + tmp); ::close(fd); ::unlink(tmp.c_str()); return {{}, e}; } off += std::size_t(n); }
    if (::fsync(fd)) { auto e = last_error("fsync " + tmp); ::close(fd); ::unlink(tmp.c_str()); return {{}, e}; }
    ::close(fd);
    if (::rename(tmp.c_str(), path.c_str())) { auto e = last_error("rename " + path); ::unlink(tmp.c_str()); return {{}, e}; }
    // Make the rename itself durable: fsync the containing directory (best effort; some filesystems refuse).
    auto slash = path.find_last_of('/');
    std::string dir = slash == std::string::npos ? std::string(".") : slash == 0 ? std::string("/") : path.substr(0, slash);
    int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dfd >= 0) { (void)::fsync(dfd); ::close(dfd); }
    return {};
#else
    auto a = widen(tmp), b = widen(path);
    HANDLE h = CreateFileW(a.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, (mode & 0222) ? FILE_ATTRIBUTE_NORMAL : FILE_ATTRIBUTE_READONLY, nullptr);
    if (h == INVALID_HANDLE_VALUE) return {{}, last_error("write " + tmp)};
    auto fail = [&](const char* what) {
        auto e = last_error(std::string(what) + " " + tmp); CloseHandle(h);
        SetFileAttributesW(a.c_str(), FILE_ATTRIBUTE_NORMAL); DeleteFileW(a.c_str()); return status{{}, e};
    };
    for (std::size_t off = 0; off < data.size();) {
        DWORD put = 0; if (!WriteFile(h, data.data() + off, file::chunk(data.size() - off), &put, nullptr)) return fail("write");
        off += put;
    }
    if (!FlushFileBuffers(h)) return fail("flush");   // the data is on disk before the rename can expose it
    CloseHandle(h);
    if (!MoveFileExW(a.c_str(), b.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        auto e = last_error("rename " + path); SetFileAttributesW(a.c_str(), FILE_ATTRIBUTE_NORMAL); DeleteFileW(a.c_str()); return {{}, e};
    }
    return {};
#endif
}

// ---------------------------------------------------------------- fast copy and clone
// How copy_file moved the bytes: clone = the new file shares the source's blocks (copy-on-write, no data copied);
// kernel = copied inside the kernel (copy_file_range / copyfile / CopyFileEx, which may itself clone on ZFS, ReFS
// and Dev Drive volumes); user = a read / write loop.
enum class copy_method { clone, kernel, user };
namespace detail {
#if defined(FORGE_OS_POSIX)
inline status copy_loop(int in, int out, const std::string& what) {
    std::vector<char> buf(1 << 20);
    for (;;) {
        ssize_t n = ::read(in, buf.data(), buf.size());
        if (n < 0) { if (errno == EINTR) continue; return {{}, last_error("read " + what)}; }
        if (n == 0) return {};
        for (ssize_t off = 0; off < n;) { ssize_t w = ::write(out, buf.data() + off, std::size_t(n - off)); if (w < 0) { if (errno == EINTR) continue; return {{}, last_error("write " + what)}; } off += w; }
    }
}
#endif
inline bool cannot_clone(int e) {
#if defined(FORGE_OS_POSIX)
    return e == EOPNOTSUPP || e == ENOTSUP || e == EXDEV || e == EINVAL || e == ENOTTY || e == ENOSYS || e == EBADF || e == EPERM || e == ETXTBSY;
#else
    (void)e; return true;
#endif
}
}
// A copy-on-write clone of `from` at `to` (which must not exist): instant and shares storage until either file
// changes. Linux / Android: FICLONE (Btrfs, XFS with reflink, bcachefs, OCFS2, overlayfs over those). macOS /
// iOS: clonefile (APFS). FreeBSD and Windows have no call that clones or fails (their copy calls clone on their
// own where the file system can: ZFS block cloning, ReFS / Dev Drive), so clone_file is unsupported there, as it
// is on a file system that cannot share blocks. Use copy_file to get the best the system offers.
inline status clone_file(const std::string& from, const std::string& to) {
#if defined(__linux__)
    int in = ::open(from.c_str(), O_RDONLY | O_CLOEXEC); if (in < 0) return {{}, last_error("open " + from)};
    struct stat st{}; if (::fstat(in, &st)) { auto e = last_error("stat " + from); ::close(in); return {{}, e}; }
    int out = ::open(to.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, st.st_mode & 0777);
    if (out < 0) { auto e = last_error("create " + to); ::close(in); return {{}, e}; }
    const unsigned long ficlone = _IOW(0x94, 9, int);
    int r = ::ioctl(out, ficlone, in); int en = errno;
    ::close(in); ::close(out);
    if (r == 0) return {};
    ::unlink(to.c_str());
    if (detail::cannot_clone(en)) return {{}, unsupported("clone_file (the file system cannot share blocks)")};
    errno = en; return {{}, last_error("clone " + from)};
#elif defined(__APPLE__)
    if (::clonefile(from.c_str(), to.c_str(), 0) == 0) return {};
    if (errno == ENOTSUP || errno == EXDEV) return {{}, unsupported("clone_file (the file system cannot share blocks)")};
    return {{}, last_error("clonefile " + from)};
#else
    (void)from; (void)to; return {{}, unsupported("clone_file")};
#endif
}
// Copy one regular file the fastest way the OS has; overwrite = false fails if `to` exists. The mode bits are
// copied (POSIX); other metadata is not. Returns how the data was moved.
inline result<copy_method> copy_file(const std::string& from, const std::string& to, bool overwrite = false) {
#if defined(FORGE_OS_WINDOWS)
    if (!CopyFileExW(wpath(from).c_str(), wpath(to).c_str(), nullptr, nullptr, nullptr, overwrite ? 0 : COPY_FILE_FAIL_IF_EXISTS))
        return {copy_method::kernel, last_error("copy " + from + " -> " + to)};
    return {copy_method::kernel, {}};
#else
#if defined(__APPLE__)
    if (!overwrite || !os::exists(to)) {
        if (::clonefile(from.c_str(), to.c_str(), 0) == 0) return {copy_method::clone, {}};
        if (errno == EEXIST && !overwrite) return {copy_method::clone, last_error("copy " + from + " -> " + to)};
    }
#endif
    int in = ::open(from.c_str(), O_RDONLY | O_CLOEXEC); if (in < 0) return {copy_method::user, last_error("open " + from)};
    struct stat st{}; if (::fstat(in, &st)) { auto e = last_error("stat " + from); ::close(in); return {copy_method::user, e}; }
    if (!S_ISREG(st.st_mode)) { ::close(in); return {copy_method::user, errno_error(EINVAL, "copy " + from + ": not a regular file")}; }
    int out = ::open(to.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC | (overwrite ? O_TRUNC : O_EXCL), st.st_mode & 0777);
    if (out < 0) { auto e = last_error("create " + to); ::close(in); return {copy_method::user, e}; }
    auto done = [&](copy_method m, error e) { ::close(in); if (::close(out) != 0 && !e) e = last_error("close " + to); if (e) ::unlink(to.c_str()); return result<copy_method>{m, std::move(e)}; };
#if defined(__linux__)
    if (::ioctl(out, _IOW(0x94, 9, int) /*FICLONE*/, in) == 0) return done(copy_method::clone, {});
#endif
#if defined(__APPLE__)
    if (::fcopyfile(in, out, nullptr, COPYFILE_DATA) == 0) return done(copy_method::kernel, {});
#elif defined(__linux__) || (defined(__FreeBSD__) && __FreeBSD_version >= 1300037)
    {   // copy_file_range: in-kernel, reflinks or server-side copies where it can (Linux 4.5; across file systems 5.3)
        std::uint64_t left = std::uint64_t(st.st_size); bool any = false;
        while (left && android_api_level() >= 34) {
#if defined(__linux__)
            ssize_t n = ::syscall(SYS_copy_file_range, in, nullptr, out, nullptr, std::size_t(std::min<std::uint64_t>(left, 1u << 30)), 0u);
#else
            ssize_t n = ::copy_file_range(in, nullptr, out, nullptr, std::size_t(std::min<std::uint64_t>(left, 1u << 30)), 0);
#endif
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;   // error, or a source that reports a size it does not have (procfs): finish below
            left -= std::uint64_t(n); any = true;
        }
        if (!left && (any || st.st_size == 0) && android_api_level() >= 34) return done(copy_method::kernel, {});
        if (any) {   // partial: continue from where both descriptors stand
            auto s = detail::copy_loop(in, out, to); return done(copy_method::kernel, s.err);
        }
    }
#endif
    auto s = detail::copy_loop(in, out, to);
    return done(copy_method::user, s.err);
#endif
}

// ---------------------------------------------------------------- preallocation and sparse files
// Reserve disk blocks for the first `size` bytes of `f` (growing it to `size` if shorter), so later writes cannot
// fail for lack of space and the file is laid out in few extents. Linux / Android / FreeBSD: posix_fallocate.
// macOS / iOS: F_PREALLOCATE (contiguous if possible) then ftruncate. Windows: the allocation size
// (FileAllocationInfo) and the end of file. A file system that cannot reserve (ZFS, some network mounts)
// returns unsupported.
inline status preallocate(file& f, std::uint64_t size) {
#if defined(FORGE_OS_WINDOWS)
    auto cur = f.size(); if (!cur) return {{}, cur.err};
    if (cur.value >= size) return {};   // an allocation size below the end of file would truncate it
    FILE_ALLOCATION_INFO ai{}; ai.AllocationSize.QuadPart = LONGLONG(size);
    if (!SetFileInformationByHandle(f.native(), FileAllocationInfo, &ai, sizeof ai)) {
        const DWORD e = GetLastError();
        if (e == ERROR_INVALID_FUNCTION || e == ERROR_NOT_SUPPORTED || e == ERROR_CALL_NOT_IMPLEMENTED) return {{}, unsupported("preallocate (this file system cannot reserve space)")};
        SetLastError(e); return {{}, last_error("preallocate")};
    }
    return f.resize(size);
#elif defined(__APPLE__)
    auto cur = f.size(); if (!cur) return {{}, cur.err};
    if (size > cur.value) {
        fstore_t st{}; st.fst_flags = F_ALLOCATECONTIG | F_ALLOCATEALL; st.fst_posmode = F_PEOFPOSMODE; st.fst_offset = 0; st.fst_length = off_t(size - cur.value);
        if (::fcntl(f.native(), F_PREALLOCATE, &st) == -1) {
            st.fst_flags = F_ALLOCATEALL;
            if (::fcntl(f.native(), F_PREALLOCATE, &st) == -1) { if (errno == ENOTSUP || errno == EINVAL) return {{}, unsupported("preallocate (this file system cannot reserve space)")}; return {{}, last_error("F_PREALLOCATE")}; }
        }
        return f.resize(size);
    }
    return {};
#else
    int e;
    do e = ::posix_fallocate(f.native(), 0, off_t(size)); while (e == EINTR);
    if (e == EOPNOTSUPP || e == EINVAL || e == ENOSYS) { if (size == 0) return {}; return {{}, unsupported("preallocate (this file system cannot reserve space)")}; }
    if (e) return {{}, {e, "posix_fallocate: " + error_text(e)}};
    return {};
#endif
}
// Give the blocks behind [offset, offset + length) back to the file system; the range then reads as zeros and
// the file keeps its size. Linux / Android: fallocate(PUNCH_HOLE). macOS / iOS: F_PUNCHHOLE (APFS; offset and
// length must be multiples of the block size). FreeBSD 14+: fspacectl. Windows: the file is marked sparse, then
// FSCTL_SET_ZERO_DATA. Unsupported where the file system has no holes.
inline status punch_hole(file& f, std::uint64_t offset, std::uint64_t length) {
    if (!length) return {};
#if defined(FORGE_OS_WINDOWS)
    DWORD got = 0;
    if (!DeviceIoControl(f.native(), FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &got, nullptr)) {
        if (GetLastError() == ERROR_INVALID_FUNCTION || GetLastError() == ERROR_NOT_SUPPORTED) return {{}, unsupported("punch_hole (no sparse files on this file system)")};
        return {{}, last_error("FSCTL_SET_SPARSE")};
    }
    FILE_ZERO_DATA_INFORMATION z{}; z.FileOffset.QuadPart = LONGLONG(offset); z.BeyondFinalZero.QuadPart = LONGLONG(offset + length);
    if (!DeviceIoControl(f.native(), FSCTL_SET_ZERO_DATA, &z, sizeof z, nullptr, 0, &got, nullptr)) {
        if (GetLastError() == ERROR_INVALID_FUNCTION || GetLastError() == ERROR_NOT_SUPPORTED) return {{}, unsupported("punch_hole (no sparse files on this file system)")};
        return {{}, last_error("FSCTL_SET_ZERO_DATA")};
    }
    return {};
#elif defined(__linux__)
    if (::fallocate(f.native(), 0x02 /*FALLOC_FL_PUNCH_HOLE*/ | 0x01 /*FALLOC_FL_KEEP_SIZE*/, off_t(offset), off_t(length)) == 0) return {};
    if (errno == EOPNOTSUPP || errno == ENOSYS) return {{}, unsupported("punch_hole (no holes on this file system)")};
    return {{}, last_error("fallocate(PUNCH_HOLE)")};
#elif defined(__APPLE__) && defined(F_PUNCHHOLE)
    fpunchhole_t h{}; h.fp_offset = off_t(offset); h.fp_length = off_t(length);
    if (::fcntl(f.native(), F_PUNCHHOLE, &h) == 0) return {};
    if (errno == ENOTSUP || errno == ENOTTY) return {{}, unsupported("punch_hole (no holes on this file system)")};
    return {{}, last_error("F_PUNCHHOLE")};
#elif defined(__FreeBSD__) && __FreeBSD_version >= 1400032
    struct spacectl_range r{}; r.r_offset = off_t(offset); r.r_len = off_t(length);
    if (::fspacectl(f.native(), SPACECTL_DEALLOC, &r, 0, nullptr) == 0) return {};
    if (errno == EOPNOTSUPP || errno == ENODEV) return {{}, unsupported("punch_hole (no holes on this file system)")};
    return {{}, last_error("fspacectl")};
#else
    (void)f; (void)offset; return {{}, unsupported("punch_hole")};
#endif
}
// The parts of `f` that hold data, in order; the gaps between them (and after the last one) are holes that read
// as zeros. Linux, macOS, FreeBSD, Android: lseek SEEK_DATA / SEEK_HOLE. Windows:
// FSCTL_QUERY_ALLOCATED_RANGES. A file system without hole reporting yields one range covering the file. The
// granularity is the file system's (whole blocks or clusters).
struct file_range { std::uint64_t offset = 0, length = 0; };
inline result<std::vector<file_range>> data_ranges(file& f) {
    std::vector<file_range> out;
    auto sz = f.size(); if (!sz) return {{}, sz.err};
    if (!sz.value) return {std::move(out), {}};
#if defined(FORGE_OS_WINDOWS)
    FILE_ALLOCATED_RANGE_BUFFER in{}; in.FileOffset.QuadPart = 0; in.Length.QuadPart = LONGLONG(sz.value);
    std::vector<FILE_ALLOCATED_RANGE_BUFFER> buf(64);
    for (;;) {
        DWORD got = 0;
        BOOL ok = DeviceIoControl(f.native(), FSCTL_QUERY_ALLOCATED_RANGES, &in, sizeof in, buf.data(), DWORD(buf.size() * sizeof buf[0]), &got, nullptr);
        const DWORD e = ok ? 0 : GetLastError();
        if (!ok && e != ERROR_MORE_DATA) {
            if (out.empty() && (e == ERROR_INVALID_FUNCTION || e == ERROR_NOT_SUPPORTED)) { out.push_back({0, sz.value}); return {std::move(out), {}}; }
            SetLastError(e); return {{}, last_error("FSCTL_QUERY_ALLOCATED_RANGES")};
        }
        const std::size_t n = got / sizeof buf[0];
        for (std::size_t i = 0; i < n; ++i) out.push_back({std::uint64_t(buf[i].FileOffset.QuadPart), std::uint64_t(buf[i].Length.QuadPart)});
        if (ok || !n) break;
        const std::uint64_t next = out.back().offset + out.back().length;
        in.FileOffset.QuadPart = LONGLONG(next); in.Length.QuadPart = LONGLONG(sz.value - next);
    }
    return {std::move(out), {}};
#elif defined(SEEK_DATA) && defined(SEEK_HOLE)
    const int fd = f.native(); const off_t keep = ::lseek(fd, 0, SEEK_CUR);
    off_t pos = 0; const off_t end = off_t(sz.value);
    while (pos < end) {
        off_t d = ::lseek(fd, pos, SEEK_DATA);
        if (d < 0) { if (errno == ENXIO) break; out.assign(1, {0, sz.value}); break; }   // ENXIO: only a hole remains; else: no hole reporting
        off_t h = ::lseek(fd, d, SEEK_HOLE);
        if (h < 0 || h <= d) h = end;
        out.push_back({std::uint64_t(d), std::uint64_t(h - d)}); pos = h;
    }
    if (keep >= 0) ::lseek(fd, keep, SEEK_SET);
    return {std::move(out), {}};
#else
    out.push_back({0, sz.value}); return {std::move(out), {}};
#endif
}

// ---------------------------------------------------------------- file identity
// Two paths name the same file exactly when device and inode match (Windows: volume serial number and file
// index). links is the hard-link count. follow = false describes a symbolic link itself.
struct file_identity {
    std::uint64_t device = 0, inode = 0, links = 0;
    bool operator==(const file_identity& o) const { return device == o.device && inode == o.inode; }
};
inline result<file_identity> file_id(const std::string& p, bool follow = true) {
#if defined(FORGE_OS_WINDOWS)
    HANDLE h = CreateFileW(wpath(p).c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                           FILE_FLAG_BACKUP_SEMANTICS | (follow ? 0 : FILE_FLAG_OPEN_REPARSE_POINT), nullptr);
    if (h == INVALID_HANDLE_VALUE) return {{}, last_error("file_id " + p)};
    BY_HANDLE_FILE_INFORMATION i{}; BOOL ok = GetFileInformationByHandle(h, &i);
    auto e = ok ? error{} : last_error("file_id " + p); CloseHandle(h);
    if (!ok) return {{}, e};
    return {{i.dwVolumeSerialNumber, (std::uint64_t(i.nFileIndexHigh) << 32) | i.nFileIndexLow, i.nNumberOfLinks}, {}};
#else
    struct stat st{};
    if ((follow ? ::stat(p.c_str(), &st) : ::lstat(p.c_str(), &st)) != 0) return {{}, last_error("file_id " + p)};
    return {{std::uint64_t(st.st_dev), std::uint64_t(st.st_ino), std::uint64_t(st.st_nlink)}, {}};
#endif
}
inline bool same_file(const std::string& a, const std::string& b) { auto x = file_id(a), y = file_id(b); return x && y && x.value == y.value; }

// ---------------------------------------------------------------- extended attributes
// Small named values attached to a file. Linux / Android: the "user." namespace (the prefix is added and
// removed here, so names are the same everywhere). macOS / iOS: extended attributes. FreeBSD: the user
// namespace (extattr). Windows: NTFS alternate data streams ("file:name"). A missing attribute is an error
// (ENODATA / ENOATTR / ERROR_FILE_NOT_FOUND); a file system without them (tmpfs before Linux 6.6, FAT, some
// network mounts) returns unsupported.
namespace detail {
inline error xattr_error(const std::string& what) {
#if defined(FORGE_OS_POSIX)
    if (errno == ENOTSUP || errno == EOPNOTSUPP) return unsupported("extended attributes on this file system");
#endif
    return last_error(what);
}
#if defined(__linux__)
inline std::string xname(const std::string& n) { return "user." + n; }
#endif
}
inline status xattr_set(const std::string& path, const std::string& name, std::string_view value) {
    if (name.empty() || name.find_first_of(":/\\") != std::string::npos) return {{}, errno_error(EINVAL, "xattr: bad name")};
#if defined(FORGE_OS_WINDOWS)
    HANDLE h = CreateFileW(wpath(path + ":" + name).c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) { if (GetLastError() == ERROR_INVALID_NAME || GetLastError() == ERROR_INVALID_PARAMETER) return {{}, unsupported("alternate data streams on this file system")}; return {{}, last_error("xattr set " + path)}; }
    file f(h); return f.write_all(value.data(), value.size());
#elif defined(__linux__)
    if (::setxattr(path.c_str(), detail::xname(name).c_str(), value.data(), value.size(), 0) != 0) return {{}, detail::xattr_error("setxattr " + path)};
    return {};
#elif defined(__APPLE__)
    if (::setxattr(path.c_str(), name.c_str(), value.data(), value.size(), 0, 0) != 0) return {{}, detail::xattr_error("setxattr " + path)};
    return {};
#elif defined(__FreeBSD__)
    if (::extattr_set_file(path.c_str(), EXTATTR_NAMESPACE_USER, name.c_str(), value.data(), value.size()) < 0) return {{}, detail::xattr_error("extattr_set_file " + path)};
    return {};
#else
    (void)path; (void)value; return {{}, unsupported("extended attributes")};
#endif
}
inline result<std::string> xattr_get(const std::string& path, const std::string& name) {
    if (name.empty() || name.find_first_of(":/\\") != std::string::npos) return {{}, errno_error(EINVAL, "xattr: bad name")};
#if defined(FORGE_OS_WINDOWS)
    if (!os::exists(path)) return {{}, {int(ERROR_FILE_NOT_FOUND), "xattr get " + path + ": no such file"}};
    return os::read_file(path + ":" + name);
#elif defined(FORGE_OS_POSIX) && (defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__))
    for (int attempt = 0; attempt < 4; ++attempt) {   // the size can change between the two calls
#if defined(__linux__)
        ssize_t n = ::getxattr(path.c_str(), detail::xname(name).c_str(), nullptr, 0);
#elif defined(__APPLE__)
        ssize_t n = ::getxattr(path.c_str(), name.c_str(), nullptr, 0, 0, 0);
#else
        ssize_t n = ::extattr_get_file(path.c_str(), EXTATTR_NAMESPACE_USER, name.c_str(), nullptr, 0);
#endif
        if (n < 0) return {{}, detail::xattr_error("getxattr " + path)};
        std::string v(std::size_t(n), '\0');
        if (!n) return {std::move(v), {}};
#if defined(__linux__)
        ssize_t m = ::getxattr(path.c_str(), detail::xname(name).c_str(), v.data(), v.size());
#elif defined(__APPLE__)
        ssize_t m = ::getxattr(path.c_str(), name.c_str(), v.data(), v.size(), 0, 0);
#else
        ssize_t m = ::extattr_get_file(path.c_str(), EXTATTR_NAMESPACE_USER, name.c_str(), v.data(), v.size());
#endif
        if (m >= 0) { v.resize(std::size_t(m)); return {std::move(v), {}}; }
        if (errno != ERANGE) return {{}, detail::xattr_error("getxattr " + path)};
    }
    return {{}, errno_error(ERANGE, "getxattr " + path + ": value keeps changing")};
#else
    (void)path; return {{}, unsupported("extended attributes")};
#endif
}
inline status xattr_remove(const std::string& path, const std::string& name) {
    if (name.empty() || name.find_first_of(":/\\") != std::string::npos) return {{}, errno_error(EINVAL, "xattr: bad name")};
#if defined(FORGE_OS_WINDOWS)
    if (!DeleteFileW(wpath(path + ":" + name).c_str())) return {{}, last_error("xattr remove " + path)};
    return {};
#elif defined(__linux__)
    if (::removexattr(path.c_str(), detail::xname(name).c_str()) != 0) return {{}, detail::xattr_error("removexattr " + path)};
    return {};
#elif defined(__APPLE__)
    if (::removexattr(path.c_str(), name.c_str(), 0) != 0) return {{}, detail::xattr_error("removexattr " + path)};
    return {};
#elif defined(__FreeBSD__)
    if (::extattr_delete_file(path.c_str(), EXTATTR_NAMESPACE_USER, name.c_str()) != 0) return {{}, detail::xattr_error("extattr_delete_file " + path)};
    return {};
#else
    (void)path; return {{}, unsupported("extended attributes")};
#endif
}
inline result<std::vector<std::string>> xattr_list(const std::string& path) {
    std::vector<std::string> out;
#if defined(FORGE_OS_WINDOWS)
    WIN32_FIND_STREAM_DATA d{};
    HANDLE h = FindFirstStreamW(wpath(path).c_str(), FindStreamInfoStandard, &d, 0);
    if (h == INVALID_HANDLE_VALUE) {
        const DWORD e = GetLastError();
        if (e == ERROR_HANDLE_EOF) return {std::move(out), {}};   // no streams at all (a directory)
        if (e == ERROR_INVALID_PARAMETER || e == ERROR_INVALID_FUNCTION || e == ERROR_NOT_SUPPORTED) return {{}, unsupported("alternate data streams on this file system")};
        SetLastError(e); return {{}, last_error("xattr list " + path)};
    }
    do {   // ":name:$DATA"; the unnamed stream "::$DATA" is the file's contents
        std::string n = narrow(d.cStreamName);
        if (n.size() > 1 && n[0] == ':') { auto e = n.find(':', 1); n = n.substr(1, e == std::string::npos ? std::string::npos : e - 1); if (!n.empty()) out.push_back(n); }
    } while (FindNextStreamW(h, &d));
    FindClose(h);
    return {std::move(out), {}};
#elif defined(__linux__) || defined(__APPLE__)
#if defined(__linux__)
    ssize_t n = ::listxattr(path.c_str(), nullptr, 0);
#else
    ssize_t n = ::listxattr(path.c_str(), nullptr, 0, 0);
#endif
    if (n < 0) return {{}, detail::xattr_error("listxattr " + path)};
    std::string b(std::size_t(n) + 256, '\0');
#if defined(__linux__)
    n = ::listxattr(path.c_str(), b.data(), b.size());
#else
    n = ::listxattr(path.c_str(), b.data(), b.size(), 0);
#endif
    if (n < 0) return {{}, detail::xattr_error("listxattr " + path)};
    for (std::size_t i = 0; i < std::size_t(n);) {
        std::string name = b.c_str() + i; i += name.size() + 1;
#if defined(__linux__)
        if (name.rfind("user.", 0) != 0) continue;   // security.*, system.*, trusted.*: other namespaces
        name = name.substr(5);
#endif
        if (!name.empty()) out.push_back(std::move(name));
    }
    return {std::move(out), {}};
#elif defined(__FreeBSD__)
    ssize_t n = ::extattr_list_file(path.c_str(), EXTATTR_NAMESPACE_USER, nullptr, 0);
    if (n < 0) return {{}, detail::xattr_error("extattr_list_file " + path)};
    std::string b(std::size_t(n) + 256, '\0');
    n = ::extattr_list_file(path.c_str(), EXTATTR_NAMESPACE_USER, b.data(), b.size());
    if (n < 0) return {{}, detail::xattr_error("extattr_list_file " + path)};
    for (std::size_t i = 0; i < std::size_t(n);) {   // length byte, then the name (no terminator)
        std::size_t len = static_cast<unsigned char>(b[i]); ++i;
        if (i + len > std::size_t(n)) break;
        out.emplace_back(b.data() + i, len); i += len;
    }
    return {std::move(out), {}};
#else
    (void)path; return {{}, unsupported("extended attributes")};
#endif
}

// ---------------------------------------------------------------- renames that do not replace, exchanges
// Rename that fails (already_exists) instead of replacing `to`, atomically. Linux / Android: renameat2
// RENAME_NOREPLACE (3.15+). macOS / iOS: renamex_np RENAME_EXCL. Windows: MoveFileEx without REPLACE_EXISTING.
// FreeBSD, and a Linux file system without the flag: link + unlink, which is atomic for files (both names exist
// for a moment) and unsupported for directories.
inline status rename_noreplace(const std::string& from, const std::string& to) {
#if defined(FORGE_OS_WINDOWS)
    if (!MoveFileExW(wpath(from).c_str(), wpath(to).c_str(), MOVEFILE_WRITE_THROUGH)) return {{}, last_error("rename " + from + " -> " + to)};
    return {};
#else
#if defined(__linux__) && defined(SYS_renameat2)
    if (android_api_level() >= 30) {   // renameat2: Android 11
        if (::syscall(SYS_renameat2, AT_FDCWD, from.c_str(), AT_FDCWD, to.c_str(), 1u /*RENAME_NOREPLACE*/) == 0) return {};
        if (errno != ENOSYS && errno != EINVAL) return {{}, last_error("rename " + from + " -> " + to)};
    }
#elif defined(__APPLE__)
    if (::renamex_np(from.c_str(), to.c_str(), RENAME_EXCL) == 0) return {};
    if (errno != ENOTSUP) return {{}, last_error("rename " + from + " -> " + to)};
#endif
    if (::link(from.c_str(), to.c_str()) != 0) {
        if (errno == EPERM || errno == EACCES) { struct stat st{}; if (::lstat(from.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) return {{}, unsupported("rename_noreplace of a directory on this system")}; }
        return {{}, last_error("rename " + from + " -> " + to)};
    }
    if (::unlink(from.c_str()) != 0) return {{}, last_error("unlink " + from)};
    return {};
#endif
}
// Swap two existing paths atomically (files or directories). Linux / Android: renameat2 RENAME_EXCHANGE.
// macOS / iOS: renamex_np RENAME_SWAP. Windows and FreeBSD have no atomic exchange: unsupported.
inline status rename_exchange(const std::string& a, const std::string& b) {
#if defined(__linux__) && defined(SYS_renameat2)
    if (android_api_level() < 30) return {{}, unsupported("rename_exchange before Android 11")};
    if (::syscall(SYS_renameat2, AT_FDCWD, a.c_str(), AT_FDCWD, b.c_str(), 2u /*RENAME_EXCHANGE*/) == 0) return {};
    if (errno == ENOSYS || errno == EINVAL) return {{}, unsupported("rename_exchange on this kernel or file system")};
    return {{}, last_error("exchange " + a + " <-> " + b)};
#elif defined(__APPLE__)
    if (::renamex_np(a.c_str(), b.c_str(), RENAME_SWAP) == 0) return {};
    if (errno == ENOTSUP) return {{}, unsupported("rename_exchange on this file system")};
    return {{}, last_error("exchange " + a + " <-> " + b)};
#else
    (void)a; (void)b; return {{}, unsupported("rename_exchange")};
#endif
}
// Flush a directory's entries to disk, so a create / rename / remove inside it survives a power loss. POSIX:
// fsync of the directory. Windows: FlushFileBuffers on a directory handle (NTFS journals directory updates;
// this forces the log out).
inline status sync_dir(const std::string& dir) {
#if defined(FORGE_OS_WINDOWS)
    HANDLE h = CreateFileW(wpath(dir).c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) return {{}, last_error("sync_dir " + dir)};
    BOOL ok = FlushFileBuffers(h); auto e = ok ? error{} : last_error("sync_dir " + dir); CloseHandle(h);
    if (!ok) return {{}, e};
    return {};
#else
    int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return {{}, last_error("sync_dir " + dir)};
    int r; do r = ::fsync(fd); while (r != 0 && errno == EINTR);
    int en = errno; ::close(fd);
    if (r != 0 && en != EINVAL && en != EROFS) { errno = en; return {{}, last_error("sync_dir " + dir)}; }   // EINVAL: nothing to sync there
    return {};
#endif
}

// ---------------------------------------------------------------- anonymous temporary file
// A read / write file with no name that vanishes when the handle closes, even if the process is killed. Linux /
// Android: O_TMPFILE in `dir` (never visible). Other POSIX systems, and file systems without O_TMPFILE: an
// exclusive 0600 file that is unlinked at once. Windows: FILE_FLAG_DELETE_ON_CLOSE (the name exists, privately,
// until then). dir defaults to temp_dir().
inline result<file> anonymous_file(const std::string& dir = "") {
    const std::string base = dir.empty() ? temp_dir() : dir;
#if defined(FORGE_OS_WINDOWS)
    for (int i = 0; i < 16; ++i) {
        auto p = u8str(u8path(base) / u8path("forge-anon-" + random_suffix(12)));
        detail_priv::private_sa sa(false);
        HANDLE h = CreateFileW(wpath(p).c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_DELETE, sa.get(), CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        if (h != INVALID_HANDLE_VALUE) return {file(h), {}};
        if (GetLastError() != ERROR_FILE_EXISTS && GetLastError() != ERROR_ALREADY_EXISTS) return {{}, last_error("anonymous_file")};
    }
    return {{}, error{-1, "anonymous_file: could not create a unique file"}};
#else
#if defined(__linux__) && defined(O_TMPFILE)
    { int fd = ::open(base.c_str(), O_TMPFILE | O_RDWR | O_CLOEXEC, 0600); if (fd >= 0) return {file(fd), {}}; }
#endif
    for (int i = 0; i < 16; ++i) {
        auto p = u8str(u8path(base) / u8path("forge-anon-" + random_suffix(12)));
        int fd = ::open(p.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (fd >= 0) { ::unlink(p.c_str()); return {file(fd), {}}; }
        if (errno != EEXIST) return {{}, last_error("anonymous_file")};
    }
    return {{}, error{-1, "anonymous_file: could not create a unique file"}};
#endif
}

} // namespace forge::os::fs

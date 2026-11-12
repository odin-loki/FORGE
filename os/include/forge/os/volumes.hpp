#pragma once
// forge::os::volumes(): mounted storage volumes with their sizes and what kind of storage they are.
//   Linux/Android: /proc/self/mountinfo + statvfs; removable from the block device's sysfs `removable` flag;
//                  network by filesystem type (nfs, cifs, smb3, sshfs, 9p, ceph, ...). Kernel pseudo filesystems
//                  (proc, sysfs, cgroup, devpts, ...) are left out unless include_pseudo is set.
//   macOS/iOS/FreeBSD: getfsstat; network = not MNT_LOCAL; removable = MNT_REMOVABLE (macOS; FreeBSD does not
//                  say, so false there); the label is the last path component of /Volumes/<name> on macOS.
//   Windows: one volume per drive letter (GetLogicalDriveStringsW): GetDriveTypeW gives removable / network /
//                  optical / ram, GetVolumeInformationW the label and filesystem, GetDiskFreeSpaceExW the sizes.
//                  A removable drive without a medium is listed with sizes 0 (no "insert disk" dialog is shown).
// volume_for(path) finds the volume a path lives on (longest mount-point prefix).
#include <forge/os/detail.hpp>
#include <fstream>
#include <sstream>
#if defined(FORGE_OS_POSIX)
#include <sys/statvfs.h>
#endif
#if defined(__APPLE__) || defined(FORGE_OS_FREEBSD)
#include <sys/param.h>
#include <sys/mount.h>
#endif

namespace forge::os {
struct volume_info {
    std::string mount_point;   // "/", "/home", "C:\\"
    std::string device;        // "/dev/sda1", "server:/export", "\\\\?\\Volume{guid}\\"
    std::string fs_type;       // ext4, apfs, zfs, NTFS, ...
    std::string label;         // volume label where the OS reports one cheaply, else ""
    std::uint64_t total = 0, free = 0, available = 0;   // bytes; available = what an unprivileged user can still write
    bool removable = false, network = false, read_only = false, optical = false, ram = false, pseudo = false;
};
namespace detail {
inline std::string mountinfo_unescape(const std::string& s) {
    auto oct = [](char c) { return c >= '0' && c <= '7'; };
    std::string o;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 3 < s.size() && oct(s[i + 1]) && oct(s[i + 2]) && oct(s[i + 3])) { o += char(((s[i + 1] - '0') << 6) | ((s[i + 2] - '0') << 3) | (s[i + 3] - '0')); i += 3; }
        else o += s[i];
    }
    return o;
}
inline bool network_fs(const std::string& t) {
    static const char* const n[] = {"nfs", "nfs4", "cifs", "smb", "smb3", "smbfs", "sshfs", "fuse.sshfs", "9p", "ceph", "glusterfs", "fuse.glusterfs", "afs", "ncpfs", "davfs", "fuse.davfs", "lustre", "beegfs", "afpfs", "webdav", "fuse.rclone", "fuse.s3fs", "virtiofs"};
    for (auto x : n) if (t == x) return true;
    return false;
}
inline bool pseudo_fs(const std::string& t) {
    static const char* const n[] = {"proc", "sysfs", "devtmpfs", "devpts", "cgroup", "cgroup2", "securityfs", "debugfs", "tracefs", "pstore", "bpf", "configfs", "fusectl", "mqueue", "hugetlbfs", "autofs", "binfmt_misc", "rpc_pipefs", "nsfs", "efivarfs", "selinuxfs", "devfs", "fdescfs", "procfs", "linprocfs", "linsysfs", "squashfs", "overlay", "tmpfs", "ramfs", "fuse.portal", "fuse.gvfsd-fuse", "nullfs", "autofs"};
    for (auto x : n) if (t == x) return true;
    return false;
}
// One line of /proc/<pid>/mountinfo: "36 35 98:0 /root /mnt rw,noatime shared:1 - ext3 /dev/root rw,errors=continue".
inline bool parse_mountinfo_line(const std::string& line, volume_info& v) {
    std::istringstream in(line); std::string id, parent, devno, root, mp, opts, tok;
    if (!(in >> id >> parent >> devno >> root >> mp >> opts)) return false;
    while (in >> tok && tok != "-") {}
    std::string type, dev, sopts; if (tok != "-" || !(in >> type >> dev)) return false;
    in >> sopts;
    v.mount_point = mountinfo_unescape(mp); v.device = mountinfo_unescape(dev); v.fs_type = type;
    v.read_only = ("," + opts + ",").find(",ro,") != std::string::npos;
    v.network = network_fs(type); v.ram = type == "tmpfs" || type == "ramfs";
    v.pseudo = pseudo_fs(type) && !(type == "overlay" && v.mount_point == "/") && !(type == "tmpfs" && (v.mount_point == "/tmp" || v.mount_point == "/dev/shm"));
    v.optical = type == "iso9660" || type == "udf";
    return true;
}
}

inline result<std::vector<volume_info>> volumes(bool include_pseudo = false) {
    std::vector<volume_info> out;
#if defined(__linux__)
    std::ifstream f("/proc/self/mountinfo"); if (!f) return {{}, last_error("/proc/self/mountinfo")};
    std::string line;
    while (std::getline(f, line)) {
        volume_info v; if (!detail::parse_mountinfo_line(line, v) || (v.pseudo && !include_pseudo)) continue;
        bool dup = false; for (auto& x : out) dup |= x.mount_point == v.mount_point;   // an over-mount: the later line wins
        if (dup) { for (auto& x : out) if (x.mount_point == v.mount_point) x = v; continue; }
        out.push_back(std::move(v));
    }
    for (auto& v : out) {
        struct statvfs s{};
        if (!v.pseudo && ::statvfs(v.mount_point.c_str(), &s) == 0) {
            std::uint64_t fr = s.f_frsize ? s.f_frsize : s.f_bsize;
            v.total = std::uint64_t(s.f_blocks) * fr; v.free = std::uint64_t(s.f_bfree) * fr; v.available = std::uint64_t(s.f_bavail) * fr;
        }
        if (v.device.rfind("/dev/", 0) == 0) {   // /dev/sdb1 -> /sys/class/block/sdb1 (-> ../sdb)/removable
            std::string name = v.device.substr(5), sys = "/sys/class/block/" + name;
            long long r = detail::slurp_ll(sys + "/removable", -1);
            if (r < 0) { char real[4096] = {}; if (::realpath(sys.c_str(), real)) { std::string p(real); auto sl = p.rfind('/'); if (sl != std::string::npos) r = detail::slurp_ll(p.substr(0, sl) + "/removable", -1); } }
            v.removable = r == 1;
            if (name.rfind("sr", 0) == 0) v.optical = true;
        }
    }
#elif defined(__APPLE__) || defined(FORGE_OS_FREEBSD)
    int n = ::getfsstat(nullptr, 0, MNT_NOWAIT); if (n < 0) return {{}, last_error("getfsstat")};
    std::vector<struct statfs> st(std::size_t(n) + 8);
    n = ::getfsstat(st.data(), int(st.size() * sizeof(struct statfs)), MNT_NOWAIT); if (n < 0) return {{}, last_error("getfsstat")};
    for (int i = 0; i < n; ++i) {
        const auto& s = st[std::size_t(i)]; volume_info v;
        v.mount_point = s.f_mntonname; v.device = s.f_mntfromname; v.fs_type = s.f_fstypename;
        v.read_only = (s.f_flags & MNT_RDONLY) != 0; v.network = !(s.f_flags & MNT_LOCAL) || detail::network_fs(v.fs_type);
        v.pseudo = detail::pseudo_fs(v.fs_type) || v.fs_type == "autofs" || v.fs_type == "devfs";
#if defined(__APPLE__)
        v.removable = (s.f_flags & 0x00000200 /*MNT_REMOVABLE*/) != 0;
        if (s.f_flags & 0x00100000 /*MNT_DONTBROWSE*/) v.pseudo = v.pseudo || v.mount_point.rfind("/System/Volumes/", 0) == 0 || v.mount_point == "/dev";
        if (v.mount_point.rfind("/Volumes/", 0) == 0) v.label = v.mount_point.substr(9);
        v.optical = v.fs_type == "cd9660" || v.fs_type == "udf" || v.fs_type == "cddafs";
#else
        v.optical = v.fs_type == "cd9660" || v.fs_type == "udf";
        v.ram = v.fs_type == "tmpfs";
        if (v.fs_type == "tmpfs" && v.mount_point == "/tmp") v.pseudo = false;
#endif
        v.total = std::uint64_t(s.f_blocks) * std::uint64_t(s.f_bsize); v.free = std::uint64_t(s.f_bfree) * std::uint64_t(s.f_bsize);
        v.available = s.f_bavail > 0 ? std::uint64_t(s.f_bavail) * std::uint64_t(s.f_bsize) : 0;
        if (v.pseudo && !include_pseudo) continue;
        out.push_back(std::move(v));
    }
#elif defined(FORGE_OS_WINDOWS)
    (void)include_pseudo;
    wchar_t drives[1024] = {}; DWORD n = GetLogicalDriveStringsW(1023, drives);
    if (n == 0 || n > 1023) return {{}, last_error("GetLogicalDriveStrings")};
    UINT old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);   // no "there is no disk in the drive" dialog
    for (const wchar_t* p = drives; *p; p += std::wcslen(p) + 1) {
        volume_info v; v.mount_point = narrow(p);
        UINT ty = GetDriveTypeW(p);
        if (ty == DRIVE_NO_ROOT_DIR) continue;
        v.removable = ty == DRIVE_REMOVABLE; v.network = ty == DRIVE_REMOTE; v.optical = ty == DRIVE_CDROM; v.ram = ty == DRIVE_RAMDISK;
        wchar_t label[MAX_PATH + 1] = {}, fsn[MAX_PATH + 1] = {}, guid[64] = {}; DWORD flags = 0;
        if (GetVolumeInformationW(p, label, MAX_PATH, nullptr, nullptr, &flags, fsn, MAX_PATH)) { v.label = narrow(label); v.fs_type = narrow(fsn); v.read_only = (flags & FILE_READ_ONLY_VOLUME) != 0; }
        if (!v.network && GetVolumeNameForVolumeMountPointW(p, guid, 64)) v.device = narrow(guid);
        ULARGE_INTEGER avail{}, total{}, fr{};
        if (GetDiskFreeSpaceExW(p, &avail, &total, &fr)) { v.total = total.QuadPart; v.free = fr.QuadPart; v.available = avail.QuadPart; }
        if (v.optical) v.read_only = true;
        out.push_back(std::move(v));
    }
    SetErrorMode(old_mode);
#else
    (void)include_pseudo; return {{}, unsupported("volumes")};
#endif
    return {std::move(out), {}};
}

// The volume `path` is on: the listed volume whose mount point is the longest prefix of the absolute path.
inline result<volume_info> volume_for(const std::string& path) {
    auto all = volumes(true); if (!all) return {{}, all.err};
    std::string abs = absolute(path);
#if defined(FORGE_OS_WINDOWS)
    auto norm = [](std::string s) { for (auto& c : s) { if (c == '/') c = '\\'; else if (c >= 'a' && c <= 'z') c = char(c - 'a' + 'A'); } return s; };
    const char sep = '\\';
#else
    auto norm = [](std::string s) { return s; };
    const char sep = '/';
#endif
    std::string a = norm(abs); const volume_info* best = nullptr; std::size_t best_len = 0;
    for (auto& v : all.value) {
        std::string m = norm(v.mount_point); while (m.size() > 1 && m.back() == sep) m.pop_back();
        bool hit = a == m || (a.size() > m.size() && a.compare(0, m.size(), m) == 0 && (a[m.size()] == sep || m.back() == sep || m == "/"));
        if (hit && (!best || m.size() >= best_len)) { best = &v; best_len = m.size(); }
    }
    if (!best) return {{}, errno_error(ENOENT, "volume_for: no volume contains " + path)};
    return {*best, {}};
}
} // namespace forge::os

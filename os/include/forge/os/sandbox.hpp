#pragma once
// Process sandboxing (one-way: once applied it cannot be lifted).
//   Linux/Android: no_new_privs + a seccomp-BPF filter returning EPERM for the denied syscalls (network sockets
//   other than AF_UNIX and i386 socketcall, execve/execveat, file writes via open flags / creat / unlink / rename /
//   mkdir / truncate / link / symlink / mknod / chmod / chown / utimes / xattrs; io_uring whenever network or
//   writes are denied). Another architecture's syscalls kill the process, and so do x86_64 x32 calls.
//   FreeBSD: Capsicum cap_enter() (capability mode: no global namespaces at all; coarser than requested).
//   run_sandboxed() there opens the binary, the run-time linker and the library directories first and starts
//   the target with fexecve of ld-elf.so.1 -f <fd> (LD_LIBRARY_PATH_FDS), since capability mode allows no
//   lookups by path: the target runs, but only on descriptors it inherited (no files, no network, no exec).
//   OpenBSD (bonus): pledge().  macOS: sandbox_init named profiles (no-network / no-write / pure-computation);
//   App Sandbox is detected from APP_SANDBOX_CONTAINER_ID.  iOS: apps are always sandboxed; apply() reports that.
//   Windows: the current process joins a job object (ActiveProcessLimit=1 blocks child processes; network and
//   filesystem denial cannot be added to a running process, so run_sandboxed() starts children in an
//   AppContainer (no network capability unless allowed; writes only to the container folder) inside a
//   kill-on-close job with UI limits, or, for an exec-only policy, with a restricted token in the job).
#include <forge/os/detail.hpp>
#include <forge/os/process.hpp>   // find_executable, detail_proc::win_cmdline
#if defined(_WIN32)
#include <userenv.h>   // CreateAppContainerProfile (link userenv)
#if defined(_MSC_VER)
#pragma comment(lib, "userenv.lib")
#endif
#endif
#if defined(__linux__)
#include <sys/prctl.h>
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <linux/audit.h>
#include <stddef.h>
#elif defined(FORGE_OS_FREEBSD)
#include <sys/capsicum.h>
#include <elf.h>
#endif

namespace forge::os {
struct sandbox_policy { bool deny_network = false, deny_exec = false, deny_fs_write = false; };
struct sandbox_info { bool sandboxed = false; std::string mechanism; };

inline sandbox_info sandbox_status() {
    sandbox_info i;
#if defined(__linux__)
    auto s = detail::slurp("/proc/self/status"); auto p = s.find("Seccomp:");
    if (p != std::string::npos && std::atoi(s.c_str() + p + 8) > 0) { i.sandboxed = true; i.mechanism = std::atoi(s.c_str() + p + 8) == 2 ? "seccomp-bpf" : "seccomp-strict"; }
#elif defined(FORGE_OS_FREEBSD)
    unsigned m = 0; if (::cap_getmode(&m) == 0 && m) { i.sandboxed = true; i.mechanism = "capsicum"; }
#elif defined(FORGE_OS_IOS)
    i.sandboxed = true; i.mechanism = "ios-app-sandbox";
#elif defined(FORGE_OS_MACOS)
    if (!getenv("APP_SANDBOX_CONTAINER_ID").empty()) { i.sandboxed = true; i.mechanism = "app-sandbox"; }
#elif defined(FORGE_OS_WINDOWS)
    BOOL in = FALSE; if (IsProcessInJob(GetCurrentProcess(), nullptr, &in) && in) { i.sandboxed = true; i.mechanism = "job-object"; }
#endif
    return i;
}

#if defined(__linux__)
namespace detail {
#if defined(__x86_64__)
constexpr std::uint32_t audit_arch = AUDIT_ARCH_X86_64;
#elif defined(__aarch64__)
constexpr std::uint32_t audit_arch = AUDIT_ARCH_AARCH64;
#elif defined(__i386__)
constexpr std::uint32_t audit_arch = AUDIT_ARCH_I386;
#elif defined(__arm__)
constexpr std::uint32_t audit_arch = AUDIT_ARCH_ARM;
#elif defined(__riscv) && __riscv_xlen == 64
constexpr std::uint32_t audit_arch = AUDIT_ARCH_RISCV64;
#else
constexpr std::uint32_t audit_arch = 0;
#endif
// The BPF program for `p` (built in the caller, so a forked child only has to install it).
inline std::vector<sock_filter> seccomp_filter(const sandbox_policy& p) {
    std::vector<sock_filter> f;
    auto st = [&](std::uint16_t code, std::uint32_t k, std::uint8_t jt = 0, std::uint8_t jf = 0) { f.push_back(sock_filter{code, jt, jf, k}); };
    const std::uint32_t deny = SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA), allow = SECCOMP_RET_ALLOW;
    st(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, arch));
    st(BPF_JMP | BPF_JEQ | BPF_K, audit_arch, 1, 0); st(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS);
    auto load_nr = [&] { st(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)); };
    auto deny_nr = [&](long nr) { load_nr(); st(BPF_JMP | BPF_JEQ | BPF_K, std::uint32_t(nr), 0, 1); st(BPF_RET | BPF_K, deny); };
#if defined(__x86_64__)
    // x32 ABI calls share AUDIT_ARCH_X86_64 but carry bit 30 in nr (0x40000000 + the x32 number): every rule
    // below would miss them, so any such call kills the process.
    load_nr(); st(BPF_JMP | BPF_JGE | BPF_K, 0x40000000u, 0, 1); st(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS);
#endif
    if (p.deny_network || p.deny_fs_write) {
        // io_uring performs socket/connect/openat/unlinkat/... inside the kernel without those syscalls, so the
        // rules below never see them. Numbers 425-427 are shared by every architecture (since Linux 5.1). A ring
        // set up before the sandbox stops working too (io_uring_enter is denied): os::async_file reports EPERM.
        for (long nr : {425L /*io_uring_setup*/, 426L /*io_uring_enter*/, 427L /*io_uring_register*/}) deny_nr(nr);
    }
    // deny when (nr == NR) && (args[argi] & mask)
    auto deny_flags = [&](long nr, int argi, std::uint32_t mask) {
        load_nr(); st(BPF_JMP | BPF_JEQ | BPF_K, std::uint32_t(nr), 0, 3);
        st(BPF_LD | BPF_W | BPF_ABS, std::uint32_t(offsetof(seccomp_data, args) + std::size_t(argi) * 8));
        st(BPF_JMP | BPF_JSET | BPF_K, mask, 0, 1); st(BPF_RET | BPF_K, deny);
    };
    if (p.deny_network) {   // socket(domain != AF_UNIX)
        load_nr(); st(BPF_JMP | BPF_JEQ | BPF_K, std::uint32_t(__NR_socket), 0, 3);
        st(BPF_LD | BPF_W | BPF_ABS, std::uint32_t(offsetof(seccomp_data, args)));
        st(BPF_JMP | BPF_JEQ | BPF_K, AF_UNIX, 1, 0); st(BPF_RET | BPF_K, deny);
#ifdef __NR_socketcall
        deny_nr(__NR_socketcall);   // i386 multiplexer: its arguments sit in user memory the filter cannot read
#endif
    }
    if (p.deny_exec) {
        deny_nr(__NR_execve);
#ifdef __NR_execveat
        deny_nr(__NR_execveat);
#endif
    }
    if (p.deny_fs_write) {
        const std::uint32_t wr = O_WRONLY | O_RDWR | O_CREAT | O_TRUNC | O_APPEND;
#ifdef __NR_open
        deny_flags(__NR_open, 1, wr);
#endif
#ifdef __NR_creat
        deny_nr(__NR_creat);
#endif
        deny_flags(__NR_openat, 2, wr);
#ifdef __NR_openat2
        deny_nr(__NR_openat2);   // flags live in a struct the filter cannot read
#endif
        for (long nr : {long(__NR_unlinkat), long(__NR_mkdirat), long(__NR_truncate), long(__NR_ftruncate)}) deny_nr(nr);
#ifdef __NR_renameat   // absent on riscv64 (renameat2 only)
        deny_nr(__NR_renameat);
#endif
#ifdef __NR_unlink
        deny_nr(__NR_unlink); deny_nr(__NR_rename); deny_nr(__NR_mkdir); deny_nr(__NR_rmdir);
#endif
#ifdef __NR_renameat2
        deny_nr(__NR_renameat2);
#endif
        // links, device nodes, permission / owner / time / extended-attribute changes
        for (long nr : {long(__NR_linkat), long(__NR_symlinkat), long(__NR_mknodat), long(__NR_fchmodat), long(__NR_fchmod), long(__NR_fchownat),
                        long(__NR_fchown), long(__NR_utimensat), long(__NR_setxattr), long(__NR_lsetxattr), long(__NR_fsetxattr),
                        long(__NR_removexattr), long(__NR_lremovexattr), long(__NR_fremovexattr)}) deny_nr(nr);
        deny_nr(452);   // fchmodat2 (Linux 6.6, same number everywhere; older headers lack __NR_fchmodat2)
#ifdef __NR_link   // legacy non-at calls (x86, arm; absent on aarch64 / riscv64)
        deny_nr(__NR_link); deny_nr(__NR_symlink); deny_nr(__NR_mknod); deny_nr(__NR_chmod); deny_nr(__NR_chown); deny_nr(__NR_lchown);
#endif
#ifdef __NR_utime
        deny_nr(__NR_utime);
#endif
#ifdef __NR_utimes
        deny_nr(__NR_utimes);
#endif
#ifdef __NR_futimesat
        deny_nr(__NR_futimesat);
#endif
#ifdef __NR_chown32   // i386 / arm 32-bit uid variants
        deny_nr(__NR_chown32); deny_nr(__NR_lchown32); deny_nr(__NR_fchown32);
#endif
#ifdef __NR_utimensat_time64
        deny_nr(__NR_utimensat_time64);
#endif
#ifdef __NR_truncate64
        deny_nr(__NR_truncate64); deny_nr(__NR_ftruncate64);
#endif
    }
    st(BPF_RET | BPF_K, allow);
    return f;
}
// Install a filter: two prctl calls, async-signal-safe (run_sandboxed builds the filter before fork).
inline bool install_seccomp(const std::vector<sock_filter>& f) noexcept {
    if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) return false;
    sock_fprog prog{static_cast<unsigned short>(f.size()), const_cast<sock_filter*>(f.data())};
    return ::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog, 0, 0) == 0;
}
inline status apply_seccomp(const sandbox_policy& p) {
    if (!install_seccomp(seccomp_filter(p))) return {{}, last_error("seccomp (PR_SET_NO_NEW_PRIVS / PR_SET_SECCOMP)")};
    return {};
}
}
#endif

inline status apply_sandbox(const sandbox_policy& p) {
    if (!p.deny_network && !p.deny_exec && !p.deny_fs_write) return {};
#if defined(__linux__)
    if (!detail::audit_arch) return {{}, unsupported("seccomp on this architecture")};
    return detail::apply_seccomp(p);
#elif defined(FORGE_OS_FREEBSD)
    if (::cap_enter()) return {{}, last_error("cap_enter")}; return {};
#elif defined(__OpenBSD__)
    std::string pr = "stdio rpath flock"; if (!p.deny_fs_write) pr += " wpath cpath"; if (!p.deny_network) pr += " inet unix dns"; if (!p.deny_exec) pr += " proc exec";
    if (::pledge(pr.c_str(), nullptr)) return {{}, last_error("pledge")}; return {};
#elif defined(FORGE_OS_MACOS)
    auto init = reinterpret_cast<int (*)(const char*, std::uint64_t, char**)>(::dlsym(RTLD_DEFAULT, "sandbox_init"));
    if (!init) return {{}, unsupported("sandbox_init")};
    const char* prof = (p.deny_network && p.deny_fs_write) || p.deny_exec ? "pure-computation" : p.deny_network ? "no-internet" : "no-write";
    char* err = nullptr; if (init(prof, 1 /*SANDBOX_NAMED*/, &err)) { std::string e = err ? err : "sandbox_init failed"; return {{}, {1, e}}; }
    return {};
#elif defined(FORGE_OS_IOS)
    return {};   // already confined by the iOS app sandbox (no exec, container-only fs); nothing further to apply
#elif defined(FORGE_OS_WINDOWS)
    if (p.deny_network || p.deny_fs_write) return {{}, unsupported("in-process network/fs denial on Windows (use run_sandboxed)")};
    HANDLE j = CreateJobObjectW(nullptr, nullptr); if (!j) return {{}, last_error("CreateJobObject")};
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{}; li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_ACTIVE_PROCESS; li.BasicLimitInformation.ActiveProcessLimit = 1;
    if (!SetInformationJobObject(j, JobObjectExtendedLimitInformation, &li, sizeof li) || !AssignProcessToJobObject(j, GetCurrentProcess())) { auto e = last_error("job object"); CloseHandle(j); return {{}, e}; }
    return {};   // the job handle is intentionally leaked: it must live as long as the process
#else
    return {{}, unsupported("sandbox")};
#endif
}

// Run a child under the policy; returns its exit code (POSIX: 126 = the policy could not be applied, 127 = the
// target could not be started, 128+n = killed by signal n). argv[0] is resolved like spawn_wait (PATH).
// deny_exec: Windows (ActiveProcessLimit = 1) and FreeBSD (capability mode) keep the target itself from
// starting programs; Linux and macOS cannot deny exec yet allow the one that starts the target, so there the
// target runs under the policy's other parts only (Landlock is planned).
inline result<int> run_sandboxed(const std::vector<std::string>& argv, const sandbox_policy& p) {
    if (argv.empty()) return {-1, {22, "run_sandboxed: empty argv"}};
#if defined(FORGE_OS_WINDOWS)
    std::string exe = find_executable(argv[0]);
    if (exe.empty()) return {-1, {ERROR_FILE_NOT_FOUND, "run_sandboxed " + argv[0] + ": not found"}};
    error cmd_err{};
    std::wstring cmd = detail_proc::win_cmdline(argv, exe, &cmd_err), wexe = widen(exe);   // same quoting as spawn_wait
    if (cmd.empty()) return {-1, cmd_err};
    // Job object for every child: kill-on-close, no breakaway, UI limits; ActiveProcessLimit=1 when exec is denied.
    HANDLE j = CreateJobObjectW(nullptr, nullptr); if (!j) return {-1, last_error("CreateJobObject")};
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
    li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION | (p.deny_exec ? JOB_OBJECT_LIMIT_ACTIVE_PROCESS : 0);
    li.BasicLimitInformation.ActiveProcessLimit = 1;
    JOBOBJECT_BASIC_UI_RESTRICTIONS ui{}; ui.UIRestrictionsClass = JOB_OBJECT_UILIMIT_DESKTOP | JOB_OBJECT_UILIMIT_EXITWINDOWS | JOB_OBJECT_UILIMIT_GLOBALATOMS | JOB_OBJECT_UILIMIT_HANDLES | JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS;
    if (!SetInformationJobObject(j, JobObjectExtendedLimitInformation, &li, sizeof li) || !SetInformationJobObject(j, JobObjectBasicUIRestrictions, &ui, sizeof ui)) {
        auto e = last_error("SetInformationJobObject"); CloseHandle(j); return {-1, e};
    }
    PROCESS_INFORMATION pi{};
    if (p.deny_fs_write || p.deny_network) {
        // AppContainer: the child gets a per-policy container SID. With no capabilities it has no network
        // (internetClient / privateNetworkClientServer are not granted) and can write only to its own
        // container folder (%LOCALAPPDATA%\Packages\<name>\AC); reads of user files are denied unless their ACL
        // grants ALL APPLICATION PACKAGES (system directories do, so programs still load).
        const wchar_t* name = p.deny_network && p.deny_fs_write ? L"forge.sandbox.nonet.nofs" : p.deny_network ? L"forge.sandbox.nonet" : L"forge.sandbox.nofs";
        PSID ac = nullptr;
        HRESULT hr = CreateAppContainerProfile(name, name, L"FORGE sandboxed child", nullptr, 0, &ac);
        if (hr == HRESULT_FROM_WIN32(ERROR_ALREADY_EXISTS)) hr = DeriveAppContainerSidFromAppContainerName(name, &ac);
        if (FAILED(hr) || !ac) { CloseHandle(j); return {-1, {int(hr), "AppContainer profile failed"}}; }
        SID_AND_ATTRIBUTES caps[1]{}; DWORD ncaps = 0; PSID net = nullptr;
        if (!p.deny_network) {   // fs denied only: keep outbound network
            SID_IDENTIFIER_AUTHORITY app = SECURITY_APP_PACKAGE_AUTHORITY;
            if (AllocateAndInitializeSid(&app, SECURITY_BUILTIN_CAPABILITY_RID_COUNT, SECURITY_CAPABILITY_BASE_RID, SECURITY_CAPABILITY_INTERNET_CLIENT, 0, 0, 0, 0, 0, 0, &net)) { caps[0].Sid = net; caps[0].Attributes = SE_GROUP_ENABLED; ncaps = 1; }
        }
        SECURITY_CAPABILITIES sc{}; sc.AppContainerSid = ac; sc.Capabilities = ncaps ? caps : nullptr; sc.CapabilityCount = ncaps;
        SIZE_T sz = 0; InitializeProcThreadAttributeList(nullptr, 1, 0, &sz);
        std::vector<unsigned char> al(sz); auto* attrs = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(al.data());
        bool ok = InitializeProcThreadAttributeList(attrs, 1, 0, &sz) && UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES, &sc, sizeof sc, nullptr, nullptr);
        STARTUPINFOEXW si{}; si.StartupInfo.cb = sizeof si; si.lpAttributeList = attrs;
        if (ok) ok = CreateProcessW(wexe.c_str(), cmd.data(), nullptr, nullptr, FALSE, EXTENDED_STARTUPINFO_PRESENT | CREATE_SUSPENDED, nullptr, nullptr, &si.StartupInfo, &pi);
        error e{}; if (!ok) e = last_error("CreateProcess (AppContainer)");
        if (attrs) DeleteProcThreadAttributeList(attrs);
        if (net) FreeSid(net);
        FreeSid(ac);
        if (!ok) { CloseHandle(j); return {-1, e}; }
    } else {
        // exec-only policy: restricted token (max privileges dropped) inside the job.
        HANDLE tok = nullptr, rt = nullptr;
        if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ALL_ACCESS, &tok)) { CloseHandle(j); return {-1, last_error("OpenProcessToken")}; }
        BOOL ok = CreateRestrictedToken(tok, DISABLE_MAX_PRIVILEGE, 0, nullptr, 0, nullptr, 0, nullptr, &rt); CloseHandle(tok);
        if (!ok) { CloseHandle(j); return {-1, last_error("CreateRestrictedToken")}; }
        STARTUPINFOW si{}; si.cb = sizeof si;
        if (!CreateProcessAsUserW(rt, wexe.c_str(), cmd.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED, nullptr, nullptr, &si, &pi)) {
            auto e = last_error("CreateProcessAsUser"); CloseHandle(rt); CloseHandle(j); return {-1, e};
        }
        CloseHandle(rt);
    }
    if (!AssignProcessToJobObject(j, pi.hProcess)) { auto e = last_error("AssignProcessToJobObject"); TerminateProcess(pi.hProcess, 1); CloseHandle(pi.hThread); CloseHandle(pi.hProcess); CloseHandle(j); return {-1, e}; }
    ResumeThread(pi.hThread); WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1; GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread); CloseHandle(pi.hProcess); CloseHandle(j); return {int(code), {}};
#elif defined(FORGE_OS_IOS)
    (void)p; return {-1, unsupported("child processes on iOS")};
#else
    // Everything is prepared before fork; the child makes only async-signal-safe calls (apart from macOS
    // sandbox_init, which has no such variant).
    const std::string exe = find_executable(argv[0]);
    if (exe.empty()) return {-1, {ENOENT, "run_sandboxed " + argv[0] + ": command not found"}};
    std::vector<char*> a; for (auto& s : argv) a.push_back(const_cast<char*>(s.c_str())); a.push_back(nullptr);
    const bool confine = p.deny_network || p.deny_exec || p.deny_fs_write;
#if defined(__linux__)
    // deny_exec is left out: the filter would refuse the execve that starts the target.
    sandbox_policy q = p; q.deny_exec = false;
    if (confine && !detail::audit_arch) return {-1, unsupported("seccomp on this architecture")};
    const std::vector<sock_filter> filter = q.deny_network || q.deny_fs_write ? detail::seccomp_filter(q) : std::vector<sock_filter>{};
#elif defined(FORGE_OS_FREEBSD)
    // Capability mode refuses every lookup by path, execve included, so the child opens nothing itself: the
    // binary (and for a dynamic one the run-time linker named by PT_INTERP plus the default library
    // directories) is opened here, the child calls cap_enter() and fexecve()s - a static binary directly, a
    // dynamic one as `ld-elf.so.1 -f <fd> -- argv...` with LD_LIBRARY_PATH_FDS (the kernel would otherwise
    // have to open the interpreter by path, and the exec fails past the point of no return). Scripts (#!)
    // cannot start this way; libraries outside /lib, /usr/lib, /usr/lib/compat, /usr/local/lib are not found.
    int bin = -1, interp = -1; std::vector<int> dirs; std::vector<std::string> xargs, xenv; std::vector<char*> xa, xe;
    auto close_all = [&] { for (int f : dirs) ::close(f); if (bin >= 0) ::close(bin); if (interp >= 0) ::close(interp); };
    if (confine) {
        if ((bin = ::open(exe.c_str(), O_RDONLY)) < 0) return {-1, last_error("open " + exe)};
        Elf_Ehdr eh{}; std::string interp_path;
        if (::pread(bin, &eh, sizeof eh, 0) != ssize_t(sizeof eh) || std::memcmp(eh.e_ident, ELFMAG, SELFMAG) != 0) {
            close_all(); return {-1, {ENOEXEC, "run_sandboxed " + exe + ": not an ELF binary (a script cannot start in capability mode)"}};
        }
        for (unsigned i = 0; i < eh.e_phnum; ++i) {
            Elf_Phdr ph{};
            if (::pread(bin, &ph, sizeof ph, off_t(eh.e_phoff + std::uint64_t(i) * eh.e_phentsize)) != ssize_t(sizeof ph)) break;
            if (ph.p_type != PT_INTERP || ph.p_filesz < 2 || ph.p_filesz > 4096) continue;
            interp_path.resize(std::size_t(ph.p_filesz));
            if (::pread(bin, interp_path.data(), interp_path.size(), off_t(ph.p_offset)) != ssize_t(interp_path.size())) interp_path.clear();
            interp_path.resize(::strnlen(interp_path.c_str(), interp_path.size()));
            break;
        }
        if (!interp_path.empty()) {
            if ((interp = ::open(interp_path.c_str(), O_RDONLY | O_CLOEXEC)) < 0) { auto e = last_error("open " + interp_path); close_all(); return {-1, e}; }
            std::string fds;
            for (const char* d : {"/lib", "/usr/lib", "/usr/lib/compat", "/usr/local/lib"})
                if (int f = ::open(d, O_RDONLY | O_DIRECTORY); f >= 0) { dirs.push_back(f); if (!fds.empty()) fds += ':'; fds += std::to_string(f); }
            xargs = {"ld-elf.so.1", "-f", std::to_string(bin), "--"}; xargs.insert(xargs.end(), argv.begin(), argv.end());
            for (char** e = environ; e && *e; ++e) if (std::strncmp(*e, "LD_LIBRARY_PATH_FDS=", 20) != 0) xenv.emplace_back(*e);
            xenv.push_back("LD_LIBRARY_PATH_FDS=" + fds);
            for (auto& s : xargs) xa.push_back(s.data()); xa.push_back(nullptr);
            for (auto& s : xenv) xe.push_back(s.data()); xe.push_back(nullptr);
        } else {
            ::fcntl(bin, F_SETFD, FD_CLOEXEC);   // static: the kernel needs no path, the target no descriptor
        }
    }
#elif defined(FORGE_OS_MACOS)
    sandbox_policy q = p; q.deny_exec = false;   // see above: the target must still be exec'd
    (void)confine;
#else
    (void)confine;
#endif
    pid_t c = forge::os::serialized_fork();
    if (c < 0) {
        auto e = last_error("fork");
#if defined(FORGE_OS_FREEBSD)
        close_all();
#endif
        return {-1, e};
    }
    if (c == 0) {
#if defined(__linux__)
        if (!filter.empty() && !detail::install_seccomp(filter)) ::_exit(126);
#elif defined(FORGE_OS_FREEBSD)
        if (confine) {
            if (::cap_enter()) ::_exit(126);
            if (interp >= 0) ::fexecve(interp, xa.data(), xe.data()); else ::fexecve(bin, a.data(), environ);
            ::_exit(127);
        }
#elif defined(FORGE_OS_MACOS)
        if (!apply_sandbox(q)) ::_exit(126);
#else
        if (!apply_sandbox(p)) ::_exit(126);
#endif
        ::execv(exe.c_str(), a.data()); ::_exit(127);
    }
#if defined(FORGE_OS_FREEBSD)
    close_all();
#endif
    int st = 0; while (::waitpid(c, &st, 0) < 0 && errno == EINTR) {}
    return {WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st), {}};
#endif
}
} // namespace forge::os

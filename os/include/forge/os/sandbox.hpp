#pragma once
// Process sandboxing (one-way: once applied it cannot be lifted).
//   Linux/Android: no_new_privs + a seccomp-BPF filter returning EPERM for the denied syscalls (network sockets
//   other than AF_UNIX, execve/execveat, file writes via open flags / unlink / rename / mkdir).
//   FreeBSD: Capsicum cap_enter() (capability mode: no global namespaces at all; coarser than requested).
//   OpenBSD (bonus): pledge().  macOS: sandbox_init named profiles (no-network / no-write / pure-computation);
//   App Sandbox is detected from APP_SANDBOX_CONTAINER_ID.  iOS: apps are always sandboxed; apply() reports that.
//   Windows: the current process joins a job object (ActiveProcessLimit=1 blocks child processes; network and
//   filesystem denial cannot be added to a running process, so run_sandboxed() starts children in an
//   AppContainer (no network capability unless allowed; writes only to the container folder) inside a
//   kill-on-close job with UI limits, or, for an exec-only policy, with a restricted token in the job).
#include <forge/os/detail.hpp>
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
inline status apply_seccomp(const sandbox_policy& p) {
    std::vector<sock_filter> f;
    auto st = [&](std::uint16_t code, std::uint32_t k, std::uint8_t jt = 0, std::uint8_t jf = 0) { f.push_back(sock_filter{code, jt, jf, k}); };
    const std::uint32_t deny = SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA), allow = SECCOMP_RET_ALLOW;
    st(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, arch));
    st(BPF_JMP | BPF_JEQ | BPF_K, audit_arch, 1, 0); st(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS);
    auto load_nr = [&] { st(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)); };
    auto deny_nr = [&](long nr) { load_nr(); st(BPF_JMP | BPF_JEQ | BPF_K, std::uint32_t(nr), 0, 1); st(BPF_RET | BPF_K, deny); };
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
    }
    st(BPF_RET | BPF_K, allow);
    if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) return {{}, last_error("PR_SET_NO_NEW_PRIVS")};
    sock_fprog prog{static_cast<unsigned short>(f.size()), f.data()};
    if (::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog, 0, 0)) return {{}, last_error("PR_SET_SECCOMP")};
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

// Run a child under the policy; returns its exit code.
inline result<int> run_sandboxed(const std::vector<std::string>& argv, const sandbox_policy& p) {
    if (argv.empty()) return {-1, {22, "run_sandboxed: empty argv"}};
#if defined(FORGE_OS_WINDOWS)
    // Job object for every child: kill-on-close, no breakaway, UI limits; ActiveProcessLimit=1 when exec is denied.
    HANDLE j = CreateJobObjectW(nullptr, nullptr); if (!j) return {-1, last_error("CreateJobObject")};
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
    li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION | (p.deny_exec ? JOB_OBJECT_LIMIT_ACTIVE_PROCESS : 0);
    li.BasicLimitInformation.ActiveProcessLimit = 1; SetInformationJobObject(j, JobObjectExtendedLimitInformation, &li, sizeof li);
    JOBOBJECT_BASIC_UI_RESTRICTIONS ui{}; ui.UIRestrictionsClass = JOB_OBJECT_UILIMIT_DESKTOP | JOB_OBJECT_UILIMIT_EXITWINDOWS | JOB_OBJECT_UILIMIT_GLOBALATOMS | JOB_OBJECT_UILIMIT_HANDLES | JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS;
    SetInformationJobObject(j, JobObjectBasicUIRestrictions, &ui, sizeof ui);
    std::wstring cmd; for (auto& a : argv) { if (!cmd.empty()) cmd += L' '; cmd += L'"' + widen(a) + L'"'; }
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
        if (ok) ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, EXTENDED_STARTUPINFO_PRESENT | CREATE_SUSPENDED, nullptr, nullptr, &si.StartupInfo, &pi);
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
        if (!CreateProcessAsUserW(rt, nullptr, cmd.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED, nullptr, nullptr, &si, &pi)) {
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
    pid_t c = forge::os::serialized_fork(); if (c < 0) return {-1, last_error("fork")};
    if (c == 0) {
        sandbox_policy q = p; q.deny_exec = false;   // exec the target first; the filter is inherited across exec
#if defined(__linux__)
        if (p.deny_exec) q.deny_exec = false;
#endif
        if (!apply_sandbox(q)) ::_exit(126);
        std::vector<char*> a; for (auto& s : argv) a.push_back(const_cast<char*>(s.c_str())); a.push_back(nullptr);
        ::execvp(a[0], a.data()); ::_exit(127);
    }
    int st = 0; while (::waitpid(c, &st, 0) < 0 && errno == EINTR) {}
    return {WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st), {}};
#endif
}
} // namespace forge::os

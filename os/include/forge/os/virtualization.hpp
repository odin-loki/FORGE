#pragma once
// forge::os::virtualization(): is this process inside a virtual machine, a container, a binary translator or an
// emulator? Read-only probes, nothing privileged:
//   hypervisor   x86: the CPUID hypervisor bit and vendor leaf 0x40000000 (kvm, hyperv, vmware, xen, virtualbox,
//                qemu, bhyve, parallels, acrn, qnx).  Otherwise / in addition: Linux DMI (/sys/class/dmi/id),
//                /sys/hypervisor/type and the device tree; macOS kern.hv_vmm_present + hw.model; FreeBSD
//                kern.vm_guest; Windows the BIOS registry key.
//   container    Linux: /.dockerenv, /run/.containerenv (podman), /run/systemd/container, the `container`
//                variable, cgroup paths (docker, kubepods, lxc), Kubernetes and Flatpak / Snap markers; WSL 1 and 2
//                are reported as "wsl".  FreeBSD: security.jail.jailed -> "jail".  Windows: the ContainerType
//                value of a Windows container -> "windows".
//   translated   macOS: Rosetta 2 (sysctl.proc_translated).  Windows: x64 or x86 code on ARM64 (IsWow64Process2),
//                and wine (ntdll wine_get_version).  Linux: none detectable (qemu-user leaves no reliable mark).
//   emulator     Android: the emulator properties (ro.kernel.qemu, goldfish / ranchu hardware, sdk_gphone models).
//                iOS: a simulator build (TARGET_OS_SIMULATOR) or SIMULATOR_DEVICE_NAME.
#include <forge/os/detail.hpp>
#include <forge/os/cpu_topology.hpp>
#include <fstream>

namespace forge::os {
struct virt_info {
    std::string hypervisor;   // "" on bare metal (or undetected)
    std::string container;    // "" outside a container
    std::string translator;   // "rosetta", "wine", "arm64-emulation" or ""
    bool emulator = false;    // a device emulator / simulator (mobile)
    bool virtualized() const { return !hypervisor.empty(); }
    bool containerized() const { return !container.empty(); }
    bool translated() const { return !translator.empty(); }
};
namespace detail {
// The 12-byte hypervisor vendor of CPUID leaf 0x40000000 -> a short name ("" when unknown).
inline std::string hypervisor_from_cpuid_vendor(const std::string& v) {
    static const std::pair<const char*, const char*> known[] = {
        {"KVMKVMKVM", "kvm"}, {"Linux KVM Hv", "kvm"}, {"Microsoft Hv", "hyperv"}, {"VMwareVMware", "vmware"}, {"XenVMMXenVMM", "xen"},
        {"VBoxVBoxVBox", "virtualbox"}, {"TCGTCGTCGTCG", "qemu"}, {"bhyve bhyve ", "bhyve"}, {"BHyVE BHyVE ", "bhyve"}, {" lrpepyh  vr", "parallels"},
        {"prl hyperv  ", "parallels"}, {"ACRNACRNACRN", "acrn"}, {" QNXQVMBSQG ", "qnx"}, {"VirtualApple", "apple"}, {"UnisysSpar64", "unisys"}};
    for (auto& [k, n] : known) if (v.compare(0, std::strlen(k), k) == 0) return n;
    return {};
}
// DMI / SMBIOS vendor and product strings -> hypervisor name ("" when they describe real hardware).
inline std::string hypervisor_from_dmi(const std::string& vendor, const std::string& product) {
    auto has = [](const std::string& s, const char* w) { return s.find(w) != std::string::npos; };
    if (has(product, "KVM") || has(vendor, "KVM")) return "kvm";
    if (has(vendor, "QEMU") || has(product, "QEMU") || has(product, "Standard PC")) return "qemu";
    if (has(vendor, "VMware") || has(product, "VMware")) return "vmware";
    if (has(vendor, "innotek") || has(product, "VirtualBox") || has(vendor, "Oracle Corporation")) return "virtualbox";
    if (has(vendor, "Microsoft Corporation") && has(product, "Virtual")) return "hyperv";
    if (has(vendor, "Xen") || has(product, "HVM domU")) return "xen";
    if (has(vendor, "Parallels") || has(product, "Parallels")) return "parallels";
    if (has(vendor, "Amazon EC2")) return "kvm";
    if (has(vendor, "Google") && has(product, "Google Compute Engine")) return "kvm";
    if (has(vendor, "Apple") && has(product, "Virtual")) return "apple";
    if (has(product, "BHYVE") || has(vendor, "bhyve")) return "bhyve";
    if (has(vendor, "OpenStack") || has(product, "OpenStack")) return "kvm";
    return {};
}
// Container runtime named by a /proc/<pid>/cgroup text ("" when none).
inline std::string container_from_cgroup(const std::string& text) {
    if (text.find("kubepods") != std::string::npos) return "kubernetes";
    if (text.find("docker") != std::string::npos) return "docker";
    if (text.find("libpod") != std::string::npos) return "podman";
    if (text.find("/lxc/") != std::string::npos || text.find("lxc.payload") != std::string::npos) return "lxc";
    if (text.find("machine.slice/systemd-nspawn") != std::string::npos) return "systemd-nspawn";
    return {};
}
// Kernel release -> true for WSL ("...-microsoft-standard-WSL2", "...-Microsoft").
inline bool wsl_kernel(const std::string& release) {
    std::string r = release; for (auto& c : r) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return r.find("microsoft") != std::string::npos || r.find("wsl") != std::string::npos;
}
}

inline virt_info virtualization() {
    virt_info v;
#if defined(FORGE_OS_HAVE_CPUID)
    {
        unsigned a = 0, b = 0, c = 0, d = 0; detail::cpuid(1, 0, a, b, c, d);
        if (c & (1u << 31)) {
            detail::cpuid(0x40000000u, 0, a, b, c, d);
            char vend[13] = {}; std::memcpy(vend, &b, 4); std::memcpy(vend + 4, &c, 4); std::memcpy(vend + 8, &d, 4);
            v.hypervisor = detail::hypervisor_from_cpuid_vendor(vend);
            if (v.hypervisor.empty()) v.hypervisor = "unknown";
        }
    }
#endif
#if defined(__linux__)
#if defined(__ANDROID__)
    {
        auto hw = detail::android_prop("ro.hardware"), model = detail::android_prop("ro.product.model"), fp = detail::android_prop("ro.build.fingerprint");
        v.emulator = detail::android_prop("ro.kernel.qemu") == "1" || detail::android_prop("ro.boot.qemu") == "1" || hw == "goldfish" || hw == "ranchu" ||
                     model.find("sdk_gphone") != std::string::npos || model.find("Emulator") != std::string::npos || model.find("Android SDK built for") != std::string::npos ||
                     fp.rfind("generic", 0) == 0;
        if (v.emulator && v.hypervisor.empty()) v.hypervisor = "qemu";
    }
#else
    {
        if (v.hypervisor.empty() || v.hypervisor == "unknown") {
            auto h = detail::hypervisor_from_dmi(detail::trim(detail::slurp("/sys/class/dmi/id/sys_vendor")), detail::trim(detail::slurp("/sys/class/dmi/id/product_name")));
            if (h.empty()) { auto t = detail::trim(detail::slurp("/sys/hypervisor/type")); if (!t.empty()) h = t; }
            if (h.empty()) { auto dt = detail::slurp("/proc/device-tree/hypervisor/compatible"); if (dt.find("xen") != std::string::npos) h = "xen"; else if (dt.find("kvm") != std::string::npos || dt.find("qemu") != std::string::npos) h = "kvm"; else if (!dt.empty()) h = "unknown"; }
            if (!h.empty()) v.hypervisor = h;
        }
        std::string release = detail::trim(detail::slurp("/proc/sys/kernel/osrelease"));
        if (detail::wsl_kernel(release) || has_env("WSL_DISTRO_NAME") || stat("/proc/sys/fs/binfmt_misc/WSLInterop").exists) v.container = "wsl";
        else if (stat("/.dockerenv").exists) v.container = "docker";
        else if (stat("/run/.containerenv").exists) v.container = "podman";
        else if (auto sc = detail::trim(detail::slurp("/run/systemd/container")); !sc.empty()) v.container = sc;
        else if (auto ce = getenv("container"); !ce.empty()) v.container = ce;
        else if (has_env("KUBERNETES_SERVICE_HOST")) v.container = "kubernetes";
        else if (auto cg = detail::container_from_cgroup(detail::slurp("/proc/self/cgroup")); !cg.empty()) v.container = cg;
        else if (stat("/.flatpak-info").exists) v.container = "flatpak";
        else if (has_env("SNAP") && has_env("SNAP_NAME")) v.container = "snap";
    }
#endif
#elif defined(__APPLE__)
    {
        int tr = 0; std::size_t n = sizeof tr;
        if (::sysctlbyname("sysctl.proc_translated", &tr, &n, nullptr, 0) == 0 && tr == 1) v.translator = "rosetta";
        int hv = 0; n = sizeof hv;
        if (::sysctlbyname("kern.hv_vmm_present", &hv, &n, nullptr, 0) == 0 && hv == 1 && v.hypervisor.empty()) v.hypervisor = "unknown";
        if (!v.hypervisor.empty()) {
            auto model = detail::sysctl_str("hw.model");
            if (model.find("VMware") != std::string::npos) v.hypervisor = "vmware";
            else if (model.find("Parallels") != std::string::npos) v.hypervisor = "parallels";
            else if (model.find("VirtualBox") != std::string::npos) v.hypervisor = "virtualbox";
            else if (model.find("VirtualMac") != std::string::npos || v.hypervisor == "unknown") v.hypervisor = "apple";
        }
#if defined(FORGE_OS_IOS)
#if defined(TARGET_OS_SIMULATOR) && TARGET_OS_SIMULATOR
        v.emulator = true;
#else
        v.emulator = has_env("SIMULATOR_DEVICE_NAME");
#endif
#endif
    }
#elif defined(FORGE_OS_FREEBSD)
    {
        auto g = detail::sysctl_str("kern.vm_guest");
        if (!g.empty() && g != "none") v.hypervisor = g == "hv" ? "hyperv" : g == "vbox" ? "virtualbox" : g == "generic" ? (v.hypervisor.empty() ? "unknown" : v.hypervisor) : g;
        if (detail::sysctl_ll("security.jail.jailed", 0) == 1) v.container = "jail";
    }
#elif defined(FORGE_OS_WINDOWS)
    {
        auto reg_str = [](const wchar_t* key, const wchar_t* name) -> std::string {
            HKEY k = nullptr; if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, key, 0, KEY_QUERY_VALUE | KEY_WOW64_64KEY, &k) != ERROR_SUCCESS) return {};
            wchar_t w[256] = {}; DWORD sz = sizeof w - sizeof(wchar_t), ty = 0;
            LSTATUS rc = RegQueryValueExW(k, name, nullptr, &ty, reinterpret_cast<BYTE*>(w), &sz); RegCloseKey(k);
            return rc == ERROR_SUCCESS && ty == REG_SZ ? narrow(w) : std::string();
        };
        if (v.hypervisor.empty() || v.hypervisor == "unknown") {
            auto h = detail::hypervisor_from_dmi(reg_str(L"HARDWARE\\DESCRIPTION\\System\\BIOS", L"SystemManufacturer"), reg_str(L"HARDWARE\\DESCRIPTION\\System\\BIOS", L"SystemProductName"));
            if (!h.empty()) v.hypervisor = h;
        }
        {
            HKEY k = nullptr; DWORD ct = 0, sz = sizeof ct, ty = 0;
            if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control", 0, KEY_QUERY_VALUE | KEY_WOW64_64KEY, &k) == ERROR_SUCCESS) {
                if (RegQueryValueExW(k, L"ContainerType", nullptr, &ty, reinterpret_cast<BYTE*>(&ct), &sz) == ERROR_SUCCESS && ty == REG_DWORD) v.container = "windows";
                RegCloseKey(k);
            }
        }
        if (HMODULE nt = GetModuleHandleW(L"ntdll.dll"); nt && GetProcAddress(nt, "wine_get_version")) v.translator = "wine";
        else {
            using fn = BOOL(WINAPI*)(HANDLE, USHORT*, USHORT*);
            auto f = reinterpret_cast<fn>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "IsWow64Process2")));
            USHORT pm = 0, nm = 0;
            if (f && f(GetCurrentProcess(), &pm, &nm)) {
#if defined(_M_ARM64) || defined(__aarch64__)
                (void)pm; (void)nm;
#else
                if (nm == 0xAA64 /*IMAGE_FILE_MACHINE_ARM64*/) v.translator = "arm64-emulation";   // x86 / x64 code on an ARM64 system
#endif
            }
        }
    }
#endif
    return v;
}
} // namespace forge::os

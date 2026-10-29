#pragma once
// Huge pages for allocations and mappings.
//   Linux/Android: MAP_HUGETLB (+MAP_HUGE_2MB/1GB) from the hugetlbfs pool, else an aligned mapping with
//   madvise(MADV_HUGEPAGE) (transparent huge pages).  Windows: VirtualAlloc(MEM_LARGE_PAGES) after enabling
//   SeLockMemoryPrivilege (needs the "Lock pages in memory" right).  macOS x86_64: VM_FLAGS_SUPERPAGE_SIZE_2MB;
//   Apple Silicon/iOS have no superpage API (16K pages): falls back.  FreeBSD: MAP_ALIGNED_SUPER (transparent
//   superpage promotion).  With fallback=true a normal allocation is returned and `kind` says what you got.
#include <forge/os/detail.hpp>

namespace forge::os {
enum class page_kind { normal, transparent_huge, huge_2m, huge_1g };
inline const char* page_kind_name(page_kind k) {
    switch (k) { case page_kind::transparent_huge: return "thp"; case page_kind::huge_2m: return "2M"; case page_kind::huge_1g: return "1G"; default: return "normal"; }
}
struct huge_region {
    void* ptr = nullptr; std::size_t size = 0; page_kind kind = page_kind::normal;
    explicit operator bool() const { return ptr != nullptr; }
};
// Page sizes the OS can hand out explicitly (beyond the base page).
inline std::vector<std::size_t> huge_page_sizes() {
    std::vector<std::size_t> v;
#if defined(__linux__)
    if (auto l = list_dir("/sys/kernel/mm/hugepages"))
        for (auto& e : l.value) if (e.name.rfind("hugepages-", 0) == 0) v.push_back(std::size_t(std::strtoull(e.name.c_str() + 10, nullptr, 10)) * 1024);
#elif defined(FORGE_OS_WINDOWS)
    if (auto m = GetLargePageMinimum()) v.push_back(m);
#elif defined(FORGE_OS_FREEBSD)
    std::size_t ps[8] = {}; int n = ::getpagesizes(ps, 8); for (int i = 1; i < n; ++i) v.push_back(ps[i]);
#elif defined(FORGE_OS_MACOS) && defined(__x86_64__)
    v.push_back(std::size_t(2) << 20);
#endif
    std::sort(v.begin(), v.end()); return v;
}
inline bool thp_enabled() {
#if defined(__linux__)
    auto s = detail::slurp("/sys/kernel/mm/transparent_hugepage/enabled");
    return s.find("[always]") != std::string::npos || s.find("[madvise]") != std::string::npos;
#elif defined(FORGE_OS_FREEBSD)
    return detail::sysctl_ll("vm.pmap.pg_ps_enabled", 0) == 1;
#else
    return false;
#endif
}
#if defined(FORGE_OS_WINDOWS)
inline bool enable_lock_memory_privilege() {
    static int st = -1; if (st >= 0) return st == 1;
    HANDLE t = nullptr; st = 0;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &t)) return false;
    TOKEN_PRIVILEGES tp{}; tp.PrivilegeCount = 1; tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    if (LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid) && AdjustTokenPrivileges(t, FALSE, &tp, 0, nullptr, nullptr) && GetLastError() == ERROR_SUCCESS) st = 1;
    CloseHandle(t); return st == 1;
}
#endif
namespace detail {
inline std::size_t round_up(std::size_t n, std::size_t a) { return (n + a - 1) / a * a; }
#if defined(FORGE_OS_POSIX)
// Anonymous mapping of n bytes aligned to `a` (over-map then trim).
inline void* mmap_aligned(std::size_t n, std::size_t a, int extra_flags = 0) {
    std::size_t len = n + a;
    auto* p = static_cast<std::uint8_t*>(::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | extra_flags, -1, 0));
    if (p == MAP_FAILED) return nullptr;
    auto al = reinterpret_cast<std::uint8_t*>((reinterpret_cast<std::uintptr_t>(p) + a - 1) & ~(std::uintptr_t(a) - 1));
    if (al > p) ::munmap(p, std::size_t(al - p));
    std::size_t tail = std::size_t((p + len) - (al + n)); if (tail) ::munmap(al + n, tail);
    return al;
}
#endif
}
// want = huge_2m or huge_1g. fallback: degrade 1G -> 2M -> transparent -> normal instead of failing.
inline result<huge_region> huge_alloc(std::size_t n, page_kind want = page_kind::huge_2m, bool fallback = true) {
    huge_region r; if (!n) return {r, {22, "huge_alloc: size 0"}};
    const std::size_t two = std::size_t(2) << 20, gig = std::size_t(1) << 30;
    error last = unsupported("huge pages");
#if defined(__linux__)
    if (want == page_kind::huge_1g) {
        std::size_t len = detail::round_up(n, gig);
        void* p = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_HUGETLB | (30 << 26), -1, 0);
        if (p != MAP_FAILED) return {{p, len, page_kind::huge_1g}, {}};
        last = last_error("mmap MAP_HUGE_1GB"); if (!fallback) return {r, last};
    }
    if (want == page_kind::huge_1g || want == page_kind::huge_2m) {
        std::size_t len = detail::round_up(n, two);
        void* p = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_HUGETLB | (21 << 26), -1, 0);
        if (p != MAP_FAILED) return {{p, len, page_kind::huge_2m}, {}};
        last = last_error("mmap MAP_HUGE_2MB"); if (!fallback) return {r, last};
    }
    {   // transparent huge pages: 2M-aligned mapping + MADV_HUGEPAGE
        std::size_t len = detail::round_up(n, two);
        void* p = detail::mmap_aligned(len, two);
        if (!p) return {r, last_error("mmap")};
        bool thp = thp_enabled() && ::madvise(p, len, 14 /*MADV_HUGEPAGE*/) == 0;
        return {{p, len, thp ? page_kind::transparent_huge : page_kind::normal}, {}};
    }
#elif defined(FORGE_OS_WINDOWS)
    if (want != page_kind::normal && want != page_kind::transparent_huge) {
        if (std::size_t lp = GetLargePageMinimum(); lp && enable_lock_memory_privilege()) {
            std::size_t len = detail::round_up(n, lp);
            if (void* p = VirtualAlloc(nullptr, len, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE))
                return {{p, len, lp >= gig ? page_kind::huge_1g : page_kind::huge_2m}, {}};
            last = last_error("VirtualAlloc MEM_LARGE_PAGES");
        } else last = {1314, "large pages: SeLockMemoryPrivilege not held (grant 'Lock pages in memory')"};
        if (!fallback) return {r, last};
    }
    std::size_t len = detail::round_up(n, page_size());
    void* p = VirtualAlloc(nullptr, len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!p) return {r, last_error("VirtualAlloc")};
    return {{p, len, page_kind::normal}, {}};
#elif defined(FORGE_OS_FREEBSD)
    std::size_t len = detail::round_up(n, two);
    void* p = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_ALIGNED_SUPER, -1, 0);
    if (p == MAP_FAILED) return {r, last_error("mmap MAP_ALIGNED_SUPER")};
    if (!fallback && want == page_kind::huge_1g) { ::munmap(p, len); return {r, unsupported("1G superpages")}; }
    return {{p, len, thp_enabled() ? page_kind::transparent_huge : page_kind::normal}, {}};
#else
#if defined(FORGE_OS_MACOS) && defined(__x86_64__)
    if (want == page_kind::huge_2m || want == page_kind::huge_1g) {
        std::size_t len = detail::round_up(n, two);
        void* p = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, int(2 << 16) /*VM_FLAGS_SUPERPAGE_SIZE_2MB*/, 0);
        if (p != MAP_FAILED) return {{p, len, page_kind::huge_2m}, {}};
        last = last_error("mmap VM_FLAGS_SUPERPAGE_SIZE_2MB");
    }
#endif
    if (!fallback && want != page_kind::normal) return {r, last};
    std::size_t len = detail::round_up(n, page_size());
    void* p = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) return {r, last_error("mmap")};
    return {{p, len, page_kind::normal}, {}};
#endif
}
inline void huge_free(huge_region& r) {
    if (!r.ptr) return;
#if defined(FORGE_OS_WINDOWS)
    VirtualFree(r.ptr, 0, MEM_RELEASE);
#else
    ::munmap(r.ptr, r.size);
#endif
    r = {};
}
// Ask for huge pages on an existing mapping (e.g. an mmapped weight file). Linux: MADV_HUGEPAGE (file THP needs
// CONFIG_READ_ONLY_THP_FOR_FS); FreeBSD promotes automatically; elsewhere unsupported.
inline status advise_huge(void* p, std::size_t n) {
#if defined(__linux__)
    auto a = reinterpret_cast<std::uintptr_t>(p) & ~(std::uintptr_t(page_size()) - 1);
    if (::madvise(reinterpret_cast<void*>(a), n + (reinterpret_cast<std::uintptr_t>(p) - a), 14)) return {{}, last_error("madvise(MADV_HUGEPAGE)")};
    return {};
#elif defined(FORGE_OS_FREEBSD)
    (void)p; (void)n; return {};
#else
    (void)p; (void)n; return {{}, unsupported("advise_huge")};
#endif
}
} // namespace forge::os

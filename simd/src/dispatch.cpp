#if defined(__aarch64__) && defined(__linux__)
#include <sys/prctl.h>
#endif
#include <forge/simd/dispatch.hpp>
#include <cstdlib>
#include <cstring>
#include <format>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#define FORGE_X86_CPUID 1
#endif
#if defined(__linux__)
#include <sys/auxv.h>
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(__FreeBSD__)
#include <sys/auxv.h>
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <sys/sysctl.h>
#elif defined(__FreeBSD__)
#include <sys/auxv.h>
#elif defined(_WIN32)
extern "C" __declspec(dllimport) int __stdcall IsProcessorFeaturePresent(unsigned long);
#endif
#endif

namespace forge::simd {
namespace mv_scalar { extern const kernels table; }
// Which tables exist is decided by CMake (FORGE_SIMD_HAVE_<TABLE> on this file), one set per architecture.
#if defined(FORGE_SIMD_HAVE_AVX2)
namespace mv_avx2 { extern const kernels table; }
namespace mv_avx512 { extern const kernels table; }
#endif
#if defined(FORGE_SIMD_HAVE_SSE)
namespace mv_sse2 { extern const kernels table; }
namespace mv_sse42 { extern const kernels table; }
#endif
#if defined(FORGE_SIMD_HAVE_LSX)
namespace mv_lsx { extern const kernels table; }
#endif
#if defined(FORGE_SIMD_HAVE_LASX)
namespace mv_lasx { extern const kernels table; }
#endif
#if defined(FORGE_SIMD_HAVE_VSX)
namespace mv_vsx { extern const kernels table; }
#endif
#if defined(FORGE_SIMD_HAVE_VXE)
namespace mv_vxe { extern const kernels table; }
#endif
#if defined(FORGE_SIMD_HAVE_WASM128)
namespace mv_simd128 { extern const kernels table; }
#endif
#if defined(FORGE_SIMD_HAVE_MSA)
namespace mv_msa { extern const kernels table; }
#endif
#if defined(FORGE_SIMD_HAVE_NEON32)
namespace mv_neon { extern const kernels table; }
#endif
#if defined(__aarch64__)
namespace mv_neon { extern const kernels table; }
#endif
// Fixed-length scalable tables, one per built length (FORGE_SIMD_SVE<B> / FORGE_SIMD_RVV<B> set by CMake).
#if defined(FORGE_SIMD_SVE128)
namespace mv_sve128 { extern const kernels table; }
#endif
#if defined(FORGE_SIMD_SVE256)
namespace mv_sve256 { extern const kernels table; }
#endif
#if defined(FORGE_SIMD_SVE512)
namespace mv_sve512 { extern const kernels table; }
#endif
#if defined(FORGE_SIMD_SVE1024)
namespace mv_sve1024 { extern const kernels table; }
#endif
#if defined(FORGE_SIMD_RVV128)
namespace mv_rvv128 { extern const kernels table; }
#endif
#if defined(FORGE_SIMD_RVV256)
namespace mv_rvv256 { extern const kernels table; }
#endif
#if defined(FORGE_SIMD_RVV512)
namespace mv_rvv512 { extern const kernels table; }
#endif
#if defined(FORGE_SIMD_RVV1024)
namespace mv_rvv1024 { extern const kernels table; }
#endif
// Read the hardware vector length without SVE/V codegen flags in this TU (raw encodings, numeric CSR).
#if defined(__aarch64__) && defined(__linux__)
static int sve_vl_bytes() { long v; __asm__ volatile(".inst 0x04bf5020\n\tmov %0, x0" : "=r"(v) : : "x0"); return int(v); }  // rdvl x0, #1
#endif
#if defined(__riscv) && defined(__linux__)
static int rvv_vlenb() { unsigned long v; __asm__ volatile("csrr %0, 0xc22" : "=r"(v)); return int(v); }  // vlenb
#endif
static const kernels* sve_table(int bytes) {
    switch (bytes * 8) {
#if defined(FORGE_SIMD_SVE128)
    case 128: return &mv_sve128::table;
#endif
#if defined(FORGE_SIMD_SVE256)
    case 256: return &mv_sve256::table;
#endif
#if defined(FORGE_SIMD_SVE512)
    case 512: return &mv_sve512::table;
#endif
#if defined(FORGE_SIMD_SVE1024)
    case 1024: return &mv_sve1024::table;
#endif
    default: (void)bytes; return nullptr;
    }
}
// SVE hardware can have any VL that is a multiple of 128 bits (384, 640, ...), but fixed-length kernels exist
// only for powers of two. Linux lets a process shorten its own VL, so at load time (before main, before
// any thread exists; new threads copy it) a VL with no table (384, 640, 2048, ...) is lowered to the largest
// power of two we have a table for. FORGE_SVE_KEEP_VL=1 disables this (the sve lane then falls back to NEON).
#if defined(__aarch64__) && defined(__linux__)
#ifndef PR_SVE_SET_VL
#define PR_SVE_SET_VL 50
#endif
static int g_sve_hw_bytes = 0;   // VL before any adjustment (0 = no SVE / not adjusted)
__attribute__((constructor)) static void forge_sve_normalise_vl() {
    if (!(getauxval(AT_HWCAP) & (1ul << 22))) return;
    int vl = sve_vl_bytes();
    g_sve_hw_bytes = vl;
    if (sve_table(vl)) return;
    if (const char* e = std::getenv("FORGE_SVE_KEEP_VL"); e && *e == '1') return;
    int want = 0;
    for (int b : {16, 32, 64, 128}) if (b <= vl && sve_table(b)) want = b;
    // Threads created later copy the caller's VL; PR_SVE_VL_INHERIT (keep it across execve) is not needed.
    if (want) prctl(PR_SVE_SET_VL, want, 0, 0, 0);
}
#endif
static const kernels* rvv_table(int bytes) {
    switch (bytes * 8) {
#if defined(FORGE_SIMD_RVV128)
    case 128: return &mv_rvv128::table;
#endif
#if defined(FORGE_SIMD_RVV256)
    case 256: return &mv_rvv256::table;
#endif
#if defined(FORGE_SIMD_RVV512)
    case 512: return &mv_rvv512::table;
#endif
#if defined(FORGE_SIMD_RVV1024)
    case 1024: return &mv_rvv1024::table;
#endif
    default: (void)bytes; return nullptr;
    }
}

#if defined(FORGE_X86_CPUID)
static unsigned long long xgetbv0() {
    unsigned eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<unsigned long long>(edx) << 32) | eax;
}
#endif

// ELF auxiliary vector (AT_HWCAP = 16, AT_HWCAP2 = 26); 0 where the OS has none.
[[maybe_unused]] static unsigned long hwcap(int which) {
#if defined(__linux__)
    return getauxval(static_cast<unsigned long>(which));
#elif defined(__FreeBSD__)
    unsigned long v = 0;
    return elf_aux_info(which, &v, int(sizeof v)) == 0 ? v : 0;
#else
    (void)which; return 0;
#endif
}

static cpu_features do_detect() {
    cpu_features f;
#if defined(FORGE_X86_CPUID)
    unsigned a, b, c, d;
    __cpuid(0, a, b, c, d);
    unsigned maxl = a;
    char v[13]; std::memcpy(v, &b, 4); std::memcpy(v + 4, &d, 4); std::memcpy(v + 8, &c, 4); v[12] = 0;
    f.vendor = v;
    __cpuid(1, a, b, c, d);
    f.sse2 = d >> 26 & 1; f.sse4_2 = c >> 20 & 1; f.popcnt = c >> 23 & 1; f.fma = c >> 12 & 1; f.f16c = c >> 29 & 1;
    bool osxsave = c >> 27 & 1, avx_hw = c >> 28 & 1;
    unsigned long long xcr0 = osxsave ? xgetbv0() : 0;
    f.os_avx = (xcr0 & 0x6) == 0x6;
    f.os_avx512 = f.os_avx && (xcr0 & 0xe0) == 0xe0;
    f.avx = avx_hw && f.os_avx;
    if (!f.os_avx) f.fma = f.f16c = false;
    if (maxl >= 7) {
        __cpuid_count(7, 0, a, b, c, d);
        f.avx2 = (b >> 5 & 1) && f.os_avx; f.bmi2 = b >> 8 & 1;
        if (f.os_avx512) {
            f.avx512f = b >> 16 & 1; f.avx512dq = b >> 17 & 1; f.avx512ifma = b >> 21 & 1; f.avx512cd = b >> 28 & 1;
            f.avx512bw = b >> 30 & 1; f.avx512vl = b >> 31 & 1; f.avx512vbmi = c >> 1 & 1; f.avx512vbmi2 = c >> 6 & 1;
            f.avx512vnni = c >> 11 & 1; f.avx512bitalg = c >> 12 & 1; f.avx512vpopcntdq = c >> 14 & 1;
            f.avx512fp16 = d >> 23 & 1;
        }
        f.hybrid = d >> 15 & 1;
        bool amx_tile_hw = d >> 24 & 1;
        f.amx_bf16 = d >> 22 & 1; f.amx_int8 = d >> 25 & 1;
        unsigned a1, b1, c1, d1;
        __cpuid_count(7, 1, a1, b1, c1, d1);
        f.avxvnni = (a1 >> 4 & 1) && f.os_avx; f.avx512bf16 = (a1 >> 5 & 1) && f.os_avx512; f.amx_fp16 = a1 >> 21 & 1;
        f.avx10 = d1 >> 19 & 1;
        if (f.avx10 && maxl >= 0x24) { __cpuid_count(0x24, 0, a, b, c, d); f.avx10_version = int(b & 0xff); }
        // AMX: XCR0 bits 17/18 must be enabled and Linux requires a permission request.
        f.amx_tile = amx_tile_hw && (xcr0 & 0x60000) == 0x60000;
#if defined(__linux__) && defined(__x86_64__)
        if (f.amx_tile) {
            constexpr int ARCH_REQ_XCOMP_PERM = 0x1023, XFEATURE_XTILEDATA = 18;
            f.amx_permitted = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
        }
#endif
        if (!f.amx_tile) f.amx_bf16 = f.amx_int8 = f.amx_fp16 = false;
    }
    __cpuid(0x80000000, a, b, c, d);
    if (a >= 0x80000004) {
        char br[49] = {};
        for (unsigned i = 0; i < 3; ++i) { __cpuid(0x80000002 + i, a, b, c, d);
            std::memcpy(br + 16 * i, &a, 4); std::memcpy(br + 16 * i + 4, &b, 4);
            std::memcpy(br + 16 * i + 8, &c, 4); std::memcpy(br + 16 * i + 12, &d, 4); }
        f.brand = br;
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    f.neon = true; // mandatory in AArch64
    f.vendor = "aarch64";
    // Integer dot products: ASIMDDP (sdot/udot, optional from ARMv8.2, mandatory from ARMv8.4) and I8MM (usdot).
#if defined(__linux__)   // Linux and Android: HWCAP_ASIMDDP, HWCAP_SVE, HWCAP2_SVE2, HWCAP2_I8MM
    unsigned long hw = getauxval(AT_HWCAP), hw2 = getauxval(AT_HWCAP2);
    f.dotprod = hw & (1ul << 20); f.i8mm = hw2 & (1ul << 13);
    f.sve = hw & (1ul << 22); f.sve2 = hw2 & (1ul << 1);
    if (f.sve) { f.sve_bytes = sve_vl_bytes(); f.sve_hw_bytes = g_sve_hw_bytes ? g_sve_hw_bytes : f.sve_bytes; }
#elif defined(__APPLE__)
    auto sysctl_on = [](const char* name) { int v = 0; std::size_t n = sizeof v; return sysctlbyname(name, &v, &n, nullptr, 0) == 0 && v != 0; };
    f.dotprod = sysctl_on("hw.optional.arm.FEAT_DotProd");
    f.i8mm = sysctl_on("hw.optional.arm.FEAT_I8MM");
#if TARGET_OS_OSX
    f.dotprod = true;   // every Apple Silicon Mac (M1 and later); the FEAT_ sysctl names only exist from macOS 12
#endif
#elif defined(__FreeBSD__)   // same bit numbers as Linux
    unsigned long hw = 0, hw2 = 0;
    elf_aux_info(AT_HWCAP, &hw, sizeof hw); elf_aux_info(AT_HWCAP2, &hw2, sizeof hw2);
    f.dotprod = hw & (1ul << 20); f.i8mm = hw2 & (1ul << 13);
#elif defined(_WIN32)   // PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE = 43, PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE = 66
    f.dotprod = IsProcessorFeaturePresent(43) != 0; f.i8mm = IsProcessorFeaturePresent(66) != 0;
#endif
#elif defined(__riscv)
    f.vendor = "riscv64";
#if defined(__linux__)
    f.rvv = getauxval(AT_HWCAP) & (1ul << ('V' - 'A'));
    if (f.rvv) f.rvv_bytes = rvv_vlenb();
#endif
#elif defined(__loongarch__)
    f.vendor = "loongarch64";
    { unsigned long hw = hwcap(16);   // HWCAP_LOONGARCH_LSX = 1 << 4, HWCAP_LOONGARCH_LASX = 1 << 5
      f.lsx = hw & (1ul << 4); f.lasx = hw & (1ul << 5); }
#if defined(__loongarch_sx)
    f.lsx = true;    // the build's baseline already requires it
#endif
#if defined(__loongarch_asx)
    f.lasx = true;
#endif
#elif defined(__powerpc64__)
    f.vendor = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ ? "ppc64le" : "ppc64";
    { unsigned long hw = hwcap(16), hw2 = hwcap(26);
      f.vsx = hw & 0x00000080ul;               // PPC_FEATURE_HAS_VSX
      f.power8 = hw2 & 0x80000000ul;           // PPC_FEATURE2_ARCH_2_07
      f.power9 = hw2 & 0x00800000ul;           // PPC_FEATURE2_ARCH_3_00
      f.power10 = hw2 & 0x00040000ul; }        // PPC_FEATURE2_ARCH_3_1
#if defined(__VSX__) && defined(_ARCH_PWR8)
    f.vsx = f.power8 = true;   // baseline of the build (ELFv2 little-endian requires POWER8)
#endif
#elif defined(__s390x__)
    f.vendor = "s390x";
    { unsigned long hw = hwcap(16);   // HWCAP_S390_VXRS = 2048, _VXRS_EXT = 8192 (z14), _VXRS_EXT2 = 32768 (z15)
      f.s390_vx = hw & 2048ul; f.s390_vxe = f.s390_vx && (hw & 8192ul); f.s390_vxe2 = f.s390_vxe && (hw & 32768ul); }
#elif defined(__wasm__)
    f.vendor = "wasm";
#if defined(__wasm_simd128__) || defined(FORGE_SIMD_HAVE_WASM128)
    f.wasm_simd128 = true;   // a module is validated as a whole: there is nothing to detect at run time
#endif
#elif defined(__mips__)
    f.vendor = "mips";
    f.msa = hwcap(16) & (1ul << 1);   // HWCAP_MIPS_MSA
#elif defined(__arm__)
    f.vendor = "arm";
    { unsigned long hw = hwcap(16);   // HWCAP_NEON = 1 << 12, HWCAP_VFPv4 = 1 << 16
      f.neon = hw & (1ul << 12); f.vfpv4 = hw & (1ul << 16); }
#endif
    return f;
}

const cpu_features& detect() { static const cpu_features f = do_detect(); return f; }

std::string cpu_features::summary() const {
    std::string s = std::format("{} | {}", vendor, brand);
    auto add = [&](bool b, const char* n) { if (b) { s += ' '; s += n; } };
    add(sse4_2, "sse4.2"); add(avx, "avx"); add(avx2, "avx2"); add(fma, "fma"); add(f16c, "f16c");
    add(avx512f, "avx512f"); add(avx512bw, "avx512bw"); add(avx512dq, "avx512dq"); add(avx512vl, "avx512vl");
    add(avx512vnni, "avx512vnni"); add(avx512bf16, "avx512bf16"); add(avx512fp16, "avx512fp16");
    add(avx512ifma, "avx512ifma"); add(avx512vbmi, "avx512vbmi"); add(avxvnni, "avxvnni"); add(avx10, "avx10");
    add(amx_tile, "amx-tile"); add(amx_permitted, "(amx-permitted)"); add(hybrid, "hybrid"); add(neon, "neon");
    add(dotprod, "dotprod"); add(i8mm, "i8mm");
    add(sve, "sve"); add(sve2, "sve2"); if (sve_bytes) s += std::format("(vl={}b)", sve_bytes * 8);
    if (sve_hw_bytes && sve_hw_bytes != sve_bytes) s += std::format("(hw vl={}b, lowered)", sve_hw_bytes * 8);
    add(rvv, "rvv"); if (rvv_bytes) s += std::format("(vlen={}b)", rvv_bytes * 8);
    add(lsx, "lsx"); add(lasx, "lasx"); add(vsx, "vsx"); add(power8, "power8"); add(power9, "power9"); add(power10, "power10");
    add(s390_vx, "vx"); add(s390_vxe, "vxe"); add(s390_vxe2, "vxe2"); add(wasm_simd128, "simd128"); add(msa, "msa"); add(vfpv4, "vfpv4");
    return s;
}

const kernels* by_name(const char* t) {
    [[maybe_unused]] const auto& f = detect();
    std::string_view n = t ? t : "";
    if (n == "scalar") return &mv_scalar::table;
#if defined(__aarch64__)
    if (n == "neon") return f.neon ? &mv_neon::table : nullptr;
    if (n == "sve") return f.sve ? sve_table(f.sve_bytes) : nullptr;  // table built for exactly this VL, if any
#elif defined(__riscv)
    if (n == "rvv") return f.rvv ? rvv_table(f.rvv_bytes) : nullptr;
#endif
#if defined(FORGE_SIMD_HAVE_AVX2)
    // The whole x86-64-v3 / -v4 feature set the TUs are built for: nested VMs (e.g. the FreeBSD CI guest on AMD)
    // can report AVX2 while masking F16C/BMI2, and a kernel using them then dies with SIGILL.
    const bool v3 = f.sse2 && f.sse4_2 && f.popcnt && f.avx && f.avx2 && f.fma && f.f16c && f.bmi2;
    if (n == "avx2") return v3 ? &mv_avx2::table : nullptr;
    if (n == "avx512") return v3 && f.avx512f && f.avx512bw && f.avx512dq && f.avx512vl && f.avx512cd ? &mv_avx512::table : nullptr;
#endif
#if defined(FORGE_SIMD_HAVE_SSE)
    if (n == "sse4.2" || n == "sse42") return f.sse2 && f.sse4_2 && f.popcnt ? &mv_sse42::table : nullptr;   // x86-64-v2 TU
    if (n == "sse2") return f.sse2 ? &mv_sse2::table : nullptr;
#endif
#if defined(FORGE_SIMD_HAVE_LASX)
    if (n == "lasx") return f.lsx && f.lasx ? &mv_lasx::table : nullptr;
#endif
#if defined(FORGE_SIMD_HAVE_LSX)
    if (n == "lsx") return f.lsx ? &mv_lsx::table : nullptr;
#endif
#if defined(FORGE_SIMD_HAVE_VSX)
    if (n == "vsx") return f.vsx && f.power8 ? &mv_vsx::table : nullptr;   // TU is built for POWER8
#endif
#if defined(FORGE_SIMD_HAVE_VXE)
    if (n == "vxe") return f.s390_vxe ? &mv_vxe::table : nullptr;          // TU is built for z14
#endif
#if defined(FORGE_SIMD_HAVE_WASM128)
    if (n == "simd128") return &mv_simd128::table;
#endif
#if defined(FORGE_SIMD_HAVE_MSA)
    if (n == "msa") return f.msa ? &mv_msa::table : nullptr;
#endif
#if defined(FORGE_SIMD_HAVE_NEON32)
    // Opt-in: Advanced SIMD on AArch32 always flushes subnormals to zero, so results in the subnormal range
    // (exp below -87, cbrt/erfc of tiny values) miss the oracle tolerance. Never chosen automatically.
    if (n == "neon") {
        const char* e = std::getenv("FORGE_SIMD_ARM32_NEON");
        return e && *e == '1' && f.neon && f.vfpv4 ? &mv_neon::table : nullptr;
    }
#endif
    return nullptr;
}

// Every table name this build knows, preferred first; `scalar` is last and always usable.
std::span<const char* const> target_names() {
    static const char* const names[] = {"avx512", "avx2", "sse4.2", "sse2", "sve", "neon", "rvv", "lasx", "lsx", "vsx", "vxe",
                                        "simd128", "msa", "scalar"};
    return names;
}

const kernels& active() {
    static const kernels* k = [] {
        // FORGE_LANE_TARGET is the old name, still honoured
        for (const char* var : {"FORGE_SIMD_TARGET", "FORGE_LANE_TARGET"})
            if (const char* e = std::getenv(var))
                if (auto* p = by_name(e)) return p;
        for (const char* t : target_names()) if (auto* p = by_name(t)) return p;
        return &mv_scalar::table;
    }();
    return *k;
}

} // namespace forge::simd

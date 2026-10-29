#pragma once
// Power and thermal state: battery, AC, low-power mode, thermal pressure/throttling.
//   Linux: /sys/class/power_supply, /sys/firmware/acpi/platform_profile, /sys/class/thermal (temp vs passive trip),
//   cpufreq scaling_max < cpuinfo_max.  Android: same sysfs where readable, plus platform hooks that JNI glue calls
//   with PowerManager.isPowerSaveMode()/getCurrentThermalStatus() and BatteryManager.  Windows: GetSystemPowerStatus
//   (battery saver flag) + CallNtPowerInformation(ProcessorInformation) MHz limit.  macOS/iOS: NSProcessInfo
//   (lowPowerModeEnabled, thermalState) through the objc runtime; IOKit IOPS for the mac power source; UIDevice
//   battery on iOS.  FreeBSD: hw.acpi.battery / hw.acpi.acline / hw.acpi.thermal.
#include <forge/os/detail.hpp>

namespace forge::os {
enum class thermal_state { unknown, nominal, fair, serious, critical };
inline const char* thermal_name(thermal_state t) {
    switch (t) { case thermal_state::nominal: return "nominal"; case thermal_state::fair: return "fair"; case thermal_state::serious: return "serious"; case thermal_state::critical: return "critical"; default: return "unknown"; }
}
struct power_state {
    bool has_battery = false, on_battery = false, charging = false, low_power_mode = false, throttled = false;
    int battery_percent = -1;   // -1 unknown
    thermal_state thermal = thermal_state::unknown;
    double temperature_c = -1;  // hottest sensor we could read, -1 unknown
    std::string source;         // which backend filled this in
};
// Mobile glue (JNI on Android, app code on iOS) can push authoritative values here; they override probing.
struct power_hooks {
    static std::mutex& m() { static std::mutex x; return x; }
    static bool& has() { static bool b = false; return b; }
    static power_state& st() { static power_state s; return s; }
    static void report(const power_state& s) { std::lock_guard<std::mutex> g(m()); st() = s; has() = true; }
    static void clear() { std::lock_guard<std::mutex> g(m()); has() = false; }
};

inline power_state power_status() {
    power_state p;
    {
        std::lock_guard<std::mutex> g(power_hooks::m());
        if (power_hooks::has()) { p = power_hooks::st(); if (p.source.empty()) p.source = "hook"; return p; }
    }
#if defined(__linux__)
    p.source = "sysfs";
    if (auto l = list_dir("/sys/class/power_supply")) {
        for (auto& e : l.value) {
            std::string d = "/sys/class/power_supply/" + e.name, type = detail::trim(detail::slurp(d + "/type"));
            if (type == "Battery") {
                p.has_battery = true; auto cap = detail::slurp_ll(d + "/capacity", -1); if (cap >= 0) p.battery_percent = int(cap);
                auto st = detail::trim(detail::slurp(d + "/status")); p.charging = st == "Charging" || st == "Full"; if (st == "Discharging") p.on_battery = true;
            } else if (type == "Mains" || type == "USB") { if (detail::slurp_ll(d + "/online", 0) == 1) p.on_battery = false; }
        }
    }
    auto prof = detail::trim(detail::slurp("/sys/firmware/acpi/platform_profile")); if (prof == "low-power" || prof == "quiet") p.low_power_mode = true;
    double worst = 0;   // fraction of the passive trip point
    if (auto l = list_dir("/sys/class/thermal")) {
        for (auto& e : l.value) {
            if (e.name.rfind("thermal_zone", 0)) continue;
            std::string d = "/sys/class/thermal/" + e.name; auto t = detail::slurp_ll(d + "/temp", -1); if (t <= 0) continue;
            double c = double(t) / 1000.0; if (c > p.temperature_c) p.temperature_c = c;
            for (int i = 0; i < 8; ++i) {
                auto ty = detail::trim(detail::slurp(d + "/trip_point_" + std::to_string(i) + "_type")); if (ty.empty()) break;
                auto tt = detail::slurp_ll(d + "/trip_point_" + std::to_string(i) + "_temp", -1);
                if ((ty == "passive" || ty == "critical" || ty == "hot") && tt > 0) worst = std::max(worst, double(t) / double(tt) * (ty == "passive" ? 1.0 : 1.15));
            }
        }
    }
    auto smax = detail::slurp_ll("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq", 0), cmax = detail::slurp_ll("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", 0);
    if (smax > 0 && cmax > 0 && smax < cmax * 9 / 10 && !p.low_power_mode) p.throttled = true;
    if (p.temperature_c >= 0) p.thermal = worst >= 1.1 ? thermal_state::critical : worst >= 1.0 ? thermal_state::serious : worst >= 0.85 ? thermal_state::fair : thermal_state::nominal;
    if (p.thermal >= thermal_state::serious) p.throttled = true;
#elif defined(FORGE_OS_WINDOWS)
    p.source = "win32";
    SYSTEM_POWER_STATUS s{};
    if (GetSystemPowerStatus(&s)) {
        p.has_battery = s.BatteryFlag != 128 && s.BatteryFlag != 255; p.on_battery = s.ACLineStatus == 0;
        p.charging = (s.BatteryFlag & 8) != 0; if (s.BatteryLifePercent <= 100) p.battery_percent = s.BatteryLifePercent;
        p.low_power_mode = s.SystemStatusFlag == 1;   // battery saver
    }
    struct ppi { ULONG Number, MaxMhz, CurrentMhz, MhzLimit, MaxIdleState, CurrentIdleState; };
    if (HMODULE h = LoadLibraryW(L"powrprof.dll")) {
        using fn = LONG(WINAPI*)(int, PVOID, ULONG, PVOID, ULONG);
        if (auto f = reinterpret_cast<fn>(reinterpret_cast<void*>(GetProcAddress(h, "CallNtPowerInformation")))) {
            std::vector<ppi> v(cpu_count());
            if (f(11 /*ProcessorInformation*/, nullptr, 0, v.data(), ULONG(v.size() * sizeof(ppi))) == 0 && !v.empty() && v[0].MaxMhz) {
                p.throttled = v[0].MhzLimit < v[0].MaxMhz * 9 / 10;
                p.thermal = v[0].MhzLimit < v[0].MaxMhz / 2 ? thermal_state::serious : p.throttled ? thermal_state::fair : thermal_state::nominal;
            }
        }
        FreeLibrary(h);
    }
#elif defined(__APPLE__)
    p.source = "foundation";
    auto& o = detail::objc_api::get();
    if (o.ok()) {
        if (void* pi = o.send(o.cls("NSProcessInfo"), "processInfo")) {
            p.low_power_mode = o.send<signed char>(pi, "isLowPowerModeEnabled") != 0;
            long ts = o.send<long>(pi, "thermalState");   // NSProcessInfoThermalState 0..3
            p.thermal = thermal_state(int(thermal_state::nominal) + int(ts < 0 ? 0 : ts > 3 ? 3 : ts));
            p.throttled = ts >= 2;
        }
    }
#if defined(FORGE_OS_IOS)
    if (o.ok()) if (void* dev = o.send(o.cls("UIDevice"), "currentDevice")) {
        o.send<void, signed char>(dev, "setBatteryMonitoringEnabled:", 1);
        float lvl = reinterpret_cast<float (*)(void*, void*)>(o.msg)(dev, o.sel("batteryLevel"));
        long st = o.send<long>(dev, "batteryState");   // 1 unplugged, 2 charging, 3 full
        p.has_battery = true; if (lvl >= 0) p.battery_percent = int(lvl * 100 + 0.5f); p.charging = st == 2 || st == 3; p.on_battery = st == 1;
    }
#else
    if (void* io = ::dlopen("/System/Library/Frameworks/IOKit.framework/IOKit", RTLD_LAZY)) {
        auto& cf = detail::cf_api::get();
        using copy_fn = detail::CFTypeRef (*)(); using type_fn = detail::CFStringRef (*)(detail::CFTypeRef);
        auto info = reinterpret_cast<copy_fn>(::dlsym(io, "IOPSCopyPowerSourcesInfo"));
        auto prov = reinterpret_cast<type_fn>(::dlsym(io, "IOPSGetProvidingPowerSourceType"));
        auto list = reinterpret_cast<detail::CFTypeRef (*)(detail::CFTypeRef)>(::dlsym(io, "IOPSCopyPowerSourcesList"));
        auto desc = reinterpret_cast<detail::CFDictionaryRef (*)(detail::CFTypeRef, detail::CFTypeRef)>(::dlsym(io, "IOPSGetPowerSourceDescription"));
        if (cf.ok && info && prov) if (auto blob = info()) {
            p.on_battery = cf.to_string(prov(blob)) == "Battery Power";
            if (list && desc) if (auto arr = list(blob)) {
                for (detail::CFIndex i = 0; i < cf.CFArrayGetCount(arr); ++i) {
                    auto d = desc(blob, cf.CFArrayGetValueAtIndex(arr, i)); if (!d) continue;
                    auto kcur = cf.str("Current Capacity"), kmax = cf.str("Max Capacity"), kch = cf.str("Is Charging");
                    long long cur = cf.to_ll(cf.CFDictionaryGetValue(d, kcur)), mx = cf.to_ll(cf.CFDictionaryGetValue(d, kmax));
                    if (cur >= 0 && mx > 0) { p.has_battery = true; p.battery_percent = int(cur * 100 / mx); }
                    p.charging = cf.CFDictionaryGetValue(d, kch) == cf.kCFBooleanTrue;
                    cf.CFRelease(kcur); cf.CFRelease(kmax); cf.CFRelease(kch);
                }
                cf.CFRelease(arr);
            }
            cf.CFRelease(blob);
        }
    }
#endif
#elif defined(FORGE_OS_FREEBSD)
    p.source = "acpi";
    long long life = detail::sysctl_ll("hw.acpi.battery.life", -1), ac = detail::sysctl_ll("hw.acpi.acline", -1), st = detail::sysctl_ll("hw.acpi.battery.state", -1);
    if (life >= 0) { p.has_battery = true; p.battery_percent = int(life); }
    if (ac >= 0) p.on_battery = ac == 0; if (st >= 0) p.charging = (st & 2) != 0;
    long long tz = detail::sysctl_ll("hw.acpi.thermal.tz0.temperature", -1), psv = detail::sysctl_ll("hw.acpi.thermal.tz0._PSV", -1);
    if (tz > 0) { p.temperature_c = double(tz) / 10.0 - 273.15; p.thermal = psv > 0 && tz >= psv ? thermal_state::serious : psv > 0 && tz >= psv - 100 ? thermal_state::fair : thermal_state::nominal; }
    p.throttled = p.thermal >= thermal_state::serious;
#endif
    return p;
}
} // namespace forge::os

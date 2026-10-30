// forge::os::locale: Apple hooks (CoreFoundation) for the system locale, preferred languages and time zone.
#include "locale_internal.hpp"
#include <CoreFoundation/CoreFoundation.h>
#include <cstring>

namespace forge::os::locale::detail {
namespace {
std::string to_std(CFStringRef s) {
    if (!s) return {};
    if (const char* p = CFStringGetCStringPtr(s, kCFStringEncodingUTF8)) return p;
    CFIndex len = CFStringGetMaximumSizeForEncoding(CFStringGetLength(s), kCFStringEncodingUTF8) + 1;
    std::string r(std::size_t(len), '\0');
    if (!CFStringGetCString(s, r.data(), len, kCFStringEncodingUTF8)) return {};
    r.resize(std::strlen(r.c_str())); return r;
}
}  // namespace
std::string apple_system_locale() {
    CFLocaleRef l = CFLocaleCopyCurrent(); if (!l) return {};
    std::string r = to_std(CFLocaleGetIdentifier(l));   // "en_AU", "zh-Hans_CN", "en_AU@calendar=japanese"
    CFRelease(l); return r;
}
std::vector<std::string> apple_preferred_languages() {
    std::vector<std::string> r;
    CFArrayRef a = CFLocaleCopyPreferredLanguages(); if (!a) return r;
    for (CFIndex i = 0, n = CFArrayGetCount(a); i < n; ++i) r.push_back(to_std((CFStringRef)CFArrayGetValueAtIndex(a, i)));
    CFRelease(a); return r;
}
std::string apple_local_zone() {
    CFTimeZoneRef z = CFTimeZoneCopySystem(); if (!z) return {};
    std::string r = to_std(CFTimeZoneGetName(z));
    CFRelease(z); return r;
}
}  // namespace forge::os::locale::detail

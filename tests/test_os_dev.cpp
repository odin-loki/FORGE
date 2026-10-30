// forge::os phase 4: devices, clipboard.
#include <forge/os/devices.hpp>
#include <forge/os/clipboard.hpp>
#include "check.hpp"
using namespace forge;

TEST(devices_enumerate) {
    auto b = os::devices::block(); auto p = os::devices::pci(); auto u = os::devices::usb();
#if defined(__linux__)
    CHECK(b);   // may be empty in containers, but readable
    for (auto& d : b.value) CHECK(!d.name.empty());
    if (p) for (auto& d : p.value) CHECK(d.vendor_id.rfind("0x", 0) == 0);
    if (u) for (auto& d : u.value) CHECK(d.vendor_id.size() == 4);
#else
    CHECK(!b ? os::is_unsupported(b.err) : true);
#endif
}
TEST(clipboard_roundtrip_or_unsupported) {
    if (!os::clipboard::available()) { CHECK(os::is_unsupported(os::clipboard::get().err)); return; }
    auto old = os::clipboard::get();
    CHECK(os::clipboard::set("forge clipboard ✓"));
    auto g = os::clipboard::get(); CHECK(g); CHECK(g.value == "forge clipboard ✓");
    if (old) os::clipboard::set(old.value);
}

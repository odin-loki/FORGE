// forge::os phase 4: devices, clipboard.
#include <forge/os/devices.hpp>
#include <forge/os/clipboard.hpp>
#include "check.hpp"
using namespace forge;

TEST(devices_enumerate) {
    auto b = os::devices::block(); auto p = os::devices::pci(); auto u = os::devices::usb();
    std::printf("  block %s, pci %s, usb %s\n", b ? std::to_string(b->size()).c_str() : b.err.what.c_str(), p ? std::to_string(p->size()).c_str() : p.err.what.c_str(),
                u ? std::to_string(u->size()).c_str() : u.err.what.c_str());
    for (auto& d : b.value) {
        std::printf("  disk %s %llu B '%s' '%s'%s%s%s parts:", d.name.c_str(), (unsigned long long)d.size_bytes, d.vendor.c_str(), d.model.c_str(), d.removable ? " removable" : "", d.rotational ? " rotational" : "", d.read_only ? " ro" : "");
        for (auto& x : d.partitions) std::printf(" %s", x.c_str());
        std::printf("\n"); CHECK(!d.name.empty());
    }
    for (auto& d : p.value) {
        CHECK(d.vendor_id.size() == 6 && d.vendor_id.rfind("0x", 0) == 0); CHECK(d.device_id.rfind("0x", 0) == 0);
        CHECK(d.class_id.empty() || d.class_id.size() == 8); CHECK(!d.address.empty());
    }
    if (!p.value.empty()) { auto& d = p.value[0]; std::printf("  pci %s %s:%s class %s driver '%s'\n", d.address.c_str(), d.vendor_id.c_str(), d.device_id.c_str(), d.class_id.c_str(), d.driver.c_str()); }
    for (auto& d : u.value) {
        CHECK(d.vendor_id.size() == 4 && d.product_id.size() == 4); CHECK(!d.path.empty());
        std::printf("  usb %s %s:%s '%s' '%s' bus %d dev %d\n", d.path.c_str(), d.vendor_id.c_str(), d.product_id.c_str(), d.manufacturer.c_str(), d.product.c_str(), d.bus, d.devnum);
    }
#if defined(__linux__) || defined(_WIN32) || defined(__FreeBSD__) || (defined(__APPLE__) && !TARGET_OS_IPHONE)
    CHECK(b);   // may be empty in containers, but readable
    CHECK(p || os::is_unsupported(p.err)); CHECK(u || os::is_unsupported(u.err));   // Linux containers may lack /sys/bus/*
#if !defined(__linux__)
    CHECK(p); CHECK(u);
#endif
#if defined(_WIN32) || defined(__FreeBSD__) || defined(__APPLE__)
    CHECK(!b.value.empty());   // a real or virtual machine always has a boot disk
#endif
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

#pragma once
// forge::os devices: read-only enumeration of block devices, PCI functions and USB devices.
// Linux/Android read sysfs; other platforms return unsupported() for now (SetupAPI / IOKit are planned).
// Guide: docs/os-wrapper.md.
#include <forge/os/os.hpp>
#include <algorithm>
#include <fstream>

namespace forge::os::devices {
struct block_device { std::string name, model, vendor, serial; std::uint64_t size_bytes = 0; bool removable = false, rotational = false, read_only = false; std::vector<std::string> partitions; };
struct pci_device { std::string address, vendor_id, device_id, class_id, driver; };
struct usb_device { std::string path, vendor_id, product_id, manufacturer, product, serial; int bus = 0, devnum = 0; };

#if defined(__linux__)
inline std::string sysfs_read_(const std::string& p) {
    std::ifstream f(p); std::string s; std::getline(f, s);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ' || s.back() == '\r')) s.pop_back();
    while (!s.empty() && s.front() == ' ') s.erase(s.begin());
    return s;
}
inline std::vector<std::string> sysfs_list_(const std::string& dir) {
    std::vector<std::string> v; auto l = list_dir(dir); if (!l) return v;
    for (auto& e : l.value) if (e.name != "." && e.name != "..") v.push_back(e.name);
    std::sort(v.begin(), v.end()); return v;
}
#endif

inline result<std::vector<block_device>> block() {
#if defined(__linux__)
    std::vector<block_device> out;
    for (auto& n : sysfs_list_("/sys/block")) {
        if (n.rfind("loop", 0) == 0 || n.rfind("ram", 0) == 0 || n.rfind("zram", 0) == 0) continue;
        std::string b = "/sys/block/" + n;
        block_device d; d.name = n;
        d.size_bytes = std::strtoull(sysfs_read_(b + "/size").c_str(), nullptr, 10) * 512ull;
        d.removable = sysfs_read_(b + "/removable") == "1"; d.rotational = sysfs_read_(b + "/queue/rotational") == "1"; d.read_only = sysfs_read_(b + "/ro") == "1";
        d.model = sysfs_read_(b + "/device/model"); d.vendor = sysfs_read_(b + "/device/vendor"); d.serial = sysfs_read_(b + "/device/serial");
        for (auto& p : sysfs_list_(b)) if (p.rfind(n, 0) == 0 && stat(b + "/" + p + "/partition").exists) d.partitions.push_back(p);
        out.push_back(std::move(d));
    }
    return {std::move(out), {}};
#else
    return {{}, unsupported("block device enumeration")};
#endif
}
inline result<std::vector<pci_device>> pci() {
#if defined(__linux__)
    std::vector<pci_device> out; std::string root = "/sys/bus/pci/devices";
    if (!stat(root).is_dir) return {{}, unsupported("pci (no /sys/bus/pci)")};
    for (auto& a : sysfs_list_(root)) {
        std::string b = root + "/" + a; pci_device d; d.address = a;
        d.vendor_id = sysfs_read_(b + "/vendor"); d.device_id = sysfs_read_(b + "/device"); d.class_id = sysfs_read_(b + "/class");
        char lk[512]; auto n = ::readlink((b + "/driver").c_str(), lk, sizeof lk - 1); if (n > 0) d.driver = basename(std::string(lk, std::size_t(n)));
        out.push_back(std::move(d));
    }
    return {std::move(out), {}};
#else
    return {{}, unsupported("pci enumeration")};
#endif
}
inline result<std::vector<usb_device>> usb() {
#if defined(__linux__)
    std::vector<usb_device> out; std::string root = "/sys/bus/usb/devices";
    if (!stat(root).is_dir) return {{}, unsupported("usb (no /sys/bus/usb)")};
    for (auto& a : sysfs_list_(root)) {
        if (a.find(':') != std::string::npos) continue;   // interfaces, not devices
        std::string b = root + "/" + a; usb_device d; d.path = a;
        d.vendor_id = sysfs_read_(b + "/idVendor"); d.product_id = sysfs_read_(b + "/idProduct"); if (d.vendor_id.empty()) continue;
        d.manufacturer = sysfs_read_(b + "/manufacturer"); d.product = sysfs_read_(b + "/product"); d.serial = sysfs_read_(b + "/serial");
        d.bus = std::atoi(sysfs_read_(b + "/busnum").c_str()); d.devnum = std::atoi(sysfs_read_(b + "/devnum").c_str());
        out.push_back(std::move(d));
    }
    return {std::move(out), {}};
#else
    return {{}, unsupported("usb enumeration")};
#endif
}
} // namespace forge::os::devices

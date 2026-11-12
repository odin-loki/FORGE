#pragma once
// forge::os::net network configuration: link-level interface details, default routes, DNS servers, the system
// proxy, and change notifications. Same API on every OS; what each one answers from:
//   links()           Linux/Android: getifaddrs (AF_PACKET) + /sys/class/net.  macOS/iOS/FreeBSD: getifaddrs
//                     (AF_LINK) + SIOCGIFMTU / SIOCGIFMEDIA.  Windows: GetAdaptersAddresses.
//   default_routes()  Linux/Android: /proc/net/route and /proc/net/ipv6_route.  macOS/FreeBSD: the routing sysctl
//                     (NET_RT_FLAGS, RTF_GATEWAY).  Windows: the adapters' gateway lists.  iOS: unsupported (the
//                     SDK has no <net/route.h>).
//   dns_servers()     resolv.conf (the systemd-resolved upstream list when resolv.conf only names the stub);
//                     macOS: SystemConfiguration "State:/Network/Global/DNS" first.  Windows: the adapters' DNS
//                     lists.  Android: the net.dnsN properties (gone in Android 8+: unsupported there).
//   system_proxy()    HTTP_PROXY / HTTPS_PROXY / ALL_PROXY / NO_PROXY first, on every OS.  Otherwise Windows: the
//                     user's WinINet settings (WinHttpGetIEProxyConfigForCurrentUser).  macOS/iOS:
//                     CFNetworkCopySystemProxySettings.  Linux/FreeBSD: GNOME settings through gsettings.
//                     Android: set_proxy_hook() (the Java system properties are only visible through JNI).
//   network_monitor   Linux/Android: an rtnetlink socket.  macOS/FreeBSD: a PF_ROUTE socket.  Windows:
//                     NotifyIpInterfaceChange / NotifyUnicastIpAddressChange / NotifyRouteChange2.  Where the
//                     kernel source is refused (Android apps from API 30, iOS, sandboxes) or with
//                     force_polling: a snapshot of links() compared once a second.
// System libraries (iphlpapi, winhttp, SystemConfiguration, CFNetwork) are loaded at run time.
#include <forge/os/net.hpp>
#include <forge/os/detail.hpp>
#include <forge/os/process.hpp>
#include <fstream>
#include <sstream>
#if defined(FORGE_OS_POSIX)
#include <net/if.h>
#include <sys/ioctl.h>
#endif
#if defined(__linux__)
#include <linux/if_packet.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#elif defined(__APPLE__) || defined(FORGE_OS_FREEBSD)
#include <net/if_dl.h>
#include <net/if_types.h>
#include <sys/sockio.h>
#if __has_include(<net/if_media.h>)
#include <net/if_media.h>
#define FORGE_NET_IFMEDIA 1
#endif
#if __has_include(<net/route.h>)
#include <net/route.h>
#define FORGE_NET_ROUTE_SOCKET 1
#endif
#endif

namespace forge::os::net {
struct link_address { std::string ip; int family = 4; int prefix = -1; };   // prefix: network prefix length, -1 unknown
struct link_info {
    std::string name;          // kernel name ("eth0", "en0"); Windows: the friendly name ("Ethernet")
    std::string id;            // Windows: the adapter GUID; elsewhere the name
    std::string description;   // Windows: the adapter description; "" elsewhere
    std::string mac;           // "aa:bb:cc:dd:ee:ff"; "" when the link has none or the OS hides it (Android, iOS)
    std::string type;          // loopback, ethernet, wifi, cellular, tunnel, bridge, ppp or other
    unsigned index = 0, mtu = 0;
    std::uint64_t speed_bps = 0;   // 0 unknown
    bool up = false, running = false, loopback = false;   // up: administratively; running: carrier / operational
    std::vector<link_address> addresses;
};
struct route_info { std::string gateway, interface_name; int family = 4; unsigned metric = 0; };

namespace detail_net {
inline std::string mac_text(const unsigned char* p, std::size_t n) {
    bool any = false; for (std::size_t i = 0; i < n; ++i) any |= p[i] != 0;
    if (!n || !any) return {};
    std::string s; char b[4];
    for (std::size_t i = 0; i < n; ++i) { std::snprintf(b, sizeof b, i ? ":%02x" : "%02x", unsigned(p[i])); s += b; }
    return s;
}
inline std::string sa_ip(const sockaddr* sa) {
    if (!sa) return {};
    char b[INET6_ADDRSTRLEN] = {};
    if (sa->sa_family == AF_INET) { if (!::inet_ntop(AF_INET, &reinterpret_cast<const sockaddr_in*>(sa)->sin_addr, b, sizeof b)) return {}; }
    else if (sa->sa_family == AF_INET6) { if (!::inet_ntop(AF_INET6, &reinterpret_cast<const sockaddr_in6*>(sa)->sin6_addr, b, sizeof b)) return {}; }
    else return {};
    return b;
}
inline int mask_prefix(const sockaddr* sa) {
    if (!sa) return -1;
    const unsigned char* p = nullptr; std::size_t n = 0;
    if (sa->sa_family == AF_INET) { p = reinterpret_cast<const unsigned char*>(&reinterpret_cast<const sockaddr_in*>(sa)->sin_addr); n = 4; }
    else if (sa->sa_family == AF_INET6) { p = reinterpret_cast<const unsigned char*>(&reinterpret_cast<const sockaddr_in6*>(sa)->sin6_addr); n = 16; }
    else return -1;
    int bits = 0; for (std::size_t i = 0; i < n; ++i) for (int k = 7; k >= 0; --k) { if (!(p[i] & (1u << k))) return bits; ++bits; }
    return bits;
}
// Linux ARPHRD_* value (/sys/class/net/<if>/type) and uevent DEVTYPE -> link type.
inline std::string linux_link_type(long arphrd, const std::string& devtype, bool wireless, const std::string& name) {
    if (arphrd == 772) return "loopback";
    if (wireless || devtype == "wlan" || arphrd == 801 || arphrd == 802 || arphrd == 803) return "wifi";
    if (devtype == "wwan" || name.rfind("rmnet", 0) == 0 || name.rfind("wwan", 0) == 0 || name.rfind("ccmni", 0) == 0) return "cellular";
    if (devtype == "bridge") return "bridge";
    if (arphrd == 512) return "ppp";
    if (arphrd == 65534 || arphrd == 768 || arphrd == 769 || arphrd == 776 || arphrd == 778 || arphrd == 823 || devtype == "wireguard" || devtype == "vxlan" || devtype == "geneve") return "tunnel";
    if (arphrd == 1) return "ethernet";
    return "other";
}
#if defined(FORGE_OS_WINDOWS)
inline HMODULE iphlpapi() { static HMODULE h = LoadLibraryExW(L"iphlpapi.dll", nullptr, 0x800 /*LOAD_LIBRARY_SEARCH_SYSTEM32*/); return h; }
// GetAdaptersAddresses into buf; the adapter list head or nullptr (err set unless the list is just empty).
inline IP_ADAPTER_ADDRESSES* adapters(std::vector<unsigned char>& buf, ULONG flags, error& err) {
    net_init();
    HMODULE h = iphlpapi(); if (!h) { err = last_error("iphlpapi.dll"); return nullptr; }
    using gaa_fn = ULONG(WINAPI*)(ULONG, ULONG, PVOID, IP_ADAPTER_ADDRESSES*, PULONG);
    auto gaa = reinterpret_cast<gaa_fn>(reinterpret_cast<void*>(GetProcAddress(h, "GetAdaptersAddresses")));
    if (!gaa) { err = unsupported("GetAdaptersAddresses"); return nullptr; }
    ULONG n = 16384, rc = 0;
    for (int k = 0; k < 4; ++k) {
        buf.assign(n, 0);
        rc = gaa(AF_UNSPEC, flags, nullptr, reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buf.data()), &n);
        if (rc != ERROR_BUFFER_OVERFLOW) break;
    }
    if (rc == ERROR_NO_DATA) return nullptr;
    if (rc == ERROR_NOT_SUPPORTED) { err = unsupported("GetAdaptersAddresses with these flags"); return nullptr; }   // wine: no gateway list
    if (rc != ERROR_SUCCESS) { err = error{int(rc), "GetAdaptersAddresses: " + error_text(int(rc))}; return nullptr; }
    return reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buf.data());
}
inline const char* win_if_type(DWORD t) {
    switch (t) { case 24: return "loopback"; case 6: return "ethernet"; case 71: return "wifi"; case 131: return "tunnel"; case 23: return "ppp";
                 case 243: case 244: return "cellular"; default: return "other"; }
}
#endif
} // namespace detail_net

// Every network interface with its link-level details and addresses.
inline result<std::vector<link_info>> links() {
    std::vector<link_info> out;
#if defined(FORGE_OS_WINDOWS)
    std::vector<unsigned char> buf; error err;
    auto* list = detail_net::adapters(buf, 0x2 | 0x4 | 0x8 /*SKIP_ANYCAST|SKIP_MULTICAST|SKIP_DNS_SERVER*/, err);
    if (!list && err) return {{}, err};
    for (auto* a = list; a; a = a->Next) {
        link_info l;
        l.name = a->FriendlyName ? narrow(a->FriendlyName) : std::string(a->AdapterName ? a->AdapterName : "");
        l.id = a->AdapterName ? a->AdapterName : ""; l.description = a->Description ? narrow(a->Description) : std::string();
        l.mac = detail_net::mac_text(a->PhysicalAddress, a->PhysicalAddressLength);
        l.type = detail_net::win_if_type(a->IfType); l.loopback = a->IfType == 24;
        l.index = a->IfIndex ? unsigned(a->IfIndex) : unsigned(a->Ipv6IfIndex); l.mtu = a->Mtu == 0xFFFFFFFFu ? 0u : unsigned(a->Mtu);
        l.running = a->OperStatus == IfOperStatusUp; l.up = a->OperStatus != IfOperStatusDown && a->OperStatus != IfOperStatusNotPresent;
        if (a->TransmitLinkSpeed != ~0ull && l.running) l.speed_bps = a->TransmitLinkSpeed;
        for (auto* u = a->FirstUnicastAddress; u; u = u->Next) {
            auto ip = detail_net::sa_ip(u->Address.lpSockaddr); if (ip.empty()) continue;
            l.addresses.push_back({ip, u->Address.lpSockaddr->sa_family == AF_INET ? 4 : 6, int(u->OnLinkPrefixLength)});
        }
        out.push_back(std::move(l));
    }
    return {std::move(out), {}};
#elif defined(FORGE_OS_POSIX)
    auto find = [&](const std::string& n) -> link_info& {
        for (auto& x : out) if (x.name == n) return x;
        link_info l; l.name = l.id = n; out.push_back(std::move(l)); return out.back();
    };
#if defined(FORGE_NET_IFADDRS)
    ifaddrs* ifs = nullptr; if (::getifaddrs(&ifs)) return {{}, last_error("getifaddrs")};
    for (auto* p = ifs; p; p = p->ifa_next) {
        if (!p->ifa_name) continue;
        link_info& l = find(p->ifa_name);
        l.up = (p->ifa_flags & IFF_UP) != 0; l.running = (p->ifa_flags & IFF_RUNNING) != 0; l.loopback = (p->ifa_flags & IFF_LOOPBACK) != 0;
        if (!p->ifa_addr) continue;
        int fam = p->ifa_addr->sa_family;
        if (fam == AF_INET || fam == AF_INET6) {
            auto ip = detail_net::sa_ip(p->ifa_addr); if (!ip.empty()) l.addresses.push_back({ip, fam == AF_INET ? 4 : 6, detail_net::mask_prefix(p->ifa_netmask)});
        }
#if defined(__linux__)
        else if (fam == AF_PACKET) {
            auto* ll = reinterpret_cast<const sockaddr_ll*>(p->ifa_addr);
            l.index = unsigned(ll->sll_ifindex); if (ll->sll_halen <= 8) l.mac = detail_net::mac_text(ll->sll_addr, ll->sll_halen);
        }
#else
        else if (fam == AF_LINK) {
            auto* dl = reinterpret_cast<const sockaddr_dl*>(p->ifa_addr);
            l.index = dl->sdl_index;
            if (dl->sdl_alen && dl->sdl_alen <= 8) l.mac = detail_net::mac_text(reinterpret_cast<const unsigned char*>(LLADDR(dl)), dl->sdl_alen);
            switch (dl->sdl_type) {
                case IFT_LOOP: l.type = "loopback"; break;
                case IFT_ETHER: l.type = "ethernet"; break;
                case IFT_PPP: l.type = "ppp"; break;
                case IFT_BRIDGE: l.type = "bridge"; break;
                case 0x47 /*IFT_IEEE80211*/: l.type = "wifi"; break;
                case 0xff /*IFT_CELLULAR (Apple)*/: l.type = "cellular"; break;
                case IFT_GIF: case IFT_STF: case 0xf5 /*IFT_PFLOG..*/: case 0x83 /*IFT_TUNNEL*/: l.type = "tunnel"; break;
                default: l.type = "other";
            }
        }
#endif
    }
    ::freeifaddrs(ifs);
#else
    if (auto v = interfaces()) for (auto& i : v.value) { link_info& l = find(i.name); l.up = l.running = i.up; l.loopback = i.loopback; for (auto& a : i.addresses) l.addresses.push_back({a.ip, a.family, -1}); }
#endif
    int fd = cloexec_socket(AF_INET, SOCK_DGRAM);
    for (auto& l : out) {
        if (!l.index) l.index = ::if_nametoindex(l.name.c_str());
        if (fd >= 0 && l.name.size() < IFNAMSIZ) {
            ifreq r{}; std::memcpy(r.ifr_name, l.name.c_str(), l.name.size());
            if (::ioctl(fd, SIOCGIFMTU, &r) == 0 && r.ifr_mtu > 0) l.mtu = unsigned(r.ifr_mtu);
#if defined(FORGE_NET_IFMEDIA)
            ifmediareq m{}; std::memcpy(m.ifm_name, l.name.c_str(), l.name.size());
            if (::ioctl(fd, SIOCGIFMEDIA, &m) == 0) {
                if (IFM_TYPE(m.ifm_current) == IFM_IEEE80211) l.type = "wifi";
                if (m.ifm_status & IFM_AVALID) l.running = l.up && (m.ifm_status & IFM_ACTIVE) != 0;
            }
#endif
        }
#if defined(__linux__)
        std::string d = "/sys/class/net/" + l.name;
        long ty = long(detail::slurp_ll(d + "/type", -1)); auto ue = detail::slurp(d + "/uevent"); std::string devtype;
        if (auto q = ue.find("DEVTYPE="); q != std::string::npos) devtype = ue.substr(q + 8, ue.find('\n', q) - q - 8);
        bool wireless = stat(d + "/wireless").exists || stat(d + "/phy80211").exists;
        l.type = l.loopback ? "loopback" : ty < 0 && devtype.empty() && !wireless ? std::string("other") : detail_net::linux_link_type(ty, devtype, wireless, l.name);
        if (ty < 0 && !l.loopback && (l.name.rfind("wlan", 0) == 0)) l.type = "wifi";   // Android apps cannot read sysfs: go by the name
        if (ty < 0 && !l.loopback && (l.name.rfind("rmnet", 0) == 0 || l.name.rfind("ccmni", 0) == 0)) l.type = "cellular";
        if (ty < 0 && !l.loopback && l.name.rfind("eth", 0) == 0) l.type = "ethernet";
        long long sp = detail::slurp_ll(d + "/speed", -1); if (sp > 0) l.speed_bps = std::uint64_t(sp) * 1000000ull;
        if (l.mac.empty()) { auto m = detail::trim(detail::slurp(d + "/address")); if (m.size() == 17 && m != "00:00:00:00:00:00") l.mac = m; }
        auto oper = detail::trim(detail::slurp(d + "/operstate")); if (oper == "down" || oper == "lowerlayerdown" || oper == "notpresent") l.running = false;
#else
        if (l.type.empty()) l.type = l.loopback ? "loopback" : "other";
#endif
    }
    if (fd >= 0) ::close(fd);
    return {std::move(out), {}};
#else
    return {{}, unsupported("network links")};
#endif
}

namespace detail_net {
// /proc/net/route text -> IPv4 default routes ("Iface Destination Gateway Flags RefCnt Use Metric Mask ...", hex, host order).
inline std::vector<route_info> parse_proc_route(const std::string& text) {
    std::vector<route_info> v; std::istringstream in(text); std::string line; std::getline(in, line);
    while (std::getline(in, line)) {
        std::istringstream ls(line); std::string ifn, dst, gw, flags, refcnt, use, metric, mask;
        if (!(ls >> ifn >> dst >> gw >> flags >> refcnt >> use >> metric >> mask)) continue;
        unsigned long fl = std::strtoul(flags.c_str(), nullptr, 16), d = std::strtoul(dst.c_str(), nullptr, 16), m = std::strtoul(mask.c_str(), nullptr, 16), g = std::strtoul(gw.c_str(), nullptr, 16);
        if (!(fl & 0x1) || !(fl & 0x2) || d != 0 || m != 0) continue;   // RTF_UP, RTF_GATEWAY, 0.0.0.0/0
        in_addr a{}; std::uint32_t raw = std::uint32_t(g); std::memcpy(&a, &raw, 4);   // the file prints the in-memory (network order) word
        char b[INET_ADDRSTRLEN] = {}; if (!::inet_ntop(AF_INET, &a, b, sizeof b)) continue;
        v.push_back({b, ifn, 4, unsigned(std::strtoul(metric.c_str(), nullptr, 10))});
    }
    return v;
}
// /proc/net/ipv6_route: "dst(32 hex) plen src splen nexthop(32 hex) metric refcnt use flags iface".
inline std::vector<route_info> parse_proc_ipv6_route(const std::string& text) {
    std::vector<route_info> v; std::istringstream in(text); std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line); std::string dst, plen, src, splen, nh, metric, refcnt, use, flags, ifn;
        if (!(ls >> dst >> plen >> src >> splen >> nh >> metric >> refcnt >> use >> flags >> ifn)) continue;
        if (dst.size() != 32 || nh.size() != 32 || dst.find_first_not_of('0') != std::string::npos || plen != "00") continue;
        unsigned long fl = std::strtoul(flags.c_str(), nullptr, 16); if (!(fl & 0x1) || !(fl & 0x2)) continue;
        in6_addr a{}; bool ok = true;
        for (int i = 0; i < 16 && ok; ++i) { int hi = hexv(static_cast<unsigned char>(nh[std::size_t(i) * 2])), lo = hexv(static_cast<unsigned char>(nh[std::size_t(i) * 2 + 1])); if (hi < 0 || lo < 0) ok = false; else a.s6_addr[i] = static_cast<unsigned char>(hi * 16 + lo); }
        char b[INET6_ADDRSTRLEN] = {}; if (!ok || !::inet_ntop(AF_INET6, &a, b, sizeof b)) continue;
        v.push_back({b, ifn, 6, unsigned(std::strtoul(metric.c_str(), nullptr, 16))});
    }
    return v;
}
// "nameserver <ip>" lines of a resolv.conf (a %zone suffix is kept).
inline std::vector<std::string> parse_resolv_conf(const std::string& text) {
    std::vector<std::string> v; std::istringstream in(text); std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line); std::string k, ip; if (!(ls >> k >> ip) || k != "nameserver") continue;
        bool dup = false; for (auto& x : v) dup |= x == ip; if (!dup) v.push_back(ip);
    }
    return v;
}
} // namespace detail_net

// Default gateways (0.0.0.0/0 and ::/0), lowest metric first where the OS reports one. No default route is an
// empty list, not an error.
inline result<std::vector<route_info>> default_routes() {
    std::vector<route_info> out;
#if defined(__linux__)
    auto r4 = detail::slurp("/proc/net/route"), r6 = detail::slurp("/proc/net/ipv6_route");
    if (r4.empty() && r6.empty() && !stat("/proc/net/route").exists) return {{}, unsupported("default routes (/proc/net/route not readable)")};
    out = detail_net::parse_proc_route(r4);
    for (auto& r : detail_net::parse_proc_ipv6_route(r6)) out.push_back(std::move(r));
#elif defined(FORGE_OS_WINDOWS)
    std::vector<unsigned char> buf; error err;
    auto* list = detail_net::adapters(buf, 0x2 | 0x4 | 0x8 | 0x80 /*INCLUDE_GATEWAYS*/, err);
    if (!list && err) return {{}, err};
    for (auto* a = list; a; a = a->Next) {
        if (a->OperStatus != IfOperStatusUp) continue;
        for (auto* g = a->FirstGatewayAddress; g; g = g->Next) {
            auto ip = detail_net::sa_ip(g->Address.lpSockaddr); if (ip.empty()) continue;
            bool v4 = g->Address.lpSockaddr->sa_family == AF_INET;
            out.push_back({ip, a->FriendlyName ? narrow(a->FriendlyName) : std::string(), v4 ? 4 : 6, unsigned(v4 ? a->Ipv4Metric : a->Ipv6Metric)});
        }
    }
#elif defined(FORGE_NET_ROUTE_SOCKET)
    for (int af : {AF_INET, AF_INET6}) {
        int mib[6] = {CTL_NET, PF_ROUTE, 0, af, NET_RT_FLAGS, RTF_GATEWAY};
        std::size_t len = 0; if (::sysctl(mib, 6, nullptr, &len, nullptr, 0) || !len) continue;
        std::vector<char> buf(len + 4096); len = buf.size();
        if (::sysctl(mib, 6, buf.data(), &len, nullptr, 0)) continue;
        for (std::size_t off = 0; off + sizeof(rt_msghdr) <= len;) {
            auto* rtm = reinterpret_cast<const rt_msghdr*>(buf.data() + off);
            if (rtm->rtm_msglen < sizeof(rt_msghdr) || off + rtm->rtm_msglen > len) break;
            const char* p = reinterpret_cast<const char*>(rtm + 1); const char* end = buf.data() + off + rtm->rtm_msglen;
            off += rtm->rtm_msglen;
            if (!(rtm->rtm_flags & RTF_GATEWAY) || !(rtm->rtm_flags & RTF_UP)) continue;
            const sockaddr* sa[RTAX_MAX] = {};
            for (int i = 0; i < RTAX_MAX && p + 2 <= end; ++i) {
                if (!(rtm->rtm_addrs & (1 << i))) continue;
                auto* s = reinterpret_cast<const sockaddr*>(p); sa[i] = s;
#if defined(__APPLE__)
                std::size_t adv = s->sa_len ? 1 + ((std::size_t(s->sa_len) - 1) | (sizeof(std::uint32_t) - 1)) : sizeof(std::uint32_t);
#else
                std::size_t adv = s->sa_len ? 1 + ((std::size_t(s->sa_len) - 1) | (sizeof(long) - 1)) : sizeof(long);
#endif
                p += adv;
            }
            const sockaddr* dst = sa[RTAX_DST]; const sockaddr* gw = sa[RTAX_GATEWAY]; const sockaddr* mask = sa[RTAX_NETMASK];
            if (!dst || !gw || dst->sa_family != af || gw->sa_family != af) continue;
            if (reinterpret_cast<const char*>(gw) + (af == AF_INET ? sizeof(sockaddr_in) : sizeof(sockaddr_in6)) > end) continue;
            bool any = true;
            if (af == AF_INET) any = reinterpret_cast<const sockaddr_in*>(dst)->sin_addr.s_addr == 0;
            else for (unsigned char c : reinterpret_cast<const sockaddr_in6*>(dst)->sin6_addr.s6_addr) any &= c == 0;
            if (!any) continue;
            if (mask && mask->sa_len > 4) {   // a default route has an empty or all-zero mask
                bool zero = true; auto* mb = reinterpret_cast<const unsigned char*>(mask);
                for (std::size_t i = (af == AF_INET ? offsetof(sockaddr_in, sin_addr) : offsetof(sockaddr_in6, sin6_addr)); i < mask->sa_len && reinterpret_cast<const char*>(mb + i) < end; ++i) zero &= mb[i] == 0;
                if (!zero) continue;
            }
            std::string ip;
            if (af == AF_INET6) {   // KAME keeps the scope of a link-local address in bytes 2-3
                sockaddr_in6 g6; std::memcpy(&g6, gw, sizeof g6);
                if (g6.sin6_addr.s6_addr[0] == 0xfe && (g6.sin6_addr.s6_addr[1] & 0xc0) == 0x80) g6.sin6_addr.s6_addr[2] = g6.sin6_addr.s6_addr[3] = 0;
                ip = detail_net::sa_ip(reinterpret_cast<const sockaddr*>(&g6));
            } else ip = detail_net::sa_ip(gw);
            if (ip.empty()) continue;
            char nm[IFNAMSIZ] = {}; ::if_indextoname(rtm->rtm_index, nm);
            out.push_back({ip, nm, af == AF_INET ? 4 : 6, 0});
        }
    }
#else
    return {{}, unsupported("default routes")};
#endif
    std::stable_sort(out.begin(), out.end(), [](const route_info& a, const route_info& b) { return a.metric < b.metric; });
    return {std::move(out), {}};
}

// The DNS servers the system resolver is configured with, in order. An empty list is possible (no network).
inline result<std::vector<std::string>> dns_servers() {
    std::vector<std::string> out;
#if defined(FORGE_OS_WINDOWS)
    std::vector<unsigned char> buf; error err;
    auto* list = detail_net::adapters(buf, 0x2 | 0x4 /*SKIP_ANYCAST|SKIP_MULTICAST*/, err);
    if (!list && err) return {{}, err};
    for (auto* a = list; a; a = a->Next) {
        if (a->OperStatus != IfOperStatusUp || a->IfType == 24) continue;
        for (auto* d = a->FirstDnsServerAddress; d; d = d->Next) {
            auto ip = detail_net::sa_ip(d->Address.lpSockaddr); if (ip.empty() || ip.rfind("fec0:0:0:ffff::", 0) == 0) continue;   // deprecated site-local placeholders
            bool dup = false; for (auto& x : out) dup |= x == ip; if (!dup) out.push_back(ip);
        }
    }
    return {std::move(out), {}};
#elif defined(FORGE_OS_ANDROID)
    for (const char* k : {"net.dns1", "net.dns2", "net.dns3", "net.dns4"}) { auto v = detail::android_prop(k); if (!v.empty()) out.push_back(v); }
    if (out.empty()) return {{}, unsupported("dns servers (net.dnsN properties are not readable on Android 8+; use ConnectivityManager)")};
    return {std::move(out), {}};
#elif defined(FORGE_OS_POSIX)
#if defined(FORGE_OS_MACOS)
    {   // configd's view: what the resolver uses even when /etc/resolv.conf is stale
        auto& cf = detail::cf_api::get();
        static void* sc = ::dlopen("/System/Library/Frameworks/SystemConfiguration.framework/SystemConfiguration", RTLD_LAZY);
        auto copy = sc ? reinterpret_cast<detail::CFTypeRef (*)(const void*, detail::CFStringRef)>(::dlsym(sc, "SCDynamicStoreCopyValue")) : nullptr;
        if (cf.ok && copy && cf.CFArrayGetCount && cf.CFArrayGetValueAtIndex && cf.CFDictionaryGetValue) {
            auto key = cf.str("State:/Network/Global/DNS");
            if (auto d = copy(nullptr, key)) {
                auto k2 = cf.str("ServerAddresses");
                if (auto arr = cf.CFDictionaryGetValue(d, k2))
                    for (detail::CFIndex i = 0; i < cf.CFArrayGetCount(arr); ++i) { auto s = cf.to_string(cf.CFArrayGetValueAtIndex(arr, i)); if (!s.empty()) out.push_back(s); }
                cf.CFRelease(k2); cf.CFRelease(d);
            }
            cf.CFRelease(key);
        }
        if (!out.empty()) return {std::move(out), {}};
    }
#endif
    if (!stat("/etc/resolv.conf").exists) return {{}, unsupported("dns servers (no /etc/resolv.conf)")};
    out = detail_net::parse_resolv_conf(detail::slurp("/etc/resolv.conf"));
    if (out.size() == 1 && out[0] == "127.0.0.53") {   // systemd-resolved stub: report its upstream servers
        auto up = detail_net::parse_resolv_conf(detail::slurp("/run/systemd/resolve/resolv.conf"));
        if (!up.empty()) out = std::move(up);
    }
    return {std::move(out), {}};
#else
    return {{}, unsupported("dns servers")};
#endif
}

// ---------------------------------------------------------------- proxies
// Proxy URLs carry their scheme ("http://proxy:3128", "socks5://host:1080"); a bare "host:port" from the OS
// settings is given the http:// (or socks5://) scheme. bypass holds NO_PROXY-style rules, see proxy_bypassed().
struct proxy_settings {
    std::string http, https, socks, pac_url; std::vector<std::string> bypass; bool auto_detect = false;
    std::string source = "none";   // env, wininet, cfnetwork, gsettings, hook or none
    bool any() const { return !http.empty() || !https.empty() || !socks.empty() || !pac_url.empty() || auto_detect; }
};
namespace detail_net {
inline std::vector<std::string> split_list(const std::string& s, const char* seps) {
    std::vector<std::string> v; std::string cur;
    for (char c : s) { if (std::strchr(seps, c)) { cur = detail::trim(cur); if (!cur.empty()) v.push_back(cur); cur.clear(); } else cur += c; }
    cur = detail::trim(cur); if (!cur.empty()) v.push_back(cur); return v;
}
inline std::string with_scheme(const std::string& hostport, const char* scheme) {
    if (hostport.empty() || hostport.find("://") != std::string::npos) return hostport;
    return std::string(scheme) + "://" + hostport;
}
// WinINet proxy string: "host:port" (every protocol) or "http=h:p;https=h:p;socks=h:p".
inline void parse_wininet_proxy(const std::string& s, proxy_settings& p) {
    for (auto& part : split_list(s, "; \t")) {
        auto eq = part.find('=');
        if (eq == std::string::npos) { if (p.http.empty()) p.http = with_scheme(part, "http"); if (p.https.empty()) p.https = with_scheme(part, "http"); continue; }
        std::string k = lower(part.substr(0, eq)), v = part.substr(eq + 1);
        if (k == "http") p.http = with_scheme(v, "http"); else if (k == "https") p.https = with_scheme(v, "http"); else if (k == "socks") p.socks = with_scheme(v, "socks5");
    }
}
// `gsettings list-recursively org.gnome.system.proxy` output.
inline proxy_settings parse_gsettings_proxy(const std::string& text) {
    proxy_settings p; std::map<std::string, std::string> kv; std::istringstream in(text); std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line); std::string schema, key; if (!(ls >> schema >> key)) continue;
        std::string val; std::getline(ls, val); kv[schema + " " + key] = detail::trim(val);
    }
    auto unq = [](std::string s) { if (s.size() >= 2 && s.front() == '\'' && s.back() == '\'') s = s.substr(1, s.size() - 2); return s; };
    std::string mode = unq(kv["org.gnome.system.proxy mode"]);
    auto hp = [&](const char* sub, const char* scheme) -> std::string {
        std::string h = unq(kv[std::string("org.gnome.system.proxy.") + sub + " host"]), port = kv[std::string("org.gnome.system.proxy.") + sub + " port"];
        if (h.empty()) return {};
        return std::string(scheme) + "://" + (h.find(':') != std::string::npos && h[0] != '[' ? "[" + h + "]" : h) + (port.empty() || port == "0" ? "" : ":" + port);
    };
    if (mode == "manual") { p.http = hp("http", "http"); p.https = hp("https", "http"); p.socks = hp("socks", "socks5"); }
    else if (mode == "auto") { p.pac_url = unq(kv["org.gnome.system.proxy autoconfig-url"]); p.auto_detect = p.pac_url.empty(); }
    else return p;
    std::string ig = kv["org.gnome.system.proxy ignore-hosts"];   // ['localhost', '127.0.0.0/8']
    for (auto& e : split_list(ig, ",[]")) { auto x = unq(detail::trim(e)); if (!x.empty() && x != "@as") p.bypass.push_back(x); }
    p.source = "gsettings"; return p;
}
inline std::function<proxy_settings()>& proxy_hook() { static std::function<proxy_settings()> h; return h; }
inline std::mutex& proxy_hook_mutex() { static std::mutex m; return m; }
} // namespace detail_net

// Platform glue (Android JNI: ProxySelector / the http.proxyHost properties) can supply the settings.
inline void set_proxy_hook(std::function<proxy_settings()> h) { std::lock_guard<std::mutex> g(detail_net::proxy_hook_mutex()); detail_net::proxy_hook() = std::move(h); }

// The conventional variables, lower case first (as curl does); HTTP_PROXY in upper case is not read when
// REQUEST_METHOD is set (CGI: the header "Proxy:" arrives under that name).
inline proxy_settings proxy_from_env() {
    proxy_settings p;
    auto env = [](const char* lo, const char* up) { auto v = getenv(lo); return v.empty() && up ? getenv(up) : v; };
    p.http = env("http_proxy", has_env("REQUEST_METHOD") ? nullptr : "HTTP_PROXY"); p.https = env("https_proxy", "HTTPS_PROXY");
    std::string all = env("all_proxy", "ALL_PROXY");
    if (!all.empty()) { if (detail_net::lower(all).rfind("socks", 0) == 0) p.socks = all; if (p.http.empty()) p.http = all; if (p.https.empty()) p.https = all; }
    p.http = detail_net::with_scheme(p.http, "http"); p.https = detail_net::with_scheme(p.https, "http");
    p.bypass = detail_net::split_list(env("no_proxy", "NO_PROXY"), ", \t");
    if (p.any()) p.source = "env";
    return p;
}
inline proxy_settings system_proxy() {
    proxy_settings p = proxy_from_env(); if (p.any()) return p;
    {
        std::function<proxy_settings()> h; { std::lock_guard<std::mutex> g(detail_net::proxy_hook_mutex()); h = detail_net::proxy_hook(); }
        if (h) { p = h(); if (p.source == "none") p.source = "hook"; return p; }
    }
#if defined(FORGE_OS_WINDOWS)
    struct ie_config { BOOL auto_detect; LPWSTR pac, proxy, bypass; };
    static HMODULE h = LoadLibraryExW(L"winhttp.dll", nullptr, 0x800);
    auto f = h ? reinterpret_cast<BOOL(WINAPI*)(ie_config*)>(reinterpret_cast<void*>(GetProcAddress(h, "WinHttpGetIEProxyConfigForCurrentUser"))) : nullptr;
    ie_config c{};
    if (f && f(&c)) {
        p.auto_detect = c.auto_detect != 0;
        if (c.pac) { p.pac_url = narrow(c.pac); GlobalFree(c.pac); }
        if (c.proxy) { detail_net::parse_wininet_proxy(narrow(c.proxy), p); GlobalFree(c.proxy); }
        if (c.bypass) { p.bypass = detail_net::split_list(narrow(c.bypass), "; \t"); GlobalFree(c.bypass); }
        if (p.any()) p.source = "wininet";
    }
#elif defined(__APPLE__)
    auto& cf = detail::cf_api::get();
    static void* cn = ::dlopen("/System/Library/Frameworks/CFNetwork.framework/CFNetwork", RTLD_LAZY);
    auto copy = cn ? reinterpret_cast<detail::CFDictionaryRef (*)()>(::dlsym(cn, "CFNetworkCopySystemProxySettings")) : nullptr;
    if (cf.ok && copy && cf.CFDictionaryGetValue && cf.CFArrayGetCount && cf.CFArrayGetValueAtIndex) if (auto d = copy()) {
        auto val = [&](const char* k) { auto ks = cf.str(k); auto v = cf.CFDictionaryGetValue(d, ks); cf.CFRelease(ks); return v; };
        auto on = [&](const char* k) { return cf.to_ll(val(k), 0) != 0; };
        auto hp = [&](const char* host, const char* port, const char* scheme) -> std::string {
            auto h2 = cf.to_string(val(host)); if (h2.empty()) return {};
            long long pt = cf.to_ll(val(port), 0);
            return std::string(scheme) + "://" + (h2.find(':') != std::string::npos && h2[0] != '[' ? "[" + h2 + "]" : h2) + (pt > 0 ? ":" + std::to_string(pt) : "");
        };
        if (on("HTTPEnable")) p.http = hp("HTTPProxy", "HTTPPort", "http");
        if (on("HTTPSEnable")) p.https = hp("HTTPSProxy", "HTTPSPort", "http");
        if (on("SOCKSEnable")) p.socks = hp("SOCKSProxy", "SOCKSPort", "socks5");
        if (on("ProxyAutoConfigEnable")) p.pac_url = cf.to_string(val("ProxyAutoConfigURLString"));
        p.auto_detect = on("ProxyAutoDiscoveryEnable");
        if (auto ex = val("ExceptionsList"))
            for (detail::CFIndex i = 0; i < cf.CFArrayGetCount(ex); ++i) { auto s = cf.to_string(cf.CFArrayGetValueAtIndex(ex, i)); if (!s.empty()) p.bypass.push_back(s); }
        cf.CFRelease(d);
        if (p.any()) p.source = "cfnetwork";
    }
#elif defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)
    if ((has_env("DBUS_SESSION_BUS_ADDRESS") || has_env("XDG_CURRENT_DESKTOP")) && !find_executable("gsettings").empty()) {
        spawn_spec sp; sp.argv = {"gsettings", "list-recursively", "org.gnome.system.proxy"}; sp.timeout_ms = 3000;
        if (auto r = spawn_wait(sp); r && r->exit_code == 0) { auto g = detail_net::parse_gsettings_proxy(r->out); if (g.any()) return g; }
    }
#endif
    return p;
}

// Does a bypass list (NO_PROXY rules) cover host? Rules: "*" everything; "<local>" names without a dot;
// "example.com" / ".example.com" / "*.example.com" the domain and its subdomains; an IP literal exactly;
// "10.0.0.0/8" an IPv4 network; a ":port" suffix on a rule is ignored.
inline bool proxy_bypassed(const std::string& host_in, const std::vector<std::string>& bypass) {
    std::string host = detail_net::lower(host_in);
    if (host.size() >= 2 && host.front() == '[' && host.back() == ']') host = host.substr(1, host.size() - 2);
    while (!host.empty() && host.back() == '.') host.pop_back();
    in_addr ha{}; bool host_v4 = ::inet_pton(AF_INET, host.c_str(), &ha) == 1;
    for (auto rule : bypass) {
        rule = detail_net::lower(detail::trim(rule)); if (rule.empty()) continue;
        if (rule == "*") return true;
        if (rule == "<local>") { if (host.find('.') == std::string::npos && host.find(':') == std::string::npos) return true; continue; }
        if (auto sl = rule.find('/'); sl != std::string::npos) {
            in_addr na{}; int bits = std::atoi(rule.c_str() + sl + 1);
            if (host_v4 && ::inet_pton(AF_INET, rule.substr(0, sl).c_str(), &na) == 1 && bits >= 0 && bits <= 32) {
                std::uint32_t m = bits == 0 ? 0u : htonl(~std::uint32_t(0) << (32 - bits));
                if ((ha.s_addr & m) == (na.s_addr & m)) return true;
            }
            continue;
        }
        if (rule[0] == '[') { auto e = rule.find(']'); if (e != std::string::npos) rule = rule.substr(1, e - 1); }
        else if (std::count(rule.begin(), rule.end(), ':') == 1) rule = rule.substr(0, rule.find(':'));   // host:port
        if (rule.rfind("*.", 0) == 0) rule = rule.substr(1);
        if (rule.empty()) continue;
        if (rule == host) return true;
        std::string suffix = rule[0] == '.' ? rule : "." + rule;
        if (!host_v4 && host.size() > suffix.size() && host.compare(host.size() - suffix.size(), suffix.size(), suffix) == 0) return true;
        if (rule[0] == '.' && host == rule.substr(1)) return true;
    }
    return false;
}
// The proxy URL to use for `target` ("" = connect directly). PAC scripts are not evaluated: with only pac_url or
// auto_detect set the answer is "".
inline std::string proxy_for(const std::string& target, const proxy_settings& p) {
    auto u = parse_url(target); if (!u) return {};
    if (proxy_bypassed(u->host, p.bypass)) return {};
    const std::string& pick = u->scheme == "https" || u->scheme == "wss" ? p.https : p.http;
    return !pick.empty() ? pick : p.socks;
}

// ---------------------------------------------------------------- change notifications
enum class net_change_kind { address, link, route, other };
inline const char* net_change_name(net_change_kind k) {
    switch (k) { case net_change_kind::address: return "address"; case net_change_kind::link: return "link"; case net_change_kind::route: return "route"; default: return "other"; }
}
struct net_change { net_change_kind kind = net_change_kind::other; std::string interface_name; unsigned index = 0; };

namespace detail_net {
// What changed between two links() snapshots: interfaces that appeared, vanished or changed state are `link`
// changes, a different address list is an `address` change.
inline std::vector<net_change> diff_links(const std::vector<link_info>& a, const std::vector<link_info>& b) {
    std::vector<net_change> out;
    auto addrs = [](const link_info& l) { std::vector<std::string> v; for (auto& x : l.addresses) v.push_back(x.ip + "/" + std::to_string(x.prefix)); std::sort(v.begin(), v.end()); return v; };
    for (auto& n : b) {
        const link_info* o = nullptr; for (auto& x : a) if (x.name == n.name) o = &x;
        if (!o) { out.push_back({net_change_kind::link, n.name, n.index}); continue; }
        if (o->up != n.up || o->running != n.running || o->mac != n.mac || o->mtu != n.mtu) out.push_back({net_change_kind::link, n.name, n.index});
        if (addrs(*o) != addrs(n)) out.push_back({net_change_kind::address, n.name, n.index});
    }
    for (auto& o : a) { bool still = false; for (auto& n : b) still |= n.name == o.name; if (!still) out.push_back({net_change_kind::link, o.name, o.index}); }
    return out;
}
} // namespace detail_net

struct network_monitor_options { bool force_polling = false; int poll_interval_ms = 1000; };
class network_monitor {
public:
    using options = network_monitor_options;
    network_monitor(const network_monitor&) = delete; network_monitor& operator=(const network_monitor&) = delete;
    static result<std::unique_ptr<network_monitor>> open(options o = {}) {
        std::unique_ptr<network_monitor> m(new network_monitor); m->opt_ = o; if (m->opt_.poll_interval_ms < 10) m->opt_.poll_interval_ms = 10;
        if (!o.force_polling) m->open_native_();
        if (!m->native_) {
            auto l = links(); if (!l) return {{}, l.err};
            m->snap_ = std::move(l.value); m->last_sample_ = monotonic_ns();
        }
        return {std::move(m), {}};
    }
    // netlink, route_socket, iphlpapi or polling.
    const char* backend() const { return native_ ? native_name_ : "polling"; }
    // Changes since the last call; waits up to timeout_ms for the first one. A burst is returned together.
    std::vector<net_change> poll(int timeout_ms) {
        std::vector<net_change> out;
        if (!native_) { poll_snapshot_(timeout_ms, out); return out; }
#if defined(FORGE_OS_WINDOWS)
        { std::lock_guard<std::mutex> g(st_->m); out.swap(st_->q); }
        if (out.empty() && timeout_ms != 0) {
            WaitForSingleObject(st_->ev, timeout_ms < 0 ? INFINITE : DWORD(timeout_ms));
            std::lock_guard<std::mutex> g(st_->m); out.swap(st_->q);
        }
#elif defined(FORGE_OS_POSIX)
        pollfd pf{fd_, POLLIN, 0}; int wait = timeout_ms;
        while (::poll(&pf, 1, wait) > 0 && (pf.revents & POLLIN)) {
            char buf[8192]; auto n = ::recv(fd_, buf, sizeof buf, MSG_DONTWAIT);
            if (n <= 0) break;
            parse_(buf, std::size_t(n), out);
            wait = out.empty() ? wait : 20;   // gather the rest of a burst
            if (out.size() > 4096) break;
        }
#endif
        return out;
    }
    ~network_monitor() {
#if defined(FORGE_OS_WINDOWS)
        if (st_) {
            auto cancel = detail_net::iphlpapi() ? reinterpret_cast<DWORD(WINAPI*)(HANDLE)>(reinterpret_cast<void*>(GetProcAddress(detail_net::iphlpapi(), "CancelMibChangeNotify2"))) : nullptr;
            for (HANDLE h : st_->handles) if (h && cancel) cancel(h);   // returns after running callbacks have finished
            CloseHandle(st_->ev);
        }
#elif defined(FORGE_OS_POSIX)
        if (fd_ >= 0) ::close(fd_);
#endif
    }
private:
    network_monitor() = default;
    void poll_snapshot_(int timeout_ms, std::vector<net_change>& out) {
        const std::uint64_t interval = std::uint64_t(opt_.poll_interval_ms) * 1000000ull;
        const std::uint64_t end = timeout_ms < 0 ? ~std::uint64_t(0) : monotonic_ns() + std::uint64_t(timeout_ms) * 1000000ull;
        for (;;) {
            std::uint64_t now = monotonic_ns();
            if (now - last_sample_ >= interval || timeout_ms == 0) {
                if (auto l = links()) { out = detail_net::diff_links(snap_, l.value); snap_ = std::move(l.value); }
                last_sample_ = now;
                if (!out.empty()) return;
            }
            if (now >= end) return;
            std::uint64_t next = last_sample_ + interval; sleep_ns(std::min(next > now ? next - now : interval, end - now));
        }
    }
    void open_native_() {
#if defined(__linux__)
        fd_ = ::socket(AF_NETLINK, SOCK_RAW | SOCK_CLOEXEC, NETLINK_ROUTE); if (fd_ < 0) return;
        sockaddr_nl sa{}; sa.nl_family = AF_NETLINK; sa.nl_groups = RTMGRP_LINK | RTMGRP_IPV4_IFADDR | RTMGRP_IPV6_IFADDR | RTMGRP_IPV4_ROUTE | RTMGRP_IPV6_ROUTE;
        if (::bind(fd_, reinterpret_cast<sockaddr*>(&sa), sizeof sa)) { ::close(fd_); fd_ = -1; return; }   // Android apps (API 30+): EACCES -> polling
        native_ = true; native_name_ = "netlink";
#elif defined(FORGE_NET_ROUTE_SOCKET)
        fd_ = cloexec_socket(PF_ROUTE, SOCK_RAW, 0); if (fd_ < 0) return;
        native_ = true; native_name_ = "route_socket";
#elif defined(FORGE_OS_WINDOWS)
        HMODULE h = detail_net::iphlpapi(); if (!h) return;
        using cb_fn = void(WINAPI*)(void*, void*, int);
        using reg_fn = DWORD(WINAPI*)(ADDRESS_FAMILY, cb_fn, void*, BOOLEAN, HANDLE*);
        auto reg = [&](const char* name, cb_fn cb) {
            auto f = reinterpret_cast<reg_fn>(reinterpret_cast<void*>(GetProcAddress(h, name))); HANDLE nh = nullptr;
            if (f && f(AF_UNSPEC, cb, st_.get(), FALSE, &nh) == NO_ERROR && nh) st_->handles.push_back(nh);
        };
        st_ = std::make_unique<win_state>(); st_->ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!st_->ev) { st_.reset(); return; }
        // The row pointers start with (family, LUID, index) for interfaces and routes; only the kind is used here.
        reg("NotifyIpInterfaceChange", &win_state::on_link);
        reg("NotifyUnicastIpAddressChange", &win_state::on_address);
        reg("NotifyRouteChange2", &win_state::on_route);
        if (st_->handles.empty()) { CloseHandle(st_->ev); st_.reset(); return; }
        native_ = true; native_name_ = "iphlpapi";
#endif
    }
#if defined(__linux__)
    static void parse_(const char* buf, std::size_t n, std::vector<net_change>& out) {
        for (std::size_t off = 0; off + sizeof(nlmsghdr) <= n;) {
            nlmsghdr h; std::memcpy(&h, buf + off, sizeof h);
            if (h.nlmsg_len < sizeof(nlmsghdr) || off + h.nlmsg_len > n) break;
            const char* body = buf + off + NLMSG_HDRLEN; std::size_t blen = h.nlmsg_len - NLMSG_HDRLEN;
            net_change c; bool known = true;
            if ((h.nlmsg_type == RTM_NEWLINK || h.nlmsg_type == RTM_DELLINK) && blen >= sizeof(ifinfomsg)) { ifinfomsg m; std::memcpy(&m, body, sizeof m); c.kind = net_change_kind::link; c.index = unsigned(m.ifi_index); }
            else if ((h.nlmsg_type == RTM_NEWADDR || h.nlmsg_type == RTM_DELADDR) && blen >= sizeof(ifaddrmsg)) { ifaddrmsg m; std::memcpy(&m, body, sizeof m); c.kind = net_change_kind::address; c.index = m.ifa_index; }
            else if (h.nlmsg_type == RTM_NEWROUTE || h.nlmsg_type == RTM_DELROUTE) c.kind = net_change_kind::route;
            else known = false;
            if (known) { char nm[IFNAMSIZ] = {}; if (c.index && ::if_indextoname(c.index, nm)) c.interface_name = nm; out.push_back(std::move(c)); }
            off += NLMSG_ALIGN(h.nlmsg_len);
        }
    }
#elif defined(FORGE_NET_ROUTE_SOCKET)
    static void parse_(const char* buf, std::size_t n, std::vector<net_change>& out) {
        for (std::size_t off = 0; off + 4 <= n;) {
            unsigned short len; std::memcpy(&len, buf + off, 2); unsigned char type = static_cast<unsigned char>(buf[off + 3]);
            if (len < 4 || off + len > n) break;
            net_change c; bool known = true;
            if (type == RTM_NEWADDR || type == RTM_DELADDR) { c.kind = net_change_kind::address; if (len >= sizeof(ifa_msghdr)) { ifa_msghdr m; std::memcpy(&m, buf + off, sizeof m); c.index = m.ifam_index; } }
            else if (type == RTM_IFINFO) { c.kind = net_change_kind::link; if (len >= sizeof(if_msghdr)) { if_msghdr m; std::memcpy(&m, buf + off, sizeof m); c.index = m.ifm_index; } }
            else if (type == RTM_ADD || type == RTM_DELETE || type == RTM_CHANGE) { c.kind = net_change_kind::route; if (len >= sizeof(rt_msghdr)) { rt_msghdr m; std::memcpy(&m, buf + off, sizeof m); c.index = m.rtm_index; } }
            else known = false;
            if (known) { char nm[IFNAMSIZ] = {}; if (c.index && ::if_indextoname(c.index, nm)) c.interface_name = nm; out.push_back(std::move(c)); }
            off += len;
        }
    }
#elif defined(FORGE_OS_POSIX)
    static void parse_(const char*, std::size_t, std::vector<net_change>&) {}
#endif
    options opt_; bool native_ = false; const char* native_name_ = "polling";
    std::vector<link_info> snap_; std::uint64_t last_sample_ = 0;
#if defined(FORGE_OS_WINDOWS)
    struct win_state {
        std::mutex m; std::vector<net_change> q; HANDLE ev = nullptr; std::vector<HANDLE> handles;
        void push(net_change_kind k) { { std::lock_guard<std::mutex> g(m); if (q.size() < 4096) q.push_back({k, {}, 0}); } SetEvent(ev); }
        static void WINAPI on_link(void* c, void*, int) { static_cast<win_state*>(c)->push(net_change_kind::link); }
        static void WINAPI on_address(void* c, void*, int) { static_cast<win_state*>(c)->push(net_change_kind::address); }
        static void WINAPI on_route(void* c, void*, int) { static_cast<win_state*>(c)->push(net_change_kind::route); }
    };
    std::unique_ptr<win_state> st_;
#elif defined(FORGE_OS_POSIX)
    int fd_ = -1;
#endif
};
} // namespace forge::os::net

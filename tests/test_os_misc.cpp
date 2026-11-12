// forge::os wave 4, smaller additions: system appearance (and its watcher), power profile, clock synchronisation
// status, SNTP against an in-process server.
#include <forge/os/appearance.hpp>
#include <forge/os/power.hpp>
#include <forge/os/timesync.hpp>
#include <forge/os/fs.hpp>
#include <atomic>
#include <thread>
#include "check.hpp"
using namespace forge;

TEST(appearance_query) {
    auto a = os::system_appearance();
    if (!a) { std::printf("  skip: %s\n", a.err.what.c_str()); CHECK(os::is_unsupported(a.err)); }
    else {
        std::printf("  %s: scheme %s, high contrast %d, reduced motion %d, accent %s#%02x%02x%02x\n", a->source.c_str(),
                    a->scheme == os::color_scheme::dark ? "dark" : a->scheme == os::color_scheme::light ? "light" : "unknown", int(a->high_contrast), int(a->reduced_motion),
                    a->has_accent ? "" : "(none) ", a->accent_r, a->accent_g, a->accent_b);
        CHECK(!a->source.empty());
#if defined(_WIN32) || defined(FORGE_OS_MACOS)
        CHECK(a->scheme != os::color_scheme::unknown);
#endif
    }
#if defined(__ANDROID__)
    CHECK(!a); CHECK(os::is_unsupported(a.err));   // Java-only without the hook
#endif
    // the hook takes over everywhere (Android / tests), and the watcher reports each change once
    std::atomic<int> dark{0};
    os::set_appearance_hook([&] { os::appearance x; x.scheme = dark.load() ? os::color_scheme::dark : os::color_scheme::light; x.has_accent = true; x.accent_b = 255; return os::result<os::appearance>{x, {}}; });
    auto h = os::system_appearance(); CHECK(h); CHECK(h->scheme == os::color_scheme::light); CHECK(h->source == "hook"); CHECK(h->accent_b == 255);
    std::atomic<int> calls{0}, saw_dark{0};
    {
        os::appearance_watcher w; CHECK(!w.running());
        CHECK(w.start([&](const os::appearance& x) { ++calls; if (x.scheme == os::color_scheme::dark) ++saw_dark; }, 10)); CHECK(w.running());
        for (int i = 0; i < 300 && calls.load() < 1; ++i) os::sleep_ns(5'000'000);
        CHECK(calls.load() == 1); os::sleep_ns(60'000'000); CHECK(calls.load() == 1);   // nothing changed: no further callback
        dark = 1;
        for (int i = 0; i < 400 && saw_dark.load() < 1; ++i) os::sleep_ns(5'000'000);
        CHECK(saw_dark.load() == 1); CHECK(calls.load() == 2);
        auto t0 = os::monotonic_ns(); w.stop(); CHECK(os::monotonic_ns() - t0 < 2'000'000'000ull); CHECK(!w.running());
    }
    os::set_appearance_hook({});
}

TEST(appearance_settings_files) {
    auto k = os::detail_appearance::parse_kdeglobals("[General]\nColorScheme=BreezeDark\nAccentColor=61,174,233\n\n[KDE]\nAnimationDurationFactor=0\n[WM]\nColorScheme=Light\n");
    CHECK(k.scheme == os::color_scheme::dark); CHECK(k.reduced_motion); CHECK(k.has_accent && k.accent_r == 61 && k.accent_g == 174 && k.accent_b == 233); CHECK(k.source == "kdeglobals");
    auto l = os::detail_appearance::parse_kdeglobals("[General]\nColorScheme=BreezeLight\n[KDE]\nAnimationDurationFactor=1\n"); CHECK(l.scheme == os::color_scheme::light); CHECK(!l.reduced_motion);
    CHECK(os::detail_appearance::parse_kdeglobals("").source.empty());
    CHECK(os::detail_appearance::parse_dconf("").source.empty()); CHECK(os::detail_appearance::parse_dconf(std::string(100, 'x')).source.empty());
}

TEST(power_profile_query) {
    auto p = os::power::profile();
    if (!p) { std::printf("  skip: %s\n", p.err.what.c_str()); CHECK(os::is_unsupported(p.err)); return; }
    std::printf("  power profile: %s\n", p->c_str());
    CHECK(*p == "power-saver" || *p == "balanced" || *p == "performance" || *p == "custom");
}

TEST(time_sync_status_query) {
    auto t = os::time_sync_status();
    if (!t) { std::printf("  skip: %s\n", t.err.what.c_str()); CHECK(os::is_unsupported(t.err)); return; }
    std::printf("  %s: synchronized %d, estimated error %.6f s\n", t->source.c_str(), int(t->synchronized), t->estimated_error_s);
    CHECK(!t->source.empty()); CHECK(t->estimated_error_s >= 0);
#if defined(_WIN32) || defined(FORGE_OS_IOS) || defined(__ANDROID__)
    CHECK(false);   // these report unsupported
#endif
}

TEST(sntp_against_local_server) {
    using namespace os::detail_sntp;
    {   // timestamp conversion, both eras
        unsigned char b[8]; from_unix_ns(1700000000ll * 1000000000ll + 500000000ll, b);
        CHECK(be(b, 4) == 1700000000ull + 2208988800ull); CHECK(be(b + 4, 4) == 0x80000000ull);
        CHECK(to_unix_ns(b) == 1700000000ll * 1000000000ll + 500000000ll);
        const std::int64_t y2040 = 2208988800ll * 1000000000ll;   // 2040-01-01: past the 2036 wrap of the 32-bit seconds field
        from_unix_ns(y2040, b); CHECK(be(b, 4) < 0x80000000ull); CHECK(to_unix_ns(b) == y2040);
    }
    // A UDP server on loopback that answers like an NTP server whose clock is `skew` ahead; `mode` selects a fault.
    os::net_init();
    os::socket srv = os::socket::make(AF_INET, SOCK_DGRAM); CHECK(srv.valid());
    sockaddr_in sa{}; sa.sin_family = AF_INET; sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK); sa.sin_port = 0;
    CHECK(::bind(srv.native(), reinterpret_cast<sockaddr*>(&sa), sizeof sa) == 0);
    const int port = srv.local_port(); CHECK(port > 0);
    std::atomic<bool> stop{false}; std::atomic<int> fault{0}; const std::int64_t skew = 3'500'000'000ll;
    std::thread th([&] {
        while (!stop) {
            if (!srv.wait_readable(50)) continue;
            unsigned char rq[512]; sockaddr_storage from{}; socklen_t fl = sizeof from;
            auto n = ::recvfrom(srv.native(), reinterpret_cast<char*>(rq), sizeof rq, 0, reinterpret_cast<sockaddr*>(&from), &fl); if (n < 48) continue;
            unsigned char rp[48] = {}; rp[0] = 0x24 /*v4, server*/; rp[1] = 2 /*stratum*/;
            std::memcpy(rp + 24, rq + 40, 8);   // originate = the client's transmit
            const std::int64_t now = os::wall_ns() + skew; from_unix_ns(now, rp + 32); from_unix_ns(now, rp + 40);
            const int f = fault.load();
            if (f == 1) rp[24] ^= 0xff;                                    // answers some other request
            if (f == 2) { rp[1] = 0; std::memcpy(rp + 12, "RATE", 4); }    // kiss-of-death
            if (f == 3) rp[0] = 0x23;                                      // a client packet, not a server's
            if (f == 4) continue;                                          // silence
            ::sendto(srv.native(), reinterpret_cast<const char*>(rp), f == 5 ? 20 : 48, 0, reinterpret_cast<sockaddr*>(&from), fl);
        }
    });
    auto r = os::sntp_query("127.0.0.1", port, 3000);
    CHECK_MSG(bool(r), "%s", r.err.what.c_str());
    if (r) {
        std::printf("  offset %.6f s, delay %.6f s, stratum %d\n", r->offset_s, r->delay_s, r->stratum);
        CHECK(r->stratum == 2); CHECK(r->offset_s > 3.3 && r->offset_s < 3.7); CHECK(r->delay_s >= 0 && r->delay_s < 1.0);
        CHECK(r->server_unix_ns > os::wall_ns() + 3'000'000'000ll && r->server_unix_ns < os::wall_ns() + 4'000'000'000ll);
    }
    for (int f = 1; f <= 5; ++f) {   // every malformed or mismatched reply is refused; none hangs
        fault = f; auto t0 = os::monotonic_ns(); auto bad = os::sntp_query("127.0.0.1", port, 300);
        CHECK_MSG(!bad, "fault %d accepted", f); CHECK(os::monotonic_ns() - t0 < 5'000'000'000ull);
        if (f == 2) CHECK_MSG(bad.err.what.find("RATE") != std::string::npos, "%s", bad.err.what.c_str());
    }
    stop = true; th.join();
    // nobody listening: an error within the timeout, not a hang
    auto t0 = os::monotonic_ns(); auto none = os::sntp_query("127.0.0.1", port, 300); CHECK(!none); CHECK(os::monotonic_ns() - t0 < 5'000'000'000ull);
    CHECK(!os::sntp_query(std::string(70, 'x') + ".invalid", 123, 300));
}

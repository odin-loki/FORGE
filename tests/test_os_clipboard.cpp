// forge::os::clipboard on X11 without helper programs: runs against a private Xvfb when one is installed (no
// window ever appears), is skipped otherwise. Elsewhere: the backend hook contract.
#include <forge/os/clipboard.hpp>
#include "check.hpp"
using namespace forge;
namespace clip = forge::os::clipboard;

TEST(clipboard_backend_hook) {
    std::string held = "initial"; int sets = 0;
    clip::set_backend({[&] { return os::result<std::string>{held, {}}; }, [&](const std::string& t) { held = t; ++sets; return os::status{}; }});
    CHECK(clip::available()); CHECK(clip::get().value == "initial"); CHECK(clip::set("x\xC3\xA9y")); CHECK(clip::get().value == "x\xC3\xA9y"); CHECK(sets == 1);
    clip::set_backend({});
}

#if defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)
namespace {
// Starts Xvfb on a free display and points DISPLAY at it for the lifetime of the object.
struct private_x {
    os::child server; std::string display, old; bool had = false, ok = false;
    private_x() {
        if (os::find_executable("Xvfb").empty()) return;
        had = os::has_env("DISPLAY"); old = os::getenv("DISPLAY");
        for (int n = 150 + int(os::pid() % 400), tries = 0; tries < 5 && !ok; ++n, ++tries) {
            const std::string sock = "/tmp/.X11-unix/X" + std::to_string(n);
            if (os::exists(sock)) continue;
            auto c = os::child::start({"Xvfb", ":" + std::to_string(n), "-screen", "0", "64x64x8", "-nolisten", "tcp", "-noreset"}, true); if (!c) return;
            for (int i = 0; i < 300 && !os::exists(sock) && c->running() && c->try_wait() == -1; ++i) os::sleep_ns(10'000'000);
            if (os::exists(sock)) { server = std::move(c.value); display = ":" + std::to_string(n); os::setenv("DISPLAY", display); ok = true; }
            else { c->kill(); c->wait(2000); }
        }
    }
    ~private_x() {
        clip::x11::release();
        if (ok) { server.kill(); server.wait(5000); }
        if (had) os::setenv("DISPLAY", old); else os::unsetenv("DISPLAY");
    }
};
}

TEST(clipboard_x11_without_a_display) {
    bool had = os::has_env("DISPLAY"); std::string old = os::getenv("DISPLAY");
    os::unsetenv("DISPLAY");
    CHECK(!clip::x11::available());
    auto g = clip::x11::get(200); CHECK(!g); CHECK(os::is_unsupported(g.err));
    auto s = clip::x11::set("x"); CHECK(!s); CHECK(os::is_unsupported(s.err));
    os::setenv("DISPLAY", ":999");   // nobody listens there: unsupported, promptly
    auto t0 = os::monotonic_ns(); auto g2 = clip::x11::get(200); CHECK(!g2); CHECK(os::is_unsupported(g2.err)); CHECK(os::monotonic_ns() - t0 < 5'000'000'000ull);
    if (had) os::setenv("DISPLAY", old); else os::unsetenv("DISPLAY");
}

TEST(clipboard_x11_native) {
    private_x x;
    if (!x.ok) { std::printf("  skip: no Xvfb to run a private X server\n"); return; }
    if (!clip::x11::available()) { std::printf("  skip: libxcb is not installed\n"); return; }
    // nothing copied yet: an empty string, not an error
    { auto g = clip::x11::get(2000); CHECK_MSG(bool(g), "%s", g.err.what.c_str()); CHECK(g.value.empty()); CHECK(!clip::x11::owned()); }
    const std::string small_ = "forge clipboard \xE2\x9C\x93 caf\xC3\xA9\nline two";
    CHECK(clip::x11::set(small_)); CHECK(clip::x11::owned());
    { auto g = clip::x11::get(3000); CHECK_MSG(bool(g), "%s", g.err.what.c_str()); CHECK(g.value == small_); }
    CHECK(clip::x11::set("second")); CHECK(clip::x11::get().value == "second");   // replacing while owning
    CHECK(clip::x11::set("")); { auto g = clip::x11::get(); CHECK(g); CHECK(g.value.empty()); }
    // a text far over one request: sent and received in INCR chunks
    std::string big(3u << 20, '\0'); for (std::size_t i = 0; i < big.size(); ++i) big[i] = char('a' + (i * 7 + i / 1000) % 26);
    CHECK(clip::x11::set(big));
    { auto g = clip::x11::get(20000); CHECK_MSG(bool(g), "%s", g.err.what.c_str()); CHECK(g.value.size() == big.size()); CHECK(g.value == big); }
    // the public API takes the same path (no wl-copy session here)
    bool had_w = os::has_env("WAYLAND_DISPLAY"); std::string old_w = os::getenv("WAYLAND_DISPLAY"); os::unsetenv("WAYLAND_DISPLAY");
    CHECK(clip::available()); CHECK(clip::set("via the public call")); CHECK(clip::get().value == "via the public call");
    // an independent implementation agrees, in both directions
    if (!os::find_executable("xsel").empty()) {
        os::spawn_spec out; out.argv = {"xsel", "--clipboard", "--output"}; out.timeout_ms = 5000;
        auto r = os::spawn_wait(out); CHECK(r); if (r) CHECK_MSG(r->out == "via the public call", "xsel read [%s]", r->out.c_str());
        os::spawn_spec in; in.argv = {"xsel", "--clipboard", "--input"}; in.stdin_data = "from xsel \xC3\xA9"; in.timeout_ms = 5000; in.capture_stdout = false; in.capture_stderr = false;
        auto w = os::spawn_wait(in); CHECK(w);
        if (w && w->exit_code == 0) {
            for (int i = 0; i < 100 && clip::x11::owned(); ++i) os::sleep_ns(10'000'000);   // SelectionClear reaches our owner thread
            CHECK(!clip::x11::owned());
            auto g = clip::x11::get(3000); CHECK_MSG(bool(g), "%s", g.err.what.c_str()); CHECK_MSG(g.value == "from xsel \xC3\xA9", "[%s]", g.value.c_str());
            os::spawn_spec kill; kill.argv = {"xsel", "--clipboard", "--clear"}; kill.timeout_ms = 5000; (void)os::spawn_wait(kill);
        }
    } else std::printf("  (no xsel: cross-check skipped)\n");
    if (had_w) os::setenv("WAYLAND_DISPLAY", old_w);
    clip::x11::release(); CHECK(!clip::x11::owned());
}
#endif

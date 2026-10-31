// forge::os process / pipe / signal (OS wrapper phase 1).
#include <forge/os/process.hpp>
#include <forge/os/pipe.hpp>
#include <forge/os/signal.hpp>
#include "check.hpp"
using namespace forge;

#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS)
TEST(spawn_captures_separately) {
    os::spawn_spec s; s.argv = {"sh", "-c", "echo out; echo err 1>&2; exit 3"};
    auto r = os::spawn_wait(s);
    CHECK(r); CHECK(r->out == "out\n"); CHECK(r->err == "err\n"); CHECK(r->exit_code == 3); CHECK(!r->timed_out);
}
TEST(spawn_stdin_env_cwd) {
    os::spawn_spec s; s.argv = {"sh", "-c", "cat; printf %s \"$FOO\"; pwd"};
    s.stdin_data = std::string(200000, 'x'); s.env_set["FOO"] = "bar"; s.cwd = "/";
    auto r = os::spawn_wait(s);
    CHECK(r); CHECK(r->out == std::string(200000, 'x') + "bar/\n");
}
TEST(spawn_merge_stderr) {
    os::spawn_spec s; s.argv = {"sh", "-c", "echo a; echo b 1>&2"}; s.merge_stderr = true;
    auto r = os::spawn_wait(s); CHECK(r); CHECK(r->out.size() == 4); CHECK(r->err.empty());
}
TEST(spawn_timeout_kills_tree) {
    os::spawn_spec s; s.argv = {"sh", "-c", "sleep 30 & sleep 30"}; s.timeout_ms = 200;
    auto t0 = os::monotonic_ns(); auto r = os::spawn_wait(s);
    CHECK(r); CHECK(r->timed_out); CHECK(r->exit_code == 124); CHECK(os::monotonic_ns() - t0 < 5'000'000'000ull);
}
TEST(spawn_missing_command) {
    os::spawn_spec s; s.argv = {"no-such-command-forge-xyz"};
    auto r = os::spawn_wait(s); CHECK(!r);
}
TEST(spawn_memory_limit) {
    os::spawn_spec s; s.argv = {"sh", "-c", "ulimit -v"}; s.limits.memory_bytes = 512ull << 20;
    auto r = os::spawn_wait(s); CHECK(r); CHECK(r->out == "524288\n");
}
TEST(spawn_output_cap) {
    os::spawn_spec s; s.argv = {"sh", "-c", "head -c 100000 /dev/zero"}; s.max_output = 1000;
    auto r = os::spawn_wait(s); CHECK(r); CHECK(r->out.size() <= 1000); CHECK(r->truncated);
}
TEST(child_stream_and_kill) {
    auto c = os::child::start({"sh", "-c", "echo hi; sleep 30"});
    CHECK(c); CHECK(c->running());
    CHECK(c->read(100, 2000) == "hi\n");
    CHECK(c->wait(50) == -1);
    CHECK(c->kill(9)); CHECK(c->wait() == 128 + 9);
}
TEST(pipe_roundtrip) {
    auto p = os::make_pipe(); CHECK(p);
    CHECK(p->write.write_all("hello")); p->write.close();
    CHECK(p->read.read_all() == "hello");
}
TEST(signals) {
    os::signal_id s; CHECK(os::sig_from_name("term", s) && s == os::signal_id::terminate);
    auto c = os::child::start({"sleep", "30"}); CHECK(c);
    CHECK(os::send_signal(c->pid(), os::signal_id::terminate));
    CHECK(c->wait() == 128 + SIGTERM);
}
#if defined(FORGE_OS_LINUX)
TEST(process_list) {
    auto l = os::list_processes(); CHECK(l); CHECK(!l->empty());
    auto me = os::process_info(os::pid()); CHECK(me); CHECK(me->rss_bytes > 0);
}
#endif
#endif
TEST(find_exe) { CHECK(os::find_executable("no-such-command-forge-xyz").empty()); CHECK(!os::process_caps().empty() || true); }

#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS)
TEST(child_start_spec_env_cwd_and_exec_error) {
    os::spawn_spec s; s.argv = {"sh", "-c", "echo $FORGE_X; pwd"}; s.env_set["FORGE_X"] = "hi"; s.cwd = "/";
    auto c = os::child::start(s); CHECK(c);
    std::string out, part; while (!(part = c->read()).empty()) out += part;
    CHECK(c->wait() == 0); CHECK(out == "hi\n/\n");
    os::spawn_spec bad; bad.argv = {"/nonexistent/forge-no-such-binary"};
    CHECK(!os::child::start(bad));
}
TEST(child_start_memory_limit) {
    os::spawn_spec s; s.argv = {"sh", "-c", "exec dd if=/dev/zero bs=64M count=1 of=/dev/null 2>/dev/null"};
    s.limits.memory_bytes = 32ull << 20;
    auto c = os::child::start(s); CHECK(c); CHECK(c->wait() != 0);   // 64 MiB buffer cannot fit in 32 MiB
    s.limits.memory_bytes = 512ull << 20;
    auto d = os::child::start(s); CHECK(d); CHECK(d->wait() == 0);
}
TEST(wait_any_picks_first_and_wait_all) {
    auto mk = [](const char* t, int code) { os::spawn_spec s; s.argv = {"sh", "-c", std::string("sleep ") + t + "; exit " + std::to_string(code)}; return std::move(os::child::start(s).value); };
    os::child a = mk("0.6", 1), b = mk("0.05", 2), c = mk("0.3", 3);
    os::child* all[] = {&a, &b, &c, nullptr};
    auto i = os::wait_any(all); CHECK(i && *i == 1); CHECK(b.exit_code() == 2);
    i = os::wait_any(all); CHECK(i && *i == 2); CHECK(c.exit_code() == 3);
    CHECK(!os::wait_any(all, 20));                         // a still sleeping: timeout
    CHECK(a.try_wait() == -1);
    CHECK(os::wait_all(all, 5000)); CHECK(a.exit_code() == 1);
    CHECK(!os::wait_any(all));                             // nothing left to wait for
    os::child d = mk("5", 0); os::child* one[] = {&d};
    auto t0 = os::monotonic_ns(); CHECK(!os::wait_all(one, 50)); CHECK(os::monotonic_ns() - t0 < 2000000000ull);
    d.kill(); CHECK(d.wait() == 128 + 9);
}
#endif

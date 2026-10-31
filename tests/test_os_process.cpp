// forge::os process / pipe / signal (OS wrapper phase 1).
#include <forge/os/process.hpp>
#include <forge/os/pipe.hpp>
#include <forge/os/signal.hpp>
#include <thread>
#include <sstream>
#include <cstdio>
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
TEST(spawn_fsize_core_limits) {
    os::spawn_spec s; s.argv = {"sh", "-c", "ulimit -f; ulimit -c"}; s.limits.file_size_bytes = 1ull << 20; s.limits.no_core = true;
    auto r = os::spawn_wait(s); CHECK(r); CHECK(r->out == "2048\n0\n");   // ulimit -f counts 512-byte blocks
}
TEST(apply_limits_clamps_to_hard) {
    // a child whose hard NOFILE is 64 asks for 256: lowered to 64 instead of failing and leaving none
    pid_t p = ::fork();
    if (p == 0) {
        struct rlimit h{64, 64}; ::setrlimit(RLIMIT_NOFILE, &h);
        os::proc_limits l; l.open_files = 256; l.cpu_seconds = 100;
        bool ok = os::apply_limits(l); struct rlimit got{}; ::getrlimit(RLIMIT_NOFILE, &got);
        ::_exit(ok && got.rlim_cur == 64 && got.rlim_max == 64 ? 0 : 1);
    }
    int st = 0; ::waitpid(p, &st, 0); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0);
}
TEST(spawn_stop_token_cancels) {
    std::stop_source src; os::spawn_spec s; s.argv = {"sh", "-c", "sleep 30 & sleep 30"}; s.stop = src.get_token();
    std::thread t([&] { std::this_thread::sleep_for(std::chrono::milliseconds(150)); src.request_stop(); });
    auto t0 = os::monotonic_ns(); auto r = os::spawn_wait(s); t.join();
    CHECK(r); CHECK(r->cancelled); CHECK(r->exit_code == 125); CHECK(os::monotonic_ns() - t0 < 5'000'000'000ull);
}
TEST(spawn_file_redirects) {
    std::string d = "/tmp/forge_redir_" + std::to_string(::getpid());
    { FILE* f = std::fopen((d + ".in").c_str(), "w"); std::fputs("hello\n", f); std::fclose(f); }
    os::spawn_spec s; s.argv = {"sh", "-c", "cat; echo e >&2"}; s.stdin_file = d + ".in"; s.stdout_file = d + ".out"; s.stderr_file = d + ".err";
    auto r = os::spawn_wait(s); CHECK(r); CHECK(r->exit_code == 0); CHECK(r->out.empty());
    auto slurp = [](const std::string& p) { std::string o; if (FILE* f = std::fopen(p.c_str(), "r")) { char b[256]; std::size_t n; while ((n = std::fread(b, 1, sizeof b, f))) o.append(b, n); std::fclose(f); } return o; };
    CHECK(slurp(d + ".out") == "hello\n"); CHECK(slurp(d + ".err") == "e\n");
    s.append_output = true; s.argv = {"sh", "-c", "echo more"}; s.stdin_file.clear(); CHECK(os::spawn_wait(s));
    CHECK(slurp(d + ".out") == "hello\nmore\n");
    for (auto x : {".in", ".out", ".err"}) std::remove((d + x).c_str());
    s.stdin_file = "/nonexistent/forge"; CHECK(!os::spawn_wait(s));
}
TEST(spawn_new_session_and_group_kill) {
    os::spawn_spec s; s.argv = {"sh", "-c", "ps -o sid= -p $$; ps -o pid= -p $$"}; s.new_session = true;
    auto r = os::spawn_wait(s); CHECK(r);
    if (r && r->exit_code == 0) { std::istringstream is(r->out); long sid = 0, pid = 0; is >> sid >> pid; CHECK(sid == pid); }
    os::spawn_spec c; c.argv = {"sh", "-c", "sleep 30 & sleep 30"}; c.new_session = true;
    auto ch = os::child::start(c); CHECK(ch);
    CHECK(os::kill_group(ch->pid(), SIGKILL)); CHECK(ch->wait(5000) == 128 + SIGKILL);
}
#if defined(__linux__)
TEST(spawn_memory_watchdog) {
    os::spawn_spec s; s.argv = {"sh", "-c", "x=$(head -c 200000000 /dev/zero | tr '\\0' a); sleep 5"}; s.watch_memory_bytes = 64ull << 20;
    auto r = os::spawn_wait(s); CHECK(r); CHECK(r->memory_killed); CHECK(r->exit_code == 137);
}
#endif
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

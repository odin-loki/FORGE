#include "check.hpp"
#include <forge/core/log.hpp>
#include <forge/os/os.hpp>
#include <filesystem>
#include <fstream>
#include <set>
#include <thread>
namespace L = forge::log;

struct capture {
    std::vector<std::string> lines; std::vector<std::string> mods; int id;
    capture() { id = L::add_sink([this](const L::record& r, std::string_view s) { lines.emplace_back(s); mods.emplace_back(r.module); }); }
    ~capture() { L::remove_sink(id); }
};

TEST(levels_and_threshold) {
    L::set_stderr(false); L::configure("warn");
    capture c;
    L::info("hidden {}", 1); L::warn("shown {}", 2); L::error("e");
    CHECK(c.lines.size() == 2);
    CHECK(c.lines[0] == "[forge W] shown 2");
    L::set_level(L::level::trace); L::trace("t"); L::debug("d");
    CHECK(c.lines.size() == 4);
    L::set_level(L::level::off); L::error("x"); CHECK(c.lines.size() == 4);
    CHECK(L::parse_level("DEBUG") == L::level::debug);
}

TEST(module_tags_and_overrides) {
    L::configure("error,mind=debug");
    capture c; L::logger m{"mind"}, f{"flow"};
    m.debug("kv {}", 3); f.warn("no"); L::warn("no");
    CHECK(c.lines.size() == 1);
    CHECK(c.lines[0] == "[forge:mind D] kv 3");
    CHECK(c.mods[0] == "mind");
    CHECK(m.enabled(L::level::debug) && !f.enabled(L::level::warn));
    L::set_module_level("flow", L::level::info); f.info("yes"); CHECK(c.lines.size() == 2);
}

TEST(json_format_escapes) {
    L::configure("info"); L::set_format(L::format::json);
    capture c; L::logger{"a\"b"}.info("line\n\"q\"");
    L::set_format(L::format::text);
    CHECK(c.lines.size() == 1);
    CHECK(c.lines[0].find("\"level\":\"info\"") != std::string::npos);
    CHECK(c.lines[0].find("\"module\":\"a\\\"b\"") != std::string::npos);
    CHECK(c.lines[0].find("\"msg\":\"line\\n\\\"q\\\"\"") != std::string::npos);
    CHECK(c.lines[0].find('\n') == std::string::npos);
}

TEST(file_sink_and_threads_do_not_interleave) {
    L::configure("info");
    auto p = (std::filesystem::temp_directory_path() / ("forge_log_" + std::to_string(forge::os::pid()) + ".txt")).string();
    std::filesystem::remove(p);
    int id = L::add_file_sink(p); CHECK(id > 0);
    std::vector<std::thread> ts;
    for (int t = 0; t < 4; ++t) ts.emplace_back([t] { L::logger g{"th"}; for (int i = 0; i < 200; ++i) g.info("t{}-i{}-{}", t, i, std::string(40, char('a' + t))); });
    for (auto& t : ts) t.join();
    L::remove_sink(id);
    std::ifstream in(p); std::string s; int n = 0; std::set<std::string> seen;
    while (std::getline(in, s)) {
        ++n; CHECK_MSG(s.rfind("[forge:th I] t", 0) == 0, "bad line %s", s.c_str());
        auto tail = s.substr(s.size() - 40); CHECK(tail == std::string(40, tail[0]));
        seen.insert(s);
    }
    CHECK(n == 800); CHECK(seen.size() == 800);
    in.close(); std::filesystem::remove(p);
    CHECK(L::add_file_sink("/nonexistent-dir/x/y.log") == 0);
}

TEST(env_spec_parsing) {
    L::configure("i,mind=t,flow=o");
    CHECK(L::threshold() == L::level::info);
    CHECK(L::threshold("mind") == L::level::trace);
    CHECK(!L::enabled(L::level::error, "flow"));
    CHECK(L::enabled(L::level::trace, "mind"));
    L::configure("warn");
}

TEST(fatal_level_and_macros_skip_arguments) {
    L::configure("warn"); capture c; int evals = 0;
    auto bump = [&] { return ++evals; };
    FORGE_LOG_DEBUG("m", "x {}", bump());   // disabled at run time: argument not evaluated
    CHECK(evals == 0);
    FORGE_LOG_WARN("m", "y {}", bump()); CHECK(evals == 1);
    L::fatal("down"); CHECK(c.lines.size() == 2); CHECK(c.lines[1] == "[forge F] down");
    CHECK(L::parse_level("fatal") == L::level::fatal);
}

TEST(file_sink_rotation) {
    L::configure("info");
    auto p = (std::filesystem::temp_directory_path() / ("forge_rot_" + std::to_string(forge::os::pid()) + ".log")).string();
    for (auto s : {"", ".1", ".2", ".3"}) std::filesystem::remove(p + s);
    int id = L::add_file_sink(p, 200, 2);
    for (int i = 0; i < 30; ++i) L::info("line {:02} {}", i, std::string(20, 'x'));   // 36 bytes each
    L::remove_sink(id);
    CHECK(std::filesystem::exists(p)); CHECK(std::filesystem::exists(p + ".1")); CHECK(std::filesystem::exists(p + ".2"));
    CHECK(!std::filesystem::exists(p + ".3"));
    for (auto s : {"", ".1", ".2"}) CHECK(std::filesystem::file_size(p + s) <= 200);
    std::ifstream in(p); std::string last, s; while (std::getline(in, s)) last = s;
    CHECK(last.find("line 29") != std::string::npos);
    for (auto s : {"", ".1", ".2"}) std::filesystem::remove(p + s);
}

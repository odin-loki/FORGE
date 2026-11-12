// Native ahead-of-time compilation (docs/aot-models.md): generate the plans' straight-line code, compile it with the
// host C++ compiler, load it through the normal session API and compare with plans compiled at run time, bit for bit.
// Exits 77 (ctest: skipped) when no compiler can be run or the build description is missing. Slow: every case
// compiles a translation unit that includes the FORGE headers.
#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/inference/inference.hpp>
#include <forge/inference/aot_build.hpp>
#include <chrono>
#include <filesystem>
#if !defined(_WIN32)
#include <unistd.h>
#endif

using namespace forge;
using namespace forge::inference;
namespace fs = std::filesystem;

static std::string tmp(const std::string& n) { return (fs::temp_directory_path() / ("forge_aotn_" + std::to_string(::getpid()) + "_" + n)).string(); }
static bool same_bits(std::span<const float> a, std::span<const float> b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0; }
#if defined(_WIN32)
static const char* so_ext = ".dll"; static const char* exe_ext = ".exe";
#elif defined(__APPLE__)
static const char* so_ext = ".dylib"; static const char* exe_ext = "";
#else
static const char* so_ext = ".so"; static const char* exe_ext = "";
#endif

static void skip(const std::string& why) { std::printf("SKIP: %s\n", why.c_str()); std::exit(77); }

// Build a native artefact, or skip the whole binary when this machine cannot.
static aot_native_result build(const std::string& fpk, aot_build_options bo, aot_native_options no) {
    try {
        auto r = aot_build_native(fpk, bo, no);
        std::printf("  %s: %zu plans, %zu direct + %zu kernel-call steps, generated in %.2fs, compiled in %.1fs by %s (simd %s)\n", no.emit.c_str(),
                    r.plans.targets.size(), r.direct_steps, r.generic_steps, r.generate_s, r.compile_s, r.compiler.c_str(), r.simd.c_str());
        return r;
    } catch (const forge_exception& e) {
        if (e.code == errc::unsupported || e.code == errc::not_found) skip(e.what());
        throw;
    }
}

static std::shared_ptr<model> jit_model(const std::string& path) { auto m = model::open(path); m->set_aot(false); return m; }

// The same token stream through the compiled library and through plans compiled at run time.
static void expect_identical(const std::string& lib, const std::string& fpk, const std::vector<int>& chunks, session_options so = {}, std::size_t min_native = 1) {
    auto ma = model::open(lib); auto mr = jit_model(fpk);
    session a(ma, so), r(mr, so);
    int tok = 3;
    for (int n : chunks) {
        std::vector<int> ids; for (int i = 0; i < n; ++i) ids.push_back((tok = (tok * 37 + 11) % 250));
        CHECK_MSG(same_bits(a.forward(ids), r.forward(ids)), "logits differ for a chunk of %d at position %lld", n, (long long)r.position());
    }
    CHECK_MSG(ma->aot().stats().native >= min_native, "native plans: %zu, precompiled %zu, rejected %zu (%s)", ma->aot().stats().native, ma->aot().stats().used,
              ma->aot().stats().rejected, ma->aot().stats().last_reject.c_str());
    sampler g1, g2; a.reset(); r.reset();
    std::vector<int> p = {1, 5, 9, 33};
    CHECK(a.generate(p, 16, g1) == r.generate(p, 16, g2));
    // sampled generation with a seed, and a rewind in the middle
    sampler_config sc; sc.temperature = 0.8f; sc.top_k = 20; sc.seed = 42;
    sampler s1(sc), s2(sc); a.reset(); r.reset();
    CHECK(a.generate(p, 16, s1, {}, nullptr, false) == r.generate(p, 16, s2, {}, nullptr, false));
    a.rewind(6); r.rewind(6);
    std::vector<int> more = {7, 8};
    CHECK(same_bits(a.forward(more), r.forward(more)));
}

static const std::string& deep_f32() { static std::string p = forge_test::make_tiny_model(tmp("deep.fpk"), {.d = 64, .layers = 12, .heads = 4, .kv = 2, .ffn = 256}); return p; }

TEST(native_shared_library_is_bit_identical) {
    auto so = tmp(std::string("deep") + so_ext);
    auto r = build(deep_f32(), {.shapes = {1, 4, 16}, .kv_codecs = {"f32", "int8"}}, {.out = so, .work_dir = tmp("deep_src")});
    CHECK(fs::exists(so) && fs::exists(r.source));
    CHECK(r.direct_steps > 0);
    CHECK(aot_native::library::is_library(so) && !aot_native::library::is_library(deep_f32()));
    expect_identical(so, deep_f32(), {4, 1, 1, 16, 1, 4}, {}, 3);
    expect_identical(so, deep_f32(), {4, 1, 1}, {.kv_codec = "int8"}, 2);
    expect_identical(so, deep_f32(), {16, 1}, {.page_rows = 2}, 2);
    // shapes and codecs the library has no code for run on plans compiled here, in the same session
    expect_identical(so, deep_f32(), {5, 1, 3}, {}, 1);
    expect_identical(so, deep_f32(), {4, 1}, {.kv_codec = "tbq4"}, 0);
    {   // the weights are found next to the library, or where FORGE_AOT_FPK says; another model is refused
        auto m = model::open(so);
        CHECK(m->aot().native() && m->fpk_path() == forge::os::absolute(deep_f32()));
        auto other = forge_test::make_tiny_model(tmp("other.fpk"), {.d = 64, .layers = 12, .heads = 4, .kv = 2, .ffn = 256, .seed = 3});
        forge_test::set_env("FORGE_AOT_FPK", other.c_str());
        auto m2 = model::open(so);   // warns and runs `other` without the compiled code
        CHECK(!m2->aot().native() && m2->aot().stats().rejected == 1);
        session s2(m2); auto mr = jit_model(other); session sr(mr);
        std::vector<int> ids = {1, 2, 3, 4};
        CHECK(same_bits(s2.forward(ids), sr.forward(ids)));
        forge_test::set_env("FORGE_AOT_FPK", nullptr);
        // strict loading: compiled code is not covered by any model signature
        forge_test::set_env("FORGE_REQUIRE_SIGNED", "1");
        CHECK_THROWS(model(so));
        forge_test::set_env("FORGE_REQUIRE_SIGNED", nullptr);
    }
    // the C++ session API on top: grammar-constrained generation gives the same tokens
    {
        auto ma = model::open(so); auto mr = jit_model(deep_f32());
        session a(ma), b(mr);
        auto gen = [](session& s) {
            auto g = std::make_shared<grammar>("root ::= [a-z]+ \" \" [0-9]+");
            grammar_constraint gc(g, *s.tokenizer(), int(s.vocab()), s.tokenizer()->eos());
            sampler smp; smp.filter = [&](std::span<float> l) { gc.apply(l); }; smp.allow = [&](int t) { return gc.allows(t); };
            auto p = s.tokenizer()->encode("Hi");
            return s.generate(p, 12, smp, [&](int t) { return gc.accept(t) && !gc.finished(); });
        };
        CHECK(gen(a) == gen(b));
    }
    // start-up and decode throughput: run-time compiled plans, precompiled plans, native code
    {
        graph::aot::image img({r.plans.blob.data(), r.plans.blob.size()});
        auto planned = tmp("deep_plans.fpk");
        modelfile::aot::write_section(deep_f32(), planned, r.plans.blob);
        std::vector<int> prompt; for (int i = 0; i < 16; ++i) prompt.push_back(1 + i * 7);
        using clk = std::chrono::steady_clock;
        auto ms = [](clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
        { session warm(deep_f32()); warm.forward(prompt); }
        const char* name[3] = {"compiled at run time", "precompiled plans", "native code"};
        double start[3] = {1e9, 1e9, 1e9}, tps[3] = {0, 0, 0};
        std::vector<float> ref;
        for (int rep = 0; rep < 5; ++rep) for (int k = 0; k < 3; ++k) {
            auto t0 = clk::now();
            auto m = model::open(k == 2 ? so : k == 1 ? planned : deep_f32()); if (k == 0) m->set_aot(false);
            session s(m);
            s.forward(prompt);
            int one = 9; auto l = s.forward(std::span<const int>(&one, 1));
            start[k] = std::min(start[k], ms(clk::now() - t0));
            if (ref.empty()) ref = l; else CHECK(same_bits(ref, l));
            auto t1 = clk::now();
            for (int i = 0; i < 128; ++i) { int tk = 3 + i; s.forward(std::span<const int>(&tk, 1)); }
            tps[k] = std::max(tps[k], 128.0 / std::chrono::duration<double>(clk::now() - t1).count());
        }
        std::printf("  12-layer d=64 tiny model (best of 5):\n");
        for (int k = 0; k < 3; ++k) std::printf("    %-22s start-up to the first decoded token %7.2f ms, decode %6.0f tok/s\n", name[k], start[k], tps[k]);
    }
}

TEST(native_quantised_embedded_weights) {
    auto src = forge_test::make_tiny_model(tmp("q.fpk"), {.codec = "q4_k"});
    auto so = tmp(std::string("q") + so_ext);
    build(src, {.shapes = {1, 6}}, {.out = so, .embed_weights = true});
    {
        auto m = model::open(so);
        CHECK(m->aot().native() && m->aot().native()->embedded());
    }
    fs::path moved = fs::path(tmp("moved")) / (std::string("q") + so_ext);
    fs::create_directories(moved.parent_path()); fs::rename(so, moved);   // the library alone is the model
    auto ref = tmp("q_ref.fpk"); fs::rename(src, ref);
    expect_identical(moved.string(), ref, {6, 1, 1, 6}, {}, 2);
    // above the embed limit: refused, with the reason
    try { aot_build_native(ref, {.shapes = {1}}, {.out = tmp("big.so"), .embed_weights = true, .embed_limit = 1024}); CHECK(false); }
    catch (const forge_exception& e) { CHECK(e.code == errc::unsupported && std::string(e.what()).find("embed limit") != std::string::npos); }
}

TEST(native_recurrent_state_model) {   // state_op steps between the compiled segments
    auto src = forge_test::make_tiny_qwen35_model(tmp("q35.fpk"));
    auto so = tmp(std::string("q35") + so_ext);
    build(src, {.shapes = {1, 5}}, {.out = so});
    auto ma = model::open(so); auto mr = jit_model(src);
    session a(ma), r(mr);
    for (auto ids : {std::vector<int>{1, 2, 3, 4, 5}, std::vector<int>{6}, std::vector<int>{7}, std::vector<int>{8, 9, 10, 11, 12}})
        CHECK(same_bits(a.forward(ids), r.forward(ids)));
    CHECK(ma->aot().stats().native == 2);
    sampler g1, g2; a.reset(); r.reset();
    std::vector<int> p = {1, 5, 9, 33, 2};
    CHECK(a.generate(p, 12, g1) == r.generate(p, 12, g2));
}

#if defined(FORGE_TEST_BINARY_DIR)
TEST(native_moe_golden_model) {   // the golden Mixtral / DeepSeek-V2 models, when the golden tests have written them
    int n = 0;
    for (const char* name : {"golden_moe/deepseek_v2.fpk"}) {   // MLA attention + sparse MoE
        auto src = std::string(FORGE_TEST_BINARY_DIR) + "/" + name;
        if (!fs::exists(src)) { std::printf("  (no %s: run the golden_moe_mla test first)\n", name); continue; }
        auto so = tmp(std::string("moe") + std::to_string(n++) + so_ext);
        build(src, {.shapes = {1, 6}}, {.out = so});
        expect_identical(so, src, {6, 1, 1, 6}, {}, 2);
    }
}
#endif

#if defined(FORGE_TEST_FORGE_BIN)
static std::string run(std::vector<std::string> argv, int* code = nullptr, std::string* err = nullptr) {
    forge::os::spawn_spec sp; sp.argv = std::move(argv);
    auto r = forge::os::spawn_wait(sp);
    if (!r) skip("cannot start a process: " + r.err.what);
    if (code) *code = r.value.exit_code;
    if (err) *err = r.value.err;
    return r.value.out;
}
// `forge aot --emit exe`: the program prints what `forge run` prints for the same arguments.
TEST(native_executable_matches_forge_run) {
    const std::string forge_bin = FORGE_TEST_FORGE_BIN;
    auto src = forge_test::make_tiny_model(tmp("exe.fpk"), {.seed = 11});
    auto exe = tmp(std::string("model") + exe_ext);
    int code = 0; std::string err;
    auto out = run({forge_bin, "aot", src, "-o", exe, "--aot-mode", "native", "--emit", "exe", "--shapes", "1,4,6"}, &code, &err);
    if (code != 0 && (err.find("no C++ compiler") != std::string::npos || err.find("cannot run") != std::string::npos)) skip(err);
    CHECK_MSG(code == 0, "forge aot failed: %s %s", out.c_str(), err.c_str());
    if (code != 0) return;
    for (auto args : {std::vector<std::string>{"--prompt", "Hello", "-n", "24"},
                      std::vector<std::string>{"--tokens", "1,5,9,33", "-n", "16", "--ids"},
                      std::vector<std::string>{"--prompt", "abc", "-n", "20", "--temp", "0.9", "--top-k", "30", "--seed", "7"},
                      std::vector<std::string>{"--prompt", "a longer prompt than any precompiled shape", "-n", "8", "--no-bos"}}) {
        std::vector<std::string> a = {exe}, b = {forge_bin, "run", src};
        a.insert(a.end(), args.begin(), args.end()); b.insert(b.end(), args.begin(), args.end());
        int ca = 0, cb = 0; std::string ea, eb;
        auto oa = run(a, &ca, &ea), ob = run(b, &cb, &eb);
        CHECK_MSG(ca == 0 && cb == 0, "exit codes %d %d: %s | %s", ca, cb, ea.c_str(), eb.c_str());
        CHECK_MSG(oa == ob && !oa.empty(), "the executable printed '%s', forge run printed '%s'", oa.c_str(), ob.c_str());
        if (args[1] == "Hello") { CHECK_MSG(ea.find("(2 native)") != std::string::npos || ea.find("native)") != std::string::npos, "stats: %s", ea.c_str()); std::printf("  %s", ea.c_str()); }
    }
    // forge run on a compiled library, through the CLI
    auto so = tmp(std::string("cli") + so_ext);
    out = run({forge_bin, "aot", src, "-o", so, "--aot-mode", "native", "--shapes", "1,6"}, &code, &err);
    CHECK_MSG(code == 0, "forge aot failed: %s %s", out.c_str(), err.c_str());
    if (code == 0) {
        std::string e1, e2;
        auto o1 = run({forge_bin, "run", so, "--prompt", "Hello", "-n", "24"}, &code, &e1), o2 = run({forge_bin, "run", src, "--prompt", "Hello", "-n", "24"}, nullptr, &e2);
        CHECK_MSG(code == 0 && o1 == o2, "'%s' vs '%s' (%s)", o1.c_str(), o2.c_str(), e1.c_str());
        CHECK_MSG(e1.find("native") != std::string::npos, "%s", e1.c_str());
    }
    // a static library: the object and the exported entry point are there
    auto lib = tmp("model.a");
    out = run({forge_bin, "aot", src, "-o", lib, "--aot-mode", "native", "--emit", "static", "--shapes", "1"}, &code, &err);
    CHECK_MSG(code == 0 && fs::exists(lib) && fs::file_size(lib) > 1000, "forge aot --emit static failed: %s %s", out.c_str(), err.c_str());
    // a device target is refused with a reason
    out = run({forge_bin, "aot", src, "-o", tmp("dev.fpk"), "--device", "vulkan"}, &code, &err);
    CHECK_MSG(code != 0 && err.find("CPU") != std::string::npos, "%d %s", code, err.c_str());
}
#endif

// forge::compress persistence: SSD-backed buffers (mapped.hpp), lazy tensors, and crash / OOM checkpoints of
// compress jobs (checkpoint.hpp). Tiny random models (tiny_model.hpp).
//
// FORGE_COMPRESS_RSS_MODEL=<model.fpk> also measures the memory of loading a real model (printed, not checked):
// run once as is and once with FORGE_COMPRESS_LAZY=0 FORGE_COMPRESS_MMAP_MIN=off for the in-RAM baseline.
#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/compress/cli.hpp>

#include <filesystem>
#include <fstream>
#include <random>

using namespace forge;
using namespace forge::compress;
using forge_test::tiny_cfg;
using forge_test::make_tiny_model;

namespace {

const int exact_int8 = (forge_test::set_env("FORGE_CPU_INT8", "0"), 0);

std::string tmpdir() {
    static const std::string d = [] {
        auto p = std::filesystem::temp_directory_path() / std::format("forge_compress_persist_{}", ::getpid());
        std::filesystem::create_directories(p);
        return p.string();
    }();
    return d;
}
std::string tmp(const std::string& n) { return (std::filesystem::path(tmpdir()) / n).string(); }

std::string read_bytes(const std::string& p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }

// Every tensor, the blueprint and the manifest of two .fpk files are identical.
bool same_model_file(const std::string& a, const std::string& b, std::string* why = nullptr) {
    forge::modelfile::reader ra(a), rb(b);
    auto fail = [&](const std::string& w) { if (why) *why = w; return false; };
    if (ra.tensors().size() != rb.tensors().size()) return fail("tensor count");
    for (auto& [n, e] : ra.tensors()) {
        if (!rb.tensors().count(n)) return fail("missing " + n);
        const auto& f = rb.entry(n);
        if (e.codec != f.codec || e.shape != f.shape) return fail("codec/shape of " + n);
        auto x = ra.tensor_bytes(n), y = rb.tensor_bytes(n);
        if (x.size() != y.size() || std::memcmp(x.data(), y.data(), x.size()) != 0) return fail("bytes of " + n);
    }
    if (ra.section_text("blueprint") != rb.section_text("blueprint")) return fail("blueprint");
    if (forge::json::dump(ra.manifest()) != forge::json::dump(rb.manifest())) return fail("manifest");
    return true;
}

std::string calib_text() {
    static const std::string p = [] {
        std::string t;
        std::mt19937 rng(5);
        const char* words[] = {"the", "model", "compresses", "layers", "while", "a", "checkpoint", "keeps", "its", "progress", "on", "disk", "and", "resumes", "after", "crash"};
        for (int i = 0; i < 4000; ++i) { t += words[rng() % 16]; t += (i % 13 == 12) ? ".\n" : " "; }
        std::string path = tmp("calib.txt");
        std::ofstream(path) << t;
        return path;
    }();
    return p;
}

std::string tiny_dense_file(const std::string& name, unsigned seed, int layers = 3) {
    return make_tiny_model(tmp(name), tiny_cfg{.d = 32, .layers = layers, .heads = 4, .kv = 2, .ffn = 64, .seed = seed});
}

// A tiny Qwen3-MoE (4 experts, top-2) file, as in test_compress.cpp.
std::string tiny_moe_file(const std::string& name, unsigned seed = 3) {
    std::string base = make_tiny_model(tmp("moe_base.fpk"), tiny_cfg{.d = 32, .layers = 3, .heads = 4, .kv = 2, .ffn = 48, .seed = seed});
    Model m = Model::load(base);
    std::mt19937 rng(seed); std::normal_distribution<float> N(0, 1);
    m.spec.qk_norm = true; m.spec.family = "qwen3";
    const i64 d = m.spec.d_model, E = 4, f = 24, hd = m.spec.head_dim;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        auto& L = m.spec.layers[std::size_t(l)];
        for (const char* n : {"w_gate", "w_up", "w_down"}) m.w.t.erase(std::format("layer.{}.{}", l, n));
        L.ffn = LayerSpec::Ffn::moe; L.moe_k = 2; L.moe_norm = true;
        auto put = [&](const std::string& n, shape_t s, float sd, float mean = 0) { tensor t(s); for (auto& x : t.data) x = mean + sd * N(rng); m.w.t[n] = t; };
        const std::string p = std::format("layer.{}.", l);
        put(p + "q_norm", {hd}, .1f, 1); put(p + "k_norm", {hd}, .1f, 1);
        put(p + "router", {E, d}, 1.f / std::sqrt(float(d)));
        put(p + "experts.w_gate", {E, f, d}, 1.f / std::sqrt(float(d)));
        put(p + "experts.w_up", {E, f, d}, 1.f / std::sqrt(float(d)));
        put(p + "experts.w_down", {E, d, f}, 1.f / std::sqrt(float(f)));
    }
    std::string p = tmp(name);
    m.save(p);
    std::filesystem::remove(base);
    return p;
}

cli::opts job(const std::string& in, const std::string& out, const std::string& recipe, const std::string& ckpt = "none") {
    cli::opts o; o.in = in; o.out = out;
    o.kv = {{"recipe", recipe}, {"calib-data", calib_text()}, {"nsamples", "6"}, {"seqlen", "24"}, {"probes", "4"}, {"probe-len", "24"},
            {"checkpoint", ckpt}, {"checkpoint-every", "1"}};
    return o;
}

struct env_guard {
    std::string k;
    env_guard(std::string key, const std::string& v) : k(std::move(key)) { forge_test::set_env(k.c_str(), v.c_str()); }
    ~env_guard() { forge_test::set_env(k.c_str(), nullptr); }
};

// Runs `recipe` uninterrupted, then again with checkpoints, aborted at each of `aborts` in turn (each rerun resumes
// the previous one), and finally to the end: the two outputs must be identical.
void check_abort_resume(const std::string& in, const std::string& recipe, const std::vector<std::string>& aborts, const char* what) {
    const std::string ref = tmp(std::string(what) + "_ref.fpk"), out = tmp(std::string(what) + "_resumed.fpk"), ck = tmp(std::string(what) + ".ckpt");
    cli::compress_file(job(in, ref, recipe));
    for (auto& a : aborts) {
        env_guard g("FORGE_COMPRESS_ABORT_AFTER", a);
        bool threw = false;
        try { cli::compress_file(job(in, out, recipe, ck)); } catch (const std::runtime_error& e) { threw = std::string(e.what()).find("FORGE_COMPRESS_ABORT_AFTER") != std::string::npos; }
        CHECK_MSG(threw, "%s: no abort at %s", what, a.c_str());
        CHECK_MSG(std::filesystem::exists(ck + "/state.json"), "%s: no checkpoint after the abort at %s", what, a.c_str());
        CHECK(!std::filesystem::exists(out));
    }
    auto rep = cli::compress_file(job(in, out, recipe, ck));
    CHECK_MSG(!aborts.empty() && rep.find("resumed_after_stages"), "%s: the last run did not resume", what);
    std::string why;
    CHECK_MSG(same_model_file(ref, out, &why), "%s: resumed output differs from the uninterrupted run (%s)", what, why.c_str());
    CHECK_MSG(read_bytes(ref) == read_bytes(out), "%s: resumed output file is not byte-identical", what);
    CHECK_MSG(!std::filesystem::exists(ck), "%s: checkpoint directory not removed after success", what);
    std::filesystem::remove(ref); std::filesystem::remove(out);
}

#if defined(__linux__)
std::map<std::string, long> proc_status() {   // kB
    std::map<std::string, long> r;
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) { auto c = line.find(':'); if (c == std::string::npos) continue; r[line.substr(0, c)] = std::atol(line.c_str() + c + 1); }
    return r;
}
int mapped_scratch_regions() {   // live forge-compress scratch mappings of this process
    std::ifstream f("/proc/self/maps"); std::string line; int n = 0;
    while (std::getline(f, line)) n += line.find("forge-compress-") != std::string::npos;
    return n;
}
#endif

}  // namespace

// ------------------------------------------------------------------------------------------------ buffers
TEST(persist_mapped_vector_matches_vector) {
    for (std::size_t th : {std::size_t(SIZE_MAX), std::size_t(0)}) {   // heap, then every allocation a scratch mapping
        scratch::set_threshold(th);
        std::mt19937 rng(1);
        std::vector<float> v;
        fbuf m;
        for (int step = 0; step < 300; ++step) {
            const int op = int(rng() % 7);
            const float x = float(rng() % 1000) / 7.f;
            switch (op) {
                case 0: { std::size_t n = rng() % 3000; v.resize(n); m.resize(n); break; }
                case 1: { std::size_t n = rng() % 3000; v.resize(n, x); m.resize(n, x); break; }
                case 2: v.push_back(x); m.push_back(x); break;
                case 3: { std::size_t n = rng() % 500; v.assign(n, x); m.assign(n, x); break; }
                case 4: if (!v.empty()) { auto i = rng() % v.size(); v[i] = x; m[i] = x; } break;
                case 5: { std::vector<float> s(rng() % 50, x); auto at = v.empty() ? 0 : rng() % v.size(); v.insert(v.begin() + long(at), s.begin(), s.end()); m.insert(m.begin() + at, s.begin(), s.end()); break; }
                case 6: if (v.size() > 10) { v.erase(v.begin() + 2, v.begin() + 7); m.erase(m.begin() + 2, m.begin() + 7); } break;
            }
            CHECK(m.size() == v.size());
            if (!(m == v)) { CHECK_MSG(false, "mapped_vector differs from std::vector after step %d (op %d, threshold %zu)", step, op, th); break; }
        }
        fbuf c = m; CHECK(c == m);
        fbuf mv = std::move(c); CHECK(mv == v && c.empty());
        std::vector<float> back = m; CHECK(back == v);
        fbuf from = v; CHECK(from == m);
        mapped_vector<double> z(1000, 0.0); bool zero = true; for (double d : z) zero &= d == 0.0; CHECK(zero);
        CHECK(th != 0 || m.empty() || m.is_mapped());
    }
    scratch::set_threshold(std::size_t(64) << 20);
}

TEST(persist_scratch_files_are_freed) {
    const std::string d = tmp("scratch");
    std::filesystem::create_directories(d);
    scratch::set_dir(d);
    forge_test::set_env("FORGE_COMPRESS_SCRATCH", nullptr);
    scratch::set_threshold(1 << 16);
    const auto s0 = scratch::current();
    {
        std::vector<fbuf> bufs;
        for (int i = 0; i < 12; ++i) { bufs.emplace_back(std::size_t(1 << 16) + std::size_t(i) * 1000, float(i)); }
        bufs.emplace_back(100, 1.f);   // below the threshold: heap
        const auto s1 = scratch::current();
        CHECK_MSG(s1.files_live - s0.files_live == 12, "%lld live scratch files (expected 12)", (long long)(s1.files_live - s0.files_live));
        CHECK(s1.bytes_live - s0.bytes_live >= 12 * (1 << 18));
        for (int i = 0; i < 12; ++i) CHECK(bufs[std::size_t(i)].is_mapped() && bufs[std::size_t(i)][5] == float(i));
        CHECK(!bufs.back().is_mapped());
        bufs[0].resize(0);   // a big buffer emptied gives its file back
        CHECK(scratch::current().files_live - s0.files_live == 11);
#if defined(__linux__)
        CHECK(mapped_scratch_regions() >= 11);
#endif
        // a buffer that outgrows its mapping moves to a bigger one and frees the old file
        bufs[1].resize(std::size_t(1) << 20, 2.f);
        CHECK(scratch::current().files_live - s0.files_live == 11 && bufs[1][1 << 19] == 2.f && bufs[1][5] == 1.f);
    }
    const auto s2 = scratch::current();
    CHECK_MSG(s2.files_live == s0.files_live && s2.bytes_live == s0.bytes_live, "scratch files left: %lld (%lld bytes)", (long long)(s2.files_live - s0.files_live), (long long)(s2.bytes_live - s0.bytes_live));
#if defined(__linux__)
    CHECK_MSG(mapped_scratch_regions() == 0, "%d scratch mappings still in /proc/self/maps", mapped_scratch_regions());
#endif
    std::size_t left = 0; for (auto& e : std::filesystem::directory_iterator(d)) { (void)e; ++left; }
    CHECK_MSG(left == 0, "%zu files left in the scratch directory", left);
    scratch::set_threshold(std::size_t(64) << 20);
    scratch::set_dir("");
}

TEST(persist_lazy_tensors) {
    const std::string p = tiny_dense_file("lazy.fpk", 11);
    Model a = Model::load(p);
    Model b;
    { env_guard g("FORGE_COMPRESS_LAZY", "0"); b = Model::load(p); }
    CHECK(a.w.at("layer.0.wq").data.is_lazy() && !b.w.at("layer.0.wq").data.is_lazy());
    bool same = true;
    for (auto& [n, t] : b.w.t) same &= std::as_const(a).w.at(n).data == t.data && a.w.at(n).shape == t.shape;
    CHECK(same);
    CHECK(a.w.at("layer.0.wq").data.is_lazy());   // reading does not copy
    const float* src = std::as_const(a.w.at("layer.0.wq").data).data();
    wtensor copy = a.w.at("layer.0.wq");           // a copy shares the source
    CHECK(copy.data.is_lazy() && std::as_const(copy.data).data() == src);
    a.w.at("layer.0.wq").data[3] += 1.f;          // the first write gives the tensor its own storage
    CHECK(!a.w.at("layer.0.wq").data.is_lazy() && std::as_const(a.w.at("layer.0.wq").data).data() != src);
    CHECK(copy.data[3] + 1.f == a.w.at("layer.0.wq").data[3] && copy.data == b.w.at("layer.0.wq").data);
    // the tied head shares the embedding's source
    CHECK(count(a).params == count(b).params);
    std::filesystem::remove(p);
}

// ------------------------------------------------------------------------------------------------ pipelines
TEST(persist_mapped_pipeline_equals_ram) {
    const std::string in = tiny_dense_file("pipe.fpk", 21, 4);
    const std::string recipe = "width(s=0.25,score=obs,round=16)+prune(method=sparsegpt,s=0.5,refit=mcr,iters=5)+quant(method=mixed,ladder=q4_0/q8_0/f16,ref=q8_0,head=none,kl=0.2)+calibrate";
    std::string ram = tmp("pipe_ram.fpk"), mapped = tmp("pipe_mapped.fpk");
    {
        env_guard g("FORGE_COMPRESS_LAZY", "0");
        scratch::set_threshold(SIZE_MAX);
        auto o = job(in, ram, recipe); o.kv["calib-data"] = "self";
        cli::compress_file(o);
    }
    {
        env_guard g("FORGE_COMPRESS_SCRATCH", tmp("scratch2"));
        std::filesystem::create_directories(tmp("scratch2"));
        scratch::set_threshold(0);   // every buffer of the run in a scratch-file mapping
        const auto s0 = scratch::current();
        auto o = job(in, mapped, recipe); o.kv["calib-data"] = "self";
        cli::compress_file(o);
        CHECK_MSG(scratch::current().files_created - s0.files_created > 100, "only %llu scratch mappings were used", (unsigned long long)(scratch::current().files_created - s0.files_created));
        scratch::set_threshold(std::size_t(64) << 20);
    }
    std::string why;
    CHECK_MSG(same_model_file(ram, mapped, &why), "mapped run differs from the in-RAM run (%s)", why.c_str());
    CHECK(read_bytes(ram) == read_bytes(mapped));
    std::filesystem::remove(ram); std::filesystem::remove(mapped); std::filesystem::remove(in);
}

TEST(persist_abort_resume_prune) {
    const std::string in = tiny_dense_file("prune_in.fpk", 31, 4);
    check_abort_resume(in, "prune(method=sparsegpt,s=0.5,refit=mcr,iters=5)", {"prune:1"}, "prune");
    // two stages: an abort inside the second one, then one in its last layer
    check_abort_resume(in, "prune(method=wanda,s=0.3)+prune(method=sparsegpt,nm=2:4)", {"stage:0", "prune:2", "prune:3"}, "prune2");
    std::filesystem::remove(in);
}

TEST(persist_abort_resume_quant_mixed) {
    const std::string in = tiny_dense_file("mixed_in.fpk", 41, 4);
    const std::string q = "quant(method=mixed,ladder=q4_0/q8_0/f16,ref=q8_0,head=q8_0,kl=0.2)";
    check_abort_resume(in, q, {"mixed.err:1"}, "mixed_err");        // inside the error-curve pass
    check_abort_resume(in, q, {"mixed.apply:1"}, "mixed_apply");    // inside the GPTQ pass
    check_abort_resume(in, "prune(method=wanda,s=0.4)+" + q + "+calibrate", {"prune:0", "mixed.err:2", "mixed.apply:0", "stage:1"}, "chain");
    std::filesystem::remove(in);
}

TEST(persist_abort_resume_structural) {
    const std::string in = tiny_dense_file("width_in.fpk", 51, 4);
    check_abort_resume(in, "width(s=0.25,score=obs,round=16)", {"width:1"}, "width");           // shapes change
    check_abort_resume(in, "quant(method=gptq,codec=q8_0)", {"quant:2"}, "gptq");
    std::filesystem::remove(in);
    const std::string moe = tiny_moe_file("moe_in.fpk");
    check_abort_resume(moe, "moe2dense(width=0.5)", {"moe2dense:1"}, "moe2dense");     // the spec changes
    check_abort_resume(moe, "merge(drop=1,method=ls)", {"merge:0"}, "merge");
    std::filesystem::remove(moe);
}

TEST(persist_oom_drill_and_lock) {
    const std::string in = tiny_dense_file("oom_in.fpk", 61, 3);
    const std::string out = tmp("oom.fpk"), ck = tmp("oom.ckpt");
    {
        env_guard g("FORGE_COMPRESS_ABORT_AFTER", "prune:0");
        env_guard o("FORGE_COMPRESS_ABORT_OOM", "1");
        bool bad_alloc = false;
        try { cli::compress_file(job(in, out, "prune(method=wanda,s=0.5)", ck)); } catch (const std::bad_alloc&) { bad_alloc = true; }
        CHECK(bad_alloc);
    }
    {   // a second process (here: a second Checkpoint) may not use a directory that is in use
        CheckpointCfg cfg; cfg.dir = ck; cfg.key = "x";
        Checkpoint held(cfg);
        bool refused = false;
        try { cli::compress_file(job(in, out, "prune(method=wanda,s=0.5)", ck)); } catch (const std::runtime_error& e) { refused = std::string(e.what()).find("in use") != std::string::npos; }
        CHECK(refused);
    }
    // the directory now belongs to key "x": the real job starts afresh and completes
    auto rep = cli::compress_file(job(in, out, "prune(method=wanda,s=0.5)", ck));
    CHECK(!rep.find("resumed_after_stages") && std::filesystem::exists(out) && !std::filesystem::exists(ck));
    std::filesystem::remove(out); std::filesystem::remove(in);
}

// ------------------------------------------------------------------------------------------------ memory
TEST(persist_rss_real_model) {
#if defined(__linux__)
    const std::string p = os::getenv("FORGE_COMPRESS_RSS_MODEL", "");
    if (p.empty()) { std::printf("  (FORGE_COMPRESS_RSS_MODEL not set: real-model memory measurement skipped)\n"); return; }
    auto show = [](const char* what) {
        auto s = proc_status();
        std::printf("  %-34s VmHWM %7.0f MB  VmRSS %7.0f MB  RssAnon %7.0f MB  RssFile %7.0f MB  RssShmem %6.0f MB\n", what, s["VmHWM"] / 1024.0, s["VmRSS"] / 1024.0,
                    s["RssAnon"] / 1024.0, s["RssFile"] / 1024.0, s["RssShmem"] / 1024.0);
    };
    std::printf("  FORGE_COMPRESS_LAZY=%s FORGE_COMPRESS_MMAP_MIN=%s\n", os::getenv("FORGE_COMPRESS_LAZY", "1").c_str(), os::getenv("FORGE_COMPRESS_MMAP_MIN", "64M").c_str());
    show("start");
    const double t0 = now_s();
    Model m = Model::load(p);
    std::printf("  load %.2fs\n", now_s() - t0);
    show("after Model::load");
    auto c = count(m);
    show("after count (reads every tensor)");
    for (auto& [n, t] : m.w.t) (void)t.data.data();   // what a compression pass does to the tensors it writes
    show("after first write to every tensor");
    CalibSet cs; cs.seq = 512; for (int i = 0; i < 32; ++i) cs.seqs.push_back(std::vector<int>(512, 100 + i));
    Hidden h = Hidden::embed(m, cs);
    la::Gram g(3072);
    show("+ hidden 32x512 + Gram 3072^2");
    std::printf("  params %lld\n", (long long)c.params);
#endif
}

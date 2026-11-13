#pragma once
// Build side of ahead-of-time compilation (`forge aot`, docs/aot-models.md): compile a model's plans for a list of
// targets and write them as a plan blob (plan mode), or also generate and compile their straight-line code (native
// mode, aot_codegen.hpp).
#include <forge/graph/aot_codegen.hpp>
#include <forge/inference/session.hpp>
#include <forge/os/fs.hpp>
#include <forge/os/process.hpp>
#include <chrono>
#include <deque>

namespace forge::inference {

struct aot_plans {
    std::vector<std::uint8_t> blob;               // graph::aot blob: the "aot" section / sidecar bytes
    std::vector<graph::aot::target> targets;      // one per blob entry, in order
    std::size_t nodes = 0, steps = 0;             // totals over the plans, for reporting
};

// Compile the plans of `m` for every (shape, all_rows, KV codec) in `o` with the CPU lowerings, exactly as a session
// would at run time, and serialise them. `each(index, target, plan)` sees every compiled plan (native code
// generation hooks in here). Precompiled plans already attached to `m` are ignored (set_aot(false)).
template <class Each>
inline aot_plans aot_compile_plans(model& m, aot_build_options o, Each&& each) {
    if (m.device() != "cpu") raise(errc::unsupported, "aot: plans can only be precompiled for the CPU (device is '{}'); a device plan holds kernels the backend creates per process", m.device());
    m.set_aot(false);
    const bool generative = m.blueprint().session == "generative_sequence";
    if (o.shapes.empty()) o.shapes = generative ? aot_default_shapes() : std::vector<i64>{1};
    if (o.kv_codecs.empty()) o.kv_codecs = {"f32"};
    i64 ctx = o.context ? o.context : m.max_context();
    std::sort(o.shapes.begin(), o.shapes.end());
    o.shapes.erase(std::unique(o.shapes.begin(), o.shapes.end()), o.shapes.end());
    graph::aot::builder b(modelfile::aot::model_key(m.file()));
    aot_plans out;
    std::deque<graph::state_store> stores;   // plans point into their store's slots while we serialise them
    for (auto& kc : o.kv_codecs) {
        auto kp = graph::kv_codec_pair_from_name(kc);
        for (auto n : o.shapes) {
            if (n < 1 || n > ctx) raise(errc::invalid_argument, "aot: shape {} is outside the context 1..{}", n, ctx);
            for (int all = 0; all < (o.all_rows ? 2 : 1); ++all) {
                auto& st = stores.emplace_back();
                st.capacity = ctx; st.page_rows = 256;
                st.codec = kp.k; st.codec_v = kp.v != kp.k ? kp.v : graph::kv_codec::f32;
                graph::aot::target t;
                t.seq = n; t.all_rows = all != 0; t.kv_codec = graph::kv_codec_pair_name({st.codec, st.codec_v == graph::kv_codec::f32 ? st.codec : st.codec_v});
                bool dup = false; for (auto& x : out.targets) dup = dup || x.name() == t.name();
                if (dup) continue;   // two spellings of one codec
                auto& p = m.plan_for(n, all != 0, &st);
                b.add(t, p);
                each(out.targets.size(), t, p);
                out.targets.push_back(t);
                out.nodes += p.g.size(); out.steps += p.steps.size();
            }
        }
    }
    out.blob = b.finish(simd::active().target);
    return out;
}
inline aot_plans aot_compile_plans(model& m, aot_build_options o = {}) {
    return aot_compile_plans(m, std::move(o), [](std::size_t, const graph::aot::target&, graph::plan&) {});
}

// ---- native mode: generate the plans' code and compile it with the host C++ compiler ----

// What a translation unit that includes the FORGE headers needs: written by the FORGE build as forge_aot_sdk.txt
// (key=value lines: cxx, ar, flag, include, define, lib, syslib).
struct aot_sdk {
    std::string file, cxx, cxx_id, ar;
    std::vector<std::string> flags, includes, defines, libs, syslibs;
};
// The SDK description: `given`, else $FORGE_AOT_SDK, else the build tree this binary came from (FORGE_AOT_SDK_FILE),
// else forge_aot_sdk.txt next to the program or in ../share/forge.
inline aot_sdk aot_load_sdk(const std::string& given = {}) {
    std::vector<std::string> tried;
    auto pick = [&]() -> std::string {
        if (!given.empty()) return given;
        if (const char* e = std::getenv("FORGE_AOT_SDK"); e && *e) return e;
        std::vector<std::string> c;
#if defined(FORGE_AOT_SDK_FILE)
        c.push_back(FORGE_AOT_SDK_FILE);
#endif
        auto dir = forge::os::dirname(forge::os::exe_path());
        c.push_back(dir + "/forge_aot_sdk.txt"); c.push_back(dir + "/../share/forge/forge_aot_sdk.txt");
        for (auto& p : c) { if (forge::os::exists(p)) return p; tried.push_back(p); }
        return {};
    };
    aot_sdk k; k.file = pick();
    if (k.file.empty()) raise(errc::not_found, "aot: no FORGE build description (forge_aot_sdk.txt) found; pass --sdk FILE or set FORGE_AOT_SDK (tried {})", tried.empty() ? "nothing" : tried.front());
    std::ifstream f(k.file);
    if (!f) raise(errc::io_error, "aot: cannot read {}", k.file);
    for (std::string line; std::getline(f, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto eq = line.find('=');
        if (line.empty() || line[0] == '#' || eq == std::string::npos) continue;
        auto key = line.substr(0, eq), val = line.substr(eq + 1);
        // an installed description names its paths relative to the prefix: <prefix>/share/forge/forge_aot_sdk.txt
        if (auto at = val.find("${prefix}"); at != std::string::npos) val.replace(at, 9, forge::os::dirname(forge::os::dirname(forge::os::dirname(forge::os::absolute(k.file)))));
        if (key == "cxx") k.cxx = val; else if (key == "cxx_id") k.cxx_id = val; else if (key == "ar") k.ar = val;
        else if (key == "flag") k.flags.push_back(val); else if (key == "include") k.includes.push_back(val);
        else if (key == "define") k.defines.push_back(val); else if (key == "lib") k.libs.push_back(val);
        else if (key == "syslib") k.syslibs.push_back(val);
    }
    return k;
}

struct aot_native_options {
    std::string out;                      // the artefact
    std::string emit = "shared";          // shared | static | exe
    std::string cxx;                      // compiler; empty: $CXX, the compiler FORGE was built with, then c++ / clang++ / clang++-23 / g++ / cl on PATH
    std::string target;                   // passed as --target=<triple> (cross compiling; the SDK libraries must be for it)
    std::string simd;                     // lowest SIMD target of the code; empty: this machine's (simd::active())
    std::string march;                    // -march value; empty: from `simd`; "none": no -march
    std::string sdk;                      // forge_aot_sdk.txt; empty: found by aot_load_sdk()
    std::string work_dir;                 // where the generated source goes; empty: a temp directory, removed afterwards
    bool embed_weights = false;           // put the .fpk into the artefact
    std::uint64_t embed_limit = std::uint64_t(1) << 30;   // refuse to embed files above this (2 GB relocation limit of the small code model)
    std::vector<std::string> extra_flags; // appended to the compile line
    float temperature = 0; int top_k = 0; float top_p = 1; int max_new_tokens = 32;   // defaults of the executable
};
struct aot_native_result {
    std::string out, source, compiler, command, simd;
    bool same_compiler = true;            // compiled by the compiler that built FORGE (what the bit-identity tests cover)
    aot_plans plans;
    std::size_t direct_steps = 0, generic_steps = 0;
    double generate_s = 0, compile_s = 0;
};

namespace aot_detail {
// -march for the generic (non-kernel) code. The SIMD kernels come precompiled from libforge_simd_mv and dispatch at
// run time; everything else in the unit is ordinary C++ that may use the target's instructions. On x86 the FORGE
// build compiles that code for baseline x86-64, where there is no FMA to contract a*b+c into, so -ffp-contract=off
// goes with -march to keep its results bit-identical (docs/aot-models.md).
inline std::vector<std::string> march_flags(const std::string& simd, const std::string& march, const aot_sdk& sdk) {
    for (auto& f : sdk.flags) if (f.rfind("-m", 0) == 0) return {};   // FORGE itself was built for a specific machine: same flags
    std::string m = march;
    if (m == "none") return {};
#if defined(__x86_64__) || defined(_M_X64)
    if (m.empty()) m = simd == "avx512" ? "x86-64-v4" : simd == "avx2" ? "x86-64-v3" : "";
    if (m.empty()) return {};
    return {"-march=" + m, "-ffp-contract=off"};
#else
    (void)simd;
    if (m.empty()) return {};
    return {"-march=" + m};
#endif
}
inline std::string join(const std::vector<std::string>& v) { std::string s; for (auto& x : v) { if (!s.empty()) s += ' '; s += x; } return s; }
inline std::string tail(const std::string& s, std::size_t n = 4000) { return s.size() > n ? "..." + s.substr(s.size() - n) : s; }
} // namespace aot_detail

// Compile the plans of the model at `fpk_path` to native code. Raises errc::unsupported where no compiler can be
// run (iOS, a sandbox without process spawning, a compiler that is not GCC- or Clang-compatible).
inline aot_native_result aot_build_native(const std::string& fpk_path, const aot_build_options& bo, aot_native_options o) {
#if defined(FORGE_OS_IOS)
    raise(errc::unsupported, "aot: native compilation needs a host compiler, which this platform cannot run; use --aot-mode plan");
#endif
    using clk = std::chrono::steady_clock;
    if (o.out.empty()) raise(errc::invalid_argument, "aot: no output path");
    if (o.emit != "shared" && o.emit != "static" && o.emit != "exe") raise(errc::invalid_argument, "aot: --emit {} (shared, static or exe)", o.emit);
    aot_native_result res;
    auto sdk = aot_load_sdk(o.sdk);
    // compiler
    std::string cxx = o.cxx;
    if (cxx.empty()) if (const char* e = std::getenv("CXX"); e && *e) cxx = e;
    if (cxx.empty() && !sdk.cxx.empty() && !forge::os::find_executable(sdk.cxx).empty()) cxx = sdk.cxx;
    if (cxx.empty()) for (const char* c : {"c++", "clang++", "clang++-23", "g++", "cl"}) if (!forge::os::find_executable(c).empty()) { cxx = c; break; }
    if (cxx.empty() || forge::os::find_executable(cxx).empty()) raise(errc::unsupported, "aot: no C++ compiler found (--cxx, $CXX, {}, c++, clang++, clang++-23, g++, cl); use --aot-mode plan", sdk.cxx.empty() ? "the build's compiler" : sdk.cxx);
    // Another compiler than the one that built FORGE: the unit's generic code is compiled by it while the SIMD kernels
    // stay the build's. Keep floating point strict (no contraction) so the two cannot differ there; the build's own
    // compiler gets the build's flags unchanged.
    res.same_compiler = sdk.cxx.empty() || forge::os::find_executable(cxx) == forge::os::find_executable(sdk.cxx);
    {
        auto base = forge::os::basename(cxx);
        for (auto& c : base) c = char(std::tolower(static_cast<unsigned char>(c)));
        if (base == "cl" || base == "cl.exe") raise(errc::unsupported, "aot: native compilation drives a GCC- or Clang-compatible compiler; use clang-cl's clang++ or --aot-mode plan");
    }
    res.compiler = cxx;
    // plans + code
    auto t0 = clk::now();
    auto abs_fpk = forge::os::absolute(fpk_path);
    if (o.embed_weights) {
        auto sz = forge::os::fs::file_size(abs_fpk);
        if (!sz) raise(errc::io_error, "aot: cannot stat {}", abs_fpk);
        if (sz.value > o.embed_limit) raise(errc::unsupported, "aot: {} is {} MB, above the embed limit of {} MB (the 2 GB limit of the default code model is close); keep the weights in the .fpk", abs_fpk, sz.value >> 20, o.embed_limit >> 20);
    }
    res.simd = o.simd.empty() ? std::string(simd::active().target) : o.simd;
    auto mflags = aot_detail::march_flags(res.simd, o.march, sdk);
    if (mflags.empty() && o.simd.empty()) res.simd = "scalar";   // nothing in the unit needs more than the baseline
    graph::aot::codegen gen;
    {
        model m(fpk_path);
        if (m.file().has_section("encryption") && o.embed_weights) raise(errc::unsupported, "aot: an encrypted model cannot be embedded");
        if (m.file().sharded() && o.embed_weights) raise(errc::unsupported, "aot: a sharded model cannot be embedded (its tensors are in blob files); join it first (forge shard join)");
        res.plans = aot_compile_plans(m, bo, [&](std::size_t, const graph::aot::target& t, graph::plan& p) { gen.add(t, p); });
    }
    std::vector<std::string> cmd = {cxx};
    for (auto& f : sdk.flags) cmd.push_back(f);
    cmd.insert(cmd.end(), {"-O2", "-DNDEBUG", "-fPIC", "-fvisibility=hidden", "-fvisibility-inlines-hidden"});
    for (auto& f : mflags) cmd.push_back(f);
    if (!res.same_compiler && std::find(mflags.begin(), mflags.end(), "-ffp-contract=off") == mflags.end()) cmd.push_back("-ffp-contract=off");
    if (!o.target.empty()) cmd.push_back("--target=" + o.target);
    for (auto& i : sdk.includes) cmd.push_back("-I" + i);
    for (auto& d : sdk.defines) cmd.push_back("-D" + d);
    if (o.emit == "exe") cmd.push_back("-DFORGE_AOT_EXE=1");
    for (auto& f : o.extra_flags) cmd.push_back(f);
    graph::aot::codegen_options co;
    co.fpk_path = abs_fpk; co.embed_fpk = o.embed_weights ? abs_fpk : std::string(); co.simd_target = res.simd;
    co.compiler = (sdk.cxx_id.empty() ? cxx : sdk.cxx_id) + (mflags.empty() ? std::string() : " " + aot_detail::join(mflags));
    co.temperature = o.temperature; co.top_k = o.top_k; co.top_p = o.top_p; co.max_new_tokens = o.max_new_tokens;
    std::string src = gen.finish(res.plans.blob, co);
    res.direct_steps = gen.direct_steps(); res.generic_steps = gen.generic_steps();
    bool temp = o.work_dir.empty();
    std::string dir = o.work_dir;
    if (temp) { auto d = forge::os::fs::make_temp_dir("forge-aot-"); if (!d) raise(errc::io_error, "aot: cannot create a work directory: {}", d.err.what); dir = d.value; }
    else if (auto mk = forge::os::fs::make_dirs(dir); !mk) raise(errc::io_error, "aot: cannot create {}: {}", dir, mk.err.what);
    struct cleanup { std::string d; bool on; ~cleanup() { if (on) forge::os::remove_all(d); } } cl{dir, temp};
    res.source = dir + "/forge_aot_model.cpp";
    { std::ofstream f(res.source, std::ios::binary | std::ios::trunc); f << src; if (!f) raise(errc::io_error, "aot: cannot write {}", res.source); }
    res.generate_s = std::chrono::duration<double>(clk::now() - t0).count();
    // compile (and link)
    auto t1 = clk::now();
    auto run = [&](const std::vector<std::string>& argv) {
        forge::os::spawn_spec sp; sp.argv = argv; sp.merge_stderr = true;
        auto r = forge::os::spawn_wait(sp);
        if (!r) raise(errc::unsupported, "aot: cannot run {}: {}; use --aot-mode plan where no compiler can be started", argv[0], r.err.what);
        if (r.value.exit_code != 0) raise(errc::invalid_argument, "aot: {} failed with exit code {} (source kept with --work DIR)\n{}\n{}", argv[0], r.value.exit_code, aot_detail::join(argv), aot_detail::tail(r.value.out));
    };
    auto out = forge::os::absolute(o.out);
    if (o.emit == "static") {
        auto obj = dir + "/forge_aot_model.o";
        auto c = cmd; c.insert(c.end(), {"-c", res.source, "-o", obj});
        res.command = aot_detail::join(c);
        run(c);
        std::string ar;
        if (const char* e = std::getenv("AR"); e && *e) ar = e;
        if (ar.empty() && !sdk.ar.empty() && !forge::os::find_executable(sdk.ar).empty()) ar = sdk.ar;
        if (ar.empty()) for (const char* c2 : {"ar", "llvm-ar"}) if (!forge::os::find_executable(c2).empty()) { ar = c2; break; }
        if (ar.empty()) raise(errc::unsupported, "aot: no archiver found ($AR, ar, llvm-ar) for --emit static");
        forge::os::remove(out);
        run({ar, "rcs", out, obj});
    } else {
        auto c = cmd;
        c.push_back(res.source);
        if (o.emit == "shared") c.push_back("-shared");
        c.insert(c.end(), {"-o", out});
        for (auto& l : sdk.libs) c.push_back(l);
        for (auto& l : sdk.syslibs) {   // "-framework Metal" (macOS) is two arguments
            if (l.rfind("-framework ", 0) == 0) { c.push_back("-framework"); c.push_back(l.substr(11)); } else c.push_back(l);
        }
#if !defined(__APPLE__) && !defined(_WIN32)
        if (o.emit == "shared" && o.target.empty()) c.push_back("-Wl,-Bsymbolic");   // the library calls its own copies of the header code
#endif
        res.command = aot_detail::join(c);
        run(c);
    }
    res.compile_s = std::chrono::duration<double>(clk::now() - t1).count();
    res.out = out;
    if (!temp) res.source = dir + "/forge_aot_model.cpp"; else res.source.clear();
    return res;
}

} // namespace forge::inference

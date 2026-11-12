// forge::script AOT driver (docs/aot.md): `forge script --aot`.
//   native: translate (aot_codegen.cpp), then run the host C++ compiler against the AOT runtime library described
//           by forge-aot-runtime.cfg (written by the CMake build and by the install rules);
//   embed:  copy a runner stub and append the payload and its trailer.
#include <forge/script/aot.hpp>
#include "aot.hpp"
#include <forge/os/os.hpp>
#include <forge/os/fs.hpp>
#include <forge/os/process.hpp>
#include <forge/hash/blake3.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

#ifndef FORGE_AOT_BUILD_RUNTIME_DIR
#define FORGE_AOT_BUILD_RUNTIME_DIR ""
#endif

namespace forge::script {

bool aot_supported() {
#if defined(FORGE_OS_IOS) || defined(FORGE_OS_ANDROID) || defined(__ANDROID__)
    return false;
#else
    return true;
#endif
}

namespace {
namespace fs = std::filesystem;

#if defined(_WIN32)
constexpr const char* exe_suffix = ".exe";
#else
constexpr const char* exe_suffix = "";
#endif

// forge-aot-runtime.cfg: `key=value` lines; include / lib / cxxflag / ldflag / define / clang_cxxflag repeat.
// Relative paths are relative to the directory of the file.
struct runtime_cfg {
    std::string dir;
    std::vector<std::string> includes, libs, cxxflags, clang_cxxflags, ldflags, defines;
    std::string stub, abi, triple;
    bool found = false;
};
std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
    std::size_t i = 0; while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.substr(i);
}
runtime_cfg load_cfg(const std::string& dir) {
    runtime_cfg c; c.dir = dir;
    std::ifstream f(fs::path(dir) / "forge-aot-runtime.cfg");
    if (!f) return c;
    c.found = true;
    auto path = [&](const std::string& v) { fs::path p(v); return p.is_absolute() ? p.string() : (fs::path(dir) / p).string(); };
    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = trim(line.substr(0, eq)), v = trim(line.substr(eq + 1));
        if (v.empty()) continue;
        if (k == "include") c.includes.push_back(path(v));
        else if (k == "lib") c.libs.push_back(path(v));
        else if (k == "cxxflag") c.cxxflags.push_back(v);
        else if (k == "clang_cxxflag") c.clang_cxxflags.push_back(v);
        else if (k == "ldflag") c.ldflags.push_back(v);
        else if (k == "define") c.defines.push_back(v);
        else if (k == "stub") c.stub = path(v);
        else if (k == "abi") c.abi = v;
        else if (k == "triple") c.triple = v;
    }
    return c;
}
runtime_cfg find_runtime(const aot_options& o, std::string& tried) {
    std::vector<std::string> dirs;
    if (!o.runtime_dir.empty()) dirs.push_back(o.runtime_dir);
    else {
        if (auto e = os::getenv("FORGE_AOT_RUNTIME"); !e.empty()) dirs.push_back(e);
        if (auto exe = os::exe_path(); !exe.empty()) {
            fs::path d = fs::path(exe).parent_path();
            dirs.push_back((d / ".." / "lib" / "forge" / "aot").lexically_normal().string());
            dirs.push_back((d / "aot-runtime").string());
        }
        if (std::string b = FORGE_AOT_BUILD_RUNTIME_DIR; !b.empty()) dirs.push_back(b);
    }
    for (auto& d : dirs) {
        auto c = load_cfg(d);
        if (c.found) return c;
        tried += (tried.empty() ? "" : ", ") + d;
    }
    return {};
}

std::string find_cxx(const aot_options& o) {
    if (!o.cxx.empty()) { auto p = os::find_executable(o.cxx); return p.empty() ? std::string() : p; }
    if (auto e = os::getenv("CXX"); !e.empty()) { if (auto p = os::find_executable(e); !p.empty()) return p; }
#if defined(_WIN32)
    for (const char* n : {"cl", "clang++", "clang++-23", "g++", "c++"})
#else
    for (const char* n : {"c++", "clang++", "clang++-23", "g++"})
#endif
        if (auto p = os::find_executable(n); !p.empty()) return p;
    return {};
}
bool is_msvc(const std::string& cxx) {
    std::string b = fs::path(cxx).stem().string();
    for (auto& ch : b) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    return b == "cl" || b == "clang-cl";
}

os::result<os::proc_result> run(const std::vector<std::string>& argv, std::uint64_t timeout_ms) {
    os::spawn_spec sp; sp.argv = argv; sp.merge_stderr = true; sp.timeout_ms = timeout_ms;
    return os::spawn_wait(sp);
}
std::string join_cmd(const std::vector<std::string>& argv) {
    std::string s;
    for (auto& a : argv) { if (!s.empty()) s += ' '; bool q = a.find_first_of(" \t\"'") != std::string::npos; s += q ? "'" + a + "'" : a; }
    return s;
}

void write_file(const std::string& path, std::string_view data) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(data.data(), std::streamsize(data.size()));
    f.close();
    if (!f) throw aot_error(0, "cannot write " + path, 1);
}
void make_executable(const std::string& path) {
    std::error_code ec;
    fs::permissions(path, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec, fs::perm_options::add, ec);
}
// Apple Silicon refuses to run a Mach-O whose code signature does not cover the file: after appending the
// payload the copy is re-signed ad hoc when codesign is available (docs/aot.md).
std::string resign_if_macho(const std::string& path) {
#if defined(__APPLE__)
    auto cs = os::find_executable("codesign");
    if (cs.empty()) return "; codesign not found: sign it with `codesign -s - -f` before running it on Apple Silicon";
    auto r = run({cs, "-s", "-", "-f", path}, 60000);
    if (!r || r->exit_code != 0) return "; ad hoc code signing failed (" + (r ? trim(r->out) : r.err.what) + "): sign it with `codesign -s - -f`";
    return "; re-signed ad hoc";
#else
    (void)path; return {};
#endif
}

aot_result build_embed(const aot::translated& tr, const std::string& out, const aot_options& o, const runtime_cfg& cfg) {
    std::string stub = o.stub;
    if (stub.empty() && !cfg.stub.empty() && fs::exists(cfg.stub)) stub = cfg.stub;
    const std::string self = os::exe_path();
    if (stub.empty() && !self.empty()) {
        fs::path cand = fs::path(self).parent_path() / (std::string("forge-script-run") + exe_suffix);
        std::error_code ec; if (fs::is_regular_file(cand, ec)) stub = cand.string();
    }
    if (stub.empty() && o.self_is_runner) stub = self;
    if (stub.empty()) throw aot_error(0, "aot embed: no runner stub found (forge-script-run next to this executable or in the AOT runtime directory; or pass a stub)", 1);
    auto stub_bytes = os::read_file(stub);
    if (!stub_bytes) throw aot_error(0, "aot embed: cannot read runner stub " + stub + ": " + stub_bytes.err.what, 1);
    std::string body = stub_bytes.value;
    // A stub that already carries a program (an AOT binary used as the stub) is cut back to the bare runner.
    {
        std::vector<std::uint8_t> old; std::string why;
        if (aot::read_embedded(stub, old, &why) == 1 && body.size() >= old.size() + aot::trailer_size) body.resize(body.size() - old.size() - aot::trailer_size);
    }
    auto pb = aot::encode_payload(tr.pl);
    auto trailer = aot::make_trailer(pb);
    body.append(reinterpret_cast<const char*>(pb.data()), pb.size());
    body.append(reinterpret_cast<const char*>(trailer.data()), trailer.size());
    std::error_code ec; fs::remove(out, ec);   // never write through an existing executable that may be running
    write_file(out, body);
    make_executable(out);
    aot_result r; r.mode = aot_mode::embed; r.output = out;
    r.detail = "embed: runner stub " + stub + " + " + std::to_string(pb.size()) + " byte program" + resign_if_macho(out);
    return r;
}

aot_result build_native(const aot::translated& tr, const std::string& out, const aot_options& o, const runtime_cfg& cfg, const std::string& cxx) {
    if (!cfg.abi.empty() && cfg.abi != std::to_string(aot::abi_version))
        throw aot_error(0, "aot native: the runtime in " + cfg.dir + " has ABI " + cfg.abi + ", this code generator emits ABI " + std::to_string(aot::abi_version), 1);
    for (auto& l : cfg.libs) { std::error_code ec; if (!fs::exists(l, ec)) throw aot_error(0, "aot native: runtime library " + l + " is missing (build the forge_script_rt target)", 1); }
    // Work in a private temporary directory; the generated source is kept only on request.
    auto tmp = os::fs::make_temp_dir("forge-aot-");
    if (!tmp) throw aot_error(0, "aot native: cannot create a temporary directory: " + tmp.err.what, 1);
    struct Cleanup { std::string d; ~Cleanup() { std::error_code ec; fs::remove_all(d, ec); } } cleanup{tmp.value};
    const std::string stem = fs::path(out).stem().string().empty() ? std::string("program") : fs::path(out).stem().string();
    const std::string src = (fs::path(tmp.value) / (stem + ".aot.cpp")).string();
    write_file(src, tr.cxx);

    const bool msvc = is_msvc(cxx);
    bool clang = false;
    if (!msvc) { auto v = run({cxx, "--version"}, 20000); clang = v && v->out.find("clang") != std::string::npos; }
    std::vector<std::string> cmd{cxx};
    const std::string& O = o.opt_level;
    std::string pch_note;
    if (msvc) {
        cmd.insert(cmd.end(), {"/nologo", "/std:c++latest", "/EHsc", "/bigobj", "/w", "/MD", "/DNDEBUG", O == "0" ? "/Od" : (O == "s" || O == "z") ? "/O1" : "/O2"});
        for (auto& d : cfg.defines) cmd.push_back("/D" + d);
        for (auto& i : cfg.includes) cmd.push_back("/I" + i);
        for (auto& f : o.cxx_flags) cmd.push_back(f);
        cmd.push_back(src);
        cmd.push_back("/Fo" + (fs::path(tmp.value) / (stem + ".obj")).string());
        cmd.push_back("/Fe" + out);
        cmd.push_back("/link");
        for (auto& l : cfg.libs) cmd.push_back(l);
        for (auto& f : cfg.ldflags) cmd.push_back(f);
    } else {
        cmd.insert(cmd.end(), {"-std=c++23", "-O" + O, "-w", "-DNDEBUG"});
        if (!o.target.empty()) cmd.push_back("--target=" + o.target);
        for (auto& f : cfg.cxxflags) cmd.push_back(f);
        if (clang) for (auto& f : cfg.clang_cxxflags) cmd.push_back(f);
        for (auto& d : cfg.defines) cmd.push_back("-D" + d);
        for (auto& f : o.cxx_flags) cmd.push_back(f);
        // Optional precompiled runtime header (aot_options::pch_dir): most of a compile is parsing the runtime's
        // headers, which are the same for every script. One PCH per (compiler, flags, runtime) lives in a
        // directory the caller names; it must be private to the user, because a PCH is trusted like source.
        std::string pch_dir = o.pch_dir.empty() ? os::getenv("FORGE_AOT_PCH_DIR") : o.pch_dir;
        if (!pch_dir.empty()) {
            std::string header;
            for (auto& i : cfg.includes) { std::error_code e2; if (fs::exists(fs::path(i) / "forge_aot_rt.hpp", e2)) { header = (fs::path(i) / "forge_aot_rt.hpp").string(); break; } }
            auto ver = run({cxx, "--version"}, 20000);
            std::string key_src = cxx + "\n" + (ver ? ver->out : std::string()) + "\n" + cfg.dir + "\n" + header + "\n";
            for (std::size_t k = 1; k < cmd.size(); ++k) key_src += cmd[k] + "\n";
            for (auto& i : cfg.includes) key_src += i + "\n";
            // GCC does not notice that a header behind a .gch changed. The runtime library is rebuilt whenever one
            // of them does, so its size and time stamp (and the entry header's) are part of the key.
            std::vector<std::string> stamped(cfg.libs); stamped.push_back(header);
            for (auto& f : stamped) {
                std::error_code e5;
                key_src += std::to_string(fs::file_size(f, e5)) + ":" + std::to_string(fs::last_write_time(f, e5).time_since_epoch().count()) + "\n";
            }
            const std::string key = hash::blake3::hex(hash::blake3::of(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(key_src.data()), key_src.size()))).substr(0, 24);
            const fs::path dir = fs::path(pch_dir) / key;
            const fs::path pch = dir / (clang ? "forge_aot_rt.hpp.pch" : "forge_aot_rt.hpp.gch");
            std::error_code e3;
            bool have = fs::exists(pch, e3);
            if (!have && !header.empty()) {
                fs::create_directories(dir, e3);
                const std::string part = pch.string() + ".tmp-" + std::to_string(os::pid());
                std::vector<std::string> pc(cmd);
                for (auto& i : cfg.includes) pc.push_back("-I" + i);
                pc.insert(pc.end(), {"-x", "c++-header", header, "-o", part});
                auto pr = run(pc, 30ull * 60 * 1000);
                if (pr && pr->exit_code == 0) { fs::rename(part, pch, e3); have = !e3; } else fs::remove(part, e3);
            }
            if (have) {
                if (clang) { cmd.push_back("-include-pch"); cmd.push_back(pch.string()); }
                else cmd.push_back("-I" + dir.string());   // GCC looks for <dir>/forge_aot_rt.hpp.gch before the header itself
                pch_note = ", precompiled runtime header " + pch.string();
            }
        }
        for (auto& i : cfg.includes) cmd.push_back("-I" + i);
        cmd.push_back(src);
        cmd.push_back("-o"); cmd.push_back(out);
        for (auto& l : cfg.libs) cmd.push_back(l);
        for (auto& f : cfg.ldflags) cmd.push_back(f);
        // The runtime library carries the interpreter's debug info; leave it out of the program unless asked (-Og).
        if (O != "g") {
            bool apple = o.target.find("apple") != std::string::npos || o.target.find("darwin") != std::string::npos;
#if defined(__APPLE__)
            if (o.target.empty()) apple = true;
#endif
            cmd.push_back(apple ? "-Wl,-S" : "-Wl,--strip-debug");
        }
    }
    if (!o.keep_source_dir.empty()) {
        std::error_code ec; fs::create_directories(o.keep_source_dir, ec);
        write_file((fs::path(o.keep_source_dir) / (stem + ".aot.cpp")).string(), tr.cxx);
        write_file((fs::path(o.keep_source_dir) / (stem + ".aot.cmd")).string(), join_cmd(cmd) + "\n");
    }
    std::error_code ec; fs::remove(out, ec);
    auto r = run(cmd, 30ull * 60 * 1000);
    if (r && r->exit_code != 0 && !pch_note.empty()) {
        // A stale precompiled header (the runtime's headers changed) is rejected by the compiler: drop it and
        // compile from the headers.
        std::vector<std::string> plain;
        for (std::size_t k = 0; k < cmd.size(); ++k) {
            if (cmd[k] == "-include-pch") { std::error_code e4; fs::remove(cmd[k + 1], e4); ++k; continue; }
            if (cmd[k].starts_with("-I") && fs::exists(fs::path(cmd[k].substr(2)) / "forge_aot_rt.hpp.gch", ec)) { fs::remove(fs::path(cmd[k].substr(2)) / "forge_aot_rt.hpp.gch", ec); continue; }
            plain.push_back(cmd[k]);
        }
        cmd = plain; pch_note.clear();
        r = run(cmd, 30ull * 60 * 1000);
    }
    if (!r) throw aot_error(0, "aot native: could not run " + cxx + ": " + r.err.what, 1);
    if (r->exit_code != 0 || !fs::exists(out, ec)) {
        std::string log = r->out; if (log.size() > 8000) log = log.substr(0, 8000) + "\n...";
        throw aot_error(0, "aot native: " + cxx + " failed (exit " + std::to_string(r->exit_code) + (r->timed_out ? ", timed out" : "") + ")\n" + log, 1);
    }
    aot_result res; res.mode = aot_mode::native; res.output = out; res.compiler_output = r->out;
    res.detail = "native: " + cxx + " -O" + O + (o.target.empty() ? "" : " --target=" + o.target) + ", runtime " + cfg.dir + pch_note + "; " + std::to_string(tr.unboxed) + " unboxed int local(s)";
    if (tr.delegated) { res.detail += ", " + std::to_string(tr.delegated) + " construct(s) left to the tree-walker ("; bool first = true; for (auto& k : tr.delegated_kinds) { res.detail += (first ? "" : ", ") + k; first = false; } res.detail += ")"; }
    return res;
}

} // namespace

aot_result aot_compile(std::string_view src, const std::string& file, const std::string& out, const aot_options& o) {
    if (!aot_supported()) throw aot_error(0, "aot: compiling is not supported on this platform (build the binary on a desktop host; an AOT binary's embedded program still runs here)", 2);
    if (out.empty()) throw aot_error(0, "aot: no output path", 2);
    static const std::set<std::string> levels{"0", "1", "2", "3", "s", "z", "g", "fast"};
    if (!levels.count(o.opt_level)) throw aot_error(0, "aot: unknown optimisation level -O" + o.opt_level, 2);
    std::string tried;
    runtime_cfg cfg = find_runtime(o, tried);
    if (!o.runtime_dir.empty() && !cfg.found) throw aot_error(0, "aot: no forge-aot-runtime.cfg in " + o.runtime_dir, 2);

    std::string why_not;   // why native is not possible, when it is not
    std::string cxx;
    if (o.mode != aot_mode::embed) {
        if (!cfg.found) why_not = "the AOT runtime was not found (looked in: " + (tried.empty() ? std::string("nowhere") : tried) + "; pass --aot-runtime DIR)";
        else if (cfg.libs.empty()) why_not = "the AOT runtime in " + cfg.dir + " lists no library";
        else { cxx = find_cxx(o); if (cxx.empty()) why_not = o.cxx.empty() ? "no C++ compiler found ($CXX, c++, clang++, clang++-23, g++, cl)" : "C++ compiler '" + o.cxx + "' not found"; }
        if (!why_not.empty() && o.mode == aot_mode::native) throw aot_error(0, "aot native: " + why_not, 1);
    }
    const bool try_native = o.mode != aot_mode::embed && why_not.empty();
    // Syntax and static-check errors are the script's fault in every mode: they propagate (exit 2).
    aot::translated tr = aot::translate_program(src, file, o, false);
    if (try_native) {
        try {
            aot::translated full = aot::translate_program(src, file, o, true);
            return build_native(full, out, o, cfg, cxx);
        } catch (const aot_error& e) {
            if (o.mode == aot_mode::native) throw;
            why_not = e.what();
        }
    }
    aot_result r = build_embed(tr, out, o, cfg);
    if (o.mode == aot_mode::automatic) r.fallback_reason = why_not;
    return r;
}

} // namespace forge::script

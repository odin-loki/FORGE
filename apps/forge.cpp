// forge — command-line front end: info | import | inspect | run | bench
#include <forge/import/gguf.hpp>
#include <forge/lane/dispatch.hpp>
#include <forge/mind/mind.hpp>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <chrono>
#include <forge/grid/prims.hpp>

using namespace forge;

namespace {

struct args {
    std::vector<std::string> pos;
    std::map<std::string, std::string> kv;
    std::set<std::string> flags;
    std::map<std::string, std::vector<std::string>> multi;  // repeated options (e.g. --pubkey a --pubkey b)
    std::vector<std::string> all(const std::string& k) const { auto it = multi.find(k); return it == multi.end() ? std::vector<std::string>{} : it->second; }
    std::string get(const std::string& k, const std::string& d = "") const { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
    double num(const std::string& k, double d) const { auto it = kv.find(k); return it == kv.end() ? d : std::stod(it->second); }
    bool has(const std::string& f) const { return flags.count(f) || kv.count(f); }
};

args parse(int argc, char** argv, int from, std::set<std::string> boolean) {
    args a;
    for (int i = from; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) == 0 || (s.size() == 2 && s[0] == '-')) {
            std::string k = s.substr(s[1] == '-' ? 2 : 1);
            if (auto eq = k.find('='); eq != std::string::npos) { a.kv[k.substr(0, eq)] = k.substr(eq + 1); a.multi[k.substr(0, eq)].push_back(k.substr(eq + 1)); continue; }
            if (boolean.count(k) || i + 1 >= argc) a.flags.insert(k); else { a.kv[k] = argv[++i]; a.multi[k].push_back(a.kv[k]); }
        } else a.pos.push_back(s);
    }
    return a;
}

int usage() {
    std::puts(
        "usage: forge <command> [options]\n"
        "  info                                   CPU features, lane target, grid backends\n"
        "  import <hf_dir|model.gguf> -o <out.fpk> [--codec f32|f16|bf16|fp8_e4m3|q8_0|q4_k] [--template llama]\n"
        "  inspect <model.fpk> [--verify] [--tensors] [--blueprint]\n"
        "  run <model.fpk> (--prompt TEXT | --tokens 1,2,3) [-n 32] [--temp 0] [--top-k 0] [--top-p 1]\n"
        "      [--min-p 0] [--repeat-penalty 1] [--seed 0] [--ctx N] [--ids] [--logits out.f32] [--verify]\n"
        "      [--grammar file.gbnf] [--draft draft.fpk [--spec-k 4]] [--codec C (for hf/gguf inputs)]\n"
        "      (<model> may also be a Hugging Face directory or a .gguf: imported once into ~/.cache/forge)\n"
        "  bench <model.fpk> [-p 32] [-n 32]  |  bench --gemm 512\n"
        "  dump <model.fpk> <tensor> <out.f32>       decode one tensor to raw f32\n"
        "  keygen -o <name>                       Ed25519 key pair: <name>.key (secret seed), <name>.pub\n"
        "  sign <model.fpk> --key <name.key> [-o out.fpk]   add/extend the signature section\n"
        "  verify <model.fpk> [--pubkey <name.pub>]...      check hashes + signature (exit 1 on failure)\n"
        "  (strict loading: FORGE_REQUIRE_SIGNED=1 and/or FORGE_TRUSTED_KEYS=<hex>[,<hex>])");
    return 2;
}

// `forge run <hf-dir|model.gguf>`: import on the fly into a content-addressed cache and reuse it next time.
std::string resolve_model(const std::string& path, const std::string& codec) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(path) && !path.ends_with(".gguf")) return path;
    std::string key = fs::absolute(path).string() + "|" + codec + "|" + std::to_string(int(pack::format_version));
    auto stamp = [&](const fs::path& f) { key += std::format("|{}:{}:{}", f.filename().string(), (unsigned long long)fs::file_size(f),
                                                            (long long)fs::last_write_time(f).time_since_epoch().count()); };
    if (fs::is_directory(path)) { std::vector<fs::path> fl; for (auto& e : fs::directory_iterator(path)) if (e.is_regular_file()) fl.push_back(e.path());
                                  std::sort(fl.begin(), fl.end()); for (auto& f : fl) stamp(f); }
    else stamp(path);
    const char* home = std::getenv("HOME");
    fs::path dir = std::getenv("FORGE_CACHE") ? fs::path(std::getenv("FORGE_CACHE")) : fs::path(home ? home : "/tmp") / ".cache" / "forge";
    fs::create_directories(dir);
    auto h = hash::blake3::hex(hash::blake3::of({reinterpret_cast<const std::uint8_t*>(key.data()), key.size()})).substr(0, 24);
    fs::path out = dir / (fs::path(path).filename().string() + "-" + codec + "-" + h + ".fpk");
    if (!fs::exists(out)) {
        import::import_options o; o.codec = codec;
        auto tmp = out.string() + ".partial";
        auto r = path.ends_with(".gguf") ? import::import_gguf(path, tmp, o) : import::import_hf(path, tmp, o);
        fs::rename(tmp, out);
        std::fprintf(stderr, "[imported %s (%s, %zu tensors) -> %s]\n", path.c_str(), r.template_name.c_str(), r.tensors, out.c_str());
    } else std::fprintf(stderr, "[using cached %s]\n", out.c_str());
    return out.string();
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> r; std::size_t p = 0;
    while (p < s.size()) { auto e = s.find(',', p); if (e == std::string::npos) e = s.size(); if (e > p) r.push_back(std::stoi(s.substr(p, e - p))); p = e + 1; }
    return r;
}

int cmd_info() {
    auto& f = lane::detect();
    std::printf("cpu      %s\nfeatures %s\nlane     %s\n", f.brand.c_str(), f.summary().c_str(), lane::active().target);
    for (auto& b : grid::probe_backends()) std::printf("grid     %-10s %s  %s\n", grid::name(b.kind), b.available ? "yes" : "no ", b.detail.c_str());
    return 0;
}

int cmd_import(const args& a) {
    if (a.pos.empty() || a.get("o").empty()) return usage();
    import::import_options o; o.codec = a.get("codec", "f32"); o.template_name = a.get("template");
    auto r = a.pos[0].ends_with(".gguf") ? import::import_gguf(a.pos[0], a.get("o"), o) : import::import_hf(a.pos[0], a.get("o"), o);
    std::printf("imported %s via template '%s': %zu tensors, %zu bytes -> %s\n", r.arch.c_str(), r.template_name.c_str(), r.tensors, r.bytes, a.get("o").c_str());
    return 0;
}

int cmd_inspect(const args& a) {
    if (a.pos.empty()) return usage();
    pack::reader r(a.pos[0]);
    std::printf("file      %s (%zu bytes)\nmanifest  %s\n", a.pos[0].c_str(), r.file_size(), json::dump(r.manifest()).c_str());
    for (auto& [n, s] : r.sections()) std::printf("section   %-10s offset %10llu size %10llu blake3 %s\n", n.c_str(),
                                                   (unsigned long long)s.offset, (unsigned long long)s.size, hash::blake3::hex(s.hash).substr(0, 16).c_str());
    auto bp = pack::parse_blueprint(r.section_text("blueprint"));
    std::printf("blueprint \"%s\" opset %d session %s, %zu params, %zu states, %zu tensors\n", bp.name.c_str(), bp.opset,
                bp.session.c_str(), bp.params.size(), bp.states.size(), r.tensors().size());
    if (a.has("tensors")) for (auto& [n, e] : r.tensors()) std::printf("  %-28s %-9s %s\n", n.c_str(), e.codec.c_str(), shape_str(e.shape).c_str());
    if (a.has("blueprint")) std::printf("%.*s\n", int(r.section_text("blueprint").size()), r.section_text("blueprint").data());
    if (a.has("verify")) {
        auto bad = r.verify();
        if (bad.empty()) std::puts("verify    ok (all sections and tensors match BLAKE3)");
        else { for (auto& b : bad) std::printf("verify    MISMATCH %s\n", b.c_str()); return 1; }
    }
    return 0;
}

int cmd_dump(const args& a) {  // decode one tensor to raw little-endian f32
    if (a.pos.size() < 3) return usage();
    pack::reader r(a.pos[0]);
    auto t = r.load(a.pos[1]);
    std::ofstream f(a.pos[2], std::ios::binary);
    f.write(reinterpret_cast<const char*>(t.data.data()), std::streamsize(t.size() * 4));
    std::printf("%s %s %s -> %s\n", a.pos[1].c_str(), r.entry(a.pos[1]).codec.c_str(), shape_str(t.shape).c_str(), a.pos[2].c_str());
    return 0;
}

int cmd_keygen(const args& a) {
    if (a.get("o").empty()) return usage();
    auto k = pack::generate_keypair();
    auto base = a.get("o");
    { std::ofstream f(base + ".key"); f << pack::detail::to_hex(k.seed) << "\n"; }
    std::filesystem::permissions(base + ".key", std::filesystem::perms::owner_read | std::filesystem::perms::owner_write);
    { std::ofstream f(base + ".pub"); f << pack::detail::to_hex(k.pub) << "\n"; }
    std::printf("key id %s\npublic %s\nwrote %s.key (keep secret) and %s.pub\n", pack::key_id(k.pub).c_str(), pack::detail::to_hex(k.pub).c_str(), base.c_str(), base.c_str());
    return 0;
}
int cmd_sign(const args& a) {
    if (a.pos.empty() || a.get("key").empty()) return usage();
    auto k = pack::load_secret_key(a.get("key"));
    auto out = a.get("o").empty() ? a.pos[0] : a.get("o");
    pack::sign_file(a.pos[0], out, k);
    std::printf("signed %s with key %s -> %s\n", a.pos[0].c_str(), pack::key_id(k.pub).c_str(), out.c_str());
    return 0;
}
int cmd_verify(const args& a) {
    if (a.pos.empty()) return usage();
    pack::reader r(a.pos[0]);
    auto bad = r.verify();
    for (auto& b : bad) std::printf("hash MISMATCH %s\n", b.c_str());
    std::vector<std::string> trusted;
    for (auto& p : a.all("pubkey")) trusted.push_back(pack::load_public_key(p));
    auto rep = pack::verify_signature(r, trusted);
    for (auto& k : rep.valid_keys) std::printf("signature ok   key %s\n", k.c_str());
    for (auto& k : rep.invalid_keys) std::printf("signature BAD  key %s\n", k.c_str());
    std::printf("%s: %s\n", a.pos[0].c_str(), bad.empty() && rep.trusted ? "VERIFIED" : rep.message.c_str());
    return bad.empty() && rep.trusted ? 0 : 1;
}

int cmd_run(const args& a) {
    if (a.pos.empty()) return usage();
    mind::session_options so; so.context = i64(a.num("ctx", 0)); so.verify = a.has("verify");
    mind::session s(resolve_model(a.pos[0], a.get("codec", "f32")), so);
    std::vector<int> prompt;
    if (a.kv.count("tokens")) prompt = parse_ids(a.get("tokens"));
    else if (a.kv.count("prompt")) {
        if (!s.tokenizer()) raise(errc::not_found, "model has no tokenizer; use --tokens");
        prompt = s.tokenizer()->encode(a.get("prompt"));
        if (s.tokenizer()->bos() >= 0 && !a.has("no-bos")) prompt.insert(prompt.begin(), s.tokenizer()->bos());
    } else return usage();
    mind::sampler_config sc;
    sc.temperature = float(a.num("temp", 0)); sc.top_k = int(a.num("top-k", 0)); sc.top_p = float(a.num("top-p", 1));
    sc.min_p = float(a.num("min-p", 0)); sc.repetition_penalty = float(a.num("repeat-penalty", 1)); sc.seed = std::uint64_t(a.num("seed", 0));
    mind::sampler smp(sc);
    bool ids = a.has("ids") || !s.tokenizer();
    if (a.kv.count("logits")) {  // dump prefill logits for golden comparison
        auto l = s.forward(prompt);
        std::ofstream f(a.get("logits"), std::ios::binary);
        f.write(reinterpret_cast<const char*>(l.data()), std::streamsize(l.size() * 4));
        s.reset();
    }
    std::unique_ptr<mind::grammar_constraint> gc;
    if (a.kv.count("grammar")) {
        if (!s.tokenizer()) raise(errc::unsupported, "--grammar needs a model tokenizer");
        auto g = std::make_shared<mind::grammar>(import::read_file(a.get("grammar")));
        gc = std::make_unique<mind::grammar_constraint>(g, *s.tokenizer(), int(s.vocab()), s.tokenizer()->eos());
        smp.filter = [&](std::span<float> l) { gc->apply(l); };
    }
    auto show = [&](int t) {
        if (gc && !gc->accept(t)) return false;
        if (ids) std::printf("%d ", t); else std::printf("%s", s.tokenizer()->decode_one(t).c_str());
        std::fflush(stdout);
        return !(gc && gc->finished());
    };
    if (a.kv.count("draft")) {
        mind::session d(resolve_model(a.get("draft"), a.get("codec", "f32")), so);
        mind::speculative_stats sst;
        mind::speculative_generate(s, d, prompt, int(a.num("n", 32)), smp, int(a.num("spec-k", 4)), show, &sst);
        std::printf("\n");
        std::fprintf(stderr, "[speculative: %zu tokens in %.3fs = %.1f tok/s; %zu rounds, acceptance %.0f%%]\n", sst.new_tokens, sst.seconds,
                     double(sst.new_tokens) / sst.seconds, sst.rounds, 100 * sst.acceptance());
        return 0;
    }
    mind::generate_stats st;
    std::vector<int> out = s.generate(prompt, int(a.num("n", 32)), smp, show, &st);
    std::printf("\n");
    std::fprintf(stderr, "[%zu prompt tokens in %.3fs, %zu new tokens in %.3fs = %.1f tok/s; plans %zu compiled, %zu reused]\n",
                 st.prompt_tokens, st.prefill_s, st.new_tokens, st.decode_s, st.new_tokens > 1 ? double(st.new_tokens - 1) / st.decode_s : 0.0,
                 s.plans().misses(), s.plans().hits());
    return 0;
}

int cmd_bench_gemm(int n) {
    std::vector<float> A(std::size_t(n) * n, 0.5f), B(std::size_t(n) * n, 0.25f), C(std::size_t(n) * n);
    for (const char* t : {"scalar", "avx2", "avx512", "neon"}) {
        auto* k = lane::by_name(t); if (!k) continue;
        k->gemm_nt(A.data(), B.data(), C.data(), std::size_t(n), std::size_t(n), std::size_t(n));
        auto t0 = std::chrono::steady_clock::now(); int reps = 0;
        do { k->gemm_nt(A.data(), B.data(), C.data(), std::size_t(n), std::size_t(n), std::size_t(n)); ++reps; }
        while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(500));
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("lane::%-7s gemm_nt %d^3: %8.2f GFLOP/s (1 thread)\n", t, n, 2.0 * n * n * n / s / 1e9);
    }
    auto t0 = std::chrono::steady_clock::now(); int reps = 0;
    do { grid::prims::gemm_nt(A.data(), B.data(), C.data(), std::size_t(n), std::size_t(n), std::size_t(n)); ++reps; }
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(500));
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
    std::printf("grid::prims gemm_nt %d^3: %8.2f GFLOP/s (all threads)\n", n, 2.0 * n * n * n / s / 1e9);
    return 0;
}

int cmd_bench(const args& a) {
    if (a.kv.count("gemm")) return cmd_bench_gemm(int(a.num("gemm", 512)));
    if (a.pos.empty()) return usage();
    mind::session s(resolve_model(a.pos[0], a.get("codec", "f32")));
    int p = int(a.num("p", 32)), n = int(a.num("n", 32));
    std::vector<int> prompt(std::size_t(p), 1);
    mind::sampler smp;
    mind::generate_stats st;
    s.generate(prompt, n, smp, {}, &st);
    std::printf("prefill %d tokens: %.1f tok/s\ndecode  %zu tokens: %.1f tok/s\n", p, p / st.prefill_s, st.new_tokens,
                st.new_tokens > 1 ? double(st.new_tokens - 1) / st.decode_s : 0.0);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    std::string c = argv[1];
    try {
        auto a = parse(argc, argv, 2, {"verify", "tensors", "blueprint", "ids", "no-bos"});
        if (c == "info") return cmd_info();
        if (c == "import") return cmd_import(a);
        if (c == "inspect") return cmd_inspect(a);
        if (c == "run") return cmd_run(a);
        if (c == "bench") return cmd_bench(a);
        if (c == "dump") return cmd_dump(a);
        if (c == "keygen") return cmd_keygen(a);
        if (c == "sign") return cmd_sign(a);
        if (c == "verify") return cmd_verify(a);
        return usage();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "forge: error: %s\n", e.what());
        return 1;
    }
}

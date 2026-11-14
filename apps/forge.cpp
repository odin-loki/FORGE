#include <iterator>
#include <iostream>
#include <forge/os/process.hpp>
// forge — command-line front end: info | import | inspect | run | bench
#include <sys/stat.h>
#include <forge/import/gguf.hpp>
#include <forge/import/onnx.hpp>
#include <forge/import/requant.hpp>
#include <forge/import/blueprint_dir.hpp>
#include <forge/modelfile/shard.hpp>
#include <forge/simd/dispatch.hpp>
#include <forge/inference/inference.hpp>
#include <forge/inference/aot_build.hpp>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <mutex>
#include <chrono>
#include <random>
#include <forge/simd/qdot_i8.hpp>
#include <forge/gpu/prims.hpp>
#include <forge/gpu/cuda.hpp>
#include <forge/script/script.hpp>
#include <forge/script/check.hpp>
#include <forge/script/local_ai.hpp>
#include <forge/script/aot.hpp>
#include <forge/security/security.hpp>
#include <forge/compress/cli.hpp>

// ASan and TSan reserve terabytes of address space for their shadow when a process starts, so an address-space cap
// (RLIMIT_AS) put on a child that is this same sanitized binary stops it before main(): `forge rpc-serve` scripts
// all failed with "could not start script child" in the sanitizer CI jobs. Such builds leave that one cap to the
// child itself (limit_own_memory() counts what is mapped already).
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define FORGE_APP_SANITIZED 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define FORGE_APP_SANITIZED 1
#endif
#endif
#ifndef FORGE_APP_SANITIZED
#define FORGE_APP_SANITIZED 0
#endif

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
        "  info                                   CPU features, lane target, grid backends, memory budget\n"
        "  import <dir with blueprint.fbp + *.safetensors> -o <out.fpk>   a model described by its own blueprint (any session kind)\n"
        "  import <hf_dir|model.gguf> -o <out.fpk> [--codec f32|f16|bf16|fp8_e4m3|fp8_e5m2|q8_0|q4_k|q4_g32|q8_g32|mxfp8|mxfp6|mxfp4|mxint8|nvfp4|tq2] [--template llama] [--emb-codec q8_0|q4_k|f32]\n"
        "      more codecs (GGUF block layouts): q5_k q6_k q4_0 q4_1 q5_0 q5_1 q2_k q3_k iq4_nl iq4_xs tq1_0 tq2_0\n"
        "      --codec keep (default for .gguf and AWQ/GPTQ checkpoints): tensors stay in their stored quantisation; another codec re-quantises them\n"
        "      mixed precision (hf inputs): --codec q4_k_m|q5_k_m [--recipe forge|llama] [--output-tensor-type T] [--token-embedding-type T]\n"
        "      [--tensor-type REGEX=TYPE[,..]] [--pure] [--bf16-head]   (llama.cpp llama-quant.cpp rules; forge adds q6_k DeltaNet/attn_output, q8_0 head)\n"
        "      [--allow-unused] import although source tensors are left over (default: error listing them)\n"
        "      [--ignore-tensor NAME[,NAME..]] leftovers known to be harmless (NAME, prefix* or *suffix)\n"
        "      [--align 64|4096|65536] file alignment of every tensor (default: 4096 when the source weights are >= 64 MiB, else 64)\n"
        "      [--calib] record statistics of the full-precision weights (\"calib\" section) for a later forge requant\n"
        "      [--shard-size SIZE [--blob-dir DIR]] write a sharded model: tensors in content-addressed blobs (0 = one per tensor)\n"
        "      (default for float Hugging Face checkpoints: optimised while imported - rotation, mixed-precision GPTQ checked\n"
        "      against the model's own outputs, output calibration; --no-optimise / FORGE_OPTIMISE=0 / --codec f32|f16|bf16 import as stored)\n"
        "      [--compress RECIPE --calib-data FILE [--nsamples 64] [--seqlen 512] [--report R.json]] compress while importing: the\n"
        "      checkpoint is imported in f32, compressed (see compress below) and written with --codec\n"
        "  compress <model.fpk|hf_dir> -o <out.fpk> --recipe RECIPE --calib-data FILE [--nsamples 64] [--seqlen 512] [--seed 0]\n"
        "      [--codec C] [--report R.json] [--data DIR ...eval options]   prune / sparsify / shrink / convert a model (docs/compress.md)\n"
        "      RECIPE: stage(k=v,..)+stage(..); stages prune, width, depth, experts, merge, moe2dense\n"
        "  eval <model.fpk> --data DIR [--ppl-windows 40] [--ppl-window 512] [--tasks arc_easy,hellaswag,..] [--limit 200] [--report R.json]\n"
        "      WikiText-2 perplexity and zero-shot accuracy on FORGE's inference engine (data: tools/compress_data.py)\n"
        "  shard split <in.fpk> -o <out.fpk> [--shard-size 2G] [--blob-dir DIR]   tensors into blobs named by their BLAKE3 hash\n"
        "      (default one blob per tensor: models that share tensors store them once); blobs go to DIR, $FORGE_BLOB_DIR or ./blobs next to out\n"
        "  shard join <in.fpk> -o <out.fpk>  |  shard list <model.fpk>             back to one file; which blobs are present\n"
        "  requant <in.fpk> -o <out.fpk> --codec C   re-encode the float weights of a model file (codecs and the mixed-precision\n"
        "      options of import); reads the calib section, or measures f32 weights [--no-calib] [--calib-crest 12] [--align A]\n"
        "  adapter list <model.fpk> | add <model.fpk> --name N --from lora.fpk [-o out.fpk] | remove <model.fpk> --name N [-o out.fpk]\n"
        "      LoRA adapters carried in the model file; select one with run --adapter N\n"
        "  blueprint <model.fpk> --to binary|text [-o out.fpk]     store the model's blueprint in the binary or the text form\n"
        "  blueprint <file.fbp|file.fbpb> --to binary|text|canonical -o <out>   convert a bare blueprint file\n"
        "  inspect <model.fpk> [--verify] [--tensors] [--blueprint [text|binary|canonical] [-o FILE]] [--license] [--state]\n"
        "  run <model.fpk> (--prompt TEXT | --tokens 1,2,3) [-n 32] [--temp 0] [--top-k 0] [--top-p 1]\n"
        "      [--min-p 0] [--repeat-penalty 1] [--seed 0] [--ctx N] [--ids] [--logits out.f32] [--verify]\n"
        "      [--grammar file.gbnf] [--draft draft.fpk | --lookup [--ngram 3] [--ngram-min 1] | --self-spec [--exit-layer K]] [--spec-k 0 (0 = planner)]\n"
        "      [--codec C (for hf/gguf inputs)] [--allow-unused (for hf/gguf inputs)]\n"
        "      (<model> may also be a Hugging Face directory or a .gguf: imported once into ~/.cache/forge)\n"
        "      one-shot graphs (imported ONNX): --input name=file.f32 ... [--dim name=N ...] [--out prefix]; integer inputs take file.i64 / file.i32\n"
        "  embed <model.fpk> (--text TEXT [--text-pair TEXT] [--ids] | --tokens 1,2,3 [--token-types ..] [--mask 1,1,0] | --input name=file.f32 ...)\n"
        "      [--out file.f32] [--device D]     embedding models (BERT family, ViT / CLIP / SigLIP, Whisper encoder):\n"
        "      prints the vector, or writes it as raw f32; --input feeds a declared input (an image, mel frames)\n"
        "      --text: the model's tokenizer adds its special tokens ([CLS] .. [SEP]) and token types; --ids prints the ids to stderr\n"
        "  denoise <model.fpk> [--steps N] [--sampler ddim|ddpm|euler] [--guidance G] [--eta E] [--seed S | --init latent.f32]\n"
        "      [--input name=file ...] [--uncond name=file ...] [--dim name=N] [--out latent.f32]   models with `session diffusion_loop`\n"
        "  stream <model.fpk> --input samples.f32 [--chunk N] [--out frames.f32]    models with `session streaming`: framing from the file,\n"
        "      one run per frame with the model's state carried over\n"
        "  aot <model.fpk> [-o out] [--aot-mode plan|native]   compile the model's plans ahead of time (docs/aot-models.md)\n"
        "      [--shapes 1,8,16 (sequence lengths; default 1,2,4,...,64)] [--all-rows] [--kv-codec f32,int8] [--ctx N]\n"
        "      plan:   plans go into the .fpk's \"aot\" section (-o out.fpk, default in place) [--sidecar] [--key name.key | --drop-signature]\n"
        "              [--strip] remove them  [--list] show them\n"
        "      native: -o model.so|model.a|model [--emit shared|static|exe] [--embed-weights] [--cxx C] [--target TRIPLE] [--simd T]\n"
        "              [--march M|none] [--cxxflag F]... [--sdk forge_aot_sdk.txt] [--work DIR]; `forge run model.so` loads the result\n"
        "  bench <model.fpk> [-p 32] [-n 32]  |  bench --gemm 512  |  bench --kernels 1024 (MB, quantised matvec GB/s)\n"
        "  dump <model.fpk> <tensor> <out.f32>       decode one tensor to raw f32\n"
        "  keygen -o <name>                       Ed25519 key pair: <name>.key (secret seed), <name>.pub\n"
        "  sign <model.fpk> --key <name.key> [-o out.fpk]   add/extend the signature section\n"
        "  sign <file> --key <name.key> [-o file.sig] [--name N] [--detached]   any other file (plugin, script):\n"
        "      detached signature <file>.sig (Ed25519 over BLAKE3, binds the file name)\n"
        "  verify <model.fpk> [--pubkey <name.pub>]...      check hashes + signature (exit 1 on failure)\n"
        "  verify <file> [--sig file.sig] [--pubkey <name.pub>]...   check a detached signature (exit 1 on failure)\n"
        "  encrypt <in.fpk> -o <out.fpk> --key k.mkey | --new-key k.mkey   seal tensor data (XChaCha20-Poly1305);\n"
        "      load with FORGE_MODEL_KEY_FILE=k.mkey (or FORGE_MODEL_KEY=<hex>)\n"
        "      [--chunk 1M] sealed chunk size: tensors are decrypted one at a time when first used (FORGE_LOCK_WEIGHTS=1: into locked memory)\n"
        "      [--single-aead] the older scheme (one AEAD over all tensor data, decrypted whole at load)\n"
        "  script <file.cpp> [args...]            run a FORGE script (interpreted, memory-safe C++23 subset)\n"
        "  script -e '<code>' [args...]          run a snippet;  script (no file): interactive REPL\n"
        "      [--tree] tree-walker only  [--debug] race/deadlock checks  [--timeout MS]  [--max-steps N]  [--heap BYTES]  [--no-proc] [--no-fs] [--no-net]\n"
        "      [--mem-limit SIZE]  hard memory cap (\"512M\", \"2G\"): managed heap, child processes, and this process at OS level\n"
        "      [--policy F]  FSS policy: capabilities from the policy, lockdown() before the script runs\n"
        "  script --aot <out> <file.cpp> [--aot-mode native|embed] [--cxx C] [-O<n>] [--target T] [--aot-runtime DIR]\n"
        "      [--aot-keep-source DIR] [--aot-allow-widen]  compile a script to a native executable (docs/aot.md)\n"
        "  security check [--policy F] [--json]   FSS controls: state, backend, verdict (exit 1 if a required one cannot be met)\n"
        "  run, bench, script and rpc-serve take --policy F (else $FORGE_SECURITY_POLICY, ./forge-security.json, the config\n"
        "      dir, $FORGE_SECURITY_PROFILE; with none of them no FSS profile is active and nothing changes; docs/security-api.md)\n"
        "  (strict loading: FORGE_REQUIRE_SIGNED=1 and/or FORGE_TRUSTED_KEYS=<hex>[,<hex>])");
    return 2;
}

// FORGE runs optimised models: a plain (float) Hugging Face checkpoint is optimised while it is imported - the
// data-free forge::compress pipeline `optimise` (rotation / equalisation, mixed-precision GPTQ checked against the
// model's own outputs, output calibration; docs/compress.md). Off with --no-optimise, FORGE_OPTIMISE=0, or an
// explicit lossless --codec (f32, f16, bf16, keep); --codec C optimises into C instead of a mixed allocation.
bool auto_optimise(const std::string& path, const std::string& codec_opt, bool no_opt) {
    namespace fs = std::filesystem;
    if (no_opt) return false;
    if (const char* e = std::getenv("FORGE_OPTIMISE"); e && std::string(e) == "0") return false;
    if (!fs::is_directory(path) || import::is_blueprint_dir(path) || !fs::exists(fs::path(path) / "config.json")) return false;
    if (import::read_file(fs::path(path) / "config.json").find("\"quantization_config\"") != std::string::npos) return false;   // AWQ / GPTQ: kept
    return !(codec_opt == "f32" || codec_opt == "f16" || codec_opt == "bf16" || codec_opt == "keep");
}
std::string optimise_recipe(const std::string& codec_opt) { return "optimise(codec=" + (codec_opt.empty() ? std::string("mixed") : codec_opt) + ")"; }

// `forge run <hf-dir|model.gguf>`: import on the fly into a content-addressed cache and reuse it next time.
std::string resolve_model(const std::string& path, const std::string& codec_opt, bool allow_unused = false, bool no_opt = false) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(path) && !path.ends_with(".gguf") && !path.ends_with(".onnx")) return path;
    const bool optimise = auto_optimise(path, codec_opt, no_opt);
    // `codec_opt` is --codec as given (empty when absent): Hugging Face inputs default to f32, a GGUF keeps its stored
    // types unless a codec is asked for (then it is re-quantised, and cached under a separate key).
    const bool quantised_src = path.ends_with(".gguf") || (fs::is_directory(path) && fs::exists(fs::path(path) / "config.json") &&
                               import::read_file(fs::path(path) / "config.json").find("\"quantization_config\"") != std::string::npos);
    const bool requant = quantised_src && !codec_opt.empty() && codec_opt != "keep";
    const std::string codec = optimise ? "optimised-" + (codec_opt.empty() ? std::string("mixed") : codec_opt) : codec_opt.empty() || codec_opt == "keep" ? "f32" : codec_opt;
    std::string key = fs::absolute(path).string() + "|" + codec + "|" + std::to_string(int(modelfile::format_version)) + (requant ? "|requant" : "");
    if (allow_unused) key += "|allow-unused";
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
        import::import_options o; o.codec = optimise ? std::string("f32") : codec; if (requant) o.requant = codec;
        o.allow_unused = allow_unused;
        auto tmp = out.string() + ".partial";
        auto r = import::is_blueprint_dir(path) ? import::import_blueprint_dir(path, tmp, o)
               : path.ends_with(".gguf") ? import::import_gguf(path, tmp, o) : path.ends_with(".onnx") ? import::import_onnx(path, tmp, o) : import::import_hf(path, tmp, o);
        if (optimise) {
            std::fprintf(stderr, "[optimising %s: %s]\n", path.c_str(), optimise_recipe(codec_opt).c_str());
            compress::cli::opts co; co.in = tmp; co.out = out.string() + ".opt.partial"; co.kv["recipe"] = optimise_recipe(codec_opt);
            try { compress::cli::compress_file(co); } catch (...) { fs::remove(tmp); throw; }
            fs::remove(tmp);
            tmp = co.out;
        }
        fs::rename(tmp, out);
        std::fprintf(stderr, "[imported %s (%s, %zu tensors) -> %s]\n", path.c_str(), r.template_name.c_str(), r.tensors, out.c_str());
        for (auto& n : r.notes) std::fprintf(stderr, "[import] %s\n", n.c_str());
    } else std::fprintf(stderr, "[using cached %s]\n", out.c_str());
    return out.string();
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> r; std::size_t p = 0;
    while (p < s.size()) { auto e = s.find(',', p); if (e == std::string::npos) e = s.size(); if (e > p) r.push_back(std::stoi(s.substr(p, e - p))); p = e + 1; }
    return r;
}

// `forge security check [--policy F] [--json]`: what the FORGE Security Standard controls would be under the policy
// (load_policy order: --policy, $FORGE_SECURITY_POLICY, ./forge-security.json, the user config dir). Exit 0 when
// every required control can be met, 1 when one cannot, 2 for usage errors or a bad policy.
int cmd_security(const args& a) {
    if (a.pos.empty() || a.pos[0] != "check") return usage();
    security::policy p;
    try { p = security::load_policy(a.get("policy")); }
    catch (const security::policy_violation& e) { std::fprintf(stderr, "forge security: %s\n", e.what()); return 2; }
    auto r = security::check(p);
    std::fputs(a.has("json") ? (r.json() + "\n").c_str() : r.text().c_str(), stdout);
    return r.ok ? 0 : 1;
}

// The FSS policy for a command (--policy F, else the load_policy() search). An active one becomes the process's policy
// for enforce() (model loads check fs.read and integrity.models through it); profile none changes nothing.
security::policy activate_policy(const std::string& cli_path) {
    auto p = security::load_policy(cli_path);
    if (p.active()) security::set_active_policy(p);
    return p;
}

// SBX-1 for a server: the listen address must be allowed (net_listen), then lockdown() once the sockets are bound and
// everything is loaded, before the first request. Audited; throws permission_denied / sandbox_error.
void server_lockdown(const security::policy& p, const char* who, const std::string& listen) {
    if (!p.active()) return;
    auto r = security::lockdown(p);
    for (auto& c : r.controls)
        if (c.verdict != "ok") std::fprintf(stderr, "%s: security %s %s: %s%s\n", who, c.id.c_str(), c.name.c_str(), c.state.c_str(), c.detail.empty() ? "" : (" (" + c.detail + ")").c_str());
    security::audit("sandbox", security::decision::allow, "SBX-1", listen, std::string(who) + ": locked down (profile " + security::name(p.prof) + ", " + p.source + ") before serving");
}

int cmd_info() {
    auto& f = simd::detect();
    std::printf("cpu      %s\nfeatures %s\nlane     %s\n", f.brand.c_str(), f.summary().c_str(), simd::active().target);
    std::string usable;   // kernel tables this CPU can run, preferred first (FORGE_SIMD_TARGET picks one)
    for (const char* t : simd::target_names()) if (simd::by_name(t)) { usable += usable.empty() ? "" : " "; usable += t; }
    std::printf("targets  %s\n", usable.c_str());
    for (auto& b : gpu::probe_backends()) std::printf("grid     %-10s %s  %s\n", gpu::name(b.kind), b.available ? "yes" : "no ", b.detail.c_str());
    std::fputs(mem::governor::instance().describe().c_str(), stdout);
    return 0;
}

std::uint64_t parse_size_arg(const std::string& v, const char* what);
int cmd_import(const args& a) {
    if (a.pos.empty() || a.get("o").empty()) return usage();
    import::import_options o; o.codec = a.get("codec", "f32"); o.template_name = a.get("template"); o.emb_codec = a.get("emb-codec");
    o.requant = a.get("codec", "keep");   // GGUF inputs: stored types unless a codec is asked for
    if (o.codec == "keep") o.codec = "f32";   // keep: GGUF types / AWQ and GPTQ codes as stored; plain checkpoints are f32
    o.recipe.recipe = a.get("recipe", "forge");
    o.recipe.output_type = a.get("output-tensor-type");
    o.recipe.token_embedding_type = a.get("token-embedding-type");
    o.recipe.pure = a.has("pure");
    o.recipe.head_bf16 = a.has("bf16-head");
    for (std::string tt = a.get("tensor-type"); !tt.empty();) {   // REGEX=TYPE[,REGEX=TYPE...]
        auto c = tt.find(','); o.recipe.tensor_types.push_back(import::parse_tensor_type(tt.substr(0, c)));
        tt = c == std::string::npos ? "" : tt.substr(c + 1);
    }
    o.allow_unused = a.has("allow-unused");
    for (std::string it = a.get("ignore-tensor"); !it.empty();) {   // NAME[,NAME...]
        auto c = it.find(','); o.ignore_tensors.push_back(it.substr(0, c));
        it = c == std::string::npos ? "" : it.substr(c + 1);
    }
    // --align 64|4096|65536 (4k, 64k): tensor alignment of the output; default by size (import::import_alignment)
    if (!a.get("align").empty()) o.align = std::size_t(parse_size_arg(a.get("align"), "--align"));
    o.calib = a.has("calib");   // record weight statistics for a later `forge requant` ("calib" section)
    activate_policy(a.get("policy"));   // FSS: importers check fs_read (input) / fs_write (output) under an active policy
    // --shard-size SIZE (0 = one blob per tensor) [--blob-dir DIR]: write a sharded model. The import goes to a
    // single file next to the output first, which is then split (modelfile/shard.hpp) and removed.
    const bool shard = a.kv.count("shard-size") != 0;
    const bool optimise = a.get("compress").empty() && !shard && auto_optimise(a.pos[0], a.get("codec"), a.has("no-optimise"));
    const bool squeeze = !a.get("compress").empty() || optimise;   // --compress RECIPE (or the default optimisation): import in f32, compress, encode
    const std::string out_codec = o.codec;
    if (squeeze) { o.codec = "f32"; o.emb_codec = "f32"; }
    const std::string final_out = a.get("o"), import_out = shard ? final_out + ".unsharded" : squeeze ? final_out + ".f32.tmp" : final_out;
    // a directory with a blueprint.fbp is a model described by its blueprint (import/blueprint_dir.hpp)
    auto r = import::is_blueprint_dir(a.pos[0]) ? import::import_blueprint_dir(a.pos[0], import_out, o)
           : a.pos[0].ends_with(".gguf") ? import::import_gguf(a.pos[0], import_out, o) : a.pos[0].ends_with(".onnx") ? import::import_onnx(a.pos[0], import_out, o) : import::import_hf(a.pos[0], import_out, o);
    if (shard) {
        modelfile::shard_options so; so.shard_bytes = parse_size_arg(a.get("shard-size"), "--shard-size"); so.blob_dir = a.get("blob-dir");
        modelfile::shard_report sr;
        try { sr = modelfile::shard_split(import_out, final_out, so); } catch (...) { os::remove(import_out); throw; }
        os::remove(import_out);
        r.bytes = std::filesystem::file_size(final_out);
        std::printf("sharded   %zu tensors in %zu blobs under %s (%zu written, %zu already there: %llu bytes not stored again)\n", sr.tensors, sr.shards, sr.blob_dir.c_str(), sr.written, sr.reused, (unsigned long long)sr.bytes_reused);
    }
    if (squeeze) {
        compress::cli::opts co; co.in = import_out; co.out = final_out; co.kv = a.kv;
        co.kv["recipe"] = optimise ? optimise_recipe(a.get("codec")) : a.get("compress"); co.kv["codec"] = out_codec;
        if (optimise) std::fprintf(stderr, "forge import: optimising (%s; --no-optimise to import as stored)\n", co.kv["recipe"].c_str());
        json::value rep;
        try { rep = compress::cli::compress_file(co); } catch (...) { os::remove(import_out); throw; }
        os::remove(import_out);
        compress::cli::write_json(a.get("report", final_out + ".compress.json"), rep);
        r.bytes = std::filesystem::file_size(final_out);
    }
    std::printf("imported %s via template '%s': %zu tensors, %zu bytes -> %s\n", r.arch.c_str(), r.template_name.c_str(), r.tensors, r.bytes, a.get("o").c_str());
    for (auto& n : r.notes) std::fprintf(stderr, "forge import: %s\n", n.c_str());
    std::printf("license   %s\n", r.license.empty() ? "(none found in the source)" : r.license.c_str());
    if (!r.unused.empty()) {   // only under --allow-unused: without it the import has already failed
        std::fprintf(stderr, "warning: %zu source tensor(s) not used by the template:\n", r.unused.size());
        for (auto& u : r.unused) std::fprintf(stderr, "  %s\n", u.c_str());
    }
    return 0;
}

int cmd_inspect(const args& a) {
    if (a.pos.empty()) return usage();
    modelfile::reader r(a.pos[0]);
    std::printf("file      %s (%zu bytes)\nmanifest  %s\n", a.pos[0].c_str(), r.file_size(), json::dump(r.manifest()).c_str());
    for (auto& [n, s] : r.sections()) std::printf("section   %-10s offset %10llu size %10llu blake3 %s\n", n.c_str(),
                                                   (unsigned long long)s.offset, (unsigned long long)s.size, hash::blake3::hex(s.hash).substr(0, 16).c_str());
    auto bp = modelfile::load_blueprint(r);
    std::printf("blueprint \"%s\" opset %d session %s, %zu params, %zu states, %zu tensors%s\n", bp.name.c_str(), bp.opset,
                bp.session.c_str(), bp.params.size(), bp.states.size(), r.tensors().size(),
                modelfile::is_binary_blueprint(r.section_text("blueprint")) ? " (stored in the binary form)" : "");
    if (bp.opset < modelfile::opset_min || bp.opset > modelfile::opset_max)
        std::printf("          opset %d is outside this engine's range %d..%d: the model will not run here\n", bp.opset, modelfile::opset_min, modelfile::opset_max);
    {   // the model's licence as recorded at import (manifest fields + the optional `license` section)
        const auto& m = r.manifest();
        std::string lic = m.str_or("license", ""), ln = m.str_or("license_name", ""), ll = m.str_or("license_link", "");
        std::string line = !lic.empty() ? lic : !ln.empty() ? ln : std::string();
        if (!lic.empty() && !ln.empty()) line += " (" + ln + ")";
        if (!ll.empty()) line += (line.empty() ? "" : " ") + ("<" + ll + ">");
        if (auto* lf = m.find("license_files"); lf && lf->is_arr()) for (auto& f : lf->arr()) if (f.is_str()) line += (line.empty() ? "" : " ") + ("[" + f.str() + "]");
        if (r.has_section("license")) line += std::format("{}text: {} bytes (--license prints it)", line.empty() ? "" : ", ", r.section_text("license").size());
        std::printf("license   %s\n", line.empty() ? "(not recorded)" : line.c_str());
    }
    if (r.sharded()) {   // where the tensor bytes are, and how much of them is here
        auto miss = r.missing_shards(); std::string dirs; for (auto& d : r.blob_dirs()) dirs += (dirs.empty() ? "" : ", ") + d;
        std::printf("shards    %zu content-addressed blobs, %llu bytes; %zu present, %zu missing; searched: %s\n", r.shards().size(), (unsigned long long)r.data_size(),
                    r.shards().size() - miss.size(), miss.size(), dirs.c_str());
    }
    {   // tensor alignment: the recorded value and what the offsets actually are
        std::size_t al = r.tensor_alignment(); auto off = r.misaligned_tensors();
        bool recorded = r.manifest().find("tensor_alignment") != nullptr;
        std::printf("alignment %zu bytes%s: %zu of %zu tensors start on a multiple of it in the file%s\n", al, recorded ? "" : " (not recorded: file written before the option)",
                    r.tensors().size() - off.size(), r.tensors().size(), off.empty() ? "" : " (file rewritten by a tool that does not keep the alignment?)");
    }
    {   // optional sections: declared state, embedded adapters, calibration statistics
        if (r.has_section(modelfile::state_section)) {
            auto sc = modelfile::state_schema::parse(r.section_text(modelfile::state_section));
            std::size_t rec = 0; for (auto& sl : sc.slots) rec += sl.recurrent;
            std::printf("state     %zu slots declared (%zu appended rows, %zu recurrent)%s\n", sc.slots.size(), sc.slots.size() - rec, rec, a.has("state") ? ":" : " (--state lists them)");
            if (a.has("state")) for (auto& sl : sc.slots) std::printf("  %-28s %-9s %-12s %s %s\n", sl.name.c_str(), sl.recurrent ? "recurrent" : "append", sl.op.c_str(), sl.dtype.c_str(), shape_str(sl.shape).c_str());
        } else if (!bp.states.empty()) std::printf("state     not declared in the file (derived from the blueprint when needed)\n");
        for (auto& ad : modelfile::read_adapters(r))
            std::printf("adapter   %-16s rank %lld alpha %g, %zu target weights (forge run --adapter %s)\n", ad.name.c_str(), (long long)ad.rank, ad.alpha, ad.targets.size(), ad.name.c_str());
        if (r.has_section(modelfile::calib_section)) {
            auto ct = modelfile::calib_table::parse(r.section_text(modelfile::calib_section));
            double worst = 0; std::string wn;
            for (auto& [n, st] : ct.tensors) if (st.crest() > worst) { worst = st.crest(); wn = n; }
            std::printf("calib     statistics of %zu tensors (source: %s); largest crest factor %.1f (%s)\n", ct.tensors.size(), ct.source.c_str(), worst, wn.c_str());
        }
    }
    if (r.has_section("encryption")) {   // which sealing scheme, and whether this process holds the key
        json::value e; try { e = json::parse(r.section_text("encryption")); } catch (const std::exception&) {}
        auto alg = e.str_or("alg", "?");
        std::string how = alg == "xchacha20-poly1305-chunked" ? std::format("{} chunks of {} bytes, decrypted per tensor on use", (unsigned long long)e.num_or("chunks", 0), (unsigned long long)e.num_or("chunk_bytes", 0))
                        : alg == "xchacha20-poly1305" ? std::string("one AEAD over the data section, decrypted whole at load") : std::string("unknown scheme");
        std::printf("encrypted %s, key id %s (%s); %s\n", alg.c_str(), e.str_or("key_id", "?").c_str(), how.c_str(), r.locked() ? "no key given: tensors unreadable" : "key accepted");
    }
    if (a.has("tensors")) for (auto& [n, e] : r.tensors()) std::printf("  %-28s %-9s %s\n", n.c_str(), e.codec.c_str(), shape_str(e.shape).c_str());
    if (a.has("blueprint")) {
        // --blueprint [text|binary|canonical] [-o FILE]: the stored text (or, for a binary section, its canonical text);
        // the binary form (a summary on the terminal, the bytes with -o); the canonical text of either.
        std::string form = a.get("blueprint");
        if (form.empty() && a.pos.size() > 1) form = a.pos[1];
        if (form.empty()) form = "text";
        auto stored = r.section_text("blueprint");
        if (form == "binary") {
            std::vector<std::uint8_t> bin = modelfile::is_binary_blueprint(stored) ? std::vector<std::uint8_t>(stored.begin(), stored.end()) : modelfile::encode_blueprint(stored);
            modelfile::binary_blueprint_info bi; (void)modelfile::decode_blueprint(bin, &bi);
            std::string ops; for (auto& c : bi.calls) ops += (ops.empty() ? "" : " ") + c;
            std::printf("binary blueprint: version %u, %zu bytes (text form: %zu bytes), opset %u, \"%s\", session %s, %zu strings, %zu tokens\ncalls: %s\n",
                        bi.version, bin.size(), modelfile::blueprint_text(stored).size(), bi.opset, bi.name.c_str(), bi.session.c_str(), bi.strings, bi.tokens, ops.c_str());
            if (!a.get("o").empty()) { std::ofstream f(a.get("o"), std::ios::binary); f.write(reinterpret_cast<const char*>(bin.data()), std::streamsize(bin.size())); if (!f) throw std::runtime_error("cannot write " + a.get("o")); }
        } else if (form == "text" || form == "canonical") {
            std::string text = form == "canonical" ? modelfile::canonical_blueprint_text(modelfile::blueprint_text(stored)) : modelfile::blueprint_text(stored);
            if (!a.get("o").empty()) { std::ofstream f(a.get("o"), std::ios::binary); f << text; if (!f) throw std::runtime_error("cannot write " + a.get("o")); }
            else std::printf("%s\n", text.c_str());
        } else throw std::runtime_error("--blueprint takes text, binary or canonical, not '" + form + "'");
    }
    if (a.has("license") && r.has_section("license")) std::printf("%.*s\n", int(r.section_text("license").size()), r.section_text("license").data());
    if (a.has("verify")) {
        auto bad = r.verify();
        if (bad.empty()) std::puts("verify    ok (all sections and tensors match BLAKE3)");
        else { for (auto& b : bad) std::printf("verify    MISMATCH %s\n", b.c_str()); return 1; }
    }
    return 0;
}

// forge adapter list <model.fpk> | add <model.fpk> --name N --from lora.fpk [-o out.fpk] | remove <model.fpk> --name N [-o out.fpk]
int cmd_adapter(const args& a) {
    if (a.pos.size() < 2) return usage();
    const std::string& verb = a.pos[0]; const std::string& f = a.pos[1];
    auto out = a.get("o").empty() ? f : a.get("o");
    if (verb == "list") {
        modelfile::reader r(f); auto all = modelfile::read_adapters(r);
        if (all.empty()) std::puts("no adapters in this file");
        for (auto& ad : all) {
            std::printf("%-16s rank %lld alpha %g\n", ad.name.c_str(), (long long)ad.rank, ad.alpha);
            for (auto& w : ad.targets) std::printf("  %s\n", w.c_str());
        }
        return 0;
    }
    activate_policy(a.get("policy"));
    if (verb == "add") {
        if (a.get("name").empty() || a.get("from").empty()) return usage();
        import::check_write(out);
        auto ad = modelfile::embed_adapter(f, out, a.get("name"), a.get("from"));
        std::printf("added adapter '%s' (rank %lld, alpha %g, %zu target weights) -> %s\n", ad.name.c_str(), (long long)ad.rank, ad.alpha, ad.targets.size(), out.c_str());
        std::puts("note: a signature and precompiled plans do not survive this; sign / run forge aot again");
        return 0;
    }
    if (verb == "remove") {
        if (a.get("name").empty()) return usage();
        import::check_write(out);
        modelfile::remove_adapter(f, out, a.get("name"));
        std::printf("removed adapter '%s' -> %s\n", a.get("name").c_str(), out.c_str());
        return 0;
    }
    return usage();
}

// forge shard split <in.fpk> -o <out.fpk> [--shard-size SIZE] [--blob-dir DIR] | join <in.fpk> -o <out.fpk> | list <model.fpk>
int cmd_shard(const args& a) {
    if (a.pos.size() < 2) return usage();
    const std::string& verb = a.pos[0]; const std::string& f = a.pos[1];
    if (verb == "list") {
        modelfile::reader r(f);
        if (!r.sharded()) { std::puts("not a sharded file (its tensors are in the data section)"); return 0; }
        std::map<int, std::size_t> count; for (auto& [n, e] : r.tensors()) ++count[e.shard];
        for (std::size_t i = 0; i < r.shards().size(); ++i) {
            auto p = r.find_shard(i);
            std::printf("%s%s  %12llu bytes  %4zu tensors  %s\n", r.shards()[i].blake3.c_str(), std::string(modelfile::blob_ext).c_str(), (unsigned long long)r.shards()[i].bytes, count[int(i)], p.empty() ? "MISSING" : p.c_str());
        }
        return 0;
    }
    if (a.get("o").empty()) return usage();
    activate_policy(a.get("policy"));
    import::check_write(a.get("o"));
    modelfile::shard_report sr;
    if (verb == "split") {
        modelfile::shard_options so; so.blob_dir = a.get("blob-dir");
        if (!a.get("shard-size").empty()) so.shard_bytes = parse_size_arg(a.get("shard-size"), "--shard-size");
        sr = modelfile::shard_split(f, a.get("o"), so);
        std::printf("split %s -> %s: %zu tensors in %zu blobs under %s\n  %zu blobs written (%llu bytes), %zu already in the store (%llu bytes not stored again)\n", f.c_str(), a.get("o").c_str(), sr.tensors, sr.shards,
                    sr.blob_dir.c_str(), sr.written, (unsigned long long)sr.bytes_written, sr.reused, (unsigned long long)sr.bytes_reused);
    } else if (verb == "join") {
        sr = modelfile::shard_join(f, a.get("o"));
        std::printf("joined %s -> %s: %zu tensors from %zu blobs, %llu bytes\n", f.c_str(), a.get("o").c_str(), sr.tensors, sr.shards, (unsigned long long)sr.bytes);
    } else return usage();
    for (auto& n : sr.notes) std::fprintf(stderr, "forge shard: %s\n", n.c_str());
    return 0;
}

// forge blueprint <model.fpk | file.fbp | file.fbpb> --to binary|text|canonical [-o out]
// A model file: its blueprint section is rewritten in that form. A bare blueprint file: converted to -o.
bool is_fpk(const std::string& f);
int cmd_blueprint(const args& a) {
    if (a.pos.empty() || a.get("to").empty()) return usage();
    const std::string to = a.get("to");
    if (to != "binary" && to != "text" && to != "canonical") throw std::runtime_error("--to takes binary, text or canonical");
    activate_policy(a.get("policy"));
    if (is_fpk(a.pos[0])) {
        if (to == "canonical") throw std::runtime_error("--to canonical is for bare blueprint files; a model file stores text or binary");
        auto out = a.get("o").empty() ? a.pos[0] : a.get("o");
        import::check_write(out);
        bool changed = modelfile::convert_blueprint_section(a.pos[0], out, to == "binary");
        if (!changed) { std::printf("%s: the blueprint is already in the %s form\n", a.pos[0].c_str(), to.c_str()); return 0; }
        modelfile::reader r(out);
        std::printf("%s: blueprint now %s (%zu bytes) -> %s\n", a.pos[0].c_str(), to.c_str(), r.section_text("blueprint").size(), out.c_str());
        std::puts("note: a signature and precompiled plans do not survive this; sign / run forge aot again");
        return 0;
    }
    if (a.get("o").empty()) return usage();
    auto src = import::read_file(a.pos[0]);
    std::string text = modelfile::blueprint_text(src), res;
    if (to == "binary") { auto b = modelfile::encode_blueprint(text); res.assign(reinterpret_cast<const char*>(b.data()), b.size()); }
    else res = to == "canonical" ? modelfile::canonical_blueprint_text(text) : text;
    import::check_write(a.get("o"));
    std::ofstream f(a.get("o"), std::ios::binary); f << res;
    if (!f) throw std::runtime_error("cannot write " + a.get("o"));
    std::printf("%s (%zu bytes) -> %s (%s, %zu bytes)\n", a.pos[0].c_str(), src.size(), a.get("o").c_str(), to.c_str(), res.size());
    return 0;
}

// forge requant <in.fpk> -o <out.fpk> --codec C [mixed-precision options of import] [--no-calib] [--calib-crest X] [--align A]
int cmd_requant(const args& a) {
    if (a.pos.empty() || a.get("o").empty() || a.get("codec").empty()) return usage();
    import::requant_options ro; auto& o = ro.base;
    o.codec = a.get("codec"); o.emb_codec = a.get("emb-codec");
    o.recipe.recipe = a.get("recipe", "forge");
    o.recipe.output_type = a.get("output-tensor-type");
    o.recipe.token_embedding_type = a.get("token-embedding-type");
    o.recipe.pure = a.has("pure");
    o.recipe.head_bf16 = a.has("bf16-head");
    o.recipe.calib_crest = a.num("calib-crest", o.recipe.calib_crest);
    for (std::string tt = a.get("tensor-type"); !tt.empty();) {
        auto c = tt.find(','); o.recipe.tensor_types.push_back(import::parse_tensor_type(tt.substr(0, c)));
        tt = c == std::string::npos ? "" : tt.substr(c + 1);
    }
    if (!a.get("align").empty()) o.align = std::size_t(parse_size_arg(a.get("align"), "--align"));
    ro.use_calib = !a.has("no-calib");
    activate_policy(a.get("policy"));
    auto r = import::requantise(a.pos[0], a.get("o"), ro);
    std::printf("requantised %s -> %s: %zu tensors (%zu re-encoded as %s, %zu copied), %llu bytes\n", a.pos[0].c_str(), a.get("o").c_str(), r.tensors, r.requantised, o.codec.c_str(), r.kept, (unsigned long long)r.bytes);
    if (ro.use_calib) std::printf("calib     %s; it changed the type of %zu tensor(s)\n", r.calib_from_file ? "statistics read from the input's calib section" : "no calib section: statistics computed from the f32 weights", r.calib_moved);
    for (auto& n : r.notes) std::fprintf(stderr, "forge requant: %s\n", n.c_str());
    return 0;
}

int cmd_dump(const args& a) {  // decode one tensor to raw little-endian f32
    if (a.pos.size() < 3) return usage();
    modelfile::reader r(a.pos[0]);
    auto t = r.load(a.pos[1]);
    std::ofstream f(a.pos[2], std::ios::binary);
    f.write(reinterpret_cast<const char*>(t.data.data()), std::streamsize(t.size() * 4));
    std::printf("%s %s %s -> %s\n", a.pos[1].c_str(), r.entry(a.pos[1]).codec.c_str(), shape_str(t.shape).c_str(), a.pos[2].c_str());
    return 0;
}

int cmd_keygen(const args& a) {
    if (a.get("o").empty()) return usage();
    auto k = modelfile::generate_keypair();
    auto base = a.get("o");
    {   // FSS-FIO-1: the secret is 0600 from its first byte (exclusive temp file + rename), never world-readable
        auto hex = modelfile::detail::to_hex(k.seed) + "\n";
        auto w = os::fs::write_file_atomic(base + ".key", hex, 0600); crypto_wipe(hex.data(), hex.size());
        if (!w) throw std::runtime_error("cannot write " + base + ".key: " + w.err.what);
    }
    { std::ofstream f(base + ".pub"); f << modelfile::detail::to_hex(k.pub) << "\n"; }
    std::printf("key id %s\npublic %s\nwrote %s.key (keep secret) and %s.pub\n", modelfile::key_id(k.pub).c_str(), modelfile::detail::to_hex(k.pub).c_str(), base.c_str(), base.c_str());
    return 0;
}
// .fpk files carry an embedded signature; anything else (plugins, scripts, ...) or --detached gets `<file>.sig`
// (FSS INT-2, security/integrity.hpp).
bool is_fpk(const std::string& f) {
    std::ifstream s(f, std::ios::binary); char m[8] = {};
    return s.read(m, 8) && std::memcmp(m, modelfile::magic, 8) == 0;
}
int cmd_sign(const args& a) {
    if (a.pos.empty() || a.get("key").empty()) return usage();
    auto k = modelfile::load_secret_key(a.get("key"));
    if (a.has("detached") || !is_fpk(a.pos[0])) {   // forge sign FILE --key K [-o FILE.sig] [--name NAME]
        const std::string& f = a.pos[0];
        auto bytes = os::read_file(f);
        if (!bytes) throw std::runtime_error("cannot read " + f + ": " + bytes.err.what);
        auto out = a.get("o").empty() ? security::detached_sig_path(f) : a.get("o");
        std::string old; if (auto e = os::read_file(out)) old = e.value;   // extend an existing .sig (another signer)
        auto nm = a.get("name").empty() ? security::signed_name(f) : a.get("name");
        auto text = security::detached_sign({reinterpret_cast<const std::uint8_t*>(bytes.value.data()), bytes.value.size()}, nm, k.seed, old);
        auto tmp = out + ".tmp";
        if (auto w = os::write_file(tmp, text); !w) throw std::runtime_error("cannot write " + tmp + ": " + w.err.what);
        if (auto rs = os::rename(tmp, out); !rs) { os::remove(tmp); throw std::runtime_error("cannot replace " + out + ": " + rs.err.what); }
        std::printf("signed %s (as '%s') with key %s -> %s\n", f.c_str(), nm.c_str(), modelfile::key_id(k.pub).c_str(), out.c_str());
        return 0;
    }
    auto out = a.get("o").empty() ? a.pos[0] : a.get("o");
    modelfile::sign_file(a.pos[0], out, k);
    std::printf("signed %s with key %s -> %s\n", a.pos[0].c_str(), modelfile::key_id(k.pub).c_str(), out.c_str());
    return 0;
}
// "4096", "64k", "1M", "2G" -> bytes (binary multiples)
std::uint64_t parse_size_arg(const std::string& v, const char* what) {
    std::size_t used = 0; double n = 0;
    try { n = std::stod(v, &used); } catch (const std::exception&) { throw std::runtime_error(std::string(what) + ": bad size '" + v + "'"); }
    std::string unit = v.substr(used); for (auto& c : unit) c = char(std::tolower(static_cast<unsigned char>(c)));
    double mul = unit.empty() || unit == "b" ? 1 : unit == "k" || unit == "kb" || unit == "kib" ? 1024.0 : unit == "m" || unit == "mb" || unit == "mib" ? 1048576.0
               : unit == "g" || unit == "gb" || unit == "gib" ? 1073741824.0 : -1;
    if (mul < 0 || n < 0 || n * mul > 9e15) throw std::runtime_error(std::string(what) + ": bad size '" + v + "'");
    return std::uint64_t(n * mul);
}
int cmd_encrypt(const args& a) {   // forge encrypt <in.fpk> -o <out.fpk> [--key k.mkey | --new-key k.mkey]
    if (a.pos.empty() || a.get("o").empty() || (a.get("key").empty() && a.get("new-key").empty())) return usage();
    modelfile::model_key k;
    if (!a.get("new-key").empty()) {
        k = modelfile::generate_model_key();
        auto hex = modelfile::detail::hex(k.data(), 32) + "\n";   // FSS-FIO-1: 0600 from the first byte
        auto st = os::fs::write_file_atomic(a.get("new-key"), hex, 0600); crypto_wipe(hex.data(), hex.size());
        if (!st) throw std::runtime_error("cannot write " + a.get("new-key") + ": " + st.err.what);
    } else {
        activate_policy(a.get("policy"));
        security::require_private_file("model key", a.get("key"), security::file_kind::secret);   // FSS-FIO-2
        std::ifstream f(a.get("key")); std::string h, w; while (f >> w) h += w;
        auto b = modelfile::detail::unhex(h); if (b.size() != 32) throw std::runtime_error("model key must be 64 hex chars");
        std::memcpy(k.data(), b.data(), 32);
    }
    // --chunk SIZE: plaintext bytes per sealed chunk (default 1M); --single-aead: the old one-AEAD scheme
    modelfile::encrypt_options eo;
    if (!a.get("chunk").empty()) eo.chunk_bytes = std::size_t(parse_size_arg(a.get("chunk"), "--chunk"));
    eo.single_aead = a.has("single-aead");
    modelfile::encrypt_file(a.pos[0], a.get("o"), k, eo);
    std::printf("encrypted %s -> %s (key id %s, %s)\n", a.pos[0].c_str(), a.get("o").c_str(), modelfile::detail::enc_key_id(k).c_str(),
                eo.single_aead ? "single AEAD" : std::format("{}-byte chunks", eo.chunk_bytes).c_str());
    return 0;
}
int cmd_verify(const args& a) {
    if (a.pos.empty()) return usage();
    std::vector<std::string> trusted;
    for (auto& p : a.all("pubkey")) trusted.push_back(modelfile::load_public_key(p));
    if (a.has("detached") || !a.get("sig").empty() || !is_fpk(a.pos[0])) {   // forge verify FILE [--sig FILE.sig] [--pubkey K]...
        auto rep = security::detached_verify_file(a.pos[0], trusted, a.get("sig"));
        for (auto& k : rep.valid_keys) std::printf("signature ok   key %s\n", k.c_str());
        for (auto& k : rep.invalid_keys) std::printf("signature BAD  key %s\n", k.c_str());
        std::printf("%s: %s\n", a.pos[0].c_str(), rep.trusted ? "VERIFIED" : rep.message.c_str());
        return rep.trusted ? 0 : 1;
    }
    modelfile::reader r(a.pos[0]);
    auto bad = r.verify();
    for (auto& b : bad) std::printf("hash MISMATCH %s\n", b.c_str());
    auto rep = modelfile::verify_signature(r, trusted);
    for (auto& k : rep.valid_keys) std::printf("signature ok   key %s\n", k.c_str());
    for (auto& k : rep.invalid_keys) std::printf("signature BAD  key %s\n", k.c_str());
    std::printf("%s: %s\n", a.pos[0].c_str(), bad.empty() && rep.trusted ? "VERIFIED" : rep.message.c_str());
    return bad.empty() && rep.trusted ? 0 : 1;
}

// one_shot models: --input name=file.f32 (raw little-endian f32, repeatable), --out prefix -> prefix.<output>.f32
int run_one_shot(inference::session& s, const args& a) {
    std::map<std::string, std::vector<float>> in; i64 seq = 1;
    const json::value* decl = s.manifest().find("onnx_inputs");
    // symbolic dimensions: --dim name=N binds one; the rest are worked out from the input sizes where an input has
    // exactly one unbound dimension
    std::map<std::string, i64> dims;
    for (auto& spec : a.all("dim")) {
        auto eq = spec.find('=');
        if (eq == std::string::npos || eq == 0) raise(errc::invalid_argument, "--dim expects name=N, got '{}'", spec);
        dims[spec.substr(0, eq)] = i64(std::stoll(spec.substr(eq + 1)));
    }
    for (auto& spec : a.all("input")) {
        auto eq = spec.find('=');
        if (eq == std::string::npos) raise(errc::invalid_argument, "--input expects name=file, got '{}'", spec);
        auto name = spec.substr(0, eq); auto file = spec.substr(eq + 1); auto bytes = import::read_file(file);
        // raw little-endian f32, or integers for integer inputs: file.i64 / file.i32 (token ids, masks)
        std::vector<float> v;
        if (file.ends_with(".i64") || file.ends_with(".i32")) {
            const std::size_t w = file.ends_with(".i64") ? 8 : 4;
            if (bytes.size() % w) raise(errc::invalid_argument, "input file {} is not raw {}", file, w == 8 ? "int64" : "int32");
            v.resize(bytes.size() / w);
            for (std::size_t i = 0; i < v.size(); ++i) { if (w == 8) { std::int64_t x; std::memcpy(&x, bytes.data() + 8 * i, 8); v[i] = float(x); } else { std::int32_t x; std::memcpy(&x, bytes.data() + 4 * i, 4); v[i] = float(x); } }
        } else {
            if (bytes.size() % 4) raise(errc::invalid_argument, "input file {} is not raw f32", file);
            v.resize(bytes.size() / 4); std::memcpy(v.data(), bytes.data(), bytes.size());
        }
        if (decl) for (auto& d : decl->arr()) {
            if (d["onnx"].str() != name && d["name"].str() != name) continue;
            name = d["name"].str();
            if (auto* dt = d.find("dims"); dt && dt->is_arr()) {   // dimension names: solve the one that is not bound yet
                i64 k = 1; std::string unknown; int n_unknown = 0;
                for (auto& x : dt->arr()) {
                    const std::string t = x.str();
                    if (!t.empty() && std::isdigit(static_cast<unsigned char>(t[0]))) k *= i64(std::stoll(t));
                    else if (auto it = dims.find(t); it != dims.end()) k *= it->second;
                    else { if (unknown != t) ++n_unknown; unknown = t; }
                }
                if (n_unknown == 1 && k > 0) {
                    if (i64(v.size()) % k) raise(errc::invalid_argument, "input {} has {} values, which is not a multiple of {}", name, v.size(), k);
                    dims[unknown] = i64(v.size()) / k;
                } else if (n_unknown > 1) raise(errc::invalid_argument, "input {} has {} unbound dimensions; give them with --dim name=N", name, n_unknown);
            } else {   // files imported before dimension names were recorded: one symbolic dimension, `seq`
                i64 k = 1; bool sym = false;
                for (auto& x : d["shape"].arr()) { if (x.num() < 0) sym = true; else k *= x.i(); }
                if (sym && k) dims["seq"] = i64(v.size()) / k;
            }
        }
        in[name] = std::move(v);
    }
    if (auto it = dims.find("seq"); it != dims.end()) seq = it->second;
    auto out = s.infer(in, seq, dims);
    auto prefix = a.get("out");
    for (auto& [k, v] : out) {
        if (prefix.empty()) { std::printf("%s:", k.c_str()); for (auto x : v) std::printf(" %.6g", x); std::printf("\n"); continue; }
        std::ofstream f(prefix + "." + k + ".f32", std::ios::binary);
        f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * 4));
    }
    return 0;
}

// name=file arguments -> {name: values}; files are raw f32, or .i64 / .i32 integers
inference::tensor_map read_named_inputs(const std::vector<std::string>& specs, const char* opt) {
    inference::tensor_map m;
    for (auto& spec : specs) {
        auto eq = spec.find('=');
        if (eq == std::string::npos || eq == 0) raise(errc::invalid_argument, "{} expects name=file, got '{}'", opt, spec);
        auto file = spec.substr(eq + 1); auto bytes = import::read_file(file);
        std::vector<float> v;
        if (file.ends_with(".i64") || file.ends_with(".i32")) {
            const std::size_t w = file.ends_with(".i64") ? 8 : 4;
            if (bytes.size() % w) raise(errc::invalid_argument, "input file {} is not raw {}", file, w == 8 ? "int64" : "int32");
            v.resize(bytes.size() / w);
            for (std::size_t i = 0; i < v.size(); ++i) { if (w == 8) { std::int64_t x; std::memcpy(&x, bytes.data() + 8 * i, 8); v[i] = float(x); } else { std::int32_t x; std::memcpy(&x, bytes.data() + 4 * i, 4); v[i] = float(x); } }
        } else {
            if (bytes.size() % 4) raise(errc::invalid_argument, "input file {} is not raw f32", file);
            v.resize(bytes.size() / 4); std::memcpy(v.data(), bytes.data(), bytes.size());
        }
        m[spec.substr(0, eq)] = std::move(v);
    }
    return m;
}
void write_f32(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * 4));
    if (!f) throw std::runtime_error("cannot write " + path);
}

// `forge denoise`: a model whose blueprint declares `session diffusion_loop`: scheduler, step plan and guidance from
// the file (overridable), the model's graph as the denoiser. Writes or prints the final latent.
int cmd_denoise(const args& a) {
    if (a.pos.empty()) return usage();
    activate_policy(a.get("policy"));
    inference::session_options so; so.verify = a.has("verify");
    inference::session s(a.pos[0], so);
    if (!a.get("device").empty()) s.shared_model()->set_device(a.get("device"));
    std::map<std::string, i64> dims;
    for (auto& spec : a.all("dim")) { auto eq = spec.find('='); if (eq == std::string::npos || eq == 0) raise(errc::invalid_argument, "--dim expects name=N, got '{}'", spec); dims[spec.substr(0, eq)] = i64(std::stoll(spec.substr(eq + 1))); }
    inference::diffusion_runner d(s, dims);
    auto o = d.defaults();
    o.steps = int(a.num("steps", o.steps)); o.guidance = float(a.num("guidance", o.guidance)); o.eta = a.num("eta", o.eta); o.seed = std::uint64_t(a.num("seed", 0));
    if (auto k = a.get("sampler"); !k.empty()) {
        if (k == "ddim") o.kind = inference::diffusion::sampler_kind::ddim; else if (k == "ddpm") o.kind = inference::diffusion::sampler_kind::ddpm;
        else if (k == "euler" || k == "euler_flow") o.kind = inference::diffusion::sampler_kind::euler_flow; else raise(errc::invalid_argument, "--sampler takes ddim, ddpm or euler, not '{}'", k);
    }
    auto cond = read_named_inputs(a.all("input"), "--input"), uncond = read_named_inputs(a.all("uncond"), "--uncond");
    inference::diffusion::latent x;
    if (!a.get("init").empty()) { auto m = read_named_inputs({"x=" + a.get("init")}, "--init"); x = std::move(m.at("x")); }   // start from a given latent instead of seeded noise
    else x = d.initial_noise(o.seed);
    auto t0 = std::chrono::steady_clock::now();
    auto y = d.run(std::move(x), o, cond, uncond);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "[diffusion: %d steps, %zu denoiser calls, latent of %zu values, %.3fs]\n", o.steps, d.model_calls(), y.size(), sec);
    if (!a.get("out").empty()) write_f32(a.get("out"), y);
    else { for (auto v : y) std::printf("%.6g ", v); std::printf("\n"); }
    return 0;
}

// `forge stream`: a model whose blueprint declares `session streaming`: samples from --input are fed in chunks of
// --chunk, framed by the file's window / hop, and the model runs once per frame with its state carried over.
int cmd_stream(const args& a) {
    if (a.pos.empty() || a.get("input").empty()) return usage();
    activate_policy(a.get("policy"));
    inference::session_options so; so.verify = a.has("verify");
    inference::session s(a.pos[0], so);
    if (!a.get("device").empty()) s.shared_model()->set_device(a.get("device"));
    inference::stream_runner r(s);
    auto samples = read_named_inputs({"x=" + a.get("input")}, "--input").at("x");
    std::size_t chunk = std::size_t(std::max(1.0, a.num("chunk", double(r.options().hop))));
    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t at = 0; at < samples.size(); at += chunk) r.feed({samples.data() + at, std::min(chunk, samples.size() - at)});
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto& st = r.stat();
    std::fprintf(stderr, "[stream: %zu samples in chunks of %zu, window %zu hop %zu: %zu frames, %zu dropped, %zu values per frame, worst frame %.3f ms, %.3fs]\n",
                 st.samples_in, chunk, r.options().window, r.options().hop, st.frames, st.dropped, r.frame_size(), st.worst_ms, sec);
    if (!a.get("out").empty()) write_f32(a.get("out"), r.output());
    else { for (auto v : r.output()) std::printf("%.6g ", v); std::printf("\n"); }
    return 0;
}

// `forge embed`: one pass of an embedding model (blueprint `session embedding`); prints or writes the result vector.
int cmd_embed(const args& a) {
    if (a.pos.empty()) return usage();
    activate_policy(a.get("policy"));
    inference::session_options so; so.verify = a.has("verify");
    inference::session s(resolve_model(a.pos[0], a.get("codec", "f32")), so);
    if (a.kv.count("device")) s.shared_model()->set_device(a.get("device"));
    std::vector<float> v;
    if (!a.all("input").empty()) {   // named raw-f32 inputs: an image, mel frames, or the token inputs themselves
        std::map<std::string, std::vector<float>> in; i64 seq = 1;
        for (auto& spec : a.all("input")) {
            auto eq = spec.find('=');
            if (eq == std::string::npos) raise(errc::invalid_argument, "--input expects name=file, got '{}'", spec);
            auto bytes = import::read_file(spec.substr(eq + 1));
            if (bytes.size() % 4) raise(errc::invalid_argument, "input file {} is not raw f32", spec.substr(eq + 1));
            std::vector<float> x(bytes.size() / 4); std::memcpy(x.data(), bytes.data(), bytes.size());
            for (auto& d : s.blueprint().inputs) if (d.name == spec.substr(0, eq) && d.dims.size() == 1 && d.dims[0] == "seq") seq = i64(x.size());
            in[spec.substr(0, eq)] = std::move(x);
        }
        v = s.embed_inputs(in, seq);
    } else {
        std::vector<int> ids, types;
        if (a.kv.count("tokens")) ids = parse_ids(a.get("tokens"));
        else if (a.kv.count("text")) {   // the tokenizer's post-processor adds [CLS] / [SEP] and assigns the token types
            if (!s.tokenizer()) raise(errc::not_found, "model has no tokenizer the runtime can use; pass --tokens");
            auto e = a.kv.count("text-pair") ? s.tokenizer()->encode_pair(a.get("text"), a.get("text-pair")) : s.tokenizer()->encode_typed(a.get("text"));
            ids = std::move(e.ids); types = std::move(e.type_ids);
            if (a.has("ids")) { for (std::size_t i = 0; i < ids.size(); ++i) std::fprintf(stderr, i ? " %d" : "%d", ids[i]); std::fprintf(stderr, "\n"); }
        } else return usage();
        if (a.kv.count("token-types")) types = parse_ids(a.get("token-types"));
        v = s.embed(ids, types, a.kv.count("mask") ? parse_ids(a.get("mask")) : std::vector<int>{});
    }
    if (auto& as = s.shared_model()->aot().stats(); as.used)   // precompiled plans (forge aot), as `forge run` reports them
        std::fprintf(stderr, "[aot: %zu of the plans were precompiled (%zu native), from %s]\n", as.used, as.native, as.origin.c_str());
    if (a.kv.count("out")) {
        std::ofstream f(a.get("out"), std::ios::binary);
        f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * 4));
        if (!f) raise(errc::io_error, "cannot write {}", a.get("out"));
    } else {
        for (std::size_t i = 0; i < v.size(); ++i) std::printf(i ? " %.6g" : "%.6g", double(v[i]));
        std::printf("\n");
    }
    return 0;
}

int cmd_run(const args& a) {
    if (a.pos.empty()) return usage();
    activate_policy(a.get("policy"));
    inference::session_options so; so.context = i64(a.num("ctx", 0)); so.verify = a.has("verify");
    inference::session s(resolve_model(a.pos[0], a.get("codec"), a.has("allow-unused"), a.has("no-optimise")), so);
    if (!a.get("adapter").empty()) s.set_adapter(a.get("adapter"));   // a LoRA adapter carried in the model file, by name
    if (s.blueprint().session != "generative_sequence") return run_one_shot(s, a);
    std::vector<int> prompt;
    if (a.kv.count("tokens")) prompt = parse_ids(a.get("tokens"));
    else if (a.kv.count("prompt")) {
        if (!s.tokenizer()) raise(errc::not_found, "model has no tokenizer; use --tokens");
        if (a.has("chat")) {   // --prompt is a user message (after an optional --system one), put through the model's chat template
            json::array msgs;
            if (a.kv.count("system")) msgs.push_back(json::object{{"role", "system"}, {"content", a.get("system")}});
            msgs.push_back(json::object{{"role", "user"}, {"content", a.get("prompt")}});
            prompt = s.tokenizer()->encode(inference::apply_chat_template(*s.shared_model(), json::value(msgs)));
        } else {
            prompt = s.tokenizer()->encode(a.get("prompt"));
            if (s.tokenizer()->add_bos() && !a.has("no-bos")) prompt.insert(prompt.begin(), s.tokenizer()->bos());
        }
    } else return usage();
    inference::sampler_config sc;
    sc.temperature = float(a.num("temp", 0)); sc.top_k = int(a.num("top-k", 0)); sc.top_p = float(a.num("top-p", 1));
    sc.min_p = float(a.num("min-p", 0)); sc.repetition_penalty = float(a.num("repeat-penalty", 1)); sc.seed = std::uint64_t(a.num("seed", 0));
    inference::sampler smp(sc);
    bool ids = a.has("ids") || !s.tokenizer();
    if (a.kv.count("logits")) {  // dump prefill logits for golden comparison
        auto l = s.forward(prompt);
        std::ofstream f(a.get("logits"), std::ios::binary);
        f.write(reinterpret_cast<const char*>(l.data()), std::streamsize(l.size() * 4));
        s.reset();
    }
    std::unique_ptr<inference::grammar_constraint> gc;
    if (a.kv.count("grammar")) {
        if (!s.tokenizer()) raise(errc::unsupported, "--grammar needs a model tokenizer");
        auto g = std::make_shared<inference::grammar>(import::read_file(a.get("grammar")));
        gc = std::make_unique<inference::grammar_constraint>(g, *s.tokenizer(), int(s.vocab()), s.tokenizer()->eos());
        smp.filter = [&](std::span<float> l) { gc->apply(l); }; smp.allow = [&](int t) { return gc->allows(t); };
    }
    // --chat: the turn ends at the model's end-of-turn tokens (<|im_end|>, <|eot_id|>, ...: inference/chat_tools.hpp), not only at EOS
    std::vector<int> turn_ends;
    if (a.has("chat")) for (int t : inference::end_of_turn_tokens(*s.shared_model())) if (t != s.tokenizer()->eos()) turn_ends.push_back(t);
    auto show = [&](int t) {
        if (gc && !gc->accept(t)) return false;
        if (std::find(turn_ends.begin(), turn_ends.end(), t) != turn_ends.end()) return false;
        if (ids) std::printf("%d ", t); else std::printf("%s", s.tokenizer()->decode_one(t).c_str());
        std::fflush(stdout);
        return !(gc && gc->finished());
    };
    // Speculative decoding: --draft model.fpk or --lookup (n-gram drafts from the prompt and output).
    // --spec-k N: N drafts per pass; 0 (default) = chosen per pass by the planner from measured costs.
    // Not with --grammar: the constraint advances one token at a time.
    // --self-spec [--exit-layer K]: drafts from the model's own early exit (its first K layers and the LM head);
    // K defaults to the blueprint's `early_exit`, else half the depth. The output is that of plain decoding.
    if ((a.kv.count("draft") || a.has("lookup") || a.has("self-spec")) && !gc) {
        inference::speculative_stats sst;
        int k = int(a.num("spec-k", 0)), n = int(a.num("n", 32));
        if (a.has("self-spec")) {
            inference::self_speculative_generate(s, prompt, n, smp, i64(a.num("exit-layer", 0)), k, show, &sst);
        } else if (a.kv.count("draft")) {
            inference::session d(resolve_model(a.get("draft"), a.get("codec")), so);
            inference::speculative_generate(s, d, prompt, n, smp, k, show, &sst);
        } else {
            int ng = std::max(1, int(a.num("ngram", 3)));
            inference::lookup_generate(s, prompt, n, smp, k, ng, show, &sst, nullptr, int(a.num("ngram-min", 1)));
        }
        std::printf("\n");
        std::string pos;
        for (std::size_t i = 0; i < 8 && sst.pos_tried[i]; ++i) pos += std::format("{}{:.0f}%", i ? " " : "", 100.0 * double(sst.pos_won[i]) / double(sst.pos_tried[i]));
        std::fprintf(stderr, "[speculative: %zu tokens in %.3fs = %.1f tok/s decode (prefill %.3fs); %zu passes, %zu with drafts, %zu drafted, acceptance %.0f%% (by position: %s), %.2f tokens/pass, %zu replay rows, %zu probes]\n",
                     sst.new_tokens, sst.seconds, sst.decode_tps(), sst.prefill_seconds, sst.passes, sst.rounds, sst.drafted, 100 * sst.acceptance(), pos.c_str(),
                     sst.tokens_per_pass(), sst.replayed, sst.probes);
        return 0;
    }
    inference::generate_stats st;
    std::vector<int> out = s.generate(prompt, int(a.num("n", 32)), smp, show, &st);
    std::printf("\n");
    std::fprintf(stderr, "[%zu prompt tokens in %.3fs, %zu new tokens in %.3fs = %.1f tok/s; plans %zu compiled, %zu reused]\n",
                 st.prompt_tokens, st.prefill_s, st.new_tokens, st.decode_s, st.new_tokens > 1 ? double(st.new_tokens - 1) / st.decode_s : 0.0,
                 s.plans().misses(), s.plans().hits());
    if (auto& as = s.shared_model()->aot().stats(); as.used)   // precompiled plans (forge aot)
        std::fprintf(stderr, "[aot: %zu of the plans were precompiled (%zu native), from %s]\n", as.used, as.native, as.origin.c_str());
    return 0;
}

int cmd_bench_gemm(int n) {
    std::vector<float> A(std::size_t(n) * n, 0.5f), B(std::size_t(n) * n, 0.25f), C(std::size_t(n) * n);
    for (const char* t : simd::target_names()) {
        auto* k = simd::by_name(t); if (!k) continue;
        k->gemm_nt(A.data(), B.data(), C.data(), std::size_t(n), std::size_t(n), std::size_t(n));
        auto t0 = std::chrono::steady_clock::now(); int reps = 0;
        do { k->gemm_nt(A.data(), B.data(), C.data(), std::size_t(n), std::size_t(n), std::size_t(n)); ++reps; }
        while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(500));
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("simd::%-7s gemm_nt %d^3: %8.2f GFLOP/s (1 thread)\n", t, n, 2.0 * n * n * n / s / 1e9);
    }
    auto t0 = std::chrono::steady_clock::now(); int reps = 0;
    do { gpu::prims::gemm_nt(A.data(), B.data(), C.data(), std::size_t(n), std::size_t(n), std::size_t(n)); ++reps; }
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(500));
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
    std::printf("gpu::prims gemm_nt %d^3: %8.2f GFLOP/s (all threads)\n", n, 2.0 * n * n * n / s / 1e9);
    return 0;
}

// Quantised matvec bandwidth per weight format: a matrix of `mb` MB (well past the last-level cache) times 1..8
// activation vectors, on the integer kernels and on the f32 decode + dot path, with every pool thread.
int cmd_bench_kernels(int mb) {
    using namespace forge::graph;
    const std::size_t k = 4096;
    std::mt19937 rng(1);
    std::printf("quantised matvec, k=%zu, %d MB matrix, %u threads, integer kernels: %s, prefetch %d B\n", k, mb, task::engine_threads(),
                simd::qi8_active().target, simd::qi8_prefetch_bytes);
    for (const char* cn : {"q4_k", "q5_k", "q6_k", "q8_0"}) {
        const auto* c = codecs::find(cn); const int fmt = qweights::fmt_of(c);
        const std::size_t rb = c->bytes(k), n = std::size_t(mb) * 1000000 / rb, bytes = n * rb;
        std::vector<std::uint8_t> w(bytes);
        for (auto& b : w) b = std::uint8_t(rng());
        const std::uint16_t one = 0x2000;   // a small finite f16 for every scale field
        for (std::size_t j = 0; j < n * (k / c->block); ++j) {
            std::uint8_t* blk = w.data() + j * c->block_bytes;
            if (c->name == "q6_k") std::memcpy(blk + 208, &one, 2);
            else { std::memcpy(blk, &one, 2); if (c->name != "q8_0") std::memcpy(blk + 2, &one, 2); }
        }
        std::vector<float> x(8 * k), y(8 * n);
        for (auto& v : x) v = float(rng() % 2001) / 1000.f - 1.f;
        auto timeit = [&](auto&& f) {
            f(); double best = 1e30;
            for (int rep = 0; rep < 5; ++rep) {
                auto t0 = std::chrono::steady_clock::now(); f();
                best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            return best;
        };
        std::printf("  %-5s", cn);
        for (std::size_t nv : {1, 2, 4, 8}) {
            double s = timeit([&] { qweights::gemm_nt_i8(fmt, w.data(), rb, n, x.data(), nv, k, y.data()); });
            std::printf("  int8 nv=%zu %6.1f GB/s", nv, double(bytes) / s / 1e9);
        }
        double s = timeit([&] {
            task::parallel_for(n, 8, [&](std::size_t b, std::size_t e) {
                const auto& ker = simd::active();
                for (std::size_t j = b; j < e; ++j) y[j] = ker.qdot(fmt, w.data() + j * rb, x.data(), k, c->decode, c->block, c->block_bytes);
            });
        });
        std::printf("  | f32 nv=1 %6.1f GB/s\n", double(bytes) / s / 1e9);
    }
    return 0;
}

int cmd_bench(const args& a) {
    if (a.kv.count("gemm")) return cmd_bench_gemm(int(a.num("gemm", 512)));
    if (a.kv.count("kernels")) return cmd_bench_kernels(int(a.num("kernels", 1024)));
    if (a.pos.empty()) return usage();
    activate_policy(a.get("policy"));
    inference::session s(resolve_model(a.pos[0], a.get("codec")));
    int p = int(a.num("p", 32)), n = int(a.num("n", 32));
    std::vector<int> prompt(std::size_t(p), 1);
    inference::sampler smp;
    // --warmup N (default 1): untimed short runs first, so one-off weight uploads / page-ins of the mmapped
    // model and kernel loads are not counted as prefill or decode time (llama-bench does the same).
    for (int w = int(a.num("warmup", 1)); w > 0; --w) {
        std::vector<int> wp(std::size_t(std::min(p, 64)), 1); inference::generate_stats ws;
        s.generate(wp, 2, smp, {}, &ws); s.reset();
    }
    // GPU driver calls (CUDA/HIP): a prefill-only run (1 new token) and the full run, so the decode share is
    // (full - prefill-only) / (n - 1) per token. "launches" are kernels issued directly, "captured" are kernels
    // recorded into a CUDA graph (replayed by one graph launch), instantiates are graphs (re)built, node_updates are
    // kernel nodes patched in place on replay, dtoh/htod/dtod are copies (dtod a copy kernel under graphs), syncs are stream syncs.
    using snap = gpu::cuda_hip::call_counters::snapshot;
    auto now = [] { return gpu::cuda_hip::counters().read(); };
    auto diff = [](const snap& a, const snap& b) {
        return snap{b.launches - a.launches, b.captured - a.captured, b.graph_launches - a.graph_launches, b.dtoh - a.dtoh,
                    b.htod - a.htod, b.dtod - a.dtod, b.syncs - a.syncs, b.instantiates - a.instantiates, b.node_updates - a.node_updates, b.replayed_runs - a.replayed_runs};
    };
    snap c0 = now();
    { inference::generate_stats ps; s.generate(prompt, 1, smp, {}, &ps, false); s.reset(); }
    snap pre = diff(c0, now());
    inference::generate_stats st;
    snap c1 = now();
    s.generate(prompt, n, smp, {}, &st, /*stop_at_eos=*/false);   // bench: always decode n tokens
    snap all = diff(c1, now());
    std::printf("prefill %d tokens: %.1f tok/s\ndecode  %zu tokens: %.1f tok/s\n", p, p / st.prefill_s, st.new_tokens,
                st.new_tokens > 1 ? double(st.new_tokens - 1) / st.decode_s : 0.0);
    if (all.launches + all.captured > 0) {
        auto pr = [](const char* what, const snap& c, double per) {
            std::printf("%s: launches %.1f captured %.1f graph_launches %.1f instantiates %.1f node_updates %.1f replayed %.2f dtoh %.1f htod %.1f dtod %.1f syncs %.1f\n", what,
                        c.launches / per, c.captured / per, c.graph_launches / per, double(std::int64_t(c.instantiates)) / per, double(std::int64_t(c.node_updates)) / per, double(std::int64_t(c.replayed_runs)) / per, c.dtoh / per, c.htod / per, c.dtod / per, c.syncs / per);
        };
        pr("gpu prefill (total)", pre, 1.0);
        if (st.new_tokens > 1) pr("gpu decode (per token)", diff(pre, all), double(st.new_tokens - 1));
    }
    return 0;
}

// forge rpc-serve [--port 7000] [--bind 127.0.0.1] [--device cpu|vulkan|cuda:0...]
//   [--token T | --token-file F]... [--authorized-keys FILE | --pubkey HEX|FILE]...
//   [--tls-cert C --tls-key K [--tls-client-ca CA [--tls-client-map FILE]... [--tls-crl PEM]... [--tls-revoked SHA256|FILE]...]]
//   [--quiet]
//   [--allow-scripts [--script-timeout MS] [--script-mem-mb N] [--script-no-proc] [--script-no-fs] [--script-no-net]]
// Without tokens/keys it is unauthenticated (keep it on loopback). Scripts require auth. See docs/rpc.md.
int cmd_rpc_serve(const args& a) {
    // FSS (docs/rpc.md "Security policy"): with an active policy the listen address must be allowed, script children
    // need exec, and the server locks itself down after binding, before it accepts a request.
    const auto pol = activate_policy(a.get("policy"));
    std::string why, dev = a.get("device", "cpu");
    auto reg = graph::device_registry(dev, &why, false);
    if (!reg && dev != "cpu") std::fprintf(stderr, "forge: %s; serving on cpu\n", why.c_str());
    graph::rpc::server_security sec;
    auto toks = a.all("token"); if (toks.empty() && a.has("token")) toks.push_back(a.get("token"));
    for (auto& t : toks) {   // "--token T" (full access) or "--token 'allow=device T'" (a token-file line)
        if (!graph::rpc::sec::is_option(t)) { sec.tokens.push_back(t); continue; }
        for (auto& e : graph::rpc::parse_tokens(t, "--token")) sec.tokens.push_back(std::move(e));
    }
    auto files = a.all("token-file"); if (files.empty() && a.has("token-file")) files.push_back(a.get("token-file"));
    // FSS-FIO-2: secrets (token files, the TLS key) must be owner-only and trusted lists (authorized keys, client
    // maps) writable only by their owner; profile strict refuses otherwise, other profiles warn (audit).
    for (auto& f : files) security::require_private_file(pol, "--token-file", f, security::file_kind::secret);
    for (auto& f : files) {   // "[allow=...] [ops=...] [max-bytes=N] [script-timeout=MS] <token> [name]" per line (graph/rpc_secure.hpp)
        auto r = os::read_file(f); if (!r) throw std::runtime_error("cannot read " + f);
        struct wipe_on_exit { std::string& s; ~wipe_on_exit() { if (!s.empty()) crypto_wipe(s.data(), s.size()); } } w{r.value};
        auto ts = graph::rpc::parse_tokens(r.value, f);
        if (ts.empty()) throw std::runtime_error(f + ": no tokens");
        for (auto& e : ts) sec.tokens.push_back(std::move(e));
    }
    auto keyfiles = a.all("authorized-keys"); if (keyfiles.empty() && a.has("authorized-keys")) keyfiles.push_back(a.get("authorized-keys"));
    for (auto& f : keyfiles) {   // "[options] <hex> [comment]" per line, options as for tokens (graph/rpc_secure.hpp)
        security::require_private_file(pol, "--authorized-keys", f, security::file_kind::trusted);
        auto r = os::read_file(f); if (!r) throw std::runtime_error("cannot read " + f);
        try { for (auto& k : graph::rpc::parse_authorized_keys(r.value)) sec.authorized_keys.push_back(std::move(k)); }
        catch (const std::exception& e) { throw std::runtime_error(f + ": " + e.what()); }
    }
    auto pubs = a.all("pubkey"); if (pubs.empty() && a.has("pubkey")) pubs.push_back(a.get("pubkey"));
    for (auto& p : pubs) sec.authorized_keys.push_back(p.size() == 64 && p.find('/') == std::string::npos && !std::filesystem::exists(p) ? p : modelfile::load_public_key(p));
    sec.tls.cert_file = a.get("tls-cert"); sec.tls.key_file = a.get("tls-key"); sec.tls.client_ca_file = a.get("tls-client-ca");
    if (sec.tls.cert_file.empty() != sec.tls.key_file.empty()) throw std::runtime_error("--tls-cert and --tls-key go together");
    security::require_private_file(pol, "--tls-key", sec.tls.key_file, security::file_kind::secret);
    if (!sec.tls.client_ca_file.empty() && sec.tls.cert_file.empty()) throw std::runtime_error("--tls-client-ca needs --tls-cert and --tls-key");
    auto maps = a.all("tls-client-map"); if (maps.empty() && a.has("tls-client-map")) maps.push_back(a.get("tls-client-map"));
    for (auto& f : maps) {   // "[options] sha256:<hex> | san:<type>:<value> | cn:<name> [comment]" per line (graph/rpc_secure.hpp)
        security::require_private_file(pol, "--tls-client-map", f, security::file_kind::trusted);
        auto r = os::read_file(f); if (!r) throw std::runtime_error("cannot read " + f);
        auto rs = graph::rpc::parse_client_map(r.value, f);
        if (rs.empty()) throw std::runtime_error(f + ": no rules (an empty client map would accept no certificate)");
        for (auto& x : rs) sec.client_map.push_back(std::move(x));
    }
    if (!sec.client_map.empty() && sec.tls.client_ca_file.empty()) throw std::runtime_error("--tls-client-map needs --tls-client-ca");
    // Revocation of client certificates: PEM CRLs checked in the TLS handshake, and a deny-list of SHA-256 certificate
    // fingerprints (a fingerprint, or a file of them). The server re-reads the files when they change.
    sec.tls.crl_files = a.all("tls-crl"); if (sec.tls.crl_files.empty() && a.has("tls-crl")) sec.tls.crl_files.push_back(a.get("tls-crl"));
    auto revoked = a.all("tls-revoked"); if (revoked.empty() && a.has("tls-revoked")) revoked.push_back(a.get("tls-revoked"));
    for (auto& r : revoked) {
        if (!os::exists(r)) {   // not a file: one fingerprint
            try { auto one = graph::rpc::parse_revoked(r, "--tls-revoked"); if (one.size() != 1) throw std::runtime_error("empty"); sec.revoked_certs.push_back(one[0]); }
            catch (const std::exception&) { throw std::runtime_error("--tls-revoked wants a SHA-256 certificate fingerprint (64 hex) or a file of them: " + r); }
            continue;
        }
        security::require_private_file(pol, "--tls-revoked", r, security::file_kind::trusted);
        sec.revoked_files.push_back(r);
    }
    for (auto& f : sec.tls.crl_files) security::require_private_file(pol, "--tls-crl", f, security::file_kind::trusted);
    if ((!sec.tls.crl_files.empty() || !revoked.empty()) && sec.tls.client_ca_file.empty()) throw std::runtime_error("--tls-crl and --tls-revoked need --tls-client-ca");
    sec.allow_scripts = a.has("allow-scripts");
    std::string bind = a.get("bind", "127.0.0.1");
    if (pol.active()) {
        security::enforce(pol, security::capability::net_listen, (bind.find(':') != std::string::npos ? "[" + bind + "]" : bind) + ":" + std::to_string(int(a.num("port", 7000))));
        if (sec.allow_scripts && !security::try_enforce(pol, security::capability::exec, os::exe_path()))
            throw security::policy_violation("rpc-serve --allow-scripts starts a `forge script` child per script, but the security policy (" + pol.source + ") denies exec", "SBX-1");
    }
    if (!sec.auth_required() && sec.tls.client_ca_file.empty() && bind != "127.0.0.1" && bind != "::1" && bind != "localhost")
        std::fprintf(stderr, "forge rpc: WARNING: no authentication and bound to %s; anyone who can reach it can use it\n", bind.c_str());
    graph::rpc::server srv(reg.get(), int(a.num("port", 7000)), bind, sec);
    if (!a.has("quiet")) srv.log = [](const std::string& m) { std::fprintf(stderr, "forge rpc: %s\n", m.c_str()); };
    if (sec.allow_scripts) {
        auto def_tmo = std::uint64_t(a.num("script-timeout", 60000));
        auto mem_mb = std::uint64_t(a.num("script-mem-mb", 2048));
        bool no_proc = a.has("script-no-proc"), no_fs = a.has("script-no-fs"), no_net = a.has("script-no-net");
        std::string self = os::exe_path();
        if (self.empty()) throw std::runtime_error("rpc-serve --allow-scripts: cannot find this executable to start script children");
        // children load the same policy (an explicit --policy is passed on as an absolute path; otherwise they find it
        // the way this process did) and lock themselves down before running the script
        const std::string child_policy = a.get("policy").empty() ? std::string() : os::absolute(a.get("policy"));
        // Each remote script runs in its own child process (`forge script --rpc-child`), so ForgeScript's OS sandbox
        // (seccomp / Capsicum / pledge / sandbox_init / restricted token) applies to the script without touching the
        // server's own sockets, and a crash, leak or runaway loop dies with the child. The source goes over stdin.
        // The peer's permissions can only take capabilities away: a script keeps fs / net / proc when the server
        // flags allow it and its token or key (and client certificate) has scripts.fs / scripts.net / scripts.proc.
        srv.script_request_handler = [self, def_tmo, mem_mb, srv_no_proc = no_proc, srv_no_fs = no_fs, srv_no_net = no_net, child_policy](const graph::rpc::server::script_request& q) {
            const auto& src = q.source; const auto& name = q.name; const auto& argv = q.argv; const auto& signature = q.signature;
            const bool no_proc = srv_no_proc || !q.allow_proc, no_fs = srv_no_fs || !q.allow_fs, no_net = srv_no_net || !q.allow_net;
            auto t = q.timeout_ms ? std::min<std::uint64_t>(q.timeout_ms, def_tmo) : def_tmo;
            os::spawn_spec sp;
            sp.argv = {self, "script", "--rpc-child", name.empty() ? "<remote>" : name, "--timeout", std::to_string(t)};
            if (no_proc) sp.argv.push_back("--no-proc");
            if (no_fs) sp.argv.push_back("--no-fs");
            if (no_net) sp.argv.push_back("--no-net");
            // the same cap inside the interpreter, so a runaway script gets OutOfMemoryError before the OS cap below
            if (mem_mb) { sp.argv.push_back("--mem-limit"); sp.argv.push_back(std::to_string(mem_mb) + "M"); }
            if (!child_policy.empty()) { sp.argv.push_back("--policy"); sp.argv.push_back(child_policy); }
            // the client's detached signature (FSS INT-2), hex so it survives the command line; the child verifies it
            if (!signature.empty()) { sp.argv.push_back("--rpc-sig"); sp.argv.push_back(security::detail::hex({reinterpret_cast<const std::uint8_t*>(signature.data()), signature.size()})); }
            // Windows cannot deny fs/net inside a running process (only run_sandboxed's AppContainer can), so there the
            // child applies what it can (job object) and the interpreter checks the rest.
#if defined(_WIN32)
            if (no_proc || no_fs || no_net) { sp.argv.push_back("--os-sandbox"); sp.argv.push_back("best-effort"); }
#else
            if (no_proc || no_fs || no_net) { sp.argv.push_back("--os-sandbox"); sp.argv.push_back("require"); }
#endif
            sp.argv.push_back("--args"); for (auto& x : argv) sp.argv.push_back(x);
            sp.stdin_data = src; sp.timeout_ms = t + 5000; sp.max_output = 32u << 20;
#if !FORGE_APP_SANITIZED
            sp.limits.memory_bytes = mem_mb << 20;
#endif   // (a sanitized child maps its shadow before main(): see FORGE_APP_SANITIZED; --mem-limit above still caps it)
            sp.env_unset = {"FORGE_SCRIPT_CACHE_SIGN_SEED", "FORGE_RPC_TOKEN", "FORGE_RPC_KEY", "FORGE_RPC_TLS_KEY"};
            auto r = os::spawn_wait(sp);
            if (!r) return graph::rpc::server::script_result{false, "RuntimeError", "could not start script child: " + r.err.what, ""};
            auto& pr = r.value;
            if (pr.timed_out) return graph::rpc::server::script_result{false, "TimeoutError", "script exceeded " + std::to_string(t) + " ms", pr.out};
            if (pr.exit_code == 0) return graph::rpc::server::script_result{true, "", "", pr.out};
            // child protocol: stderr = "<ErrorType>\n<message>"; a signal death or sandbox failure has none.
            // Log lines ("[forge:security W] ...", from the child's policy and lockdown) precede the protocol: skip them.
            std::string err = pr.err;
            while (err.rfind("[forge:", 0) == 0) { auto nl = err.find('\n'); err = nl == std::string::npos ? std::string() : err.substr(nl + 1); }
            std::string et = "RuntimeError", em = err.empty() ? pr.err : err;
            if (auto nl = err.find('\n'); nl != std::string::npos && nl > 0 && nl < 64 && err.find(' ') > nl) { et = err.substr(0, nl); em = err.substr(nl + 1); if (!et.empty() && et.back() == '\r') et.pop_back(); }   // CRLF on Windows
            if (pr.term_signal) { et = "RuntimeError"; em = "script child killed by signal " + std::to_string(pr.term_signal) + (pr.err.empty() ? "" : ": " + pr.err); }
            while (!em.empty() && em.back() == '\n') em.pop_back();
            return graph::rpc::server::script_result{false, et, em, pr.out};
        };
    }
    server_lockdown(pol, "forge rpc", bind + ":" + std::to_string(srv.port()));
    std::printf("forge rpc: listening on %s:%d (%s)%s%s%s%s%s%s\n", bind.c_str(), srv.port(), reg ? dev.c_str() : "cpu",
                srv.tls() ? " tls" : "", sec.tls.client_ca_file.empty() ? "" : " mtls", sec.client_map.empty() ? "" : " cert-map",
                sec.tls.crl_files.empty() && revoked.empty() ? "" : " revocation",
                sec.auth_required() ? " auth" : " NO-AUTH", sec.allow_scripts ? " scripts" : "");
    std::fflush(stdout);
    srv.run();
    return 0;
}

} // namespace

// FSS for `forge script` (docs/scripting.md "Security policy"). With an active policy (--policy F or the load_policy()
// search) the capabilities come from the policy (P1: net only if net.mode is not none, proc only with exec, env only
// with env) and the --no-* flags narrow the policy in turn; fs operations are path-scoped by the policy's fs.read /
// fs.write roots (enforce_scope in the runtime). integrity.scripts (INT-1/INT-2) is checked where the source is read:
// engine::run_file and imports verify `<file>.sig` on the bytes they parse, an rpc child verifies the source it got
// on stdin against the signature the client sent (--rpc-sig). Source without a file or signature (-e, the REPL)
// cannot pass "required" and is refused here. Then lockdown(policy) confines the process before any script code runs;
// it replaces options::os_sandbox. No active policy: nothing here changes anything.
void script_policy(forge::script::options& o, const std::string& cli_path, const std::string& what, bool verifiable) {
    auto p = security::load_policy(cli_path);
    if (!p.active()) return;
    if (p.integrity.scripts == security::requirement::required && !verifiable) {
        security::audit("integrity", security::decision::fail, "INT-4", what, "script: integrity.scripts is required and this source has no detached signature");
        throw security::integrity_error(what + ": integrity.scripts is required in " + p.source + ", and code given with -e, typed into the REPL or sent without a signature cannot be verified (run a signed file)", "INT-4");
    }
    if (!o.allow_fs) { p.fs.read.clear(); p.fs.write.clear(); }
    if (!o.allow_net) p.net.mode = security::net_mode::none;
    if (!o.allow_proc) p.exec = false;
    if (!o.allow_env) p.env = false;
    o.allow_fs = !(p.fs.read.empty() && p.fs.write.empty());
    o.allow_net = p.net.mode != security::net_mode::none;
    o.allow_proc = p.exec;
    o.allow_env = p.env;
    o.os_sandbox = forge::script::options::os_sandbox_mode::off;   // lockdown() below applies the OS confinement
    security::audit("script", security::decision::allow, "CAP-1", what,
                    std::string("script runs with fs=") + (o.allow_fs ? "scoped" : "off") + " net=" + security::name(p.net.mode) +
                    " proc=" + (o.allow_proc ? "on" : "off") + " env=" + (o.allow_env ? "on" : "off") + " (profile " + security::name(p.prof) + ")");
    security::lockdown(p);
}

// forge script --mem-limit, OS part: cap this process's memory the way spawn_spec::limits caps a child
// (os::apply_limits -> RLIMIT_AS; Windows: the job object of os::detail_proc::win_job). RLIMIT_AS counts address
// space and a job object counts committed memory, both including what the process already uses, so the cap is
// what is in use now plus `bytes` plus headroom (half of `bytes`, at least 64 MiB). The headroom is for what the
// managed-heap accounting does not count (allocator overhead and reserves, thread stacks, the interpreter's own
// structures): without it this cap would fire before the interpreter's, as a bare std::bad_alloc in the middle of
// native code instead of a clean OutOfMemoryError at an allocation the interpreter accounts for. Returns "" when
// applied, "unsupported" where no mechanism exists for a running process (macOS, iOS, Android and the BSDs: see
// docs/scripting.md "Resource limits"), else the failure.
std::string limit_own_memory(std::uint64_t bytes) {
    bytes += std::max<std::uint64_t>(bytes / 2, 64ull << 20);
#if defined(__linux__) && !defined(__ANDROID__)
    unsigned long long pages = 0;
    if (std::FILE* f = std::fopen("/proc/self/statm", "r")) { if (std::fscanf(f, "%llu", &pages) != 1) pages = 0; std::fclose(f); }
    if (!pages) return "cannot read /proc/self/statm";
    long page = ::sysconf(_SC_PAGESIZE);
    os::proc_limits lim; lim.memory_bytes = pages * std::uint64_t(page > 0 ? page : 4096) + bytes;
    return os::apply_limits(lim) ? std::string() : std::string("setrlimit(RLIMIT_AS) failed");
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{}; pmc.cb = sizeof pmc;
    std::uint64_t used = K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc) ? std::uint64_t(pmc.PagefileUsage) : 0;
    os::proc_limits lim; lim.memory_bytes = used + bytes;
    HANDLE job = os::detail_proc::win_job(lim, false);   // kept open for the life of the process
    if (!job) return "CreateJobObject failed";
    if (!AssignProcessToJobObject(job, GetCurrentProcess())) { CloseHandle(job); return "AssignProcessToJobObject failed"; }
    return {};
#else
    (void)bytes;
    return "unsupported";
#endif
}

// forge script [opts] <file> [args...] | -e <code> [args...] | (REPL)
int cmd_script(int argc, char** argv) {
    forge::script::options o;
    std::string file, code, emit_target, emit_out; bool have_code = false, os_sandbox_set = false, check_only = false, check_json = false; std::vector<std::string> module_dirs;
    std::string rpc_child, policy_path, rpc_sig_hex;
    std::string aot_out; forge::script::aot_options aot;   // --aot OUT: compile to an executable instead of running (docs/aot.md)
    int i = 2;
    for (; i < argc; ++i) {
        std::string s = argv[i];
        auto val = [&](const char* what) -> std::string { if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + what); return argv[++i]; };
        if (s == "-e") { code = val("-e"); have_code = true; ++i; break; }
        else if (s == "--rpc-child") { rpc_child = val("--rpc-child"); have_code = true; }   // rpc-serve: source on stdin
        else if (s == "--rpc-sig") rpc_sig_hex = val("--rpc-sig");   // rpc-serve: the client's detached signature (hex of the .sig text)
        else if (s == "--args") { ++i; break; }
        else if (s == "--emit") emit_target = val("--emit");
        else if (s == "-o") emit_out = val("-o");
        else if (s == "--aot") aot_out = val("--aot");
        else if (s == "--aot-mode") { auto m = val("--aot-mode"); aot.mode = m == "native" ? forge::script::aot_mode::native : m == "embed" ? forge::script::aot_mode::embed : m == "auto" ? forge::script::aot_mode::automatic : throw std::runtime_error("--aot-mode takes native|embed|auto"); }
        else if (s == "--cxx") aot.cxx = val("--cxx");
        else if (s == "--target") aot.target = val("--target");
        else if (s == "--aot-keep-source") aot.keep_source_dir = val("--aot-keep-source");
        else if (s == "--aot-runtime") aot.runtime_dir = val("--aot-runtime");
        else if (s == "--aot-stub") aot.stub = val("--aot-stub");
        else if (s == "--aot-allow-widen") aot.allow_widen = true;
        else if (s == "--aot-pch") aot.pch_dir = val("--aot-pch");
        else if (s == "--aot-cxx-flag") aot.cxx_flags.push_back(val("--aot-cxx-flag"));
        else if (s.size() > 2 && s.starts_with("-O")) aot.opt_level = s.substr(2);
        else if (s == "--emit-targets") { for (auto& t : forge::script::emit_targets()) std::printf("%s\n", t.c_str()); return 0; }
        else if (s == "--check") check_only = true;           // static checks only (docs/scripting.md "Static checks")
        else if (s == "--json") check_json = true;
        else if (s == "--module-dir") module_dirs.push_back(val("--module-dir"));   // extra import dir for --check
        else if (s == "--no-check") o.static_check = false;
        else if (s == "--ask-model") o.ask_model = val("--ask-model");   // in-process model for forge::ask & co
        else if (s == "--ask-max-tokens") o.ask_max_tokens = std::stoi(val("--ask-max-tokens"));
        else if (s == "--strict-check") o.strict_check = true;
        else if (s == "--tree") o.use_vm = false;
        else if (s == "--debug") o.debug = true;
        else if (s == "--timeout") o.timeout_ms = std::stoull(val("--timeout"));
        else if (s == "--max-steps") o.max_steps = std::stoull(val("--max-steps"));
        else if (s == "--heap") o.heap_bytes = std::stoull(val("--heap"));
        else if (s == "--mem-limit") {   // size string, as limit {mem: "2G"}
            auto v = val("--mem-limit"); std::string why;
            if (!forge::script::parse_size(v, o.mem_limit_bytes, &why)) throw std::runtime_error("--mem-limit: " + why);
            if (!o.mem_limit_bytes) throw std::runtime_error("--mem-limit: the limit must be greater than 0");
        }
        else if (s == "--no-proc") o.allow_proc = false;
        else if (s == "--cache-dir") o.fsc_cache_dir = val("--cache-dir");
        else if (s == "--cache-trust") o.fsc_trusted_keys.push_back(val("--cache-trust"));
        else if (s == "--cache-require-signed") o.fsc_require_signed = true;
        else if (s == "--cache-sign-seed") o.fsc_sign_seed = val("--cache-sign-seed");
        else if (s == "--policy") policy_path = val("--policy");
        else if (s == "--no-fs") o.allow_fs = false;
        else if (s == "--no-net") o.allow_net = false;
        else if (s == "--os-sandbox") { os_sandbox_set = true; auto m = val("--os-sandbox"); o.os_sandbox = m == "off" ? forge::script::options::os_sandbox_mode::off : m == "require" ? forge::script::options::os_sandbox_mode::require : m == "best-effort" ? forge::script::options::os_sandbox_mode::best_effort : throw std::runtime_error("--os-sandbox takes off|best-effort|require"); }
        else if (s == "--") { ++i; if (i < argc) file = argv[i++]; break; }
        else if (s.starts_with("--")) throw std::runtime_error("unknown script option " + s);
        else { file = s; ++i; break; }
    }
    for (; i < argc; ++i) o.argv.push_back(argv[i]);
    (void)os_sandbox_set;   // ForgeScript enforces cap flags at the OS level by default (options::os_sandbox)
    // AI-drafted code (forge::ai_run / ai_eval) runs only after an explicit "y" on the terminal.
    o.confirm = [](const std::string& code) {
        std::fprintf(stderr, "\nAI drafted:\n%s\nRun it? [y/N] ", code.c_str()); std::fflush(stderr);
        char buf[16] = {}; if (!std::fgets(buf, sizeof buf, stdin)) return false;
        return buf[0] == 'y' || buf[0] == 'Y';
    };
    if (!emit_target.empty()) for (std::size_t k = 0; k + 1 < o.argv.size(); ++k) if (o.argv[k] == "-o") { emit_out = o.argv[k + 1]; o.argv.erase(o.argv.begin() + k, o.argv.begin() + k + 2); break; }
    if (check_only) {
        if (!have_code) { if (file.empty()) throw std::runtime_error("--check needs a script file or -e"); std::ifstream f(file, std::ios::binary); if (!f) throw std::runtime_error("cannot read " + file); code.assign(std::istreambuf_iterator<char>(f), {}); }
        forge::script::check_options co{o.allow_fs, o.allow_net, o.allow_proc, o.allow_env, true, module_dirs};
        auto ds = forge::script::check(code, have_code ? "<-e>" : file, co);
        auto txt = check_json ? forge::script::to_json(ds) + "\n" : forge::script::to_text(ds);
        std::fwrite(txt.data(), 1, txt.size(), check_json ? stdout : stderr);
        return forge::script::has_errors(ds) || (o.strict_check && !ds.empty()) ? 2 : 0;
    }
    if (!aot_out.empty()) {   // ahead-of-time compilation: write an executable instead of running (docs/aot.md)
        if (!have_code) { if (file.empty()) throw std::runtime_error("--aot needs a script file or -e"); std::ifstream f(file, std::ios::binary); if (!f) throw std::runtime_error("cannot read " + file); code.assign(std::istreambuf_iterator<char>(f), {}); }
        aot.run = o; aot.run.argv.clear(); aot.policy_file = policy_path; aot.module_dirs = module_dirs; aot.self_is_runner = true;
        try {
            auto r = forge::script::aot_compile(code, have_code ? "<-e>" : file, aot_out, aot);
            if (!r.fallback_reason.empty()) std::fprintf(stderr, "forge script --aot: native mode not used: %s\n", r.fallback_reason.c_str());
            std::fprintf(stderr, "forge script --aot: wrote %s (%s)\n", r.output.c_str(), r.detail.c_str());
            return 0;
        } catch (const forge::script::aot_error& ex) { std::fprintf(stderr, "%s\n", ex.what()); return ex.exit_code; }
    }
    if (!emit_target.empty()) {   // write once, emit anywhere: translate instead of running
        if (!have_code) { if (file.empty()) throw std::runtime_error("--emit needs a script file or -e"); std::ifstream f(file, std::ios::binary); if (!f) throw std::runtime_error("cannot read " + file); code.assign(std::istreambuf_iterator<char>(f), {}); }
        try {
            auto text = forge::script::emit(code, have_code ? "<-e>" : file, emit_target);
            if (emit_out.empty()) std::fwrite(text.data(), 1, text.size(), stdout);
            else { std::ofstream f(emit_out, std::ios::binary); f << text; f.close(); std::filesystem::permissions(emit_out, std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec | std::filesystem::perms::others_exec, std::filesystem::perm_options::add); }
            return 0;
        } catch (const forge::script::emit_error& ex) { std::fprintf(stderr, "%s:%d: emit %s: %s\n", file.empty() ? "<-e>" : file.c_str(), ex.line, emit_target.c_str(), ex.what()); return 2; }
    }
    try { script_policy(o, policy_path, !rpc_child.empty() ? rpc_child : have_code ? std::string("<-e>") : file.empty() ? std::string("<repl>") : file,
                        !rpc_child.empty() ? !rpc_sig_hex.empty() : !have_code && !file.empty()); }
    catch (const security::security_error& e) {
        if (rpc_child.empty()) throw;
        // rpc-serve child protocol: stderr = "<ErrorType>\n<message>" (script classes, FSS-ERR-1)
        const char* t = e.sc == security::sec_code::integrity ? "IntegrityError" : e.sc == security::sec_code::policy ? "PolicyViolation"
                      : e.sc == security::sec_code::permission ? "PermissionError" : "SandboxViolationError";
        std::fprintf(stderr, "%s\n%s", t, e.what());
        return 1;
    }
    // --mem-limit at OS level: after the policy lockdown (which may set its own, lower RLIMIT_AS), before any script
    // code. The interpreter cap (options::mem_limit_bytes) applies either way.
    if (o.mem_limit_bytes)
        if (auto why = limit_own_memory(o.mem_limit_bytes); !why.empty() && why != "unsupported")
            std::fprintf(stderr, "forge script: --mem-limit: no OS-level cap on this process (%s); the interpreter cap still applies\n", why.c_str());
    if (!rpc_child.empty()) {
        code.assign(std::istreambuf_iterator<char>(std::cin), {});
        o.confirm = nullptr;   // no terminal: AI-drafted code never runs remotely
        try {   // INT-1/INT-2: the source as received, against the client's signature for the script's name
            std::vector<std::uint8_t> sig;
            if (!rpc_sig_hex.empty() && !security::detail::unhex(rpc_sig_hex, sig)) throw security::integrity_error(rpc_child + ": malformed --rpc-sig", "INT-4");
            security::verify_artefact(*security::active_policy(), security::artefact::script, rpc_child,
                                      {reinterpret_cast<const std::uint8_t*>(code.data()), code.size()}, std::string(sig.begin(), sig.end()), security::signed_name(rpc_child));
        } catch (const security::integrity_error& e) { std::fprintf(stderr, "IntegrityError\n%s", e.what()); return 1; }
        forge::script::engine e(o);
        auto r = e.run(code, rpc_child);
        std::fflush(stdout);
        if (!r.ok) { std::fprintf(stderr, "%s\n%s", r.error_type.empty() ? "RuntimeError" : r.error_type.c_str(), r.error.c_str()); return r.exit_code ? r.exit_code : 1; }
        return r.exit_code;
    }
    forge::script::engine e(o);
    if (!have_code && file.empty()) return e.repl();
    auto r = have_code ? e.run(code, "<-e>") : e.run_file(file);
    std::fflush(stdout);
    if (!r.ok) std::fprintf(stderr, "%s\n", r.error.c_str());
    return r.exit_code;
}

// ---- forge aot: ahead-of-time compilation of a model's plans (docs/aot-models.md) ----
int cmd_aot(const args& a) {
    if (a.pos.empty()) return usage();
    activate_policy(a.get("policy"));
    const std::string& in = a.pos[0];
    const std::string mode = a.get("aot-mode", "plan");
    if (a.has("list")) {
        modelfile::reader r(in);
        if (!r.has_section(modelfile::aot::section)) { std::printf("%s: no precompiled plans\n", in.c_str()); return 0; }
        graph::aot::image img(r.raw_section_bytes(modelfile::aot::section));
        bool match = img.model() == modelfile::aot::model_key(r);
        std::string why; bool covered = !r.has_section("signature") || modelfile::aot::covered_by_signature(r, &why);
        std::printf("%s: %zu precompiled plan(s), FORGE %s, pipeline %u, built on %s, %s%s\n", in.c_str(), img.entries().size(), img.forge().c_str(), img.pipeline(),
                    img.simd().c_str(), match ? "key matches this model and build" : "key does NOT match (other model, weights or FORGE version): ignored at load",
                    covered ? "" : (", NOT covered by the signature: ignored at load (" + why + ")").c_str());
        for (auto& e : img.entries()) std::printf("  %-28s %8llu bytes  key %s\n", e.t.name().c_str(), (unsigned long long)e.size, hash::blake3::hex(e.key).substr(0, 16).c_str());
        return 0;
    }
    if (a.has("strip")) {
        auto out = a.get("o").empty() ? in : a.get("o");
        bool dropped = modelfile::aot::write_section(in, out, {});
        std::printf("removed the precompiled plans -> %s%s\n", out.c_str(), dropped ? " (the signature was dropped: sign again)" : "");
        return 0;
    }
    if (a.kv.count("adapter")) raise(errc::unsupported, "aot: LoRA adapters are not compiled ahead of time; sessions that select an adapter compile their plans at run time");
    if (auto dev = a.get("device", "cpu"); dev != "cpu" && !dev.empty())
        raise(errc::unsupported, "aot: plans can only be precompiled for the CPU, not for device '{}' (a device plan holds kernels the backend creates per process)", dev);
    inference::aot_build_options bo;
    for (int n : parse_ids(a.get("shapes"))) bo.shapes.push_back(n);
    bo.all_rows = a.has("all-rows"); bo.context = i64(a.num("ctx", 0));
    { std::string k = a.get("kv-codec"); for (std::size_t p = 0, q; p < k.size(); p = q + 1) { q = k.find(',', p); if (q == std::string::npos) q = k.size(); if (q > p) bo.kv_codecs.push_back(k.substr(p, q - p)); } }
    if (mode == "native") {
        inference::aot_native_options no;
        no.emit = a.get("emit", "shared"); no.out = a.get("o");
        if (no.out.empty()) return usage();
        no.cxx = a.get("cxx"); no.target = a.get("target"); no.simd = a.get("simd"); no.march = a.get("march"); no.sdk = a.get("sdk"); no.work_dir = a.get("work");
        no.embed_weights = a.has("embed-weights"); no.extra_flags = a.all("cxxflag");
        if (a.kv.count("embed-limit-mb")) no.embed_limit = std::uint64_t(a.num("embed-limit-mb", 1024)) << 20;
        no.temperature = float(a.num("temp", 0)); no.top_k = int(a.num("top-k", 0)); no.top_p = float(a.num("top-p", 1)); no.max_new_tokens = int(a.num("n", 32));
        auto r = inference::aot_build_native(in, bo, no);
        std::printf("compiled %zu plan(s) (%zu steps as direct calls, %zu as kernel calls) with %s in %.1fs -> %s (%s, SIMD target %s, weights %s)\n",
                    r.plans.targets.size(), r.direct_steps, r.generic_steps, r.compiler.c_str(), r.compile_s, r.out.c_str(), no.emit.c_str(), r.simd.c_str(),
                    no.embed_weights ? "embedded" : ("in " + os::absolute(in)).c_str());
        if (!r.source.empty()) std::printf("source kept: %s\n", r.source.c_str());
        if (!r.same_compiler) std::fprintf(stderr, "forge aot: note: %s is not the compiler this FORGE was built with; the code was compiled with -ffp-contract=off, but bit-identical results are only tested for the build's compiler\n", r.compiler.c_str());
        return 0;
    }
    if (mode != "plan") raise(errc::invalid_argument, "aot: --aot-mode {} (plan or native)", mode);
    bool is_signed = false;
    { modelfile::reader r(in); is_signed = r.has_section("signature"); }
    const bool sidecar = a.has("sidecar");
    if (is_signed && !sidecar && a.get("key").empty() && !a.has("drop-signature"))
        raise(errc::invalid_argument, "aot: {} is signed and adding plans changes what the signature covers; pass --key name.key to sign the result, or --drop-signature", in);
    inference::aot_plans pl;
    { inference::model m(in); pl = inference::aot_compile_plans(m, bo); }
    std::string out = a.get("o");
    if (sidecar) {
        if (out.empty()) out = in + modelfile::aot::sidecar_ext;
        modelfile::aot::write_sidecar(out, pl.blob);
        if (is_signed) std::fprintf(stderr, "forge aot: note: %s is signed; a sidecar is not covered by its signature and is ignored when it is loaded\n", in.c_str());
    } else {
        if (out.empty()) out = in;
        modelfile::aot::write_section(in, out, pl.blob);
        if (!a.get("key").empty()) modelfile::sign_file(out, out, modelfile::load_secret_key(a.get("key")));
    }
    std::printf("precompiled %zu plan(s) (%zu nodes, %zu steps, %zu bytes) -> %s%s\n", pl.targets.size(), pl.nodes, pl.steps, pl.blob.size(), out.c_str(),
                sidecar ? " (sidecar)" : is_signed ? (a.get("key").empty() ? " (signature dropped)" : " (signed again)") : "");
    for (auto& t : pl.targets) std::printf("  %s\n", t.name().c_str());
    return 0;
}

int main(int argc, char** argv) {
    if (forge::script::run_process_worker_if_requested(argc, argv)) return 0;   // ProcessPool spawn-mode worker
    forge::script::install_local_ai();   // forge::ask & co run FORGE_ASK_MODEL / --ask-model in-process
    if (auto rc = forge::script::aot_run_embedded(argc, argv)) return *rc;   // a copy of forge carrying a script (--aot embed mode)
    if (argc < 2) return usage();
    std::string c = argv[1];
    if (c == "script") {
        try { return cmd_script(argc, argv); }
        catch (const std::exception& e) { std::fprintf(stderr, "forge script: %s\n", e.what()); return 2; }
    }
    try {
        auto a = parse(argc, argv, 2, {"verify", "tensors", "blueprint", "license", "allow-unused", "ids", "no-bos", "chat", "allow-scripts", "script-no-proc", "script-no-fs", "script-no-net", "quiet", "json", "detached", "lookup", "self-spec", "single-aead", "state", "calib", "no-calib", "pure", "bf16-head", "no-optimise"});
        if (c == "aot") return cmd_aot(parse(argc, argv, 2, {"all-rows", "sidecar", "strip", "list", "embed-weights", "drop-signature"}));
        if (c == "info") return cmd_info();
        if (c == "import") return cmd_import(a);
        if (c == "inspect") return cmd_inspect(a);
        if (c == "run") return cmd_run(a);
        if (c == "embed") return cmd_embed(a);
        if (c == "denoise") return cmd_denoise(a);
        if (c == "stream") return cmd_stream(a);
        if (c == "bench") return cmd_bench(a);
        if (c == "dump") return cmd_dump(a);
        if (c == "keygen") return cmd_keygen(a);
        if (c == "sign") { activate_policy(a.get("policy")); return cmd_sign(a); }   // FIO-2 on the secret key under a policy
        if (c == "verify") return cmd_verify(a);
        if (c == "encrypt") return cmd_encrypt(a);
        if (c == "adapter") return cmd_adapter(a);
        if (c == "shard") return cmd_shard(a);
        if (c == "blueprint") return cmd_blueprint(a);
        if (c == "requant") return cmd_requant(a);
        if (c == "compress" || c == "eval") {
            if (a.pos.empty()) return usage();
            compress::cli::opts co; co.in = a.pos[0]; co.out = a.get("o"); co.kv = a.kv;
            if (c == "eval") return compress::cli::eval(co);
            const bool dir = std::filesystem::is_directory(co.in);
            if (dir) {   // a checkpoint directory: import it in f32 first
                import::import_options io; io.codec = "f32";
                co.in = co.out + ".f32.tmp";
                import::import_hf(a.pos[0], co.in, io);
            }
            int rc = 1;
            try { rc = compress::cli::compress(co); } catch (...) { if (dir) os::remove(co.in); throw; }
            if (dir) os::remove(co.in);
            return rc;
        }
        if (c == "rpc-serve") return cmd_rpc_serve(a);
        if (c == "security") return cmd_security(a);
        return usage();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "forge: error: %s\n", e.what());
        return 1;
    }
}

#pragma once
// main() of a natively compiled model executable (`forge aot --aot-mode native --emit exe`, docs/aot-models.md):
// `forge run` for the one model the program was compiled from, with the same sampler, tokenizer, grammar and
// generation loop (inference::session::generate) and the same output.
//
//   ./model (--prompt TEXT | --tokens 1,2,3) [-n N] [--temp T] [--top-k K] [--top-p P] [--min-p P]
//           [--repeat-penalty R] [--seed S] [--ctx N] [--ids] [--no-bos] [--grammar file.gbnf] [--model file.fpk]
#include <forge/inference/inference.hpp>
#include <cstdio>
#include <fstream>
#include <set>

namespace forge::inference {

inline int aot_main(int argc, char** argv, const forge_aot_model_v1* d) {
    try {
        std::map<std::string, std::string> kv; std::set<std::string> flags;
        const std::set<std::string> boolean = {"ids", "no-bos", "help", "verify"};
        for (int i = 1; i < argc; ++i) {
            std::string s = argv[i];
            if (s.rfind("--", 0) != 0 && !(s.size() == 2 && s[0] == '-')) { std::fprintf(stderr, "%s: unexpected argument '%s'\n", argv[0], s.c_str()); return 2; }
            std::string k = s.substr(s[1] == '-' ? 2 : 1);
            if (auto eq = k.find('='); eq != std::string::npos) { kv[k.substr(0, eq)] = k.substr(eq + 1); continue; }
            if (boolean.count(k) || i + 1 >= argc) flags.insert(k); else kv[k] = argv[++i];
        }
        auto num = [&](const char* k, double dflt) { auto it = kv.find(k); return it == kv.end() ? dflt : std::stod(it->second); };
        if (flags.count("help") || (!kv.count("prompt") && !kv.count("tokens"))) {
            std::fprintf(stderr, "usage: %s (--prompt TEXT | --tokens 1,2,3) [-n %d] [--temp %g] [--top-k %d] [--top-p %g] [--min-p 0]\n"
                                 "       [--repeat-penalty 1] [--seed 0] [--ctx N] [--ids] [--no-bos] [--grammar file.gbnf] [--model file.fpk]\n"
                                 "compiled model: %s (FORGE %s, %s, %s)\n",
                         argv[0], d->max_new_tokens, double(d->temperature), d->top_k, double(d->top_p), d->fpk_data ? "embedded weights" : d->fpk_path, d->forge_version, d->simd_target, d->compiler);
            return 2;
        }
        auto lib = aot_native::library::in_image(d, forge::os::exe_path());
        std::string path = kv.count("model") ? kv["model"] : lib->embedded() ? std::string("<embedded>") : lib->fpk_path();
        session_options so; so.context = i64(num("ctx", 0)); so.verify = flags.count("verify") != 0;
        session s(std::make_shared<model>(path, so.verify, lib), so);
        s.check_generative();
        std::vector<int> prompt;
        if (kv.count("tokens")) {
            const std::string& t = kv["tokens"];
            for (std::size_t p = 0, q; p < t.size(); p = q + 1) { q = t.find(',', p); if (q == std::string::npos) q = t.size(); if (q > p) prompt.push_back(std::stoi(t.substr(p, q - p))); }
        } else {
            if (!s.tokenizer()) raise(errc::not_found, "model has no tokenizer; use --tokens");
            prompt = s.tokenizer()->encode(kv["prompt"]);
            if (s.tokenizer()->add_bos() && !flags.count("no-bos")) prompt.insert(prompt.begin(), s.tokenizer()->bos());
        }
        sampler_config sc;
        sc.temperature = float(num("temp", d->temperature)); sc.top_k = int(num("top-k", d->top_k)); sc.top_p = float(num("top-p", d->top_p));
        sc.min_p = float(num("min-p", 0)); sc.repetition_penalty = float(num("repeat-penalty", 1)); sc.seed = std::uint64_t(num("seed", 0));
        sampler smp(sc);
        bool ids = flags.count("ids") || !s.tokenizer();
        std::unique_ptr<grammar_constraint> gc;
        if (kv.count("grammar")) {
            if (!s.tokenizer()) raise(errc::unsupported, "--grammar needs a model tokenizer");
            std::ifstream gf(kv["grammar"], std::ios::binary);
            if (!gf) raise(errc::io_error, "cannot read {}", kv["grammar"]);
            std::string text((std::istreambuf_iterator<char>(gf)), std::istreambuf_iterator<char>());
            gc = std::make_unique<grammar_constraint>(std::make_shared<grammar>(text), *s.tokenizer(), int(s.vocab()), s.tokenizer()->eos());
            smp.filter = [&](std::span<float> l) { gc->apply(l); }; smp.allow = [&](int t) { return gc->allows(t); };
        }
        auto show = [&](int t) {
            if (gc && !gc->accept(t)) return false;
            if (ids) std::printf("%d ", t); else std::printf("%s", s.tokenizer()->decode_one(t).c_str());
            std::fflush(stdout);
            return !(gc && gc->finished());
        };
        generate_stats st;
        s.generate(prompt, int(num("n", d->max_new_tokens)), smp, show, &st);
        std::printf("\n");
        const auto& a = s.shared_model()->aot().stats();
        std::fprintf(stderr, "[%zu prompt tokens in %.3fs, %zu new tokens in %.3fs = %.1f tok/s; plans %zu compiled, %zu precompiled (%zu native), %zu reused]\n",
                     st.prompt_tokens, st.prefill_s, st.new_tokens, st.decode_s, st.new_tokens > 1 ? double(st.new_tokens - 1) / st.decode_s : 0.0,
                     s.plans().misses() - a.used, a.used, a.native, s.plans().hits());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: error: %s\n", argc ? argv[0] : "model", e.what());
        return 1;
    }
}

} // namespace forge::inference

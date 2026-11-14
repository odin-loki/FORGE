// The `forge compress`, `forge eval` and `forge import --compress` commands (apps/forge.cpp forwards to these).
#pragma once
#include <forge/compress/checkpoint.hpp>
#include <forge/compress/compress.hpp>
#include <forge/compress/eval.hpp>
#include <forge/compress/probes.hpp>

#include <fstream>
#include <iostream>

namespace forge::compress::cli {

struct opts {
    std::string in, out;
    std::map<std::string, std::string> kv;
    std::string get(const std::string& k, const std::string& d = "") const { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
    bool has(const std::string& k) const { return kv.count(k) != 0; }
};

inline std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> v; std::string cur;
    for (char ch : s) { if (ch == c) { if (!cur.empty()) v.push_back(cur); cur.clear(); } else cur += ch; }
    if (!cur.empty()) v.push_back(cur);
    return v;
}

inline void write_json(const std::string& path, const value& v) {
    if (path.empty()) { std::cout << forge::json::dump(v) << "\n"; return; }
    std::ofstream f(path); f << forge::json::dump(v) << "\n";
}

inline value eval_json(const EvalResult& r) {
    jobj acc; for (auto& [k, v] : r.acc) acc.push_back({k, value(v)});
    jobj n; for (auto& [k, v] : r.n) n.push_back({k, value(v)});
    jobj o{{"ppl", value(r.ppl)}, {"nll", value(r.nll)}, {"ppl_mix", value(r.ppl_mix)}, {"ppl_tokens", value(r.ppl_tokens)}, {"acc", value(acc)}, {"n", value(n)},
           {"mean_acc", value(r.mean_acc)}, {"seconds", value(r.seconds)}};
    if (!r.kv_codec.empty()) o.push_back({"kv_codec", value(r.kv_codec)});
    if (r.spec) {
        const auto& S = *r.spec;
        o.push_back({"speculative", value(jobj{{"prompts", value(S.prompts)}, {"new_tokens", value(S.new_tokens)}, {"acceptance", value(S.acceptance)},
                                               {"tokens_per_pass", value(S.tokens_per_pass)}, {"plain_tps", value(S.plain_tps)}, {"spec_tps", value(S.spec_tps)},
                                               {"speedup", value(S.speedup)}, {"plain_seconds", value(S.plain_seconds)}, {"spec_seconds", value(S.spec_seconds)},
                                               {"identical", value(S.identical)}, {"lossless", value(S.lossless())}})});
    }
    return value(o);
}

inline EvalCfg eval_cfg(const opts& a) {
    EvalCfg c;
    c.data_dir = a.get("data");
    c.ppl_windows = std::stoll(a.get("ppl-windows", "40"));
    c.ppl_window = std::stoll(a.get("ppl-window", "512"));
    c.tasks = split(a.get("tasks", ""), ',');
    c.task_limit = std::stoll(a.get("limit", "200"));
    c.mix = a.get("mix", "0") == "1";
    c.kv_codec = a.get("kv-codec");
    c.draft = a.get("draft");
    c.spec_tokens = std::stoll(a.get("spec-tokens", "64"));
    c.spec_prompts = std::stoll(a.get("spec-prompts", "8"));
    c.spec_prompt_len = std::stoll(a.get("spec-prompt-len", "128"));
    c.spec_k = std::stoi(a.get("spec-k", "4"));
    return c;
}

inline bool is_2of4(const tensor& t) { return compress::is_2of4(t.data.data(), t.data.size(), t.shape.back()); }
inline bool is_2of4(const wtensor& t) { return compress::is_2of4(t.data.data(), t.data.size(), t.shape.back()); }

// Codec for every 2-D+ weight whose rows fit the codec's block (others stay f32; norms and routers always f32).
// The 2:4 codecs (s24_f16 / s24_q8 / s24_q4) are used only on tensors that are 2:4 sparse; the others get the dense
// codec of the same value format (f16 / q8_0 / q4_0), so nothing is pruned by the storage format.
inline std::function<std::string(const std::string&, const wtensor&)> codec_picker(const std::string& codec) {
    if (codec.empty() || codec == "f32") return {};
    const auto* cd = forge::graph::codecs::find(codec);
    if (!cd) die("unknown codec " + codec);
    const i64 block = i64(cd->block);
    const bool s24 = codec.starts_with("s24_");
    const std::string dense = dense_sibling(codec);
    const i64 dblock = i64(forge::graph::codecs::find(dense)->block);
    return [=](const std::string&, const wtensor& t) {
        if (s24 && t.shape.back() % block == 0 && is_2of4(t)) return codec;
        if (t.shape.back() % dblock == 0) return dense;
        return std::string("f32");
    };
}

// Compresses the model file a.in into a.out with the recipe in a.kv["recipe"]; returns the JSON report.
// Checkpointing of a compress job (checkpoint.hpp): --checkpoint DIR ("none" = off; `forge compress` defaults to
// <out>.ckpt), --checkpoint-every N layers, --checkpoint-minutes M (default 10), --keep-checkpoint.
inline CheckpointCfg checkpoint_cfg(const opts& a) {
    CheckpointCfg c;
    const std::string d = a.get("checkpoint");
    if (d.empty() || d == "none" || d == "off" || d == "0") return c;
    c.dir = d;
    c.key = checkpoint_key(a.in, a.kv);
    c.every_layers = std::stoll(a.get("checkpoint-every", "0"));
    c.every_s = 60.0 * std::stod(a.get("checkpoint-minutes", "10"));
    return c;
}

inline Counts counts_of(const value& v) {
    Counts c; c.params = v["params"].i(); c.nonzero = v["nonzero"].i(); c.body_params = v["body_params"].i(); c.body_nonzero = v["body_nonzero"].i();
    return c;
}

// Compresses the model file a.in into a.out with the recipe in a.kv["recipe"]; returns the JSON report.
// With --checkpoint DIR the job saves its progress there and a rerun of the same command resumes it.
inline value compress_file(const opts& a) {
    const double t0 = now_s();
    const std::string recipe = a.get("recipe");
    if (a.in.empty() || a.out.empty() || recipe.empty()) die("compress needs <model>, -o OUT and --recipe RECIPE");
    // big buffers (mapped.hpp) go to scratch files next to the output unless FORGE_COMPRESS_SCRATCH says otherwise
    scratch::set_dir(os::dirname(os::absolute(a.out)));
    Checkpoint ck(checkpoint_cfg(a));
    if (ck.complete() && os::exists(a.out)) { say("checkpoint {}: {} is already done", ck.dir(), a.out); return ck.report(); }
    const i64 n = std::stoll(a.get("nsamples", "64")), seq = std::stoll(a.get("seqlen", "512"));
    // calibration: a dataset, or "self" (text the model writes itself: no data needed)
    const std::string src = a.get("calib-data", "self");
    const std::uint64_t seed = std::stoull(a.get("seed", "0"));
    // self: decoded by default; --self-mode jacobi [--jacobi-passes 6]: Jacobi prefill sampling (probes.hpp)
    const std::string self_mode = a.get("self-mode", "decode");
    const i64 passes = std::stoll(a.get("jacobi-passes", "6"));
    const bool need_ref = recipe.find("calibrate") != std::string::npos || recipe.find("kl=") != std::string::npos || recipe.find("optimi") != std::string::npos;
    Model m;
    Counts before;
    CalibSet calib;
    Reference ref;
    std::vector<value> stages;
    if (ck.resumed()) {
        const std::string mp = ck.model_path();
        say("checkpoint {}: resuming after {} finished stage(s) from {}", ck.dir(), ck.stages_done(), mp.empty() ? a.in : mp);
        m = Model::load(mp.empty() ? a.in : mp);
        m.src_path = a.in;
        ck.restore(m);
        before = counts_of(ck.before());
        calib = ck.calib();
        if (ck.has_ref()) ref = ck.reference();
        stages = ck.reports();
    } else {
        say("loading {}", a.in);
        m = Model::load(a.in);
        before = count(m);
        calib = src == "self" ? self_text(a.in, m, n, seq, seed, self_mode, passes) : make_calib(m, src, n, seq, seed);
        // reference distributions of the uncompressed model for feedback / calibrate: self-generated probes held
        // apart from the calibration text (different seed)
        if (need_ref) {
            const i64 np = std::stoll(a.get("probes", "8")), pl = std::stoll(a.get("probe-len", "256"));
            auto probes = self_text(a.in, m, np, pl, seed + 7919, self_mode, passes);
            ref = capture_reference(m, probes);
            say("reference: {} x {} self-generated probe tokens", probes.seqs.size(), probes.seq);
        }
        ck.start(m, calib, &ref, counts_json(before));
    }
    say("calibration: {} x {} tokens from {}", calib.seqs.size(), calib.seq, src);
    Ctx c{m, calib, {}, std::stod(a.get("damp", "0.01"))};
    if (!ref.empty()) c.ref = &ref;
    if (ck.enabled()) c.on_layer = [&](const LayerEvent& e) { return ck.on_layer(m, e); };
    const auto recipe_stages = parse_recipe(recipe);
    for (std::size_t i = std::size_t(ck.stages_done()); i < recipe_stages.size(); ++i) {
        ck.begin_stage(i64(i));
        stages.push_back(run_stage(c, recipe_stages[i]));
        ck.stage_done(i64(i), m, stages);
    }
    auto after = count(m);
    const std::string codec = a.get("codec", "f32");
    say("writing {} (codec {})", a.out, codec);
    m.save(a.out, codec_picker(codec));
    jobj rep{{"model", value(a.in)}, {"recipe", value(recipe)}, {"out", value(a.out)}, {"codec", value(codec)},
             {"calib", value(jobj{{"source", value(src)}, {"nsamples", value(n)}, {"seqlen", value(seq)},
                                  {"self_mode", value(src == "self" ? self_mode : std::string())}, {"jacobi_passes", value(passes)}})},
             {"before", counts_json(before)}, {"after", counts_json(after)},
             {"file_bytes", value(i64(std::filesystem::file_size(a.out)))}, {"stages", value(stages)},
             {"compress_seconds", value(now_s() - t0)}};
    if (ck.resumed()) rep.push_back({"resumed_after_stages", value(ck.stages_done())});
    say("done in {:.1f}s: params {} -> {}, nonzero {} -> {}", now_s() - t0, before.params, after.params, before.nonzero, after.nonzero);
    ck.mark_complete(value(rep));
    { Model gone = std::move(m); }   // release the checkpoint's model file before removing it (Windows)
    if (!a.has("keep-checkpoint") && !a.has("defer-checkpoint-cleanup")) ck.finish();
    return value(rep);
}

inline int compress(const opts& a) {
    opts b = a;   // `forge compress` checkpoints by default; the directory goes once the report is written
    if (!b.has("checkpoint")) b.kv["checkpoint"] = b.out + ".ckpt";
    b.kv["defer-checkpoint-cleanup"] = "1";
    auto rep = compress_file(b);
    if (a.has("data")) {
        say("evaluating {}", a.out);
        rep.obj_mut().push_back({"eval", eval_json(evaluate(a.out, eval_cfg(a)))});
    }
    write_json(a.get("report"), rep);
    if (const auto d = b.get("checkpoint"); !a.has("keep-checkpoint") && d != "none" && d != "off" && d != "0" && os::exists(d)) {
        if (auto s = os::remove_all(d); !s) say("checkpoint: could not remove {}: {}", d, s.err.what);
    }
    return 0;
}

inline int eval(const opts& a) {
    if (a.in.empty()) die("eval needs <model.fpk>");
    Model m = Model::load(a.in);
    auto cnt = count(m);
    auto r = evaluate(a.in, eval_cfg(a));
    jobj rep{{"model", value(a.in)}, {"counts", counts_json(cnt)}, {"file_bytes", value(i64(std::filesystem::file_size(a.in)))}, {"eval", eval_json(r)}};
    write_json(a.get("report"), value(rep));
    return 0;
}

}  // namespace forge::compress::cli

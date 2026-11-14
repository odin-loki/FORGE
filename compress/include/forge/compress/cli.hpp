// The `forge compress`, `forge eval` and `forge import --compress` commands (apps/forge.cpp forwards to these).
#pragma once
#include <forge/compress/compress.hpp>
#include <forge/compress/eval.hpp>

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
    return value(jobj{{"ppl", value(r.ppl)}, {"nll", value(r.nll)}, {"ppl_tokens", value(r.ppl_tokens)}, {"acc", value(acc)}, {"n", value(n)},
                      {"mean_acc", value(r.mean_acc)}, {"seconds", value(r.seconds)}});
}

inline EvalCfg eval_cfg(const opts& a) {
    EvalCfg c;
    c.data_dir = a.get("data");
    c.ppl_windows = std::stoll(a.get("ppl-windows", "40"));
    c.ppl_window = std::stoll(a.get("ppl-window", "512"));
    c.tasks = split(a.get("tasks", ""), ',');
    c.task_limit = std::stoll(a.get("limit", "200"));
    return c;
}

inline bool is_2of4(const tensor& t) { return compress::is_2of4(t.data.data(), t.data.size(), t.shape.back()); }

// Codec for every 2-D+ weight whose rows fit the codec's block (others stay f32; norms and routers always f32).
// The 2:4 codecs (s24_f16 / s24_q8 / s24_q4) are used only on tensors that are 2:4 sparse; the others get the dense
// codec of the same value format (f16 / q8_0 / q4_0), so nothing is pruned by the storage format.
inline std::function<std::string(const std::string&, const tensor&)> codec_picker(const std::string& codec) {
    if (codec.empty() || codec == "f32") return {};
    const auto* cd = forge::graph::codecs::find(codec);
    if (!cd) die("unknown codec " + codec);
    const i64 block = i64(cd->block);
    const bool s24 = codec.starts_with("s24_");
    const std::string dense = dense_sibling(codec);
    const i64 dblock = i64(forge::graph::codecs::find(dense)->block);
    return [=](const std::string&, const tensor& t) {
        if (s24 && t.shape.back() % block == 0 && is_2of4(t)) return codec;
        if (t.shape.back() % dblock == 0) return dense;
        return std::string("f32");
    };
}

// Compresses the model file a.in into a.out with the recipe in a.kv["recipe"]; returns the JSON report.
inline value compress_file(const opts& a) {
    const double t0 = now_s();
    const std::string recipe = a.get("recipe");
    if (a.in.empty() || a.out.empty() || recipe.empty()) die("compress needs <model>, -o OUT and --recipe RECIPE");
    say("loading {}", a.in);
    Model m = Model::load(a.in);
    auto before = count(m);
    const i64 n = std::stoll(a.get("nsamples", "64")), seq = std::stoll(a.get("seqlen", "512"));
    const std::string src = a.get("calib-data");
    if (src.empty()) die("--calib-data FILE is required (a .jsonl of {\"text\": ...} documents or a plain text file)");
    auto calib = make_calib(m, src, n, seq, std::stoull(a.get("seed", "0")));
    say("calibration: {} x {} tokens from {}", calib.seqs.size(), calib.seq, src);
    Ctx c{m, calib, {}, std::stod(a.get("damp", "0.01"))};
    std::vector<value> stages;
    for (auto& st : parse_recipe(recipe)) stages.push_back(run_stage(c, st));
    auto after = count(m);
    const std::string codec = a.get("codec", "f32");
    say("writing {} (codec {})", a.out, codec);
    m.save(a.out, codec_picker(codec));
    jobj rep{{"model", value(a.in)}, {"recipe", value(recipe)}, {"out", value(a.out)}, {"codec", value(codec)},
             {"calib", value(jobj{{"source", value(src)}, {"nsamples", value(n)}, {"seqlen", value(seq)}})},
             {"before", counts_json(before)}, {"after", counts_json(after)},
             {"file_bytes", value(i64(std::filesystem::file_size(a.out)))}, {"stages", value(stages)},
             {"compress_seconds", value(now_s() - t0)}};
    say("done in {:.1f}s: params {} -> {}, nonzero {} -> {}", now_s() - t0, before.params, after.params, before.nonzero, after.nonzero);
    return value(rep);
}

inline int compress(const opts& a) {
    auto rep = compress_file(a);
    if (a.has("data")) {
        say("evaluating {}", a.out);
        rep.obj_mut().push_back({"eval", eval_json(evaluate(a.out, eval_cfg(a)))});
    }
    write_json(a.get("report"), rep);
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

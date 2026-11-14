// Recipes: chains of compression stages applied to a Model, each stage calibrated on the model as left by the
// previous ones (layer by layer, with the calibration hidden states propagated through the already-compressed
// layers - the "sequential" scheme of SparseGPT, applied to every stage).
//
// Recipe syntax:  stage(key=value,...)+stage(...)   e.g.  "width(s=0.15,score=obs)+prune(method=sparsegpt,s=0.3,refit=mcr)"
//   prune      method=magnitude|wanda|ria|sparsegpt  s=<sparsity>  nm=2:4  alloc=uniform|owl|eba  refit=none|mcr
//              (nm + alloc=eba: partial N:M on the cheapest projections, frac=<fraction of prunable parameters>)
//              iters=<CG iterations>  scope=all|attn|ffn  max_s=<per-unit cap for eba>
//   width      s=<fraction of FFN neurons removed>  score=obs|act|mag  alloc=uniform|eba  round=<multiple>
//   depth      n=<sublayers removed>  unit=sublayer|layer|ffn|attn  score=probe|bi  shortlist=<k>  probe=<seqs>
//              replace=none|linear  rank=<r>
//   experts    drop=<experts per layer>  score=reap|freq|gate
//   merge      drop=<experts per layer>  method=ls|avg  score=reap|freq|gate
//   moe2dense  width=<fraction of E*f neurons kept>  max_err=<auto threshold, 1 = convert all>
//   quant      method=gptq|rtn  codec=q8_0|q4_0|q4_k|...  scope=all|attn|ffn  head=none|rtn (also quantise tok_emb / lm_head)
//              method=mixed: a codec per tensor from ladder=q4_k/q5_k/q6_k/q8_0 (low -> high), the fewest bytes whose
//              predicted error stays within slack x the error of uniform ref=q8_0; round=gptq|rtn; head=<codec>|none;
//              kl=<target>: measured KL feedback against the reference (halve the error budget until it holds)
//   rotate     seed=<n> heads=1  randomised Hadamard rotation of the residual stream (+ value heads); exact, data-free
//   equalize   alpha=1            SwiGLU up/down and V/O channel balancing; exact, data-free
//   calibrate  bias=1             output temperature + logit bias fitted to the reference distributions
//   optimise   codec=mixed|<codec> the default pipeline: rotate (dense models) + equalize + quant + calibrate
#pragma once
#include <forge/compress/moe.hpp>
#include <forge/compress/quant.hpp>
#include <forge/compress/reparam.hpp>
#include <forge/compress/probes.hpp>

namespace forge::compress {


struct Stage {
    std::string name;
    std::map<std::string, std::string> kv;
    std::string str(const std::string& k, const std::string& d) const { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
    double num(const std::string& k, double d) const { auto it = kv.find(k); return it == kv.end() ? d : std::stod(it->second); }
    std::string text() const { std::string s = name + "("; bool first = true; for (auto& [k, v] : kv) { s += (first ? "" : ",") + k + "=" + v; first = false; } return s + ")"; }
};

inline std::vector<Stage> parse_recipe(const std::string& r) {
    std::vector<Stage> out;
    std::size_t p = 0;
    while (p < r.size()) {
        while (p < r.size() && (r[p] == '+' || r[p] == ' ')) ++p;
        if (p >= r.size()) break;
        Stage s;
        auto q = r.find('(', p);
        auto plus = r.find('+', p);
        if (q == std::string::npos || (plus != std::string::npos && plus < q)) {   // bare name
            auto e = plus == std::string::npos ? r.size() : plus;
            s.name = r.substr(p, e - p); p = e; out.push_back(s); continue;
        }
        s.name = r.substr(p, q - p);
        auto e = r.find(')', q);
        if (e == std::string::npos) die("recipe: missing ')' in " + r);
        std::string args = r.substr(q + 1, e - q - 1);
        for (std::size_t a = 0; a < args.size();) {
            auto c = args.find(',', a); if (c == std::string::npos) c = args.size();
            auto kvs = args.substr(a, c - a);
            auto eq = kvs.find('=');
            if (eq == std::string::npos) die("recipe: expected key=value in " + kvs);
            s.kv[kvs.substr(0, eq)] = kvs.substr(eq + 1);
            a = c + 1;
        }
        out.push_back(s);
        p = e + 1;
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ accounting
struct Counts { i64 params = 0, nonzero = 0, body_params = 0, body_nonzero = 0; };
inline Counts count(const Model& m) {
    Counts c;
    std::set<std::string> used;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) for (auto& n : layer_tensors(m.spec, l)) used.insert(n);
    const bool tied = m.w.has("lm_head") && m.w.has("tok_emb") && m.w.at("lm_head").data == m.w.at("tok_emb").data;
    for (auto& n : used) {
        const auto& t = m.w.at(n);
        i64 nz = 0; for (float x : t.data) nz += x != 0.f;
        c.body_params += forge::numel(t.shape); c.body_nonzero += nz;
    }
    c.params = c.body_params; c.nonzero = c.body_nonzero;
    for (const char* n : {"tok_emb", "out_norm", "lm_head", "logit_bias", "head_mix"}) {
        if (std::string(n) == "lm_head" && tied) continue;
        if (!m.w.has(n)) continue;
        const auto& t = m.w.at(n);
        i64 nz = 0; for (float x : t.data) nz += x != 0.f;
        c.params += forge::numel(t.shape); c.nonzero += nz;
    }
    return c;
}

inline value counts_json(const Counts& c) {
    return value(jobj{{"params", value(c.params)}, {"nonzero", value(c.nonzero)}, {"body_params", value(c.body_params)}, {"body_nonzero", value(c.body_nonzero)}});
}

struct Ctx;
// Stage registry: modules (spectral.hpp, heads.hpp, ...) register their stages with
//   inline const bool reg_x = register_stage("name", fn);
using stage_fn = value (*)(Ctx&, const struct Stage&);
inline std::map<std::string, stage_fn>& stage_registry() { static std::map<std::string, stage_fn> r; return r; }
inline bool register_stage(const std::string& n, stage_fn f) { stage_registry()[n] = f; return true; }
// Allocation registry for prune(alloc=NAME): fn(model, calibration, criterion config, target sparsity, stage) -> unit -> sparsity
using alloc_fn = std::map<std::string, double> (*)(Ctx&, const struct PruneCfg&, double, const struct Stage&);
inline std::map<std::string, alloc_fn>& alloc_registry() { static std::map<std::string, alloc_fn> r; return r; }
inline bool register_alloc(const std::string& n, alloc_fn f) { alloc_registry()[n] = f; return true; }

// Per-layer progress of a sequential stage loop, for checkpoint / resume (checkpoint.hpp installs Ctx::on_layer).
// `phase` names the loop within its stage; `h` is the hidden state entering layer+1 (after the layer) and `carry`
// the loop's accumulators (its report entries), which a resumed run gets back.
struct LayerEvent { const char* phase; i64 layer; bool done; bool modified; Hidden* h; value* carry; };

struct Ctx {
    Model& m;
    const CalibSet& calib;
    std::vector<value> reports;
    double damp = 0.01;
    const Reference* ref = nullptr;   // the uncompressed model's distributions on probes (feedback / calibrate)
    std::function<bool(const LayerEvent&)> on_layer;   // empty: no checkpointing
};

// Checkpoint hooks of a sequential loop (no-ops without Ctx::on_layer):
//   for (i64 l = 0; l < n; ++l) {
//       if (layer_resumed(c, "prune", l, h, carry)) { <read the accumulators back from carry>; continue; }
//       ... compress layer l ...; layer_forward(m, l, h, nx); std::swap(h, nx);
//       carry = <the accumulators>; layer_finished(c, "prune", l, h, carry);
//   }
// layer_resumed: true when an interrupted earlier run finished layer l; its tensors, spec and codecs, the carry and
// (at the last finished layer) the hidden state have been restored, and the loop skips the layer.
// layer_finished: may write a checkpoint (every N layers / M minutes). `modified`: the loop changed layer l's
// tensors (else only the hidden state and carry are saved).
inline bool layer_resumed(Ctx& c, const char* phase, i64 l, Hidden& h, value& carry) {
    return c.on_layer && c.on_layer({phase, l, false, false, &h, &carry});
}
inline void layer_finished(Ctx& c, const char* phase, i64 l, Hidden& h, value& carry, bool modified = true) {
    if (c.on_layer) c.on_layer({phase, l, true, modified, &h, &carry});
}
// Doubles that must survive a checkpoint exactly (the JSON writer prints 9 digits): stored as hex-float strings.
inline value exact_num(double x) {
    std::string s = std::format("{:a}", std::abs(x));
    if (std::isfinite(x)) s = "0x" + s;
    return value(std::signbit(x) ? "-" + s : s);
}
inline double exact_of(const value& v) { return v.is_num() ? v.num() : std::strtod(v.str().c_str(), nullptr); }

// A linear layer view: a dense matrix, or one expert's slice of a stacked tensor.
struct LinView { std::string name; float* W; i64 rows, cols; const la::Gram* H; const std::vector<double>* diag; };

inline std::vector<LinView> layer_linears(Model& m, i64 l, LayerStats& S, const std::string& scope) {
    std::vector<LinView> v;
    const auto& L = m.spec.layers[std::size_t(l)];
    const std::string p = std::format("layer.{}.", l);
    auto add = [&](const std::string& leaf, const std::string& grp) {
        auto& t = m.w.at(p + leaf);
        auto git = S.gram.find(grp);
        v.push_back({p + leaf, t.data.data(), t.shape[0], t.shape[1], git == S.gram.end() ? nullptr : &git->second, &S.diag[grp]});
    };
    const bool attn_ok = scope == "all" || scope == "attn", ffn_ok = scope == "all" || scope == "ffn";
    auto addp = [&](const std::string& leaf, const std::string& grp) {   // a factored projection: its two factors
        if (!m.spec.factored(l, leaf.c_str())) { add(leaf, grp); return; }
        if (!m.spec.alias.count(p + leaf + "_a")) add(leaf + "_a", grp);
        if (!m.spec.alias.count(p + leaf + "_b")) add(leaf + "_b", "lr:" + leaf);
    };
    if (L.attn && attn_ok) { addp("wq", "attn"); addp("wk", "attn"); addp("wv", "attn"); addp("wo", "o"); }
    if (L.ffn == LayerSpec::Ffn::dense && ffn_ok) { addp("w_gate", "mlp"); addp("w_up", "mlp"); addp("w_down", "down"); }
    if (L.ffn == LayerSpec::Ffn::linear && ffn_ok) { add("lin_a", "lin"); }
    if (L.ffn == LayerSpec::Ffn::moe && ffn_ok) {
        for (const char* n : {"experts.w_gate", "experts.w_up", "experts.w_down"}) {
            auto& t = m.w.at(p + n);
            const i64 E = t.shape[0], r = t.shape[1], c = t.shape[2];
            for (i64 e = 0; e < E && e < i64(S.experts.size()); ++e) {
                auto& es = S.experts[std::size_t(e)];
                const bool down = std::string(n) == "experts.w_down";
                const la::Gram* g = down ? &es.hid : &es.in;
                v.push_back({std::format("{}{}#{}", p, n, e), t.data.data() + e * r * c, r, c, g->d ? g : nullptr, nullptr});
            }
        }
    }
    return v;
}

inline std::vector<double> diag_of(const la::Gram& g) { std::vector<double> d(static_cast<std::size_t>(g.d)); for (i64 i = 0; i < g.d; ++i) d[std::size_t(i)] = g.H(i, i); return d; }

// ------------------------------------------------------------------------------------------------ prune stage
inline value stage_prune(Ctx& c, const Stage& st) {
    Model& m = c.m;
    PruneCfg pc;
    pc.crit = criterion_from(st.str("method", "wanda"));
    pc.sparsity = st.num("s", 0.5);
    pc.damp = st.num("damp", c.damp);
    if (auto nm = st.str("nm", ""); !nm.empty()) { pc.nm_n = std::stoi(nm.substr(0, nm.find(':'))); pc.nm_m = std::stoi(nm.substr(nm.find(':') + 1)); pc.sparsity = 1.0 - double(pc.nm_n) / pc.nm_m; }
    const std::string alloc = st.str("alloc", "uniform"), refit = st.str("refit", "none"), scope = st.str("scope", "all");
    const int iters = int(st.num("iters", 25));
    const bool need_gram = pc.crit == Criterion::sparsegpt || refit == "mcr";
    jobj rep{{"stage", value(st.text())}};

    // layer-wise allocation from a dense pre-pass (diagonal statistics)
    std::map<std::string, double> unit_s;
    if (alloc == "eba" && pc.nm_m > 0) {
        // partial N:M: N:M only on the projections where it costs the least residual-relative error per parameter,
        // until `frac` of the prunable parameters are N:M (the rest stay dense)
        const double frac = st.num("frac", 0.5);
        struct Cand { std::string name; i64 params; double cost; };
        std::vector<Cand> cands;
        Hidden h = Hidden::embed(m, c.calib), nx;
        StatsReq rq; rq.gram = false; rq.moe = true;
        for (i64 l = 0; l < m.spec.n_layers(); ++l) {
            if (m.spec.layers[std::size_t(l)].ffn == LayerSpec::Ffn::moe) rq.gram = true, rq.only = {"__none__"};
            else rq.gram = false;
            auto S = layer_pass(m, l, h, rq, &nx);
            for (auto& lv : layer_linears(m, l, S, scope)) {
                std::vector<double> dg = lv.diag && !lv.diag->empty() ? *lv.diag : (lv.H ? diag_of(*lv.H) : std::vector<double>(static_cast<std::size_t>(lv.cols), 1.0));
                PruneCfg w = pc; if (w.crit == Criterion::sparsegpt) w.crit = Criterion::wanda;
                auto sal = saliency(lv.W, lv.rows, lv.cols, dg, w);
                auto mask = select_mask(sal, lv.W, lv.rows, lv.cols, w);
                double e = 0;
                for (i64 r = 0; r < lv.rows; ++r) for (i64 j = 0; j < lv.cols; ++j) if (!mask[std::size_t(r * lv.cols + j)]) { double x = lv.W[r * lv.cols + j]; e += x * x * dg[std::size_t(j)]; }
                cands.push_back({lv.name, lv.rows * lv.cols, e / std::max(S.resid_out, 1e-30)});
            }
            std::swap(h, nx);
        }
        double total = 0; for (auto& cd : cands) total += double(cd.params);
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.cost / double(a.params) < b.cost / double(b.params); });
        double have = 0;
        for (auto& cd : cands) { const bool take = have < frac * total; unit_s[cd.name] = take ? 1.0 : 0.0; if (take) have += double(cd.params); }
        std::vector<value> al;
        for (auto& [k, v] : unit_s) if (v > 0) al.push_back(value(k));
        rep.push_back({"nm_units", value(al)});
        rep.push_back({"nm_fraction", value(have / std::max(total, 1.0))});
    }
    if (alloc != "uniform" && pc.nm_m == 0) {
        auto grid = sparsity_grid(0.025, std::min(0.95, st.num("max_s", 0.9)));
        std::vector<Unit> units;
        Hidden h = Hidden::embed(m, c.calib), nx;
        StatsReq rq; rq.gram = false; rq.moe = true;
        for (i64 l = 0; l < m.spec.n_layers(); ++l) {
            if (m.spec.layers[std::size_t(l)].ffn == LayerSpec::Ffn::moe) rq.gram = true, rq.only = {"__none__"};
            else rq.gram = false;
            auto S = layer_pass(m, l, h, rq, &nx);
            for (auto& lv : layer_linears(m, l, S, scope)) {
                std::vector<double> dg = lv.diag ? *lv.diag : (lv.H ? diag_of(*lv.H) : std::vector<double>(static_cast<std::size_t>(lv.cols), 1.0));
                if (dg.empty()) dg.assign(static_cast<std::size_t>(lv.cols), 1.0);
                units.push_back(make_unit(lv.name, lv.W, lv.rows, lv.cols, dg, pc, grid, S.resid_out));
            }
            std::swap(h, nx);
        }
        if (auto it = alloc_registry().find(alloc); it != alloc_registry().end()) unit_s = it->second(c, pc, pc.sparsity, st);
        else if (alloc == "eba") unit_s = eba_allocate(units, grid, pc.sparsity, st.num("min_s", 0.0), st.num("max_s", 0.9));
        else if (alloc == "owl") {
            auto ls = owl_allocate(units, pc.sparsity, st.num("lambda", 0.08));
            for (auto& u : units) unit_s[u.name] = ls[u.layer];
        } else die("unknown alloc " + alloc);
        std::vector<value> al;
        for (auto& [k, v] : unit_s) al.push_back(value(jobj{{"unit", value(k)}, {"s", value(v)}}));
        rep.push_back({"allocation", value(al)});
    }

    Hidden h = Hidden::embed(m, c.calib), nx;
    std::vector<value> layers;
    double err_sum = 0; i64 err_n = 0;
    value carry;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        if (layer_resumed(c, "prune", l, h, carry)) { layers = carry["layers"].arr(); err_sum = exact_of(carry["err_sum"]); err_n = carry["err_n"].i(); continue; }
        StatsReq rq; rq.gram = need_gram; rq.moe = m.spec.layers[std::size_t(l)].ffn == LayerSpec::Ffn::moe;
        if (rq.moe) rq.gram = true;   // per-expert statistics need the routed Gram (diagonal at least)
        auto S = layer_pass(m, l, h, rq, nullptr);
        jobj lr{{"layer", value(l)}};
        for (auto& lv : layer_linears(m, l, S, scope)) {
            PruneCfg u = pc;
            if (auto it = unit_s.find(lv.name); it != unit_s.end()) {
                if (u.nm_m > 0) { if (it->second <= 0) continue; }   // partial N:M: this projection stays dense
                else u.sparsity = it->second;
            }
            if (u.sparsity <= 0 && u.nm_m == 0) continue;
            std::vector<float> W0(lv.W, lv.W + lv.rows * lv.cols);
            std::vector<double> dg = lv.diag && !lv.diag->empty() ? *lv.diag : (lv.H ? diag_of(*lv.H) : std::vector<double>(static_cast<std::size_t>(lv.cols), 1.0));
            const bool has_h = lv.H && lv.H->count > 0;
            if (u.crit == Criterion::sparsegpt && has_h) sparsegpt(lv.W, lv.rows, lv.cols, lv.H->H, u);
            else {
                PruneCfg us = u; if (us.crit == Criterion::sparsegpt) us.crit = Criterion::wanda;
                auto sal = saliency(lv.W, lv.rows, lv.cols, dg, us);
                auto mask = select_mask(sal, lv.W, lv.rows, lv.cols, us);
                apply_mask(lv.W, mask);
            }
            if (refit == "mcr" && has_h) {
                auto mask = mask_of(lv.W, std::size_t(lv.rows * lv.cols));
                auto rr = mcr_refit(lv.W, W0.data(), lv.rows, lv.cols, lv.H->H, mask, iters, pc.damp);
                err_sum += rr.err_after; ++err_n;
                lr.push_back({lv.name, value(jobj{{"err_before", value(rr.err_before)}, {"err_after", value(rr.err_after)}, {"iters", value(rr.iters)}})});
            } else if (has_h) {
                double e = rel_error(lv.W, W0.data(), lv.rows, lv.cols, lv.H->H);
                err_sum += e; ++err_n;
                lr.push_back({lv.name, value(jobj{{"err", value(e)}})});
            }
        }
        layer_forward(m, l, h, nx);
        std::swap(h, nx);
        layers.push_back(value(lr));
        say("prune layer {}/{} done", l + 1, m.spec.n_layers());
        carry = value(jobj{{"layers", value(layers)}, {"err_sum", exact_num(err_sum)}, {"err_n", value(err_n)}});
        layer_finished(c, "prune", l, h, carry);
    }
    rep.push_back({"layers", value(layers)});
    rep.push_back({"mean_rel_err", value(err_n ? err_sum / double(err_n) : 0.0)});
    return value(rep);
}

// ------------------------------------------------------------------------------------------------ width stage
inline bool is_sparse_model(const Model& m) {
    i64 z = 0, n = 0;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        const auto& L = m.spec.layers[std::size_t(l)];
        if (L.ffn != LayerSpec::Ffn::dense) continue;
        const auto& t = m.w.at(std::format("layer.{}.w_down", l));
        for (float x : t.data) { z += x == 0.f; ++n; }
    }
    return n && double(z) / double(n) > 0.25;
}

inline value stage_width(Ctx& c, const Stage& st) {
    Model& m = c.m;
    const double s = st.num("s", 0.2);
    const auto sc = width_score_from(st.str("score", "obs"));
    const std::string alloc = st.str("alloc", "uniform");
    const i64 round = i64(st.num("round", 64));
    const std::string sp = st.str("sparse", "auto");
    const bool sparse = sp == "auto" ? is_sparse_model(m) : sp == "1";
    jobj rep{{"stage", value(st.text())}, {"sparse_refit", value(sparse)}};

    std::map<i64, double> layer_s;
    if (alloc == "eba") {
        auto grid = sparsity_grid(0.025, std::min(0.9, st.num("max_s", 0.6)));
        std::vector<Unit> units;
        Hidden h = Hidden::embed(m, c.calib), nx;
        StatsReq rq; rq.gram = false;
        for (i64 l = 0; l < m.spec.n_layers(); ++l) {
            auto S = layer_pass(m, l, h, rq, &nx);
            std::swap(h, nx);
            if (m.spec.layers[std::size_t(l)].ffn != LayerSpec::Ffn::dense) continue;
            const auto& wd = m.w.at(std::format("layer.{}.w_down", l));
            const i64 d = wd.shape[0], f = wd.shape[1];
            const auto& dg = S.diag["down"];
            std::vector<double> sc_v(static_cast<std::size_t>(f));
            for (i64 j = 0; j < f; ++j) { double dn = 0; for (i64 r = 0; r < d; ++r) dn += double(wd.data[std::size_t(r * f + j)]) * wd.data[std::size_t(r * f + j)]; sc_v[std::size_t(j)] = dn * dg[std::size_t(j)]; }
            std::sort(sc_v.begin(), sc_v.end());
            Unit u; u.name = std::to_string(l); u.layer = l; u.params = 3 * d * f;
            std::vector<double> pre(static_cast<std::size_t>(f) + 1, 0); for (i64 j = 0; j < f; ++j) pre[std::size_t(j) + 1] = pre[std::size_t(j)] + sc_v[std::size_t(j)];
            for (double g : grid) u.err.push_back(pre[std::size_t(std::llround(g * double(f)))] / std::max(S.resid_out, 1e-30));
            units.push_back(u);
        }
        auto a = eba_allocate(units, grid, s, 0.0, st.num("max_s", 0.6));
        for (auto& [k, v] : a) layer_s[std::stoll(k)] = v;
    }

    Hidden h = Hidden::embed(m, c.calib), nx;
    std::vector<value> layers;
    value carry;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        if (layer_resumed(c, "width", l, h, carry)) { layers = carry["layers"].arr(); continue; }
        if (m.spec.layers[std::size_t(l)].ffn == LayerSpec::Ffn::dense && !m.spec.factored(l, "w_down") && !m.spec.factored(l, "w_gate") && !m.spec.factored(l, "w_up")) {
            StatsReq rq; rq.gram = true; rq.only = {"down"};
            auto S = layer_pass(m, l, h, rq, nullptr);
            const i64 f = m.w.at(std::format("layer.{}.w_down", l)).shape[1];
            const double ls = layer_s.count(l) ? layer_s[l] : s;
            i64 keep = i64(std::llround(double(f) * (1.0 - ls) / double(round))) * round;
            keep = std::clamp<i64>(keep, round, f);
            if (keep < f) {
                auto r = prune_ffn_width(m, l, S.gram.at("down").H, keep, sc, c.damp, sparse);
                layers.push_back(value(jobj{{"layer", value(l)}, {"ffn", value(f)}, {"kept", value(keep)}, {"rel_err", value(r.err)}}));
                say("width layer {}: {} -> {} neurons, rel err {:.4g}", l, f, keep, r.err);
            }
        }
        layer_forward(m, l, h, nx);
        std::swap(h, nx);
        carry = value(jobj{{"layers", value(layers)}});
        layer_finished(c, "width", l, h, carry);
    }
    rep.push_back({"layers", value(layers)});
    return value(rep);
}

// ------------------------------------------------------------------------------------------------ depth stage
inline double logprob_row(const float* row, i64 vocab, int t) {
    double mx = -INFINITY; for (i64 j = 0; j < vocab; ++j) mx = std::max(mx, double(row[j]));
    double s = 0; for (i64 j = 0; j < vocab; ++j) s += std::exp(double(row[j]) - mx);
    return double(row[t]) - mx - std::log(s);
}

inline double calib_nll(Runner& r, const CalibSet& c, i64 n_seq, i64 vocab) {
    double nll = 0; i64 cnt = 0;
    for (i64 i = 0; i < n_seq && i < i64(c.seqs.size()); ++i) {
        const auto& ids = c.seqs[std::size_t(i)];
        r.run_tokens(ids);
        auto lg = r.output();
        std::vector<double> part(ids.size() - 1);
        la::parallel_for(ids.size() - 1, 8, [&](std::size_t b, std::size_t e) {
            for (std::size_t t = b; t < e; ++t) part[t] = -logprob_row(lg.data() + i64(t) * vocab, vocab, ids[t + 1]);
        });
        for (double x : part) nll += x;
        cnt += i64(ids.size()) - 1;
    }
    return nll / double(std::max<i64>(cnt, 1));
}

inline value stage_depth(Ctx& c, const Stage& st) {
    Model& m = c.m;
    const i64 n = i64(st.num("n", 2));
    const std::string unit = st.str("unit", "sublayer"), score = st.str("score", "probe"), replace = st.str("replace", "none");
    const i64 shortlist = i64(st.num("shortlist", 8)), probe = i64(st.num("probe", 8)), rank = i64(st.num("rank", 256));
    jobj rep{{"stage", value(st.text())}};
    // block influence of every sublayer on the current model
    std::map<std::string, double> bi;
    {
        Hidden h = Hidden::embed(m, c.calib), nx;
        StatsReq rq; rq.gram = false; rq.sublayer = true;
        for (i64 l = 0; l < m.spec.n_layers(); ++l) {
            auto S = layer_pass(m, l, h, rq, &nx);
            std::swap(h, nx);
            const auto& L = m.spec.layers[std::size_t(l)];
            const double T = double(S.tokens);
            if (unit == "layer") { if (L.attn || L.ffn != LayerSpec::Ffn::none) bi[std::format("layer{}", l)] = 1 - S.cos_block / T; continue; }
            if (L.attn && unit != "ffn") bi[std::format("attn{}", l)] = 1 - S.cos_attn / T;
            if (L.ffn != LayerSpec::Ffn::none && unit != "attn") bi[std::format("ffn{}", l)] = 1 - S.cos_ffn / T;
        }
    }
    auto subs_of = [&](const std::string& k) {
        std::vector<Sublayer> v;
        if (k.rfind("layer", 0) == 0) {
            i64 l = std::stoll(k.substr(5));
            if (m.spec.layers[std::size_t(l)].attn) v.push_back({l, true});
            if (m.spec.layers[std::size_t(l)].ffn != LayerSpec::Ffn::none) v.push_back({l, false});
        } else if (k.rfind("attn", 0) == 0) v.push_back({std::stoll(k.substr(4)), true});
        else v.push_back({std::stoll(k.substr(3)), false});
        return v;
    };
    std::vector<std::string> chosen;
    std::vector<value> steps;
    if (score == "bi") {
        std::vector<std::pair<double, std::string>> v; for (auto& [k, x] : bi) v.push_back({x, k});
        std::sort(v.begin(), v.end());
        for (i64 i = 0; i < n && i < i64(v.size()); ++i) chosen.push_back(v[std::size_t(i)].second);
    } else {
        Runner r(m.spec, m.w, c.calib.seq, {}, true);
        const double base = calib_nll(r, c.calib, probe, m.spec.vocab);
        say("depth: calibration nll {:.4f}", base);
        std::map<std::string, std::vector<std::pair<Sublayer, std::vector<float>>>> zeroed;
        for (i64 step = 0; step < n; ++step) {
            std::vector<std::pair<double, std::string>> v;
            for (auto& [k, x] : bi) if (!zeroed.count(k)) v.push_back({x, k});
            std::sort(v.begin(), v.end());
            std::string best; double best_nll = INFINITY;
            for (i64 i = 0; i < shortlist && i < i64(v.size()); ++i) {
                const auto& k = v[std::size_t(i)].second;
                std::vector<std::pair<Sublayer, std::vector<float>>> saved;
                for (auto& s : subs_of(k)) saved.push_back({s, zero_sublayer(m, s)});
                const double x = calib_nll(r, c.calib, probe, m.spec.vocab);
                for (auto& [s, w] : saved) restore_sublayer(m, s, std::move(w));
                if (x < best_nll) { best_nll = x; best = k; }
            }
            if (best.empty()) break;
            std::vector<std::pair<Sublayer, std::vector<float>>> saved;
            for (auto& s : subs_of(best)) saved.push_back({s, zero_sublayer(m, s)});
            zeroed[best] = std::move(saved);
            chosen.push_back(best);
            steps.push_back(value(jobj{{"removed", value(best)}, {"nll", value(best_nll)}, {"bi", value(bi[best])}}));
            say("depth step {}: remove {} -> calibration nll {:.4f} (base {:.4f})", step + 1, best, best_nll, base);
        }
        for (auto& [k, sv] : zeroed) for (auto& [s, w] : sv) restore_sublayer(m, s, std::move(w));
        rep.push_back({"base_nll", value(base)});
    }
    // structural removal, layer by layer; FFNs may be replaced by fitted low-rank linear maps
    std::set<std::string> rm(chosen.begin(), chosen.end());
    std::vector<Sublayer> todo;
    for (auto& k : chosen) for (auto& s : subs_of(k)) todo.push_back(s);
    Hidden h = Hidden::embed(m, c.calib), nx;
    std::vector<value> repl;
    value carry;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        if (replace == "linear" && layer_resumed(c, "depth", l, h, carry)) { repl = carry["replaced"].arr(); continue; }
        bool attn_rm = false, ffn_rm = false;
        for (auto& s : todo) if (s.layer == l) (s.attn ? attn_rm : ffn_rm) = true;
        if (attn_rm) remove_sublayer(m, {l, true});
        if (ffn_rm) {
            if (replace == "linear" && m.spec.layers[std::size_t(l)].ffn != LayerSpec::Ffn::none) {
                StatsReq rq; rq.gram = false; rq.ffn_io = true;
                auto S = layer_pass(m, l, h, rq, nullptr);
                double expl = replace_ffn_linear(m, l, S, rank);
                repl.push_back(value(jobj{{"layer", value(l)}, {"rank", value(rank)}, {"explained", value(expl)}}));
                say("depth: layer {} FFN -> rank-{} linear ({:.3f} of output energy)", l, rank, expl);
            } else remove_sublayer(m, {l, false});
        }
        if (replace == "linear") {
            layer_forward(m, l, h, nx); std::swap(h, nx);
            carry = value(jobj{{"replaced", value(repl)}});
            layer_finished(c, "depth", l, h, carry);
        }
    }
    std::vector<value> ch; for (auto& k : chosen) ch.push_back(value(k));
    rep.push_back({"removed", value(ch)});
    rep.push_back({"steps", value(steps)});
    rep.push_back({"replaced", value(repl)});
    return value(rep);
}

// ------------------------------------------------------------------------------------------------ MoE stages
inline value stage_experts(Ctx& c, const Stage& st) {
    Model& m = c.m;
    const i64 drop = i64(st.num("drop", 1));
    const auto sc = expert_score_from(st.str("score", "reap"));
    jobj rep{{"stage", value(st.text())}};
    Hidden h = Hidden::embed(m, c.calib), nx;
    std::vector<value> layers;
    value carry;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        if (layer_resumed(c, "experts", l, h, carry)) { layers = carry["layers"].arr(); continue; }
        const bool moe = m.spec.layers[std::size_t(l)].ffn == LayerSpec::Ffn::moe;
        if (moe) {
            StatsReq rq; rq.gram = false; rq.moe = true;
            auto S = layer_pass(m, l, h, rq, nullptr);
            auto scores = expert_scores(S, sc);
            auto keep = prune_experts(m, l, S, drop, sc);
            std::vector<value> sv; for (double x : scores) sv.push_back(value(x));
            std::vector<value> kv; for (i64 x : keep) kv.push_back(value(x));
            layers.push_back(value(jobj{{"layer", value(l)}, {"scores", value(sv)}, {"kept", value(kv)}}));
        }
        layer_forward(m, l, h, nx);
        std::swap(h, nx);
        carry = value(jobj{{"layers", value(layers)}});
        layer_finished(c, "experts", l, h, carry, moe);
    }
    rep.push_back({"layers", value(layers)});
    return value(rep);
}

inline value stage_merge(Ctx& c, const Stage& st) {
    Model& m = c.m;
    const i64 drop = i64(st.num("drop", 1));
    const auto sc = expert_score_from(st.str("score", "reap"));
    const auto how = st.str("method", "ls") == "avg" ? MergeMethod::avg : MergeMethod::ls;
    jobj rep{{"stage", value(st.text())}};
    Hidden h = Hidden::embed(m, c.calib), nx;
    std::vector<value> layers;
    value carry;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        if (layer_resumed(c, "merge", l, h, carry)) { layers = carry["layers"].arr(); continue; }
        const bool moe = m.spec.layers[std::size_t(l)].ffn == LayerSpec::Ffn::moe;
        if (moe) {
            StatsReq rq; rq.gram = false; rq.moe = true; rq.moe_means = true;
            auto S = layer_pass(m, l, h, rq, nullptr);
            auto plan = plan_merges(S, drop, sc);
            auto fit = merge_pass(m, l, h, plan);
            auto errs = merge_experts(m, l, plan, how, &fit);
            std::vector<value> pv;
            for (std::size_t q = 0; q < plan.size(); ++q) pv.push_back(value(jobj{{"absorbed", value(plan[q].first)}, {"into", value(plan[q].second)}, {"fit_err", value(errs[q])}}));
            layers.push_back(value(jobj{{"layer", value(l)}, {"merges", value(pv)}}));
            say("merge layer {}: {} merges", l, plan.size());
        }
        layer_forward(m, l, h, nx);
        std::swap(h, nx);
        carry = value(jobj{{"layers", value(layers)}});
        layer_finished(c, "merge", l, h, carry, moe);
    }
    rep.push_back({"layers", value(layers)});
    return value(rep);
}

inline value stage_moe2dense(Ctx& c, const Stage& st) {
    Model& m = c.m;
    const double width = st.num("width", 0.5), max_err = st.num("max_err", 1.0);
    const i64 round = i64(st.num("round", 64));
    jobj rep{{"stage", value(st.text())}};
    Hidden h = Hidden::embed(m, c.calib), nx;
    std::vector<value> layers;
    value carry;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        if (layer_resumed(c, "moe2dense", l, h, carry)) { layers = carry["layers"].arr(); continue; }
        const bool moe = m.spec.layers[std::size_t(l)].ffn == LayerSpec::Ffn::moe;
        if (moe) {
            StatsReq rq; rq.gram = false; rq.moe_dense_acts = true;
            auto S = layer_pass(m, l, h, rq, nullptr);
            const auto& wg = m.w.at(std::format("layer.{}.experts.w_gate", l));
            const i64 n = wg.shape[0] * wg.shape[1];
            const i64 keep = std::clamp<i64>(i64(std::llround(double(n) * width / double(round))) * round, round, n);
            // try, and keep the MoE layer if the dense fit is worse than max_err
            auto saved_spec = m.spec.layers[std::size_t(l)];
            std::map<std::string, wtensor> saved;
            for (auto& nm : layer_tensors(m.spec, l)) saved[nm] = m.w.at(nm);
            const double err = moe_to_dense(m, l, S, keep);
            bool ok = err <= max_err;
            if (!ok) {
                for (auto& nm : layer_tensors(m.spec, l)) m.w.t.erase(nm);
                m.spec.layers[std::size_t(l)] = saved_spec;
                for (auto& [k, v] : saved) m.w.t[k] = v;
            }
            layers.push_back(value(jobj{{"layer", value(l)}, {"width", value(keep)}, {"rel_err", value(err)}, {"converted", value(ok)}}));
            say("moe2dense layer {}: {} neurons, rel err {:.4f} {}", l, keep, err, ok ? "converted" : "kept as MoE");
        }
        layer_forward(m, l, h, nx);
        std::swap(h, nx);
        carry = value(jobj{{"layers", value(layers)}});
        layer_finished(c, "moe2dense", l, h, carry, moe);
    }
    rep.push_back({"layers", value(layers)});
    return value(rep);
}

// ------------------------------------------------------------------------------------------------ quant stage
inline double abs_error(const float* W, const float* W0, i64 rows, i64 cols, const std::vector<float>& Hf) {
    std::vector<float> D(static_cast<std::size_t>(rows * cols)), DH(D.size());
    for (std::size_t i = 0; i < D.size(); ++i) D[i] = W0[i] - W[i];
    la::gemm_nt(D.data(), Hf.data(), DH.data(), rows, cols, cols);
    double a = 0; for (std::size_t i = 0; i < D.size(); ++i) a += double(D[i]) * DH[i];
    return a;
}

// Mixed precision: per tensor, the predicted residual-relative output error of every ladder codec (RTN), then the
// cheapest assignment within the error budget; applied with GPTQ (or RTN); optionally checked against the
// reference distributions and tightened until the measured KL meets its target.
inline value stage_quant_mixed(Ctx& c, const Stage& st) {
    Model& m = c.m;
    std::vector<std::string> ladder;
    { std::string l = st.str("ladder", "q4_k/q5_k/q6_k/q8_0"); for (std::size_t p = 0; p < l.size();) { auto q = l.find('/', p); if (q == std::string::npos) q = l.size(); ladder.push_back(l.substr(p, q - p)); p = q + 1; } }
    const std::string refc = st.str("ref", "q8_0"), round = st.str("round", "gptq"), head = st.str("head", "q8_0"), scope = st.str("scope", "all");
    double slack = st.num("slack", 1.0);
    const double kl_target = st.num("kl", 0.0);
    const int max_fb = int(st.num("feedback", 4));
    const bool sparse = is_sparse_model(m);
    auto eff = [&](std::string cn, const wtensor& t) {   // block-32 sibling when rows do not fit a 256 block; 2:4 codecs on 2:4 tensors
        if (t.shape.back() % 256 != 0) { if (cn == "q4_k") cn = "q4_0"; else if (cn == "q5_k") cn = "q5_0"; else if (cn == "q6_k" || cn == "q3_k" || cn == "q2_k") cn = cn == "q6_k" ? "q8_0" : "q4_0"; }
        if (sparse && t.shape.back() % 32 == 0 && is_2of4(t.data.data(), t.data.size(), t.shape.back())) {
            if (cn == "q8_0" || cn == "q6_k") return std::string("s24_q8");
            if (cn == "q4_k" || cn == "q5_k" || cn == "q4_0") return std::string("s24_q4");
            if (cn == "f16") return std::string("s24_f16");
        }
        return cn;
    };
    struct UnitQ { std::string name; std::vector<std::string> codec; std::vector<double> err; std::vector<double> bytes; std::size_t pick = 0; };
    std::vector<UnitQ> units;
    std::map<std::string, std::size_t> uidx;
    // checkpoint carry of pass 1: the error curves so far
    auto pack_units = [&] {
        std::vector<value> us;
        for (auto& u : units) {
            std::vector<value> cs, es, bs;
            for (std::size_t k = 0; k < u.codec.size(); ++k) { cs.push_back(value(u.codec[k])); es.push_back(exact_num(u.err[k])); bs.push_back(exact_num(u.bytes[k])); }
            us.push_back(value(jobj{{"name", value(u.name)}, {"codec", value(cs)}, {"err", value(es)}, {"bytes", value(bs)}}));
        }
        return value(jobj{{"units", value(us)}});
    };
    auto unpack_units = [&](const value& v) {
        units.clear(); uidx.clear();
        for (auto& x : v["units"].arr()) {
            UnitQ u; u.name = x["name"].str();
            for (std::size_t k = 0; k < x["codec"].arr().size(); ++k) { u.codec.push_back(x["codec"][k].str()); u.err.push_back(exact_of(x["err"][k])); u.bytes.push_back(exact_of(x["bytes"][k])); }
            uidx[u.name] = units.size(); units.push_back(u);
        }
    };
    // 1. predicted errors on the current model (one sequential pass, unquantised upstream)
    {
        Hidden h = Hidden::embed(m, c.calib), nx;
        value carry;
        for (i64 l = 0; l < m.spec.n_layers(); ++l) {
            if (layer_resumed(c, "mixed.err", l, h, carry)) { unpack_units(carry); continue; }
            StatsReq rq; rq.gram = true; rq.moe = m.spec.layers[std::size_t(l)].ffn == LayerSpec::Ffn::moe;
            auto S = layer_pass(m, l, h, rq, &nx);
            const double R = std::max(S.resid_out, 1e-30);
            for (auto& lv : layer_linears(m, l, S, scope)) {
                const std::string tname = lv.name.substr(0, lv.name.find('#'));
                const auto& t = m.w.at(tname);
                if (!uidx.count(tname)) {
                    UnitQ u; u.name = tname;
                    for (auto& cn : ladder) {
                        auto e = eff(cn, t);
                        if (!codec_fits(codec_of(e), t.shape.back())) continue;
                        u.codec.push_back(e); u.err.push_back(0); u.bytes.push_back(double(codec_of(e).bytes(static_cast<std::size_t>(forge::numel(t.shape)))));
                    }
                    if (u.codec.empty()) continue;
                    uidx[tname] = units.size(); units.push_back(u);
                }
                auto& u = units[uidx[tname]];
                if (!lv.H || lv.H->count == 0) continue;
                std::vector<float> Hf = la::to_f32(lv.H->H);
                for (std::size_t k = 0; k < u.codec.size(); ++k) {
                    std::vector<float> Wq(lv.W, lv.W + lv.rows * lv.cols);
                    rtn_quantise(Wq.data(), lv.rows, lv.cols, codec_of(u.codec[k]), true);
                    u.err[k] += abs_error(Wq.data(), lv.W, lv.rows, lv.cols, Hf) / R;
                }
            }
            std::swap(h, nx);
            say("mixed: error curves for layer {}/{}", l + 1, m.spec.n_layers());
            carry = pack_units();
            layer_finished(c, "mixed.err", l, h, carry, false);
        }
    }
    double ref_err = 0;
    for (auto& u : units) {
        auto it = std::find(u.codec.begin(), u.codec.end(), eff(refc, m.w.at(u.name)));
        ref_err += it != u.codec.end() ? u.err[std::size_t(it - u.codec.begin())] : u.err.back();
    }
    // keep the pre-quantisation weights for feedback rounds
    std::map<std::string, fbuf> orig;
    for (auto& u : units) orig[u.name] = m.w.at(u.name).data;
    for (const char* n : {"tok_emb", "lm_head"}) orig[n] = m.w.at(n).data;
    std::vector<value> rounds;
    // allocation (Lagrangian): every tensor picks argmin bytes + lambda * err on its own; lambda is bisected so the
    // total predicted error meets the budget (the convex-hull optimum of the knapsack)
    auto allocate = [&](double sl) {
        const double budget = ref_err * sl;
        auto assign = [&](double lam) {
            double e = 0;
            for (auto& u : units) {
                std::size_t best = 0; double bc = INFINITY;
                for (std::size_t k = 0; k < u.codec.size(); ++k) { const double cst = u.bytes[k] + lam * u.err[k]; if (cst < bc) { bc = cst; best = k; } }
                u.pick = best; e += u.err[best];
            }
            return e;
        };
        double lo = 0, hi = 1;
        while (assign(hi) > budget && hi < 1e30) hi *= 10;
        for (int it = 0; it < 80; ++it) { const double mid = std::sqrt(std::max(lo, 1e-30) * hi); if (assign(mid) > budget) lo = mid; else hi = mid; }
        return assign(hi);
    };
    auto restore = [&] {
        for (auto& u : units) m.w.at(u.name).data = orig[u.name];
        for (const char* n : {"tok_emb", "lm_head"}) m.w.at(n).data = orig[n];
        m.codec.clear();
    };
    auto quant_head = [&] {
        if (head == "none") return;
        for (const char* n : {"tok_emb", "lm_head"}) {
            auto& t = m.w.at(n);
            const auto& cd = codec_of(head);
            if (!codec_fits(cd, t.shape.back())) continue;
            rtn_quantise(t.data.data(), t.shape[0], t.shape[1], cd, false);
            m.codec[n] = head;
        }
    };
    auto apply = [&](bool gptq) {
        restore();
        Hidden h = Hidden::embed(m, c.calib), nx;
        value carry;
        for (i64 l = 0; l < m.spec.n_layers(); ++l) {
            if (gptq && layer_resumed(c, "mixed.apply", l, h, carry)) continue;
            StatsReq rq; rq.gram = gptq; rq.moe = m.spec.layers[std::size_t(l)].ffn == LayerSpec::Ffn::moe;
            if (rq.moe && gptq) rq.gram = true;
            LayerStats S;
            if (gptq) S = layer_pass(m, l, h, rq, nullptr);
            for (auto& lv : layer_linears(m, l, S, scope)) {
                const std::string tname = lv.name.substr(0, lv.name.find('#'));
                if (!uidx.count(tname)) continue;
                const auto& u = units[uidx[tname]];
                if (u.codec[u.pick] == "f32") continue;
                const auto& cd = codec_of(u.codec[u.pick]);
                if (gptq && lv.H && lv.H->count > 0) gptq_quantise(lv.W, lv.rows, lv.cols, lv.H->H, cd, c.damp, true);
                else rtn_quantise(lv.W, lv.rows, lv.cols, cd, true);
                m.codec[tname] = u.codec[u.pick];
            }
            if (gptq) { layer_forward(m, l, h, nx); std::swap(h, nx); layer_finished(c, "mixed.apply", l, h, carry); }
        }
        quant_head();
    };
    auto summary = [&](double sl, double tot, double klv) {
        double bytes = 0, params = 0;
        std::map<std::string, i64> hist;
        for (auto& u : units) { bytes += u.bytes[u.pick]; params += double(forge::numel(m.w.at(u.name).shape)); hist[u.codec[u.pick]] += forge::numel(m.w.at(u.name).shape); }
        jobj hj; for (auto& [k, v] : hist) hj.push_back({k, value(v)});
        say("mixed: slack {:.3g}: {:.3f} bits/weight, predicted err {:.4g} (uniform {} {:.4g}), KL {:.5f}", sl, 8 * bytes / std::max(params, 1.0), tot, refc, ref_err, klv);
        return value(jobj{{"slack", value(sl)}, {"predicted_err", value(tot)}, {"ref_err", value(ref_err)}, {"body_bits_per_weight", value(8 * bytes / std::max(params, 1.0))}, {"codecs", value(hj)}, {"kl", value(klv)}});
    };
    // feedback: with a measured KL target, search the largest error budget whose (RTN) model meets it
    const bool fb = kl_target > 0 && c.ref && !c.ref->empty();
    if (fb) {
        double good = 0, bad = 0, prev_bpw = -1;
        for (int it = 0; it < max_fb + 6; ++it) {
            const double sl = good == 0 && bad == 0 ? slack : bad == 0 ? good * 4 : good == 0 ? bad / 4 : std::sqrt(good * bad);
            const double tot = allocate(sl);
            apply(false);
            const double klv = kl(m, *c.ref);
            rounds.push_back(summary(sl, tot, klv));
            (klv <= kl_target ? good : bad) = sl;
            const double bpw = rounds.back()["body_bits_per_weight"].num();
            if (bad == 0 && it > 0 && std::abs(bpw - prev_bpw) < 1e-9) break;   // the cheapest allocation is reached
            prev_bpw = bpw;
            if (good > 0 && bad > 0 && bad / good < 1.5) break;
            if (bad == 0 && good > 1e4) break;
            if (good == 0 && bad < 1e-3) break;
        }
        slack = good > 0 ? good : (bad > 0 ? bad / 4 : slack);
    }
    const double tot = allocate(slack);
    double klv = -1, kl_rtn = -1;
    if (round == "gptq" && c.ref && !c.ref->empty()) { apply(false); kl_rtn = kl(m, *c.ref); }
    apply(round == "gptq");
    if (c.ref && !c.ref->empty()) klv = kl(m, *c.ref);
    std::string rounding = round;
    if (kl_rtn >= 0 && klv > kl_rtn) {   // feedback: GPTQ on scarce calibration statistics can overfit - keep the better
        apply(false); klv = kl_rtn; rounding = "rtn (measured better than gptq)";
    }
    rounds.push_back(summary(slack, tot, klv));
    say("mixed: final rounding {}", rounding);
    std::vector<value> al;
    for (auto& u : units) al.push_back(value(jobj{{"tensor", value(u.name)}, {"codec", value(u.codec[u.pick])}}));
    return value(jobj{{"stage", value(st.text())}, {"rounds", value(rounds)}, {"allocation", value(al)}});
}

inline value stage_quant(Ctx& c, const Stage& st) {
    if (st.str("method", "gptq") == "mixed") return stage_quant_mixed(c, st);
    Model& m = c.m;
    const std::string method = st.str("method", "gptq"), scope = st.str("scope", "all"), cname = st.str("codec", "q8_0");
    const auto& cd = codec_of(cname);
    const bool gptq = method == "gptq";
    if (!gptq && method != "rtn") die("quant: method gptq or rtn");
    jobj rep{{"stage", value(st.text())}};
    Hidden h = Hidden::embed(m, c.calib), nx;
    double err_sum = 0; i64 err_n = 0, quantised = 0, skipped = 0;
    std::set<std::string> done_stacked;
    value carry;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        if (gptq && layer_resumed(c, "quant", l, h, carry)) {
            err_sum = exact_of(carry["err_sum"]); err_n = carry["err_n"].i(); quantised = carry["quantised"].i(); skipped = carry["skipped"].i();
            continue;
        }
        StatsReq rq; rq.gram = gptq; rq.moe = m.spec.layers[std::size_t(l)].ffn == LayerSpec::Ffn::moe;
        if (rq.moe) rq.gram = true;
        LayerStats S;
        if (gptq) S = layer_pass(m, l, h, rq, nullptr);
        for (auto& lv : layer_linears(m, l, S, scope)) {
            const std::string tname = lv.name.substr(0, lv.name.find('#'));
            // a 2:4 codec only where the weights are 2:4 sparse (stacked experts: the whole tensor must be)
            std::string use = cname;
            if (cname.starts_with("s24_")) {
                const auto& t = m.w.at(tname);
                if (!is_2of4(t.data.data(), t.data.size(), t.shape.back())) use = dense_sibling(cname);
            }
            const auto& cd = codec_of(use);
            if (!codec_fits(cd, lv.cols)) { ++skipped; continue; }
            std::vector<float> W0(lv.W, lv.W + lv.rows * lv.cols);
            const bool has_h = lv.H && lv.H->count > 0;
            if (gptq && has_h) gptq_quantise(lv.W, lv.rows, lv.cols, lv.H->H, cd, c.damp, true);
            else rtn_quantise(lv.W, lv.rows, lv.cols, cd, true);
            if (has_h) { err_sum += rel_error(lv.W, W0.data(), lv.rows, lv.cols, lv.H->H); ++err_n; }
            m.codec[tname] = use;
            ++quantised;
        }
        if (gptq) { layer_forward(m, l, h, nx); std::swap(h, nx); }
        if (gptq) {
            say("quant layer {}/{} done", l + 1, m.spec.n_layers());
            carry = value(jobj{{"err_sum", exact_num(err_sum)}, {"err_n", value(err_n)}, {"quantised", value(quantised)}, {"skipped", value(skipped)}});
            layer_finished(c, "quant", l, h, carry);
        }
    }
    if (st.str("head", "none") == "rtn") {
        for (const char* n : {"tok_emb", "lm_head"}) {
            auto& t = m.w.at(n);
            if (!codec_fits(cd, t.shape.back())) continue;
            const bool tied = std::string(n) == "lm_head" && m.w.at("tok_emb").data == t.data;
            rtn_quantise(t.data.data(), t.shape[0], t.shape[1], cd, false);
            (void)tied;
            m.codec[n] = cname;
        }
    }
    rep.push_back({"quantised", value(quantised)});
    rep.push_back({"skipped", value(skipped)});
    rep.push_back({"mean_rel_err", value(err_n ? err_sum / double(err_n) : 0.0)});
    rep.push_back({"bits_per_weight", value(8.0 * double(cd.block_bytes) / double(cd.block))});
    return value(rep);
}

inline value run_stage(Ctx& c, const Stage& st);

inline value stage_optimise(Ctx& c, const Stage& st) {
    Model& m = c.m;
    std::vector<value> sub;
    const bool sparse = is_sparse_model(m);
    if (!sparse && is_pow2(m.spec.d_model) && st.str("rotate", "1") == "1") sub.push_back(run_stage(c, Stage{"rotate", {{"seed", st.str("seed", "0")}}}));
    if (st.str("equalize", "1") == "1") sub.push_back(run_stage(c, Stage{"equalize", {}}));
    const std::string codec = st.str("codec", "mixed");
    if (codec == "mixed") sub.push_back(run_stage(c, Stage{"quant", {{"method", "mixed"}, {"kl", st.str("kl", "0.005")}, {"slack", st.str("slack", "1")}, {"head", st.str("head", "q8_0")}}}));
    else if (codec != "f32") sub.push_back(run_stage(c, Stage{"quant", {{"method", "gptq"}, {"codec", codec}, {"head", "rtn"}}}));
    if (c.ref && st.str("calibrate", "1") == "1") sub.push_back(run_stage(c, Stage{"calibrate", {}}));
    return value(jobj{{"stage", value(st.text())}, {"sparse", value(sparse)}, {"stages", value(sub)}});
}

inline value run_stage(Ctx& c, const Stage& st) {
    const double t0 = now_s();
    say("stage {}", st.text());
    value r;
    if (st.name == "none") r = value(jobj{{"stage", value("none")}});   // storage only (e.g. a new codec)
    else if (st.name == "prune") r = stage_prune(c, st);
    else if (st.name == "width") r = stage_width(c, st);
    else if (st.name == "depth") r = stage_depth(c, st);
    else if (st.name == "experts") r = stage_experts(c, st);
    else if (st.name == "merge") r = stage_merge(c, st);
    else if (st.name == "moe2dense") r = stage_moe2dense(c, st);
    else if (st.name == "quant") r = stage_quant(c, st);
    else if (st.name == "rotate") {
        if (is_sparse_model(c.m) && st.str("force", "0") != "1") r = value(jobj{{"stage", value(st.text())}, {"skipped", value("sparse model: rotation would densify it")}});
        else r = rotate_model(c.m, std::uint64_t(st.num("seed", 0)), st.str("heads", "1") == "1");
    }
    else if (st.name == "equalize") r = equalize_model(c.m, st.num("alpha", 1.0));
    else if (st.name == "calibrate") {
        if (!c.ref || c.ref->empty()) die("calibrate needs reference probes (forge compress captures them when the recipe has calibrate / kl)");
        r = calibrate_distribution(c.m, *c.ref, st.str("bias", "1") == "1");
    }
    else if (st.name == "optimise" || st.name == "optimize") r = stage_optimise(c, st);
    else if (auto it = stage_registry().find(st.name); it != stage_registry().end()) r = it->second(c, st);
    else die("unknown stage " + st.name);
    auto& o = r.obj_mut();
    o.push_back({"seconds", value(now_s() - t0)});
    o.push_back({"counts", counts_json(count(c.m))});
    return r;
}

}  // namespace forge::compress

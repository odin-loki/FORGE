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
#pragma once
#include <forge/compress/moe.hpp>
#include <forge/compress/quant.hpp>

namespace forge::compress {

using forge::json::value;
using jobj = forge::json::object;

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
    for (const char* n : {"tok_emb", "out_norm", "lm_head"}) {
        if (std::string(n) == "lm_head" && tied) continue;
        const auto& t = m.w.at(n);
        i64 nz = 0; for (float x : t.data) nz += x != 0.f;
        c.params += forge::numel(t.shape); c.nonzero += nz;
    }
    return c;
}

inline value counts_json(const Counts& c) {
    return value(jobj{{"params", value(c.params)}, {"nonzero", value(c.nonzero)}, {"body_params", value(c.body_params)}, {"body_nonzero", value(c.body_nonzero)}});
}

struct Ctx {
    Model& m;
    const CalibSet& calib;
    std::vector<value> reports;
    double damp = 0.01;
};

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
    if (L.attn && attn_ok) { add("wq", "attn"); add("wk", "attn"); add("wv", "attn"); add("wo", "o"); }
    if (L.ffn == LayerSpec::Ffn::dense && ffn_ok) { add("w_gate", "mlp"); add("w_up", "mlp"); add("w_down", "down"); }
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
                std::vector<double> dg = lv.diag && !lv.diag->empty() ? *lv.diag : (lv.H ? diag_of(*lv.H) : std::vector<double>(std::size_t(lv.cols), 1.0));
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
        if (alloc == "eba") unit_s = eba_allocate(units, grid, pc.sparsity, st.num("min_s", 0.0), st.num("max_s", 0.9));
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
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
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
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        if (m.spec.layers[std::size_t(l)].ffn == LayerSpec::Ffn::dense) {
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
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
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
        if (replace == "linear") { layer_forward(m, l, h, nx); std::swap(h, nx); }
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
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        if (m.spec.layers[std::size_t(l)].ffn == LayerSpec::Ffn::moe) {
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
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        if (m.spec.layers[std::size_t(l)].ffn == LayerSpec::Ffn::moe) {
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
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        if (m.spec.layers[std::size_t(l)].ffn == LayerSpec::Ffn::moe) {
            StatsReq rq; rq.gram = false; rq.moe_dense_acts = true;
            auto S = layer_pass(m, l, h, rq, nullptr);
            const auto& wg = m.w.at(std::format("layer.{}.experts.w_gate", l));
            const i64 n = wg.shape[0] * wg.shape[1];
            const i64 keep = std::clamp<i64>(i64(std::llround(double(n) * width / double(round))) * round, round, n);
            // try, and keep the MoE layer if the dense fit is worse than max_err
            auto saved_spec = m.spec.layers[std::size_t(l)];
            std::map<std::string, tensor> saved;
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
    }
    rep.push_back({"layers", value(layers)});
    return value(rep);
}

// ------------------------------------------------------------------------------------------------ quant stage
inline value stage_quant(Ctx& c, const Stage& st) {
    Model& m = c.m;
    const std::string method = st.str("method", "gptq"), scope = st.str("scope", "all"), cname = st.str("codec", "q8_0");
    const auto& cd = codec_of(cname);
    const bool gptq = method == "gptq";
    if (!gptq && method != "rtn") die("quant: method gptq or rtn");
    jobj rep{{"stage", value(st.text())}};
    Hidden h = Hidden::embed(m, c.calib), nx;
    double err_sum = 0; i64 err_n = 0, quantised = 0, skipped = 0;
    std::set<std::string> done_stacked;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
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
        if (gptq) say("quant layer {}/{} done", l + 1, m.spec.n_layers());
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

inline value run_stage(Ctx& c, const Stage& st) {
    const double t0 = now_s();
    say("stage {}", st.text());
    value r;
    if (st.name == "prune") r = stage_prune(c, st);
    else if (st.name == "width") r = stage_width(c, st);
    else if (st.name == "depth") r = stage_depth(c, st);
    else if (st.name == "experts") r = stage_experts(c, st);
    else if (st.name == "merge") r = stage_merge(c, st);
    else if (st.name == "moe2dense") r = stage_moe2dense(c, st);
    else if (st.name == "quant") r = stage_quant(c, st);
    else die("unknown stage " + st.name);
    auto& o = r.obj_mut();
    o.push_back({"seconds", value(now_s() - t0)});
    o.push_back({"counts", counts_json(count(c.m))});
    return r;
}

}  // namespace forge::compress

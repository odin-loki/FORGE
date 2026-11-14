// Attention head pruning, exact neuron / value-channel permutations for N:M pruning, and calibrated activation
// sparsity. Three stages, registered with compress.hpp's stage registry:
//
//   heads     s=<fraction of query heads removed>  score=obs|act|mag  alloc=uniform|eba  refit=ls|none  max_s=<cap>
//             swap=<cols>  exhaustive=<n>
//             Query-head pruning. Every KV group keeps the same number of query heads (at least one), so a layer
//             keeps a multiple of n_kv heads; wq loses the head's rows (and bq its entries), wo the head's columns;
//             wk / wv / q_norm / k_norm are untouched. Sequential, layer by layer (like `width`).
//               mag  ||wq[head rows]|| * ||wo[:, head cols]||
//               act  sum over the head's columns c of H_cc * ||wo[:, c]||^2  (H = Gram of the wo input: the
//                    diagonal approximation of the head's output energy, Wanda-sp / FLAP style)
//               obs  structured OBS over whole head blocks B of wo's input: removing B costs
//                    tr(W_B (Hinv_BB)^-1 W_B^T) and the alive columns get the optimal update; heads are removed
//                    greedily in rounds of one head per KV group so every intermediate state is balanced
//                    then a refinement under the exact set cost tr(W_R (Hinv_RR)^-1 W_R^T) when at most
//                    swap=<columns, default 1024> are removed: all balanced selections when there are at most
//                    exhaustive=<n, default 512> of them, else in-group swaps of a removed and a kept head
//             refit=ls (default): least-squares refit of wo on the kept columns (la::ls_refit_columns).
//             alloc=eba: per-layer head budgets from the error curves (removal of one head per group = one step;
//             greedy marginal cost over the convex hull, as width's eba), cap max_s per layer.
//   permute   nm=2:4  values=1  iters=<local-search trials per group>
//             Exact symmetry: SwiGLU neurons (rows of w_gate / w_up, columns of w_down) are reordered so that the
//             N:M groups along w_down's input keep more Wanda saliency |W| * ||x|| (balanced serpentine assignment
//             of high-saliency neurons, then pairwise group swaps, as RIA's channel permutation). values=1 also
//             permutes the value channels within each KV head (rows of wv / bv and the matching wo columns of every
//             query head of its group). The function is unchanged; the report gives the retained N:M saliency
//             fraction before / after. Run it before prune(nm=...).
//   actsparse p=<target fraction of zeroed FFN activations>  mode=threshold  alloc=uniform|eba  max_p=<cap>
//             samples=<per-layer sample cap>  measure=1
//             TEAL / CATS-style activation sparsity of dense SwiGLU FFNs: the hidden a = silu(gate) * up is
//             zeroed where |a| <= t, with t per layer the p-quantile of |a| on the calibration data (computed from
//             the "mlp" input with swiglu_hidden; sequential, each layer sees the thresholded earlier layers).
//             alloc=eba spreads p across layers by the output-error proxy sum a^2 ||w_down[:, j]||^2 of the
//             zeroed activations. This is a k-winner / tropical (max-plus, idempotent) sparsification: keeping
//             the large activations is a selection that applied twice gives the same result. FORGE does not yet
//             skip the zeroed work in its kernels, so the gain today is accuracy-measured readiness for a sparse
//             (gather-then-GEMV) w_down kernel; measure=1 reports the zero fraction actually seen at w_down's input
//             in the FORGE graph.
#pragma once
#include <forge/compress/compress.hpp>

#include <bit>
#include <queue>
#include <random>

namespace forge::compress {

// ================================================================================================ heads
enum class HeadScore { mag, act, obs };
inline HeadScore head_score_from(const std::string& s) {
    if (s == "mag") return HeadScore::mag;
    if (s == "act") return HeadScore::act;
    if (s == "obs") return HeadScore::obs;
    die("unknown heads score " + s);
}

// Block OBS over the head blocks of W [rows, nh*hd] with input Gram H, greedy in rounds of one head per group
// (group of head h = h / (nh / n_groups)). Stops when every group has `keep_pg` heads, or runs to 1 per group
// when keep_pg <= 0. Returns the removal order; `curve[k]` = cumulative absolute output error after k rounds.
inline std::vector<i64> obs_heads_order(const float* W0, i64 rows, i64 nh, i64 hd, i64 n_groups, const la::Sym& H, i64 keep_pg,
                                        double damp, std::vector<double>* curve) {
    const i64 n = nh * hd, g = nh / n_groups;
    if (keep_pg <= 0) keep_pg = 1;
    la::Sym L = H;
    la::cholesky_damped(L, damp);
    la::Sym Hi = la::inverse_from_cholesky(L);
    std::vector<float> W(W0, W0 + rows * n);
    std::vector<std::uint8_t> alive(static_cast<std::size_t>(nh), 1);
    std::vector<i64> left(static_cast<std::size_t>(n_groups), g), order;
    double acc = 0;
    if (curve) curve->assign(1, 0.0);
    // cost of removing head h (and X = W_B Hinv_BB^-1 when wanted)
    auto block_cost = [&](i64 h, std::vector<float>* X) {
        std::vector<i64> B(static_cast<std::size_t>(hd)); std::iota(B.begin(), B.end(), h * hd);
        la::Sym Hb = la::submatrix(Hi, B);
        std::vector<float> Wb(static_cast<std::size_t>(rows * hd));
        for (i64 r = 0; r < rows; ++r) std::copy_n(W.data() + r * n + h * hd, hd, Wb.data() + r * hd);
        std::vector<float> Y = Wb;
        la::spd_solve_rows(Hb, 1e-9, Y.data(), rows);
        double c = 0; for (std::size_t i = 0; i < Y.size(); ++i) c += double(Y[i]) * Wb[i];
        if (X) *X = std::move(Y);
        return c;
    };
    while (true) {
        std::vector<std::uint8_t> pending(static_cast<std::size_t>(n_groups), 0);
        i64 n_pending = 0;
        for (i64 q = 0; q < n_groups; ++q) if (left[std::size_t(q)] > keep_pg) { pending[std::size_t(q)] = 1; ++n_pending; }
        if (!n_pending) break;
        while (n_pending) {   // one round: the cheapest head of each pending group, cheapest group first
            std::vector<double> cost(static_cast<std::size_t>(nh), INFINITY);
            for (i64 hh = 0; hh < nh; ++hh) if (alive[std::size_t(hh)] && pending[std::size_t(hh / g)]) cost[std::size_t(hh)] = block_cost(hh, nullptr);
            const i64 h = i64(std::min_element(cost.begin(), cost.end()) - cost.begin());
            std::vector<float> X;
            acc += block_cost(h, &X);   // X = W_B Hinv_BB^-1 [rows, hd]
            // Y = Hinv_{., B} Hinv_BB^-1 [n, hd]
            std::vector<i64> B(static_cast<std::size_t>(hd)); std::iota(B.begin(), B.end(), h * hd);
            std::vector<float> HB(static_cast<std::size_t>(n * hd));
            for (i64 i = 0; i < n; ++i) for (i64 k = 0; k < hd; ++k) HB[std::size_t(i * hd + k)] = float(Hi(i, h * hd + k));
            std::vector<float> Y = HB;
            la::spd_solve_rows(la::submatrix(Hi, B), 1e-9, Y.data(), n);
            std::vector<float> dW(static_cast<std::size_t>(rows * n)), dH(static_cast<std::size_t>(n * n));
            la::gemm_nt(X.data(), HB.data(), dW.data(), rows, n, hd);   // X Hinv_{B,.}
            la::gemm_nt(Y.data(), HB.data(), dH.data(), n, n, hd);      // Hinv_{.,B} Hinv_BB^-1 Hinv_{B,.}
            for (std::size_t i = 0; i < W.size(); ++i) W[i] -= dW[i];
            la::parallel_for(static_cast<std::size_t>(n), 16, [&](std::size_t b, std::size_t e) {
                for (std::size_t i = b; i < e; ++i) { double* hr = Hi.row(i64(i)); const float* d = dH.data() + i * std::size_t(n); for (i64 j = 0; j < n; ++j) hr[j] -= double(d[j]); }
            });
            for (i64 k : B) { for (i64 j = 0; j < n; ++j) { Hi(k, j) = 0; Hi(j, k) = 0; } Hi(k, k) = 1; }
            for (i64 r = 0; r < rows; ++r) std::fill_n(W.data() + r * n + h * hd, hd, 0.f);
            alive[std::size_t(h)] = 0;
            --left[std::size_t(h / g)];
            pending[std::size_t(h / g)] = 0; --n_pending;
            order.push_back(h);
        }
        if (curve) curve->push_back(acc);
    }
    return order;
}

// Per-head scores (higher = more important) for mag / act.
inline std::vector<double> head_scores(const float* Wq, const float* Wo, i64 d, i64 nh, i64 hd, const la::Sym* H,
                                       const std::vector<double>* diag, HeadScore sc) {
    const i64 n = nh * hd;
    std::vector<double> s(static_cast<std::size_t>(nh), 0.0);
    for (i64 h = 0; h < nh; ++h) {
        double on = 0, qn = 0, act = 0;
        for (i64 c = h * hd; c < (h + 1) * hd; ++c) {
            double cn = 0; for (i64 r = 0; r < d; ++r) cn += double(Wo[r * n + c]) * Wo[r * n + c];
            on += cn;
            const double hc = H ? (*H)(c, c) : (diag && !diag->empty() ? (*diag)[std::size_t(c)] : 1.0);
            act += hc * cn;
        }
        for (i64 i = h * hd * d; i < (h + 1) * hd * d; ++i) qn += double(Wq[i]) * Wq[i];
        s[std::size_t(h)] = sc == HeadScore::act ? act : std::sqrt(on) * std::sqrt(qn);
    }
    return s;
}

// Removal order by score, in balanced rounds (round r removes the r-th weakest head of every group), with the
// cumulative removed score after each round.
inline std::vector<i64> score_heads_order(const std::vector<double>& s, i64 n_groups, i64 keep_pg, std::vector<double>* curve) {
    const i64 nh = i64(s.size()), g = nh / n_groups;
    if (keep_pg <= 0) keep_pg = 1;
    std::vector<std::vector<i64>> by(static_cast<std::size_t>(n_groups));
    for (i64 q = 0; q < n_groups; ++q) {
        for (i64 h = q * g; h < (q + 1) * g; ++h) by[std::size_t(q)].push_back(h);
        std::sort(by[std::size_t(q)].begin(), by[std::size_t(q)].end(), [&](i64 a, i64 b) { return s[std::size_t(a)] < s[std::size_t(b)]; });
    }
    std::vector<i64> order;
    double acc = 0;
    if (curve) curve->assign(1, 0.0);
    for (i64 r = 0; r < g - keep_pg; ++r) {
        for (i64 q = 0; q < n_groups; ++q) { const i64 h = by[std::size_t(q)][std::size_t(r)]; order.push_back(h); acc += s[std::size_t(h)]; }
        if (curve) curve->push_back(acc);
    }
    return order;
}

inline bool heads_prunable(const Model& m, i64 l) {
    const auto& sp = m.spec;
    if (!sp.layers[std::size_t(l)].attn || sp.factored(l, "wq") || sp.factored(l, "wo")) return false;
    for (const char* x : {"wq", "wo", "bq"}) if (sp.alias.count(std::format("layer.{}.{}", l, x))) return false;
    return sp.heads_of(l) / sp.n_kv > 1;
}

struct HeadsResult { std::vector<i64> kept; double rel_err = 0; i64 swaps = 0; };

// Exact error of removing a set of head blocks R at once (the optimal refit of the rest): tr(W_R (Hinv_RR)^-1 W_R^T).
inline double obs_set_cost(const float* W, i64 rows, i64 hd, i64 n, const la::Sym& Hi, const std::vector<std::uint8_t>& drop) {
    std::vector<i64> R;
    for (std::size_t h = 0; h < drop.size(); ++h) if (drop[h]) for (i64 c = i64(h) * hd; c < (i64(h) + 1) * hd; ++c) R.push_back(c);
    if (R.empty()) return 0;
    std::vector<float> Wr = take_cols(W, rows, n, R), Y = Wr;
    la::spd_solve_rows(la::submatrix(Hi, R), 1e-9, Y.data(), rows);
    double c = 0; for (std::size_t i = 0; i < Y.size(); ++i) c += double(Y[i]) * Wr[i];
    return c;
}

// Refinement of a greedy head selection under the exact set cost. When the balanced selections are few
// (C(g, removed per group)^groups <= `exhaustive`) all of them are evaluated; otherwise, within each KV group, a
// removed and a kept head are exchanged while the cost decreases (first improvement, a few passes). Returns the
// number of changes made to the greedy selection.
inline i64 obs_heads_refine(const float* W, i64 rows, i64 nh, i64 hd, i64 n_groups, const la::Sym& H, double damp, std::vector<std::uint8_t>& drop,
                            i64 exhaustive = 512, int passes = 3) {
    const i64 n = nh * hd, g = nh / n_groups;
    la::Sym L = H;
    la::cholesky_damped(L, damp);
    const la::Sym Hi = la::inverse_from_cholesky(L);
    double cur = obs_set_cost(W, rows, hd, n, Hi, drop);
    i64 rem = 0; for (i64 h = 0; h < g; ++h) rem += drop[std::size_t(h)];
    std::vector<std::uint32_t> subsets;   // removal patterns of one group (bit masks with `rem` bits)
    if (g <= 20) for (std::uint32_t mk = 0; mk < (1u << g); ++mk) if (std::popcount(mk) == rem) subsets.push_back(mk);
    double combos = 1; for (i64 q = 0; q < n_groups; ++q) combos *= double(subsets.size());
    if (!subsets.empty() && combos <= double(exhaustive)) {
        const std::vector<std::uint8_t> greedy = drop;
        std::vector<std::size_t> digit(static_cast<std::size_t>(n_groups), 0);
        std::vector<std::uint8_t> best = drop, trial(drop.size());
        while (true) {
            for (i64 q = 0; q < n_groups; ++q) for (i64 h = 0; h < g; ++h) trial[std::size_t(q * g + h)] = (subsets[digit[std::size_t(q)]] >> h) & 1;
            const double c = obs_set_cost(W, rows, hd, n, Hi, trial);
            if (c < cur * (1 - 1e-9)) { cur = c; best = trial; }
            std::size_t q = 0;
            while (q < digit.size() && ++digit[q] == subsets.size()) digit[q++] = 0;
            if (q == digit.size()) break;
        }
        drop = best;
        i64 changed = 0; for (std::size_t h = 0; h < drop.size(); ++h) changed += drop[h] != greedy[h];
        return changed / 2;
    }
    i64 swaps = 0;
    for (int ps = 0; ps < passes; ++ps) {
        bool any = false;
        for (i64 q = 0; q < n_groups; ++q)
            for (i64 r = q * g; r < (q + 1) * g; ++r) {
                if (!drop[std::size_t(r)]) continue;
                for (i64 k = q * g; k < (q + 1) * g; ++k) {
                    if (drop[std::size_t(k)]) continue;
                    drop[std::size_t(r)] = 0; drop[std::size_t(k)] = 1;
                    const double c = obs_set_cost(W, rows, hd, n, Hi, drop);
                    if (c < cur * (1 - 1e-9)) { cur = c; ++swaps; any = true; break; }
                    drop[std::size_t(r)] = 1; drop[std::size_t(k)] = 0;
                }
            }
        if (!any) break;
    }
    return swaps;
}

// Prunes layer l's query heads to `keep_pg` per KV group. H = Gram of the wo input (all current heads).
// obs: a swap refinement of the greedy selection runs when at most `swap_cols` columns are removed (its set
// cost needs a solve of that size per candidate swap).
inline HeadsResult prune_heads(Model& m, i64 l, const la::Sym& H, const std::vector<double>& diag, i64 keep_pg, HeadScore sc,
                               double damp, bool refit, i64 swap_cols = 1024, i64 exhaustive = 512) {
    const std::string p = std::format("layer.{}.", l);
    auto& wq = m.w.at(p + "wq"); auto& wo = m.w.at(p + "wo");
    const i64 d = wo.shape[0], nh = m.spec.heads_of(l), hd = m.spec.head_dim, G = m.spec.n_kv, n = nh * hd;
    if (wo.shape[1] != n || wq.shape[0] != n) die(std::format("heads: layer {} wq/wo shapes do not match {} heads", l, nh));
    HeadsResult res;
    std::vector<i64> order = sc == HeadScore::obs ? obs_heads_order(wo.data.data(), d, nh, hd, G, H, keep_pg, damp, nullptr)
                                                  : score_heads_order(head_scores(wq.data.data(), wo.data.data(), d, nh, hd, &H, &diag, sc), G, keep_pg, nullptr);
    std::vector<std::uint8_t> drop(static_cast<std::size_t>(nh), 0);
    for (i64 h : order) drop[std::size_t(h)] = 1;
    if (sc == HeadScore::obs && swap_cols >= i64(order.size()) * hd) res.swaps = obs_heads_refine(wo.data.data(), d, nh, hd, G, H, damp, drop, exhaustive);
    std::vector<i64> cols, rows_q;
    for (i64 h = 0; h < nh; ++h) if (!drop[std::size_t(h)]) {
        res.kept.push_back(h);
        for (i64 c = h * hd; c < (h + 1) * hd; ++c) cols.push_back(c);
    }
    const std::vector<float> W0 = wo.data;
    std::vector<float> Wn = refit ? la::ls_refit_columns(W0.data(), d, H, cols, damp) : take_cols(W0.data(), d, n, cols);
    {   // relative output error of the new wo against the old one (full-width, zeros at removed columns)
        std::vector<float> full(static_cast<std::size_t>(d * n), 0.f);
        for (i64 r = 0; r < d; ++r) for (std::size_t j = 0; j < cols.size(); ++j) full[std::size_t(r * n + cols[j])] = Wn[std::size_t(r) * cols.size() + j];
        res.rel_err = rel_error(full.data(), W0.data(), d, n, H);
    }
    const i64 k = i64(cols.size());
    wo.data = std::move(Wn); wo.shape = {d, k};
    wq.data = take_rows(wq.data.data(), d, cols); wq.shape = {k, d};
    if (m.spec.qkv_bias && m.w.has(p + "bq")) {
        auto& bq = m.w.at(p + "bq");
        bq.data = take_rows(bq.data.data(), 1, cols); bq.shape = {k};
    }
    const i64 nk = i64(res.kept.size());
    if (nk == m.spec.n_heads) m.spec.heads.erase(l); else m.spec.heads[l] = nk;
    return res;
}

// Per-layer heads-per-group budget from the error curves: steps (one head per group) go where the marginal
// error per step is smallest (convex-hull greedy).
inline std::map<i64, i64> heads_eba(const std::map<i64, std::vector<double>>& curves, const std::map<i64, i64>& max_steps, i64 steps) {
    std::map<i64, i64> at;
    for (auto& [l, c] : curves) at[l] = 0;
    using item = std::tuple<double, i64, i64>;   // (cost per step, layer, to-step)
    auto best = [&](i64 l) -> std::optional<item> {
        const auto& c = curves.at(l);
        std::optional<item> b;
        const i64 a = at[l];
        for (i64 t = a + 1; t <= max_steps.at(l) && t < i64(c.size()); ++t) {
            const double cost = (c[std::size_t(t)] - c[std::size_t(a)]) / double(t - a);
            if (!b || cost < std::get<0>(*b)) b = item{cost, l, t};
        }
        return b;
    };
    std::priority_queue<item, std::vector<item>, std::greater<item>> pq;
    for (auto& [l, c] : curves) if (auto s = best(l)) pq.push(*s);
    i64 have = 0;
    while (have < steps && !pq.empty()) {
        auto [cost, l, t] = pq.top(); pq.pop();
        if (t <= at[l]) continue;
        t = std::min(t, at[l] + (steps - have));   // do not overshoot the budget
        have += t - at[l]; at[l] = t;
        if (auto s = best(l)) pq.push(*s);
    }
    return at;
}

inline value stage_heads(Ctx& c, const Stage& st) {
    Model& m = c.m;
    const double s = st.num("s", 0.25);
    const auto sc = head_score_from(st.str("score", "obs"));
    const std::string alloc = st.str("alloc", "uniform");
    const bool refit = st.str("refit", "ls") != "none";
    const double damp = st.num("damp", c.damp), max_s = st.num("max_s", 0.75);
    const i64 swap_cols = i64(st.num("swap", 1024)), exhaustive = i64(st.num("exhaustive", 512));
    const i64 G = m.spec.n_kv;
    jobj rep{{"stage", value(st.text())}};

    std::map<i64, i64> keep_pg;   // layer -> query heads kept per KV group
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        if (!heads_prunable(m, l)) continue;
        const i64 g = m.spec.heads_of(l) / G;
        keep_pg[l] = std::clamp<i64>(i64(std::llround(double(g) * (1.0 - s))), 1, g);
    }
    if (alloc == "eba") {
        std::map<i64, std::vector<double>> curves;
        std::map<i64, i64> max_steps;
        i64 total_groups_heads = 0;   // sum over layers of heads per group (one step = one head per group)
        Hidden h = Hidden::embed(m, c.calib), nx;
        for (i64 l = 0; l < m.spec.n_layers(); ++l) {
            const bool pr = keep_pg.count(l) != 0;
            StatsReq rq; rq.gram = pr && sc == HeadScore::obs; rq.only = {"o"};
            auto S = layer_pass(m, l, h, rq, &nx);
            std::swap(h, nx);
            if (m.spec.layers[std::size_t(l)].attn) total_groups_heads += m.spec.heads_of(l) / G;
            if (!pr) continue;
            const std::string p = std::format("layer.{}.", l);
            const auto& wq = m.w.at(p + "wq"); const auto& wo = m.w.at(p + "wo");
            const i64 nh = m.spec.heads_of(l), hd = m.spec.head_dim, g = nh / G;
            std::vector<double> curve;
            if (sc == HeadScore::obs) obs_heads_order(wo.data.data(), wo.shape[0], nh, hd, G, S.gram.at("o").H, 1, damp, &curve);
            else {
                auto scores = head_scores(wq.data.data(), wo.data.data(), wo.shape[0], nh, hd, nullptr, &S.diag["o"], sc);
                score_heads_order(scores, G, 1, &curve);
                if (sc == HeadScore::mag) { double t = 0; for (double x : scores) t += x; for (auto& x : curve) x /= std::max(t, 1e-30); }
            }
            if (sc != HeadScore::mag) for (auto& x : curve) x /= std::max(S.resid_out, 1e-30);
            curves[l] = curve;
            max_steps[l] = std::min<i64>(g - 1, i64(std::floor(max_s * double(g) + 1e-9)));
        }
        const i64 steps = i64(std::llround(s * double(total_groups_heads)));
        auto at = heads_eba(curves, max_steps, steps);
        std::vector<value> al;
        for (auto& [l, t] : at) {
            keep_pg[l] = m.spec.heads_of(l) / G - t;
            al.push_back(value(jobj{{"layer", value(l)}, {"heads", value(keep_pg[l] * G)}}));
        }
        rep.push_back({"allocation", value(al)});
    } else if (alloc != "uniform") die("heads: unknown alloc " + alloc);

    Hidden h = Hidden::embed(m, c.calib), nx;
    std::vector<value> layers;
    i64 before = 0, after = 0;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        if (m.spec.layers[std::size_t(l)].attn) before += m.spec.heads_of(l);
        if (keep_pg.count(l) && keep_pg[l] < m.spec.heads_of(l) / G) {
            StatsReq rq; rq.gram = true; rq.only = {"o"};
            auto S = layer_pass(m, l, h, rq, nullptr);
            const i64 nh = m.spec.heads_of(l);
            auto r = prune_heads(m, l, S.gram.at("o").H, S.diag["o"], keep_pg[l], sc, damp, refit, swap_cols, exhaustive);
            std::vector<value> kept; for (i64 k : r.kept) kept.push_back(value(k));
            layers.push_back(value(jobj{{"layer", value(l)}, {"heads", value(nh)}, {"kept", value(kept)}, {"rel_err", value(r.rel_err)}, {"swaps", value(r.swaps)}}));
            say("heads layer {}: {} -> {} query heads, rel err {:.4g}", l, nh, r.kept.size(), r.rel_err);
        }
        if (m.spec.layers[std::size_t(l)].attn) after += m.spec.heads_of(l);
        layer_forward(m, l, h, nx);
        std::swap(h, nx);
    }
    rep.push_back({"heads_before", value(before)});
    rep.push_back({"heads_after", value(after)});
    rep.push_back({"layers", value(layers)});
    return value(rep);
}

// ================================================================================================ permute
// Saliency stored column-major: col j holds R values (one per output row). The retained N:M saliency of a
// permutation is the sum over rows and groups of M consecutive positions of the N largest saliencies; positions
// past the last full group are not pruned (select_mask skips a partial tail) and keep everything.
struct PermProblem {
    i64 n = 0, R = 0; int M = 4, N = 2;
    std::vector<float> sal;   // [n][R]
    const float* col(i64 j) const { return sal.data() + j * R; }
    i64 groups() const { return (n + M - 1) / M; }
    double group_keep(const i64* cols, i64 cnt) const {
        double s = 0;
        if (cnt < M) { for (i64 k = 0; k < cnt; ++k) { const float* c = col(cols[k]); for (i64 r = 0; r < R; ++r) s += c[r]; } return s; }
        const float* cs[16];
        for (i64 k = 0; k < cnt; ++k) cs[k] = col(cols[k]);
        float v[16];
        for (i64 r = 0; r < R; ++r) {
            for (int k = 0; k < M; ++k) v[k] = cs[k][r];
            std::partial_sort(v, v + N, v + M, std::greater<float>());
            for (int k = 0; k < N; ++k) s += v[k];
        }
        return s;
    }
    double retained(const std::vector<i64>& perm) const {
        double s = 0;
        for (i64 g = 0; g < groups(); ++g) s += group_keep(perm.data() + g * M, std::min<i64>(M, n - g * M));
        return s;
    }
    double total() const { double s = 0; for (float x : sal) s += x; return s; }
};

// A permutation (position -> original column) that increases the retained N:M saliency: the better of the
// identity and a balanced serpentine assignment (columns by total saliency, dealt across groups), then pairwise
// swap local search between random group pairs (best of the M*M swaps, kept when it improves).
inline std::vector<i64> optimise_permutation(const PermProblem& P, i64 trials_per_group, std::uint64_t seed) {
    const i64 n = P.n, M = P.M, G = P.groups(), full = n / M;
    std::vector<i64> id(static_cast<std::size_t>(n)); std::iota(id.begin(), id.end(), 0);
    if (M > 16 || n < 2 * M || M <= P.N) return id;
    std::vector<double> tot(static_cast<std::size_t>(n));
    for (i64 j = 0; j < n; ++j) { double s = 0; const float* c = P.col(j); for (i64 r = 0; r < P.R; ++r) s += c[r]; tot[std::size_t(j)] = s; }
    std::vector<i64> byw = id;
    std::sort(byw.begin(), byw.end(), [&](i64 a, i64 b) { return tot[std::size_t(a)] > tot[std::size_t(b)]; });
    std::vector<i64> bal(static_cast<std::size_t>(n), -1);
    {   // the heaviest columns go to the unpruned tail, the rest are dealt serpentine over the full groups
        i64 q = 0;
        for (i64 p = full * M; p < n; ++p) bal[std::size_t(p)] = byw[std::size_t(q++)];
        std::vector<i64> fill(static_cast<std::size_t>(full), 0);
        for (i64 i = 0; q < n; ++i, ++q) {
            const i64 round = i / full, k = i % full, g = (round % 2 == 0) ? k : full - 1 - k;
            bal[std::size_t(g * M + fill[std::size_t(g)]++)] = byw[std::size_t(q)];
        }
    }
    std::vector<i64> perm = P.retained(bal) > P.retained(id) ? bal : id;
    std::vector<double> gk(static_cast<std::size_t>(G));
    for (i64 g = 0; g < G; ++g) gk[std::size_t(g)] = P.group_keep(perm.data() + g * M, std::min<i64>(M, n - g * M));
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<i64> pick(0, G - 1);
    const i64 trials = trials_per_group * G;
    for (i64 t = 0; t < trials; ++t) {
        const i64 a = pick(rng), b = pick(rng);
        if (a == b) continue;
        const i64 ca = std::min<i64>(M, n - a * M), cb = std::min<i64>(M, n - b * M);
        double best = 0; i64 bi = -1, bj = -1;
        i64 A[16], B[16];
        for (i64 i = 0; i < ca; ++i) for (i64 j = 0; j < cb; ++j) {
            std::copy_n(perm.data() + a * M, ca, A); std::copy_n(perm.data() + b * M, cb, B);
            std::swap(A[i], B[j]);
            const double dlt = P.group_keep(A, ca) + P.group_keep(B, cb) - gk[std::size_t(a)] - gk[std::size_t(b)];
            if (dlt > best + 1e-9 * (gk[std::size_t(a)] + gk[std::size_t(b)])) { best = dlt; bi = i; bj = j; }
        }
        if (bi < 0) continue;
        std::swap(perm[std::size_t(a * M + bi)], perm[std::size_t(b * M + bj)]);
        gk[std::size_t(a)] = P.group_keep(perm.data() + a * M, ca);
        gk[std::size_t(b)] = P.group_keep(perm.data() + b * M, cb);
    }
    return perm;
}

inline bool aliased_any(const Model& m, i64 l, std::initializer_list<const char*> xs) {
    for (const char* x : xs) if (m.spec.alias.count(std::format("layer.{}.{}", l, x)) || m.spec.factored(l, x)) return true;
    return false;
}

inline value stage_permute(Ctx& c, const Stage& st) {
    Model& m = c.m;
    const std::string nm = st.str("nm", "2:4");
    const int N = std::stoi(nm.substr(0, nm.find(':'))), M = std::stoi(nm.substr(nm.find(':') + 1));
    if (M <= 0 || N <= 0 || N >= M || M > 16) die("permute: nm must be N:M with 0 < N < M <= 16");
    const bool values = st.str("values", "1") == "1";
    const i64 trials = i64(st.num("iters", 16));
    const std::uint64_t seed = std::uint64_t(st.num("seed", 0));
    jobj rep{{"stage", value(st.text())}};
    std::vector<value> layers;
    double ffn_b = 0, ffn_a = 0, ffn_t = 0, v_b = 0, v_a = 0, v_t = 0;
    Hidden h = Hidden::embed(m, c.calib), nx;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        StatsReq rq; rq.gram = false;
        auto S = layer_pass(m, l, h, rq, &nx);   // the permutations are exact: the propagated states stay valid
        std::swap(h, nx);
        const auto& L = m.spec.layers[std::size_t(l)];
        const std::string p = std::format("layer.{}.", l);
        jobj lr{{"layer", value(l)}};
        if (L.ffn == LayerSpec::Ffn::dense && !aliased_any(m, l, {"w_gate", "w_up", "w_down"})) {
            auto& wg = m.w.at(p + "w_gate"); auto& wu = m.w.at(p + "w_up"); auto& wd = m.w.at(p + "w_down");
            const i64 d = wd.shape[0], f = wd.shape[1];
            const auto& dg = S.diag["down"];
            PermProblem P; P.n = f; P.R = d; P.M = M; P.N = N; P.sal.resize(static_cast<std::size_t>(f * d));
            for (i64 j = 0; j < f; ++j) { const double xn = std::sqrt(std::max(dg.empty() ? 1.0 : dg[std::size_t(j)], 0.0)); for (i64 r = 0; r < d; ++r) P.sal[std::size_t(j * d + r)] = float(std::abs(wd.data[std::size_t(r * f + j)]) * xn); }
            std::vector<i64> id(static_cast<std::size_t>(f)); std::iota(id.begin(), id.end(), 0);
            const double b = P.retained(id), t = P.total();
            auto perm = optimise_permutation(P, trials, seed + std::uint64_t(l));
            const double a = P.retained(perm);
            if (a > b) {
                wg.data = take_rows(wg.data.data(), wg.shape[1], perm);
                wu.data = take_rows(wu.data.data(), wu.shape[1], perm);
                wd.data = take_cols(wd.data.data(), d, f, perm);
            }
            const double af = std::max(a, b);
            ffn_b += b; ffn_a += af; ffn_t += t;
            lr.push_back({"ffn_before", value(b / std::max(t, 1e-30))});
            lr.push_back({"ffn_after", value(af / std::max(t, 1e-30))});
        }
        if (values && L.attn && m.spec.head_dim >= 2 * M && !aliased_any(m, l, {"wv", "wo", "bv"})) {
            auto& wv = m.w.at(p + "wv"); auto& wo = m.w.at(p + "wo");
            const i64 hd = m.spec.head_dim, nh = m.spec.heads_of(l), G = m.spec.n_kv, g = nh / G, d = wo.shape[0], n = wo.shape[1];
            const auto& dg = S.diag["o"];
            double lb = 0, la_ = 0, lt = 0;
            for (i64 k = 0; k < G; ++k) {
                PermProblem P; P.n = hd; P.R = d * g; P.M = M; P.N = N; P.sal.resize(static_cast<std::size_t>(hd * d * g));
                for (i64 cc = 0; cc < hd; ++cc) for (i64 q = 0; q < g; ++q) {
                    const i64 col = (k * g + q) * hd + cc;
                    const double xn = std::sqrt(std::max(dg.empty() ? 1.0 : dg[std::size_t(col)], 0.0));
                    for (i64 r = 0; r < d; ++r) P.sal[std::size_t(cc * d * g + q * d + r)] = float(std::abs(wo.data[std::size_t(r * n + col)]) * xn);
                }
                std::vector<i64> id(static_cast<std::size_t>(hd)); std::iota(id.begin(), id.end(), 0);
                const double b = P.retained(id), t = P.total();
                auto perm = optimise_permutation(P, trials * 4, seed + std::uint64_t(1000 * l + k));
                const double a = P.retained(perm);
                lb += b; lt += t; la_ += std::max(a, b);
                if (a <= b) continue;
                // wv rows (and bv) of KV head k, wo columns of its query heads
                std::vector<float> rows(static_cast<std::size_t>(hd * wv.shape[1]));
                for (i64 cc = 0; cc < hd; ++cc) std::copy_n(wv.data.data() + (k * hd + perm[std::size_t(cc)]) * wv.shape[1], wv.shape[1], rows.data() + cc * wv.shape[1]);
                std::copy(rows.begin(), rows.end(), wv.data.begin() + k * hd * wv.shape[1]);
                if (m.spec.qkv_bias && m.w.has(p + "bv")) {
                    auto& bv = m.w.at(p + "bv");
                    std::vector<float> bb(static_cast<std::size_t>(hd));
                    for (i64 cc = 0; cc < hd; ++cc) bb[std::size_t(cc)] = bv.data[std::size_t(k * hd + perm[std::size_t(cc)])];
                    std::copy(bb.begin(), bb.end(), bv.data.begin() + k * hd);
                }
                std::vector<float> tmp(static_cast<std::size_t>(hd));
                for (i64 q = 0; q < g; ++q) for (i64 r = 0; r < d; ++r) {
                    float* o = wo.data.data() + r * n + (k * g + q) * hd;
                    for (i64 cc = 0; cc < hd; ++cc) tmp[std::size_t(cc)] = o[perm[std::size_t(cc)]];
                    std::copy(tmp.begin(), tmp.end(), o);
                }
            }
            v_b += lb; v_a += la_; v_t += lt;
            lr.push_back({"wo_before", value(lb / std::max(lt, 1e-30))});
            lr.push_back({"wo_after", value(la_ / std::max(lt, 1e-30))});
        }
        layers.push_back(value(lr));
    }
    rep.push_back({"ffn_retained_before", value(ffn_b / std::max(ffn_t, 1e-30))});
    rep.push_back({"ffn_retained_after", value(ffn_a / std::max(ffn_t, 1e-30))});
    if (values) {
        rep.push_back({"wo_retained_before", value(v_b / std::max(v_t, 1e-30))});
        rep.push_back({"wo_retained_after", value(v_a / std::max(v_t, 1e-30))});
    }
    rep.push_back({"layers", value(layers)});
    say("permute {}: ffn retained {:.4f} -> {:.4f}, wo {:.4f} -> {:.4f}", nm, ffn_b / std::max(ffn_t, 1e-30), ffn_a / std::max(ffn_t, 1e-30),
        v_b / std::max(v_t, 1e-30), v_a / std::max(v_t, 1e-30));
    return value(rep);
}

// ================================================================================================ actsparse
inline bool actsparse_ok(const Model& m, i64 l) {
    return m.spec.layers[std::size_t(l)].ffn == LayerSpec::Ffn::dense && !aliased_any(m, l, {"w_gate", "w_up", "w_down"});
}

// Runs layer l over the calibration states (block graph with the "mlp" input tapped) and returns a strided
// sample of |a| for a = silu(gate) * up (unthresholded), with the output-error weight a^2 ||w_down[:, j]||^2 of
// each sample; fills `out` with the block outputs (the thresholds already set on earlier layers apply) and
// `resid` with sum ||out||^2.
struct ActSamples { std::vector<std::pair<float, float>> v; double resid = 0; i64 total = 0; };
inline ActSamples act_samples(Model& m, i64 l, const Hidden& in, Hidden* out, i64 cap) {
    ActSamples A;
    const std::string p = std::format("layer.{}.", l);
    const i64 d = m.spec.d_model, T = in.seq;
    EmitOptions eo; eo.mode = EmitOptions::Mode::block; eo.block = l;
    Runner r(m.spec, m.w, T, eo, true, [&](const std::string& w) -> std::string { return w == p + "w_gate" ? "mlp" : ""; });
    if (out) { out->n = in.n; out->seq = T; out->d = d; out->x.resize(in.x.size()); }
    const auto& wg = m.w.at(p + "w_gate"); const auto& wu = m.w.at(p + "w_up"); const auto& wd = m.w.at(p + "w_down");
    const i64 f = wg.shape[0];
    std::vector<double> dn(static_cast<std::size_t>(f), 0.0);
    for (i64 rr = 0; rr < d; ++rr) for (i64 j = 0; j < f; ++j) dn[std::size_t(j)] += double(wd.data[std::size_t(rr * f + j)]) * wd.data[std::size_t(rr * f + j)];
    A.total = in.n * T * f;
    const i64 stride = std::max<i64>(1, (A.total + cap - 1) / cap);
    i64 idx = 0;
    for (i64 i = 0; i < in.n; ++i) {
        r.run_hidden({in.at(i), std::size_t(T * d)});
        auto o = r.output();
        if (out) std::copy(o.begin(), o.end(), out->at(i));
        for (float x : o) A.resid += double(x) * x;
        auto x = r.tap("tap:mlp");
        auto a = swiglu_hidden(x.data(), T, d, wg.data.data(), wu.data.data(), f);
        for (i64 k = (stride - idx % stride) % stride; k < T * f; k += stride) {
            const float v = a[std::size_t(k)];
            A.v.push_back({std::abs(v), float(double(v) * v * dn[std::size_t(k % f)])});
        }
        idx += T * f;
    }
    std::sort(A.v.begin(), A.v.end());
    return A;
}

// Fraction of exact zeros at each dense FFN's w_down input in the FORGE graph (what the sparse kernel would skip),
// on the calibration set, through the model as it is now. Layers without a dense FFN get -1.
inline std::vector<double> measure_act_sparsity(Model& m, const CalibSet& calib) {
    std::vector<double> frac(static_cast<std::size_t>(m.spec.n_layers()), -1.0);
    Hidden h = Hidden::embed(m, calib), nx;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        if (m.spec.layers[std::size_t(l)].ffn != LayerSpec::Ffn::dense || m.spec.factored(l, "w_down")) { layer_forward(m, l, h, nx); std::swap(h, nx); continue; }
        const std::string wn = std::format("layer.{}.w_down", l);
        EmitOptions eo; eo.mode = EmitOptions::Mode::block; eo.block = l;
        Runner r(m.spec, m.w, h.seq, eo, true, [&](const std::string& w) -> std::string { return w == wn ? "down" : ""; });
        nx.n = h.n; nx.seq = h.seq; nx.d = h.d; nx.x.resize(h.x.size());
        i64 z = 0, tot = 0;
        for (i64 i = 0; i < h.n; ++i) {
            r.run_hidden({h.at(i), std::size_t(h.seq * h.d)});
            auto o = r.output(); std::copy(o.begin(), o.end(), nx.at(i));
            for (float v : r.tap("tap:down")) { z += v == 0.f; ++tot; }
        }
        frac[std::size_t(l)] = tot ? double(z) / double(tot) : 0.0;
        std::swap(h, nx);
    }
    return frac;
}

inline value stage_actsparse(Ctx& c, const Stage& st) {
    Model& m = c.m;
    const double p = st.num("p", 0.5);
    if (st.str("mode", "threshold") != "threshold") die("actsparse: only mode=threshold");
    const std::string alloc = st.str("alloc", "uniform");
    const i64 cap = i64(st.num("samples", 1 << 21));
    const double max_p = st.num("max_p", std::min(0.95, p + 0.3));
    jobj rep{{"stage", value(st.text())}};
    std::map<i64, double> layer_p;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) if (actsparse_ok(m, l)) layer_p[l] = p;
    if (alloc == "eba") {
        auto grid = sparsity_grid(0.025, max_p);
        std::vector<Unit> units;
        Hidden h = Hidden::embed(m, c.calib), nx;
        for (i64 l = 0; l < m.spec.n_layers(); ++l) {
            if (!actsparse_ok(m, l)) { layer_forward(m, l, h, nx); std::swap(h, nx); continue; }
            auto A = act_samples(m, l, h, &nx, cap);
            std::swap(h, nx);
            const double scale = double(A.total) / double(std::max<std::size_t>(A.v.size(), 1));
            std::vector<double> pre(A.v.size() + 1, 0.0);
            for (std::size_t i = 0; i < A.v.size(); ++i) pre[i + 1] = pre[i] + double(A.v[i].second);
            Unit u; u.name = std::to_string(l); u.layer = l; u.params = m.w.at(std::format("layer.{}.w_gate", l)).shape[0];
            for (double g : grid) u.err.push_back(scale * pre[std::size_t(std::llround(g * double(A.v.size())))] / std::max(A.resid, 1e-30));
            units.push_back(u);
        }
        auto a = eba_allocate(units, grid, p, 0.0, max_p);
        std::vector<value> al;
        for (auto& [k, v] : a) { layer_p[std::stoll(k)] = v; al.push_back(value(jobj{{"layer", value(std::stoll(k))}, {"p", value(v)}})); }
        rep.push_back({"allocation", value(al)});
    } else if (alloc != "uniform") die("actsparse: unknown alloc " + alloc);

    Hidden h = Hidden::embed(m, c.calib), nx;
    std::vector<value> layers;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        if (!layer_p.count(l)) { layer_forward(m, l, h, nx); std::swap(h, nx); continue; }
        auto A = act_samples(m, l, h, nullptr, cap);
        const double pl = layer_p[l];
        double t = 0;
        if (pl > 0 && !A.v.empty()) {
            const std::size_t k = std::min(A.v.size() - 1, std::size_t(std::max<double>(0.0, std::ceil(pl * double(A.v.size())) - 1)));
            t = A.v[k].first;
        }
        m.spec.layers[std::size_t(l)].act_thresh = t;
        layer_forward(m, l, h, nx);   // with the threshold: the next layer is calibrated on the sparsified stream
        std::swap(h, nx);
        layers.push_back(value(jobj{{"layer", value(l)}, {"p", value(pl)}, {"thresh", value(t)}}));
        say("actsparse layer {}: p {:.3f} threshold {:.4g}", l, pl, t);
    }
    if (st.str("measure", "1") == "1") {
        auto fr = measure_act_sparsity(m, c.calib);
        double s = 0; i64 n = 0;
        std::vector<value> mv;
        for (double x : fr) { mv.push_back(value(x)); if (x >= 0) { s += x; ++n; } }
        rep.push_back({"measured", value(mv)});
        rep.push_back({"measured_mean", value(n ? s / double(n) : 0.0)});
    }
    rep.push_back({"layers", value(layers)});
    return value(rep);
}

inline const bool reg_heads = register_stage("heads", &stage_heads);
inline const bool reg_permute = register_stage("permute", &stage_permute);
inline const bool reg_actsparse = register_stage("actsparse", &stage_actsparse);

}  // namespace forge::compress

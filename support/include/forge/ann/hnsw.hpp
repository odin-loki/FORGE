#pragma once
// forge::ann::hnsw: Hierarchical Navigable Small World graph (Malkov & Yashunin) with a deterministic parallel build.
//
//   hnsw<std::vector<float>, ann::l2> ix({.M = 16, .ef_construction = 200, .ef_search = 160, .seed = 7});
//   for (...) ix.add(id, vec);          // queued; searchable after build()
//   ix.build(&pool);                    // may be called again after more add()s (incremental)
//   auto hits = ix.search(q, 10);       // ordered by (dist, id); const, thread-safe
//   ix.save(conn, "ann_main");  auto ix2 = hnsw<V, M>::load(conn, "ann_main");
//
// Determinism (a fixed seed and insertion order give the identical graph for ANY pool size, also nullptr):
//   * a node's level comes from splitmix64(seed, its insertion index), never from a shared random stream;
//   * build() inserts the queued nodes in batches whose sizes depend only on how many nodes are already in the graph
//     (max(1, built / 16), at most 128);
//   * phase 1 (parallel, read-only): every node of a batch searches the graph FROZEN as it was before the batch (efConstruction
//     candidates per level, greedy descent from the entry point) and also considers the earlier nodes of its own batch by brute
//     force, then picks its neighbours with the paper's heuristic (a candidate is kept if it is closer to the new node than to
//     every neighbour already kept; pruned ones refill up to M);
//   * phase 2 (serial, in insertion order): commit the nodes, their forward links and the entry point;
//   * phase 3 (parallel over target nodes, each target owned by one task): add the back links in insertion order; a
//     list that exceeds its cap (2M at level 0, M above) is re-selected with the same heuristic.
//   Every comparison orders by (distance, node index), so ties cannot depend on timing.
// Not provided: removal and updates. Search before build() sees only the nodes already built.
#include <forge/ann/metrics.hpp>
#include <forge/core/error.hpp>
#include <forge/db/sqlite.hpp>
#include <forge/task/algorithms.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <string>
#include <string_view>
#include <vector>

namespace forge::ann {

struct hnsw_params {
    std::size_t M = 16, ef_construction = 200, ef_search = 160;   // ef_search 160: recall@10 0.9695 on 20000 random 64-d vectors (64 gave 0.858; docs/ann.md)
    std::uint64_t seed = 0x5eed;
};

template <class V, class M> requires metric<M, V>
class hnsw {
public:
    explicit hnsw(hnsw_params p = {}, M m = {}) : p_(p), m_(std::move(m)) {
        if (p_.M < 2) p_.M = 2;
        if (p_.ef_construction < p_.M) p_.ef_construction = p_.M;
        if (p_.ef_search < 1) p_.ef_search = 1;
    }

    // Queue a vector (searchable after the next build()).
    void add(std::uint64_t id, V v) {
        node n; n.id = id; n.v = std::move(v); n.level = level_for_(nodes_.size());
        nodes_.push_back(std::move(n));
    }
    std::size_t size() const { return nodes_.size(); }       // added
    std::size_t built() const { return built_; }             // searchable
    const hnsw_params& params() const { return p_; }

    void build(forge::task::pool* pool = nullptr) {
        while (built_ < nodes_.size()) {
            const std::size_t b0 = built_, sz = std::min<std::size_t>(std::max<std::size_t>(1, b0 / 16), 128), b1 = std::min(nodes_.size(), b0 + sz);
            insert_batch_(pool, b0, b1);
            built_ = b1;
        }
    }

    std::vector<hit> search(const V& q, std::size_t k, std::size_t ef = 0) const {
        if (built_ == 0 || k == 0) return {};
        if (ef == 0) ef = p_.ef_search;
        ef = std::max(ef, k);
        std::uint32_t cur = entry_; float cd = dist_(q, nodes_[cur].v);
        for (int l = max_level_; l > 0; --l) descend_(q, cur, cd, l);
        auto c = search_layer_(q, {cand{cd, cur}}, ef, 0, built_);
        std::vector<hit> out; out.reserve(std::min(k, c.size()));
        for (auto& x : c) out.push_back({nodes_[x.idx].id, x.d});
        std::sort(out.begin(), out.end(), hit_before);
        if (out.size() > k) out.resize(k);
        return out;
    }

    // A hash of the whole graph (ids, levels, links, entry point): equal graphs have equal hashes.
    std::uint64_t graph_hash() const {
        std::uint64_t h = 1469598103934665603ull;
        auto mix = [&](std::uint64_t x) { h = (h ^ detail::splitmix64(x)) * 1099511628211ull; };
        mix(built_); mix(entry_); mix(std::uint64_t(max_level_ + 1));
        for (std::size_t i = 0; i < built_; ++i) {
            mix(nodes_[i].id); mix(std::uint64_t(nodes_[i].level));
            for (auto& l : nodes_[i].links) { mix(l.size()); for (auto x : l) mix(x); }
        }
        return h;
    }
    std::size_t edges() const { std::size_t e = 0; for (std::size_t i = 0; i < built_; ++i) for (auto& l : nodes_[i].links) e += l.size(); return e; }

    // ---- persistence: one SQLite table (kind, k, data BLOB), see docs/ann.md ------------------------------------
    // Saves the built nodes (throws if some are still queued). Replaces `table` in one transaction.
    void save(forge::db::connection& c, std::string_view table) const {
        if (built_ != nodes_.size()) forge::raise(forge::errc::invalid_argument, "hnsw::save: {} nodes are queued; call build() first", nodes_.size() - built_);
        const std::string t = forge::db::quote_ident(table);
        forge::db::transaction tx(c);
        c.exec("DROP TABLE IF EXISTS " + t);
        c.exec("CREATE TABLE " + t + "(kind TEXT NOT NULL, k INTEGER NOT NULL, data BLOB NOT NULL, PRIMARY KEY(kind, k)) WITHOUT ROWID");
        auto ins = c.prepare("INSERT INTO " + t + "(kind, k, data) VALUES(?1, ?2, ?3)");
        std::vector<std::byte> b;
        detail::put1(b, std::uint64_t(0x31574e5348474f46ull));   // "FOGHSNW1"
        detail::put1(b, std::uint64_t(p_.M)); detail::put1(b, std::uint64_t(p_.ef_construction)); detail::put1(b, std::uint64_t(p_.ef_search)); detail::put1(b, p_.seed);
        ins.run("params", 0, b);
        b.clear();
        detail::put1(b, std::uint64_t(built_)); detail::put1(b, std::uint32_t(entry_)); detail::put1(b, std::int32_t(max_level_));
        ins.run("graph", 0, b);
        for (std::size_t i = 0; i < built_; ++i) {
            const node& n = nodes_[i]; b.clear();
            detail::put1(b, n.id); detail::put1(b, std::int32_t(n.level));
            std::vector<std::byte> vb; codec<V>::encode(n.v, vb);
            detail::put1(b, std::uint32_t(vb.size())); b.insert(b.end(), vb.begin(), vb.end());
            for (const auto& l : n.links) { detail::put1(b, std::uint32_t(l.size())); detail::put(b, l.data(), l.size()); }
            ins.run("node", std::int64_t(i), b);
        }
        tx.commit();
    }
    static hnsw load(forge::db::connection& c, std::string_view table, M m = {}) {
        const std::string t = forge::db::quote_ident(table);
        auto bad = [&](const char* what) { forge::raise(forge::errc::format_error, "hnsw::load({}): {}", table, what); };
        auto get = [&](const char* kind, std::int64_t k) {
            auto st = c.prepare("SELECT data FROM " + t + " WHERE kind=?1 AND k=?2"); st.bind(1, kind).bind(2, k);
            if (!st.step()) bad(kind);
            return st.get<std::vector<std::byte>>(0);
        };
        auto pb = get("params", 0); if (pb.size() != 40) bad("params");
        std::uint64_t magic, v[4]; std::memcpy(&magic, pb.data(), 8); std::memcpy(v, pb.data() + 8, 32);
        if (magic != 0x31574e5348474f46ull) bad("not an hnsw table");
        hnsw_params p; p.M = v[0]; p.ef_construction = v[1]; p.ef_search = v[2]; p.seed = v[3];
        hnsw h(p, std::move(m));
        auto gb = get("graph", 0); if (gb.size() != 16) bad("graph");
        std::uint64_t n; std::uint32_t entry; std::int32_t ml; std::memcpy(&n, gb.data(), 8); std::memcpy(&entry, gb.data() + 8, 4); std::memcpy(&ml, gb.data() + 12, 4);
        h.nodes_.reserve(n);
        auto st = c.prepare("SELECT k, data FROM " + t + " WHERE kind='node' ORDER BY k");
        for (auto& r : st.rows()) {
            if (r.get<std::int64_t>(0) != std::int64_t(h.nodes_.size())) bad("node numbering");
            auto b = r.get<std::span<const std::byte>>(1);
            node nd; std::size_t o = 0;
            auto need = [&](std::size_t x) { if (o + x > b.size()) bad("truncated node"); };
            need(16); std::memcpy(&nd.id, b.data(), 8); std::int32_t lv; std::memcpy(&lv, b.data() + 8, 4); std::uint32_t vl; std::memcpy(&vl, b.data() + 12, 4); o = 16;
            if (lv < 0 || lv > 64) bad("level");
            nd.level = lv; need(vl); nd.v = codec<V>::decode(b.subspan(o, vl)); o += vl;
            nd.links.resize(std::size_t(lv) + 1);
            for (auto& l : nd.links) {
                need(4); std::uint32_t c32; std::memcpy(&c32, b.data() + o, 4); o += 4;
                need(std::size_t(c32) * 4); l.resize(c32); if (c32) std::memcpy(l.data(), b.data() + o, std::size_t(c32) * 4); o += std::size_t(c32) * 4;
            }
            h.nodes_.push_back(std::move(nd));
        }
        if (h.nodes_.size() != n) bad("node count");
        for (auto& nd : h.nodes_) for (std::size_t l = 0; l < nd.links.size(); ++l) for (auto x : nd.links[l])
            if (x >= n || h.nodes_[x].level < int(l)) bad("link out of range");
        if (n && (entry >= n || h.nodes_[entry].level != ml)) bad("entry point");
        h.built_ = n; h.entry_ = entry; h.max_level_ = n ? ml : -1;
        return h;
    }

private:
    struct node { std::uint64_t id = 0; V v{}; int level = 0; std::vector<std::vector<std::uint32_t>> links; };
    struct cand { float d; std::uint32_t idx; };
    static bool cand_before(const cand& a, const cand& b) { return a.d != b.d ? a.d < b.d : a.idx < b.idx; }
    struct cand_after { bool operator()(const cand& a, const cand& b) const { return cand_before(b, a); } };   // min-heap on cand_before
    struct cand_less { bool operator()(const cand& a, const cand& b) const { return cand_before(a, b); } };    // max-heap (worst on top)

    float dist_(const V& a, const V& b) const { float d = float(m_.distance(a, b)); return d != d ? std::numeric_limits<float>::infinity() : d; }
    int level_for_(std::size_t idx) const {
        const std::uint64_t h = detail::splitmix64(p_.seed ^ (std::uint64_t(idx) + 1) * 0x9E3779B97F4A7C15ull);
        const double u = (double(h >> 11) + 1.0) / 9007199254740992.0;   // (0, 1]
        const double mL = 1.0 / std::log(double(p_.M));
        return std::min(int(std::floor(-std::log(u) * mL)), 30);
    }
    std::size_t cap_(int level) const { return level == 0 ? 2 * p_.M : p_.M; }

    // per-thread visited marks (epoch counter: no clearing between searches)
    struct visited {
        std::vector<std::uint32_t>& m; std::uint32_t e;
        explicit visited(std::size_t n) : m(store_()), e(++epoch_()) {
            if (m.size() < n) m.resize(n, 0);
            if (e == 0) { std::fill(m.begin(), m.end(), 0); e = epoch_() = 1; }
        }
        bool seen(std::uint32_t i) { if (m[i] == e) return true; m[i] = e; return false; }
        static std::vector<std::uint32_t>& store_() { thread_local std::vector<std::uint32_t> s; return s; }
        static std::uint32_t& epoch_() { thread_local std::uint32_t e = 0; return e; }
    };

    // Best-first search of one level over nodes with index < limit. Returns the ef closest, ascending.
    std::vector<cand> search_layer_(const V& q, const std::vector<cand>& eps, std::size_t ef, int level, std::size_t limit) const {
        visited vis(limit);
        std::priority_queue<cand, std::vector<cand>, cand_after> cands;
        std::priority_queue<cand, std::vector<cand>, cand_less> res;
        for (const cand& e : eps) { if (vis.seen(e.idx)) continue; cands.push(e); res.push(e); if (res.size() > ef) res.pop(); }
        while (!cands.empty()) {
            const cand c = cands.top();
            if (res.size() >= ef && cand_before(res.top(), c)) break;
            cands.pop();
            for (std::uint32_t nb : nodes_[c.idx].links[std::size_t(level)]) {
                if (nb >= limit || vis.seen(nb)) continue;
                const cand x{dist_(q, nodes_[nb].v), nb};
                if (res.size() < ef || cand_before(x, res.top())) { cands.push(x); res.push(x); if (res.size() > ef) res.pop(); }
            }
        }
        std::vector<cand> out(res.size());
        for (std::size_t i = out.size(); i-- > 0;) { out[i] = res.top(); res.pop(); }
        return out;
    }
    void descend_(const V& q, std::uint32_t& cur, float& cd, int level) const {
        for (bool moved = true; moved;) {
            moved = false;
            for (std::uint32_t nb : nodes_[cur].links[std::size_t(level)]) {
                const float d = dist_(q, nodes_[nb].v);
                if (cand_before({d, nb}, {cd, cur})) { cd = d; cur = nb; moved = true; }
            }
        }
    }
    // The paper's heuristic over candidates sorted ascending by (d, idx) with d = distance to the point being linked.
    std::vector<std::uint32_t> select_(const std::vector<cand>& cs, std::size_t cap) const {
        std::vector<std::uint32_t> keep, pruned;
        for (const cand& c : cs) {
            if (keep.size() >= cap) break;
            bool good = true;
            for (std::uint32_t r : keep) if (dist_(nodes_[c.idx].v, nodes_[r].v) < c.d) { good = false; break; }
            (good ? keep : pruned).push_back(c.idx);
        }
        for (std::size_t i = 0; keep.size() < cap && i < pruned.size(); ++i) keep.push_back(pruned[i]);
        return keep;
    }

    void insert_batch_(forge::task::pool* pool, std::size_t b0, std::size_t b1) {
        const std::size_t cnt = b1 - b0;
        const int frozen_max = max_level_; const std::uint32_t frozen_entry = entry_;
        std::vector<std::vector<std::vector<std::uint32_t>>> sel(cnt);   // [node in batch][level] -> neighbours
        forge::task::parallel_blocks(pool, cnt, 1, [&](std::size_t bi, std::size_t, std::size_t) {
            const std::size_t i = b0 + bi; const node& nd = nodes_[i]; const int L = nd.level;
            sel[bi].resize(std::size_t(L) + 1);
            std::vector<cand> eps;
            if (b0 > 0) {
                std::uint32_t cur = frozen_entry; float cd = dist_(nd.v, nodes_[cur].v);
                for (int l = frozen_max; l > L; --l) descend_(nd.v, cur, cd, l);
                eps = {cand{cd, cur}};
            }
            for (int l = L; l >= 0; --l) {
                std::vector<cand> cs;
                if (b0 > 0 && l <= frozen_max) { cs = search_layer_(nd.v, eps, p_.ef_construction, l, b0); eps = cs; }
                for (std::size_t j = b0; j < i; ++j) if (nodes_[j].level >= l) cs.push_back({dist_(nd.v, nodes_[j].v), std::uint32_t(j)});
                std::sort(cs.begin(), cs.end(), cand_before);
                if (cs.size() > p_.ef_construction) cs.resize(p_.ef_construction);
                sel[bi][std::size_t(l)] = select_(cs, cap_(l));
            }
        });
        // phase 2: commit in insertion order
        struct back { std::uint32_t target; int level; std::uint32_t from; };
        std::vector<back> backs;
        for (std::size_t bi = 0; bi < cnt; ++bi) {
            const std::size_t i = b0 + bi; node& nd = nodes_[i];
            nd.links.assign(std::size_t(nd.level) + 1, {});
            for (int l = 0; l <= nd.level; ++l) {
                nd.links[std::size_t(l)] = sel[bi][std::size_t(l)];
                for (std::uint32_t t : nd.links[std::size_t(l)]) backs.push_back({t, l, std::uint32_t(i)});
            }
            if (b0 == 0 && bi == 0) { entry_ = 0; max_level_ = nd.level; }
            else if (nd.level > max_level_) { entry_ = std::uint32_t(i); max_level_ = nd.level; }
        }
        // phase 3: back links, one task per target (stable sort keeps insertion order within a target)
        std::stable_sort(backs.begin(), backs.end(), [](const back& a, const back& b) { return a.target < b.target; });
        std::vector<std::size_t> starts;
        for (std::size_t i = 0; i < backs.size(); ++i) if (i == 0 || backs[i].target != backs[i - 1].target) starts.push_back(i);
        starts.push_back(backs.size());
        forge::task::parallel_blocks(pool, starts.size() - 1, 1, [&](std::size_t g, std::size_t, std::size_t) {
            node& t = nodes_[backs[starts[g]].target];
            for (int l = 0; l <= t.level; ++l) {
                bool touched = false;
                auto& lst = t.links[std::size_t(l)];
                for (std::size_t k = starts[g]; k < starts[g + 1]; ++k) if (backs[k].level == l) { lst.push_back(backs[k].from); touched = true; }
                if (!touched || lst.size() <= cap_(l)) continue;
                std::vector<cand> cs; cs.reserve(lst.size());
                for (std::uint32_t x : lst) cs.push_back({dist_(t.v, nodes_[x].v), x});
                std::sort(cs.begin(), cs.end(), cand_before);
                lst = select_(cs, cap_(l));
            }
        });
    }

    hnsw_params p_;
    M m_;
    std::vector<node> nodes_;
    std::size_t built_ = 0;
    std::uint32_t entry_ = 0;
    int max_level_ = -1;
};

} // namespace forge::ann

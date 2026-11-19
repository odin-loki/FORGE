#pragma once
// ANN over a db::shard_set: one HNSW graph per shard, saved inside that shard (table "ann" by default), searched in parallel
// on forge::task and merged by (dist, id).
//
// Design choice: per-shard graphs rather than one global graph. A shard is a self-contained, content-addressed file, so
// its graph is rebuilt and re-hashed only when its rows change (a global graph would change every shard on any insert),
// shards are built in parallel by the shard_writer, and a query can search just the shards a key range selects. The price is
// recall and cost: a query searches every selected shard's graph (the work grows with the shard count) and the union of the
// per-shard top-k contains the global top-k whenever each shard graph finds its own true top-k, so recall is the per-shard
// recall; docs/ann.md lists the measured recall against flat_index.
//
//   sc.finalize = ann::ann_finalize<fvec, ann::l2>("SELECT id, vec FROM t ORDER BY rowid", {.M = 16, .seed = 7}, {});
//   ann::shard_ann<fvec, ann::l2> ix(set);          auto hits = ix.search(q, 10, &pool);
#include <forge/ann/hnsw.hpp>
#include <forge/db/shards.hpp>
#include <memory>
#include <mutex>

namespace forge::ann {

// A shard_schema::finalize hook: reads (id INTEGER, vec BLOB) rows with `select_sql`, builds an hnsw (serially: shards are
// built in parallel) and saves it into `table` of the shard being written.
template <class V, class M>
std::function<void(db::connection&)> ann_finalize(std::string select_sql, hnsw_params p, M m, std::string table = "ann") {
    return [=](db::connection& c) {
        hnsw<V, M> h(p, m);
        auto st = c.prepare(select_sql);
        for (auto& r : st.rows()) h.add(std::uint64_t(r.get<std::int64_t>(0)), codec<V>::decode(r.get<std::span<const std::byte>>(1)));
        h.build(nullptr);
        h.save(c, table);
    };
}

template <class V, class M> requires metric<M, V>
class shard_ann {
public:
    explicit shard_ann(db::shard_set& s, std::string table = "ann", M m = {}) : set_(&s), table_(std::move(table)), m_(std::move(m)) {
        for (std::size_t i = 0; i < s.size(); ++i) slots_.push_back(std::make_unique<slot>());
    }
    // k closest over the given shards (default: all), ordered by (dist, id). `ef` as in hnsw::search (0: the graph's ef_search).
    std::vector<hit> search(const V& q, std::size_t k, forge::task::pool* tp = nullptr, std::size_t ef = 0, const std::vector<std::size_t>* shards = nullptr) {
        const auto idx = shards ? *shards : set_->all_shards();
        auto per = db::shard_map<std::vector<hit>>(*set_, tp, idx, [&](db::connection& c, std::size_t i) { return graph_(c, i)->search(q, k, ef); }, db::run_options{false, db::shard_schedule::dynamic});
        auto out = forge::task::merge_runs(tp, std::move(per), hit_before);
        if (out.size() > k) out.resize(k);
        return out;
    }
    std::size_t loaded() const { std::size_t n = 0; for (auto& s : slots_) n += s->g ? 1 : 0; return n; }
    void unload() { for (auto& s : slots_) { std::lock_guard g(s->m); s->g.reset(); } }

private:
    struct slot { std::mutex m; std::shared_ptr<const hnsw<V, M>> g; };
    std::shared_ptr<const hnsw<V, M>> graph_(db::connection& c, std::size_t i) {
        slot& s = *slots_[i]; std::lock_guard g(s.m);
        if (!s.g) s.g = std::make_shared<const hnsw<V, M>>(hnsw<V, M>::load(c, table_, m_));
        return s.g;
    }
    db::shard_set* set_; std::string table_; M m_;
    std::vector<std::unique_ptr<slot>> slots_;
};

} // namespace forge::ann

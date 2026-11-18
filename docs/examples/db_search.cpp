// forge::db shard sets, parallel BM25 top-k, forge::ann: build a small sharded database of synthetic documents, look keys up,
// run an exact BM25 search over the shards and an approximate nearest-neighbour search, and check both against brute force.
// Built and run by ctest as doc_example_db_search.
#include <forge/ann/ann.hpp>
#include <forge/db/shards.hpp>
#include <cstdio>
#include <filesystem>
#include <random>
using namespace forge;
using fvec = std::vector<float>;

int main() {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / "forge_doc_example_db";
    fs::remove_all(dir);
    task::pool pool(4);

    // 1. Write: rows sorted by key are cut into shards of at most 64 KiB (the default budget is 10 MiB).
    db::shard_schema sc;
    sc.ddl = "CREATE TABLE docs(k BLOB PRIMARY KEY, id INTEGER, body TEXT, vec BLOB) WITHOUT ROWID;"
             "CREATE VIRTUAL TABLE fts USING fts5(body, content='');";
    sc.table = "docs"; sc.columns = {"k", "id", "body", "vec"}; sc.fts_table = "fts";
    sc.finalize = [ann = ann::ann_finalize<fvec, ann::l2>("SELECT id, vec FROM docs ORDER BY k", {}, {})](db::connection& c) {
        c.exec("INSERT INTO fts(rowid, body) SELECT id, body FROM docs ORDER BY k");   // the index lives in the shard
        ann(c);                                                                         // so does an HNSW graph of its vectors
    };
    std::mt19937_64 g(1); std::normal_distribution<float> nd;
    std::vector<fvec> vecs;
    {
        db::shard_writer w(dir, sc, {.budget = 64 << 10, .presorted = true}, &pool);
        for (std::int64_t id = 1; id <= 3000; ++id) {
            fvec v(8); for (auto& f : v) f = nd(g); vecs.push_back(v);
            std::string body = "doc" + std::to_string(id) + " topic" + std::to_string(id % 7) + " word" + std::to_string(g() % 50);
            w.add({db::int_key(id), id, body, std::vector<std::byte>(reinterpret_cast<const std::byte*>(v.data()), reinterpret_cast<const std::byte*>(v.data()) + v.size() * 4)});
        }
        w.finish();   // writes manifest.json: shard names, key ranges, row counts, sizes, SHA-256 and BLAKE3
    }

    // 2. Open: the manifest is read, each shard's SHA-256 is checked the first time it is opened.
    auto set = db::open(dir);
    std::printf("%zu shards, %llu rows, %llu bytes\n", set.size(), (unsigned long long)set.manifest().rows, (unsigned long long)set.manifest().bytes);

    // 3. A key lookup opens only the shard whose key range contains the key.
    const auto key = db::int_key(1234);
    auto shards = set.shards_for_key(key);
    auto body = db::shard_query<std::string>(set, &pool, shards, "SELECT body FROM docs WHERE k = :k",
                                             [&](db::statement& st) { st.bind(":k", db::static_span_(key)); }, [](const db::row& r) { return r.get<std::string>(0); });
    std::printf("key 1234 -> %zu shard opened of %zu: %s\n", shards.size(), set.size(), body.at(0).c_str());

    // 4. BM25 over all shards with the global statistics: the same ranking as one table holding every row.
    auto hits = db::shard_bm25_topk(set, &pool, "fts", {"topic3", "word7"}, 5);
    for (auto& h : hits) std::printf("  bm25 id=%lld score=%.4f (shard %u)\n", (long long)h.id, h.score, h.shard);

    // 5. ANN: one HNSW graph per shard, searched in parallel and merged; compare with the exact answer.
    ann::shard_ann<fvec, ann::l2> index(set);
    ann::flat_index<fvec, ann::l2> exact;
    for (std::size_t i = 0; i < vecs.size(); ++i) exact.add(i + 1, vecs[i]);
    const fvec q = vecs[41];
    auto approx = index.search(q, 5, &pool), truth = exact.search(q, 5, &pool);
    std::size_t same = 0; for (auto& a : approx) for (auto& t : truth) same += a.id == t.id;
    std::printf("ann top-5 overlap with exact: %zu/5, nearest id %llu\n", same, (unsigned long long)approx[0].id);

    fs::remove_all(dir);
    return (hits.size() == 5 && approx[0].id == 42 && same >= 4) ? 0 : 1;
}

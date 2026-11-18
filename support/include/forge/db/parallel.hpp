#pragma once
// forge::db parallel layer: read-only connection pool, partitioned queries on forge::task, deterministic top-k,
// sort-merge, and a parallel bulk load.
//
//   db::read_pool rp("lean.fdb", 8);                         // 8 read-only connections to one file
//   task::pool tp(8);
//   auto parts = db::rowid_partitions(rp, "docs", 64);        // [lo, hi] rowid ranges, fixed count: results never depend on tp
//   auto sql   = db::range_sql("docs", "id, length(body)");   // SELECT id, length(body) FROM docs WHERE rowid BETWEEN :lo AND :hi ORDER BY rowid
//   auto rows  = db::parallel_query<std::pair<int64_t,int64_t>>(rp, &tp, sql, parts, [](const db::row& r) { return ...; });
//   auto top   = db::parallel_bm25_topk(rp, &tp, "fts_docs", db::fts5::query_from_text("bitvec add"), 10);
//
// Contract (every function): the result equals the serial run of the same call (tp == nullptr, or a pool of one
// thread) for any pool size. Partitions are fixed by the caller (or by rowid_partitions' count), partition results are
// combined in partition order, and ties are broken by id / by partition order, never by timing. A partitioned SQL
// statement must select its rows with `rowid BETWEEN :lo AND :hi` (named parameters :lo, :hi, both inclusive) and
// return them in rowid order (ORDER BY rowid) when the order of results matters; range_sql() builds that statement.
// Rows seen by different partitions come from different connections: for a consistent snapshot while a writer is
// active, open the database in WAL mode and call parallel functions between writes, or open it immutable.
//
// FTS5 note: bm25() takes its statistics (document counts, term frequencies) from the whole index, not from the rowid
// range, so the score of a row is the same in every partition and in a serial query: parallel_bm25_topk returns
// exactly fts5::search's ranking.
#include <forge/db/sqlite.hpp>
#include <forge/task/algorithms.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace forge::db {

// ---- connection pool -------------------------------------------------------------------------------------------
// N read-only connections to one database file, handed out one per user. Thread-safe.
class read_pool {
public:
    read_pool(std::string path, std::size_t n, open_options o = {}) : path_(std::move(path)) {
        o.read_only = true; o.create = false; o.wal = false;
        if (n == 0) n = 1;
        conns_.reserve(n);
        for (std::size_t i = 0; i < n; ++i) { conns_.push_back(std::make_unique<connection>(connection::open(path_, o))); free_.push_back(conns_.back().get()); }
    }
    read_pool(const read_pool&) = delete;
    read_pool& operator=(const read_pool&) = delete;
    std::size_t size() const { return conns_.size(); }
    const std::string& path() const { return path_; }

    class lease {
    public:
        lease(lease&& o) noexcept : p_(std::exchange(o.p_, nullptr)), c_(o.c_) {}
        lease(const lease&) = delete; lease& operator=(const lease&) = delete;
        ~lease() { if (p_) p_->release_(c_); }
        connection& operator*() const { return *c_; }
        connection* operator->() const { return c_; }
    private:
        friend class read_pool;
        lease(read_pool* p, connection* c) : p_(p), c_(c) {}
        read_pool* p_; connection* c_;
    };
    // Blocks until a connection is free.
    lease acquire() {
        std::unique_lock l(m_);
        cv_.wait(l, [&] { return !free_.empty(); });
        auto* c = free_.back(); free_.pop_back();
        return lease(this, c);
    }

private:
    void release_(connection* c) { { std::lock_guard l(m_); free_.push_back(c); } cv_.notify_one(); }
    std::string path_;
    std::vector<std::unique_ptr<connection>> conns_;
    std::vector<connection*> free_;
    std::mutex m_; std::condition_variable cv_;
};

// ---- partitions ------------------------------------------------------------------------------------------------
struct rowid_range { std::int64_t lo, hi; };   // inclusive

// Splits [min(rowid), max(rowid)] of `table` into at most `parts` ranges of (nearly) equal width. Empty table: no ranges.
inline std::vector<rowid_range> split_rowid_range(std::int64_t lo, std::int64_t hi, std::size_t parts) {
    std::vector<rowid_range> out;
    if (hi < lo || parts == 0) return out;
    const std::uint64_t span = std::uint64_t(hi) - std::uint64_t(lo) + 1;   // 0 when the full 64-bit range: treat as 2^64-1
    const std::uint64_t width = span == 0 ? ~std::uint64_t(0) : span;
    const std::uint64_t np = std::min<std::uint64_t>(parts, width);
    const std::uint64_t q = width / np, r = width % np;
    std::uint64_t cur = std::uint64_t(lo);
    for (std::uint64_t i = 0; i < np; ++i) {
        const std::uint64_t sz = q + (i < r ? 1 : 0);
        out.push_back({std::int64_t(cur), std::int64_t(cur + sz - 1)});
        cur += sz;
    }
    return out;
}
inline std::vector<rowid_range> rowid_partitions(connection& c, std::string_view table, std::size_t parts) {
    auto st = c.prepare("SELECT min(rowid), max(rowid) FROM " + quote_ident(table));
    if (!st.step() || st.current().is_null(0)) return {};
    return split_rowid_range(st.get<std::int64_t>(0), st.get<std::int64_t>(1), parts);
}
inline std::vector<rowid_range> rowid_partitions(read_pool& rp, std::string_view table, std::size_t parts) {
    auto l = rp.acquire(); return rowid_partitions(*l, table, parts);
}
// SELECT <select_list> FROM <table> WHERE rowid BETWEEN :lo AND :hi [AND (<extra_where>)] ORDER BY rowid
inline std::string range_sql(std::string_view table, std::string_view select_list, std::string_view extra_where = {}) {
    std::string s = "SELECT " + std::string(select_list) + " FROM " + quote_ident(table) + " WHERE rowid BETWEEN :lo AND :hi";
    if (!extra_where.empty()) s += " AND (" + std::string(extra_where) + ")";
    return s + " ORDER BY rowid";
}

// ---- generic partition map -------------------------------------------------------------------------------------
// out[i] = fn(connection&, parts[i]) with the partitions spread over `tp` (nullptr: one after another on the caller).
// Each call gets its own pooled connection. Results are in partition order.
template <class R, class Part, class Fn>
std::vector<R> parallel_map(read_pool& rp, task::pool* tp, const std::vector<Part>& parts, Fn&& fn) {
    std::vector<std::optional<R>> tmp(parts.size());
    task::parallel_blocks(tp, parts.size(), 1, [&](std::size_t i, std::size_t, std::size_t) {
        auto l = rp.acquire();
        tmp[i].emplace(fn(*l, parts[i]));
    });
    std::vector<R> out; out.reserve(tmp.size());
    for (auto& x : tmp) out.push_back(std::move(*x));
    return out;
}

namespace detail {
inline void bind_range(statement& st, const rowid_range& r) { st.bind(":lo", r.lo).bind(":hi", r.hi); }
} // namespace detail

// ---- query / scan / filter / reduce ----------------------------------------------------------------------------
// Rows of `sql` for every partition, mapped through rowfn(const row&) -> R, concatenated in partition order (= rowid
// order when the SQL says ORDER BY rowid).
template <class R, class RowFn>
std::vector<R> parallel_query(read_pool& rp, task::pool* tp, std::string_view sql, const std::vector<rowid_range>& parts, RowFn&& rowfn) {
    auto per = parallel_map<std::vector<R>>(rp, tp, parts, [&](connection& c, const rowid_range& r) {
        std::vector<R> v; auto st = c.prepare(sql); detail::bind_range(st, r);
        for (auto& row : st.rows()) v.push_back(rowfn(row));
        return v;
    });
    std::size_t n = 0; for (auto& v : per) n += v.size();
    std::vector<R> out; out.reserve(n);
    for (auto& v : per) for (auto& x : v) out.push_back(std::move(x));
    return out;
}
// visit(partition_index, const row&) for every row. Partitions run concurrently (so visit must be thread-safe across
// partitions); within one partition rows come in SQL order on one thread.
template <class Visit>
void parallel_scan(read_pool& rp, task::pool* tp, std::string_view sql, const std::vector<rowid_range>& parts, Visit&& visit) {
    task::parallel_blocks(tp, parts.size(), 1, [&](std::size_t i, std::size_t, std::size_t) {
        auto l = rp.acquire(); auto st = l->prepare(sql); detail::bind_range(st, parts[i]);
        for (auto& row : st.rows()) visit(i, row);
    });
}
// The rows with pred(row) true, mapped through proj(row), in partition order.
template <class R, class Pred, class Proj>
std::vector<R> parallel_filter(read_pool& rp, task::pool* tp, std::string_view sql, const std::vector<rowid_range>& parts, Pred&& pred, Proj&& proj) {
    auto per = parallel_map<std::vector<R>>(rp, tp, parts, [&](connection& c, const rowid_range& r) {
        std::vector<R> v; auto st = c.prepare(sql); detail::bind_range(st, r);
        for (auto& row : st.rows()) if (pred(row)) v.push_back(proj(row));
        return v;
    });
    std::vector<R> out;
    for (auto& v : per) for (auto& x : v) out.push_back(std::move(x));
    return out;
}
// result = op(...op(op(init, P0), P1)..., Pn-1), P_i = left fold of map(row) over partition i (partitions without rows
// are skipped). For an associative op this equals the serial fold; for any op it equals the serial run on the same partitions.
template <class T, class MapFn, class Op>
T parallel_reduce(read_pool& rp, task::pool* tp, std::string_view sql, const std::vector<rowid_range>& parts, T init, MapFn&& map, Op op) {
    auto per = parallel_map<std::optional<T>>(rp, tp, parts, [&](connection& c, const rowid_range& r) {
        std::optional<T> acc; auto st = c.prepare(sql); detail::bind_range(st, r);
        for (auto& row : st.rows()) { if (!acc) acc = map(row); else acc = op(std::move(*acc), map(row)); }
        return acc;
    });
    for (auto& a : per) if (a) init = op(std::move(init), std::move(*a));
    return init;
}

// ---- top-k -----------------------------------------------------------------------------------------------------
struct scored { std::int64_t id; double score; bool operator==(const scored&) const = default; };
// Result order: score descending, then id ascending. NaN scores sort last. A total order, so the answer is unique.
inline bool scored_before(const scored& a, const scored& b) {
    const bool an = a.score != a.score, bn = b.score != b.score;
    if (an != bn) return bn;
    if (!an && a.score != b.score) return a.score > b.score;
    return a.id < b.id;
}

// `sql` (partitioned as above) returns (id INTEGER, score REAL) rows. Returns the k best by (score desc, id asc).
inline std::vector<scored> parallel_topk(read_pool& rp, task::pool* tp, std::string_view sql, const std::vector<rowid_range>& parts, std::size_t k) {
    if (k == 0) return {};
    auto per = parallel_map<std::vector<scored>>(rp, tp, parts, [&](connection& c, const rowid_range& r) {
        std::vector<scored> heap;   // max-heap on "worst on top" for scored_before
        auto st = c.prepare(sql); detail::bind_range(st, r);
        for (auto& row : st.rows()) {
            scored s{row.get<std::int64_t>(0), row.get<double>(1)};
            if (heap.size() < k) { heap.push_back(s); std::push_heap(heap.begin(), heap.end(), scored_before); }
            else if (scored_before(s, heap.front())) { std::pop_heap(heap.begin(), heap.end(), scored_before); heap.back() = s; std::push_heap(heap.begin(), heap.end(), scored_before); }
        }
        std::sort_heap(heap.begin(), heap.end(), scored_before);
        return heap;
    });
    auto out = task::merge_runs(tp, std::move(per), scored_before);
    if (out.size() > k) out.resize(k);
    return out;
}

// FTS5 bm25 search of `table` for the MATCH expression `query`, partitioned by rowid. score = -bm25() (larger = better).
// Equal to fts5::search(conn, table, query, k, weights) run on one connection.
inline std::vector<scored> parallel_bm25_topk(read_pool& rp, task::pool* tp, std::string_view table, std::string_view query, std::size_t k,
                                              const std::vector<rowid_range>& parts, const std::vector<double>& weights = {}) {
    if (query.empty() || k == 0 || parts.empty()) return {};
    std::string t = quote_ident(table), w;
    for (double x : weights) w += ", " + std::to_string(x);
    const std::string sql = "SELECT rowid, -bm25(" + t + w + ") FROM " + t + " WHERE " + t + " MATCH " + quote_literal(query) + " AND rowid BETWEEN :lo AND :hi";
    return parallel_topk(rp, tp, sql, parts, k);
}
inline std::vector<scored> parallel_bm25_topk(read_pool& rp, task::pool* tp, std::string_view table, std::string_view query, std::size_t k,
                                              std::size_t nparts = 64, const std::vector<double>& weights = {}) {
    return parallel_bm25_topk(rp, tp, table, query, k, rowid_partitions(rp, table, nparts), weights);
}

// ---- sort ------------------------------------------------------------------------------------------------------
// Fetches the rows per partition, stable-sorts each partition's rows with cmp, then merges the runs with the stable
// parallel merge (task::merge_runs). The result equals std::stable_sort over the rows in partition order.
template <class R, class RowFn, class Cmp>
std::vector<R> parallel_sort(read_pool& rp, task::pool* tp, std::string_view sql, const std::vector<rowid_range>& parts, RowFn&& rowfn, Cmp cmp) {
    auto runs = parallel_map<std::vector<R>>(rp, tp, parts, [&](connection& c, const rowid_range& r) {
        std::vector<R> v; auto st = c.prepare(sql); detail::bind_range(st, r);
        for (auto& row : st.rows()) v.push_back(rowfn(row));
        std::stable_sort(v.begin(), v.end(), cmp);
        return v;
    });
    return task::merge_runs(tp, std::move(runs), cmp);
}

// ---- bulk load -------------------------------------------------------------------------------------------------
struct bulk_spec {
    std::string table;                    // target table (must exist)
    std::vector<std::string> columns;     // columns written, in the order fill() binds them
    std::size_t rows = 0;                 // total number of rows; fill(i, ...) is called for i in [0, rows)
    std::size_t parts = 8;                // staging databases, 1..120 (SQLITE_MAX_ATTACHED is 125)
    std::string conflict;                 // "", "OR IGNORE", "OR REPLACE": the INSERT's conflict clause
    std::filesystem::path staging_dir;    // where the staging files go (default: next to the target, else the temp dir)
};
// Parallel load: partition p owns rows [p*rows/parts, (p+1)*rows/parts); each fills its own staging database (one
// transaction, journal off) by calling fill(i, insert_statement), which binds the columns 1..n of row i (the loader steps the
// statement), then, in one transaction on the target, the staging tables are ATTACHed and copied in
// partition order with INSERT INTO target(cols) SELECT cols FROM stage ORDER BY rowid. The target therefore ends up
// with the same rows in the same order (and the same autogenerated rowids) as a serial loop over fill(0..rows-1).
// fill runs concurrently for different partitions. Returns the number of rows inserted. The staging files are
// deleted afterwards, also on failure (the target is rolled back).
template <class Fill>
std::int64_t parallel_bulk_load(task::pool* tp, connection& target, const bulk_spec& spec, Fill&& fill) {
    namespace fs = std::filesystem;
    if (spec.columns.empty()) raise(errc::invalid_argument, "db: parallel_bulk_load: no columns");
    const std::size_t parts = std::clamp<std::size_t>(spec.parts, 1, 120);
    static std::atomic<unsigned> ctr{0};
    fs::path dir = spec.staging_dir;
    if (dir.empty()) {
        const std::string tp_ = target.path();
        dir = (tp_.empty() || tp_ == ":memory:" || tp_[0] == ':') ? fs::temp_directory_path() : fs::path(tp_).parent_path();
        if (dir.empty()) dir = ".";
    }
    const std::string tag = "forge_stage_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" + std::to_string(ctr.fetch_add(1));
    std::vector<fs::path> files(parts);
    for (std::size_t p = 0; p < parts; ++p) files[p] = dir / (tag + "_" + std::to_string(p) + ".db");
    struct cleanup { std::vector<fs::path>& f; ~cleanup() { for (auto& x : f) { std::error_code ec; fs::remove(x, ec); } } } guard{files};

    std::string cols, qs;
    for (std::size_t i = 0; i < spec.columns.size(); ++i) { if (i) { cols += ", "; qs += ", "; } cols += quote_ident(spec.columns[i]); qs += "?"; }
    task::parallel_blocks(tp, parts, 1, [&](std::size_t p, std::size_t, std::size_t) {
        const std::size_t b = p * spec.rows / parts, e = (p + 1) * spec.rows / parts;
        open_options o; o.foreign_keys = false; o.pragmas = {"journal_mode=OFF", "synchronous=OFF", "locking_mode=EXCLUSIVE", "temp_store=MEMORY"};
        auto c = connection::open(files[p].string(), o);
        c.exec("CREATE TABLE s(" + cols + ")");
        transaction tx(c);
        auto ins = c.prepare("INSERT INTO s(" + cols + ") VALUES(" + qs + ")");
        for (std::size_t i = b; i < e; ++i) { ins.reset(); fill(i, ins); ins.step(); }
        tx.commit();
    });
    for (std::size_t p = 0; p < parts; ++p) target.attach(files[p].string(), "stg" + std::to_string(p));
    std::int64_t total = 0;
    try {
        transaction tx(target, tx_mode::immediate);
        const std::string verb = "INSERT " + spec.conflict + " INTO " + quote_ident(spec.table) + "(" + cols + ") SELECT " + cols + " FROM ";
        for (std::size_t p = 0; p < parts; ++p) {
            target.exec(verb + quote_ident("stg" + std::to_string(p)) + ".s ORDER BY rowid");
            total += target.changes();
        }
        tx.commit();
    } catch (...) {
        for (std::size_t p = 0; p < parts; ++p) { try { target.detach("stg" + std::to_string(p)); } catch (...) {} }
        throw;
    }
    for (std::size_t p = 0; p < parts; ++p) target.detach("stg" + std::to_string(p));
    return total;
}

} // namespace forge::db

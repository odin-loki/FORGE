#pragma once
// forge::db shard sets: the default storage layout of forge::db. A database is a directory of SQLite shard files, each
// at most `budget` bytes (default 10 MiB), holding consecutive ranges of rows sorted by a user key, plus a small text
// manifest (manifest.json). The layout is the search structure: key lookups, prefix and range scans open only the
// shards whose [min, max] key range can hold the answer; full scans, top-k, BM25 and ANN run over the shards in parallel
// on forge::task. A single SQLite file (db::open_single, or db::open on a plain file) is a one-shard set.
//
//   shard_schema sc;  sc.ddl = "CREATE TABLE t(k BLOB PRIMARY KEY, body TEXT) WITHOUT ROWID; ...";
//   sc.table = "t"; sc.columns = {"k", "body"}; sc.key_index = 0;
//   db::shard_writer w("lean.db", sc, {.presorted = true}, &pool);
//   w.add({std::string("key1"), std::string("text")}); ...  auto manifest = w.finish();
//
//   auto set = db::open("lean.db");                      // verifies each shard's SHA-256 the first time it is opened
//   auto idx = set.shards_for_prefix("key");             // binary search over the manifest's key ranges
//   auto rows = db::shard_query<std::string>(set, &pool, idx, "SELECT body FROM t WHERE k >= :lo AND k < :hi ORDER BY k", binder, rowfn);
//
// Reproducible builds: for the same rows, schema, finalize hook and options the shard files are byte-identical on any
// thread count (page_size fixed, rows in key order, VACUUM, journal_mode=DELETE, no clock or random values; the cut
// points depend only on the data), so a git repository stores only the shards that changed. Every shard is checked
// against the budget on its final size, after VACUUM; an over-budget shard is split in halves and rebuilt.
//
// Hashes: the manifest records SHA-256 per shard (what git and external pins use) and BLAKE3 (the library's fast internal
// integrity check and cache key, hash/blake3.hpp). The BLAKE3 of the manifest text identifies the whole database
// (shard_set::id()); it keys cached query results in a forge::cache::store.
#include <forge/cache/cache.hpp>
#include <forge/db/parallel.hpp>
#include <forge/db/sqlite.hpp>
#include <forge/hash/blake3.hpp>
#include <forge/hash/sha256.hpp>
#include <forge/json/json.hpp>
#include <forge/os/fs.hpp>
#include <forge/task/algorithms.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace forge::db {

using value = std::variant<std::monostate, std::int64_t, double, std::string, std::vector<std::byte>>;

// Big-endian with the sign bit flipped: byte order of the key equals numeric order.
inline std::string int_key(std::int64_t v) {
    std::uint64_t u = std::uint64_t(v) ^ (1ull << 63); std::string s(8, '\0');
    for (int i = 0; i < 8; ++i) s[std::size_t(i)] = char(u >> (56 - 8 * i));
    return s;
}
inline std::string to_hex_(std::string_view s) { static const char* x = "0123456789abcdef"; std::string o; for (unsigned char c : s) { o += x[c >> 4]; o += x[c & 15]; } return o; }
inline std::string from_hex_(std::string_view h) {
    auto v = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; };
    if (h.size() % 2) raise(errc::parse_error, "shards: odd hex string");
    std::string o;
    for (std::size_t i = 0; i < h.size(); i += 2) { int a = v(h[i]), b = v(h[i + 1]); if (a < 0 || b < 0) raise(errc::parse_error, "shards: bad hex"); o += char(a * 16 + b); }
    return o;
}

// ---- manifest --------------------------------------------------------------------------------------------------
struct shard_info {
    std::string file;                          // file name inside the database directory
    std::string min_key, max_key;              // key bytes of the first and last row (empty for a single-file set)
    std::uint64_t rows = 0, bytes = 0;
    std::string sha256, blake3;                // hex; empty for a single-file set
    std::int64_t fts_rows = -1, fts_tokens = -1;   // FTS5 table statistics (manifest of a set written with schema.fts_table); -1 unknown
};
struct shard_manifest {
    static constexpr const char* kFormat = "forge-shards-1";
    std::uint32_t schema_version = 1;
    std::string table, fts_table;
    std::uint32_t page_size = 4096;
    std::uint64_t budget_bytes = 0, rows = 0, bytes = 0;
    json::object stats;                        // caller-registered global statistics
    std::vector<shard_info> shards;

    std::string to_text() const {
        std::string o = "{\n\"format\": \"" + std::string(kFormat) + "\",\n\"schema_version\": " + std::to_string(schema_version) +
                        ",\n\"table\": " + json::dump(json::value(table)) + ",\n\"fts_table\": " + json::dump(json::value(fts_table)) +
                        ",\n\"page_size\": " + std::to_string(page_size) + ",\n\"budget_bytes\": " + std::to_string(budget_bytes) +
                        ",\n\"rows\": " + std::to_string(rows) + ",\n\"bytes\": " + std::to_string(bytes) +
                        ",\n\"stats\": " + json::dump(json::value(stats)) + ",\n\"shards\": [\n";
        for (std::size_t i = 0; i < shards.size(); ++i) {
            const auto& s = shards[i];
            o += "{\"file\":" + json::dump(json::value(s.file)) + ",\"min\":\"" + to_hex_(s.min_key) + "\",\"max\":\"" + to_hex_(s.max_key) + "\",\"rows\":" +
                 std::to_string(s.rows) + ",\"bytes\":" + std::to_string(s.bytes) + ",\"sha256\":\"" + s.sha256 + "\",\"blake3\":\"" + s.blake3 + "\"";
            if (s.fts_rows >= 0) o += ",\"fts_rows\":" + std::to_string(s.fts_rows) + ",\"fts_tokens\":" + std::to_string(s.fts_tokens);
            o += i + 1 < shards.size() ? "},\n" : "}\n";
        }
        return o + "]\n}\n";
    }
    static shard_manifest parse(std::string_view text) {
        auto j = json::parse(text);
        if (j.str_or("format", "") != kFormat) raise(errc::format_error, "shards: not a {} manifest", kFormat);
        shard_manifest m;
        m.schema_version = std::uint32_t(j.num_or("schema_version", 1)); m.table = j.str_or("table", ""); m.fts_table = j.str_or("fts_table", "");
        m.page_size = std::uint32_t(j.num_or("page_size", 4096)); m.budget_bytes = std::uint64_t(j.num_or("budget_bytes", 0));
        m.rows = std::uint64_t(j.num_or("rows", 0)); m.bytes = std::uint64_t(j.num_or("bytes", 0));
        if (auto* s = j.find("stats"); s && s->is_obj()) m.stats = s->obj();
        for (auto& e : j["shards"].arr()) {
            shard_info s; s.file = e["file"].str(); s.min_key = from_hex_(e["min"].str()); s.max_key = from_hex_(e["max"].str());
            s.rows = std::uint64_t(e["rows"].num()); s.bytes = std::uint64_t(e["bytes"].num()); s.sha256 = e["sha256"].str(); s.blake3 = e["blake3"].str();
            s.fts_rows = std::int64_t(e.num_or("fts_rows", -1)); s.fts_tokens = std::int64_t(e.num_or("fts_tokens", -1));
            m.shards.push_back(std::move(s));
        }
        return m;
    }
};

// ---- FTS5 totals -----------------------------------------------------------------------------------------------
// {rows, tokens} of an FTS5 table: what FTS5's own bm25() reads as the row count and the total token count.
inline std::pair<std::int64_t, std::int64_t> fts_totals(connection& c, std::string_view fts_table) {
    const std::int64_t rows = c.scalar<std::int64_t>("SELECT count(*) FROM " + quote_ident(fts_table));
    c.exec("DROP TABLE IF EXISTS temp.forge_fv");
    c.exec("CREATE VIRTUAL TABLE temp.forge_fv USING fts5vocab(main, " + quote_ident(fts_table) + ", 'row')");
    const std::int64_t tokens = c.scalar_or<std::int64_t>(0, "SELECT coalesce(sum(cnt), 0) FROM temp.forge_fv");
    c.exec("DROP TABLE temp.forge_fv");
    return {rows, tokens};
}

// ---- writer ----------------------------------------------------------------------------------------------------
struct shard_schema {
    std::string ddl;                           // CREATE statements run in every shard (tables, indexes, FTS tables)
    std::string table;                         // the table rows are inserted into
    std::vector<std::string> columns;          // columns written, in the order of add()'s values
    std::size_t key_index = 0;                 // columns[key_index] holds the sort key (stored as a BLOB; bytes compare as memcmp)
    std::uint32_t schema_version = 1;
    std::string fts_table;                     // optional FTS5 table filled by `finalize`: its row/token totals go in the manifest
    std::function<void(connection&)> finalize; // after the rows are inserted, before VACUUM: FTS rebuild, ANN save, ... Must be deterministic.
};
struct shard_options {
    std::uint64_t budget = 10ull << 20;        // bytes per shard file (checked on the final, vacuumed file)
    std::uint32_t page_size = 4096;
    bool presorted = false;                    // rows arrive in non-decreasing key order: streamed in waves of `wave_bytes`; else all are buffered and sorted
    std::uint64_t wave_bytes = 256ull << 20;
    std::filesystem::path blob_dir;            // optional content-addressed store: each shard is also kept as <blob_dir>/<sha256>.fdb
};

namespace detail {
struct built_shard { std::filesystem::path tmp; shard_info info; };
inline std::uint64_t raw_bytes(const std::vector<value>& r) {
    std::uint64_t n = 0;
    for (auto& v : r) n += 8 + (std::holds_alternative<std::string>(v) ? std::get<std::string>(v).size() : std::holds_alternative<std::vector<std::byte>>(v) ? std::get<std::vector<std::byte>>(v).size() : 0);
    return n;
}
inline void hash_file(const std::filesystem::path& p, std::string& b3, std::string& s256) {
    std::ifstream f(p, std::ios::binary); hash::blake3 h3; hash::sha256 h2; std::string buf(1 << 20, '\0');
    while (f) { f.read(buf.data(), std::streamsize(buf.size())); const auto n = std::size_t(f.gcount()); if (!n) break; h3.update(buf.data(), n); h2.update(buf.data(), n); }
    b3 = hash::blake3::hex(h3.final()); s256 = hash::sha256::hex(h2.final());
}
inline void bind_value(statement& st, int i, const value& v) {
    std::visit([&](const auto& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::monostate>) st.bind(i, nullptr); else st.bind(i, x);
    }, v);
}
} // namespace detail

class shard_writer {
public:
    shard_writer(std::filesystem::path dir, shard_schema schema, shard_options opt = {}, task::pool* tp = nullptr)
        : dir_(std::move(dir)), sc_(std::move(schema)), opt_(opt), tp_(tp) {
        if (sc_.columns.empty() || sc_.key_index >= sc_.columns.size() || sc_.table.empty()) raise(errc::invalid_argument, "shard_writer: schema needs a table, columns and a key column");
        std::filesystem::create_directories(dir_);
        // a directory that already holds shards of an older build: remove what this writer replaces on finish()
    }
    // One row of values in the order of schema.columns. The key column must hold a std::string (or bytes).
    void add(std::vector<value> row) {
        if (row.size() != sc_.columns.size()) raise(errc::invalid_argument, "shard_writer::add: expected {} values, got {}", sc_.columns.size(), row.size());
        entry e; auto& kv = row[sc_.key_index];
        if (auto* s = std::get_if<std::string>(&kv)) e.key = *s;
        else if (auto* b = std::get_if<std::vector<std::byte>>(&kv)) e.key.assign(reinterpret_cast<const char*>(b->data()), b->size());
        else raise(errc::invalid_argument, "shard_writer::add: the key must be a string or a byte vector");
        kv = std::vector<std::byte>(reinterpret_cast<const std::byte*>(e.key.data()), reinterpret_cast<const std::byte*>(e.key.data()) + e.key.size());   // keys are stored as BLOBs
        if (opt_.presorted && have_last_ && e.key < last_key_) raise(errc::invalid_argument, "shard_writer::add: keys must be non-decreasing when presorted");
        last_key_ = e.key; have_last_ = true;
        e.bytes = detail::raw_bytes(row); e.row = std::move(row);
        buf_bytes_ += e.bytes; rows_.push_back(std::move(e));
        if (opt_.presorted && buf_bytes_ >= opt_.wave_bytes) flush_();
    }
    void set_stat(std::string name, json::value v) {
        for (auto& [k, x] : stats_) if (k == name) { x = std::move(v); return; }
        stats_.emplace_back(std::move(name), std::move(v));
    }
    // Writes the remaining shards and manifest.json (atomically). Returns the manifest.
    shard_manifest finish() {
        if (!opt_.presorted)
            task::parallel_sort(tp_, rows_.begin(), rows_.end(), [](const entry& a, const entry& b) { return a.key < b.key; });   // stable: equal keys keep arrival order
        flush_();
        shard_manifest m; m.schema_version = sc_.schema_version; m.table = sc_.table; m.fts_table = sc_.fts_table; m.page_size = opt_.page_size;
        m.budget_bytes = opt_.budget; m.shards = std::move(done_); m.stats = stats_;
        for (auto& s : m.shards) { m.rows += s.rows; m.bytes += s.bytes; }
        // drop shard files of an earlier build that this one no longer names
        std::error_code ec;
        for (auto& de : std::filesystem::directory_iterator(dir_, ec)) {
            const auto fn = de.path().filename().string();
            if (fn.rfind("shard-", 0) == 0 && fn.size() > 4 && fn.substr(fn.size() - 4) == ".fdb" && std::none_of(m.shards.begin(), m.shards.end(), [&](const shard_info& s) { return s.file == fn; }))
                std::filesystem::remove(de.path(), ec);
        }
        auto st = os::fs::write_file_atomic((dir_ / "manifest.json").string(), m.to_text());
        if (!st) raise(errc::io_error, "shard_writer: cannot write manifest: {}", st.err.what);
        return m;
    }

private:
    struct entry { std::string key; std::vector<value> row; std::uint64_t bytes = 0; };
    using built = detail::built_shard;

    built build_range_one_(std::size_t b, std::size_t e, const std::filesystem::path& tmp) {
        std::error_code ec; std::filesystem::remove(tmp, ec);
        std::int64_t frows = -1, ftok = -1;
        {
            open_options o; o.foreign_keys = false; o.busy_timeout_ms = 0;
            o.pragmas = {"page_size=" + std::to_string(opt_.page_size), "journal_mode=OFF", "synchronous=OFF", "temp_store=MEMORY", "locking_mode=EXCLUSIVE", "auto_vacuum=NONE"};
            auto c = connection::open(tmp.string(), o);
            c.exec(sc_.ddl);
            std::string cols, qs;
            for (std::size_t i = 0; i < sc_.columns.size(); ++i) { if (i) { cols += ", "; qs += ", "; } cols += quote_ident(sc_.columns[i]); qs += "?"; }
            {
                transaction tx(c);
                auto ins = c.prepare("INSERT INTO " + quote_ident(sc_.table) + "(" + cols + ") VALUES(" + qs + ")");
                for (std::size_t r = b; r < e; ++r) {
                    ins.reset(); ins.clear_bindings();
                    for (std::size_t i = 0; i < sc_.columns.size(); ++i) detail::bind_value(ins, int(i + 1), rows_[r].row[i]);
                    ins.step();
                }
                if (sc_.finalize) sc_.finalize(c);
                tx.commit();
            }
            c.exec("PRAGMA user_version=" + std::to_string(sc_.schema_version));
            if (!sc_.fts_table.empty()) { auto t = fts_totals(c, sc_.fts_table); frows = t.first; ftok = t.second; }
            c.exec("PRAGMA locking_mode=NORMAL"); c.exec("PRAGMA journal_mode=DELETE");
            c.exec("VACUUM");
        }
        built bs; bs.tmp = tmp;
        bs.info.rows = e - b; bs.info.bytes = std::filesystem::file_size(tmp);
        bs.info.min_key = rows_[b].key; bs.info.max_key = rows_[e - 1].key; bs.info.fts_rows = frows; bs.info.fts_tokens = ftok;
        return bs;
    }
    // Builds rows [b, e) as one shard; splits in halves while a file exceeds the budget.
    void build_budgeted_(std::size_t b, std::size_t e, std::vector<built>& out) {
        auto tmp = dir_ / (".tmp-" + std::to_string(wave_) + "-" + std::to_string(b) + "-" + std::to_string(e) + ".fdb");
        built bs = build_range_one_(b, e, tmp);
        if (bs.info.bytes > opt_.budget) {
            std::error_code ec; std::filesystem::remove(tmp, ec);
            if (e - b == 1) raise(errc::invalid_argument, "shard_writer: one row needs {} bytes, more than the {}-byte budget", bs.info.bytes, opt_.budget);
            const std::size_t mid = b + (e - b) / 2;
            build_budgeted_(b, mid, out); build_budgeted_(mid, e, out);
            return;
        }
        out.push_back(std::move(bs));
    }
    void flush_() {
        if (rows_.empty()) return;
        const std::size_t n = rows_.size();
        std::vector<std::pair<std::size_t, std::size_t>> chunks;
        std::vector<built> first;
        std::size_t pos = 0;
        auto cut = [&](std::uint64_t target) { std::size_t b = pos; std::uint64_t acc = 0; while (pos < n && (acc < target || pos == b)) acc += rows_[pos++].bytes; return std::make_pair(b, pos); };
        if (ratio_ == 0) {   // calibrate: file bytes per raw byte, from the first shard
            auto [b, e] = cut(std::uint64_t(double(opt_.budget) * 0.5));
            std::uint64_t raw = 0; for (std::size_t i = b; i < e; ++i) raw += rows_[i].bytes;
            build_budgeted_(b, e, first);
            std::uint64_t fb = 0; for (auto& f : first) fb += f.info.bytes;
            ratio_ = std::max(0.2, double(fb) / double(std::max<std::uint64_t>(raw, 1)));
        }
        const std::uint64_t target = std::uint64_t(double(opt_.budget) * 0.9 / ratio_);
        while (pos < n) chunks.push_back(cut(target));
        std::vector<std::vector<built>> per(chunks.size());
        task::parallel_blocks(tp_, chunks.size(), 1, [&](std::size_t i, std::size_t, std::size_t) { build_budgeted_(chunks[i].first, chunks[i].second, per[i]); });
        std::vector<built> all = std::move(first);
        for (auto& v : per) for (auto& x : v) all.push_back(std::move(x));
        std::uint64_t raw = 0, fb = 0; for (auto& r : rows_) raw += r.bytes; for (auto& a : all) fb += a.info.bytes;
        // finalize names and hashes in order
        std::vector<std::string> b3(all.size()), s2(all.size());
        task::parallel_blocks(tp_, all.size(), 1, [&](std::size_t i, std::size_t, std::size_t) { detail::hash_file(all[i].tmp, b3[i], s2[i]); });
        for (std::size_t i = 0; i < all.size(); ++i) {
            char nm[32]; std::snprintf(nm, sizeof nm, "shard-%06zu.fdb", next_name_++);
            auto fin = dir_ / nm; std::error_code ec; std::filesystem::remove(fin, ec);
            std::filesystem::rename(all[i].tmp, fin);
            all[i].info.file = nm; all[i].info.blake3 = b3[i]; all[i].info.sha256 = s2[i];
            if (!opt_.blob_dir.empty()) {
                std::filesystem::create_directories(opt_.blob_dir);
                auto blob = opt_.blob_dir / (s2[i] + ".fdb");
                if (!std::filesystem::exists(blob)) { std::filesystem::create_hard_link(fin, blob, ec); if (ec) std::filesystem::copy_file(fin, blob, ec); }
            }
            done_.push_back(std::move(all[i].info));
        }
        if (raw) ratio_ = std::max(0.2, double(fb) / double(raw));
        rows_.clear(); rows_.shrink_to_fit(); buf_bytes_ = 0; ++wave_;
    }

    std::filesystem::path dir_; shard_schema sc_; shard_options opt_; task::pool* tp_;
    std::vector<entry> rows_; std::uint64_t buf_bytes_ = 0;
    std::string last_key_; bool have_last_ = false;
    std::vector<shard_info> done_; std::size_t next_name_ = 0, wave_ = 0; double ratio_ = 0;
    json::object stats_;
};
inline shard_writer create(std::filesystem::path dir, shard_schema schema, shard_options opt = {}, task::pool* tp = nullptr) { return shard_writer(std::move(dir), std::move(schema), opt, tp); }

// ---- BM25 auxiliary function with global statistics ------------------------------------------------------------
namespace detail {
// forge_bm25(fts, avgdl, idf_blob, w0, w1, ...): FTS5's bm25 (k1 = 1.2, b = 0.75) with the average document length and
// the per-phrase IDF supplied by the caller (computed from global statistics) instead of from the shard's own index.
// The arithmetic is the same expression, in the same order, as FTS5's built-in, so a shard scored with the global
// statistics gives exactly the value the unsharded table would.
inline void forge_bm25_fn(const Fts5ExtensionApi* api, Fts5Context* ctx, sqlite3_context* out, int nVal, sqlite3_value** apVal) {
    if (nVal < 2) { sqlite3_result_error(out, "forge_bm25(fts, avgdl, idf_blob, weights...)", -1); return; }
    const double k1 = 1.2, b = 0.75;
    const double avgdl = sqlite3_value_double(apVal[0]);
    const int nb = sqlite3_value_bytes(apVal[1]); const void* blob = sqlite3_value_blob(apVal[1]);
    const int nPhrase = nb / int(sizeof(double));
    double stack_idf[32], stack_freq[32];
    std::vector<double> heap_buf;
    double* idf = stack_idf; double* freq = stack_freq;
    if (nPhrase > 32) { heap_buf.resize(std::size_t(nPhrase) * 2); idf = heap_buf.data(); freq = idf + nPhrase; }
    if (nPhrase) std::memcpy(idf, blob, std::size_t(nPhrase) * sizeof(double));
    for (int i = 0; i < nPhrase; ++i) freq[i] = 0.0;
    int nInst = 0; int rc = api->xInstCount(ctx, &nInst);
    for (int i = 0; rc == SQLITE_OK && i < nInst; ++i) {
        int ip, ic, io; rc = api->xInst(ctx, i, &ip, &ic, &io);
        if (rc == SQLITE_OK) { const double w = (nVal - 2 > ic) ? sqlite3_value_double(apVal[2 + ic]) : 1.0; if (ip < nPhrase) freq[ip] += w; }
    }
    double D = 0.0;
    if (rc == SQLITE_OK) { int nTok = 0; rc = api->xColumnSize(ctx, -1, &nTok); D = double(nTok); }
    if (rc != SQLITE_OK) { sqlite3_result_error_code(out, rc); return; }
    double score = 0.0;
    for (int i = 0; i < nPhrase; ++i) score += idf[i] * ((freq[i] * (k1 + 1.0)) / (freq[i] + k1 * (1 - b + b * D / avgdl)));
    sqlite3_result_double(out, -1.0 * score);
}
inline void register_forge_bm25(connection& c) {
    fts5_api* api = nullptr;
    auto st = c.prepare("SELECT fts5(?1)");
    sqlite3_bind_pointer(st.handle(), 1, static_cast<void*>(&api), "fts5_api_ptr", nullptr);
    st.step();
    if (!api) raise(errc::unsupported, "db: FTS5 is not available in this SQLite build");
    const int rc = api->xCreateFunction(api, "forge_bm25", nullptr, &forge_bm25_fn, nullptr);
    if (rc != SQLITE_OK) throw_db(c.handle(), rc, "xCreateFunction forge_bm25");
}
inline void prefetch_file(const std::string& path) {
#if defined(__unix__) || defined(__APPLE__)
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
#if defined(POSIX_FADV_WILLNEED)
        (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
#endif
        ::close(fd);
    }
#else
    (void)path;
#endif
}
} // namespace detail

inline std::span<const std::byte> static_span_(const std::string& s) { return {reinterpret_cast<const std::byte*>(s.data()), s.size()}; }

// ---- shard set -------------------------------------------------------------------------------------------------
enum class verify_mode { none, size, blake3, sha256 };
struct shard_open_options {
    verify_mode verify = verify_mode::sha256;  // checked the first time a shard is opened (shard_set::verify_all checks all now)
    std::uint64_t max_open_bytes = 1ull << 30; // LRU budget over the shards that hold open connections (file bytes)
    std::int64_t cache_kib = 1024;             // SQLite page cache per connection (shards are mmapped, so this stays small)
    bool mmap = true;                          // PRAGMA mmap_size = the shard's size
    bool immutable = true;                     // open read-only with immutable=1: no locking, no journal or WAL files
    std::filesystem::path blob_dir;            // content-addressed fallback: <blob_dir>/<sha256>.fdb when <dir>/<file> is missing
    std::function<void(connection&, std::size_t)> on_open;   // called for each new connection (shard index)
};

class shard_set {
public:
    // Directory holding manifest.json, or the manifest file itself.
    static shard_set open(const std::filesystem::path& p, shard_open_options o = {}) {
        namespace fs = std::filesystem;
        if (fs::is_regular_file(p) && p.filename() != "manifest.json") return single_file(p, std::move(o));   // a plain SQLite file
        const fs::path mf = fs::is_directory(p) ? p / "manifest.json" : p;
        std::ifstream f(mf, std::ios::binary); std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (text.empty()) raise(errc::io_error, "shard_set: cannot read {}", mf.string());
        shard_set s; s.m_ = shard_manifest::parse(text); s.dir_ = mf.parent_path(); s.opt_ = std::move(o);
        { hash::blake3 h; h.update(text.data(), text.size()); s.id_ = hash::blake3::hex(h.final()); }
        for (std::size_t i = 0; i < s.m_.shards.size(); ++i) {
            const auto& sh = s.m_.shards[i];
            if (sh.min_key > sh.max_key || (i && s.m_.shards[i - 1].max_key > sh.min_key)) raise(errc::integrity_error, "shard_set: manifest key ranges are not sorted at shard {}", i);
        }
        s.init_slots_();
        return s;
    }
    // One SQLite file as a one-shard set (no key range, no hashes).
    static shard_set single_file(const std::filesystem::path& file, shard_open_options o = {}) {
        shard_set s; s.opt_ = std::move(o); s.opt_.verify = verify_mode::none; s.opt_.immutable = false; s.dir_ = file.parent_path();
        shard_info si; si.file = file.filename().string(); si.bytes = std::filesystem::file_size(file);
        s.m_.shards.push_back(si); s.m_.bytes = si.bytes; s.id_ = "single:" + file.string(); s.single_ = true;
        s.init_slots_();
        return s;
    }
    shard_set(shard_set&&) = default; shard_set& operator=(shard_set&&) = default;

    std::size_t size() const { return m_.shards.size(); }
    const shard_info& info(std::size_t i) const { return m_.shards[i]; }
    const shard_manifest& manifest() const { return m_; }
    const std::string& id() const { return id_; }                  // BLAKE3 of the manifest text
    std::vector<std::size_t> all_shards() const { std::vector<std::size_t> v(size()); for (std::size_t i = 0; i < v.size(); ++i) v[i] = i; return v; }
    std::filesystem::path path_of(std::size_t i) const {
        auto p = dir_ / m_.shards[i].file;
        if (!std::filesystem::exists(p) && !opt_.blob_dir.empty() && !m_.shards[i].sha256.empty()) p = opt_.blob_dir / (m_.shards[i].sha256 + ".fdb");
        return p;
    }

    // Shards whose key range can contain `key`; [lo, hi] (inclusive); or any key starting with `prefix`. Binary search.
    std::vector<std::size_t> shards_for_range(const std::string& lo, const std::string& hi) const {
        if (single_) return {0};
        const auto& sh = m_.shards;
        std::size_t a = std::size_t(std::partition_point(sh.begin(), sh.end(), [&](const shard_info& s) { return s.max_key < lo; }) - sh.begin());   // first max >= lo
        std::size_t b = std::size_t(std::partition_point(sh.begin(), sh.end(), [&](const shard_info& s) { return s.min_key <= hi; }) - sh.begin());  // first min > hi
        std::vector<std::size_t> v; for (std::size_t i = a; i < b; ++i) v.push_back(i);
        return v;
    }
    std::vector<std::size_t> shards_for_key(const std::string& key) const { return shards_for_range(key, key); }
    std::vector<std::size_t> shards_for_prefix(const std::string& prefix) const {
        if (single_) return {0};
        std::string succ = prefix;
        while (!succ.empty() && static_cast<unsigned char>(succ.back()) == 0xFF) succ.pop_back();
        const auto& sh = m_.shards;
        std::size_t a = std::size_t(std::partition_point(sh.begin(), sh.end(), [&](const shard_info& s) { return s.max_key < prefix; }) - sh.begin());
        std::vector<std::size_t> v;
        if (!succ.empty()) succ.back() = char(static_cast<unsigned char>(succ.back()) + 1);
        for (std::size_t i = a; i < sh.size(); ++i) { if (!succ.empty() && sh[i].min_key >= succ) break; v.push_back(i); }
        return v;
    }
    // [lo, hi) bounds for SQL: the successor of `prefix` ("" = unbounded, for prefixes of 0xFF bytes only).
    static std::string prefix_end(std::string prefix) {
        while (!prefix.empty() && static_cast<unsigned char>(prefix.back()) == 0xFF) prefix.pop_back();
        if (!prefix.empty()) prefix.back() = char(static_cast<unsigned char>(prefix.back()) + 1);
        return prefix;
    }

    class lease {
    public:
        lease(lease&& o) noexcept : s_(std::exchange(o.s_, nullptr)), i_(o.i_), c_(std::move(o.c_)) {}
        lease(const lease&) = delete; lease& operator=(const lease&) = delete;
        ~lease() { if (s_) s_->release_(i_, std::move(c_)); }
        connection& operator*() const { return *c_; }
        connection* operator->() const { return c_.get(); }
    private:
        friend class shard_set;
        lease(shard_set* s, std::size_t i, std::unique_ptr<connection> c) : s_(s), i_(i), c_(std::move(c)) {}
        shard_set* s_; std::size_t i_; std::unique_ptr<connection> c_;
    };
    // A read-only connection to shard i (verified on first use). Thread-safe; release by destroying the lease.
    lease acquire(std::size_t i) {
        slot& sl = *slots_[i];
        std::unique_ptr<connection> c;
        {
            std::lock_guard g(sl.m);
            if (!sl.verified) { verify_shard_(i, opt_.verify); sl.verified = true; }
            if (!sl.free.empty()) { c = std::move(sl.free.back()); sl.free.pop_back(); }
            ++sl.in_use;
        }
        if (!c) {
            try { c = open_(i); } catch (...) { std::lock_guard g(sl.m); --sl.in_use; throw; }
            std::lock_guard g(*mup_);
            if (!sl.counted) { sl.counted = true; open_bytes_ += m_.shards[i].bytes; }
        }
        { std::lock_guard g(*mup_); sl.tick = ++clock_; }
        evict_();
        return lease(this, i, std::move(c));
    }
    void prefetch(std::size_t i) const { detail::prefetch_file(path_of(i).string()); }
    // Checks every shard against the manifest now (in parallel when a pool is given). Throws errc::integrity_error.
    void verify_all(task::pool* tp = nullptr, verify_mode mode = verify_mode::sha256) {
        task::parallel_blocks(tp, size(), 1, [&](std::size_t i, std::size_t, std::size_t) { verify_shard_(i, mode); });
    }
    std::size_t open_shards() { std::lock_guard g(*mup_); std::size_t n = 0; for (auto& s : slots_) n += s->counted ? 1 : 0; return n; }
    // Drops every pooled connection (e.g. before replacing files).
    void close_all() { std::lock_guard g(*mup_); for (auto& s : slots_) { std::lock_guard gs(s->m); s->free.clear(); if (s->counted && s->in_use == 0) { s->counted = false; } } open_bytes_ = 0; for (auto& s : slots_) if (s->counted) open_bytes_ += m_.shards[std::size_t(&s - slots_.data())].bytes; }
    const shard_open_options& options() const { return opt_; }
    bool is_single_file() const { return single_; }

private:
    struct slot { std::mutex m; std::vector<std::unique_ptr<connection>> free; unsigned in_use = 0; bool verified = false, counted = false; std::uint64_t tick = 0; };
    shard_set() = default;
    void init_slots_() { for (std::size_t i = 0; i < m_.shards.size(); ++i) slots_.push_back(std::make_unique<slot>()); }
    void verify_shard_(std::size_t i, verify_mode mode) {
        if (mode == verify_mode::none || single_) return;
        const auto& s = m_.shards[i]; const auto p = path_of(i);
        std::error_code ec; const auto sz = std::filesystem::file_size(p, ec);
        if (ec) raise(errc::io_error, "shard_set: cannot read shard {} ({})", s.file, p.string());
        if (sz != s.bytes) raise(errc::integrity_error, "shard_set: shard {} is {} bytes, the manifest says {}", s.file, sz, s.bytes);
        if (mode == verify_mode::size) return;
        std::string b3, s2; detail::hash_file(p, b3, s2);
        if (mode == verify_mode::blake3 ? b3 != s.blake3 : (s2 != s.sha256 || b3 != s.blake3))
            raise(errc::integrity_error, "shard_set: shard {} does not match the manifest (sha256 {} expected {})", s.file, s2, s.sha256);
    }
    std::unique_ptr<connection> open_(std::size_t i) {
        open_options o; o.read_only = true; o.create = false; o.immutable = opt_.immutable; o.busy_timeout_ms = 2000; o.foreign_keys = false;
        if (opt_.cache_kib > 0) o.cache_size_kib = int(opt_.cache_kib);
        if (opt_.mmap) o.mmap_size = std::int64_t(std::max<std::uint64_t>(m_.shards[i].bytes, 1 << 20));
        auto c = std::make_unique<connection>(connection::open(path_of(i).string(), o));
        detail::register_forge_bm25(*c);
        if (opt_.on_open) opt_.on_open(*c, i);
        return c;
    }
    void release_(std::size_t i, std::unique_ptr<connection> c) {
        slot& sl = *slots_[i];
        std::lock_guard g(sl.m); sl.free.push_back(std::move(c)); --sl.in_use;
    }
    void evict_() {
        std::lock_guard g(*mup_);
        while (open_bytes_ > opt_.max_open_bytes) {
            slot* victim = nullptr; std::size_t vi = 0;
            for (std::size_t i = 0; i < slots_.size(); ++i) {
                slot& s = *slots_[i];
                if (!s.counted || (victim && s.tick >= victim->tick)) continue;
                std::unique_lock gl(s.m, std::try_to_lock);
                if (gl.owns_lock() && s.in_use == 0) { victim = &s; vi = i; }
            }
            if (!victim) return;
            std::lock_guard gl(victim->m); victim->free.clear(); victim->counted = false; open_bytes_ -= m_.shards[vi].bytes;
        }
    }
    shard_manifest m_; std::filesystem::path dir_; shard_open_options opt_; std::string id_; bool single_ = false;
    std::vector<std::unique_ptr<slot>> slots_;
    std::unique_ptr<std::mutex> mup_ = std::make_unique<std::mutex>(); std::uint64_t clock_ = 0, open_bytes_ = 0;
};
using database = shard_set;
// The default way to open a database: a shard set (a directory with manifest.json), or a one-shard set for a plain file.
inline shard_set open(const std::filesystem::path& p, shard_open_options o = {}) { return shard_set::open(p, std::move(o)); }
// Explicit single-file access for small or temporary databases.
inline connection open_single(std::string_view path, const open_options& o = {}) { return connection::open(path, o); }

// ---- parallel work over shards ---------------------------------------------------------------------------------
enum class shard_schedule { dynamic, blocked };   // dynamic: each worker takes the next shard; blocked: each worker takes a contiguous run (page-cache and NUMA locality)
struct run_options { bool prefetch = true; shard_schedule schedule = shard_schedule::dynamic; };

// out[i] = fn(connection&, shard_index) for each shard in `shards`, results in the order of `shards`. Prefetches the files first.
template <class R, class Fn>
std::vector<R> shard_map(shard_set& s, task::pool* tp, const std::vector<std::size_t>& shards, Fn&& fn, run_options ro = {}) {
    std::vector<std::optional<R>> tmp(shards.size());
    if (ro.prefetch) for (auto i : shards) s.prefetch(i);
    auto one = [&](std::size_t k) { auto l = s.acquire(shards[k]); tmp[k].emplace(fn(*l, shards[k])); };
    if (ro.schedule == shard_schedule::blocked && tp && tp->size() > 1) {
        const std::size_t blk = (shards.size() + tp->size() - 1) / tp->size();
        task::parallel_blocks(tp, shards.size(), std::max<std::size_t>(blk, 1), [&](std::size_t, std::size_t b, std::size_t e) { for (std::size_t k = b; k < e; ++k) one(k); });
    } else task::parallel_blocks(tp, shards.size(), 1, [&](std::size_t k, std::size_t, std::size_t) { one(k); });
    std::vector<R> out; out.reserve(tmp.size());
    for (auto& x : tmp) out.push_back(std::move(*x));
    return out;
}

// Rows of `sql` on each shard, mapped by rowfn(const row&), concatenated in shard order (= key order when sql orders by key).
// `bind(statement&)` binds the parameters (the same on every shard).
template <class R, class Bind, class RowFn>
std::vector<R> shard_query(shard_set& s, task::pool* tp, const std::vector<std::size_t>& shards, std::string_view sql, Bind&& bind, RowFn&& rowfn, run_options ro = {}) {
    auto per = shard_map<std::vector<R>>(s, tp, shards, [&](connection& c, std::size_t) {
        std::vector<R> v; auto st = c.prepare(sql); bind(st);
        for (auto& r : st.rows()) v.push_back(rowfn(r));
        return v;
    }, ro);
    std::vector<R> out; for (auto& v : per) for (auto& x : v) out.push_back(std::move(x));
    return out;
}
// Same, for the rows of one key range [lo, hi): sql uses :lo and :hi. Only the shards that can hold the range are opened.
template <class R, class RowFn>
std::vector<R> shard_range_query(shard_set& s, task::pool* tp, const std::string& lo, const std::string& hi, std::string_view sql, RowFn&& rowfn, run_options ro = {}) {
    auto idx = s.shards_for_range(lo, hi);   // [lo, hi] superset of [lo, hi)
    return shard_query<R>(s, tp, idx, sql, [&](statement& st) { st.bind(":lo", static_span_(lo)).bind(":hi", static_span_(hi)); }, rowfn, ro);
}

// Top-k over shards: sql returns (id INTEGER, score REAL) rows. Order: score descending, then id ascending.
template <class Bind>
std::vector<scored> shard_topk(shard_set& s, task::pool* tp, const std::vector<std::size_t>& shards, std::string_view sql, Bind&& bind, std::size_t k, run_options ro = {}) {
    if (k == 0) return {};
    auto per = shard_map<std::vector<scored>>(s, tp, shards, [&](connection& c, std::size_t) {
        std::vector<scored> heap; auto st = c.prepare(sql); bind(st);
        for (auto& row : st.rows()) {
            scored x{row.get<std::int64_t>(0), row.get<double>(1)};
            if (heap.size() < k) { heap.push_back(x); std::push_heap(heap.begin(), heap.end(), scored_before); }
            else if (scored_before(x, heap.front())) { std::pop_heap(heap.begin(), heap.end(), scored_before); heap.back() = x; std::push_heap(heap.begin(), heap.end(), scored_before); }
        }
        std::sort_heap(heap.begin(), heap.end(), scored_before);
        return heap;
    }, ro);
    auto out = task::merge_runs(tp, std::move(per), scored_before);
    if (out.size() > k) out.resize(k);
    return out;
}

// ---- exact BM25 over shards ------------------------------------------------------------------------------------
struct shard_hit {
    std::uint32_t shard; std::int64_t id; double score;   // id = the FTS table's rowid
    bool operator==(const shard_hit&) const = default;
};
inline bool shard_hit_before(const shard_hit& a, const shard_hit& b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.id != b.id) return a.id < b.id;
    return a.shard < b.shard;
}
struct bm25_options {
    std::string op = "OR";                  // "OR" or "AND" between the terms
    std::vector<double> weights;            // per-column weights (empty: all 1)
    bool prune = true;                      // skip shards whose best possible score cannot reach the current k-th
};
struct bm25_report { std::size_t shards = 0, with_hits = 0, scored = 0, skipped_bound = 0; std::int64_t rows = 0, tokens = 0; std::vector<std::int64_t> df; };

// The k best rows for `terms` (each one plain word, one phrase each) over all shards, scored with the global statistics
// (total rows and tokens, and each term's document count summed over the shards), so the ranking and the scores equal
// those of FTS5's bm25() on one table holding all the rows. Ordered by (score desc, id asc, shard asc); score = -bm25().
// Document counts come from per-shard count queries (what FTS5's own bm25 runs per query); the shard totals from the manifest.
// Pruning: a term adds less than (k1+1) * idf to a row, so a shard whose matching terms sum below the current k-th best score
// cannot contribute; shards are scored best bound first and skipped when their bound is below the k-th score (strictly), which
// never changes the result.
inline std::vector<shard_hit> shard_bm25_topk(shard_set& s, task::pool* tp, std::string_view fts_table, const std::vector<std::string>& terms_in, std::size_t k,
                                              const bm25_options& bo = {}, bm25_report* rep = nullptr, run_options ro = {}) {
    std::vector<std::string> terms;
    for (auto& t : terms_in) if (!t.empty() && std::find(terms.begin(), terms.end(), t) == terms.end()) terms.push_back(t);
    if (terms.empty() || k == 0) return {};
    const std::size_t ns = s.size(), nt = terms.size();
    std::string expr;
    for (std::size_t i = 0; i < nt; ++i) { if (i) expr += " " + bo.op + " "; expr += fts5::quote_phrase(terms[i]); }
    // phase 1: per-shard totals (manifest, or computed once for a single file) and document counts per term
    struct counts { std::int64_t rows = 0, tokens = 0; std::vector<std::int64_t> df; };
    auto cs = shard_map<counts>(s, tp, s.all_shards(), [&](connection& c, std::size_t i) {
        counts r; r.df.assign(nt, 0);
        const auto& si = s.info(i);
        if (si.fts_rows >= 0) { r.rows = si.fts_rows; r.tokens = si.fts_tokens; } else { auto t = fts_totals(c, fts_table); r.rows = t.first; r.tokens = t.second; }
        if (r.rows == 0) return r;
        auto st = c.prepare("SELECT count(*) FROM " + quote_ident(fts_table) + " WHERE " + quote_ident(fts_table) + " MATCH ?1");
        for (std::size_t t = 0; t < nt; ++t) { st.reset(); st.bind(1, fts5::quote_phrase(terms[t])); st.step(); r.df[t] = st.get<std::int64_t>(0); }
        return r;
    }, ro);
    std::int64_t N = 0, T = 0; std::vector<std::int64_t> df(nt, 0);
    for (auto& c : cs) { N += c.rows; T += c.tokens; for (std::size_t t = 0; t < nt; ++t) df[t] += c.df[t]; }
    if (rep) { rep->shards = ns; rep->rows = N; rep->tokens = T; rep->df = df; }
    if (N == 0) return {};
    const double avgdl = double(T) / double(N);
    std::vector<double> idf(nt);
    for (std::size_t t = 0; t < nt; ++t) { const double x = std::log((double(N - df[t]) + 0.5) / (double(df[t]) + 0.5)); idf[t] = x <= 0.0 ? 1e-6 : x; }
    std::string idf_blob(nt * sizeof(double), '\0'); std::memcpy(idf_blob.data(), idf.data(), idf_blob.size());
    // phase 2: shard bounds, then scoring best bound first
    struct cand { std::size_t shard; double bound; };
    std::vector<cand> order;
    for (std::size_t i = 0; i < ns; ++i) {
        double b = 0; std::size_t present = 0;
        for (std::size_t t = 0; t < nt; ++t) if (cs[i].df[t] > 0) { b += idf[t] * (1.2 + 1.0); ++present; }
        if (present == 0 || (bo.op == "AND" && present < nt)) continue;
        order.push_back({i, b});
    }
    if (rep) rep->with_hits = order.size();
    std::stable_sort(order.begin(), order.end(), [](const cand& a, const cand& b) { return a.bound > b.bound; });
    std::mutex mu; std::vector<shard_hit> heap; std::atomic<std::size_t> scored_n{0}, skipped{0};
    std::string wsql; for (double w : bo.weights) wsql += ", " + std::to_string(w);
    const std::string ft = quote_ident(fts_table);
    const std::string sql = "SELECT rowid, -forge_bm25(" + ft + ", ?1, ?2" + wsql + ") FROM " + ft + " WHERE " + ft + " MATCH ?3";
    std::vector<std::size_t> shard_ids; for (auto& c : order) shard_ids.push_back(c.shard);
    shard_map<int>(s, tp, shard_ids, [&](connection& c, std::size_t sh) {
        double bound = 0; for (auto& cd : order) if (cd.shard == sh) { bound = cd.bound; break; }
        if (bo.prune) { std::lock_guard g(mu); if (heap.size() >= k && bound < heap.front().score) { ++skipped; return 0; } }
        std::vector<shard_hit> local; local.reserve(std::min<std::size_t>(k, 1024));
        auto st = c.prepare(sql); st.bind(1, avgdl).bind(2, std::span<const std::byte>(reinterpret_cast<const std::byte*>(idf_blob.data()), idf_blob.size())).bind(3, expr);
        for (auto& r : st.rows()) {
            shard_hit h{std::uint32_t(sh), r.get<std::int64_t>(0), r.get<double>(1)};
            if (local.size() < k) { local.push_back(h); std::push_heap(local.begin(), local.end(), shard_hit_before); }
            else if (shard_hit_before(h, local.front())) { std::pop_heap(local.begin(), local.end(), shard_hit_before); local.back() = h; std::push_heap(local.begin(), local.end(), shard_hit_before); }
        }
        ++scored_n;
        std::lock_guard g(mu);
        for (auto& h : local) {
            if (heap.size() < k) { heap.push_back(h); std::push_heap(heap.begin(), heap.end(), shard_hit_before); }
            else if (shard_hit_before(h, heap.front())) { std::pop_heap(heap.begin(), heap.end(), shard_hit_before); heap.back() = h; std::push_heap(heap.begin(), heap.end(), shard_hit_before); }
        }
        return 0;
    }, run_options{false, ro.schedule});
    if (rep) { rep->scored = scored_n; rep->skipped_bound = skipped; }
    std::sort(heap.begin(), heap.end(), shard_hit_before);
    return heap;
}

// ---- cached results (forge::cache) -----------------------------------------------------------------------------
// Result bytes of an expensive query, kept in a content-addressed forge::cache::store (LRU by its max_bytes) under a key built
// from the database id (BLAKE3 of the manifest) and the query parts, so a rebuilt database never serves stale results.
inline std::string cached_result(cache::store& c, const shard_set& s, std::initializer_list<std::string_view> parts, const std::function<std::string()>& compute) {
    std::vector<std::string_view> p{s.id()}; p.insert(p.end(), parts.begin(), parts.end());
    hash::blake3 h; for (auto sv : p) { std::uint64_t n = sv.size(); h.update(&n, 8); h.update(sv.data(), sv.size()); }
    const cache::key k = h.final();
    if (auto v = c.get(k)) return *v;
    auto r = compute(); c.put(k, r); return r;
}
inline std::string encode_hits(const std::vector<shard_hit>& v) {
    std::string s(v.size() * sizeof(shard_hit), '\0');
    for (std::size_t i = 0; i < v.size(); ++i) { std::memcpy(s.data() + i * 24, &v[i].shard, 4); std::memcpy(s.data() + i * 24 + 8, &v[i].id, 8); std::memcpy(s.data() + i * 24 + 16, &v[i].score, 8); }
    return s;
}
inline std::vector<shard_hit> decode_hits(std::string_view s) {
    std::vector<shard_hit> v(s.size() / 24);
    for (std::size_t i = 0; i < v.size(); ++i) { std::memcpy(&v[i].shard, s.data() + i * 24, 4); std::memcpy(&v[i].id, s.data() + i * 24 + 8, 8); std::memcpy(&v[i].score, s.data() + i * 24 + 16, 8); }
    return v;
}

} // namespace forge::db

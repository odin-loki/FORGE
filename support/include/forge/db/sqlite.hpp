#pragma once
// forge::db: RAII C++ layer over the vendored SQLite (third_party/sqlite, pinned by third_party/sqlite/PIN).
//
//   db::connection c = db::connection::open("lean.fdb", {.wal = true});
//   c.exec("CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT, v BLOB)");
//   { db::transaction tx(c);                                   // rolls back unless commit() is called
//     auto ins = c.prepare("INSERT INTO t(name, v) VALUES(?, ?)");
//     ins.run("alpha", std::span<const std::byte>(bytes));     // bind, step, reset
//     tx.commit(); }
//   auto sel = c.prepare("SELECT id, name FROM t WHERE id > ?1"); sel.bind(1, 0);
//   for (auto& r : sel.rows()) use(r.get<std::int64_t>(0), r.get<std::string_view>(1));
//   auto [n] = c.query_row<std::int64_t>("SELECT count(*) FROM t");
//
// * Thread model: SQLite is built SQLITE_THREADSAFE=2 (multi-thread). One connection (and its statements) is used by
//   one thread at a time; use one connection per thread (db/parallel.hpp keeps a pool of read-only ones).
// * Errors: every failing call throws db::error, a forge::forge_exception whose errc is mapped from the SQLite result
//   code (busy/IO/cantopen -> io_error, corrupt -> integrity_error, not a database -> format_error, nomem ->
//   out_of_memory, constraint/misuse/range -> invalid_argument, syntax and other -> parse_error / invalid_argument),
//   carrying the SQLite primary and extended codes and the SQL text. try_open returns forge::result<connection>.
// * Binding: integers, bool, floating point, string-likes (copied), std::span<const std::byte> / std::vector<std::byte>
//   (blob, copied), nullptr, std::nullopt, std::optional<T>. static_text / static_blob bind without a copy and must
//   outlive the step. Columns: integers, bool, double, std::string, std::string_view and std::span<const std::byte>
//   (valid until the next step / reset), std::vector<std::byte>, std::optional<T> (NULL -> nullopt).
// * Database file contents are bit-stable for fixed schema and inserts; nothing here depends on the clock.
#include <forge/core/error.hpp>
#include <forge/core/log.hpp>
#include <sqlite3.h>
#include <atomic>
#include <cctype>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace forge::db {

// ---- errors ----------------------------------------------------------------------------------------------------
inline errc to_errc(int rc) {
    switch (rc & 0xff) {
        case SQLITE_NOMEM: return errc::out_of_memory;
        case SQLITE_CORRUPT: case SQLITE_CORRUPT_VTAB: return errc::integrity_error;
        case SQLITE_NOTADB: return errc::format_error;
        case SQLITE_BUSY: case SQLITE_LOCKED: case SQLITE_IOERR: case SQLITE_CANTOPEN: case SQLITE_FULL: case SQLITE_READONLY:
        case SQLITE_PERM: case SQLITE_PROTOCOL: case SQLITE_INTERRUPT: case SQLITE_AUTH: return errc::io_error;
        case SQLITE_NOTFOUND: return errc::not_found;
        case SQLITE_CONSTRAINT: case SQLITE_MISUSE: case SQLITE_RANGE: case SQLITE_MISMATCH: case SQLITE_TOOBIG: return errc::invalid_argument;
        case SQLITE_ERROR: return errc::parse_error;
        default: return errc::invalid_argument;
    }
}

struct error : forge_exception {
    int sqlite_code;       // primary result code (SQLITE_BUSY, ...)
    int extended_code;     // extended result code (SQLITE_CONSTRAINT_UNIQUE, ...)
    std::string sql;       // the statement, when known
    error(int primary, int extended, const std::string& msg, std::string statement = {})
        : forge_exception(to_errc(primary), msg), sqlite_code(primary), extended_code(extended), sql(std::move(statement)) {}
    bool busy() const { return sqlite_code == SQLITE_BUSY || sqlite_code == SQLITE_LOCKED; }
    bool constraint() const { return sqlite_code == SQLITE_CONSTRAINT; }
};

namespace detail {
[[noreturn]] inline void throw_db(sqlite3* db, int rc, std::string_view sql = {}) {
    std::string m = std::string(db ? sqlite3_errmsg(db) : sqlite3_errstr(rc));
    const int ext = db ? sqlite3_extended_errcode(db) : rc;
    m = "sqlite: " + m + " (code " + std::to_string(rc) + (ext != rc ? "/" + std::to_string(ext) : std::string()) + ")";
    if (!sql.empty()) m += ": " + std::string(sql.substr(0, 200));
    throw error(rc & 0xff, ext, m, std::string(sql));
}
inline void check(sqlite3* db, int rc, std::string_view sql = {}) {
    if (rc != SQLITE_OK && rc != SQLITE_ROW && rc != SQLITE_DONE) throw_db(db, rc, sql);
}
} // namespace detail

// ---- bind / column adapters ------------------------------------------------------------------------------------
struct static_text { std::string_view s; };                    // bound without a copy; must outlive the step
struct static_blob { std::span<const std::byte> b; };
struct zeroblob { std::uint64_t size; };                       // a blob of `size` zero bytes (filled by incremental I/O or update)

template <class T> struct is_optional : std::false_type {};
template <class T> struct is_optional<std::optional<T>> : std::true_type {};

template <class T> concept integer_like = std::integral<std::remove_cvref_t<T>> && !std::same_as<std::remove_cvref_t<T>, bool>
                                          && !std::same_as<std::remove_cvref_t<T>, char>;
template <class T> concept text_like = std::convertible_to<const T&, std::string_view> && !std::is_arithmetic_v<std::remove_cvref_t<T>>;
template <class T> concept blob_like = std::convertible_to<const T&, std::span<const std::byte>> && !text_like<T>;
template <class T> concept bindable = integer_like<T> || std::same_as<std::remove_cvref_t<T>, bool> || std::floating_point<std::remove_cvref_t<T>>
    || text_like<T> || blob_like<T> || std::same_as<std::remove_cvref_t<T>, std::nullptr_t> || std::same_as<std::remove_cvref_t<T>, std::nullopt_t>
    || std::same_as<std::remove_cvref_t<T>, static_text> || std::same_as<std::remove_cvref_t<T>, static_blob> || std::same_as<std::remove_cvref_t<T>, zeroblob>
    || is_optional<std::remove_cvref_t<T>>::value;

class connection;
class statement;

// One result row: a view of the statement's current row. Valid until the statement is stepped or reset.
class row {
public:
    explicit row(sqlite3_stmt* s) : s_(s) {}
    int size() const { return sqlite3_column_count(s_); }
    int type(int i) const { return sqlite3_column_type(s_, i); }   // SQLITE_INTEGER, SQLITE_FLOAT, SQLITE_TEXT, SQLITE_BLOB, SQLITE_NULL
    bool is_null(int i) const { return type(i) == SQLITE_NULL; }
    std::string_view name(int i) const { const char* n = sqlite3_column_name(s_, i); return n ? n : ""; }
    int index(std::string_view col) const {
        for (int i = 0; i < size(); ++i) if (name(i) == col) return i;
        raise(errc::not_found, "db: no result column '{}'", col);
    }
    template <class T> T get(int i) const {
        if (i < 0 || i >= size()) raise(errc::invalid_argument, "db: column index {} out of range (0..{})", i, size());
        using U = std::remove_cvref_t<T>;
        if constexpr (is_optional<U>::value) {
            if (is_null(i)) return std::nullopt;
            return U(get<typename U::value_type>(i));
        } else if constexpr (std::same_as<U, bool>) {
            return sqlite3_column_int64(s_, i) != 0;
        } else if constexpr (std::integral<U>) {
            return static_cast<U>(sqlite3_column_int64(s_, i));
        } else if constexpr (std::floating_point<U>) {
            return static_cast<U>(sqlite3_column_double(s_, i));
        } else if constexpr (std::same_as<U, std::string_view>) {
            const unsigned char* p = sqlite3_column_text(s_, i);
            return std::string_view(reinterpret_cast<const char*>(p ? p : reinterpret_cast<const unsigned char*>("")), std::size_t(sqlite3_column_bytes(s_, i)));
        } else if constexpr (std::same_as<U, std::string>) {
            return std::string(get<std::string_view>(i));
        } else if constexpr (std::same_as<U, std::span<const std::byte>>) {
            const void* p = sqlite3_column_blob(s_, i);
            return std::span<const std::byte>(static_cast<const std::byte*>(p), std::size_t(sqlite3_column_bytes(s_, i)));
        } else if constexpr (std::same_as<U, std::vector<std::byte>>) {
            auto b = get<std::span<const std::byte>>(i);
            return std::vector<std::byte>(b.begin(), b.end());
        } else {
            static_assert(!sizeof(U*), "db::row::get<T>: unsupported column type");
        }
    }
    template <class T> T get(std::string_view col) const { return get<T>(index(col)); }
    template <class... Ts> std::tuple<Ts...> tuple() const {
        int i = 0; return std::tuple<Ts...>{get<Ts>(i++)...};
    }
private:
    sqlite3_stmt* s_;
};

class statement {
public:
    statement() = default;
    statement(const statement&) = delete;
    statement& operator=(const statement&) = delete;
    statement(statement&& o) noexcept : s_(std::exchange(o.s_, nullptr)), db_(o.db_), sql_(std::move(o.sql_)) {}
    statement& operator=(statement&& o) noexcept {
        if (this != &o) { finalize(); s_ = std::exchange(o.s_, nullptr); db_ = o.db_; sql_ = std::move(o.sql_); }
        return *this;
    }
    ~statement() { finalize(); }
    explicit operator bool() const { return s_ != nullptr; }
    sqlite3_stmt* handle() const { return s_; }
    std::string_view sql() const { return sql_; }

    // 1-based, as in SQLite.
    template <class T> statement& bind(int i, const T& v) {
        static_assert(bindable<T>, "db::statement::bind: unsupported type");
        using U = std::remove_cvref_t<T>;
        int rc;
        if constexpr (is_optional<U>::value) {
            if (!v) rc = sqlite3_bind_null(s_, i); else return bind(i, *v);
        } else if constexpr (std::same_as<U, std::nullptr_t> || std::same_as<U, std::nullopt_t>) {
            rc = sqlite3_bind_null(s_, i);
        } else if constexpr (std::same_as<U, bool>) {
            rc = sqlite3_bind_int64(s_, i, v ? 1 : 0);
        } else if constexpr (std::integral<U>) {
            rc = sqlite3_bind_int64(s_, i, static_cast<sqlite3_int64>(v));
        } else if constexpr (std::floating_point<U>) {
            rc = sqlite3_bind_double(s_, i, static_cast<double>(v));
        } else if constexpr (std::same_as<U, static_text>) {
            rc = sqlite3_bind_text64(s_, i, v.s.data(), v.s.size(), SQLITE_STATIC, SQLITE_UTF8);
        } else if constexpr (std::same_as<U, static_blob>) {
            rc = sqlite3_bind_blob64(s_, i, v.b.data(), v.b.size(), SQLITE_STATIC);
        } else if constexpr (std::same_as<U, zeroblob>) {
            rc = sqlite3_bind_zeroblob64(s_, i, v.size);
        } else if constexpr (text_like<U>) {
            std::string_view sv = v;
            rc = sqlite3_bind_text64(s_, i, sv.data(), sv.size(), SQLITE_TRANSIENT, SQLITE_UTF8);
        } else {
            std::span<const std::byte> b = v;
            rc = b.empty() ? sqlite3_bind_zeroblob(s_, i, 0) : sqlite3_bind_blob64(s_, i, b.data(), b.size(), SQLITE_TRANSIENT);
        }
        detail::check(db_, rc, sql_);
        return *this;
    }
    // By parameter name including its prefix (":lo", "@x", "$y").
    template <class T> statement& bind(std::string_view name, const T& v) {
        const int i = sqlite3_bind_parameter_index(s_, std::string(name).c_str());
        if (i == 0) raise(errc::not_found, "db: no parameter '{}' in: {}", name, sql_.substr(0, 200));
        return bind(i, v);
    }
    template <class... A> statement& bind_all(const A&... a) {
        int i = 0; (bind(++i, a), ...); return *this;
    }
    int parameter_count() const { return sqlite3_bind_parameter_count(s_); }

    // true: a row is available (read it through current() or get<T>); false: finished. Throws db::error.
    bool step() {
        const int rc = sqlite3_step(s_);
        if (rc == SQLITE_ROW) return true;
        if (rc == SQLITE_DONE) return false;
        detail::throw_db(db_, rc, sql_);
    }
    row current() const { return row(s_); }
    template <class T> T get(int i) const { return row(s_).get<T>(i); }
    statement& reset() { sqlite3_reset(s_); return *this; }          // keeps the bindings
    statement& clear_bindings() { sqlite3_clear_bindings(s_); return *this; }
    // Reset, clear, bind a... then step until done; returns the rows changed by this execution.
    template <class... A> std::int64_t run(const A&... a) {
        reset(); clear_bindings(); bind_all(a...);
        while (step()) {}
        const auto ch = sqlite3_changes64(db_);
        reset();
        return ch;
    }

    // Input range over the rows. Starts at the current state: call reset() before re-iterating a used statement.
    class row_iterator {
    public:
        using iterator_category = std::input_iterator_tag; using value_type = row; using difference_type = std::ptrdiff_t;
        using pointer = const row*; using reference = const row&;
        row_iterator() = default;
        explicit row_iterator(statement* s) : st_(s) { advance(); }
        reference operator*() const { return r_; }
        pointer operator->() const { return &r_; }
        row_iterator& operator++() { advance(); return *this; }
        void operator++(int) { advance(); }
        friend bool operator==(const row_iterator& a, std::default_sentinel_t) { return a.st_ == nullptr; }
    private:
        void advance() { if (st_ && !st_->step()) st_ = nullptr; else if (st_) r_ = st_->current(); }
        statement* st_ = nullptr; row r_{nullptr};
    };
    struct row_range {
        statement* s;
        row_iterator begin() const { return row_iterator(s); }
        std::default_sentinel_t end() const { return {}; }
    };
    row_range rows() { return {this}; }

private:
    friend class connection;
    statement(sqlite3_stmt* s, sqlite3* d, std::string sql) : s_(s), db_(d), sql_(std::move(sql)) {}
    void finalize() { if (s_) { sqlite3_finalize(s_); s_ = nullptr; } }
    sqlite3_stmt* s_ = nullptr;
    sqlite3* db_ = nullptr;
    std::string sql_;
};

// ---- connection ------------------------------------------------------------------------------------------------
struct open_options {
    bool read_only = false;
    bool create = true;                 // create the file when missing (ignored for read_only)
    bool immutable = false;             // read-only and the file never changes: no locking, no WAL files (URI immutable=1)
    bool uri = false;                   // `path` is already a SQLite URI (file:...?mode=ro)
    bool in_memory_shared = false;      // path names a shared-cache in-memory database (file:<path>?mode=memory&cache=shared)
    int busy_timeout_ms = 10000;
    bool wal = false;                   // PRAGMA journal_mode=WAL (writable connections only)
    std::string synchronous;            // "", "OFF", "NORMAL", "FULL"
    bool foreign_keys = true;
    int cache_size_kib = 0;             // 0: SQLite default; else PRAGMA cache_size=-N
    std::int64_t mmap_size = 0;         // 0: off
    bool temp_store_memory = false;
    int sorter_threads = 0;             // PRAGMA threads=N: helper threads SQLite may use to sort (CREATE INDEX, ORDER BY)
    std::vector<std::string> pragmas;   // extra "name=value" pragmas, run after the above
};

// Percent-encode a filesystem path into a file: URI path (so '?' '#' '%' in names survive).
inline std::string file_uri(std::string_view path, std::string_view query) {
    std::string out = "file:";
    std::string p(path);
    for (char& c : p) if (c == '\\') c = '/';
    if (p.size() >= 2 && p[1] == ':') out += '/';   // C:/x -> /C:/x
    static const char* hex = "0123456789ABCDEF";
    for (unsigned char c : p) {
        if (std::isalnum(c) || c == '/' || c == '-' || c == '_' || c == '.' || c == '~' || c == ':' || c >= 0x80) out += char(c);
        else { out += '%'; out += hex[c >> 4]; out += hex[c & 15]; }
    }
    if (!query.empty()) { out += '?'; out += query; }
    return out;
}

// "name" quoted as an SQL identifier ("" doubled). Use for table and column names taken from data.
inline std::string quote_ident(std::string_view s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"') o += '"'; o += c; }
    o += '"'; return o;
}
// 'text' quoted as an SQL string literal. Prefer bound parameters; this is for DDL such as ATTACH or PRAGMA.
inline std::string quote_literal(std::string_view s) {
    std::string o = "'";
    for (char c : s) { if (c == '\'') o += '\''; o += c; }
    o += '\''; return o;
}

class connection {
public:
    connection() = default;
    connection(const connection&) = delete;
    connection& operator=(const connection&) = delete;
    connection(connection&& o) noexcept : db_(std::exchange(o.db_, nullptr)), path_(std::move(o.path_)) {}
    connection& operator=(connection&& o) noexcept {
        if (this != &o) { close(); db_ = std::exchange(o.db_, nullptr); path_ = std::move(o.path_); }
        return *this;
    }
    ~connection() { close(); }

    // Throws db::error. The path ":memory:" (or "") opens a private in-memory database.
    static connection open(std::string_view path, const open_options& o = {}) {
        connection c;
        int flags = SQLITE_OPEN_NOMUTEX;
        std::string target(path);
        const bool ro = o.read_only || o.immutable;
        if (o.immutable && !o.uri) { target = file_uri(path, "mode=ro&immutable=1"); flags |= SQLITE_OPEN_URI; }
        else if (o.in_memory_shared) { target = file_uri(path, "mode=memory&cache=shared"); flags |= SQLITE_OPEN_URI | SQLITE_OPEN_SHAREDCACHE; }
        if (o.uri) flags |= SQLITE_OPEN_URI;
        flags |= ro ? SQLITE_OPEN_READONLY : (SQLITE_OPEN_READWRITE | (o.create ? SQLITE_OPEN_CREATE : 0));
        if (target.empty()) target = ":memory:";
        const int rc = sqlite3_open_v2(target.c_str(), &c.db_, flags, nullptr);
        if (rc != SQLITE_OK) { const int ext = c.db_ ? sqlite3_extended_errcode(c.db_) : rc; std::string m = "sqlite: cannot open '" + std::string(path) + "': " + (c.db_ ? sqlite3_errmsg(c.db_) : sqlite3_errstr(rc)); if (c.db_) { sqlite3_close(c.db_); c.db_ = nullptr; } throw error(rc & 0xff, ext, m); }
        c.path_ = std::string(path);
        sqlite3_busy_timeout(c.db_, o.busy_timeout_ms);
        sqlite3_extended_result_codes(c.db_, 1);
        if (o.foreign_keys) c.exec("PRAGMA foreign_keys=ON");
        if (o.wal && !ro) c.exec("PRAGMA journal_mode=WAL");
        if (!o.synchronous.empty()) c.exec("PRAGMA synchronous=" + o.synchronous);
        if (o.cache_size_kib > 0) c.exec("PRAGMA cache_size=-" + std::to_string(o.cache_size_kib));
        if (o.mmap_size > 0) c.exec("PRAGMA mmap_size=" + std::to_string(o.mmap_size));
        if (o.temp_store_memory) c.exec("PRAGMA temp_store=MEMORY");
        if (o.sorter_threads > 0) c.exec("PRAGMA threads=" + std::to_string(o.sorter_threads));
        for (const auto& p : o.pragmas) c.exec("PRAGMA " + p);
        FORGE_LOG_DEBUG("db", "opened {} ({}{})", path, ro ? "read-only" : "read-write", o.immutable ? ", immutable" : "");
        return c;
    }
    static result<connection> try_open(std::string_view path, const open_options& o = {}) {
        try { return open(path, o); } catch (const forge_exception& e) { return fail(e.code, e.what()); }
    }

    explicit operator bool() const { return db_ != nullptr; }
    sqlite3* handle() const { return db_; }
    const std::string& path() const { return path_; }
    void close() {
        if (db_) { sqlite3_close_v2(db_); db_ = nullptr; }
    }

    // One or more statements, no results.
    void exec(const std::string& sql) {
        char* msg = nullptr;
        const int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &msg);
        if (rc != SQLITE_OK) { if (msg) sqlite3_free(msg); detail::throw_db(db_, rc, sql); }
    }
    statement prepare(std::string_view sql) {
        sqlite3_stmt* s = nullptr;
        const int rc = sqlite3_prepare_v3(db_, sql.data(), int(sql.size()), 0, &s, nullptr);
        if (rc != SQLITE_OK) detail::throw_db(db_, rc, sql);
        if (!s) raise(errc::invalid_argument, "db: empty statement: '{}'", sql.substr(0, 80));
        return statement(s, db_, std::string(sql));
    }
    // prepare + bind + run to completion; returns rows changed.
    template <class... A> std::int64_t execute(std::string_view sql, const A&... a) {
        auto st = prepare(sql); return st.run(a...);
    }
    // The first row as a tuple; throws not_found when the query returns no row.
    template <class... Ts, class... A> std::tuple<Ts...> query_row(std::string_view sql, const A&... a) {
        auto st = prepare(sql); st.bind_all(a...);
        if (!st.step()) raise(errc::not_found, "db: query returned no row: {}", sql.substr(0, 120));
        return st.current().template tuple<Ts...>();
    }
    // The first column of the first row; `fallback` when there is no row.
    template <class T, class... A> T scalar(std::string_view sql, const A&... a) {
        auto st = prepare(sql); st.bind_all(a...);
        if (!st.step()) raise(errc::not_found, "db: query returned no row: {}", sql.substr(0, 120));
        return st.template get<T>(0);
    }
    template <class T, class... A> T scalar_or(T fallback, std::string_view sql, const A&... a) {
        auto st = prepare(sql); st.bind_all(a...);
        return st.step() ? st.template get<T>(0) : fallback;
    }
    bool table_exists(std::string_view name, std::string_view schema = "main") {
        auto st = prepare("SELECT 1 FROM " + quote_ident(schema) + ".sqlite_master WHERE type IN ('table','view') AND name=?1");
        st.bind(1, name); return st.step();
    }
    std::string pragma(std::string_view name) {   // value of "PRAGMA name" as text ("" when no result)
        auto st = prepare("PRAGMA " + std::string(name));
        return st.step() ? st.get<std::string>(0) : std::string();
    }

    std::int64_t last_insert_rowid() const { return sqlite3_last_insert_rowid(db_); }
    std::int64_t changes() const { return sqlite3_changes64(db_); }
    std::int64_t total_changes() const { return sqlite3_total_changes64(db_); }
    bool in_transaction() const { return sqlite3_get_autocommit(db_) == 0; }
    void interrupt() { sqlite3_interrupt(db_); }       // the only call safe from another thread
    void attach(std::string_view file, std::string_view schema, bool read_only = false) {
        if (read_only) exec("ATTACH DATABASE " + quote_literal(file_uri(file, "mode=ro")) + " AS " + quote_ident(schema));
        else exec("ATTACH DATABASE " + quote_literal(file) + " AS " + quote_ident(schema));
    }
    void detach(std::string_view schema) { exec("DETACH DATABASE " + quote_ident(schema)); }

    // PRAGMA integrity_check: empty when the database is sound, else the messages SQLite returned.
    std::vector<std::string> integrity_check(int max_errors = 20) {
        std::vector<std::string> out;
        auto st = prepare("PRAGMA integrity_check(" + std::to_string(max_errors) + ")");
        for (auto& r : st.rows()) { auto s = r.get<std::string>(0); if (s != "ok") out.push_back(std::move(s)); }
        return out;
    }
    std::vector<std::string> foreign_key_check() {
        std::vector<std::string> out;
        auto st = prepare("PRAGMA foreign_key_check");
        for (auto& r : st.rows()) out.push_back(r.get<std::string>(0) + " rowid " + std::to_string(r.get<std::int64_t>(1)) + " -> " + r.get<std::string>(2));
        return out;
    }

private:
    sqlite3* db_ = nullptr;
    std::string path_;
};

// ---- transaction -----------------------------------------------------------------------------------------------
enum class tx_mode { deferred, immediate, exclusive };

// BEGIN ... COMMIT. Rolls back in the destructor unless commit() ran. Inside an open transaction it is a SAVEPOINT.
class transaction {
public:
    explicit transaction(connection& c, tx_mode m = tx_mode::deferred) : c_(&c) {
        if (c.in_transaction()) {
            static std::atomic<unsigned> ctr{0};
            sp_ = "forge_sp_" + std::to_string(ctr.fetch_add(1));
            c.exec("SAVEPOINT " + sp_);
        } else {
            c.exec(m == tx_mode::immediate ? "BEGIN IMMEDIATE" : m == tx_mode::exclusive ? "BEGIN EXCLUSIVE" : "BEGIN");
        }
        open_ = true;
    }
    transaction(const transaction&) = delete;
    transaction& operator=(const transaction&) = delete;
    ~transaction() { if (open_) { try { rollback(); } catch (...) {} } }
    void commit() {
        if (!open_) raise(errc::invalid_argument, "db: transaction already finished");
        c_->exec(sp_.empty() ? "COMMIT" : "RELEASE " + sp_);
        open_ = false;
    }
    void rollback() {
        if (!open_) return;
        open_ = false;
        if (sp_.empty()) c_->exec("ROLLBACK"); else c_->exec("ROLLBACK TO " + sp_ + "; RELEASE " + sp_);
    }
    bool active() const { return open_; }
private:
    connection* c_; std::string sp_; bool open_ = false;
};

// ---- FTS5 helpers ----------------------------------------------------------------------------------------------
namespace fts5 {

struct table_options {
    std::string content;                // external-content table ("" = the FTS table stores its text)
    std::string content_rowid;          // its rowid column when `content` is set
    std::string tokenize = "unicode61 remove_diacritics 0";
    std::string prefix;                 // e.g. "2 3"
    bool detail_none = false;           // smaller index, no phrase or NEAR queries
};

// CREATE VIRTUAL TABLE name USING fts5(columns...). A column may carry an option, e.g. "id UNINDEXED".
inline void create(connection& c, std::string_view name, const std::vector<std::string>& columns, const table_options& o = {}) {
    std::string sql = "CREATE VIRTUAL TABLE " + quote_ident(name) + " USING fts5(";
    for (std::size_t i = 0; i < columns.size(); ++i) { if (i) sql += ", "; sql += columns[i]; }
    if (!o.content.empty()) sql += ", content=" + quote_literal(o.content);
    if (!o.content_rowid.empty()) sql += ", content_rowid=" + quote_literal(o.content_rowid);
    if (!o.tokenize.empty()) sql += ", tokenize=" + quote_literal(o.tokenize);
    if (!o.prefix.empty()) sql += ", prefix=" + quote_literal(o.prefix);
    if (o.detail_none) sql += ", detail=none";
    sql += ")";
    c.exec(sql);
}
inline void rebuild(connection& c, std::string_view name) { c.exec("INSERT INTO " + quote_ident(name) + "(" + quote_ident(name) + ") VALUES('rebuild')"); }
inline void optimize(connection& c, std::string_view name) { c.exec("INSERT INTO " + quote_ident(name) + "(" + quote_ident(name) + ") VALUES('optimize')"); }
// Throws db::error (SQLITE_CORRUPT_VTAB) when the index disagrees with its content table.
inline void integrity_check(connection& c, std::string_view name, bool check_content = true) {
    c.exec("INSERT INTO " + quote_ident(name) + "(" + quote_ident(name) + ", rank) VALUES('integrity-check', " + (check_content ? "1" : "0") + ")");
}

// The word characters of `text` (ASCII letters and digits, '_' and every byte >= 0x80, so UTF-8 words stay whole).
inline std::vector<std::string> words(std::string_view text) {
    std::vector<std::string> out; std::string cur;
    auto flush = [&] { if (!cur.empty()) { out.push_back(std::move(cur)); cur.clear(); } };
    for (unsigned char ch : text) {
        if (std::isalnum(ch) || ch == '_' || ch >= 0x80) cur += char(ch); else flush();
    }
    flush(); return out;
}
inline std::string quote_phrase(std::string_view s) {   // one FTS5 string: "..." with " doubled
    std::string o = "\"";
    for (char c : s) { if (c == '"') o += '"'; o += c; }
    o += '"'; return o;
}
// A MATCH expression from free text: every word quoted (so FTS5 operators in the text are harmless), joined by `op`
// ("OR" or "AND"). Words are deduplicated in order of first use; at most `max_terms` are kept. "" when there are none.
inline std::string query_from_text(std::string_view text, std::string_view op = "OR", std::size_t max_terms = 64) {
    std::string out; std::vector<std::string> seen;
    for (auto& w : words(text)) {
        bool dup = false; for (auto& s : seen) if (s == w) { dup = true; break; }
        if (dup) continue;
        if (seen.size() >= max_terms) break;
        seen.push_back(w);
        if (!out.empty()) { out += ' '; out += op; out += ' '; }
        out += quote_phrase(w);
    }
    return out;
}

struct match { std::int64_t rowid; double score; };   // score = -bm25(): larger is better
// The k best matches of `query` (an FTS5 MATCH expression) ordered by (score descending, rowid ascending). `weights` are
// the per-column bm25 weights (empty: all 1).
inline std::vector<match> search(connection& c, std::string_view table, std::string_view query, std::size_t k, const std::vector<double>& weights = {}) {
    std::vector<match> out;
    if (query.empty() || k == 0) return out;
    std::string t = quote_ident(table), w;
    for (double x : weights) w += ", " + std::to_string(x);
    auto st = c.prepare("SELECT rowid, -bm25(" + t + w + ") AS s FROM " + t + " WHERE " + t + " MATCH ?1 ORDER BY s DESC, rowid ASC LIMIT ?2");
    st.bind(1, query).bind(2, static_cast<std::int64_t>(k));
    for (auto& r : st.rows()) out.push_back({r.get<std::int64_t>(0), r.get<double>(1)});
    return out;
}
} // namespace fts5

} // namespace forge::db

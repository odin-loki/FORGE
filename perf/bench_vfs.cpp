// perf/bench_vfs: does a FORGE SQLite VFS over core/mapped.hpp (forge::map_readonly) beat SQLite's own mmap_size on a shard?
//
//   bench_vfs shard-000010.fdb [reps]
//
// The VFS below is a minimal read-only one: xOpen maps the file with forge::map_readonly, xRead copies from the mapping, xFetch hands
// SQLite a pointer into it (so pages are used in place, as with mmap_size), locking is a no-op (the file is immutable). Everything
// else is delegated to the default VFS. Variants, each on fresh connections: sqlite-mmap (default VFS, immutable=1, mmap_size = file
// size: what db::shard_set does), forge-vfs (this VFS, same pragmas), pread (default VFS, mmap off). Operations: opening plus the
// first lookup, point lookups, a full scan, an FTS5 query. Prints the best of `reps` repetitions per operation.
#include <forge/core/mapped.hpp>
#include <forge/db/shards.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
using namespace forge;
using clk = std::chrono::steady_clock;

namespace {
struct fvfs_file { sqlite3_file base; const std::uint8_t* p; std::size_t n; };
int f_close(sqlite3_file* f) { auto* x = reinterpret_cast<fvfs_file*>(f); unmap_readonly(x->p, x->n); x->p = nullptr; return SQLITE_OK; }
int f_read(sqlite3_file* f, void* buf, int amt, sqlite3_int64 off) {
    auto* x = reinterpret_cast<fvfs_file*>(f);
    if (std::size_t(off) >= x->n) { std::memset(buf, 0, std::size_t(amt)); return SQLITE_IOERR_SHORT_READ; }
    const std::size_t got = std::min<std::size_t>(std::size_t(amt), x->n - std::size_t(off));
    std::memcpy(buf, x->p + off, got);
    if (got < std::size_t(amt)) { std::memset(static_cast<char*>(buf) + got, 0, std::size_t(amt) - got); return SQLITE_IOERR_SHORT_READ; }
    return SQLITE_OK;
}
int f_write(sqlite3_file*, const void*, int, sqlite3_int64) { return SQLITE_READONLY; }
int f_trunc(sqlite3_file*, sqlite3_int64) { return SQLITE_READONLY; }
int f_sync(sqlite3_file*, int) { return SQLITE_OK; }
int f_size(sqlite3_file* f, sqlite3_int64* s) { *s = sqlite3_int64(reinterpret_cast<fvfs_file*>(f)->n); return SQLITE_OK; }
int f_lock(sqlite3_file*, int) { return SQLITE_OK; }
int f_unlock(sqlite3_file*, int) { return SQLITE_OK; }
int f_chk(sqlite3_file*, int* r) { *r = 0; return SQLITE_OK; }
int f_ctl(sqlite3_file*, int, void*) { return SQLITE_NOTFOUND; }
int f_sector(sqlite3_file*) { return 4096; }
int f_dev(sqlite3_file*) { return SQLITE_IOCAP_IMMUTABLE; }
int f_fetch(sqlite3_file* f, sqlite3_int64 off, int amt, void** pp) {
    auto* x = reinterpret_cast<fvfs_file*>(f);
    if (std::size_t(off) + std::size_t(amt) > x->n) { *pp = nullptr; return SQLITE_OK; }
    *pp = const_cast<std::uint8_t*>(x->p) + off; return SQLITE_OK;
}
int f_unfetch(sqlite3_file*, sqlite3_int64, void*) { return SQLITE_OK; }
const sqlite3_io_methods kMethods = {3, f_close, f_read, f_write, f_trunc, f_sync, f_size, f_lock, f_unlock, f_chk, f_ctl, f_sector, f_dev, nullptr, nullptr, nullptr, nullptr, f_fetch, f_unfetch};
sqlite3_vfs* g_base = nullptr;
int v_open(sqlite3_vfs*, sqlite3_filename name, sqlite3_file* f, int flags, int* out) {
    auto* x = reinterpret_cast<fvfs_file*>(f); std::memset(x, 0, sizeof *x);
    if (!(flags & SQLITE_OPEN_MAIN_DB) || (flags & (SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE))) return SQLITE_CANTOPEN;
    try { x->p = map_readonly(sqlite3_filename_database(name), x->n); } catch (...) { return SQLITE_CANTOPEN; }
    x->base.pMethods = &kMethods; if (out) *out = flags; return SQLITE_OK;
}
int v_delete(sqlite3_vfs*, const char*, int) { return SQLITE_READONLY; }
int v_access(sqlite3_vfs*, const char* n, int fl, int* r) { return g_base->xAccess(g_base, n, fl, r); }
int v_full(sqlite3_vfs*, const char* n, int out, char* z) { return g_base->xFullPathname(g_base, n, out, z); }
int v_rand(sqlite3_vfs*, int n, char* z) { return g_base->xRandomness(g_base, n, z); }
int v_sleep(sqlite3_vfs*, int us) { return g_base->xSleep(g_base, us); }
int v_time(sqlite3_vfs*, double* t) { return g_base->xCurrentTime(g_base, t); }
int v_err(sqlite3_vfs*, int n, char* z) { return g_base->xGetLastError(g_base, n, z); }
sqlite3_vfs g_vfs;
void register_vfs() {
    g_base = sqlite3_vfs_find(nullptr); g_vfs = *g_base;
    g_vfs.zName = "forge_mmap"; g_vfs.szOsFile = int(sizeof(fvfs_file)); g_vfs.pNext = nullptr;
    g_vfs.xOpen = v_open; g_vfs.xDelete = v_delete; g_vfs.xAccess = v_access; g_vfs.xFullPathname = v_full; g_vfs.xRandomness = v_rand;
    g_vfs.xSleep = v_sleep; g_vfs.xCurrentTime = v_time; g_vfs.xGetLastError = v_err;
    g_vfs.xDlOpen = nullptr; g_vfs.xDlError = nullptr; g_vfs.xDlSym = nullptr; g_vfs.xDlClose = nullptr; g_vfs.xCurrentTimeInt64 = nullptr;
    g_vfs.xSetSystemCall = nullptr; g_vfs.xGetSystemCall = nullptr; g_vfs.xNextSystemCall = nullptr;
    sqlite3_vfs_register(&g_vfs, 0);
}
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
}   // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: bench_vfs shard.fdb [reps]\n"); return 2; }
    const std::string file = argv[1]; const int reps = argc > 2 ? std::atoi(argv[2]) : 15;
    register_vfs();
    const auto size = std::int64_t(std::filesystem::file_size(file));
    struct variant { const char* name; std::string path; std::int64_t mmap; };
    const std::string uri_default = db::file_uri(file, "mode=ro&immutable=1"), uri_forge = db::file_uri(file, "mode=ro&vfs=forge_mmap");
    std::vector<variant> vs = {{"sqlite-mmap", uri_default, size}, {"forge-vfs", uri_forge, size}, {"pread", uri_default, 0}};
    std::mt19937_64 g(3);
    std::int64_t lo = 0, hi = 0;
    { auto c = db::connection::open(uri_default, {.read_only = true, .uri = true}); lo = c.scalar<std::int64_t>("SELECT min(id) FROM t"); hi = c.scalar<std::int64_t>("SELECT max(id) FROM t"); }
    std::vector<std::string> keys; for (int i = 0; i < 4000; ++i) keys.push_back(db::int_key(lo + std::int64_t(g() % std::uint64_t(hi - lo + 1))));
    std::printf("shard %s (%.1f MiB), ids %lld..%lld; best of %d\n\n| variant | open + first lookup (us) | 4000 point lookups (us each) | full scan sum(length(body)) (ms) | FTS5 query w1 OR w2 (ms) |\n|---|---|---|---|---|\n",
                file.c_str(), double(size) / (1 << 20), (long long)lo, (long long)hi, reps);
    for (auto& v : vs) {
        double t_open = 1e9, t_pt = 1e9, t_scan = 1e9, t_fts = 1e9; std::int64_t chk = 0, chk2 = 0, chk3 = 0;
        for (int r = 0; r < reps; ++r) {
            auto t0 = clk::now();
            db::open_options o; o.uri = true; o.read_only = true; o.mmap_size = v.mmap; o.foreign_keys = false; o.cache_size_kib = 1024;
            auto c = db::connection::open(v.path, o);
            chk = c.scalar<std::int64_t>("SELECT id FROM t WHERE k = ?1", std::span<const std::byte>(reinterpret_cast<const std::byte*>(keys[0].data()), keys[0].size()));
            t_open = std::min(t_open, since(t0));
            t0 = clk::now(); auto st = c.prepare("SELECT id FROM t WHERE k = ?1"); std::int64_t s = 0;
            for (auto& k : keys) { st.reset(); st.bind(1, std::span<const std::byte>(reinterpret_cast<const std::byte*>(k.data()), k.size())); if (st.step()) s += st.get<std::int64_t>(0); }
            t_pt = std::min(t_pt, since(t0) / double(keys.size())); chk2 = s;
            t0 = clk::now(); chk3 = c.scalar<std::int64_t>("SELECT coalesce(sum(length(body)), 0) FROM t"); t_scan = std::min(t_scan, since(t0));
            t0 = clk::now(); auto q = c.prepare("SELECT rowid FROM fts WHERE fts MATCH 'w1 OR w2' ORDER BY bm25(fts) LIMIT 10"); std::int64_t f = 0; for (auto& rr : q.rows()) f += rr.get<std::int64_t>(0);
            t_fts = std::min(t_fts, since(t0)); chk += f;
        }
        std::printf("| %s | %.0f | %.2f | %.1f | %.2f |   (checks %lld %lld %lld)\n", v.name, t_open * 1e6, t_pt * 1e6, t_scan * 1e3, t_fts * 1e3, (long long)chk, (long long)chk2, (long long)chk3);
    }
    return 0;
}

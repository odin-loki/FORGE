#pragma once
// forge::db shard I/O: prefetching shard files into the page cache and placing shards on NUMA nodes.
//
// Prefetch (prefetch_kind):
//   fadvise   posix_fadvise(WILLNEED) on each file: one hint per file, the kernel reads ahead on its own (POSIX; nothing elsewhere).
//   aio       a background thread reads the files in 1 MiB chunks through forge::io::async_reader (io_uring on Linux, the pread
//             thread pool elsewhere) at queue depth 32 into scratch buffers and discards the bytes, so the page cache holds
//             the shard before a worker maps it. Files already resident (Linux: mincore) are skipped, so a warm query costs
//             one mincore per shard, not a read. Results never depend on it: it only changes what is cached.
//   automatic = fadvise: the aio thread re-read every shard on every query on a 9p mount (no page cache to ask), so it is opt-in and
//             reads each shard of a set once (shard_set::mark_prefetched).
//
// Placement (shard_schedule::numa): the shards of a query are split into contiguous runs, one per NUMA node (os::numa_nodes());
// a worker takes the next shard of the node its CPU belongs to and steals from the next node when its own run is empty.
// Shard i's result is written to slot i, so the result is the same as for any other schedule.
#include <forge/io/aio.hpp>
#include <forge/os/numa.hpp>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace forge::db {

enum class prefetch_kind { none, fadvise, aio, automatic };
inline const char* prefetch_kind_name(prefetch_kind k) { return k == prefetch_kind::none ? "none" : k == prefetch_kind::fadvise ? "fadvise" : k == prefetch_kind::aio ? "aio" : "auto"; }

namespace detail {

inline void fadvise_file(const std::string& path, int advice_willneed) {   // 1: WILLNEED, 0: DONTNEED
#if defined(__unix__) || defined(__APPLE__)
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
#if defined(POSIX_FADV_WILLNEED) && defined(POSIX_FADV_DONTNEED)
        (void)::posix_fadvise(fd, 0, 0, advice_willneed ? POSIX_FADV_WILLNEED : POSIX_FADV_DONTNEED);
#else
        (void)advice_willneed;
#endif
        ::close(fd);
    }
#else
    (void)path; (void)advice_willneed;
#endif
}

// Share of the file's pages that are in the page cache (Linux: mincore); -1 where that cannot be asked.
inline double resident_fraction(const std::string& path) {
#if defined(__linux__)
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    struct stat st{};
    double r = -1;
    if (::fstat(fd, &st) == 0 && st.st_size > 0) {
        void* m = ::mmap(nullptr, std::size_t(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
        if (m != MAP_FAILED) {
            const std::size_t pg = std::size_t(::sysconf(_SC_PAGESIZE)), n = (std::size_t(st.st_size) + pg - 1) / pg;
            std::vector<unsigned char> v(n);
            if (::mincore(m, std::size_t(st.st_size), v.data()) == 0) { std::size_t c = 0; for (auto b : v) c += b & 1; r = double(c) / double(n); }
            ::munmap(m, std::size_t(st.st_size));
        }
    }
    ::close(fd);
    return r;
#else
    (void)path; return -1;
#endif
}

inline int open_fd_ro(const std::string& path) {
#if defined(_WIN32)
    return ::_wopen(os::widen(path).c_str(), _O_RDONLY | _O_BINARY);
#else
    return ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
#endif
}
inline void close_fd(int fd) {
#if defined(_WIN32)
    ::_close(fd);
#else
    ::close(fd);
#endif
}

// Reads a list of files, in order, through io::async_reader on its own thread; the bytes are discarded (page-cache warming).
class aio_prefetcher {
public:
    struct stats_t { std::size_t files = 0, skipped_resident = 0, failed = 0; std::uint64_t bytes = 0; const char* backend = ""; };
    aio_prefetcher(std::vector<std::string> paths, unsigned depth = 32, std::size_t chunk = 1u << 20, double skip_if_resident = 0.95)
        : paths_(std::move(paths)), depth_(depth), chunk_(chunk), skip_(skip_if_resident) { th_ = std::thread([this] { run_(); }); }
    ~aio_prefetcher() { stop(); }
    aio_prefetcher(const aio_prefetcher&) = delete; aio_prefetcher& operator=(const aio_prefetcher&) = delete;
    // Stops after the chunks already submitted and joins.
    void stop() { stop_.store(true, std::memory_order_relaxed); if (th_.joinable()) th_.join(); }
    // Waits until every file has been read (or skipped).
    void wait() { if (th_.joinable()) th_.join(); }
    stats_t stats() const { return st_; }

private:
    struct chunk_t { int file; std::uint64_t off; std::size_t len; };
    void run_() {
        try {
            io::async_reader rd(depth_);
            st_.backend = rd.kind() == io::async_reader::backend::io_uring ? "io_uring" : "threads";
            std::vector<std::vector<std::uint8_t>> bufs(depth_); for (auto& b : bufs) b.resize(chunk_);
            std::vector<unsigned> freeb; for (unsigned i = 0; i < depth_; ++i) freeb.push_back(i);
            std::vector<int> fds(paths_.size(), -1); std::vector<std::size_t> left(paths_.size(), 0);
            std::vector<chunk_t> pending_q; std::size_t next_file = 0, qpos = 0; std::size_t inflight = 0;
            auto fill = [&] {   // expand files into chunks until there are enough to keep the queue busy
                while (qpos >= pending_q.size() && next_file < paths_.size() && !stop_.load(std::memory_order_relaxed)) {
                    pending_q.clear(); qpos = 0;
                    const std::size_t f = next_file++;
                    if (skip_ > 0 && resident_fraction(paths_[f]) >= skip_) { ++st_.skipped_resident; continue; }
                    int fd = open_fd_ro(paths_[f]);
                    if (fd < 0) { ++st_.failed; continue; }
                    std::uint64_t sz = 0;
#if defined(_WIN32)
                    sz = std::uint64_t(::_lseeki64(fd, 0, SEEK_END));
#else
                    { struct stat s{}; if (::fstat(fd, &s) == 0) sz = std::uint64_t(s.st_size); }
#endif
                    if (!sz) { close_fd(fd); continue; }
                    fds[f] = fd; ++st_.files;
                    for (std::uint64_t o = 0; o < sz; o += chunk_) pending_q.push_back({int(f), o, std::size_t(std::min<std::uint64_t>(chunk_, sz - o))});
                    left[f] = pending_q.size(); st_.bytes += sz;
                }
            };
            std::vector<int> owner(depth_, -1);
            for (;;) {
                fill();
                while (!freeb.empty() && qpos < pending_q.size() && !stop_.load(std::memory_order_relaxed)) {
                    auto c = pending_q[qpos++]; unsigned b = freeb.back(); freeb.pop_back(); owner[b] = c.file;
                    rd.read(fds[std::size_t(c.file)], c.off, bufs[b].data(), c.len, b); ++inflight;
                    if (qpos >= pending_q.size()) fill();
                }
                if (!inflight) break;
                for (auto& c : rd.wait(1)) {
                    --inflight; const auto b = unsigned(c.user); const int f = owner[b]; freeb.push_back(b);
                    if (c.result < 0) ++st_.failed;
                    if (--left[std::size_t(f)] == 0) { close_fd(fds[std::size_t(f)]); fds[std::size_t(f)] = -1; }
                }
            }
            for (auto fd : fds) if (fd >= 0) close_fd(fd);
        } catch (...) { ++st_.failed; }
    }
    std::vector<std::string> paths_; unsigned depth_; std::size_t chunk_; double skip_;
    std::atomic<bool> stop_{false}; std::thread th_; stats_t st_;
};

} // namespace detail

// Drops the file's pages from the page cache (posix_fadvise DONTNEED; works without root; clean pages only). For measurements.
inline void drop_page_cache(const std::string& path) { detail::fadvise_file(path, 0); }

// ---- NUMA placement ----------------------------------------------------------------------------------------------
struct placement_stats { std::atomic<std::size_t> local{0}, stolen{0}; };

// Maps a CPU to the index of its NUMA node (position in `nodes`); built once.
class node_map {
public:
    node_map() : nodes_(os::numa_nodes()) { build_(); }
    explicit node_map(std::vector<os::numa_node> nodes) : nodes_(std::move(nodes)) { build_(); }
    std::size_t count() const { return nodes_.empty() ? 1 : nodes_.size(); }
    const std::vector<os::numa_node>& nodes() const { return nodes_; }
    // Index (0 .. count()-1) of the node the calling thread runs on (0 when unknown).
    std::size_t current() const {
        const int c = os::current_cpu();
        return c >= 0 && std::size_t(c) < cpu_.size() && cpu_[std::size_t(c)] >= 0 ? std::size_t(cpu_[std::size_t(c)]) : 0;
    }
private:
    void build_() {
        for (std::size_t i = 0; i < nodes_.size(); ++i) for (auto c : nodes_[i].cpus) { if (cpu_.size() <= c) cpu_.resize(std::size_t(c) + 1, -1); cpu_[c] = int(i); }
    }
    std::vector<os::numa_node> nodes_; std::vector<int> cpu_;
};
inline const node_map& system_node_map() { static const node_map m; return m; }

// Per-node queues of work positions 0..n-1: position p has its home on node p * nodes / n (contiguous runs). take(node) hands out
// the next position of that node, else steals from the following nodes; -1 when everything is taken.
class node_queues {
public:
    node_queues(std::size_t n, std::size_t nodes) : n_(n), nodes_(nodes ? nodes : 1), cur_(nodes_) {
        for (std::size_t k = 0; k < nodes_; ++k) { lo_.push_back(k * n / nodes_); hi_.push_back((k + 1) * n / nodes_); cur_[k].store(lo_[k]); }
    }
    long take(std::size_t node, bool& stolen) {
        for (std::size_t j = 0; j < nodes_; ++j) {
            const std::size_t k = (node + j) % nodes_;
            if (cur_[k].load(std::memory_order_relaxed) >= hi_[k]) continue;
            const std::size_t p = cur_[k].fetch_add(1, std::memory_order_relaxed);
            if (p < hi_[k]) { stolen = j != 0; return long(p); }
        }
        return -1;
    }
private:
    std::size_t n_, nodes_; std::vector<std::atomic<std::size_t>> cur_; std::vector<std::size_t> lo_, hi_;
};

} // namespace forge::db

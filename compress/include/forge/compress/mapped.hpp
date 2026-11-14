// SSD-backed memory for forge::compress: a std::vector-like buffer whose large allocations live in shared mappings
// of scratch files, so the OS can write dirty pages back to the disk and drop them under memory pressure instead of
// the OOM killer ending a long run.
//
//   mapped_vector<T>   data() size() resize() assign() reserve() push_back() iterators operator[] == ...; it converts
//                      to and from std::vector<T>. Allocations of at least scratch::threshold() bytes are scratch-file
//                      mappings; smaller ones are ordinary (64-byte aligned) heap memory.
//   lazy sources       a buffer can also stand for values held elsewhere, e.g. a tensor of a mapped .fpk: an f32
//                      source is read in place until the first mutable access copies it; other codecs are decoded
//                      on first access (mapped_vector<float>::lazy).
//
// Environment:
//   FORGE_COMPRESS_MMAP_MIN   threshold in bytes ("64M", "1G", ...; default 64 MiB; "off" = never map)
//   FORGE_COMPRESS_MMAP_WEIGHTS_MIN   the threshold of model weights (set_use(use::weights); default 1 MiB)
//   FORGE_COMPRESS_SCRATCH    directory of the scratch files (default: scratch::set_dir(), which forge compress sets
//                             to the output's directory, else the system temp directory)
//
// Scratch files are named forge-compress-<pid>-<n>.tmp. They are created exclusively, sized up front (and, on
// Linux, preallocated with posix_fallocate so that a full disk is an allocation failure, not a SIGBUS on a later
// write), mapped through forge::os::mapping (shared, read-write), and unlinked right away on POSIX: the mapping
// keeps the space until it is unmapped, and nothing is left behind if the process dies. On Windows a mapped file
// cannot be deleted, so it is deleted when the buffer is freed, by an exit handler, and by a sweep of stale files
// at the first allocation of a later run. A failed scratch allocation falls back to the heap (with one warning).
#pragma once
#include <forge/os/os.hpp>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#endif

namespace forge::compress {

// ---------------------------------------------------------------------------------------------------- settings
namespace scratch {

inline std::size_t parse_bytes(std::string s, std::size_t dflt) {
    if (s.empty()) return dflt;
    for (auto& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    if (s == "off" || s == "none" || s == "never") return SIZE_MAX;
    char* e = nullptr;
    const double v = std::strtod(s.c_str(), &e);
    if (e == s.c_str() || !(v >= 0)) return dflt;
    std::string u = e;
    while (!u.empty() && u.front() == ' ') u.erase(0, 1);
    if (!u.empty() && u.back() == 'b') u.pop_back();
    if (!u.empty() && u.back() == 'i') u.pop_back();
    double m = 1;
    if (u == "k") m = 1024.0; else if (u == "m") m = 1048576.0; else if (u == "g") m = 1073741824.0; else if (u == "t") m = 1099511627776.0;
    else if (!u.empty()) return dflt;
    return std::size_t(v * m);
}

struct state {
    std::mutex mu;
    std::atomic<std::size_t> threshold{parse_bytes(os::getenv("FORGE_COMPRESS_MMAP_MIN", ""), std::size_t(64) << 20)};
    std::atomic<std::size_t> weights{parse_bytes(os::getenv("FORGE_COMPRESS_MMAP_WEIGHTS_MIN", ""), std::size_t(1) << 20)};
    std::string dir_default;                 // scratch::set_dir()
    std::set<std::string> live_paths;        // Windows: files to delete when their buffer is freed / at exit
    std::atomic<std::uint64_t> counter{0};
    std::atomic<std::int64_t> files_live{0}, bytes_live{0};
    std::atomic<std::uint64_t> files_created{0};
    bool warned = false, swept = false;
};
inline state& st() { static state* s = new state; return *s; }   // never destroyed: buffers may outlive static teardown

// What a buffer holds: model weights (WeightStore tensors) get a lower threshold than compute buffers, since they
// are long-lived and together make up most of the memory of a job.
enum class use : std::uint8_t { general, weights };
// Allocations of at least this many bytes go to scratch files (SIZE_MAX: never). Weights: the smaller of
// FORGE_COMPRESS_MMAP_WEIGHTS_MIN (default 1 MiB) and the general threshold; never when that one is off.
inline std::size_t threshold(use u = use::general) {
    const std::size_t g = st().threshold.load(std::memory_order_relaxed);
    if (u == use::general || g == SIZE_MAX) return g;
    return std::min(g, st().weights.load(std::memory_order_relaxed));
}
inline void set_threshold(std::size_t b) { st().threshold.store(b); }
inline void set_weights_threshold(std::size_t b) { st().weights.store(b); }
// Default scratch directory when FORGE_COMPRESS_SCRATCH is unset (forge compress: the output's directory).
inline void set_dir(const std::string& d) { std::lock_guard lk(st().mu); st().dir_default = d; }
inline std::string dir() {
    if (auto e = os::getenv("FORGE_COMPRESS_SCRATCH", ""); !e.empty()) return e;
    std::lock_guard lk(st().mu);
    return st().dir_default.empty() ? os::temp_dir() : st().dir_default;
}

struct stats { std::int64_t files_live = 0, bytes_live = 0; std::uint64_t files_created = 0; };
inline stats current() { return {st().files_live.load(), st().bytes_live.load(), st().files_created.load()}; }

inline void warn_once(const std::string& m) {
    std::lock_guard lk(st().mu);
    if (st().warned) return;
    st().warned = true;
    std::fprintf(stderr, "forge compress: scratch mapping unavailable (%s); large buffers stay in RAM\n", m.c_str());
}

inline void remove_live_files() {   // exit handler (Windows; a no-op on POSIX where files are unlinked at once)
    std::set<std::string> v;
    { std::lock_guard lk(st().mu); v.swap(st().live_paths); }
    for (auto& p : v) (void)os::remove(p);
}

// Removes scratch files left by runs that died (Windows; POSIX files only exist for an instant before the unlink,
// so only files older than a minute are touched there). Live files of other processes are mapped: on Windows they
// cannot be deleted, so the attempt fails harmlessly.
inline void sweep(const std::string& d) {
    auto l = os::list_dir(d);
    if (!l) return;
    const std::int64_t now = os::wall_ns();
    for (auto& e : l.value) {
        if (e.is_dir || !e.name.starts_with("forge-compress-") || !e.name.ends_with(".tmp")) continue;
        const std::string p = os::join(d, e.name);
#if !defined(_WIN32)
        auto i = os::stat(p);
        if (!i.exists || now - i.mtime_ns < std::int64_t(60) * 1000000000) continue;
#else
        (void)now;
#endif
        (void)os::remove(p);
    }
}

struct region { void* p = nullptr; std::size_t bytes = 0; std::string path; };

// A zero-filled, writable, file-backed region of `bytes` (rounded up to the allocation granularity), or {} when no
// scratch file could be made.
inline region map(std::size_t bytes) {
    const std::size_t g = std::max<std::size_t>(os::allocation_granularity(), 4096);
    bytes = (bytes + g - 1) / g * g;
    const std::string d = dir();
    {
        bool first = false;
        { std::lock_guard lk(st().mu); first = !st().swept; st().swept = true; }
        if (first) {
            (void)os::mkdirs(d);
            sweep(d);
#if defined(_WIN32)
            std::atexit(remove_live_files);
#endif
        }
    }
    std::string path;
    for (int tries = 0; tries < 8; ++tries) {
        path = os::join(d, std::format("forge-compress-{}-{}.tmp", os::pid(), st().counter.fetch_add(1)));
        auto f = os::file::open(path, os::rdwr | os::create | os::excl);
        if (!f) { if (tries == 7) { warn_once(f.err.what); return {}; } continue; }
        auto sz = f->resize(bytes);
#if defined(__linux__)
        // reserve the blocks: a full disk must fail here, not as SIGBUS on a page written later
        if (sz) { int rc = ::posix_fallocate(f->native(), 0, off_t(bytes)); if (rc != 0 && rc != EOPNOTSUPP && rc != EINVAL) sz = {{}, os::errno_error(rc, "fallocate " + path)}; }
#endif
        f->close();
        if (!sz) { (void)os::remove(path); warn_once(sz.err.what); return {}; }
        break;
    }
    auto m = os::mapping::map_file(path, os::prot_read | os::prot_write);
    if (!m) { (void)os::remove(path); warn_once(m.err.what); return {}; }
    region r; r.bytes = m->size(); r.p = m->release();
#if defined(_WIN32)
    r.path = path;
    { std::lock_guard lk(st().mu); st().live_paths.insert(path); }
#else
    (void)os::remove(path);   // unlinked early: the mapping holds the space until it is unmapped
#endif
    st().files_live.fetch_add(1); st().bytes_live.fetch_add(std::int64_t(r.bytes)); st().files_created.fetch_add(1);
    return r;
}

inline void unmap(region& r) {
    if (!r.p) return;
#if defined(_WIN32)
    UnmapViewOfFile(r.p);
    if (!r.path.empty()) { (void)os::remove(r.path); std::lock_guard lk(st().mu); st().live_paths.erase(r.path); }
#else
    ::munmap(r.p, r.bytes);
#endif
    st().files_live.fetch_sub(1); st().bytes_live.fetch_sub(std::int64_t(r.bytes));
    r = {};
}

}  // namespace scratch

// ---------------------------------------------------------------------------------------------------- buffer
template <class T>
class mapped_vector {
    static_assert(std::is_trivially_copyable_v<T>, "mapped_vector holds trivially copyable values");

public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;

    // Values held elsewhere (a mapped file): `borrow` points at n values usable in place (else nullptr), `fill`
    // writes the n values to a destination (decode / copy). `owner` keeps the source alive.
    struct lazy_source {
        std::shared_ptr<const void> owner;
        const T* borrow = nullptr;
        std::function<void(T*)> fill;
    };

    mapped_vector() = default;
    explicit mapped_vector(size_type n) { resize(n); }
    mapped_vector(size_type n, const T& v) { assign(n, v); }
    template <class It, class = std::enable_if_t<!std::is_integral_v<It>>>
    mapped_vector(It first, It last) { assign(first, last); }
    mapped_vector(std::initializer_list<T> l) { assign(l.begin(), l.end()); }
    mapped_vector(const std::vector<T>& v) { assign(v.begin(), v.end()); }   // NOLINT: implicit, like the vector it replaces
    mapped_vector(const mapped_vector& o) { copy_from_(o); }
    mapped_vector(mapped_vector&& o) noexcept { steal_(o); }
    ~mapped_vector() { release_(); }

    // Assignments keep this buffer's use(); copies and moves take the source's.
    mapped_vector& operator=(const mapped_vector& o) {
        if (this == &o) return *this;
        if (o.lz_ && !o.p_) { release_(); n_ = o.n_; lz_ = std::make_shared<lazy_state>(o.lz_->src); return *this; }
        const T* d = o.data();
        assign(d, d + o.n_);
        return *this;
    }
    mapped_vector& operator=(mapped_vector&& o) noexcept { if (this != &o) { const auto u = use_; release_(); steal_(o); use_ = u; } return *this; }
    mapped_vector& operator=(const std::vector<T>& v) { assign(v.begin(), v.end()); return *this; }
    mapped_vector& operator=(std::initializer_list<T> l) { assign(l.begin(), l.end()); return *this; }

    // A buffer of n values from a lazy source (see lazy_source).
    static mapped_vector lazy(size_type n, lazy_source src) {
        mapped_vector v;
        v.n_ = n;
        if (n) v.lz_ = std::make_shared<lazy_state>(std::move(src));
        return v;
    }
    // true while the values still live in the source (no copy made yet).
    bool is_lazy() const { return lz_ != nullptr && !p_; }
    // true when the storage is a scratch-file mapping.
    bool is_mapped() const { return reg_.p != nullptr; }
    // Which threshold decides between heap and scratch file for this buffer's future allocations.
    void set_use(scratch::use u) { use_ = u; }
    scratch::use get_use() const { return use_; }

    operator std::vector<T>() const { const T* d = data(); return std::vector<T>(d, d + n_); }   // NOLINT: FORGE APIs take vectors

    // ---- access
    T* data() { mut_(); return p_; }
    const T* data() const { return n_ ? cdata_() : p_; }
    size_type size() const { return n_; }
    bool empty() const { return n_ == 0; }
    size_type capacity() const { return lz_ && !p_ ? n_ : cap_; }
    T& operator[](size_type i) { return data()[i]; }
    const T& operator[](size_type i) const { return data()[i]; }
    T& at(size_type i) { if (i >= n_) throw std::out_of_range("mapped_vector::at"); return data()[i]; }
    const T& at(size_type i) const { if (i >= n_) throw std::out_of_range("mapped_vector::at"); return data()[i]; }
    T& front() { return data()[0]; }
    const T& front() const { return data()[0]; }
    T& back() { return data()[n_ - 1]; }
    const T& back() const { return data()[n_ - 1]; }
    iterator begin() { return data(); }
    iterator end() { return data() + n_; }
    const_iterator begin() const { return data(); }
    const_iterator end() const { return data() + n_; }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }
    std::span<T> span() { return {data(), n_}; }
    std::span<const T> span() const { return {data(), n_}; }

    // ---- modifiers
    void reserve(size_type n) { mut_(); if (n > cap_) realloc_(n); }
    void shrink_to_fit() { mut_(); if (cap_ > n_) realloc_(n_); }
    void clear() { mut_(); n_ = 0; }
    void resize(size_type n) { resize_(n, nullptr); }
    void resize(size_type n, const T& v) { resize_(n, &v); }
    void assign(size_type n, const T& v) {
        lz_.reset();
        if (n > cap_ || (n < cap_ / 2 && cap_ * sizeof(T) >= scratch::threshold(use_))) { release_(); alloc_(n); fill_(0, n, v, true); }
        else fill_(0, n, v, false);
        n_ = n;
    }
    template <class It, class = std::enable_if_t<!std::is_integral_v<It>>>
    void assign(It first, It last) {
        if constexpr (std::is_base_of_v<std::forward_iterator_tag, typename std::iterator_traits<It>::iterator_category>) {
            const size_type n = size_type(std::distance(first, last));
            lz_.reset();
            if (n > cap_ || (n < cap_ / 2 && cap_ * sizeof(T) >= scratch::threshold(use_))) { release_(); alloc_(n); }
            std::copy(first, last, p_);
            n_ = n;
        } else { clear(); for (; first != last; ++first) push_back(*first); }
    }
    void assign(std::initializer_list<T> l) { assign(l.begin(), l.end()); }
    void push_back(const T& v) { mut_(); if (n_ == cap_) { const T c = v; realloc_(grow_(n_ + 1)); p_[n_++] = c; return; } p_[n_++] = v; }
    template <class... A> T& emplace_back(A&&... a) { push_back(T(std::forward<A>(a)...)); return back(); }
    void pop_back() { mut_(); --n_; }
    template <class It, class = std::enable_if_t<!std::is_integral_v<It>>>
    iterator insert(const_iterator pos, It first, It last) {
        const std::vector<T> tmp(first, last);   // the range may alias this buffer
        const size_type at = size_type(pos - cdata_or_null_()), k = tmp.size();
        mut_();
        if (n_ + k > cap_) realloc_(grow_(n_ + k));
        std::memmove(p_ + at + k, p_ + at, (n_ - at) * sizeof(T));
        std::copy(tmp.begin(), tmp.end(), p_ + at);
        n_ += k;
        return p_ + at;
    }
    iterator insert(const_iterator pos, const T& v) { const T* b = &v; return insert(pos, b, b + 1); }
    iterator erase(const_iterator first, const_iterator last) {
        const size_type a = size_type(first - cdata_or_null_()), b = size_type(last - cdata_or_null_());
        mut_();
        std::memmove(p_ + a, p_ + b, (n_ - b) * sizeof(T));
        n_ -= b - a;
        return p_ + a;
    }
    iterator erase(const_iterator pos) { return erase(pos, pos + 1); }
    void swap(mapped_vector& o) noexcept {
        std::swap(p_, o.p_); std::swap(n_, o.n_); std::swap(cap_, o.cap_); std::swap(reg_, o.reg_); std::swap(lz_, o.lz_); std::swap(zeroed_, o.zeroed_);
    }
    friend void swap(mapped_vector& a, mapped_vector& b) noexcept { a.swap(b); }

    // ---- comparison (element-wise, like std::vector)
    friend bool operator==(const mapped_vector& a, const mapped_vector& b) { return eq_(a.data(), a.n_, b.data(), b.n_); }
    friend bool operator==(const mapped_vector& a, const std::vector<T>& b) { return eq_(a.data(), a.n_, b.data(), b.size()); }
    friend bool operator==(const std::vector<T>& a, const mapped_vector& b) { return eq_(a.data(), a.size(), b.data(), b.n_); }

private:
    struct lazy_state {
        explicit lazy_state(lazy_source s) : src(std::move(s)) {}
        lazy_source src;
        std::once_flag once;   // first const access of a source that cannot be read in place
    };

    static bool eq_(const T* a, size_type na, const T* b, size_type nb) {
        if (na != nb) return false;
        if (a == b) { if constexpr (std::is_floating_point_v<T>) { for (size_type i = 0; i < na; ++i) if (a[i] != a[i]) return false; } return true; }   // NaN != NaN
        return std::equal(a, a + na, b);
    }
    static size_type grow_(size_type need) { return std::max<size_type>(need, std::max<size_type>(16, size_type(double(need) * 1.5))); }

    const T* cdata_or_null_() const { return n_ ? data() : p_; }
    const T* cdata_() const {
        if (!lz_) return p_;
        if (lz_->src.borrow && !p_) return lz_->src.borrow;
        std::call_once(lz_->once, [this] { auto* self = const_cast<mapped_vector*>(this); self->materialize_(); });
        return p_;
    }
    // Before a mutable access: the values get their own storage.
    void mut_() {
        if (!lz_) return;
        if (!p_) materialize_();
        lz_.reset();
    }
    void materialize_() {
        if (p_) return;
        auto src = lz_->src;   // keep the source alive while copying (lz_ may be the last owner)
        const size_type n = n_;
        alloc_(n);
        if (src.borrow) std::memcpy(p_, src.borrow, n * sizeof(T));
        else if (src.fill) src.fill(p_);
    }
    // Storage for n values (contents unspecified, except that a fresh mapping reads as zero).
    void alloc_(size_type n) {
        p_ = nullptr; cap_ = 0; reg_ = {};
        if (!n) return;
        const std::size_t bytes = n * sizeof(T);
        if (bytes >= scratch::threshold(use_)) {
            reg_ = scratch::map(bytes);
            if (reg_.p) { p_ = static_cast<T*>(reg_.p); cap_ = reg_.bytes / sizeof(T); zeroed_ = true; return; }
        }
        p_ = static_cast<T*>(::operator new(bytes, std::align_val_t(64)));
        cap_ = n; zeroed_ = false;
    }
    void free_storage_() {
        if (reg_.p) scratch::unmap(reg_);
        else if (p_) ::operator delete(p_, std::align_val_t(64));
        p_ = nullptr; cap_ = 0;
    }
    void release_() { free_storage_(); n_ = 0; lz_.reset(); }
    void realloc_(size_type cap) {
        T* old = p_; const size_type n = std::min(n_, cap);
        scratch::region oreg = reg_;
        alloc_(cap);
        if (n) std::memcpy(p_, old, n * sizeof(T));
        if (oreg.p) scratch::unmap(oreg);
        else if (old) ::operator delete(old, std::align_val_t(64));
    }
    // [a, b) = v; `fresh`: the storage was just allocated (a fresh mapping is already zero).
    void fill_(size_type a, size_type b, const T& v, bool fresh) {
        if (fresh && zeroed_ && is_zero_(v)) return;
        std::fill(p_ + a, p_ + b, v);
    }
    static bool is_zero_(const T& v) { T z{}; return std::memcmp(&v, &z, sizeof(T)) == 0; }
    void resize_(size_type n, const T* v) {
        mut_();
        if (n <= n_) {
            n_ = n;
            if (!n && cap_ * sizeof(T) >= scratch::threshold(use_)) free_storage_();   // give big scratch space back
            return;
        }
        const T val = v ? *v : T{};
        bool fresh = false;
        if (n > cap_) {
            if (!n_) free_storage_(), alloc_(n);
            else realloc_(n);
            fresh = true;   // the tail [n_, n) of new storage is untouched (zero when it is a fresh mapping)
        }
        fill_(n_, n, val, fresh);
        n_ = n;
    }
    void copy_from_(const mapped_vector& o) {
        use_ = o.use_;
        if (o.lz_ && !o.p_) {   // still in its source: share the source, copy nothing
            n_ = o.n_;
            lz_ = std::make_shared<lazy_state>(o.lz_->src);
            return;
        }
        const T* d = o.data();
        alloc_(o.n_);
        if (o.n_) std::memcpy(p_, d, o.n_ * sizeof(T));
        n_ = o.n_;
    }
    void steal_(mapped_vector& o) noexcept {
        p_ = std::exchange(o.p_, nullptr); n_ = std::exchange(o.n_, 0); cap_ = std::exchange(o.cap_, 0);
        reg_ = std::exchange(o.reg_, {}); lz_ = std::move(o.lz_); zeroed_ = o.zeroed_; use_ = o.use_;
    }

    mutable T* p_ = nullptr;
    size_type n_ = 0;
    mutable size_type cap_ = 0;
    mutable scratch::region reg_;
    std::shared_ptr<lazy_state> lz_;
    mutable bool zeroed_ = false;
    scratch::use use_ = scratch::use::general;
};

using fbuf = mapped_vector<float>;

}  // namespace forge::compress

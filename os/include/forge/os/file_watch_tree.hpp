#pragma once
// forge::os recursive, debounced file watching. tree_watcher watches a whole directory tree: every directory is
// registered with the native file_watcher (inotify / kqueue / ReadDirectoryChangesW), new directories are picked
// up as they appear (their contents are reported as created, closing the mkdir-then-write race), and a
// per-path snapshot turns coarse "directory changed" notifications (kqueue) into exact created / modified /
// removed events. Where native watching is unavailable, or with force_polling, it rescans on each poll().
// Bursts are coalesced: poll() waits until the tree has been quiet for debounce_ms and reports each path once.
// Guide: docs/os-wrapper.md.
#include <forge/os/file_watch.hpp>
#include <forge/os/fs.hpp>
#include <map>
#include <set>

namespace forge::os {
struct tree_watch_opts { int debounce_ms = 50; bool force_polling = false, include_hidden = true; std::vector<std::string> ignore; };

class tree_watcher {
public:
    static result<std::unique_ptr<tree_watcher>> open(const std::string& root, tree_watch_opts o = {}) {
        if (!stat(root).is_dir) return {{}, {2, "watch: not a directory " + root}};
        auto w = std::unique_ptr<tree_watcher>(new tree_watcher(root, std::move(o)));
        w->polling_ = w->opts_.force_polling;
        if (!w->polling_ && !w->native_.add(w->root_)) w->polling_ = true;
        w->scan_(w->root_, nullptr);
        return {std::move(w), {}};
    }
    const char* backend() const { return polling_ ? "polling" : native_.backend(); }
    std::size_t watched_dirs() const { return dirs_.size(); }
    // Wait up to timeout_ms (-1 forever) for a change, then debounce and return the coalesced events.
    std::vector<watch_event> poll(int timeout_ms) {
        std::map<std::string, watch_kind> acc; std::vector<std::string> order;
        auto take = [&](std::vector<watch_event> ev) {
            for (auto& e : ev) {
                auto it = acc.find(e.path);
                if (it == acc.end()) { acc.emplace(e.path, e.kind); order.push_back(e.path); continue; }
                auto& k = it->second;
                if (k == watch_kind::created && e.kind == watch_kind::removed) k = watch_kind::other;      // came and went: drop
                else if (k == watch_kind::created && e.kind == watch_kind::modified) {}                    // still "created"
                else if (k == watch_kind::other && e.kind == watch_kind::created) k = watch_kind::created;
                else if (k == watch_kind::removed && e.kind == watch_kind::created) k = watch_kind::modified; // replaced
                else k = e.kind;
            }
        };
        auto start = monotonic_ns();
        auto remaining = [&]() -> int { if (timeout_ms < 0) return -1; auto el = int((monotonic_ns() - start) / 1000000); return el >= timeout_ms ? 0 : timeout_ms - el; };
        // First batch: block until something happens (polling mode samples at most every 100 ms).
        for (;;) {
            take(step_(polling_ ? (timeout_ms < 0 ? 100 : std::min(remaining(), 100)) : remaining()));
            if (!acc.empty() || remaining() == 0) break;
        }
        // Then keep draining until the tree is quiet for debounce_ms (bounded at 20 rounds).
        for (int i = 0; !acc.empty() && opts_.debounce_ms > 0 && i < 20; ++i) {
            auto before = acc.size(); auto snap = acc;
            take(step_(opts_.debounce_ms));
            if (acc.size() == before && acc == snap) break;
        }
        std::vector<watch_event> out;
        for (auto& p : order) { auto k = acc[p]; if (k != watch_kind::other) out.push_back({p, k}); }
        return out;
    }
private:
    struct entry { bool is_dir = false; std::int64_t mtime = 0; std::uint64_t size = 0; };
    tree_watcher(std::string root, tree_watch_opts o) : root_(normalize_(std::move(root))), opts_(std::move(o)) {}
    static std::string normalize_(std::string p) { while (p.size() > 1 && (p.back() == '/' || p.back() == '\\')) p.pop_back(); return p; }
    bool ignored_(const std::string& path) const {
        auto rel = path.size() > root_.size() ? path.substr(root_.size() + 1) : std::string();
        auto base = basename(path);
        if (!opts_.include_hidden && !base.empty() && base[0] == '.') return true;
        for (auto& g : opts_.ignore) if (fs::glob_match(g, rel) || fs::glob_match(g, base)) return true;
        return false;
    }
    static entry entry_of_(const std::string& p, bool& exists) { auto st = stat(p); exists = st.exists; return {st.is_dir, st.mtime_ns, st.size}; }
    // Record (and, when out != null, report as created) everything under dir; register dirs natively.
    void scan_(const std::string& dir, std::vector<watch_event>* out) {
        if (!dirs_.insert(dir).second && out) {}
        if (dir != root_ && !polling_) native_.add(dir);
        auto ls = list_dir(dir); if (!ls) return;
        for (auto& e : ls.value) {
            auto p = join(dir, e.name); if (ignored_(p)) continue;
            bool ex = false; auto en = entry_of_(p, ex); if (!ex) continue;
            bool fresh = !snap_.count(p);
            snap_[p] = en;
            if (fresh && out) out->push_back({p, watch_kind::created});
            if (!en.is_dir && !polling_) native_.add_file(p);
            if (en.is_dir && !dirs_.count(p)) scan_(p, out);
        }
    }
    // Compare the snapshot of `dir`'s immediate children against the disk.
    void diff_dir_(const std::string& dir, std::vector<watch_event>& out) {
        std::set<std::string> seen;
        if (auto ls = list_dir(dir)) for (auto& e : ls.value) {
            auto p = join(dir, e.name); if (ignored_(p)) continue; seen.insert(p); check_(p, out);
        }
        auto pre = dir + "/";
        std::vector<std::string> lost;   // collect first: gone_() erases whole subtrees from snap_
        for (auto it = snap_.lower_bound(pre); it != snap_.end() && it->first.compare(0, pre.size(), pre) == 0; ++it)
            if (it->first.find('/', pre.size()) == std::string::npos && !seen.count(it->first)) lost.push_back(it->first);
        for (auto& p : lost) gone_(p, out);
    }
    void gone_(const std::string& p, std::vector<watch_event>& out) {
        auto it = snap_.find(p); if (it == snap_.end()) return;
        bool was_dir = it->second.is_dir; snap_.erase(it); out.push_back({p, watch_kind::removed});
        if (was_dir) {
            dirs_.erase(p); auto pre = p + "/";
            for (auto j = snap_.lower_bound(pre); j != snap_.end() && j->first.compare(0, pre.size(), pre) == 0;) { out.push_back({j->first, watch_kind::removed}); j = snap_.erase(j); }
            for (auto d = dirs_.lower_bound(pre); d != dirs_.end() && d->compare(0, pre.size(), pre) == 0;) d = dirs_.erase(d);
        }
    }
    void check_(const std::string& p, std::vector<watch_event>& out) {
        if (ignored_(p)) return;
        bool ex = false; auto en = entry_of_(p, ex);
        auto it = snap_.find(p);
        if (!ex) { gone_(p, out); return; }
        if (it == snap_.end()) { snap_[p] = en; out.push_back({p, watch_kind::created}); if (en.is_dir) scan_(p, &out); else if (!polling_) native_.add_file(p); return; }
        if (it->second.is_dir != en.is_dir) { gone_(p, out); snap_[p] = en; out.push_back({p, watch_kind::created}); if (en.is_dir) scan_(p, &out); return; }
        if (!en.is_dir && (it->second.mtime != en.mtime || it->second.size != en.size)) { it->second = en; out.push_back({p, watch_kind::modified}); }
    }
    std::vector<watch_event> step_(int timeout_ms) {
        std::vector<watch_event> out;
        if (polling_) {
            if (timeout_ms > 0) sleep_ns(std::uint64_t(timeout_ms) * 1000000ull);
            std::vector<std::string> ds(dirs_.begin(), dirs_.end());
            for (auto& d : ds) if (dirs_.count(d)) diff_dir_(d, out);
            return out;
        }
        for (auto& e : native_.poll(timeout_ms)) {
            auto p = normalize_(e.path);
            if (dirs_.count(p) && e.kind != watch_kind::removed) diff_dir_(p, out);   // coarse (kqueue) dir event
            else check_(p, out);
        }
        return out;
    }
    std::string root_; tree_watch_opts opts_; bool polling_ = false;
    file_watcher native_; std::map<std::string, entry> snap_; std::set<std::string> dirs_;
};
} // namespace forge::os

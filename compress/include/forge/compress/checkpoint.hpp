// Crash / OOM persistence of a `forge compress` job (`--checkpoint DIR`, default <out>.ckpt/).
//
// Layout of DIR:
//   lock            held (os::fs::file_lock) while a job uses the directory
//   state.json      {key, stages_done, reports, codec, model, manifest, before, ...}: committed after every stage
//   calib.bin       the calibration token windows        ref.bin   the reference probes and their f16 log-probs
//   stage<i>.fpk    the model after stage i (Model::save, all tensors f32; the per-tensor codecs are in state.json)
//   progress.json   per-layer progress inside the running stage: for every sequential loop ("phase") the last
//                   finished layer, the loop's carry, the hidden-state file and the layer files
//   H_*.bin         hidden states entering the next layer      L_*.bin   the tensors / spec / codecs of finished layers
//
// The key is a hash of the input file (path, size, mtime), the recipe and every option that changes the result
// (nsamples, seqlen, calib-data, seed, damp, probes, probe-len). A directory with another key is cleared and the
// job starts afresh. A resumed job reloads the model of the last finished stage, then runs the interrupted stage
// again from its start: its pre-passes are recomputed (deterministic), and its sequential loops skip the layers a
// checkpoint holds (restoring their tensors, the carry and the hidden state; see layer_resumed in compress.hpp).
// The result is the same as an uninterrupted run (tests/test_compress_persist.cpp).
//
// Per-layer checkpoints are written every `every_layers` layers (--checkpoint-every N, 0 = off) or every
// `every_s` seconds (--checkpoint-minutes M, default 10). FORGE_COMPRESS_ABORT_AFTER=<layer> | <phase>:<layer> |
// stage:<i> throws right after that point was checkpointed (tests; FORGE_COMPRESS_ABORT_OOM=1 throws
// std::bad_alloc instead, to exercise the supervisor's OOM handling).
#pragma once
#include <forge/compress/compress.hpp>
#include <forge/os/fs.hpp>

#include <cstring>

namespace forge::compress {

// ---------------------------------------------------------------------------------------------------- bundles
// A binary file of named arrays: "FGCKPT01", u64 header length, JSON header {meta, entries[{name, dtype, shape,
// offset, bytes}]}, then the 64-byte aligned array data.
struct bundle_entry { std::string name, dtype; shape_t shape; const void* p = nullptr; std::size_t bytes = 0; };

inline void write_bundle(const std::string& path, const value& meta, const std::vector<bundle_entry>& es) {
    std::vector<value> ents;
    std::uint64_t off = 0;
    for (auto& e : es) {
        std::vector<value> sh; for (auto d : e.shape) sh.push_back(value(d));
        ents.push_back(value(jobj{{"name", value(e.name)}, {"dtype", value(e.dtype)}, {"shape", value(sh)}, {"offset", value(double(off))}, {"bytes", value(double(e.bytes))}}));
        off += (e.bytes + 63) / 64 * 64;
    }
    std::string hdr = forge::json::dump(value(jobj{{"meta", meta}, {"entries", value(ents)}}));
    std::string head = "FGCKPT01";
    const std::uint64_t hl = hdr.size();
    head.append(reinterpret_cast<const char*>(&hl), 8);
    head += hdr;
    head.resize((head.size() + 63) / 64 * 64, '\0');
    const std::string part = path + ".part";
    auto f = os::file::open(part, os::wr | os::create | os::truncate);
    if (!f) die("checkpoint: cannot write " + part + ": " + f.err.what);
    auto put = [&](const void* p, std::size_t n) { if (auto s = f->write_all(p, n); !s) die("checkpoint: write " + part + ": " + s.err.what); };
    put(head.data(), head.size());
    static const char zeros[64] = {};
    for (auto& e : es) {
        // write in pieces: Windows moves at most 1 GiB per call, and big mapped buffers page in as they go
        const auto* b = static_cast<const std::uint8_t*>(e.p);
        for (std::size_t o = 0; o < e.bytes; o += std::size_t(256) << 20) put(b + o, std::min(e.bytes - o, std::size_t(256) << 20));
        if (e.bytes % 64) put(zeros, 64 - e.bytes % 64);
    }
    if (auto s = f->sync(); !s) die("checkpoint: sync " + part + ": " + s.err.what);
    f->close();
    if (auto s = os::rename(part, path); !s) die("checkpoint: " + s.err.what);
}

struct bundle {
    os::mapping map;
    value header;
    std::size_t data0 = 0;
    static std::shared_ptr<bundle> open(const std::string& path) {
        auto b = std::make_shared<bundle>();
        auto m = os::mapping::map_file(path, os::prot_read);
        if (!m) die("checkpoint: cannot map " + path + ": " + m.err.what);
        b->map = std::move(m.value);
        const auto* p = b->map.data();
        if (b->map.size() < 16 || std::memcmp(p, "FGCKPT01", 8) != 0) die("checkpoint: " + path + " is not a checkpoint file");
        std::uint64_t hl; std::memcpy(&hl, p + 8, 8);
        if (hl > b->map.size() - 16) die("checkpoint: " + path + " is truncated");
        b->header = forge::json::parse(std::string_view(reinterpret_cast<const char*>(p + 16), std::size_t(hl)));
        b->data0 = std::size_t((16 + hl + 63) / 64 * 64);
        return b;
    }
    const value& meta() const { return header["meta"]; }
    const value* entry(const std::string& n) const {
        for (auto& e : header["entries"].arr()) if (e["name"].str() == n) return &e;
        return nullptr;
    }
    std::span<const std::uint8_t> bytes(const value& e) const {
        const std::size_t off = data0 + std::size_t(e["offset"].num()), n = std::size_t(e["bytes"].num());
        if (off + n > map.size()) die("checkpoint: truncated entry " + e["name"].str());
        return {map.data() + off, n};
    }
    shape_t shape(const value& e) const { shape_t s; for (auto& d : e["shape"].arr()) s.push_back(d.i()); return s; }
};

// ---------------------------------------------------------------------------------------------------- checkpoint
struct CheckpointCfg {
    std::string dir;          // empty: off
    std::string key;          // job identity (input + recipe + options)
    i64 every_layers = 0;     // per-layer checkpoint every N finished layers (0: by time only)
    double every_s = 600;     // ... or when this many seconds passed since the last checkpoint
};

class Checkpoint {
public:
    explicit Checkpoint(CheckpointCfg c) : cfg_(std::move(c)) {
        if (!enabled()) return;
        if (auto s = os::mkdirs(cfg_.dir); !s) die("checkpoint: cannot create " + cfg_.dir + ": " + s.err.what);
        auto l = os::fs::file_lock::acquire(path_("lock"), true, false);
        if (!l) die("checkpoint directory " + cfg_.dir + " is in use by another forge compress");
        lock_.emplace(std::move(l.value));
        if (os::exists(path_("state.json"))) {
            value st = forge::json::parse(read_text_(path_("state.json")));
            if (st.find("key") && st["key"].is_str() && st["key"].str() == cfg_.key) { state_ = st; resumed_ = true; }
            else { say("checkpoint {}: another job (key differs); starting afresh", cfg_.dir); clear_(); }
        } else clear_();
        if (resumed_ && os::exists(path_("progress.json"))) {
            value p = forge::json::parse(read_text_(path_("progress.json")));
            if (p.find("key") && p["key"].str() == cfg_.key) progress_ = p;
        }
        last_save_ = now_s();
    }
    Checkpoint(const Checkpoint&) = delete;
    Checkpoint& operator=(const Checkpoint&) = delete;

    bool enabled() const { return !cfg_.dir.empty(); }
    bool resumed() const { return resumed_; }
    const std::string& dir() const { return cfg_.dir; }
    i64 stages_done() const { return resumed_ ? state_["stages_done"].i() : 0; }
    bool complete() const { return resumed_ && state_.find("complete") && state_["complete"].b(); }
    value report() const { return state_["report"]; }
    std::vector<value> reports() const { return resumed_ ? state_["reports"].arr() : std::vector<value>{}; }
    value before() const { return state_["before"]; }
    // The model file to resume from ("" = the job's input).
    std::string model_path() const { return resumed_ && !state_["model"].str().empty() ? path_(state_["model"].str()) : std::string(); }
    // Model state that Model::save does not keep: per-tensor codecs and the source manifest.
    void restore(Model& m) const {
        m.codec.clear();
        for (auto& [k, v] : state_["codec"].obj()) m.codec[k] = v.str();
        if (state_.find("manifest")) m.manifest = state_["manifest"];
    }

    // A fresh job: persists its inputs (calibration windows, reference) before the first stage.
    void start(const Model& m, const CalibSet& calib, const Reference* ref, const value& before) {
        if (!enabled()) return;
        write_calib_(path_("calib.bin"), calib, "tokens");
        if (ref && !ref->empty()) write_ref_(*ref);
        state_ = value(jobj{{"version", value(1)}, {"key", value(cfg_.key)}, {"stages_done", value(0)}, {"reports", value(std::vector<value>{})},
                            {"codec", value(jobj{})}, {"model", value("")}, {"manifest", m.manifest}, {"before", before}, {"has_ref", value(ref && !ref->empty())}});
        write_state_();
        resumed_ = false;
    }
    CalibSet calib() const { return read_calib_(path_("calib.bin"), "tokens"); }
    bool has_ref() const { return resumed_ && state_["has_ref"].b(); }
    Reference reference() const {
        auto b = bundle::open(path_("ref.bin"));
        Reference R;
        R.vocab = b->meta()["vocab"].i();
        R.probes = calib_of_(*b, "probes");
        for (std::size_t s = 0; s < R.probes.seqs.size(); ++s) {
            auto* e = b->entry(std::format("logp.{}", s));
            if (!e) die("checkpoint: ref.bin lacks logp." + std::to_string(s));
            auto by = b->bytes(*e);
            mapped_vector<std::uint16_t> lp(by.size() / 2);
            std::memcpy(lp.data(), by.data(), by.size());
            R.logp.push_back(std::move(lp));
        }
        return R;
    }

    // ---- stages
    void begin_stage(i64 i) {
        stage_ = i;
        occ_.clear(); active_.clear(); live_.clear(); bundles_.clear();
        if (progress_.is_obj() && progress_["stage"].i() != i) progress_ = value();
    }
    void stage_done(i64 i, Model& m, const std::vector<value>& reports) {
        if (!enabled()) return;
        const std::string name = std::format("stage{}.fpk", i);
        {   // f32 everywhere (exact); the codecs go to state.json
            auto codecs = std::move(m.codec);
            m.codec.clear();
            try { m.save(path_(name)); } catch (...) { m.codec = std::move(codecs); throw; }
            m.codec = std::move(codecs);
        }
        jobj cj; for (auto& [k, v] : m.codec) cj.push_back({k, value(v)});
        const std::string old = state_["model"].str();
        json_set(state_, "stages_done", value(i + 1));
        json_set(state_, "reports", value(reports));
        json_set(state_, "codec", value(cj));
        json_set(state_, "model", value(name));
        write_state_();
        drop_progress_();
        if (!old.empty() && old != name) remove_later_(path_(old));
        sweep_removals_();
        last_save_ = now_s();
        say("checkpoint: stage {} saved ({})", i + 1, path_(name));
        abort_if_("stage", i);
    }
    // After the output is written: the job is complete (a rerun returns `report` without recomputing).
    void mark_complete(const value& report) {
        if (!enabled()) return;
        json_set(state_, "complete", value(true));
        json_set(state_, "report", report);
        write_state_();
    }
    // Removes the directory (the job is done). Models loaded from it must be gone first (Windows keeps mapped files).
    void finish() {
        if (!enabled()) return;
        bundles_.clear();
        lock_.reset();
        if (auto s = os::remove_all(cfg_.dir); !s) say("checkpoint: could not remove {}: {}", cfg_.dir, s.err.what);
    }

    // ---- the per-layer hook (Ctx::on_layer)
    bool on_layer(Model& m, const LayerEvent& e) {
        if (!enabled()) return false;
        const std::string ph = e.phase;
        if (!e.done) {
            if (e.layer == 0 || !active_.count(ph)) {   // a loop starts: its n-th run in this stage
                const std::string k = ph + "#" + std::to_string(occ_[ph]++);
                active_[ph] = k;
                live_[k] = Live{};
            }
            const std::string& k = active_[ph];
            const value* p = phase_(k);
            if (!p) return false;
            const i64 D = (*p)["done"].i();
            if (e.layer > D) return false;
            restore_layer_(m, *p, e.layer);
            *e.carry = (*p)["carry"];
            if (e.layer == D) { load_hidden_(path_((*p)["hidden"].str()), *e.h); say("checkpoint: {} resumed after layer {}", k, D); }
            live_[k].first_unsaved = D + 1;
            return true;
        }
        if (!active_.count(ph)) return false;
        const std::string k = active_[ph];
        auto& lv = live_[k];
        if (e.modified) lv.modified.insert(e.layer);
        const bool abort = abort_match_(ph, e.layer);
        const bool due = abort || (cfg_.every_layers > 0 && e.layer + 1 - lv.first_unsaved >= cfg_.every_layers) || now_s() - last_save_ >= cfg_.every_s;
        if (due) save_layer_(m, k, e, lv);
        if (abort) abort_if_(ph, e.layer, true);
        return false;
    }

private:
    struct Live { i64 first_unsaved = 0; std::set<i64> modified; };

    std::string path_(const std::string& n) const { return os::join(cfg_.dir, n); }
    static std::string read_text_(const std::string& p) {
        auto r = os::read_file(p);
        if (!r) die("checkpoint: cannot read " + p + ": " + r.err.what);
        return r.value;
    }
    static std::string safe_(const std::string& s) { std::string o; for (char ch : s) o += std::isalnum(static_cast<unsigned char>(ch)) ? ch : '_'; return o; }
    void write_state_() {
        if (auto s = os::fs::write_file_atomic(path_("state.json"), forge::json::dump(state_)); !s) die("checkpoint: " + s.err.what);
    }
    void clear_() {   // everything but the lock
        auto l = os::list_dir(cfg_.dir);
        if (!l) return;
        for (auto& e : l.value) if (e.name != "lock") (void)os::remove_all(path_(e.name));
    }
    void remove_later_(const std::string& p) { if (!os::remove(p)) pending_.push_back(p); }   // Windows: still mapped
    void sweep_removals_() { std::vector<std::string> keep; for (auto& p : pending_) if (os::exists(p) && !os::remove(p)) keep.push_back(p); pending_.swap(keep); }

    // ---- calibration / reference files
    static void write_calib_(const std::string& path, const CalibSet& c, const char* name, const value& meta = value(jobj{}), std::vector<bundle_entry> extra = {}) {
        std::vector<std::int32_t> tok;
        for (auto& s : c.seqs) for (int t : s) tok.push_back(std::int32_t(t));
        std::vector<bundle_entry> es{{name, "i32", {i64(c.seqs.size()), c.seq}, tok.data(), tok.size() * 4}};
        for (auto& x : extra) es.push_back(x);
        write_bundle(path, meta, es);
    }
    static CalibSet calib_of_(const bundle& b, const char* name) {
        auto* e = b.entry(name);
        if (!e) die(std::string("checkpoint: no ") + name);
        auto sh = b.shape(*e);
        auto by = b.bytes(*e);
        CalibSet c; c.seq = sh[1];
        const auto* p = reinterpret_cast<const std::int32_t*>(by.data());
        for (i64 i = 0; i < sh[0]; ++i) c.seqs.emplace_back(p + i * sh[1], p + (i + 1) * sh[1]);
        return c;
    }
    static CalibSet read_calib_(const std::string& path, const char* name) { auto b = bundle::open(path); return calib_of_(*b, name); }
    void write_ref_(const Reference& R) {
        std::vector<bundle_entry> lp;
        for (std::size_t s = 0; s < R.logp.size(); ++s) lp.push_back({std::format("logp.{}", s), "u16", {i64(R.logp[s].size())}, R.logp[s].data(), R.logp[s].size() * 2});
        write_calib_(path_("ref.bin"), R.probes, "probes", value(jobj{{"vocab", value(R.vocab)}}), lp);
    }

    // ---- hidden states
    void write_hidden_(const std::string& path, const Hidden& h) {
        write_bundle(path, value(jobj{{"n", value(h.n)}, {"seq", value(h.seq)}, {"d", value(h.d)}}),
                     {{"hidden", "f32", {h.n, h.seq, h.d}, h.x.data(), h.x.size() * 4}});
    }
    static void load_hidden_(const std::string& path, Hidden& h) {
        auto b = bundle::open(path);
        h.n = b->meta()["n"].i(); h.seq = b->meta()["seq"].i(); h.d = b->meta()["d"].i();
        auto* e = b->entry("hidden");
        if (!e) die("checkpoint: " + path + " has no hidden state");
        auto by = b->bytes(*e);
        h.x.resize(by.size() / 4);
        std::memcpy(h.x.data(), by.data(), by.size());
    }

    // ---- layers
    const value* phase_(const std::string& k) const {
        if (!progress_.is_obj()) return nullptr;
        auto* ps = progress_.find("phases");
        return ps ? ps->find(k) : nullptr;
    }
    static std::string lprefix_(i64 l) { return std::format("layer.{}.", l); }
    void save_layer_(Model& m, const std::string& k, const LayerEvent& e, Live& lv) {
        // the layers finished since the last checkpoint that the loop changed
        std::vector<i64> ls;
        for (i64 l : lv.modified) if (l >= lv.first_unsaved && l <= e.layer) ls.push_back(l);
        value ph = phase_(k) ? *phase_(k) : value(jobj{{"chunks", value(std::vector<value>{})}});
        std::vector<value> chunks = ph["chunks"].arr();
        const std::string tag = std::format("{}_{}", stage_, safe_(k));
        if (!ls.empty()) {
            jobj lm;
            std::vector<bundle_entry> es;
            for (i64 l : ls) {
                const std::string pre = lprefix_(l);
                const auto& L = m.spec.layers[std::size_t(l)];
                std::vector<value> lr, tn;
                for (auto& n : m.spec.lowrank) if (n.starts_with(pre)) lr.push_back(value(n));
                jobj al, cj;
                for (auto& [a, b] : m.spec.alias) if (a.starts_with(pre)) al.push_back({a, value(b)});
                for (auto& [a, b] : m.codec) if (a.starts_with(pre)) cj.push_back({a, value(b)});
                for (auto& [n, t] : m.w.t) if (n.starts_with(pre)) {
                    es.push_back({n, "f32", t.shape, t.data.data(), t.data.size() * 4});
                    tn.push_back(value(n));
                }
                lm.push_back({std::to_string(l), value(jobj{
                    {"attn", value(L.attn)}, {"ffn", value(int(L.ffn))}, {"moe_k", value(L.moe_k)}, {"moe_norm", value(L.moe_norm)},
                    {"moe_scale", exact_num(L.moe_scale)}, {"act_thresh", exact_num(L.act_thresh)},
                    {"heads", m.spec.heads.count(l) ? value(m.spec.heads.at(l)) : value()}, {"lowrank", value(lr)}, {"alias", value(al)},
                    {"codec", value(cj)}, {"tensors", value(tn)}})});
            }
            const std::string f = std::format("L_{}_{}_{}.bin", tag, ls.front(), ls.back());
            write_bundle(path_(f), value(jobj{{"layers", value(lm)}}), es);
            chunks.push_back(value(jobj{{"file", value(f)}, {"from", value(ls.front())}, {"to", value(ls.back())}}));
        }
        const std::string hf = std::format("H_{}_{}.bin", tag, e.layer);
        write_hidden_(path_(hf), *e.h);
        const std::string old_h = ph.find("hidden") ? ph["hidden"].str() : std::string();
        value nph(jobj{{"done", value(e.layer)}, {"carry", *e.carry}, {"hidden", value(hf)}, {"chunks", value(chunks)}});
        if (!progress_.is_obj()) progress_ = value(jobj{{"key", value(cfg_.key)}, {"stage", value(stage_)}, {"phases", value(jobj{})}});
        json_set(progress_, "stage", value(stage_));
        auto& phases = progress_.obj_mut();
        for (auto& [kk, vv] : phases) if (kk == "phases") json_set(vv, k, nph);
        if (auto s = os::fs::write_file_atomic(path_("progress.json"), forge::json::dump(progress_)); !s) die("checkpoint: " + s.err.what);
        if (!old_h.empty() && old_h != hf) remove_later_(path_(old_h));
        lv.first_unsaved = e.layer + 1;
        lv.modified.clear();
        last_save_ = now_s();
        say("checkpoint: {} after layer {} ({} layer(s) saved)", k, e.layer, ls.size());
    }
    void restore_layer_(Model& m, const value& ph, i64 l) {
        for (auto& c : ph["chunks"].arr()) {
            if (l < c["from"].i() || l > c["to"].i()) continue;
            const std::string f = path_(c["file"].str());
            auto it = bundles_.find(f);
            if (it == bundles_.end()) it = bundles_.emplace(f, bundle::open(f)).first;
            const bundle& b = *it->second;
            auto* lm = b.meta()["layers"].find(std::to_string(l));
            if (!lm) continue;
            const std::string pre = lprefix_(l);
            auto& L = m.spec.layers[std::size_t(l)];
            L.attn = (*lm)["attn"].b(); L.ffn = LayerSpec::Ffn((*lm)["ffn"].i()); L.moe_k = (*lm)["moe_k"].i(); L.moe_norm = (*lm)["moe_norm"].b();
            L.moe_scale = exact_of((*lm)["moe_scale"]); L.act_thresh = exact_of((*lm)["act_thresh"]);
            if ((*lm)["heads"].is_num()) m.spec.heads[l] = (*lm)["heads"].i(); else m.spec.heads.erase(l);
            std::erase_if(m.spec.lowrank, [&](const std::string& n) { return n.starts_with(pre); });
            for (auto& n : (*lm)["lowrank"].arr()) m.spec.lowrank.insert(n.str());
            std::erase_if(m.spec.alias, [&](const auto& kv) { return kv.first.starts_with(pre); });
            for (auto& [a, v] : (*lm)["alias"].obj()) m.spec.alias[a] = v.str();
            std::erase_if(m.codec, [&](const auto& kv) { return kv.first.starts_with(pre); });
            for (auto& [a, v] : (*lm)["codec"].obj()) m.codec[a] = v.str();
            std::erase_if(m.w.t, [&](const auto& kv) { return kv.first.starts_with(pre); });
            for (auto& n : (*lm)["tensors"].arr()) {
                auto* e = b.entry(n.str());
                if (!e) die("checkpoint: " + f + " lacks " + n.str());
                wtensor t; t.shape = b.shape(*e);
                auto by = b.bytes(*e);
                t.data.resize(by.size() / 4);
                std::memcpy(t.data.data(), by.data(), by.size());
                m.w.t[n.str()] = std::move(t);
            }
        }
    }
    void drop_progress_() {
        if (progress_.is_obj()) {
            if (auto* ps = progress_.find("phases")) for (auto& [k, p] : ps->obj()) {
                if (p.find("hidden")) remove_later_(path_(p["hidden"].str()));
                for (auto& c : p["chunks"].arr()) remove_later_(path_(c["file"].str()));
            }
        }
        bundles_.clear();
        progress_ = value();
        (void)os::remove(path_("progress.json"));
        // files of an older progress state that no longer has an index (a crash between writes)
        if (auto l = os::list_dir(cfg_.dir)) for (auto& e : l.value) if ((e.name.starts_with("H_") || e.name.starts_with("L_")) && e.name.ends_with(".bin")) remove_later_(path_(e.name));
    }

    // ---- aborts (tests / supervisor drills)
    static bool abort_match_(const std::string& phase, i64 l) {
        const std::string a = os::getenv("FORGE_COMPRESS_ABORT_AFTER", "");
        if (a.empty()) return false;
        const auto c = a.rfind(':');
        const std::string p = c == std::string::npos ? std::string() : a.substr(0, c), n = c == std::string::npos ? a : a.substr(c + 1);
        if (p.empty() && phase == "stage") return false;   // a bare layer number means a layer loop
        if (!p.empty() && p != phase) return false;
        char* end = nullptr; const long long v = std::strtoll(n.c_str(), &end, 10);
        return end != n.c_str() && *end == 0 && v == l;
    }
    void abort_if_(const std::string& phase, i64 l, bool matched = false) const {
        if (!matched && !abort_match_(phase, l)) return;
        if (os::getenv("FORGE_COMPRESS_ABORT_OOM", "") == "1") throw std::bad_alloc();
        throw std::runtime_error(std::format("aborted after {} {} (FORGE_COMPRESS_ABORT_AFTER)", phase, l));
    }

    CheckpointCfg cfg_;
    std::optional<os::fs::file_lock> lock_;
    value state_, progress_;
    bool resumed_ = false;
    i64 stage_ = -1;
    double last_save_ = 0;
    std::map<std::string, int> occ_;
    std::map<std::string, std::string> active_;
    std::map<std::string, Live> live_;
    std::map<std::string, std::shared_ptr<bundle>> bundles_;
    std::vector<std::string> pending_;
};

// The job identity of `forge compress` options (see the file comment).
inline std::string checkpoint_key(const std::string& in, const std::map<std::string, std::string>& kv) {
    auto get = [&](const char* k) { auto it = kv.find(k); return it == kv.end() ? std::string() : it->second; };
    auto st = os::stat(in);
    std::string s = std::format("forge-compress-ckpt-v1|{}|{}|{}", os::absolute(in), st.size, st.mtime_ns);
    for (const char* k : {"recipe", "nsamples", "seqlen", "calib-data", "seed", "damp", "probes", "probe-len"}) s += std::format("|{}={}", k, get(k));
    if (auto cd = get("calib-data"); !cd.empty() && cd != "self") { auto c = os::stat(cd); s += std::format("|calib:{}:{}", c.size, c.mtime_ns); }
    return forge::hash::blake3::hex(forge::hash::blake3::of({reinterpret_cast<const std::uint8_t*>(s.data()), s.size()}));
}

}  // namespace forge::compress

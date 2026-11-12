#pragma once
// Precompiled plans for a model (docs/aot-models.md): where a model's plan blob comes from and how plan_for() uses it.
//
// Sources, first match wins:
//   1. a natively compiled model library (aot_native.hpp) carries the blob it was generated from;
//   2. the "aot" section of the .fpk. On a signed file it is used only if the signature covers it;
//   3. the sidecar <model>.fplan (or $FORGE_AOT_PLAN). Never used for a signed file or under strict loading, since
//      nothing signs it.
// FORGE_AOT=0 turns precompiled plans off. Any mismatch (other model, other FORGE version, target not precompiled,
// a blob that fails a check, a lowering that does not pass the oracle here) is logged at debug level and the plan
// is compiled the usual way.
#include <forge/core/log.hpp>
#include <forge/graph/aot.hpp>
#include <forge/modelfile/aot.hpp>
#include <forge/modelfile/blueprint.hpp>
#include <forge/inference/aot_native.hpp>
#include <fstream>

namespace forge::inference {

class aot_source {
public:
    // Counters for tests and `forge run`'s summary line.
    struct stats_t { std::size_t used = 0, rejected = 0, native = 0; std::string origin, last_reject; };

    static bool env_enabled() { const char* e = std::getenv("FORGE_AOT"); return !(e && *e == '0'); }

    // Look for a plan blob for the model in `r` (opened from `path`). `strict`: signed loading is required.
    void attach(const modelfile::reader& r, const std::string& path, bool strict, std::shared_ptr<aot_native::library> lib = nullptr) {
        img_.reset(); owned_.clear(); stats_ = {}; lib_ = std::move(lib);
        if (!enabled_) return;
        try {
            key_ = modelfile::aot::model_key(r);
            const bool is_signed = r.has_section("signature");
            if (lib_) {
                open_({lib_->model().plans, std::size_t(lib_->model().plans_size)}, "native library " + lib_->path());
                if (img_ && img_->model() != key_) { reject_("the library was compiled for another model or FORGE version"); img_.reset(); lib_.reset(); }
                return;
            }
            if (r.has_section(modelfile::aot::section)) {
                std::string why;
                if (is_signed && !modelfile::aot::covered_by_signature(r, &why)) reject_("aot section ignored: " + why);
                else open_(r.raw_section_bytes(modelfile::aot::section), "section");
                return;
            }
            std::string side = path + modelfile::aot::sidecar_ext;
            if (const char* e = std::getenv("FORGE_AOT_PLAN"); e && *e) side = e;
            if (!forge::os::exists(side)) return;
            if (is_signed || strict) { reject_("sidecar " + side + " ignored: nothing signs it"); return; }
            std::ifstream f(side, std::ios::binary);
            owned_.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            open_(owned_, "sidecar " + side);
        } catch (const std::exception& e) { reject_(e.what()); img_.reset(); }
    }
    // The precompiled plan for this request, or null (the caller compiles). `bo` carries the weight resolvers.
    std::unique_ptr<graph::plan> try_plan(i64 n, bool all_rows, graph::state_store* st, const modelfile::build_options& bo,
                                          const graph::pass_options& po, const std::string& adapter) {
        if (!img_ || !enabled_ || po.lowerings || !adapter.empty()) return nullptr;
        graph::aot::target t;
        t.seq = n; t.all_rows = all_rows;
        t.kv_codec = st ? graph::kv_codec_pair_name({st->codec, st->codec_v == graph::kv_codec::f32 ? st->codec : st->codec_v}) : "f32";
        auto* e = img_->find(key_, t);
        if (!e) { if (img_->model() != key_) reject_("plan blob is for another model or FORGE version"); return nullptr; }
        try {
            graph::aot::binder b;
            b.weight = bo.weight;
            if (bo.qweight) b.qweight = [&](const std::string& name) { auto q = bo.qweight(name); return graph::aot::binder::encoded{std::move(q.shape), q.data, std::move(q.codec)}; };
            auto p = graph::aot::instantiate(*img_, *e, b, st, po);
            ++stats_.used;
            if (lib_ && lib_->attach(*p, std::size_t(e - img_->entries().data()))) ++stats_.native;
            return p;
        } catch (const std::exception& ex) { reject_(std::format("{}: {}", t.name(), ex.what())); return nullptr; }
    }
    bool available() const { return bool(img_); }
    bool enabled() const { return enabled_; }
    void set_enabled(bool on) { enabled_ = on; }
    const stats_t& stats() const { return stats_; }
    const graph::aot::image* image() const { return img_ ? &*img_ : nullptr; }
    const std::shared_ptr<aot_native::library>& native() const { return lib_; }
private:
    void open_(std::span<const std::uint8_t> blob, std::string origin) {
        img_.emplace(blob);
        stats_.origin = std::move(origin);
        log::debug("aot: {} precompiled plan(s) from {}", img_->entries().size(), stats_.origin);
    }
    void reject_(std::string why) { ++stats_.rejected; stats_.last_reject = std::move(why); log::debug("aot: {}; compiling instead", stats_.last_reject); }
    bool enabled_ = env_enabled();
    std::optional<graph::aot::image> img_;
    std::vector<std::uint8_t> owned_;
    graph::aot::digest key_{};
    std::shared_ptr<aot_native::library> lib_;
    stats_t stats_;
};

// ---- build side: compile plans for a model ahead of time ----

struct aot_build_options {
    std::vector<i64> shapes;                 // sequence lengths (empty: default_shapes())
    bool all_rows = false;                   // also precompile the all-rows variant of every shape
    std::vector<std::string> kv_codecs;      // KV codec names (empty: f32)
    i64 context = 0;                         // 0 = the manifest's
};
// Decode (1) and prefill lengths that are powers of two up to 64: a longer or odd prompt compiles its plan at run time.
inline std::vector<i64> aot_default_shapes() { return {1, 2, 4, 8, 16, 32, 64}; }

} // namespace forge::inference

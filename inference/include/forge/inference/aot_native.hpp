#pragma once
// Engine side of natively compiled models (forge_aot.h, docs/aot-models.md): load a compiled model library, find the
// model file it belongs to, and attach its code to plans.
//
// The plan object, its arena and the session state stay the engine's. A compiled library only replaces the
// interpretation of kernel steps: plan::native runs the straight-line code for each run of consecutive kernel steps,
// and the executor still performs the state steps between them.
#include <forge_aot.h>
#include <forge/core/log.hpp>
#include <forge/graph/plan.hpp>
#include <forge/os/os.hpp>
#include <fstream>
#include <memory>

namespace forge::inference::aot_native {

class library : public std::enable_shared_from_this<library> {
public:
    // True when `path` names a shared library (by its first bytes: ELF, Mach-O or PE), not a .fpk.
    static bool is_library(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        unsigned char m[4] = {};
        if (!f.read(reinterpret_cast<char*>(m), 4)) return false;
        std::uint32_t le = std::uint32_t(m[0]) | std::uint32_t(m[1]) << 8 | std::uint32_t(m[2]) << 16 | std::uint32_t(m[3]) << 24;
        return (m[0] == 0x7f && m[1] == 'E' && m[2] == 'L' && m[3] == 'F') || (m[0] == 'M' && m[1] == 'Z') ||
               le == 0xfeedfaceu || le == 0xfeedfacfu || le == 0xcefaedfeu || le == 0xcffaedfeu || le == 0xbebafecau || le == 0xbfbafecau;
    }
    // Load a compiled model library. Raises when it is not one, has another ABI version, or cannot run on this CPU.
    static std::shared_ptr<library> open(const std::string& path) {
        auto l = forge::os::library::open(forge::os::absolute(path));
        if (!l) raise(errc::io_error, "cannot load {}: {}", path, l.err.what);
        using get_fn = const forge_aot_model_v1* (*)(void);
        auto get = reinterpret_cast<get_fn>(l.value.symbol(FORGE_AOT_SYMBOL));
        if (!get) raise(errc::format_error, "{}: not a compiled FORGE model (no {} symbol)", path, FORGE_AOT_SYMBOL);
        // The library stays loaded for the life of the process: its code owns threads (its copy of the thread pool)
        // and thread-local state that must not outlive an unmapped image.
        (void)new forge::os::library(std::move(l.value));
        return std::shared_ptr<library>(new library(get(), forge::os::absolute(path)));
    }
    // Code linked into this program (a static library or the standalone executable): no loading.
    static std::shared_ptr<library> in_image(const forge_aot_model_v1* d, std::string label = "<this program>") {
        return std::shared_ptr<library>(new library(d, std::move(label)));
    }
    const forge_aot_model_v1& model() const { return *d_; }
    const std::string& path() const { return path_; }
    bool embedded() const { return d_->fpk_data && d_->fpk_size; }
    // The model file: $FORGE_AOT_FPK, else a file of the recorded name next to the library, else the recorded path.
    std::string fpk_path() const {
        if (const char* e = std::getenv("FORGE_AOT_FPK"); e && *e) return e;
        std::string rec = d_->fpk_path ? d_->fpk_path : "";
        if (!rec.empty() && path_.front() != '<') {
            auto beside = forge::os::dirname(path_) + "/" + forge::os::basename(rec);
            if (forge::os::exists(beside)) return beside;   // not "near": <windows.h> defines near and far
        }
        if (rec.empty()) raise(errc::not_found, "{}: the library does not name its model file; set FORGE_AOT_FPK", path_);
        return rec;
    }
    // Attach the compiled code for blob entry `entry` to `p` (a plan instantiated from that entry). False when the
    // library has no code for it or the plan does not have the layout the code was generated for.
    bool attach(graph::plan& p, std::size_t entry) {
        if (entry >= d_->n_plans) return false;
        const auto& pl = d_->plan[entry];
        if (!pl.bind || !pl.run || !pl.n_segments || p.hooks) return false;
        if (pl.n_values != p.g.size() || pl.n_steps != p.steps.size() || pl.arena_floats != p.arena.size()) {
            log::debug("aot: compiled code for plan {} does not fit ({} values, {} steps here)", entry, p.g.size(), p.steps.size());
            return false;
        }
        struct bound {
            std::shared_ptr<library> lib; const forge_aot_plan_v1* pl; void* inst = nullptr;
            std::vector<forge_aot_value> vals; std::vector<graph::value_id> dyn; std::vector<int> seg_at;
            ~bound() { if (inst && pl->unbind) pl->unbind(inst); }
        };
        auto b = std::make_shared<bound>();
        b->lib = shared_from_this(); b->pl = &pl;
        b->vals.resize(p.g.size());
        auto fill = [](forge_aot_value& v, const tensor_view& t) { v.data = t.data; v.qdata = t.qdata; v.rows = t.shape.empty() ? 1 : t.shape[0]; };
        for (std::size_t i = 0; i < p.g.size(); ++i) {
            fill(b->vals[i], p.views[i]);
            if (p.g.nodes()[i].kind == graph::node_kind::state_append) b->dyn.push_back(graph::value_id(i));
        }
        for (auto [a, r] : p.state_aliases) b->dyn.push_back(a);
        b->seg_at.assign(p.steps.size(), -1);
        for (std::uint32_t s = 0; s < pl.n_segments; ++s) {
            std::uint32_t first = pl.segment_first[s], n = pl.segment_steps[s];
            if (!n || first >= p.steps.size() || n > p.steps.size() - first) return false;
            for (std::uint32_t k = first; k < first + n; ++k) if (p.steps[k].kind != 0) return false;   // state steps are the engine's
            b->seg_at[first] = int(s);
        }
        char err[256] = {};
        b->inst = pl.bind(b->vals.data(), p.arena.data(), err, sizeof err);
        if (!b->inst) { log::debug("aot: compiled code for plan {} did not bind: {}", entry, err); return false; }
        graph::plan* pp = &p;
        p.native = [b, pp, fill](std::size_t si) -> std::size_t {
            int s = b->seg_at[si];
            if (s < 0) return 0;
            for (auto v : b->dyn) fill(b->vals[v], pp->views[v]);
            char e[256] = {};
            if (b->pl->run(b->inst, std::uint32_t(s), b->vals.data(), e, sizeof e)) raise(errc::invalid_argument, "{}", e);
            return b->pl->segment_steps[s];
        };
        return true;
    }
private:
    library(const forge_aot_model_v1* d, std::string path) : d_(d), path_(std::move(path)) {
        if (!d_ || d_->abi_version != FORGE_AOT_ABI_VERSION || d_->struct_size < sizeof(forge_aot_model_v1))
            raise(errc::unsupported, "{}: compiled model ABI {} (this build reads {})", path_, d_ ? d_->abi_version : 0, FORGE_AOT_ABI_VERSION);
        if (!d_->plans || d_->plans_size < 64) raise(errc::format_error, "{}: compiled model without a plan blob", path_);
        char err[256] = {};
        if (d_->check && d_->check(err, sizeof err)) raise(errc::unsupported, "{}: {}", path_, err);
    }
    const forge_aot_model_v1* d_;
    std::string path_;
};

} // namespace forge::inference::aot_native

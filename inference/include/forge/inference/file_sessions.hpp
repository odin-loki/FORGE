#pragma once
// Session kinds chosen by the model file (FORGE_PLAN E.3): a blueprint that declares `session diffusion_loop(...)` or
// `session streaming(...)` (opset 4) is opened like any other model (inference::model / inference::session), and the
// runners below attach the service the kind names to the session's graph:
//
//   diffusion_loop   noise scheduler + step plan + classifier-free guidance + latent cache (diffusion.hpp), with the
//                    blueprint's main graph as the denoiser
//   streaming        ring buffer + frame scheduler + latency budget (streaming.hpp), with the main graph run once per
//                    frame; whatever state the graph declares (conv windows, recurrent state, appended rows) is the
//                    session's state store, carried from frame to frame
//
//   blueprint "tiny-denoiser" opset 4
//   input x : f32[n]   input t : f32[1]   input cond : f32[1]
//   graph { ... eps = ... }
//   output eps : f32[n]
//   session diffusion_loop(steps: 1000, schedule: scaled_linear, beta_start: 0.00085, beta_end: 0.012,
//                          prediction: epsilon, sampler: ddim, inference_steps: 20, guidance: 1, eta: 0,
//                          latent: x, timestep: t, encoder: encode)
//
//   session streaming(window: 400, hop: 160, max_lag: 8, capacity: 65536, input: frame)
//
// Every parameter has a default (those of diffusion::options / streaming::options and of noise_scheduler::make);
// `latent`, `timestep` and `input` default to the first input, an input named `t`, and the first input.
// `encoder` names an entry point (`entry encode { ... }`) that is run once before the loop: its outputs are fed to
// the denoiser inputs of the same names (a text or class encoder whose result conditions every step).
#include <forge/inference/diffusion.hpp>
#include <forge/inference/session.hpp>
#include <forge/inference/streaming.hpp>

namespace forge::inference {

namespace detail {
// number of elements of a declared input or output with the symbolic dimensions bound
inline std::size_t decl_size(const modelfile::blueprint& bp, const modelfile::bp_io_decl& d, const std::map<std::string, i64>& dims) {
    std::uint64_t n = 1;
    for (auto& x : d.dims) {
        i64 v;
        if (auto it = dims.find(x); it != dims.end()) v = it->second;
        else if (auto pt = bp.params.find(x); pt != bp.params.end()) v = i64(pt->second);
        else if (!x.empty() && std::isdigit(static_cast<unsigned char>(x[0]))) v = i64(std::strtoll(x.c_str(), nullptr, 10));
        else raise(errc::invalid_argument, "input '{}' has the unbound dimension '{}'", d.name, x);
        if (v < 0 || (v && n > (std::uint64_t(1) << 40) / std::uint64_t(v))) raise(errc::invalid_argument, "input '{}' has a bad size", d.name);
        n *= std::uint64_t(v);
    }
    return std::size_t(n);
}
} // namespace detail

using tensor_map = std::map<std::string, std::vector<float>>;

// ---------------------------------------------------------------------------------------------- diffusion_loop
class diffusion_runner {
public:
    // `dims` binds symbolic dimensions of the denoiser's inputs (a latent declared f32[seq, c] needs seq).
    explicit diffusion_runner(session& s, std::map<std::string, i64> dims = {}) : s_(s), dims_(std::move(dims)) {
        const auto& bp = s.blueprint();
        if (bp.session != "diffusion_loop") raise(errc::unsupported, "session kind '{}' is not a diffusion model (the blueprint must declare `session diffusion_loop`)", bp.session);
        if (bp.inputs.empty() || bp.outputs.empty()) raise(errc::format_error, "diffusion_loop: the denoiser graph needs inputs and an output");
        latent_ = bp.session_str("latent", bp.inputs.front().name);
        timestep_ = bp.session_str("timestep", "t");
        auto find = [&](const std::string& n) -> const modelfile::bp_io_decl* { for (auto& d : bp.inputs) if (d.name == n) return &d; return nullptr; };
        const auto* ld = find(latent_);
        if (!ld) raise(errc::format_error, "diffusion_loop: the blueprint has no input '{}' (the latent)", latent_);
        has_t_ = find(timestep_) != nullptr;
        n_ = detail::decl_size(bp, *ld, dims_);
        if (detail::decl_size(bp, bp.outputs.front(), dims_) != n_) raise(errc::format_error, "diffusion_loop: the denoiser output '{}' is not the size of the latent '{}'", bp.outputs.front().name, latent_);
        // scheduler
        double T = bp.session_arg("steps", 1000);
        if (!(T >= 1 && T <= 100000) || T != std::floor(T)) raise(errc::format_error, "diffusion_loop: bad steps {}", T);
        auto sch = bp.session_str("schedule", "scaled_linear");
        diffusion::beta_schedule bs = sch == "linear" ? diffusion::beta_schedule::linear : sch == "cosine" ? diffusion::beta_schedule::cosine : diffusion::beta_schedule::scaled_linear;
        if (sch != "linear" && sch != "cosine" && sch != "scaled_linear") raise(errc::unsupported, "diffusion_loop: unknown schedule '{}' (linear, scaled_linear, cosine)", sch);
        double b0 = bp.session_arg("beta_start", 0.00085), b1 = bp.session_arg("beta_end", 0.012);
        if (!(b0 > 0 && b1 >= b0 && b1 < 1)) raise(errc::format_error, "diffusion_loop: bad beta range {} .. {}", b0, b1);
        sched_ = diffusion::noise_scheduler::make(int(T), bs, b0, b1);
        // defaults of a run
        auto pred = bp.session_str("prediction", "epsilon"), smp = bp.session_str("sampler", "ddim");
        if (pred != "epsilon" && pred != "v" && pred != "sample") raise(errc::unsupported, "diffusion_loop: unknown prediction '{}' (epsilon, v, sample)", pred);
        if (smp != "ddim" && smp != "ddpm" && smp != "euler_flow") raise(errc::unsupported, "diffusion_loop: unknown sampler '{}' (ddim, ddpm, euler_flow)", smp);
        opt_.pred = pred == "v" ? diffusion::prediction::v : pred == "sample" ? diffusion::prediction::sample : diffusion::prediction::epsilon;
        opt_.kind = smp == "ddpm" ? diffusion::sampler_kind::ddpm : smp == "euler_flow" ? diffusion::sampler_kind::euler_flow : diffusion::sampler_kind::ddim;
        opt_.steps = int(std::clamp(bp.session_arg("inference_steps", 20), 1.0, T));
        opt_.guidance = float(bp.session_arg("guidance", 1)); opt_.eta = bp.session_arg("eta", 0);
        encoder_ = bp.session_str("encoder", "");
        if (!encoder_.empty()) (void)bp.entry_at(encoder_);
    }
    const diffusion::noise_scheduler& scheduler() const { return sched_; }
    diffusion::options defaults() const { return opt_; }      // what the file asks for; change a copy and pass it to run()
    std::size_t latent_size() const { return n_; }
    const std::string& latent_input() const { return latent_; }
    const std::string& encoder_entry() const { return encoder_; }
    std::size_t model_calls() const { return calls_; }
    const std::map<int, diffusion::latent>& cache() const { return cache_; }

    diffusion::latent initial_noise(std::uint64_t seed) const { return loop_().initial_noise(n_, seed); }

    // Denoises `x` (pure noise for start_step 0). `cond` holds the model's other inputs for the conditional branch
    // and `uncond` for the unconditional one (classifier-free guidance, guidance != 1); an input missing from a map
    // is fed zeros. With an encoder entry, the maps hold the encoder's inputs instead and its outputs are used.
    diffusion::latent run(diffusion::latent x, const diffusion::options& o, const tensor_map& cond = {}, const tensor_map& uncond = {}) {
        if (x.size() != n_) raise(errc::shape_mismatch, "diffusion: the latent has {} values, the model's '{}' input has {}", x.size(), latent_, n_);
        const auto& bp = s_.blueprint();
        tensor_map branch[2] = {prepare_(uncond, o.guidance != 1.f), prepare_(cond, true)};   // [0] unconditional, [1] conditional
        auto den = [&](const diffusion::latent& xt, double t, int c) -> diffusion::latent {
            const tensor_map& given = branch[c ? 1 : 0];
            tensor_map in;   // exactly the declared inputs: the latent, the timestep, conditioning by name, zeros for the rest
            for (auto& d : bp.inputs) {
                if (d.name == latent_) in[d.name] = xt;
                else if (has_t_ && d.name == timestep_) in[d.name].assign(detail::decl_size(bp, d, dims_), float(t));
                else if (auto it = given.find(d.name); it != given.end()) in[d.name] = it->second;
                else in[d.name].assign(detail::decl_size(bp, d, dims_), 0.f);
            }
            auto out = s_.infer(in, seq_(), dims_);
            return std::move(out.at(bp.outputs.front().name));
        };
        diffusion::session loop(sched_, den);
        auto r = loop.run(std::move(x), o);
        calls_ += loop.model_calls(); cache_ = loop.cache();
        return r;
    }
    // From noise drawn with o.seed.
    diffusion::latent sample(const diffusion::options& o, const tensor_map& cond = {}, const tensor_map& uncond = {}) { return run(initial_noise(o.seed), o, cond, uncond); }

private:
    i64 seq_() const { auto it = dims_.find("seq"); return it == dims_.end() ? 1 : it->second; }
    diffusion::session loop_() const { return diffusion::session(sched_, [](const diffusion::latent& x, double, int) { return x; }); }
    // the denoiser's conditioning inputs for one branch: given directly, or computed once by the encoder entry
    tensor_map prepare_(const tensor_map& given, bool needed) {
        if (encoder_.empty() || !needed) return given;
        const auto& e = s_.blueprint().entry_at(encoder_);
        tensor_map in = given;
        for (auto& d : e.inputs) if (!in.count(d.name)) in[d.name].assign(detail::decl_size(s_.blueprint(), d, dims_), 0.f);
        return s_.infer_entry(encoder_, in, seq_(), dims_);
    }
    session& s_; std::map<std::string, i64> dims_;
    std::string latent_, timestep_, encoder_; bool has_t_ = false; std::size_t n_ = 0;
    diffusion::noise_scheduler sched_; diffusion::options opt_;
    std::size_t calls_ = 0; std::map<int, diffusion::latent> cache_;
};

// ---------------------------------------------------------------------------------------------- streaming
class stream_runner {
public:
    explicit stream_runner(session& s) : stream_runner(s, options_of(s.blueprint())) {}
    // `o` replaces the file's framing (window must still be the size of the model's frame input).
    stream_runner(session& s, streaming::options o) : s_(s), in_(input_of(s.blueprint())), o_(o),
        frames_(o, [this](const float* f, std::size_t n, std::size_t index) { frame_(f, n, index); }) {
        const auto& bp = s.blueprint();
        const modelfile::bp_io_decl* fd = nullptr; for (auto& d : bp.inputs) if (d.name == in_) fd = &d;
        if (!fd) raise(errc::format_error, "streaming: the blueprint has no input '{}' (the frame)", in_);
        if (detail::decl_size(bp, *fd, {{"seq", 1}}) != o.window) raise(errc::shape_mismatch, "streaming: the frame input '{}' takes {} samples, the window is {}", in_, detail::decl_size(bp, *fd, {{"seq", 1}}), o.window);
        if (bp.outputs.empty()) raise(errc::format_error, "streaming: the graph has no output");
    }
    static streaming::options options_of(const modelfile::blueprint& bp) {
        if (bp.session != "streaming") raise(errc::unsupported, "session kind '{}' is not a streaming model (the blueprint must declare `session streaming`)", bp.session);
        streaming::options o;
        auto sz = [&](const char* k, std::size_t d, double lo, double hi) { double v = bp.session_arg(k, double(d)); if (!(v >= lo && v <= hi) || v != std::floor(v)) raise(errc::format_error, "streaming: bad {} {}", k, v); return std::size_t(v); };
        o.window = sz("window", o.window, 1, 1 << 24); o.hop = sz("hop", o.hop, 1, 1 << 24);
        o.max_lag_frames = sz("max_lag", o.max_lag_frames, 1, 1 << 20); o.capacity = sz("capacity", std::max(o.capacity, 4 * o.window), double(o.window), 1 << 28);
        o.deadline = std::chrono::microseconds(std::int64_t(sz("deadline_us", 0, 0, 1e12)));
        return o;
    }
    const streaming::options& options() const { return o_; }
    // Pushes samples (any chunk size) and processes every frame that is ready. Returns the frames processed.
    // Samples beyond the ring's capacity are rejected and counted (stat().samples_rejected): feed in chunks of at
    // most options().capacity.
    std::size_t feed(std::span<const float> samples) {
        std::size_t done = 0, at = 0;
        while (at < samples.size()) {   // push what fits, drain, repeat: a long buffer never overflows the ring
            std::size_t k = frames_.feed(samples.data() + at, std::min(samples.size() - at, o_.capacity / 2 + 1));
            done += frames_.pump();
            if (!k) break;
            at += k;
        }
        return done;
    }
    // The first output of every processed frame, concatenated in frame order; frame_size() values per frame.
    const std::vector<float>& output() const { return out_; }
    std::size_t frame_size() const { return per_frame_; }
    // Indices of the frames in output() (frames dropped by the latency budget are missing from the sequence).
    const std::vector<std::size_t>& frame_indices() const { return idx_; }
    const streaming::stats& stat() const { return frames_.stat(); }
    std::vector<float> take_output() { idx_.clear(); return std::exchange(out_, {}); }

private:
    static std::string input_of(const modelfile::blueprint& bp) {
        if (bp.inputs.empty()) raise(errc::format_error, "streaming: the graph has no input");
        return bp.session_str("input", bp.inputs.front().name);
    }
    void frame_(const float* f, std::size_t n, std::size_t index) {
        const auto& bp = s_.blueprint();
        tensor_map in; in[in_].assign(f, f + n);
        for (auto& d : bp.inputs) if (!in.count(d.name)) in[d.name].assign(detail::decl_size(bp, d, {{"seq", 1}}), 0.f);
        auto out = s_.infer(in, 1);
        auto& y = out.at(bp.outputs.front().name);
        per_frame_ = y.size(); out_.insert(out_.end(), y.begin(), y.end()); idx_.push_back(index);
    }
    session& s_; std::string in_; streaming::options o_;
    std::vector<float> out_; std::vector<std::size_t> idx_; std::size_t per_frame_ = 0;
    streaming::session frames_;
};

} // namespace forge::inference

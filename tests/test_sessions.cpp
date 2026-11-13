#include "check.hpp"
#include <forge/inference/diffusion.hpp>
#include <forge/inference/streaming.hpp>
#include <forge/inference/inference.hpp>
#include <forge/modelfile/blueprint_bin.hpp>
#include "tiny_model.hpp"
#include <filesystem>
#include <random>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#include <cmath>
#include <numeric>

using namespace forge::inference;

// Oracle denoiser for data concentrated at mu_c (cond) / mu_u (uncond): the exact epsilon is (x - sqrt(ab) mu)/sqrt(1-ab).
static diffusion::denoiser oracle(const diffusion::noise_scheduler& s, float mu_c, float mu_u, diffusion::prediction p) {
    return [&s, mu_c, mu_u, p](const diffusion::latent& x, double t, int cond) {
        double ab = s.alphas_cumprod[std::size_t(t)]; float mu = cond ? mu_c : mu_u;
        diffusion::latent o(x.size());
        for (std::size_t k = 0; k < x.size(); ++k) {
            double e = (x[k] - std::sqrt(ab) * mu) / std::sqrt(1 - ab);
            o[k] = float(p == diffusion::prediction::epsilon ? e : p == diffusion::prediction::sample ? mu : std::sqrt(ab) * e - std::sqrt(1 - ab) * mu);
        }
        return o;
    };
}
static double max_err(const diffusion::latent& x, float v) { double m = 0; for (float a : x) m = std::max(m, double(std::fabs(a - v))); return m; }

TEST(diffusion_ddim_converges_all_predictions) {
    auto s = diffusion::noise_scheduler::make();
    for (auto p : {diffusion::prediction::epsilon, diffusion::prediction::v, diffusion::prediction::sample}) {
        diffusion::session d(s, oracle(s, 0.7f, 0.f, p));
        diffusion::options o; o.steps = 10; o.pred = p;
        auto x = d.run(d.initial_noise(257, 42), o);
        CHECK_MSG(max_err(x, 0.7f) < 1e-3, "pred %d err %g", int(p), max_err(x, 0.7f));
        CHECK(d.model_calls() == 10);
    }
}
TEST(diffusion_guidance_and_cache_resume) {
    auto s = diffusion::noise_scheduler::make(1000, diffusion::beta_schedule::cosine);
    diffusion::session d(s, oracle(s, 1.f, -1.f, diffusion::prediction::epsilon));
    diffusion::options o; o.steps = 25; o.guidance = 3.f; o.cache_at = {10};
    auto n = d.initial_noise(64, 7);
    auto x = d.run(n, o);
    CHECK(max_err(x, -1.f + 3.f * 2.f) < 1e-2);           // CFG is linear in mu: mu_u + w (mu_c - mu_u) = 5
    CHECK(d.model_calls() == 50);
    CHECK(d.cache().count(10) == 1);
    auto o2 = o; o2.start_step = 10; o2.cache_at = {};
    auto lat = d.cache().at(10);
    auto y = d.run(lat, o2);
    CHECK(y == x);                                         // resuming from the cached latent is bit-identical
    CHECK(d.initial_noise(64, 7) == n && d.initial_noise(64, 8) != n);
    double mean = std::accumulate(n.begin(), n.end(), 0.0) / 64; CHECK(std::fabs(mean) < 0.5);
}
TEST(diffusion_ddpm_stochastic_but_seeded) {
    auto s = diffusion::noise_scheduler::make(1000, diffusion::beta_schedule::linear, 1e-4, 0.02);
    diffusion::session d(s, oracle(s, 0.25f, 0.f, diffusion::prediction::epsilon));
    diffusion::options o; o.steps = 50; o.kind = diffusion::sampler_kind::ddpm; o.seed = 3;
    auto n = d.initial_noise(128, 1);
    auto a = d.run(n, o), b = d.run(n, o);
    CHECK(a == b);
    CHECK(max_err(a, 0.25f) < 1e-2);                       // oracle data has zero variance: the posterior collapses
}
TEST(diffusion_euler_flow) {
    // Flow matching: x_t = (1-t) x0 + t noise, v = noise - x0. Oracle knows x0 = 2, so v = (x - 2)/t... use exact form.
    diffusion::session d(diffusion::noise_scheduler::make(10), [](const diffusion::latent& x, double t, int) {
        double tt = t / 1000; diffusion::latent v(x.size()); for (std::size_t k = 0; k < x.size(); ++k) v[k] = float((x[k] - 2.0) / tt); return v; });
    diffusion::options o; o.steps = 8; o.kind = diffusion::sampler_kind::euler_flow;
    auto x = d.run(d.initial_noise(32, 9), o);
    CHECK(max_err(x, 2.f) < 1e-4);
}
TEST(streaming_framing_matches_offline) {
    std::vector<float> sig(5003); for (std::size_t i = 0; i < sig.size(); ++i) sig[i] = float(i);
    streaming::options o; o.window = 64; o.hop = 24; o.max_lag_frames = 1000; o.capacity = 4096;
    std::vector<float> starts;
    streaming::session s(o, [&](const float* f, std::size_t len, std::size_t idx) {
        CHECK(len == 64); CHECK(f[0] == float(idx * 24)); CHECK(f[63] == f[0] + 63); starts.push_back(f[0]); });
    std::size_t pos = 0, chunk = 1;
    while (pos < sig.size()) { std::size_t k = std::min(chunk, sig.size() - pos); CHECK(s.feed(sig.data() + pos, k) == k); pos += k; chunk = chunk * 3 % 97 + 1; s.pump(); }
    s.pump();
    CHECK(starts.size() == (sig.size() - 64) / 24 + 1);
    CHECK(s.stat().dropped == 0 && s.stat().samples_in == sig.size());
}
TEST(streaming_latency_budget_drops_oldest) {
    streaming::options o; o.window = 10; o.hop = 10; o.max_lag_frames = 3; o.capacity = 1000;
    std::vector<std::size_t> seen;
    streaming::session s(o, [&](const float* f, std::size_t, std::size_t idx) { seen.push_back(idx); CHECK(f[0] == float(idx * 10)); });
    std::vector<float> sig(200); for (std::size_t i = 0; i < 200; ++i) sig[i] = float(i);
    s.feed(sig.data(), 200);                               // consumer stalled: 20 frames pending, budget 3
    CHECK(s.pending_frames() == 20);
    CHECK(s.pump() == 3);
    CHECK(s.stat().dropped == 17);
    CHECK((seen == std::vector<std::size_t>{17, 18, 19}));  // only the freshest frames survive
    streaming::ring<int> r(3); int v[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    CHECK(r.capacity() == 3);
    CHECK(r.push(v, 8) == 3); int o2[3]; CHECK(r.peek(o2, 3) == 3 && o2[2] == 3); r.consume(2); CHECK(r.push(v + 5, 3) == 2);
    CHECK(r.peek(o2, 3) == 3 && o2[0] == 3 && o2[1] == 6 && o2[2] == 7);
}

// ---------------------------------------------------------------------------------------------------------------
// Session kinds chosen by the file (FORGE_PLAN E.3): a blueprint declares `session diffusion_loop(...)` or
// `session streaming(...)` (opset 4), the model opens through inference::session, and the runners attach the
// scheduler / framing services to its graph.
namespace {
using namespace forge;
std::string stmp(const std::string& n) { return (std::filesystem::temp_directory_path() / ("forge_test_sess_" + std::to_string(::getpid()) + "_" + n)).string(); }

const char* denoiser_bp = R"(blueprint "tiny-denoiser" opset 4
params { n: 8  h: 16  classes: 5  e: 4 }
input x : f32[n]
input t : f32[1]
input cond : f32[e]
graph {
  xr = reshape(x, shape: [1, n])
  hid = tanh(linear(xr, W.w1) + scale(t, alpha: 0.001) * W.wt + linear(reshape(cond, shape: [1, e]), W.wc) + W.b1)
  eps = reshape(linear(hid, W.w2) + W.b2, shape: [n])
}
output eps : f32[n]
entry encode {
  input label : f32[1]
  output cond : f32[e]
  graph { cond = reshape(embed(W.emb, label), shape: [e]) }
}
session diffusion_loop(steps: 1000, schedule: scaled_linear, beta_start: 0.00085, beta_end: 0.012, prediction: epsilon,
                       sampler: ddim, inference_steps: 12, guidance: 1, latent: x, timestep: t, encoder: encode)
)";
const char* stream_bp = R"(blueprint "tiny-stream" opset 4
params { window: 16  hop: 8  d: 6  taps: 3 }
state conv[1] : conv_window
input frame : f32[window]
graph {
  f = linear(reshape(frame, shape: [1, window]), W.proj) + W.bias
  y = tanh(conv[0].conv(f, W.cw))
}
output y : f32[1, d]
session streaming(window: window, hop: hop, max_lag: 64)
)";

struct tiny_weights { std::map<std::string, std::pair<shape_t, std::vector<float>>> w; };
tiny_weights make_weights(std::initializer_list<std::pair<const char*, shape_t>> spec, unsigned seed, float sd = 0.4f) {
    std::mt19937 rng(seed); std::normal_distribution<float> N(0, sd); tiny_weights t;
    for (auto& [n, sh] : spec) { std::vector<float> v(std::size_t(numel(sh))); for (auto& x : v) x = N(rng); t.w[n] = {sh, std::move(v)}; }
    return t;
}
std::string write_model(const std::string& path, const std::string& bp_text, const tiny_weights& tw) {
    modelfile::writer w;
    for (auto& [n, sv] : tw.w) w.add_tensor(n, sv.first, sv.second);
    auto bp = modelfile::parse_blueprint(bp_text);
    w.set_manifest(json::value(json::object{{"format", "fpk"}, {"architecture", "blueprint"}, {"session", bp.session}}));
    w.set_section("blueprint", bp_text);
    import::add_state_section(w, bp);
    w.write(path);
    return path;
}
tiny_weights denoiser_weights() { return make_weights({{"w1", {16, 8}}, {"wt", {16}}, {"wc", {16, 4}}, {"b1", {16}}, {"w2", {8, 16}}, {"b2", {8}}, {"emb", {5, 4}}}, 31); }
// the denoiser graph by hand: eps = w2 tanh(w1 x + 0.001 t wt + wc cond + b1) + b2
std::vector<float> denoise_ref(const tiny_weights& tw, const std::vector<float>& x, double t, const std::vector<float>& cond) {
    auto& W = tw.w; std::vector<double> h(16);
    for (int i = 0; i < 16; ++i) {
        double a = double(W.at("b1").second[std::size_t(i)]) + double(0.001f * float(t)) * double(W.at("wt").second[std::size_t(i)]);
        for (int j = 0; j < 8; ++j) a += double(W.at("w1").second[std::size_t(i * 8 + j)]) * double(x[std::size_t(j)]);
        for (int j = 0; j < 4; ++j) a += double(W.at("wc").second[std::size_t(i * 4 + j)]) * double(cond[std::size_t(j)]);
        h[std::size_t(i)] = std::tanh(a);
    }
    std::vector<float> e(8);
    for (int i = 0; i < 8; ++i) { double a = double(W.at("b2").second[std::size_t(i)]); for (int j = 0; j < 16; ++j) a += double(W.at("w2").second[std::size_t(i * 16 + j)]) * h[std::size_t(j)]; e[std::size_t(i)] = float(a); }
    return e;
}
double max_diff(const std::vector<float>& a, const std::vector<float>& b) { if (a.size() != b.size()) return 1e9; double m = 0; for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, double(std::fabs(a[i] - b[i]))); return m; }
}

TEST(blueprint_opset4_session_parameters_and_entries) {
    auto bp = modelfile::parse_blueprint(denoiser_bp);
    CHECK(bp.opset == 4 && bp.session == "diffusion_loop" && bp.session_arg("steps", 0) == 1000 && bp.session_arg("beta_end", 0) == 0.012 && bp.session_arg("inference_steps", 0) == 12);
    CHECK(bp.session_str("schedule", "") == "scaled_linear" && bp.session_str("latent", "") == "x" && bp.session_str("encoder", "") == "encode" && bp.session_str("nope", "dflt") == "dflt");
    CHECK(bp.entries.size() == 1 && bp.entry_at("encode").inputs.size() == 1 && bp.entry_at("encode").inputs[0].name == "label" && bp.entry_at("encode").outputs[0].name == "cond");
    CHECK_THROWS(bp.entry_at("nope"));
    auto sb = modelfile::parse_blueprint(stream_bp);
    CHECK(sb.session == "streaming" && sb.session_arg("window", 0) == 16 && sb.session_arg("hop", 0) == 8 && sb.session_arg("max_lag", 0) == 64);   // `window: window` reads the param
    // the weights of every entry point are part of the model
    auto names = modelfile::blueprint_weights(bp);
    CHECK(std::find(names.begin(), names.end(), "emb") != names.end() && std::find(names.begin(), names.end(), "w2") != names.end() && names.size() == 7);
    // each feature needs opset 4, and says so
    auto swap_opset = [](std::string t, const char* to) { auto p = t.find("opset 4"); t.replace(p, 7, to); return t; };
    for (auto* text : {denoiser_bp, stream_bp}) {
        std::string msg; try { (void)modelfile::parse_blueprint(swap_opset(text, "opset 3")); } catch (const forge_exception& e) { msg = e.what(); }
        CHECK_MSG(msg.find("needs opset 4") != std::string::npos, "message: %s", msg.c_str());
    }
    {
        std::string e3 = "blueprint \"e\" opset 3\ninput x : f32[2]\ngraph { y = relu(x) }\noutput y : f32[2]\nentry other { input a : f32[2] output b : f32[2] graph { b = relu(a) } }\n";
        std::string msg; try { (void)modelfile::parse_blueprint(e3); } catch (const forge_exception& e) { msg = e.what(); }
        CHECK(msg.find("entry point 'other'") != std::string::npos && msg.find("opset 4") != std::string::npos);
        std::string p3 = "blueprint \"e\" opset 3\ninput x : f32[2]\ngraph { y = relu(x) }\noutput y : f32[2]\nsession one_shot(a: 1)\n";
        msg.clear(); try { (void)modelfile::parse_blueprint(p3); } catch (const forge_exception& e) { msg = e.what(); }
        CHECK(msg.find("opset 4") != std::string::npos);
    }
    for (const char* bad : {"entry a { graph { b = relu(x) } }", "entry a { input q : f32[2] output b : f32[2] }", "entry main { output b : f32[2] graph { b = relu(x) } }",
                            "entry a { output b : f32[2] graph { b = relu(x) } }\nentry a { output b : f32[2] graph { b = relu(x) } }", "session streaming(window: )", "session streaming(window: -x)"}) {
        std::string t = std::string("blueprint \"e\" opset 4\ninput x : f32[2]\ngraph { y = relu(x) }\noutput y : f32[2]\n") + bad + "\n";
        CHECK_THROWS(modelfile::parse_blueprint(t));
    }
    // entry points build as graphs of their own; both forms of the blueprint agree
    modelfile::build_options o; std::vector<float> z(64, 0.f);
    o.weight = [&](const std::string& n) -> std::pair<shape_t, const float*> { return {n == "emb" ? shape_t{5, 4} : shape_t{1}, z.data()}; };
    auto g = modelfile::build_graph(bp, o, "encode");
    CHECK(g.outputs().size() == 1 && g.outputs().begin()->first == "cond");
    CHECK_THROWS(modelfile::build_graph(bp, o, "nope"));
    for (auto* text : {denoiser_bp, stream_bp}) {
        auto bin = modelfile::encode_blueprint(text);
        auto back = modelfile::blueprint_text(std::string_view(reinterpret_cast<const char*>(bin.data()), bin.size()));
        CHECK(back == modelfile::canonical_blueprint_text(text));
        auto b2 = modelfile::parse_blueprint(back), b1 = modelfile::parse_blueprint(text);
        CHECK(b2.session == b1.session && b2.session_args == b1.session_args && b2.session_strs == b1.session_strs && b2.entries.size() == b1.entries.size());
    }
    std::printf("%s", modelfile::canonical_blueprint_text(denoiser_bp).c_str());
}

TEST(diffusion_loop_session_from_a_model_file) {
    auto tw = denoiser_weights(); auto f = write_model(stmp("den.fpk"), denoiser_bp, tw);
    inference::session s(f);
    CHECK(s.blueprint().session == "diffusion_loop");
    inference::diffusion_runner d(s);
    CHECK(d.latent_size() == 8 && d.encoder_entry() == "encode" && d.defaults().steps == 12 && d.defaults().kind == inference::diffusion::sampler_kind::ddim && d.scheduler().steps() == 1000);
    // the reference: the same scheduler and loop (diffusion.hpp) around the graph computed by hand
    auto emb = [&](int label) { auto& e = tw.w.at("emb").second; return std::vector<float>(e.begin() + label * 4, e.begin() + label * 4 + 4); };
    auto ref_run = [&](inference::diffusion::latent x, const inference::diffusion::options& o, int label, int ulabel) {
        inference::diffusion::session loop(d.scheduler(), [&](const inference::diffusion::latent& xt, double t, int c) { return denoise_ref(tw, xt, t, emb(c ? label : ulabel)); });
        return loop.run(std::move(x), o);
    };
    auto o = d.defaults(); auto x0 = d.initial_noise(42);
    CHECK(x0.size() == 8 && x0 == d.initial_noise(42) && x0 != d.initial_noise(43));
    {   // DDIM, conditional only
        auto y = d.run(x0, o, {{"label", {3.f}}});
        CHECK_MSG(max_diff(y, ref_run(x0, o, 3, 0)) < 2e-5, "ddim: %g", max_diff(y, ref_run(x0, o, 3, 0)));
        CHECK(d.model_calls() == 12 && y != d.run(x0, o, {{"label", {1.f}}}));   // the label reaches the denoiser through the encoder entry
        CHECK(d.sample([&] { auto q = o; q.seed = 42; return q; }(), {{"label", {3.f}}}) == y);
    }
    {   // classifier-free guidance: two denoiser calls per step, the unconditional branch from --uncond (class 0 by default)
        auto g = o; g.guidance = 2.5f; auto before = d.model_calls();
        auto y = d.run(x0, g, {{"label", {4.f}}}, {{"label", {2.f}}});
        CHECK(d.model_calls() - before == 24);
        CHECK_MSG(max_diff(y, ref_run(x0, g, 4, 2)) < 5e-5, "cfg: %g", max_diff(y, ref_run(x0, g, 4, 2)));
        CHECK(max_diff(d.run(x0, g, {{"label", {4.f}}}), ref_run(x0, g, 4, 0)) < 5e-5);
    }
    {   // DDPM (stochastic, seeded), fewer steps, cache and resume
        auto p = o; p.kind = inference::diffusion::sampler_kind::ddpm; p.steps = 7; p.seed = 9; p.cache_at = {3};
        auto y = d.run(x0, p, {{"label", {1.f}}});
        CHECK(y == d.run(x0, p, {{"label", {1.f}}}) && max_diff(y, ref_run(x0, p, 1, 0)) < 5e-5);
        auto q = p; q.seed = 10; CHECK(d.run(x0, q, {{"label", {1.f}}}) != y);
        d.run(x0, p, {{"label", {1.f}}});
        auto mid = d.cache().at(3); auto r = p; r.start_step = 3; r.cache_at = {};
        CHECK(d.run(mid, r, {{"label", {1.f}}}) == y);
    }
    {   // conditioning given directly would need the encoder's inputs; an input the caller leaves out is zeros
        auto y0 = d.run(x0, o);                                  // label 0
        CHECK(max_diff(y0, ref_run(x0, o, 0, 0)) < 2e-5);
    }
    CHECK_THROWS(d.run(inference::diffusion::latent(5, 0.f), o));   // not the latent's size
    // the same file through the binary blueprint form
    auto fb = stmp("den_bin.fpk"); CHECK(modelfile::convert_blueprint_section(f, fb, true));
    { inference::session sb(fb); inference::diffusion_runner db(sb); CHECK(db.run(x0, o, {{"label", {3.f}}}) == d.run(x0, o, {{"label", {3.f}}})); }
    // the scheduler comes from the file: v-prediction, cosine schedule, flow sampler, no encoder
    for (auto [from, to] : {std::pair<std::string, std::string>{"prediction: epsilon", "prediction: v"}, {"schedule: scaled_linear", "schedule: cosine"}, {"sampler: ddim", "sampler: euler_flow"},
                            {"prediction: epsilon", "prediction: sample"}, {"steps: 1000", "steps: 50"}}) {
        std::string t = denoiser_bp; auto p = t.find(from); CHECK(p != std::string::npos); t.replace(p, from.size(), to);
        auto f2 = write_model(stmp("den_v.fpk"), t, tw);
        inference::session s2(f2); inference::diffusion_runner d2(s2);
        auto o2 = d2.defaults(); auto n2 = d2.initial_noise(5);
        inference::diffusion::session loop(d2.scheduler(), [&](const inference::diffusion::latent& xt, double tt, int) { return denoise_ref(tw, xt, tt, emb(2)); });
        auto want = loop.run(n2, o2), got = d2.run(n2, o2, {{"label", {2.f}}});
        // the cosine schedule starts at alpha-bar ~ 5e-5: its first x0 estimate divides by sqrt of that, so float noise
        // in the denoiser output is amplified ~150 times and the latents are large; compare relative to their size
        double scale = 1; for (float v : want) scale = std::max(scale, double(std::fabs(v)));
        CHECK_MSG(max_diff(got, want) < 1e-4 * scale, "%s: %g (latent magnitude %g)", to.c_str(), max_diff(got, want), scale);
        CHECK(to != "steps: 50" || (d2.scheduler().steps() == 50 && o2.steps == 12));
    }
    for (const char* bad : {"schedule: nope", "sampler: nope", "prediction: nope", "latent: nope", "encoder: nope", "steps: 0", "beta_start: 2"}) {
        std::string t = denoiser_bp; std::string key = std::string(bad).substr(0, std::string(bad).find(':'));
        auto p = t.find(key + ":"); auto e = t.find_first_of(",)", p); t.replace(p, e - p, bad);
        auto f3 = write_model(stmp("den_bad.fpk"), t, tw); inference::session s3(f3);
        CHECK_THROWS(inference::diffusion_runner(s3));
    }
    {   // the front door refuses the wrong kind in both directions
        auto g = forge_test::make_tiny_model(stmp("lm.fpk")); inference::session lm(g);
        CHECK_THROWS(inference::diffusion_runner(lm)); CHECK_THROWS(inference::stream_runner(lm));
        inference::sampler smp(inference::sampler_config{}); CHECK_THROWS(s.generate(std::vector<int>{1}, 2, smp));
    }
}

TEST(streaming_session_from_a_model_file) {
    auto tw = make_weights({{"proj", {6, 16}}, {"bias", {6}}, {"cw", {6, 3}}}, 77, 0.3f);
    auto f = write_model(stmp("stream.fpk"), stream_bp, tw);
    {   // the state the graph carries is declared in the file
        modelfile::reader r(f); auto sc = modelfile::state_schema::parse(r.section_text("state"));
        CHECK(sc.slots.size() == 1 && sc.slots[0].recurrent && sc.slots[0].op == "causal_conv" && sc.slots[0].shape == shape_t({2, 6}));
    }
    std::mt19937 rng(5); std::normal_distribution<float> N(0, 1);
    std::vector<float> samples(8 * 40 + 16 + 3); for (auto& x : samples) x = N(rng);   // 41 frames and 3 samples over
    // reference: frames of 16 every 8 samples; f_t = proj frame_t + bias; y_t[c] = tanh(sum_j cw[c][j] f_{t-2+j}[c]), zero history
    const std::size_t n_frames = (samples.size() - 16) / 8 + 1;
    std::vector<std::vector<double>> feat(n_frames, std::vector<double>(6));
    for (std::size_t t = 0; t < n_frames; ++t) for (int c = 0; c < 6; ++c) {
        double a = double(tw.w.at("bias").second[std::size_t(c)]);
        for (int j = 0; j < 16; ++j) a += double(tw.w.at("proj").second[std::size_t(c * 16 + j)]) * double(samples[t * 8 + std::size_t(j)]);
        feat[t][std::size_t(c)] = double(float(a));
    }
    std::vector<float> want;
    for (std::size_t t = 0; t < n_frames; ++t) for (int c = 0; c < 6; ++c) {
        double a = 0; for (int j = 0; j < 3; ++j) { auto tt = std::ptrdiff_t(t) - 2 + j; if (tt >= 0) a += double(tw.w.at("cw").second[std::size_t(c * 3 + j)]) * feat[std::size_t(tt)][std::size_t(c)]; }
        want.push_back(float(std::tanh(a)));
    }
    std::vector<float> first;
    for (std::size_t chunk : {std::size_t(1), std::size_t(5), std::size_t(8), std::size_t(16), std::size_t(37), samples.size()}) {
        inference::session s(f); inference::stream_runner r(s);
        CHECK(r.options().window == 16 && r.options().hop == 8 && r.options().max_lag_frames == 64);
        std::size_t done = 0;
        for (std::size_t at = 0; at < samples.size(); at += chunk) done += r.feed({samples.data() + at, std::min(chunk, samples.size() - at)});
        CHECK_MSG(done == n_frames && r.stat().frames == n_frames && r.stat().dropped == 0 && r.frame_size() == 6 && r.output().size() == 6 * n_frames, "chunk %zu: %zu frames", chunk, done);
        CHECK_MSG(max_diff(r.output(), want) < 2e-5, "chunk %zu: %g", chunk, max_diff(r.output(), want));
        if (first.empty()) first = r.output(); else CHECK(r.output() == first);   // the chunking of the input changes nothing, bit for bit
        CHECK(r.frame_indices().size() == n_frames && r.frame_indices().back() == n_frames - 1);
    }
    {   // the state is the session's: a second runner on a fresh session starts from zero history; reset() does the same
        inference::session s(f); inference::stream_runner r(s);
        r.feed({samples.data(), 64}); auto a = r.take_output();
        s.reset(); inference::stream_runner r2(s); r2.feed({samples.data(), 64});
        CHECK(r2.output() == a && a.size() == 6 * 7);
        r2.feed({samples.data(), 64});                                           // without a reset the conv window still holds the earlier frames
        CHECK(std::vector<float>(r2.output().begin() + 42, r2.output().end()) != a);
    }
    {   // the latency budget of the file can be overridden: a burst larger than max_lag drops the oldest frames
        inference::session s(f); auto o = inference::stream_runner::options_of(s.blueprint()); o.max_lag_frames = 4; o.capacity = 1024;
        inference::stream_runner r(s, o);
        // one large push (feed() itself drains as it goes, so go through the frame scheduler's budget with a burst of 20 frames)
        r.feed({samples.data(), 16 + 8 * 19});
        CHECK(r.stat().frames + r.stat().dropped == 20 && r.stat().dropped > 0 && r.frame_indices().front() > 0);
        auto bad = o; bad.window = 12; CHECK_THROWS(inference::stream_runner(s, bad));   // not the model's frame size
    }
    for (const char* bad : {"window: 0", "hop: 0", "window: 1.5"}) {
        std::string t = stream_bp; std::string key = std::string(bad).substr(0, std::string(bad).find(':'));
        auto p = t.rfind(key + ":"); auto e = t.find_first_of(",)", p); t.replace(p, e - p, bad);
        auto f3 = write_model(stmp("stream_bad.fpk"), t, tw); inference::session s3(f3);
        CHECK_THROWS(inference::stream_runner(s3));
    }
}

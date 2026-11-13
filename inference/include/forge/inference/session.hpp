#pragma once
// Generic inference: the .fpk's blueprint declares the session kind; the engine attaches services.
//
// model   : the immutable, shareable part (mapped file, blueprint, tokenizer, decoded weights, plans).
// session : one sequence's mutable state (KV state store, position). Many sessions may share a model;
//           runs are serialised on the model (plans own a scratch arena), state is per session.
#include <forge/graph/graph.hpp>
#include <forge/import/safetensors.hpp>
#include <forge/inference/aot.hpp>
#include <forge/inference/sampler.hpp>
#include <forge/modelfile/blueprint.hpp>
#include <forge/modelfile/blueprint_bin.hpp>
#include <forge/modelfile/fpk.hpp>
#include <forge/modelfile/sign.hpp>
#include <forge/modelfile/adapters.hpp>
#include <forge/modelfile/state_schema.hpp>
#include <forge/modelfile/tokenizer.hpp>
#include <algorithm>
#include <forge/security/enforce.hpp>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <forge/gpu/determinism.hpp>
#include <mutex>
#include <optional>
#include <span>

namespace forge::inference {

struct session_options {
    i64 context = 0;       // 0 = manifest max_position_embeddings
    bool verify = false;   // check BLAKE3 hashes on open
    i64 page_rows = 256;   // KV page size in rows; 0 = reserve the whole context up front
    std::string kv_codec;  // "", "f32", "int8", "fp8", "tbq4", "tbq3", "tbq4p", "tbq3p" (tbq + QJL residual, CPU only) or "k8v4"; empty = $FORGE_KV_CACHE_TYPE, $FORGE_KV_CODEC or f32
    int kv_offload = -1;   // 1 = keep KV on the host, drop device copies after each attention; -1 = $FORGE_KV_OFFLOAD
    // Determinism mode (gpu/determinism.hpp): 1 = order-stable device paths only (same device + same build -> bit-identical
    // results), 0 = off, -1 = $FORGE_DETERMINISTIC. Process-wide: it applies to every runtime in the process.
    int deterministic = -1;
    // Early exit: > 0 runs only the first `exit_layer` layers and applies the model's final norm and LM head to that
    // hidden state (the draft of self-speculative decoding, speculative.hpp). The blueprint is built with its depth
    // param (`n_layers`, or the one named by `session generative_sequence(depth_param: "n_blocks")`: quoted, since a
    // bare param name in a session parameter stands for the param's value) set to this value.
    // Such a session has its own state and shares the model's weights. 0 = the whole model.
    i64 exit_layer = 0;
};

struct generate_stats { std::size_t prompt_tokens = 0, new_tokens = 0; double prefill_s = 0, decode_s = 0; };

// Whether prompts start with BOS: the manifest's "add_bos" (recorded by importers from GGUF
// tokenizer.ggml.add_bos_token / HF tokenizer_config.json). Files imported before that key existed: Qwen-family
// models never add BOS (their bos_token_id is <|endoftext|>, which derails generation); others keep BOS.
inline bool add_bos_default(const json::value& man) {
    if (auto* ab = man.find("add_bos"); ab && ab->is_bool()) return ab->b();
    return man.str_or("architecture", "").rfind("qwen", 0) != 0;
}

class model {
    // FSS CAP-1 (docs/security-api.md): with an active forge::security policy the model file must be readable under
    // policy fs.read before it is opened. No active policy: nothing is checked (and nothing changes).
    static const std::string& admit(const std::string& path) {
        if (auto p = security::active_policy(); p->active()) security::enforce(*p, security::capability::fs_read, path);
        return path;
    }
public:
    // Precompiled models (docs/aot-models.md): `path` may name a natively compiled model library instead of a .fpk
    // (the weights then come from the .fpk it was compiled from, or from the library itself), and `lib` attaches code
    // that is linked into this program. A .fpk with an "aot" section or a sidecar uses its precompiled plans.
    explicit model(const std::string& path, bool verify = false, std::shared_ptr<aot_native::library> lib = nullptr)
        : lib_(open_native(path, std::move(lib))), fpk_path_(lib_ && aot_native::library::is_library(path) ? (lib_->embedded() ? lib_->path() : lib_->fpk_path()) : path),
          r_([&]() -> modelfile::reader {   // not `c ? reader(..) : reader(..)`: MSVC wants the deleted copy constructor for it
              if (lib_ && lib_->embedded()) return modelfile::reader({lib_->model().fpk_data, std::size_t(lib_->model().fpk_size)}, fpk_path_);
              return modelfile::reader(admit(fpk_path_));
          }()) {
        const std::string& fpk_path = fpk_path_;
        // Strict mode (FORGE_PLAN: "verify before load; refuse unsigned files"): FORGE_REQUIRE_SIGNED=1
        // and/or FORGE_TRUSTED_KEYS=<hex pubkey>[,<hex pubkey>...] require a valid (trusted) signature.
        std::vector<std::string> trusted;
        if (auto* e = std::getenv("FORGE_TRUSTED_KEYS")) {
            std::string s = e;
            for (std::size_t p = 0, q; p < s.size(); p = q + 1) { q = s.find(',', p); if (q == std::string::npos) q = s.size(); if (q > p) trusted.push_back(s.substr(p, q - p)); }
        }
        bool strict = !trusted.empty() || (std::getenv("FORGE_REQUIRE_SIGNED") && std::string(std::getenv("FORGE_REQUIRE_SIGNED")) == "1");
        // FSS INT-1 / INT-4: an active policy with integrity.models "required" or "optional" decides here, before the
        // blueprint is parsed or any tensor is decoded (the file is only mapped and hashed). Its trusted_keys join
        // FORGE_TRUSTED_KEYS (which lockdown() also sets); failures are security::integrity_error, audited.
        const auto pol = security::active_policy();
        const bool fss = pol->active() && pol->integrity.models != security::requirement::off;
        if (fss) {
            for (auto k : pol->integrity.trusted_keys) { for (auto& c : k) c = char(std::tolower(static_cast<unsigned char>(c))); trusted.push_back(std::move(k)); }
            if (pol->integrity.models == security::requirement::required) strict = true;
            const bool check = strict || r_.has_section("signature");
            auto fail = [&](const std::string& why) {
                security::audit("integrity", security::decision::fail, "INT-4", fpk_path, why);
                throw security::integrity_error(fpk_path + ": " + why, "INT-4");
            };
            if (check || verify) if (auto bad = r_.verify(); !bad.empty()) fail("hash mismatch in " + bad.front());
            if (check) {
                auto rep = modelfile::verify_signature(r_, trusted);
                if (!rep.trusted) fail("signature check failed (integrity.models " + std::string(security::name(pol->integrity.models)) + "): " + rep.message);
                security::audit("integrity", security::decision::allow, "INT-1", fpk_path, "model signature verified (key " + rep.valid_keys.front() + ")");
            } else security::audit("integrity", security::decision::weaken, "INT-1", fpk_path, "unsigned model loaded (integrity.models optional)");
        }
        if (!fss && (verify || strict)) {
            auto bad = r_.verify();
            if (!bad.empty()) raise(errc::integrity_error, "{}: hash mismatch in {}", fpk_path, bad.front());
        }
        if (!fss && (strict || (verify && r_.has_section("signature")))) {
            auto rep = modelfile::verify_signature(r_, trusted);
            if (!rep.trusted && (strict || rep.signed_)) raise(errc::integrity_error, "{}: signature check failed: {}", fpk_path, rep.message);
        }
        man_ = r_.manifest();
        bp_ = modelfile::load_blueprint(r_);   // text, or the binary form (modelfile/blueprint_bin.hpp)
        try { modelfile::check_opset(bp_); }   // before any plan is built: an opset this engine does not run
        catch (const forge_exception& e) { throw forge_exception(e.code, fpk_path + ": " + e.what()); }
        // Files imported before the runtime had WordPiece / Unigram tokenizers keep such a tokenizer.json in a
        // "tokenizer_json" section (and no "tokenizer" section): read from there when this build can load it.
        if (!r_.has_section("tokenizer") && r_.has_section("tokenizer_json")) {
            try { tok_.emplace(r_.section_text("tokenizer_json")); }
            catch (const std::exception& e) { tok_.reset(); log::warn("{}: the tokenizer_json section is not usable ({}); the model takes token ids", fpk_path, e.what()); }
        }
        if (r_.has_section("tokenizer")) tok_.emplace(r_.section_text("tokenizer"));
        if (tok_) {
            tok_->set_special(int(man_.num_or("bos_token_id", -1)), int(man_.num_or("eos_token_id", -1)));
            // BOS / EOS around a prompt: the manifest's add_bos / add_eos when the importer recorded them, else the
            // tokenizer.json post-processor when it has one, else the per-architecture default.
            auto* ab = man_.find("add_bos"); auto* ae = man_.find("add_eos");
            if ((ab && ab->is_bool()) || !tok_->has_post_processor()) tok_->set_add_bos(add_bos_default(man_));
            if (ae && ae->is_bool()) tok_->set_add_eos(ae->b());
        }
        if (r_.has_section("chat_template")) chat_template_ = std::string(r_.section_text("chat_template"));
        vocab_ = i64(bp_.param_or("vocab", 0));
        // Optional sections (docs/format.md): adapters carried in the file, and the declared state schema. Both are
        // checked here, before anything runs: a section that does not fit the file is a format error, not a surprise
        // at the first request. Files without them open exactly as before.
        try {
            embedded_ = modelfile::read_adapters(r_);
            if (r_.has_section(modelfile::state_section)) {
                declared_state_ = modelfile::state_schema::parse(r_.section_text(modelfile::state_section));
                if (!(*declared_state_ == derive_state_()))
                    raise(errc::format_error, "the state section does not describe the state its blueprint builds");
            }
        } catch (const forge_exception& e) { throw forge_exception(e.code, fpk_path + ": " + e.what()); }
        if (lib_ && (strict || r_.has_section("signature")))   // compiled code is not covered by the model's signature
            raise(errc::integrity_error, "{}: a compiled model library cannot be loaded for a signed model or under strict loading", path);
        aot_.attach(r_, fpk_path, strict, lib_);
        if (lib_ && !aot_.native()) log::warn("{}: {}; running {} without the compiled code", path, aot_.stats().last_reject, fpk_path);
        if (auto* d = std::getenv("FORGE_DEVICE")) set_device(d);
    }
    // Precompiled plans in use for this model (inference/aot.hpp). set_aot(false) compiles every plan here instead;
    // like set_device() it clears the cached plans, so call it before sessions run.
    const aot_source& aot() const { return aot_; }
    void set_aot(bool on) { std::lock_guard lk(run_mu_); aot_.set_enabled(on); cache_.clear(); }
    const std::string& fpk_path() const { return fpk_path_; }
    // Select the execution device ("cpu", "vulkan"). Unavailable devices fall back to the CPU with a warning.
    // Returns the device actually used. Clears cached plans.
    std::string set_device(const std::string& dev) {
        std::lock_guard lk(run_mu_);
        std::string why, use = dev;
        if (dev == "auto") {   // score every device that opens for this model (gpu/select.hpp)
            gpu::select_request rq; rq.model_bytes = std::size_t(r_.data_size());
            for (auto& [nm, e] : r_.tensors()) if (std::find(rq.codecs.begin(), rq.codecs.end(), e.codec) == rq.codecs.end()) rq.codecs.push_back(e.codec);
            std::string table; use = graph::auto_device(rq, &table);
            log::info("FORGE_DEVICE=auto -> {}\n{}", use, table);
        } else if (dev == "pipe:auto") {   // split the layers over the devices of one backend by throughput and memory (gpu::plan_pipeline)
            gpu::select_request rq; rq.model_bytes = r_.section_bytes("data").size();
            std::string table; use = graph::auto_pipeline(rq, &table);
            log::info("FORGE_DEVICE=pipe:auto -> {}\n{}", use, table);
        }
        reg_ = graph::device_registry(use, &why);
        if (!reg_ && !why.empty()) log::warn("{}; using cpu", why);
        device_ = reg_ ? use : "cpu";
        cache_.clear();
        return device_;
    }
    const std::string& device() const { return device_; }
    static std::shared_ptr<model> open(const std::string& p, bool verify = false) { return std::make_shared<model>(p, verify); }

    const modelfile::blueprint& blueprint() const { return bp_; }
    const json::value& manifest() const { return man_; }
    const modelfile::reader& file() const { return r_; }
    const modelfile::tokenizer* tokenizer() const { return tok_ ? &*tok_ : nullptr; }
    // The model's Jinja chat template (the .fpk "chat_template" section), "" when it has none. Rendered by
    // inference::apply_chat_template (chat_template.hpp).
    const std::string& chat_template() const { return chat_template_; }
    i64 vocab() const { return vocab_; }
    i64 max_context() const { return i64(man_.num_or("max_position_embeddings", 2048)); }
    graph::plan_cache& plans() { return cache_; }
    std::mutex& run_mutex() { return run_mu_; }

    // LoRA adapters: W' = W + (alpha / r) * B @ A for each named weight (A: [r, in], B: [out, r], PEFT layout).
    // Only the targeted matrices are materialised per adapter; base weights stay mapped and shared, and each
    // adapter gets its own plans, so sessions switch adapters per request without reloading anything.
    struct lora { tensor a, b; };
    void add_adapter(const std::string& name, const std::map<std::string, lora>& deltas, float alpha) {
        if (name.empty() || adapters_.count(name)) raise(errc::invalid_argument, "adapter: empty or duplicate name '{}'", name);
        std::map<std::string, tensor> merged;
        for (auto& [w, d] : deltas) {
            tensor base = r_.load(w);
            if (d.a.shape.size() != 2 || d.b.shape.size() != 2 || base.shape.size() != 2 || d.b.shape[1] != d.a.shape[0] ||
                d.b.shape[0] != base.shape[0] || d.a.shape[1] != base.shape[1])
                raise(errc::invalid_argument, "adapter {}: shape mismatch for {}", name, w);
            i64 r = d.a.shape[0], O = base.shape[0], I = base.shape[1]; float sc = alpha / float(r);
            for (i64 o = 0; o < O; ++o) for (i64 k = 0; k < r; ++k) {
                float bk = sc * d.b.data[std::size_t(o * r + k)];
                for (i64 i = 0; i < I; ++i) base.data[std::size_t(o * I + i)] += bk * d.a.data[std::size_t(k * I + i)];
            }
            merged.emplace(w, std::move(base));
        }
        std::lock_guard lk(w_mu_);
        // Plans (possibly held by live sessions) point at the merged data, so an adapter name is immutable.
        if (!adapters_.emplace(name, std::move(merged)).second) raise(errc::invalid_argument, "adapter '{}' already loaded", name);
    }
    // Load an adapter from a .fpk holding "<weight>.lora_a" / "<weight>.lora_b" tensors; alpha from the manifest.
    // A .safetensors file with the same tensor names works too (f32 / f16 / bf16 data); its alpha is the optional
    // one-element tensor "lora_alpha" (default 1).
    void add_adapter_file(const std::string& name, const std::string& path) {
        if (path.size() > 12 && path.compare(path.size() - 12, 12, ".safetensors") == 0) {
            import::safetensors st(path); std::map<std::string, lora> d; float alpha = 1.f;
            for (auto& [n, e] : st.entries()) {
                if (n == "lora_alpha") { auto t = st.load(n); if (!t.data.empty()) alpha = t.data[0]; continue; }
                auto cut = n.rfind(".lora_"); if (cut == std::string::npos) continue;
                auto& l = d[n.substr(0, cut)]; (n.substr(cut) == ".lora_a" ? l.a : l.b) = st.load(n);
            }
            if (d.empty()) raise(errc::invalid_argument, "adapter {}: {} holds no <weight>.lora_a / <weight>.lora_b tensors", name, path);
            add_adapter(name, d, alpha);
            return;
        }
        modelfile::reader ar(path); std::map<std::string, lora> d;
        for (auto& [n, e] : ar.tensors()) {
            auto cut = n.rfind(".lora_"); if (cut == std::string::npos) continue;
            auto& l = d[n.substr(0, cut)]; (n.substr(cut) == ".lora_a" ? l.a : l.b) = ar.load(n);
        }
        add_adapter(name, d, float(ar.manifest().num_or("lora_alpha", 1)));
    }
    bool has_adapter(const std::string& n) const {
        if (n.empty()) return true;
        { std::lock_guard lk(w_mu_); if (adapters_.count(n)) return true; }
        return std::any_of(embedded_.begin(), embedded_.end(), [&](auto& a) { return a.name == n; });
    }
    // Adapters carried in the model file's "adapters" section (modelfile/adapters.hpp). They are selected by name like
    // registered ones (session::set_adapter); the factors are read and merged the first time a name is used.
    const std::vector<modelfile::adapter_info>& embedded_adapters() const { return embedded_; }
    // Makes adapter `n` usable: nothing for "" and for registered adapters, a load for an embedded one not used yet.
    void ensure_adapter(const std::string& n) {
        if (n.empty()) return;
        { std::lock_guard lk(w_mu_); if (adapters_.count(n)) return; }
        auto it = std::find_if(embedded_.begin(), embedded_.end(), [&](auto& a) { return a.name == n; });
        if (it == embedded_.end()) raise(errc::not_found, "adapter '{}' not loaded", n);
        std::map<std::string, lora> d;
        for (auto& w : it->targets) d[w] = lora{r_.load(modelfile::adapter_info::tensor_a(n, w)), r_.load(modelfile::adapter_info::tensor_b(n, w))};
        try { add_adapter(n, d, float(it->alpha)); }
        catch (const forge_exception&) { std::lock_guard lk(w_mu_); if (!adapters_.count(n)) throw; }   // another thread loaded it first
    }
    // Depth of the model in the unit early exit counts in: the value of its depth param (0 when it has none).
    std::string depth_param() const { return bp_.session_str("depth_param", "n_layers"); }
    i64 depth() const { return i64(bp_.param_or(depth_param(), 0)); }
    // The exit layer the file suggests for self-speculative decoding: `session generative_sequence(early_exit: K)`
    // (opset 4), 0 when it names none.
    i64 early_exit_default() const { return i64(bp_.session_arg("early_exit", 0)); }
    // The blueprint cut at `exit_layer`: the same program with the depth param set to it, so its layer loop stops
    // there and the final norm and head read that hidden state. Raises when the model cannot be cut there.
    modelfile::blueprint early_exit_blueprint(i64 exit_layer) const {
        const auto name = depth_param(); const i64 full = depth();
        if (!bp_.params.count(name)) raise(errc::unsupported, "{}: early exit needs the blueprint's depth param '{}' (name another with `session generative_sequence(depth_param: \"name\")`)", fpk_path_, name);
        if (exit_layer < 1 || exit_layer >= full) raise(errc::invalid_argument, "{}: exit layer {} is outside 1 .. {} ({} = {})", fpk_path_, exit_layer, full - 1, name, full);
        modelfile::blueprint cut = bp_;
        cut.params[name] = double(exit_layer);
        // the cut must be a smaller graph: a blueprint whose layer loop runs on another param would give a "draft"
        // that is the whole model
        if (modelfile::blueprint_weights(cut).size() >= modelfile::blueprint_weights(bp_).size())
            raise(errc::unsupported, "{}: setting '{}' to {} does not shorten the graph; name the param that sets the depth with `session generative_sequence(depth_param: \"name\")`", fpk_path_, name, exit_layer);
        return cut;
    }
    // The model's state schema: the file's "state" section when it has one (checked against the blueprint at open),
    // else derived from the blueprint on first use. Null when it cannot be derived (the old behaviour then applies:
    // nothing is checked against it).
    const modelfile::state_schema* state_schema() const {
        if (declared_state_) return &*declared_state_;
        std::call_once(state_once_, [&] { try { derived_state_ = derive_state_(); } catch (const std::exception&) {} });
        return derived_state_ ? &*derived_state_ : nullptr;
    }
    // The adapters that can be selected by name, sorted: the registered ones and the ones carried in the model file.
    std::vector<std::string> adapter_names() const {
        std::vector<std::string> v;
        { std::lock_guard lk(w_mu_); for (auto& [k, a] : adapters_) v.push_back(k); }
        for (auto& a : embedded_) v.push_back(a.name);
        std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end());
        return v;
    }
    std::size_t loaded_weights() const { return weights_.size(); }

    // plan for `n` new tokens; all_rows keeps logits for every position
    // `dims` binds symbolic dimensions other than `seq` (graphs imported from ONNX may have several); such plans are
    // compiled here, never taken from precompiled ones (those are keyed on `seq` alone).
    // `entry` names an entry point of the blueprint (`entry name { ... }`, opset 4) to build instead of the main graph.
    // `exit_layer` > 0 builds the early-exit graph (session_options::exit_layer).
    graph::plan& plan_for(i64 n, bool all_rows, graph::state_store* st, const std::string& adapter = {}, const std::map<std::string, i64>* dims = nullptr, const std::string& entry = {}, i64 exit_layer = 0) {
        std::string dk;
        if (dims) for (auto& [k, v] : *dims) if (k != "seq") dk += std::format(";{}={}", k, v);
        if (!entry.empty() && entry != "main") dk += ";entry=" + entry;
        if (exit_layer > 0) dk += std::format(";exit={}", exit_layer);
        return cache_.get(std::format("{}:{}:{}:{}{}", all_rows ? "all" : "seq", n, adapter, st ? graph::kv_codec_pair_name({st->codec, st->codec_v == graph::kv_codec::f32 ? st->codec : st->codec_v}) : "f32", dk), [&] {
            modelfile::build_options bo;
            if (dims) for (auto& [k, v] : *dims) bo.dims[k] = v;
            bo.dims["seq"] = n;
            bo.all_rows = all_rows;
            bo.weight = [&](const std::string& name) -> std::pair<shape_t, const float*> {
                std::lock_guard lk(w_mu_);
                if (!adapter.empty()) {
                    auto& m = adapters_.at(adapter);
                    if (auto a = m.find(name); a != m.end()) return {a->second.shape, a->second.data.data()};
                }
                auto it = weights_.find(name);
                if (it == weights_.end()) it = weights_.emplace(name, r_.load(name)).first;
                return {it->second.shape, it->second.data.data()};
            };
            // Weights stay in the mapped file: f32 tensors are used in place, other codecs are read in place by
            // quantised matmul (the plan decodes a shared f32 copy only for other consumers). FORGE_DECODE_WEIGHTS=1
            // restores the old behaviour (every weight decoded to f32 up front).
            static const bool decode_all = [] { auto* e = std::getenv("FORGE_DECODE_WEIGHTS"); return e && std::string(e) == "1"; }();
            if (!decode_all) bo.qweight = [&](const std::string& name) -> modelfile::build_options::encoded {
                if (!adapter.empty() && adapters_.at(adapter).count(name)) return {};
                const auto& e = r_.entry(name);
                if (e.codec == "f32") return {};  // handled by `weight` (zero-copy below)
                return {e.shape, r_.tensor_bytes(name).data(), e.codec};
            };
            if (!decode_all) bo.weight = [&, inner = bo.weight](const std::string& name) -> std::pair<shape_t, const float*> {
                const auto& e = r_.entry(name);
                bool adapted = !adapter.empty() && adapters_.at(adapter).count(name);
                if (!adapted && e.codec == "f32")
                    if (const auto* base = r_.tensor_bytes(name).data(); reinterpret_cast<std::uintptr_t>(base) % alignof(float) == 0)
                        return {e.shape, reinterpret_cast<const float*>(base)};
                return inner(name);
            };
            graph::pass_options po; po.lowerings = reg_.get();
            if (!decode_all && dk.empty()) if (auto ap = aot_.try_plan(n, all_rows, st, bo, po, adapter)) return ap;   // precompiled
            if (exit_layer > 0) return graph::compile(modelfile::build_graph(early_exit_blueprint(exit_layer), bo, entry), st, po);
            return graph::compile(modelfile::build_graph(bp_, bo, entry), st, po);
        });
    }
private:
    modelfile::state_schema derive_state_() const {
        return modelfile::derive_state_schema(bp_, [&](const std::string& n) { return r_.entry(n).shape; });
    }
    static std::shared_ptr<aot_native::library> open_native(const std::string& path, std::shared_ptr<aot_native::library> lib) {
        if (lib || !aot_native::library::is_library(path)) return lib;
        admit(path);
        return aot_native::library::open(path);
    }
    std::shared_ptr<aot_native::library> lib_;   // before r_: an embedded model file lives in the library
    std::string fpk_path_;
    modelfile::reader r_;
    aot_source aot_;
    json::value man_;
    modelfile::blueprint bp_;
    std::optional<modelfile::tokenizer> tok_;
    std::string chat_template_;
    graph::plan_cache cache_;
    std::map<std::string, tensor> weights_;
    std::map<std::string, std::map<std::string, tensor>> adapters_;
    std::vector<modelfile::adapter_info> embedded_;
    std::optional<modelfile::state_schema> declared_state_;
    mutable std::optional<modelfile::state_schema> derived_state_;
    mutable std::once_flag state_once_;
    mutable std::mutex w_mu_; std::mutex run_mu_;
    std::unique_ptr<graph::registry> reg_;
    std::string device_ = "cpu";
    i64 vocab_ = 0;
};

// Header of session::save_prefix blobs (KV state of a token prefix).
inline constexpr char kv_blob_magic[9] = "FORGEKV1";

class session {
public:
    explicit session(const std::string& fpk_path, session_options o = {}) : session(model::open(fpk_path, o.verify), o) {}
    explicit session(std::shared_ptr<model> m, session_options o = {}) : m_(std::move(m)), opt_(o) {
        if (o.deterministic >= 0) gpu::set_deterministic(o.deterministic != 0);
        st_.capacity = o.context ? o.context : m_->max_context();
        st_.page_rows = o.page_rows;
        std::string kc = o.kv_codec;
        if (kc.empty()) if (const char* e = std::getenv("FORGE_KV_CACHE_TYPE")) kc = e;
        if (kc.empty()) if (const char* e = std::getenv("FORGE_KV_CODEC")) kc = e;
        auto kp = graph::kv_codec_pair_from_name(kc);
        st_.codec = kp.k; st_.codec_v = kp.v != kp.k ? kp.v : graph::kv_codec::f32;
        int off = o.kv_offload;
        if (off < 0) { const char* e = std::getenv("FORGE_KV_OFFLOAD"); off = e && *e && *e != '0'; }
        st_.offload = off != 0;
    }
    const std::shared_ptr<model>& shared_model() const { return m_; }
    const session_options& options() const { return opt_; }
    const modelfile::blueprint& blueprint() const { return m_->blueprint(); }
    const json::value& manifest() const { return m_->manifest(); }
    const modelfile::reader& file() const { return m_->file(); }
    const modelfile::tokenizer* tokenizer() const { return m_->tokenizer(); }
    i64 vocab() const { return m_->vocab(); }
    i64 position() const { return pos_; }
    i64 context() const { return st_.capacity; }
    const graph::state_store& state() const { return st_; }
    void reset() { st_.reset(); pos_ = 0; }
    // Select a LoRA adapter registered on the model ("" = base). The KV cache is cleared, since cached
    // keys/values were computed with the previous weights.
    void set_adapter(const std::string& a) {
        if (!m_->has_adapter(a)) raise(errc::not_found, "adapter '{}' not loaded", a);
        m_->ensure_adapter(a);   // an adapter embedded in the model file is read on first use
        if (a != adapter_) { adapter_ = a; execs_.clear(); reset(); }
    }
    const std::string& adapter() const { return adapter_; }
    // roll back to `pos` tokens (speculative rejection, prefix reuse). Recurrent state (e.g. Qwen3.5 DeltaNet,
    // conv windows) cannot be cut: this raises for such models unless pos is 0; use checkpoint() + rewind(pos, ck, tokens).
    void rewind(i64 pos) { if (pos < pos_) { st_.truncate(pos); pos_ = pos; } }
    // True when the state holds recurrent slots, which only make sense at their full length.
    bool has_recurrent_state() const { for (auto& [k, s] : st_.slots) if (s.fixed) return true; return false; }
    // Whether the first `n` tokens of this session can be shared (copy_prefix / save_prefix) or rewound to.
    bool can_share_prefix(i64 n) const {
        if (n > pos_) return false;
        for (auto& [k, s] : st_.slots) if (s.fixed && n != s.len) return false;
        return true;
    }
    // A snapshot of the recurrent slots at the current position (empty for pure KV-cache models), so a later
    // rewind can restore them and replay the tokens since, instead of cutting a state that cannot be cut.
    struct checkpoint { i64 pos = 0; std::map<std::string, graph::state_store::slot> fixed; };
    checkpoint make_checkpoint() const {
        checkpoint c; c.pos = pos_;
        if (!has_recurrent_state()) return c;
        st_.flush();   // device-resident rows back to the host copy
        for (auto& [k, s] : st_.slots) if (s.fixed) c.fixed.emplace(k, s);
        return c;
    }
    // Roll back to `pos` (ck.pos <= pos <= position()), where `since` holds the tokens fed after the checkpoint.
    // KV-only models just truncate; recurrent slots are restored from `ck` and since[0 .. pos - ck.pos) replayed.
    void rewind(i64 pos, const checkpoint& ck, std::span<const int> since) {
        if (pos >= pos_) return;
        if (ck.fixed.empty()) { rewind(pos); return; }
        if (pos < ck.pos || pos - ck.pos > i64(since.size())) raise(errc::invalid_argument, "rewind: position {} is outside the checkpoint range", pos);
        st_.forget();   // device copies are stale after the restore
        for (auto& [k, s] : ck.fixed) st_.slots[k] = s;
        st_.truncate(ck.pos); pos_ = ck.pos;
        if (pos > ck.pos) forward(since.subspan(0, std::size_t(pos - ck.pos)));
    }
    // ---- speculative verify with cheap recurrent rollback (used by speculative.hpp) ----
    // forward_spec(ids, keep) is forward_all(ids); on a model with recurrent state it also sets that state aside
    // after the first `keep` rows, inside the same pass (a device-to-device copy on GPU backends: no host transfer,
    // no extra pass). spec_rewind(pos) then rolls back to any pos in [mark, position()], mark = position before the
    // call + keep: KV-only models are cut at pos and it returns 0; recurrent models go back to the mark and it
    // returns pos - mark, the number of tokens (ids[keep ...]) the caller must feed again, e.g. at the head of its
    // next pass (they cost a few extra rows there, not a pass of their own).
    // The returned rows ([ids.size(), vocab]) are the executor's output buffer: valid until the next forward.
    std::span<const float> forward_spec(std::span<const int> ids, i64 keep) {
        keep = std::clamp<i64>(keep, 0, i64(ids.size()));
        spec_mark_ = pos_ + keep;
        if (has_recurrent_state()) st_.snap_row = keep;
        struct reset_snap { graph::state_store& s; ~reset_snap() { s.snap_row = -1; } } g{st_};
        return run_view(ids, true);
    }
    i64 spec_rewind(i64 pos) {
        if (pos >= pos_) return 0;
        if (!has_recurrent_state()) { rewind(pos); return 0; }
        if (pos < spec_mark_ || spec_mark_ > pos_ || !st_.restore_snapshots())
            raise(errc::invalid_argument, "spec_rewind: position {} is outside the last forward_spec range", pos);
        st_.truncate(spec_mark_); pos_ = spec_mark_;
        return pos - spec_mark_;
    }
    // plain-greedy step: feed `ids`, return the argmax of the last position's raw logits (on the device when it can)
    int forward_argmax(std::span<const int> ids) { return run_greedy(ids); }
    // ---- end speculative ----
    // copy the first `n` tokens of another session's state (prefix sharing across sequences)
    void copy_prefix(const session& o, i64 n) {
        if (o.m_ != m_) raise(errc::invalid_argument, "copy_prefix: sessions use different models");
        n = std::min(n, o.pos_);
        if (o.st_.codec != st_.codec) raise(errc::invalid_argument, "copy_prefix: sessions use different KV codecs");
        o.st_.flush(); st_.forget();
        st_.reset();
        for (auto& [k, s] : o.st_.slots) {
            auto& d = st_.slots[k];
            if (s.fixed) {   // recurrent state: only a full-length copy is meaningful
                if (n != s.len) raise(errc::unsupported, "copy_prefix: recurrent slot '{}' holds {} tokens, cannot share {}", k, s.len, n);
                d = s; continue;
            }
            if (!d.capacity) d.codec = s.codec;
            shape_t row = s.buf.shape;
            st_.ensure(d, row, std::max<i64>(n, 1));
            i64 rw = numel(s.buf.shape) / std::max<i64>(s.capacity, 1);
            std::copy_n(s.buf.data.data(), std::size_t(rw * n), d.buf.data.data());
            d.len = n;
        }
        pos_ = n;
    }
    // Serialise the KV state of the first `n` tokens (all, by default) to a blob, e.g. for mem::spill_store.
    // load_prefix() restores it into a session of the same model and KV codec: afterwards the session continues
    // exactly as copy_prefix() from the original would (identical logits).
    std::vector<std::byte> save_prefix(i64 n = -1) const {
        n = n < 0 ? pos_ : std::min(n, pos_);
        st_.flush();   // device-resident rows back to the host copy
        std::vector<std::byte> b;
        auto put = [&](const void* p, std::size_t k) { auto* c = static_cast<const std::byte*>(p); b.insert(b.end(), c, c + k); };
        auto u64 = [&](std::uint64_t v) { put(&v, 8); };
        put(kv_blob_magic, 8); u64(std::uint64_t(n)); u64(std::uint64_t(st_.codec)); u64(st_.slots.size());
        for (auto& [k, s] : st_.slots) {
            if (s.fixed && n != s.len) raise(errc::unsupported, "save_prefix: recurrent slot '{}' holds {} tokens, cannot save {}", k, s.len, n);
            u64(k.size()); put(k.data(), k.size());
            u64(s.fixed); u64(std::uint64_t(s.codec)); u64(std::uint64_t(s.len)); u64(std::uint64_t(s.capacity)); u64(s.buf.shape.size());
            for (auto d : s.buf.shape) u64(std::uint64_t(d));
            std::size_t nf = s.fixed ? s.buf.data.size() : std::size_t(numel(s.buf.shape) / std::max<i64>(s.capacity, 1) * n);
            u64(nf); put(s.buf.data.data(), nf * sizeof(float));
        }
        return b;
    }
    void load_prefix(std::span<const std::byte> b) {
        std::size_t at = 0;
        auto get = [&](void* p, std::size_t k) { if (at + k > b.size()) raise(errc::parse_error, "load_prefix: truncated KV blob"); std::memcpy(p, b.data() + at, k); at += k; };
        auto u64 = [&] { std::uint64_t v; get(&v, 8); return v; };
        char magic[8]; get(magic, 8);
        if (std::memcmp(magic, kv_blob_magic, 8)) raise(errc::parse_error, "load_prefix: not a KV blob");
        i64 n = i64(u64());
        if (graph::kv_codec(u64()) != st_.codec) raise(errc::invalid_argument, "load_prefix: the blob uses a different KV codec");
        if (n < 0 || n > st_.capacity) raise(errc::out_of_memory, "load_prefix: {} tokens exceed the context {}", n, st_.capacity);
        // The blob is checked against the model's state schema before anything is restored: every slot must be one
        // the model declares, of the declared kind and row shape (in this session's KV codec), and appear once.
        const modelfile::state_schema* schema = m_->state_schema();
        if (schema) check_prefix_blob_(b.subspan(at), n, *schema);
        st_.forget(); st_.reset();
        for (std::uint64_t i = 0, ns = u64(); i < ns; ++i) {
            std::uint64_t kl = u64(); if (kl > b.size() - at) raise(errc::parse_error, "load_prefix: truncated KV blob");
            std::string k(std::size_t(kl), '\0'); get(k.data(), k.size());
            bool fixed = u64() != 0; auto codec = graph::kv_codec(u64()); i64 len = i64(u64()), cap = i64(u64());
            std::uint64_t rank = u64(); if (rank > 8) raise(errc::parse_error, "load_prefix: slot '{}' has rank {}", k, rank);
            shape_t sh(static_cast<std::size_t>(rank)); for (auto& d : sh) d = i64(u64());
            std::size_t nf = u64();
            auto& d = st_.slots[k];
            if (fixed) { d = {}; d.buf = tensor(sh); d.len = len; d.capacity = cap; d.codec = codec; d.fixed = true; }
            else { if (!d.capacity) d.codec = codec; st_.ensure(d, sh, std::max<i64>(n, 1)); d.len = n; }
            if (nf > d.buf.data.size()) raise(errc::parse_error, "load_prefix: slot '{}' does not fit its shape", k);
            get(d.buf.data.data(), nf * sizeof(float));
        }
        pos_ = n;
    }
    graph::plan_cache& plans() { return m_->plans(); }
private:
    // Walks a prefix blob's slot table (after its header) without restoring anything and raises on the first slot
    // the schema does not allow. Sizes are checked against the bytes present before they are used.
    void check_prefix_blob_(std::span<const std::byte> b, i64 n, const modelfile::state_schema& schema) const {
        std::size_t at = 0;
        auto need = [&](std::uint64_t k) { if (k > b.size() - at) raise(errc::parse_error, "load_prefix: truncated KV blob"); };
        auto u64 = [&] { need(8); std::uint64_t v; std::memcpy(&v, b.data() + at, 8); at += 8; return v; };
        std::set<std::string> seen;
        for (std::uint64_t i = 0, ns = u64(); i < ns; ++i) {
            std::uint64_t kl = u64(); need(kl);
            std::string k(reinterpret_cast<const char*>(b.data() + at), std::size_t(kl)); at += std::size_t(kl);
            bool fixed = u64() != 0; auto codec = graph::kv_codec(u64()); i64 len = i64(u64()); (void)u64();
            std::uint64_t rank = u64();
            if (rank < 1 || rank > 8) raise(errc::format_error, "load_prefix: slot '{}' has rank {}", k, rank);
            shape_t sh(static_cast<std::size_t>(rank)); for (auto& d : sh) d = i64(u64());
            std::uint64_t nf = u64(); need(nf * sizeof(float) < nf ? ~std::uint64_t(0) : nf * sizeof(float)); at += std::size_t(nf * sizeof(float));
            const auto* decl = schema.find(k);
            if (!decl) raise(errc::format_error, "load_prefix: the blob holds state slot '{}', which this model does not declare", k);
            if (!seen.insert(k).second) raise(errc::format_error, "load_prefix: state slot '{}' appears twice", k);
            if (decl->recurrent != fixed) raise(errc::format_error, "load_prefix: state slot '{}' is {} in the model and {} in the blob", k, decl->recurrent ? "recurrent" : "appended rows", fixed ? "recurrent" : "appended rows");
            if (fixed) {
                if (sh != decl->shape || len < 0 || nf != std::uint64_t(numel(decl->shape)))
                    raise(errc::format_error, "load_prefix: recurrent slot '{}' has shape {}, the model declares {}", k, shape_str(sh), shape_str(decl->shape));
            } else {
                shape_t logical{1}; logical.insert(logical.end(), decl->shape.begin(), decl->shape.end());
                if (codec != graph::kv_codec::f32 && logical.size() != 3) raise(errc::format_error, "load_prefix: slot '{}' cannot be stored in a KV codec", k);
                shape_t phys = graph::state_store::physical(logical, codec);
                bool ok = sh.size() == phys.size() && sh[0] >= 0 && std::equal(sh.begin() + 1, sh.end(), phys.begin() + 1);
                std::uint64_t row = 1; for (std::size_t j = 1; j < phys.size(); ++j) row *= std::uint64_t(phys[j]);
                if (!ok || nf != row * std::uint64_t(n))
                    raise(errc::format_error, "load_prefix: slot '{}' holds rows of shape {}, the model declares {}", k, shape_str(sh), shape_str(decl->shape));
            }
        }
    }
public:

    // Runs the graph on `ids` continuing from the current position; returns logits of the last position.
    std::vector<float> forward(std::span<const int> ids) { return run(ids, false); }
    // Same, but returns logits for every input position ([ids.size(), vocab] row-major).
    std::vector<float> forward_all(std::span<const int> ids) { return run(ids, true); }

    // generative_sequence: prefill + decode loop with a sampler chain. `on_token` returning false stops.
    std::vector<int> generate(std::span<const int> prompt, int max_new, sampler& s,
                              const std::function<bool(int)>& on_token = {}, generate_stats* stats = nullptr, bool stop_at_eos = true) {
        check_generative();
        std::vector<int> hist(prompt.begin(), prompt.end()), out;
        using clk = std::chrono::steady_clock;
        auto t0 = clk::now();
        // Plain greedy: the device picks the token and only its index comes back (no logits download per token).
        const bool greedy = s.plain_greedy();
        std::span<const float> logits; int next = -1;
        if (greedy) next = run_greedy(prompt); else logits = run_view(prompt, false);   // logits: the executor's output buffer, valid until the next run
        auto t1 = clk::now();
        int eos = tokenizer() ? tokenizer()->eos() : -1;
        for (int i = 0; i < max_new; ++i) {
            int t = greedy ? next : s.sample(logits, hist);
            out.push_back(t); hist.push_back(t);
            if (on_token && !on_token(t)) break;
            if (stop_at_eos && t == eos) break;
            if (i + 1 == max_new || pos_ >= st_.capacity) break;
            if (greedy) next = run_greedy(std::span<const int>(&t, 1)); else logits = run_view(std::span<const int>(&t, 1), false);
        }
        auto t2 = clk::now();
        if (stats) *stats = {prompt.size(), out.size(), std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count()};
        return out;
    }
    void check_generative() const {
        if (blueprint().session != "generative_sequence") raise(errc::unsupported, "session kind '{}' cannot generate", blueprint().session);
    }

    // one_shot graphs (e.g. imported ONNX): bind named f32 inputs, run once, return every declared output.
    // `dims` binds further symbolic dimensions by name (a graph with more than one; see model::plan_for).
    std::map<std::string, std::vector<float>> infer(const std::map<std::string, std::vector<float>>& in, i64 seq = 1, const std::map<std::string, i64>& dims = {}) {
        std::lock_guard lk(m_->run_mutex());
        auto& p = m_->plan_for(seq, false, &st_, {}, dims.empty() ? nullptr : &dims);
        auto it = execs_.find(&p);
        if (it == execs_.end()) it = execs_.emplace(&p, std::make_unique<graph::executor>(p, &st_)).first;
        auto& ex = *it->second;
        for (auto& d : blueprint().inputs)
            if (!in.count(d.name)) raise(errc::invalid_argument, "infer: missing input '{}'", d.name);
        for (auto& [k, v] : in) ex.set_input(k, v);
        ex.run();
        std::map<std::string, std::vector<float>> out;
        for (auto& o : blueprint().outputs) { auto v = ex.output_view(o.name); out[o.name].assign(v.data, v.data + v.size()); }
        return out;
    }

    // A named entry point of the blueprint (opset 4): binds its inputs, runs it once, returns its outputs.
    std::map<std::string, std::vector<float>> infer_entry(const std::string& entry, const std::map<std::string, std::vector<float>>& in, i64 seq = 1, const std::map<std::string, i64>& dims = {}) {
        if (entry.empty() || entry == "main") return infer(in, seq, dims);
        const auto& e = blueprint().entry_at(entry);
        std::lock_guard lk(m_->run_mutex());
        auto& p = m_->plan_for(seq, false, &st_, {}, dims.empty() ? nullptr : &dims, entry);
        auto it = execs_.find(&p);
        if (it == execs_.end()) it = execs_.emplace(&p, std::make_unique<graph::executor>(p, &st_)).first;
        auto& ex = *it->second;
        for (auto& d : e.inputs) {
            auto f = in.find(d.name);
            if (f == in.end()) raise(errc::invalid_argument, "infer: entry '{}' is missing input '{}'", entry, d.name);
            ex.set_input(d.name, f->second);
        }
        ex.run();
        std::map<std::string, std::vector<float>> out;
        for (auto& o : e.outputs) { auto v = ex.output_view(o.name); out[o.name].assign(v.data, v.data + v.size()); }
        return out;
    }

    // ---- embedding sessions (`session embedding` in the blueprint) ----
    // One stateless pass; the first declared output is the result: a pooled sentence or image vector, or an
    // encoder's state sequence (flattened row-major; embedding_shape() gives its dims).
    void check_embedding() const {
        if (blueprint().session != "embedding") raise(errc::unsupported, "session kind '{}' is not an embedding model", blueprint().session);
    }
    // Token models (BERT family): feeds the inputs the blueprint declares out of `tokens` (the ids), `positions`
    // (0 .. n-1), `token_types` (zeros unless given) and `mask` (ones unless given: 0 marks padding).
    std::vector<float> embed(std::span<const int> ids, std::span<const int> token_types = {}, std::span<const int> mask = {}) {
        check_embedding();
        const i64 n = i64(ids.size());
        if (n == 0) raise(errc::invalid_argument, "embed: no tokens");
        if (n > m_->max_context()) raise(errc::out_of_memory, "embed: {} tokens exceed the model's {} positions", n, m_->max_context());
        if ((!token_types.empty() && i64(token_types.size()) != n) || (!mask.empty() && i64(mask.size()) != n))
            raise(errc::invalid_argument, "embed: token_types / mask must have one entry per token");
        std::map<std::string, std::vector<float>> in;
        bool has_tokens = false;
        for (auto& d : blueprint().inputs) {
            auto& v = in[d.name]; v.resize(std::size_t(n));
            if (d.name == "tokens") { std::copy(ids.begin(), ids.end(), v.begin()); has_tokens = true; }
            else if (d.name == "positions") for (i64 i = 0; i < n; ++i) v[std::size_t(i)] = float(i);
            else if (d.name == "token_types") { if (!token_types.empty()) std::copy(token_types.begin(), token_types.end(), v.begin()); }
            else if (d.name == "mask") { if (mask.empty()) std::fill(v.begin(), v.end(), 1.f); else std::copy(mask.begin(), mask.end(), v.begin()); }
            else raise(errc::unsupported, "embed: model input '{}' is not a token input; use embed_inputs()", d.name);
        }
        if (!has_tokens) raise(errc::unsupported, "embed: the model takes no `tokens` input; use embed_inputs()");
        return embed_inputs(in, n);
    }
    // Text through the model's tokenizer: the special tokens its post-processor puts around a sequence ([CLS] ... [SEP])
    // and the segment ids it assigns. `pair`, when given, is the second sequence (query / passage; segment 1 for BERT).
    // A text longer than the model's positions is an error, not truncated.
    std::vector<float> embed_text(std::string_view text, const std::string* pair = nullptr, std::size_t* n_tokens = nullptr) {
        check_embedding();
        auto* tk = m_->tokenizer();
        if (!tk) raise(errc::not_found, "embed: the model has no tokenizer; pass token ids");
        auto e = pair ? tk->encode_pair(text, *pair) : tk->encode_typed(text);
        if (n_tokens) *n_tokens = e.ids.size();
        return embed(e.ids, e.type_ids);
    }
    // Several token sequences in one call: one vector per input, in input order. The graph has no batch axis, so this
    // is a loop with shared setup, not one batched pass: every input is checked before anything runs (a bad one fails
    // the call without partial work), and the inputs run shortest first so each sequence length compiles its plan and
    // builds its executor once (both are cached per length on the model / session and reused by later calls).
    // `types`, when not empty, holds one token-type list per input (an empty list = zeros).
    std::vector<std::vector<float>> embed_batch(std::span<const std::vector<int>> ids, std::span<const std::vector<int>> types = {}) {
        check_embedding();
        if (!types.empty() && types.size() != ids.size()) raise(errc::invalid_argument, "embed_batch: token types must be given for every input or for none");
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (ids[i].empty()) raise(errc::invalid_argument, "embed_batch: input {} has no tokens", i);
            if (i64(ids[i].size()) > m_->max_context()) raise(errc::invalid_argument, "embed_batch: input {} is {} tokens, more than the model's {} positions", i, ids[i].size(), m_->max_context());
            if (!types.empty() && !types[i].empty() && types[i].size() != ids[i].size()) raise(errc::invalid_argument, "embed_batch: input {} has {} tokens but {} token types", i, ids[i].size(), types[i].size());
        }
        std::vector<std::size_t> order(ids.size()); std::iota(order.begin(), order.end(), std::size_t(0));
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return ids[a].size() < ids[b].size(); });
        std::vector<std::vector<float>> out(ids.size());
        for (auto i : order) out[i] = embed(ids[i], types.empty() ? std::span<const int>{} : std::span<const int>(types[i]));
        return out;
    }
    // Any embedding model: named f32 inputs as the blueprint declares them (an image [c, h, w], mel frames
    // [n_mels, frames], or the token inputs above). `seq` binds the symbolic `seq` dimension when there is one.
    std::vector<float> embed_inputs(const std::map<std::string, std::vector<float>>& in, i64 seq = 1) {
        check_embedding();
        auto out = infer(in, seq);
        return std::move(out.at(blueprint().outputs.front().name));
    }
    // Declared shape of the embedding output with `seq` bound (e.g. {384}, or {1500, 384} for encoder states).
    shape_t embedding_shape(i64 seq = 1) const {
        check_embedding();
        shape_t s;
        for (auto& d : blueprint().outputs.front().dims) {
            if (d == "seq") s.push_back(seq);
            else if (auto it = blueprint().params.find(d); it != blueprint().params.end()) s.push_back(i64(it->second));
            else s.push_back(i64(std::strtoll(d.c_str(), nullptr, 10)));
        }
        return s;
    }
    // ---- end embedding ----
private:
    std::vector<float> run(std::span<const int> ids, bool all) { auto v = run_view(ids, all); return {v.begin(), v.end()}; }
    // Runs like run_view and returns the argmax of the last position's logits: on the device when the backend can
    // (one integer read back), else over the host view (ties: the lowest index, as std::max_element).
    int run_greedy(std::span<const int> ids) {
        greedy_next_ = true;
        auto v = run_view(ids, false);
        greedy_next_ = false;
        if (last_argmax_ >= 0) return int(last_argmax_);
        return int(std::max_element(v.begin(), v.end()) - v.begin());
    }
    std::span<const float> run_view(std::span<const int> ids, bool all) {
        if (ids.empty()) raise(errc::invalid_argument, "forward: no tokens");
        if (pos_ + i64(ids.size()) > st_.capacity) raise(errc::out_of_memory, "context {} exceeded", st_.capacity);
        i64 n = i64(ids.size());
        std::lock_guard lk(m_->run_mutex());
        auto& p = m_->plan_for(n, all, &st_, adapter_, nullptr, {}, opt_.exit_layer);
        auto it = execs_.find(&p);
        if (it == execs_.end()) it = execs_.emplace(&p, std::make_unique<graph::executor>(p, &st_)).first;
        auto& ex = *it->second;
        std::vector<float> tk(ids.begin(), ids.end()); std::vector<float> ps(static_cast<std::size_t>(n));
        for (i64 i = 0; i < n; ++i) ps[std::size_t(i)] = float(pos_ + i);
        ex.set_input("tokens", tk);
        ex.set_input("positions", ps);
        ex.want_argmax(greedy_next_);
        ex.run();
        last_argmax_ = greedy_next_ ? ex.last_argmax() : -1;
        pos_ += n;
        auto v = ex.output_view(blueprint().outputs.front().name);
        return {v.data, v.size()};
    }
    std::shared_ptr<model> m_;
    session_options opt_;
    graph::state_store st_;
    std::map<graph::plan*, std::unique_ptr<graph::executor>> execs_;
    i64 pos_ = 0;
    std::string adapter_;
    bool greedy_next_ = false; long long last_argmax_ = -1;
    i64 spec_mark_ = 0;   // forward_spec: position whose recurrent state was set aside
};

} // namespace forge::inference

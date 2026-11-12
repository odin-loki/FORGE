// Swift binding over FORGE's C ABI (forge.h, ABI 4).
//
//   ForgeWeights   one opened model file (weights, tokenizer, adapters, device); shared by sessions and batches
//   ForgeModel     one sequence (KV state); ForgeModel(path:) owns private weights, weights.session() shares them
//   ForgeBatch     continuous batching over shared weights
//   Forge          library-level helpers (JSON Schema -> grammar, determinism, AOT sidecar)
//
// Thread safety follows the C ABI (docs/capi.md): one thread (or actor) at a time per object.
//
// NEVER COMPILED: written without a Mac or a Swift toolchain (HARDWARE-UNTESTED). See docs/bindings.md.
import CForge
import Foundation

public struct ForgeError: Error, CustomStringConvertible {
    public let description: String
    static func last(_ what: String) -> ForgeError { ForgeError(description: "\(what): \(String(cString: forge_last_error()))") }
    static func of(_ what: String, _ message: UnsafePointer<CChar>?) -> ForgeError {
        let m = message.map { String(cString: $0) } ?? ""
        return m.isEmpty ? last(what) : ForgeError(description: "\(what): \(m)")
    }
}

/// Why a generation ended (`ForgeModel.lastStopReason`).
public enum StopReason: Int32 { case length = 0, eos = 1, callback = 2, grammar = 3 }

/// C copies of strings and arrays that must stay alive for the duration of one C call.
private final class CArgs {
    private var frees: [() -> Void] = []
    func str(_ s: String?) -> UnsafePointer<CChar>? {
        guard let s = s, let p = strdup(s) else { return nil }
        frees.append { free(p) }
        return UnsafePointer(p)
    }
    func ints(_ a: [Int32]?) -> UnsafePointer<Int32>? {
        guard let a = a, !a.isEmpty else { return nil }
        let p = UnsafeMutablePointer<Int32>.allocate(capacity: a.count)
        p.initialize(from: a, count: a.count)
        frees.append { p.deallocate() }
        return UnsafePointer(p)
    }
    func bias(_ b: [Int32: Float]) -> UnsafePointer<forge_logit_bias>? {
        if b.isEmpty { return nil }
        let p = UnsafeMutablePointer<forge_logit_bias>.allocate(capacity: b.count)
        var i = 0
        for (token, value) in b { p[i] = forge_logit_bias(token: token, bias: value); i += 1 }
        frees.append { p.deallocate() }
        return UnsafePointer(p)
    }
    deinit { for f in frees { f() } }
}

/// A C function that writes text and returns the bytes needed: size the buffer, then read.
private func cText(_ what: String, _ call: (UnsafeMutablePointer<CChar>?, Int64) -> Int64) throws -> String {
    let need = call(nil, 0)
    if need < 0 { throw ForgeError.last(what) }
    var buf = [CChar](repeating: 0, count: Int(need) + 1)
    _ = call(&buf, Int64(buf.count))
    return String(decoding: buf.prefix(Int(need)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

/// The sampler chain: logit bias -> repetition / presence / frequency penalties -> grammar -> temperature -> top-k ->
/// top-p -> min-p -> typical -> draw (Mirostat v2 replaces the truncation steps when `mirostatTau` > 0).
public struct Sampling {
    public var temperature: Float = 0, topK: Int32 = 0, topP: Float = 1, minP: Float = 0, repetitionPenalty: Float = 1, seed: UInt64 = 0
    public var typicalP: Float = 1, mirostatTau: Float = 0, mirostatEta: Float = 0.1
    public var presencePenalty: Float = 0, frequencyPenalty: Float = 0, repetitionWindow: Int32 = 64
    /// token id -> bias added to its raw logit (a large negative value bans a token)
    public var logitBias: [Int32: Float] = [:]
    public init(temperature: Float = 0, topK: Int32 = 0, topP: Float = 1, minP: Float = 0, repetitionPenalty: Float = 1, seed: UInt64 = 0,
                typicalP: Float = 1, mirostatTau: Float = 0, mirostatEta: Float = 0.1, presencePenalty: Float = 0, frequencyPenalty: Float = 0,
                repetitionWindow: Int32 = 64, logitBias: [Int32: Float] = [:]) {
        self.temperature = temperature; self.topK = topK; self.topP = topP; self.minP = minP; self.repetitionPenalty = repetitionPenalty; self.seed = seed
        self.typicalP = typicalP; self.mirostatTau = mirostatTau; self.mirostatEta = mirostatEta
        self.presencePenalty = presencePenalty; self.frequencyPenalty = frequencyPenalty; self.repetitionWindow = repetitionWindow; self.logitBias = logitBias
    }
    /// The ABI 1 struct (the first six settings).
    var c: forge_sampling { forge_sampling(temperature: temperature, top_k: topK, top_p: topP, min_p: minP, repetition_penalty: repetitionPenalty, seed: seed) }
    /// The ABI 4 struct; `args` owns the logit-bias array it points to.
    fileprivate func params(_ args: CArgs) -> forge_sampler_params {
        var p = forge_sampler_params()
        p.size = UInt32(MemoryLayout<forge_sampler_params>.size)
        p.temperature = temperature; p.top_k = topK; p.top_p = topP; p.min_p = minP; p.typical_p = typicalP
        p.mirostat_tau = mirostatTau; p.mirostat_eta = mirostatEta; p.repetition_penalty = repetitionPenalty
        p.presence_penalty = presencePenalty; p.frequency_penalty = frequencyPenalty; p.repetition_window = repetitionWindow; p.seed = seed
        p.logit_bias = args.bias(logitBias); p.n_logit_bias = Int64(logitBias.count)
        return p
    }
}

/// What constrains and ends a generation, besides the sampler.
public struct GenerateOptions {
    /// GBNF grammar text; not together with `jsonSchema`.
    public var grammar: String?
    /// A JSON Schema as text: the output is compact JSON matching it.
    public var jsonSchema: String?
    /// Token ids that end generation like EOS (see `ForgeModel.stopTokens`); the token is not part of the result.
    public var stopTokens: [Int32] = []
    /// EOS does not end generation. Not available with `draft`.
    public var ignoreEOS = false
    /// Clear the KV state first. false (the default, as before ABI 4): continue from the current position.
    public var reset = false
    /// A draft session with the same vocabulary for speculative decoding (both sessions are reset).
    public var draft: ForgeModel?
    public var specK: Int32 = 4
    public init(grammar: String? = nil, jsonSchema: String? = nil, stopTokens: [Int32] = [], ignoreEOS: Bool = false, reset: Bool = false,
                draft: ForgeModel? = nil, specK: Int32 = 4) {
        self.grammar = grammar; self.jsonSchema = jsonSchema; self.stopTokens = stopTokens; self.ignoreEOS = ignoreEOS; self.reset = reset
        self.draft = draft; self.specK = specK
    }
}

/// Library-level helpers.
public enum Forge {
    public static var version: String { String(cString: forge_version()) }
    public static var abi: UInt32 { forge_abi_version() }
    /// GBNF text for a JSON Schema (the subset of docs/inference.md).
    public static func jsonSchemaToGrammar(_ schema: String) throws -> String {
        try cText("forge_json_schema_to_grammar") { buf, cap in forge_json_schema_to_grammar(schema, buf, cap) }
    }
    /// Throws with the parse error when `grammar` is not valid GBNF.
    public static func grammarCheck(_ grammar: String) throws {
        if forge_grammar_check(grammar) != 0 { throw ForgeError.last("grammar") }
    }
    /// Process-wide determinism mode: true / false, or nil to follow FORGE_DETERMINISTIC. Returns the mode in effect.
    @discardableResult public static func setDeterministic(_ mode: Bool?) -> Bool {
        forge_set_deterministic(mode.map { $0 ? 1 : 0 } ?? -1) == 1
    }
    /// Precompile the model's CPU plans into a sidecar ("<path>.fplan" unless `out`); returns the number of plans.
    @discardableResult public static func aotPrecompile(path: String, out: String? = nil, seqLengths: [Int32] = []) throws -> Int {
        let args = CArgs()
        let n = forge_aot_precompile(path, args.str(out), args.ints(seqLengths), Int64(seqLengths.count))
        if n < 0 { throw ForgeError.last("forge_aot_precompile") }
        return Int(n)
    }
}

/// One opened model file: weights, tokenizer, compiled plans, LoRA adapters and the device. Sessions and batches made
/// from it keep the native weights alive, so releasing this object first is allowed.
public final class ForgeWeights {
    fileprivate let w: OpaquePointer
    fileprivate init(handle: OpaquePointer) { w = handle }

    /// `device`: "metal", "cpu", "auto" (scores the devices that open) or anything FORGE_DEVICE accepts; nil follows
    /// FORGE_DEVICE (else the CPU). `deterministic` nil follows FORGE_DETERMINISTIC. `aot` false ignores precompiled plans.
    public init(path: String, verify: Bool = false, device: String? = "metal", deterministic: Bool? = nil, aot: Bool = true) throws {
        let args = CArgs()
        var p = forge_open_params()
        p.size = UInt32(MemoryLayout<forge_open_params>.size)
        p.verify_hashes = verify ? 1 : 0
        p.device = args.str(device)
        p.deterministic = deterministic.map { $0 ? 1 : 2 } ?? 0
        p.no_aot = aot ? 0 : 1
        guard let h = forge_weights_open(path, &p) else { throw ForgeError.last("forge_weights_open") }
        w = h
    }
    deinit { forge_weights_close(w) }

    /// The device in use ("cpu" when the requested one was not available).
    public var device: String { forge_weights_device(w).map { String(cString: $0) } ?? "cpu" }
    /// Select the device; returns the one in use. Not while sessions on these weights are running.
    @discardableResult public func setDevice(_ device: String) throws -> String {
        guard let d = forge_weights_set_device(w, device) else { throw ForgeError.of("forge_weights_set_device", forge_weights_error(w)) }
        return String(cString: d)
    }
    /// A new sequence on these weights. `context` 0 = the model's maximum.
    public func session(context: Int64 = 0, kvCodec: String? = nil, kvOffload: Bool = false) throws -> ForgeModel {
        let args = CArgs()
        var p = forge_session_params()
        p.size = UInt32(MemoryLayout<forge_session_params>.size)
        p.context = context; p.kv_codec = args.str(kvCodec); p.kv_offload = kvOffload ? 1 : 0
        guard let h = forge_session_open(w, &p) else { throw ForgeError.of("forge_session_open", forge_weights_error(w)) }
        return ForgeModel(handle: h)
    }
    public func batch(maxActive: Int32 = 8, context: Int64 = 0, prefillChunk: Int64 = 256, prefixCache: Int32 = 4, kvCodec: String? = nil) throws -> ForgeBatch {
        let args = CArgs()
        var p = forge_batch_params()
        p.size = UInt32(MemoryLayout<forge_batch_params>.size)
        p.max_active = maxActive; p.context = context; p.prefill_chunk = prefillChunk; p.prefix_cache = prefixCache > 0 ? prefixCache : -1
        p.kv_codec = args.str(kvCodec)
        guard let h = forge_batch_open(w, &p) else { throw ForgeError.of("forge_batch_open", forge_weights_error(w)) }
        return ForgeBatch(handle: h)
    }
    /// Load a LoRA adapter (.fpk or .safetensors with <weight>.lora_a / .lora_b). Not while sessions are running.
    public func loadAdapter(name: String, path: String) throws {
        if forge_adapter_load(w, name, path) != 0 { throw ForgeError.of("forge_adapter_load", forge_weights_error(w)) }
    }
    public func hasAdapter(_ name: String) -> Bool { forge_adapter_exists(w, name) == 1 }
}

/// One sequence: KV state and position. Not thread-safe; confine to one thread/actor.
public final class ForgeModel {
    fileprivate let m: OpaquePointer
    fileprivate init(handle: OpaquePointer) { m = handle }
    public static var version: String { String(cString: forge_version()) }
    public static var abi: UInt32 { forge_abi_version() }

    /// Opens a .fpk with private weights. `device`: "metal" (default on Apple GPUs), "cpu", or anything FORGE_DEVICE
    /// accepts; it is written to FORGE_DEVICE (process-wide), as before ABI 4. Use `ForgeWeights` to pick a device per
    /// model or to share the weights between sequences.
    public init(path: String, context: Int64 = 2048, verify: Bool = false, device: String = "metal") throws {
        setenv("FORGE_DEVICE", device, 1)
        guard let h = forge_open(path, context, verify ? 1 : 0) else { throw ForgeError.last("forge_open") }
        m = h
    }
    deinit { forge_close(m) }

    private func fail(_ what: String) -> ForgeError { ForgeError.of(what, forge_model_error(m)) }

    // MARK: model facts
    public var vocabSize: Int { Int(forge_vocab_size(m)) }
    public var position: Int64 { forge_position(m) }
    public var contextSize: Int64 { forge_context_size(m) }
    /// The session kind: "generative_sequence", "embedding", ...
    public var kind: String { forge_session_kind(m).map { String(cString: $0) } ?? "" }
    /// The model file's manifest as JSON text.
    public func manifestJSON() throws -> String { try cText("forge_manifest_json") { buf, cap in forge_manifest_json(self.m, buf, cap) } }
    /// A new handle on this session's weights.
    public func weights() throws -> ForgeWeights {
        guard let h = forge_model_weights(m) else { throw fail("forge_model_weights") }
        return ForgeWeights(handle: h)
    }
    public func reset() { forge_reset(m) }
    public func rewind(to p: Int64) throws { if forge_rewind(m, p) != 0 { throw fail("forge_rewind") } }

    // MARK: tokenizer
    /// addBOS nil: the model's default (no BOS for Qwen-family models, [CLS] .. [SEP] for BERT).
    public func tokenize(_ text: String, addBOS: Bool? = nil) throws -> [Int32] {
        let flag: Int32 = addBOS.map { $0 ? 1 : 0 } ?? -1
        let n = forge_tokenize(m, text, flag, nil, 0)
        if n < 0 { throw fail("forge_tokenize") }
        var out = [Int32](repeating: 0, count: max(Int(n), 1))
        _ = forge_tokenize(m, text, flag, &out, Int64(out.count))
        return Array(out.prefix(Int(n)))
    }
    public func detokenize(_ ids: [Int32]) throws -> String {
        let need = forge_detokenize(m, ids, Int64(ids.count), nil, 0)
        if need < 0 { throw fail("forge_detokenize") }
        var buf = [CChar](repeating: 0, count: Int(need) + 1)
        _ = forge_detokenize(m, ids, Int64(ids.count), &buf, Int64(buf.count))
        return String(decoding: buf.prefix(Int(need)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
    public var bosToken: Int32 { forge_token_bos(m) }
    public var eosToken: Int32 { forge_token_eos(m) }
    public func isSpecial(_ token: Int32) -> Bool { forge_token_is_special(m, token) == 1 }
    /// The id of a vocabulary entry given as stored ("<|im_end|>"), or -1.
    public func tokenID(_ token: String) -> Int32 { forge_token_id(m, token) }
    /// The bytes a token adds to a text (possibly part of a UTF-8 character).
    public func tokenBytes(_ token: Int32) throws -> [UInt8] {
        let need = forge_token_text(m, token, nil, 0)
        if need < 0 { throw fail("forge_token_text") }
        var buf = [CChar](repeating: 0, count: Int(need) + 1)
        _ = forge_token_text(m, token, &buf, Int64(buf.count))
        return buf.prefix(Int(need)).map { UInt8(bitPattern: $0) }
    }
    /// Ids that end an assistant turn: EOS, the manifest's eos_token_ids and (chat) the template's end-of-turn markers.
    public func stopTokens(chat: Bool = true) throws -> [Int32] {
        var out = [Int32](repeating: 0, count: 64)
        let n = forge_stop_tokens(m, chat ? 1 : 0, &out, Int64(out.count))
        if n < 0 { throw fail("forge_stop_tokens") }
        return Array(out.prefix(Int(min(n, 64))))
    }
    /// The model's Jinja chat template, "" when the file has none.
    public func chatTemplate() throws -> String { try cText("forge_chat_template") { buf, cap in forge_chat_template(self.m, buf, cap) } }
    /// Renders the chat template over `messagesJSON` (a JSON array of {role, content}); `toolsJSON` is an OpenAI tools
    /// array, `extraJSON` an object of extra template variables. Tokenize the result with addBOS: false.
    public func applyChatTemplate(messagesJSON: String, toolsJSON: String? = nil, extraJSON: String? = nil, addGenerationPrompt: Bool = true) throws -> String {
        let args = CArgs()
        let tools = args.str(toolsJSON), extra = args.str(extraJSON)
        let need = forge_chat_template_apply_ex(m, messagesJSON, tools, extra, addGenerationPrompt ? 1 : 0, nil, 0)
        if need < 0 { throw fail("forge_chat_template_apply") }
        var buf = [CChar](repeating: 0, count: Int(need) + 1)
        _ = forge_chat_template_apply_ex(m, messagesJSON, tools, extra, addGenerationPrompt ? 1 : 0, &buf, Int64(buf.count))
        return String(decoding: buf.prefix(Int(need)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    // MARK: forward passes
    /// Runs `ids` from the current position; the logits of the last position.
    public func forward(_ ids: [Int32]) throws -> [Float] {
        var lg = [Float](repeating: 0, count: vocabSize)
        if forge_forward(m, ids, Int64(ids.count), &lg, Int64(lg.count)) != 0 { throw fail("forge_forward") }
        return lg
    }
    /// Logits of every position, row-major: ids.count * vocabSize floats.
    public func forwardAll(_ ids: [Int32]) throws -> [Float] {
        var lg = [Float](repeating: 0, count: vocabSize * ids.count)
        if forge_forward_all(m, ids, Int64(ids.count), &lg, Int64(lg.count)) != 0 { throw fail("forge_forward_all") }
        return lg
    }

    // MARK: generation
    /// Why the last `generate` ended.
    public private(set) var lastStopReason: StopReason = .length

    /// The token callback's state: the closure, and (for `stream`) bytes held back until a UTF-8 character is complete.
    private final class Box {
        let f: (Int32, [UInt8]) -> Bool
        init(_ f: @escaping (Int32, [UInt8]) -> Bool) { self.f = f }
    }
    private static let tokenCallback: forge_token_cb = { tok, text, user in
        guard let user = user else { return 0 }
        let box = Unmanaged<Box>.fromOpaque(user).takeUnretainedValue()
        var bytes: [UInt8] = []
        if let text = text { bytes = Array(UnsafeBufferPointer(start: UnsafeRawPointer(text).assumingMemoryBound(to: UInt8.self), count: strlen(text))) }
        return box.f(tok, bytes) ? 1 : 0
    }
    private func run(_ prompt: [Int32], maxNew: Int32, sampling: Sampling, options: GenerateOptions, onBytes: ((Int32, [UInt8]) -> Bool)?) throws -> [Int32] {
        let args = CArgs()
        var sp = sampling.params(args)
        var gp = forge_generate_params()
        gp.size = UInt32(MemoryLayout<forge_generate_params>.size)
        gp.spec_k = options.specK
        gp.grammar = args.str(options.grammar)
        gp.json_schema = args.str(options.jsonSchema)
        gp.draft = options.draft?.m
        gp.stop_tokens = args.ints(options.stopTokens)
        gp.n_stop_tokens = Int64(options.stopTokens.count)
        gp.ignore_eos = options.ignoreEOS ? 1 : 0
        gp.reset = options.reset ? 1 : 0
        let box = onBytes.map(Box.init)
        if let box = box {
            gp.on_token = ForgeModel.tokenCallback
            gp.user = Unmanaged.passUnretained(box).toOpaque()
        }
        var out = [Int32](repeating: 0, count: Int(max(maxNew, 1)))
        var why: Int32 = 0
        let n: Int64 = withUnsafePointer(to: &sp) { spp in
            gp.sampler = spp
            return forge_generate_v4(m, prompt, Int64(prompt.count), maxNew, &gp, &out, Int64(out.count), &why)
        }
        withExtendedLifetime(box) {}
        withExtendedLifetime(args) {}
        withExtendedLifetime(options.draft) {}
        if n < 0 { throw fail("forge_generate") }
        lastStopReason = StopReason(rawValue: why) ?? .length
        return Array(out.prefix(Int(min(n, Int64(out.count)))))
    }

    /// Prefill and decode from the current position (set `options.reset` for a fresh sequence). Returns the new token
    /// ids; since ABI 4 the token that ends generation (EOS, a stop token) is not among them. `onToken` is called for
    /// every token while generation runs and returns true to stop; its text is that token's bytes decoded on their
    /// own (use `stream` for text that is correct across tokens).
    public func generate(_ prompt: [Int32], maxNew: Int32, sampling: Sampling = Sampling(), options: GenerateOptions = GenerateOptions(),
                         onToken: ((Int32, String) -> Bool)? = nil) throws -> [Int32] {
        let onBytes: ((Int32, [UInt8]) -> Bool)? = onToken.map { f in { tok, bytes in f(tok, String(decoding: bytes, as: UTF8.self)) } }
        return try run(prompt, maxNew: maxNew, sampling: sampling, options: options, onBytes: onBytes)
    }
    public func generate(_ prompt: String, maxNew: Int32, sampling: Sampling = Sampling(), options: GenerateOptions = GenerateOptions(),
                         onToken: ((Int32, String) -> Bool)? = nil) throws -> String {
        try detokenize(generate(tokenize(prompt), maxNew: maxNew, sampling: sampling, options: options, onToken: onToken))
    }
    /// The ABI 1 entry point: the first six sampler settings only, and EOS is part of the result.
    public func generateLegacy(_ prompt: [Int32], maxNew: Int32, sampling: Sampling = Sampling(), onToken: ((Int32, String) -> Bool)? = nil) throws -> [Int32] {
        var out = [Int32](repeating: 0, count: Int(max(maxNew, 1)))
        var s = sampling.c
        let box = onToken.map { f in Box { tok, bytes in f(tok, String(decoding: bytes, as: UTF8.self)) } }
        let user = box.map { Unmanaged.passUnretained($0).toOpaque() }
        let n = forge_generate(m, prompt, Int64(prompt.count), maxNew, &s, &out, Int64(out.count), box == nil ? nil : ForgeModel.tokenCallback, user)
        withExtendedLifetime(box) {}
        if n < 0 { throw fail("forge_generate") }
        return Array(out.prefix(Int(min(n, Int64(out.count)))))
    }
    /// Streams text: `onText` gets each piece as soon as its tokens are sampled (a character split over tokens is
    /// held until complete) and returns true to stop. Returns the generated token ids.
    @discardableResult public func stream(_ prompt: [Int32], maxNew: Int32, sampling: Sampling = Sampling(), options: GenerateOptions = GenerateOptions(),
                                          onText: @escaping (String) -> Bool) throws -> [Int32] {
        var pending: [UInt8] = []
        var stopped = false
        let out = try run(prompt, maxNew: maxNew, sampling: sampling, options: options) { _, bytes in
            pending.append(contentsOf: bytes)
            var cut = pending.count   // an incomplete sequence at the end stays for the next token
            for back in 1...max(1, min(4, pending.count)) where back <= pending.count {
                let c = pending[pending.count - back]
                if c & 0xC0 == 0x80 { continue }
                let need = c < 0x80 ? 1 : (c >> 5 == 0b110 ? 2 : (c >> 4 == 0b1110 ? 3 : (c >> 3 == 0b11110 ? 4 : 1)))
                if need > back { cut = pending.count - back }
                break
            }
            if cut > 0 {
                let piece = String(decoding: pending[0..<cut], as: UTF8.self)
                pending.removeFirst(cut)
                if onText(piece) { stopped = true }
            }
            return stopped
        }
        if !pending.isEmpty && !stopped { _ = onText(String(decoding: pending, as: UTF8.self)) }
        return out
    }
    /// One assistant turn: `messagesJSON` through the chat template, generation up to an end-of-turn token.
    public func chat(messagesJSON: String, maxNew: Int32, sampling: Sampling = Sampling(), toolsJSON: String? = nil) throws -> String {
        let ids = try tokenize(applyChatTemplate(messagesJSON: messagesJSON, toolsJSON: toolsJSON), addBOS: false)
        return try detokenize(generate(ids, maxNew: maxNew, sampling: sampling, options: GenerateOptions(stopTokens: stopTokens(chat: true), reset: true)))
    }

    // MARK: LoRA
    /// The selected adapter, "" = base weights.
    public var adapter: String { String(cString: forge_adapter_get(m)) }
    /// Select an adapter loaded on the weights (nil or "" = base). Clears the KV state when the adapter changes.
    public func setAdapter(_ name: String?) throws {
        let args = CArgs()
        if forge_adapter_set(m, args.str(name)) != 0 { throw fail("forge_adapter_set") }
    }

    // MARK: KV state
    /// The KV state of the first `tokens` tokens (-1 = all), for `loadState` on a session of the same model.
    public func saveState(tokens: Int64 = -1) throws -> Data {
        let need = forge_state_save(m, tokens, nil, 0)
        if need < 0 { throw fail("forge_state_save") }
        var data = Data(count: Int(need))
        let wrote: Int64 = data.withUnsafeMutableBytes { forge_state_save(m, tokens, $0.baseAddress, need) }
        if wrote != need { throw fail("forge_state_save") }
        return data
    }
    public func loadState(_ blob: Data) throws {
        let rc: Int32 = blob.withUnsafeBytes { forge_state_load(m, $0.baseAddress, Int64(blob.count)) }
        if rc != 0 { throw fail("forge_state_load") }
    }
    /// Takes the first `tokens` tokens (all by default) of `other` (a session on the same weights, KV codec and adapter).
    public func copyState(from other: ForgeModel, tokens: Int64? = nil) throws {
        if forge_state_copy(m, other.m, tokens ?? other.position) != 0 { throw fail("forge_state_copy") }
    }
    public var stateBytes: Int64 { forge_state_bytes(m) }

    // MARK: embeddings (models whose kind is "embedding")
    public func embedDim() throws -> Int {
        let d = forge_embed_dim(m)
        if d < 0 { throw fail("forge_embed_dim") }
        return Int(d)
    }
    public func embed(_ text: String, pair: String? = nil) throws -> [Float] {
        let dim = try embedDim()
        let args = CArgs()
        var out = [Float](repeating: 0, count: max(dim, 1))
        if forge_embed_text(m, text, args.str(pair), &out, Int64(dim), nil) != Int64(dim) { throw fail("forge_embed_text") }
        return Array(out.prefix(dim))
    }
    public func embed(tokens ids: [Int32], typeIDs: [Int32]? = nil) throws -> [Float] {
        let dim = try embedDim()
        if let t = typeIDs, t.count != ids.count { throw ForgeError(description: "forge_embed_tokens: token types must have one entry per token") }
        let args = CArgs()
        var out = [Float](repeating: 0, count: max(dim, 1))
        if forge_embed_tokens(m, ids, Int64(ids.count), args.ints(typeIDs), &out, Int64(dim)) != Int64(dim) { throw fail("forge_embed_tokens") }
        return Array(out.prefix(dim))
    }
    /// One vector per text, in order.
    public func embed(_ texts: [String]) throws -> [[Float]] {
        let dim = try embedDim()
        if texts.isEmpty { return [] }
        let args = CArgs()
        var ptrs: [UnsafePointer<CChar>?] = texts.map { args.str($0) }
        var out = [Float](repeating: 0, count: max(dim, 1) * texts.count)
        if forge_embed_texts(m, &ptrs, Int64(texts.count), &out, Int64(out.count)) != Int64(dim) { throw fail("forge_embed_texts") }
        return (0..<texts.count).map { Array(out[($0 * dim)..<(($0 + 1) * dim)]) }
    }
}

/// One finished batch request. `reason`: "stop", "length", "grammar", "cancelled" or "error: <message>".
public struct BatchResult {
    public let id: UInt64
    public let tokens: [Int32]
    public let reason: String
}

/// Continuous batching over shared weights: `submit` requests, then `step` or `run`. Callbacks run on the thread
/// that calls step / run / cancel, or when the batch is released. A finished request's result is in `results`.
public final class ForgeBatch {
    private let b: OpaquePointer
    fileprivate init(handle: OpaquePointer) { b = handle }
    /// Requests still waiting or running are cancelled: their onDone callbacks get reason "cancelled".
    deinit { closing = true; forge_batch_close(b) }

    public private(set) var results: [UInt64: BatchResult] = [:]
    public func takeResult(_ id: UInt64) -> BatchResult? { results.removeValue(forKey: id) }

    /// A request's callbacks; retained from submit until its done callback has run.
    private final class Request {
        unowned(unsafe) let batch: ForgeBatch   // unsafe: the done callback also runs from the batch's deinit, where a checked unowned load traps
        let onToken: ((Int32, String) -> Bool)?
        let onDone: ((BatchResult) -> Void)?
        init(_ batch: ForgeBatch, _ onToken: ((Int32, String) -> Bool)?, _ onDone: ((BatchResult) -> Void)?) { self.batch = batch; self.onToken = onToken; self.onDone = onDone }
    }
    private static let tokenCallback: forge_token_cb = { tok, text, user in
        guard let user = user else { return 0 }
        let r = Unmanaged<Request>.fromOpaque(user).takeUnretainedValue()
        guard let f = r.onToken else { return 0 }
        return f(tok, text.map { String(cString: $0) } ?? "") ? 1 : 0
    }
    private static let doneCallback: forge_done_cb = { id, tokens, n, reason, user in
        guard let user = user else { return }
        let r = Unmanaged<Request>.fromOpaque(user).takeRetainedValue()   // balances passRetained in submit
        let toks = tokens.map { Array(UnsafeBufferPointer(start: $0, count: Int(n))) } ?? []
        let result = BatchResult(id: id, tokens: toks, reason: reason.map { String(cString: $0) } ?? "")
        // during deinit the batch is going away: only the callback is told
        if !r.batch.closing { r.batch.results[id] = result }
        r.onDone?(result)
    }
    private var closing = false

    /// Queues a request; returns its id. `adapter` is a LoRA adapter loaded on the weights.
    @discardableResult public func submit(_ prompt: [Int32], maxNew: Int32, sampling: Sampling = Sampling(), grammar: String? = nil, jsonSchema: String? = nil,
                                          stopTokens: [Int32] = [], adapter: String? = nil, priority: Int32 = 0,
                                          onToken: ((Int32, String) -> Bool)? = nil, onDone: ((BatchResult) -> Void)? = nil) throws -> UInt64 {
        let args = CArgs()
        var sp = sampling.params(args)
        var rq = forge_batch_request()
        rq.size = UInt32(MemoryLayout<forge_batch_request>.size)
        rq.max_new = maxNew
        rq.prompt = args.ints(prompt); rq.n_prompt = Int64(prompt.count)
        rq.grammar = args.str(grammar); rq.json_schema = args.str(jsonSchema)
        rq.stop_tokens = args.ints(stopTokens); rq.n_stop_tokens = Int64(stopTokens.count)
        rq.adapter = args.str(adapter); rq.priority = priority
        let request = Unmanaged.passRetained(Request(self, onToken, onDone))
        if onToken != nil { rq.on_token = ForgeBatch.tokenCallback }
        rq.on_done = ForgeBatch.doneCallback
        rq.user = request.toOpaque()
        let id: UInt64 = withUnsafePointer(to: &sp) { spp in
            rq.sampler = spp
            return forge_batch_submit(b, &rq)   // copies the prompt, grammar, stop tokens and sampler
        }
        withExtendedLifetime(args) {}
        if id == 0 { request.release(); throw ForgeError.of("forge_batch_submit", forge_batch_error(b)) }   // not accepted: no done callback
        return id
    }
    /// One scheduling step; false when nothing is waiting or running.
    @discardableResult public func step() throws -> Bool {
        let r = forge_batch_step(b)
        if r < 0 { throw ForgeError.of("forge_batch_step", forge_batch_error(b)) }
        return r == 1
    }
    /// Steps until idle.
    public func run() throws { if forge_batch_run(b) != 0 { throw ForgeError.of("forge_batch_run", forge_batch_error(b)) } }
    @discardableResult public func cancel(_ id: UInt64) -> Bool { forge_batch_cancel(b, id) == 1 }
    /// steps, completed, failed, prefixHits, prefixTokensReused, peakActive, peakKVBytes, waiting, active
    public func stats() throws -> [String: UInt64] {
        var s = forge_batch_stats()
        s.size = UInt32(MemoryLayout<forge_batch_stats>.size)
        if forge_batch_stats_get(b, &s) != 0 { throw ForgeError.of("forge_batch_stats_get", forge_batch_error(b)) }
        return ["steps": s.steps, "completed": s.completed, "failed": s.failed, "prefixHits": s.prefix_hits, "prefixTokensReused": s.prefix_tokens_reused,
                "peakActive": s.peak_active, "peakKVBytes": s.peak_kv_bytes, "waiting": s.waiting, "active": s.active]
    }
    public func evictPrefixCache(keep: Int64 = 0) { forge_batch_evict_prefix_cache(b, keep) }

    /// Runs several prompts together and returns their tokens in order; a failed request throws.
    public func generateMany(_ prompts: [[Int32]], maxNew: Int32, sampling: Sampling = Sampling()) throws -> [[Int32]] {
        let ids = try prompts.map { try submit($0, maxNew: maxNew, sampling: sampling) }
        try run()
        return try ids.map { id in
            guard let r = takeResult(id) else { throw ForgeError(description: "request \(id) did not finish") }
            if r.reason.hasPrefix("error: ") { throw ForgeError(description: String(r.reason.dropFirst(7))) }
            return r.tokens
        }
    }
}

package dev.forge

import java.nio.charset.CodingErrorAction

/*
 * Kotlin binding over FORGE's C ABI (forge.h, ABI 4) via libforge_jni.so.
 *
 *   ForgeWeights   one opened model file (weights, tokenizer, adapters, device); shared by sessions and batches
 *   ForgeModel     one sequence (KV state); ForgeModel.open(path) owns private weights, weights.session() shares
 *   ForgeBatch     continuous batching over shared weights
 *   Forge          library-level helpers (version, JSON Schema -> grammar, determinism, AOT sidecar)
 *
 * Thread safety follows the C ABI (docs/capi.md): one thread at a time per object; different objects may be used
 * from different threads.
 *
 * NEVER COMPILED HERE (no Kotlin compiler or Android SDK where this was written); see docs/bindings.md.
 */

private fun String.utf8(): ByteArray = toByteArray(Charsets.UTF_8)
private fun ByteArray.text(): String = String(this, Charsets.UTF_8)   // malformed input becomes U+FFFD

/** Decodes a token stream's bytes to text, holding back a UTF-8 character that is split over tokens. */
private class Utf8Stream {
    private val decoder = Charsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPLACE).onUnmappableCharacter(CodingErrorAction.REPLACE)
    private var pending = ByteArray(0)
    fun feed(bytes: ByteArray): String {
        val all = pending + bytes
        var cut = all.size
        for (back in 1..minOf(4, all.size)) {   // an incomplete sequence at the end stays for the next token
            val c = all[all.size - back].toInt() and 0xFF
            if (c and 0xC0 == 0x80) continue
            val need = when { c < 0x80 -> 1; c shr 5 == 0b110 -> 2; c shr 4 == 0b1110 -> 3; c shr 3 == 0b11110 -> 4; else -> 1 }
            if (need > back) cut = all.size - back
            break
        }
        pending = all.copyOfRange(cut, all.size)
        return decoder.decode(java.nio.ByteBuffer.wrap(all, 0, cut)).toString()
    }
    fun finish(): String { val s = pending.text(); pending = ByteArray(0); return s }
}

/** Why a generation ended ([ForgeModel.lastStopReason]). */
enum class StopReason { LENGTH, EOS, CALLBACK, GRAMMAR }

/** What constrains and ends a generation, besides the sampler. */
data class GenerateOptions(
    /** GBNF grammar text; not together with [jsonSchema]. */
    val grammar: String? = null,
    /** A JSON Schema as text: the output is compact JSON matching it. */
    val jsonSchema: String? = null,
    /** Token ids that end generation like EOS (see [ForgeModel.stopTokens]); the token is not part of the result. */
    val stopTokens: IntArray? = null,
    /** EOS does not end generation. Not available with [draft]. */
    val ignoreEos: Boolean = false,
    /** Clear the KV state first. false (the default, as before ABI 4): continue from the current position. */
    val reset: Boolean = false,
    /** A draft session with the same vocabulary for speculative decoding (both sessions are reset). */
    val draft: ForgeModel? = null,
    val specK: Int = 4,
)

/** Library-level helpers. */
object Forge {
    @JvmStatic val version: String get() = ForgeNative.version()
    @JvmStatic val abi: Int get() = ForgeNative.abi()
    /** GBNF text for a JSON Schema (the subset of docs/inference.md). Throws IllegalStateException on a bad schema. */
    @JvmStatic fun jsonSchemaToGrammar(schema: String): String = ForgeNative.jsonSchemaToGrammar(schema.utf8()).text()
    /** Throws IllegalStateException with the parse error when [grammar] is not valid GBNF. */
    @JvmStatic fun grammarCheck(grammar: String) = ForgeNative.grammarCheck(grammar.utf8())
    /** Process-wide determinism mode: true / false, or null to follow FORGE_DETERMINISTIC. Returns the mode in effect. */
    @JvmStatic fun setDeterministic(mode: Boolean?): Boolean = ForgeNative.setDeterministic(if (mode == null) -1 else if (mode) 1 else 0) == 1
    /** Precompile the model's CPU plans into a sidecar ("<path>.fplan" unless [out]); returns the number of plans. */
    @JvmStatic fun aotPrecompile(path: String, out: String? = null, seqLengths: IntArray? = null): Long =
        ForgeNative.aotPrecompile(path.utf8(), out?.utf8(), seqLengths)
}

/**
 * One opened model file: weights, tokenizer, compiled plans, LoRA adapters and the device. Sessions and batches made
 * from it keep the native weights alive, so closing this object first is allowed.
 */
class ForgeWeights private constructor(private var handle: Long) : AutoCloseable {
    internal fun live(): Long { check(handle != 0L) { "ForgeWeights is closed" }; return handle }
    override fun close() { if (handle != 0L) { ForgeNative.weightsClose(handle); handle = 0L } }

    /** The device in use ("cpu" when the requested one was not available). */
    val device: String get() = ForgeNative.weightsDevice(live())
    /** Select the device ("cpu", "auto", "vulkan", ...); returns the one in use. Not while sessions on these weights run. */
    fun setDevice(device: String): String = ForgeNative.weightsSetDevice(live(), device.utf8())

    /** A new sequence on these weights. [context] 0 = the model's maximum. */
    fun session(context: Long = 0, kvCodec: String? = null, kvOffload: Boolean = false): ForgeModel =
        ForgeModel(ForgeNative.sessionOpen(live(), context, kvCodec?.utf8(), kvOffload))
    fun batch(maxActive: Int = 8, context: Long = 0, prefillChunk: Long = 256, prefixCache: Int = 4, kvCodec: String? = null): ForgeBatch =
        ForgeBatch(ForgeNative.batchOpen(live(), maxActive, context, prefillChunk, if (prefixCache > 0) prefixCache else -1, kvCodec?.utf8()))

    /** Load a LoRA adapter (.fpk or .safetensors with <weight>.lora_a / .lora_b). Not while sessions are running. */
    fun loadAdapter(name: String, path: String) = ForgeNative.adapterLoad(live(), name.utf8(), path.utf8())
    fun hasAdapter(name: String): Boolean = ForgeNative.adapterExists(live(), name.utf8())

    companion object {
        /**
         * Opens a .fpk. [device] null follows FORGE_DEVICE (else the CPU); "auto" scores the devices that open.
         * [deterministic] null follows FORGE_DETERMINISTIC. [aot] false ignores precompiled plans in the file.
         */
        @JvmStatic fun open(path: String, verify: Boolean = false, device: String? = null, deterministic: Boolean? = null, aot: Boolean = true): ForgeWeights =
            ForgeWeights(ForgeNative.weightsOpen(path.utf8(), verify, device?.utf8(), if (deterministic == null) 0 else if (deterministic) 1 else 2, !aot))
        internal fun wrap(handle: Long) = ForgeWeights(handle)
    }
}

/** One sequence: KV state and position. Not thread-safe: use it from one thread at a time. */
class ForgeModel internal constructor(private var handle: Long) : AutoCloseable {
    fun interface TokenListener { /** Return true to stop generation. */ fun onToken(token: Int, text: String): Boolean }

    /**
     * The sampler chain: logit bias -> repetition / presence / frequency penalties -> grammar -> temperature -> top-k ->
     * top-p -> min-p -> typical -> draw (Mirostat v2 replaces the truncation steps when [mirostatTau] > 0).
     * The first six parameters are the ones this class had before ABI 4.
     */
    data class Sampling(
        val temperature: Float = 0f, val topK: Int = 0, val topP: Float = 1f, val minP: Float = 0f,
        val repetitionPenalty: Float = 1f, val seed: Long = 0,
        val typicalP: Float = 1f, val mirostatTau: Float = 0f, val mirostatEta: Float = 0.1f,
        val presencePenalty: Float = 0f, val frequencyPenalty: Float = 0f, val repetitionWindow: Int = 64,
        /** token id -> bias added to its raw logit (a large negative value bans a token). */
        val logitBias: Map<Int, Float> = emptyMap(),
    ) {
        /** The float block forge_jni.c's sampler_fill reads (the order is fixed there). */
        internal fun pack(): FloatArray = floatArrayOf(temperature, topP, minP, typicalP, mirostatTau, mirostatEta, repetitionPenalty,
            presencePenalty, frequencyPenalty, topK.toFloat(), repetitionWindow.toFloat())
        internal fun biasTokens(): IntArray? = if (logitBias.isEmpty()) null else logitBias.keys.toIntArray()
        internal fun biasValues(): FloatArray? = if (logitBias.isEmpty()) null else logitBias.keys.map { logitBias.getValue(it) }.toFloatArray()
    }

    internal fun live(): Long { check(handle != 0L) { "ForgeModel is closed" }; return handle }
    override fun close() { if (handle != 0L) { ForgeNative.close(handle); handle = 0L } }

    // ---- model facts
    val vocabSize: Long get() = ForgeNative.vocab(live())
    val position: Long get() = ForgeNative.position(live())
    val contextSize: Long get() = ForgeNative.contextSize(live())
    /** The session kind: "generative_sequence", "embedding", ... */
    val kind: String get() = ForgeNative.kind(live())
    /** The model file's manifest as JSON text. */
    val manifestJson: String get() = ForgeNative.manifest(live()).text()
    /** A new handle on this session's weights (close it too). */
    fun weights(): ForgeWeights = ForgeWeights.wrap(ForgeNative.modelWeights(live()))
    fun reset() = ForgeNative.reset(live())
    fun rewind(position: Long) = ForgeNative.rewind(live(), position)

    // ---- tokenizer
    /** [addBos] true / false, or null for the model's default (no BOS for Qwen-family models, [CLS] .. [SEP] for BERT). */
    fun tokenize(text: String, addBos: Boolean? = true): IntArray = ForgeNative.tokenize(live(), text.utf8(), if (addBos == null) -1 else if (addBos) 1 else 0)
    fun detokenize(ids: IntArray): String = ForgeNative.detokenize(live(), ids).text()
    val bosToken: Int get() = ForgeNative.tokenBos(live())
    val eosToken: Int get() = ForgeNative.tokenEos(live())
    fun isSpecial(token: Int): Boolean = ForgeNative.tokenIsSpecial(live(), token)
    /** The id of a vocabulary entry given as stored ("<|im_end|>"), or -1. */
    fun tokenId(token: String): Int = ForgeNative.tokenId(live(), token.utf8())
    /** The bytes a token adds to a text (possibly part of a UTF-8 character). */
    fun tokenBytes(token: Int): ByteArray = ForgeNative.tokenBytes(live(), token)
    /** Ids that end an assistant turn: EOS, the manifest's eos_token_ids and ([chat]) the template's end-of-turn markers. */
    fun stopTokens(chat: Boolean = true): IntArray = ForgeNative.stopTokens(live(), chat)
    /** The model's Jinja chat template, "" when the file has none. */
    val chatTemplate: String get() = ForgeNative.chatTemplate(live()).text()
    /**
     * Renders the chat template over [messagesJson] (a JSON array of {role, content}); [toolsJson] is an OpenAI tools
     * array and [extraJson] an object of extra template variables. Tokenize the result with addBos = false.
     */
    fun applyChatTemplate(messagesJson: String, toolsJson: String? = null, extraJson: String? = null, addGenerationPrompt: Boolean = true): String =
        ForgeNative.applyChatTemplate(live(), messagesJson.utf8(), toolsJson?.utf8(), extraJson?.utf8(), addGenerationPrompt).text()

    // ---- forward passes
    /** Runs [ids] from the current position; the logits of the last position (vocabSize floats). */
    fun forward(ids: IntArray): FloatArray = ForgeNative.forward(live(), ids)
    /** Logits of every position, row-major: ids.size * vocabSize floats. */
    fun forwardAll(ids: IntArray): FloatArray = ForgeNative.forwardAll(live(), ids)

    // ---- generation
    /** Why the last [generate] ended. */
    var lastStopReason: StopReason = StopReason.LENGTH
        private set

    /**
     * Prefill and decode from the current position (call [reset], or use the overload with [GenerateOptions.reset], for
     * a fresh sequence). Returns the new token ids; since ABI 4 the token that ends generation (EOS, a stop token) is
     * not among them. [listener] is called for every token while generation runs; its text is that token's bytes
     * decoded on their own (use [generateText] or [stream] for text that is correct across tokens).
     */
    fun generate(prompt: IntArray, maxNew: Int, sampling: Sampling = Sampling(), listener: TokenListener? = null): IntArray =
        generate(prompt, maxNew, sampling, GenerateOptions(), listener)
    /** As above, with a grammar or JSON Schema, stop tokens, a draft model and the other [GenerateOptions]. */
    fun generate(prompt: IntArray, maxNew: Int, sampling: Sampling, options: GenerateOptions, listener: TokenListener? = null): IntArray {
        val sink = listener?.let { l -> ForgeNative.TokenSink { token, bytes -> l.onToken(token, bytes.text()) } }
        return run(prompt, maxNew, sampling, options, sink)
    }
    private fun run(prompt: IntArray, maxNew: Int, sampling: Sampling, options: GenerateOptions, sink: ForgeNative.TokenSink?): IntArray {
        val why = IntArray(1)
        val out = ForgeNative.generateV4(live(), prompt, maxNew, sampling.pack(), sampling.seed, sampling.biasTokens(), sampling.biasValues(),
            options.grammar?.utf8(), options.jsonSchema?.utf8(), options.draft?.live() ?: 0L, options.specK, options.stopTokens,
            options.ignoreEos, options.reset, sink, why)
        lastStopReason = StopReason.values().getOrElse(why[0]) { StopReason.LENGTH }
        return out
    }
    /** The ABI 1 entry point: the first six sampler settings only, and EOS is part of the result. */
    fun generateLegacy(prompt: IntArray, maxNew: Int, sampling: Sampling = Sampling(), listener: TokenListener? = null): IntArray {
        val sink = listener?.let { l -> ForgeNative.TokenSink { token, bytes -> l.onToken(token, bytes.text()) } }
        return ForgeNative.generate(live(), prompt, maxNew, sampling.temperature, sampling.topK, sampling.topP, sampling.minP,
            sampling.repetitionPenalty, sampling.seed, sink)
    }
    fun generateText(prompt: String, maxNew: Int, sampling: Sampling = Sampling(), listener: TokenListener? = null): String =
        detokenize(generate(tokenize(prompt), maxNew, sampling, GenerateOptions(), listener))
    fun generateText(prompt: String, maxNew: Int, sampling: Sampling, options: GenerateOptions, listener: TokenListener? = null): String =
        detokenize(generate(tokenize(prompt), maxNew, sampling, options, listener))

    /**
     * Streams text: [onText] gets each piece as soon as its tokens are sampled (a character split over tokens is held
     * until complete) and returns true to stop. Returns the generated token ids.
     */
    fun stream(prompt: IntArray, maxNew: Int, sampling: Sampling = Sampling(), options: GenerateOptions = GenerateOptions(),
               onText: (String) -> Boolean): IntArray {
        val utf8 = Utf8Stream()
        var stop = false
        val out = run(prompt, maxNew, sampling, options, ForgeNative.TokenSink { _, bytes ->
            val piece = utf8.feed(bytes)
            if (piece.isNotEmpty() && onText(piece)) stop = true
            stop
        })
        val rest = utf8.finish()
        if (rest.isNotEmpty() && !stop) onText(rest)
        return out
    }

    /** One assistant turn: [messagesJson] through the chat template, generation up to an end-of-turn token. */
    fun chat(messagesJson: String, maxNew: Int, sampling: Sampling = Sampling(), toolsJson: String? = null): String {
        val ids = tokenize(applyChatTemplate(messagesJson, toolsJson), false)
        return detokenize(generate(ids, maxNew, sampling, GenerateOptions(stopTokens = stopTokens(true), reset = true)))
    }

    // ---- LoRA
    /** The selected adapter, "" = base weights. Assigning "" selects the base; a change clears the KV state. */
    var adapter: String
        get() = ForgeNative.adapterGet(live()).text()
        set(name) = ForgeNative.adapterSet(live(), if (name.isEmpty()) null else name.utf8())

    // ---- KV state
    /** The KV state of the first [nTokens] tokens (-1 = all), for [loadState] on a session of the same model. */
    fun saveState(nTokens: Long = -1): ByteArray = ForgeNative.stateSave(live(), nTokens)
    fun loadState(blob: ByteArray) = ForgeNative.stateLoad(live(), blob)
    /** Takes the first [nTokens] tokens of [other] (a session on the same weights, KV codec and adapter). */
    fun copyStateFrom(other: ForgeModel, nTokens: Long = other.position) = ForgeNative.stateCopy(live(), other.live(), nTokens)
    val stateBytes: Long get() = ForgeNative.stateBytes(live())

    // ---- embeddings (models whose kind is "embedding")
    val embedDim: Long get() = ForgeNative.embedDim(live())
    fun embed(text: String, pair: String? = null): FloatArray = ForgeNative.embedText(live(), text.utf8(), pair?.utf8())
    fun embed(ids: IntArray, typeIds: IntArray? = null): FloatArray = ForgeNative.embedTokens(live(), ids, typeIds)
    /** One vector per text, in order. */
    fun embed(texts: List<String>): List<FloatArray> {
        val dim = embedDim.toInt()
        val flat = ForgeNative.embedTexts(live(), Array(texts.size) { texts[it].utf8() })
        return List(texts.size) { flat.copyOfRange(it * dim, (it + 1) * dim) }
    }

    companion object {
        /** Opens a .fpk with private weights. Use [ForgeWeights.open] to choose a device or to share the weights. */
        @JvmStatic fun open(path: String, context: Long = 2048, verify: Boolean = false) = ForgeModel(ForgeNative.open(path.utf8(), context, verify))
        @JvmStatic val version: String get() = ForgeNative.version()
        @JvmStatic val abi: Int get() = ForgeNative.abi()
    }
}

/** One finished batch request. [reason]: "stop", "length", "grammar", "cancelled" or "error: <message>". */
data class BatchResult(val id: Long, val tokens: IntArray, val reason: String)

/**
 * Continuous batching over shared weights: [submit] requests, then [step] or [run]. Listeners run on the thread that
 * calls step / run / cancel / close. A finished request's result is in [results] until taken.
 */
class ForgeBatch internal constructor(private var handle: Long) : AutoCloseable {
    private fun live(): Long { check(handle != 0L) { "ForgeBatch is closed" }; return handle }
    /** Requests still waiting or running are cancelled (their results get reason "cancelled"). */
    override fun close() { if (handle != 0L) { ForgeNative.batchClose(handle); handle = 0L } }

    val results: MutableMap<Long, BatchResult> = HashMap()

    /** Queues a request; returns its id. [adapter] is a LoRA adapter loaded on the weights. */
    fun submit(prompt: IntArray, maxNew: Int, sampling: ForgeModel.Sampling = ForgeModel.Sampling(), grammar: String? = null, jsonSchema: String? = null,
               stopTokens: IntArray? = null, adapter: String? = null, priority: Int = 0,
               listener: ForgeModel.TokenListener? = null, onDone: ((BatchResult) -> Unit)? = null): Long {
        val sink = listener?.let { l -> ForgeNative.TokenSink { token, bytes -> l.onToken(token, bytes.text()) } }
        val done = ForgeNative.DoneSink { id, tokens, reason ->
            val r = BatchResult(id, tokens, reason.text())
            results[id] = r
            onDone?.invoke(r)
        }
        return ForgeNative.batchSubmit(live(), prompt, maxNew, sampling.pack(), sampling.seed, sampling.biasTokens(), sampling.biasValues(),
            grammar?.utf8(), jsonSchema?.utf8(), stopTokens, adapter?.utf8(), priority, sink, done)
    }
    /** One scheduling step; false when nothing is waiting or running. */
    fun step(): Boolean = ForgeNative.batchStep(live())
    /** Steps until idle. */
    fun run() = ForgeNative.batchRun(live())
    fun cancel(id: Long): Boolean = ForgeNative.batchCancel(live(), id)
    /** steps, completed, failed, prefixHits, prefixTokensReused, peakActive, peakKvBytes, waiting, active. */
    fun stats(): Map<String, Long> {
        val v = ForgeNative.batchStats(live())
        val names = listOf("steps", "completed", "failed", "prefixHits", "prefixTokensReused", "peakActive", "peakKvBytes", "waiting", "active")
        return names.indices.associate { names[it] to v[it] }
    }
    fun evictPrefixCache(keep: Long = 0) = ForgeNative.batchEvictPrefixCache(live(), keep)

    /** Runs several prompts together and returns their tokens in order; a failed request throws. */
    fun generateMany(prompts: List<IntArray>, maxNew: Int, sampling: ForgeModel.Sampling = ForgeModel.Sampling()): List<IntArray> {
        val ids = prompts.map { submit(it, maxNew, sampling) }
        run()
        return ids.map { id ->
            val r = results.remove(id) ?: error("request $id did not finish")
            check(!r.reason.startsWith("error: ")) { r.reason.removePrefix("error: ") }
            r.tokens
        }
    }
}

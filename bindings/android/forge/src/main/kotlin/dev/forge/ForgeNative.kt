package dev.forge

/**
 * The JNI surface of libforge_jni (bindings/android/jni/forge_jni.c): one external per C ABI call (forge.h, ABI 4).
 * Internal: use [ForgeWeights], [ForgeModel], [ForgeBatch] and [Forge].
 *
 * Text crosses as UTF-8 byte arrays (JNI strings are "modified UTF-8", which cannot carry characters outside the BMP
 * as real UTF-8, and a token's text may be part of a character). Handles are Long; 0 is never a valid handle.
 *
 * NEVER COMPILED HERE: no Kotlin compiler or Android SDK in the environment this was written in. The native side is
 * compiled, and run on a host JVM against a Java declaration of the same methods (ctest `jni_smoke`,
 * tests/jni/dev/forge/ForgeNative.java). Keep the two declarations identical.
 */
internal object ForgeNative {
    init { System.loadLibrary("forge_jni") }

    /** Called by the native token callback: the token and its bytes; true stops. */
    fun interface TokenSink { fun onTokenBytes(token: Int, text: ByteArray): Boolean }
    /** Called once when a batch request ends: its id, tokens and the reason as UTF-8 bytes. */
    fun interface DoneSink { fun onDoneBytes(id: Long, tokens: IntArray, reason: ByteArray) }

    // library
    @JvmStatic external fun version(): String
    @JvmStatic external fun abi(): Int
    @JvmStatic external fun setDeterministic(mode: Int): Int
    @JvmStatic external fun jsonSchemaToGrammar(schema: ByteArray): ByteArray
    @JvmStatic external fun grammarCheck(grammar: ByteArray)
    @JvmStatic external fun aotPrecompile(path: ByteArray, out: ByteArray?, lengths: IntArray?): Long

    // weights
    @JvmStatic external fun weightsOpen(path: ByteArray, verify: Boolean, device: ByteArray?, deterministic: Int, noAot: Boolean): Long
    @JvmStatic external fun weightsClose(w: Long)
    @JvmStatic external fun modelWeights(h: Long): Long
    @JvmStatic external fun weightsDevice(w: Long): String
    @JvmStatic external fun weightsSetDevice(w: Long, device: ByteArray): String
    @JvmStatic external fun adapterLoad(w: Long, name: ByteArray, path: ByteArray)
    @JvmStatic external fun adapterExists(w: Long, name: ByteArray): Boolean

    // sessions
    @JvmStatic external fun open(path: ByteArray, ctx: Long, verify: Boolean): Long
    @JvmStatic external fun sessionOpen(w: Long, ctx: Long, kvCodec: ByteArray?, kvOffload: Boolean): Long
    @JvmStatic external fun close(h: Long)
    @JvmStatic external fun reset(h: Long)
    @JvmStatic external fun vocab(h: Long): Long
    @JvmStatic external fun position(h: Long): Long
    @JvmStatic external fun contextSize(h: Long): Long
    @JvmStatic external fun rewind(h: Long, pos: Long)
    @JvmStatic external fun kind(h: Long): String
    @JvmStatic external fun manifest(h: Long): ByteArray
    @JvmStatic external fun chatTemplate(h: Long): ByteArray
    @JvmStatic external fun applyChatTemplate(h: Long, messages: ByteArray, tools: ByteArray?, extra: ByteArray?, addGenerationPrompt: Boolean): ByteArray

    // tokenizer
    @JvmStatic external fun tokenize(h: Long, text: ByteArray, addBos: Int): IntArray
    @JvmStatic external fun detokenize(h: Long, ids: IntArray): ByteArray
    @JvmStatic external fun tokenBos(h: Long): Int
    @JvmStatic external fun tokenEos(h: Long): Int
    @JvmStatic external fun tokenIsSpecial(h: Long, id: Int): Boolean
    @JvmStatic external fun tokenId(h: Long, token: ByteArray): Int
    @JvmStatic external fun tokenBytes(h: Long, id: Int): ByteArray
    @JvmStatic external fun stopTokens(h: Long, chat: Boolean): IntArray

    // forward passes and generation
    @JvmStatic external fun forward(h: Long, ids: IntArray): FloatArray
    @JvmStatic external fun forwardAll(h: Long, ids: IntArray): FloatArray
    @JvmStatic external fun generate(h: Long, prompt: IntArray, maxNew: Int, temp: Float, topK: Int, topP: Float, minP: Float, rep: Float,
        seed: Long, sink: TokenSink?): IntArray
    /** sampler: [Sampling.pack]. stopReasonOut: IntArray(1) or null. */
    @JvmStatic external fun generateV4(h: Long, prompt: IntArray, maxNew: Int, sampler: FloatArray, seed: Long, biasTokens: IntArray?, biasValues: FloatArray?,
        grammar: ByteArray?, jsonSchema: ByteArray?, draft: Long, specK: Int, stopTokens: IntArray?, ignoreEos: Boolean, reset: Boolean,
        sink: TokenSink?, stopReasonOut: IntArray?): IntArray

    // adapters, KV state
    @JvmStatic external fun adapterSet(h: Long, name: ByteArray?)
    @JvmStatic external fun adapterGet(h: Long): ByteArray
    @JvmStatic external fun stateSave(h: Long, nTokens: Long): ByteArray
    @JvmStatic external fun stateLoad(h: Long, blob: ByteArray)
    @JvmStatic external fun stateCopy(dst: Long, src: Long, nTokens: Long)
    @JvmStatic external fun stateBytes(h: Long): Long

    // embeddings
    @JvmStatic external fun embedDim(h: Long): Long
    @JvmStatic external fun embedTokens(h: Long, ids: IntArray, typeIds: IntArray?): FloatArray
    @JvmStatic external fun embedText(h: Long, text: ByteArray, pair: ByteArray?): FloatArray
    @JvmStatic external fun embedTexts(h: Long, texts: Array<ByteArray>): FloatArray

    // batching
    @JvmStatic external fun batchOpen(w: Long, maxActive: Int, ctx: Long, prefillChunk: Long, prefixCache: Int, kvCodec: ByteArray?): Long
    @JvmStatic external fun batchClose(b: Long)
    @JvmStatic external fun batchSubmit(b: Long, prompt: IntArray, maxNew: Int, sampler: FloatArray, seed: Long, biasTokens: IntArray?, biasValues: FloatArray?,
        grammar: ByteArray?, jsonSchema: ByteArray?, stopTokens: IntArray?, adapter: ByteArray?, priority: Int, sink: TokenSink?, done: DoneSink?): Long
    @JvmStatic external fun batchCancel(b: Long, id: Long): Boolean
    @JvmStatic external fun batchStep(b: Long): Boolean
    @JvmStatic external fun batchRun(b: Long)
    @JvmStatic external fun batchStats(b: Long): LongArray
    @JvmStatic external fun batchEvictPrefixCache(b: Long, keep: Long)
}

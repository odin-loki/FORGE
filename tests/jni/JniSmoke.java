import dev.forge.ForgeNative;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Host-JVM run of the JNI bridge (bindings/android/jni/forge_jni.c) through the Java mirror of the Kotlin natives:
 * every native is called at least once, with -Xcheck:jni. It tests the C side and the declared signatures; the
 * Kotlin wrapper classes (ForgeModel.kt) are not compiled or run by it.
 *
 *   java -Xcheck:jni -Djava.library.path=<dir of libforge_jni> -cp <classes> JniSmoke <tiny_f32.fpk> <bert.fpk> <work dir>
 */
public final class JniSmoke {
    static int fails = 0;
    static void check(boolean ok, String what) { if (!ok) { System.out.println("  CHECK failed: " + what); ++fails; } }
    static byte[] b(String s) { return s.getBytes(StandardCharsets.UTF_8); }
    static String s(byte[] x) { return new String(x, StandardCharsets.UTF_8); }
    interface Call { void run(); }
    /** The call must throw IllegalStateException whose message contains `needle`. */
    static void fails(String needle, Call c) {
        try { c.run(); check(false, "expected an exception with '" + needle + "'"); }
        catch (IllegalStateException e) { check(e.getMessage().contains(needle), "message '" + e.getMessage() + "' lacks '" + needle + "'"); }
    }
    // sampler block as Sampling.pack(): temperature, topP, minP, typicalP, mirostatTau, mirostatEta, repetitionPenalty,
    // presencePenalty, frequencyPenalty, topK, repetitionWindow
    static float[] sampler(float temperature, int topK) { return new float[]{temperature, 1f, 0f, 1f, 0f, 0.1f, 1f, 0f, 0f, topK, 64}; }
    static final float[] GREEDY = sampler(0f, 0);

    static int[] gen(long h, int[] prompt, int maxNew, float[] sampler, long seed, int[] biasTok, float[] biasVal, String grammar, String schema,
                     long draft, int[] stops, boolean ignoreEos, ForgeNative.TokenSink sink, int[] why) {
        return ForgeNative.generateV4(h, prompt, maxNew, sampler, seed, biasTok, biasVal, grammar == null ? null : b(grammar), schema == null ? null : b(schema),
                                      draft, 3, stops, ignoreEos, true, sink, why);
    }

    public static void main(String[] args) throws Exception {
        String fpk = args[0], bert = args[1], work = args[2];
        check(ForgeNative.abi() >= 4 && ForgeNative.version().equals("0.1.0"), "version / abi");
        fails("nonexistent", () -> ForgeNative.open(b("/nonexistent.fpk"), 0, false));
        fails("nonexistent", () -> ForgeNative.weightsOpen(b("/nonexistent.fpk"), false, null, 0, false));

        // ---- weights, sessions, facts
        long w = ForgeNative.weightsOpen(b(fpk), true, b("cpu"), 1, false);
        check(ForgeNative.weightsDevice(w).equals("cpu") && ForgeNative.weightsSetDevice(w, b("no-such-device")).equals("cpu"), "device");
        check(ForgeNative.setDeterministic(1) == 1 && ForgeNative.setDeterministic(0) == 0 && ForgeNative.setDeterministic(-1) >= 0, "determinism");
        long a = ForgeNative.sessionOpen(w, 64, b("f32"), false), d = ForgeNative.sessionOpen(w, 0, null, false);
        long own = ForgeNative.open(b(fpk), 64, false);
        check(ForgeNative.contextSize(a) == 64 && ForgeNative.contextSize(d) == 128 && ForgeNative.vocab(a) == 300, "context / vocab");
        check(ForgeNative.kind(a).equals("generative_sequence") && s(ForgeNative.manifest(a)).contains("\"format\":\"fpk\""), "kind / manifest");
        long w2 = ForgeNative.modelWeights(a); ForgeNative.weightsClose(w2);

        // ---- tokenizer; text beyond the BMP and partial UTF-8 survive the boundary as bytes
        int[] ids = ForgeNative.tokenize(a, b("the weather"), 1);
        check(ids[0] == 290 && s(ForgeNative.detokenize(a, Arrays.copyOfRange(ids, 1, ids.length))).equals("the weather"), "tokenize round trip");
        String emoji = "café 😀 世界";
        int[] eids = ForgeNative.tokenize(a, b(emoji), 0);
        check(eids.length == b(emoji).length && s(ForgeNative.detokenize(a, eids)).equals(emoji), "UTF-8 beyond the BMP round trips (byte-level tokens)");
        check(Arrays.equals(ForgeNative.tokenBytes(a, eids[3]), new byte[]{(byte) 0xC3}), "a token that is half a character comes back as its byte");
        check(ForgeNative.tokenBos(a) == 290 && ForgeNative.tokenEos(a) == 291 && ForgeNative.tokenIsSpecial(a, 291) && !ForgeNative.tokenIsSpecial(a, 65), "special tokens");
        check(ForgeNative.tokenId(a, b("</s>")) == 291 && ForgeNative.tokenId(a, b("nope-nope")) == -1 && s(ForgeNative.tokenBytes(a, 291)).equals("</s>"), "token id / bytes");
        check(Arrays.equals(ForgeNative.stopTokens(a, true), new int[]{291}) && ForgeNative.chatTemplate(a).length == 0, "stop tokens / template");
        fails("chat template", () -> ForgeNative.applyChatTemplate(a, b("[{\"role\":\"user\",\"content\":\"hi\"}]"), null, null, true));
        fails("vocabulary", () -> ForgeNative.tokenBytes(a, 1 << 30));

        // ---- forward, generation
        ForgeNative.reset(a);
        float[] all = ForgeNative.forwardAll(a, ids);
        ForgeNative.reset(a);
        float[] last = ForgeNative.forward(a, ids);
        check(all.length == 300 * ids.length && last.length == 300 && ForgeNative.position(a) == ids.length, "forward shapes");
        float diff = 0; for (int i = 0; i < 300; ++i) diff = Math.max(diff, Math.abs(last[i] - all[300 * (ids.length - 1) + i]));
        check(diff < 1e-4f, "forward_all's last row is forward");
        ForgeNative.rewind(a, 2); check(ForgeNative.position(a) == 2, "rewind");
        ForgeNative.reset(a);
        int[] ref = ForgeNative.generate(a, ids, 12, 0f, 0, 1f, 0f, 1f, 0, null);   // ABI 1 entry point
        int[] why = new int[1];
        List<Integer> seen = new ArrayList<>(); java.io.ByteArrayOutputStream text = new java.io.ByteArrayOutputStream();
        int[] out = gen(a, ids, 12, GREEDY, 0, null, null, null, null, 0, null, true, (t, bytes) -> { seen.add(t); text.write(bytes, 0, bytes.length); return false; }, why);
        check(ref.length == 12 && Arrays.equals(out, ref) && why[0] == 0 && seen.size() == 12 && Arrays.equals(text.toByteArray(), ForgeNative.detokenize(a, out)), "generate v4 == legacy, callback per token");
        int[] calls = {0};
        out = gen(a, ids, 12, GREEDY, 0, null, null, null, null, 0, null, true, (t, bytes) -> ++calls[0] >= 1, why);
        check(out.length == 1 && why[0] == 2, "the listener stops generation");
        int[] bad = {0};
        try { gen(a, ids, 12, GREEDY, 0, null, null, null, null, 0, null, true, (t, bytes) -> { throw new UnsupportedOperationException("listener"); }, why); bad[0] = 1; }
        catch (UnsupportedOperationException e) { check(e.getMessage().equals("listener"), "the listener's own exception arrives"); }
        check(bad[0] == 0, "an exception in a listener propagates");
        int k = 1; while (k < 12 && contains(ref, ref[k], k)) ++k;
        out = gen(a, ids, 12, GREEDY, 0, null, null, null, null, 0, new int[]{ref[k]}, true, null, why);
        check(Arrays.equals(out, Arrays.copyOf(ref, k)) && why[0] == 1, "stop tokens");
        out = gen(a, ids, 3, GREEDY, 0, new int[]{65}, new float[]{1e9f}, null, null, 0, null, true, null, why);
        check(Arrays.equals(out, new int[]{65, 65, 65}), "logit bias forces a token");
        float[] hot = sampler(0.9f, 40);
        int[] s1 = gen(a, ids, 12, hot, 7, null, null, null, null, 0, null, true, null, null), s2 = gen(a, ids, 12, hot, 7, null, null, null, null, 0, null, true, null, null);
        check(Arrays.equals(s1, s2) && !Arrays.equals(s1, gen(a, ids, 12, hot, 8, null, null, null, null, 0, null, true, null, null)), "seeds");
        float[] pen = GREEDY.clone(); pen[7] = 50f; pen[8] = 5f;
        out = gen(a, ids, 12, pen, 0, null, null, null, null, 0, null, true, null, null);
        check(Arrays.stream(out).distinct().count() == 12, "presence / frequency penalties");
        out = gen(a, ids, 8, hot, 3, null, null, "root ::= \"yes\" | \"no\"", null, 0, null, true, null, why);
        String yn = s(ForgeNative.detokenize(a, out)); check((yn.equals("yes") || yn.equals("no")) && why[0] == 3, "grammar: " + yn);
        out = gen(a, ids, 12, hot, 3, null, null, null, "{\"enum\":[\"red\",7]}", 0, null, true, null, why);
        String js = s(ForgeNative.detokenize(a, out)); check(js.equals("\"red\"") || js.equals("7"), "json schema: " + js);
        fails("not both", () -> gen(a, ids, 4, GREEDY, 0, null, null, "root ::= \"a\"", "{}", 0, null, true, null, null));
        out = gen(a, ids, 12, GREEDY, 0, null, null, null, null, d, null, false, null, null);
        check(Arrays.equals(out, ref), "a draft session gives the greedy tokens");
        String g = s(ForgeNative.jsonSchemaToGrammar(b("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"boolean\"}},\"required\":[\"a\"]}")));
        check(g.startsWith("root ::= "), "schema to grammar"); ForgeNative.grammarCheck(b(g));
        fails("grammar", () -> ForgeNative.grammarCheck(b("root ::= \"open")));
        fails("json", () -> ForgeNative.jsonSchemaToGrammar(b("{nope")));

        // ---- KV state
        ForgeNative.reset(a); ForgeNative.reset(d);
        ForgeNative.forward(a, ids);
        byte[] blob = ForgeNative.stateSave(a, -1);
        check(blob.length > 16 && ForgeNative.stateBytes(a) > 0, "state save");
        ForgeNative.stateLoad(d, blob);
        check(ForgeNative.position(d) == ids.length && Arrays.equals(ForgeNative.forward(a, new int[]{ref[0]}), ForgeNative.forward(d, new int[]{ref[0]})), "state load continues identically");
        fails("load_prefix", () -> ForgeNative.stateLoad(d, Arrays.copyOf(blob, blob.length / 2)));
        check(ForgeNative.position(d) == 0, "a refused state leaves the session empty");
        ForgeNative.stateCopy(d, a, ids.length);
        check(ForgeNative.position(d) == ids.length, "state copy");
        ForgeNative.forward(own, ids);   // a session with other weights (its own copy of the file)
        fails("different models", () -> ForgeNative.stateCopy(d, own, 1));
        fails("tokens asked", () -> ForgeNative.stateSave(a, 1000));

        // ---- adapters
        Path lora = Paths.get(work, "jni_lora.safetensors");
        writeLora(lora, "layer.0.wq", 64, 64);
        check(!ForgeNative.adapterExists(w, b("tuned")) && s(ForgeNative.adapterGet(a)).isEmpty(), "no adapter yet");
        ForgeNative.adapterLoad(w, b("tuned"), b(lora.toString()));
        fails("tuned", () -> ForgeNative.adapterLoad(w, b("tuned"), b(lora.toString())));
        ForgeNative.adapterSet(a, b("tuned"));
        check(ForgeNative.adapterExists(w, b("tuned")) && s(ForgeNative.adapterGet(a)).equals("tuned") && ForgeNative.position(a) == 0, "adapter set");
        float[] noEos = {-1e9f}; int[] noEosTok = {291};
        int[] tuned = gen(a, ids, 12, GREEDY, 0, noEosTok, noEos, null, null, 0, null, true, null, null);
        ForgeNative.adapterSet(a, null);
        check(!Arrays.equals(tuned, ref) && Arrays.equals(gen(a, ids, 12, GREEDY, 0, null, null, null, null, 0, null, true, null, null), ref), "adapter changes the output, unset restores it");
        fails("missing", () -> ForgeNative.adapterSet(a, b("missing")));

        // ---- batching
        long bt = ForgeNative.batchOpen(w, 2, 0, 0, 4, null);
        List<String> doneLog = new ArrayList<>(); int[][] results = new int[8][];
        ForgeNative.DoneSink done = (id, toks, reason) -> { results[(int) id] = toks; doneLog.add(id + ":" + s(reason)); };
        int[] tokCount = {0};
        long r1 = ForgeNative.batchSubmit(bt, ids, 12, GREEDY, 0, noEosTok, noEos, null, null, null, null, 0, (t, bytes) -> { ++tokCount[0]; return false; }, done);
        long r2 = ForgeNative.batchSubmit(bt, ids, 12, hot, 2, null, null, b("root ::= \"yes\" | \"no\""), null, null, null, 0, null, done);
        long r3 = ForgeNative.batchSubmit(bt, ids, 50, GREEDY, 0, null, null, null, null, null, null, -5, null, done);
        long r4 = ForgeNative.batchSubmit(bt, ids, 12, GREEDY, 0, noEosTok, noEos, null, null, null, b("tuned"), 0, null, done);
        long r5 = ForgeNative.batchSubmit(bt, ids, 12, GREEDY, 0, null, null, null, null, new int[]{ref[k]}, null, 0, null, null);
        check(r1 > 0 && r5 > r4, "ids");
        check(ForgeNative.batchCancel(bt, r3) && !ForgeNative.batchCancel(bt, r3) && doneLog.contains(r3 + ":cancelled"), "cancel");
        check(ForgeNative.batchStep(bt), "step");
        ForgeNative.batchRun(bt);
        check(!ForgeNative.batchStep(bt), "idle after run");
        check(Arrays.equals(results[(int) r1], ref) && tokCount[0] == 12 && doneLog.contains(r1 + ":length"), "batch request 1: " + doneLog);
        String byn = s(ForgeNative.detokenize(a, results[(int) r2])); check(byn.equals("yes") || byn.equals("no"), "batch grammar: " + byn);
        check(Arrays.equals(results[(int) r4], tuned), "batch request with an adapter");
        long[] st = ForgeNative.batchStats(bt);
        check(st.length == 9 && st[1] == 5 && st[2] == 0 && st[5] == 2 && st[7] == 0 && st[8] == 0, "stats " + Arrays.toString(st));
        fails("missing", () -> ForgeNative.batchSubmit(bt, ids, 4, GREEDY, 0, null, null, null, null, null, b("missing"), 0, null, done));
        fails("empty prompt", () -> ForgeNative.batchSubmit(bt, new int[0], 4, GREEDY, 0, null, null, null, null, null, null, 0, null, done));
        ForgeNative.batchEvictPrefixCache(bt, 0);
        // a listener that throws: the exception surfaces from batchRun, the request still ends
        ForgeNative.batchSubmit(bt, ids, 6, GREEDY, 0, null, null, null, null, null, null, 0, (t, bytes) -> { throw new UnsupportedOperationException("batch listener"); }, null);
        try { ForgeNative.batchRun(bt); check(false, "expected the batch listener's exception"); }
        catch (UnsupportedOperationException e) { check(e.getMessage().equals("batch listener"), "batch listener exception"); }
        // closing with a request in flight tells it "cancelled"
        long r6 = ForgeNative.batchSubmit(bt, ids, 40, GREEDY, 0, null, null, null, null, null, null, 0, null, done);
        ForgeNative.batchStep(bt);
        ForgeNative.batchClose(bt);
        check(doneLog.contains(r6 + ":cancelled"), "close cancels: " + doneLog);

        // weights may be closed before their sessions
        ForgeNative.weightsClose(w);
        check(Arrays.equals(gen(a, ids, 12, GREEDY, 0, null, null, null, null, 0, null, true, null, null), ref), "a session outlives its weights handle");
        fails("embedding", () -> ForgeNative.embedDim(a));
        ForgeNative.close(a); ForgeNative.close(d); ForgeNative.close(own);

        // ---- embeddings
        long ew = ForgeNative.weightsOpen(b(bert), false, null, 0, false), e = ForgeNative.sessionOpen(ew, 0, null, false);
        int dim = (int) ForgeNative.embedDim(e);
        String[] texts = {"The quick brown fox jumps over the lazy dog.", "Hello, world!", "it's John's"};
        float[] one = ForgeNative.embedText(e, b(texts[1]), null);
        float[] rows = ForgeNative.embedTexts(e, new byte[][]{b(texts[0]), b(texts[1]), b(texts[2])});
        check(ForgeNative.kind(e).equals("embedding") && dim > 0 && one.length == dim && rows.length == 3 * dim && Arrays.equals(one, Arrays.copyOfRange(rows, dim, 2 * dim)), "embed text / texts");
        int[] tids = ForgeNative.tokenize(e, b(texts[1]), -1);
        check(Arrays.equals(ForgeNative.embedTokens(e, tids, null), one) && Arrays.equals(ForgeNative.embedTokens(e, tids, new int[tids.length]), one), "embed tokens");
        check(!Arrays.equals(ForgeNative.embedText(e, b("query"), b("passage")), ForgeNative.embedText(e, b("query"), null)), "embed pair");
        fails("one entry per token", () -> ForgeNative.embedTokens(e, tids, new int[1]));
        fails("no texts", () -> ForgeNative.embedTexts(e, new byte[0][]));
        fails("cannot generate", () -> gen(e, tids, 2, GREEDY, 0, null, null, null, null, 0, null, true, null, null));
        ForgeNative.close(e); ForgeNative.weightsClose(ew);

        // ---- AOT sidecar
        Path copy = Paths.get(work, "jni_aot.fpk");
        Files.copy(Paths.get(fpk), copy, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        check(ForgeNative.aotPrecompile(b(copy.toString()), null, new int[]{1, 3}) == 2 && Files.exists(Paths.get(copy + ".fplan")), "aot sidecar");
        long pm = ForgeNative.open(b(copy.toString()), 0, false);
        check(Arrays.equals(ForgeNative.generate(pm, ids, 12, 0f, 0, 1f, 0f, 1f, 0, null), ref), "same tokens with precompiled plans");
        ForgeNative.close(pm);
        Files.deleteIfExists(Paths.get(copy + ".fplan")); Files.deleteIfExists(copy); Files.deleteIfExists(lora);
        ForgeNative.close(0); ForgeNative.weightsClose(0); ForgeNative.batchClose(0);

        System.out.println("jni_smoke: " + (fails == 0 ? "ok" : fails + " FAILED") + " (ABI " + ForgeNative.abi() + ")");
        System.exit(fails == 0 ? 0 : 1);
    }
    static boolean contains(int[] a, int v, int upTo) { for (int i = 0; i < upTo; ++i) if (a[i] == v) return true; return false; }

    /** A .safetensors with one LoRA pair for `weight` ([out, in]), rank 2, lora_alpha 8. */
    static void writeLora(Path path, String weight, int out, int in) throws Exception {
        int na = 2 * in * 4, nb = out * 2 * 4;
        String hdr = "{\"" + weight + ".lora_a\":{\"dtype\":\"F32\",\"shape\":[2," + in + "],\"data_offsets\":[0," + na + "]},"
                   + "\"" + weight + ".lora_b\":{\"dtype\":\"F32\",\"shape\":[" + out + ",2],\"data_offsets\":[" + na + "," + (na + nb) + "]},"
                   + "\"lora_alpha\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[" + (na + nb) + "," + (na + nb + 4) + "]}}";
        java.nio.ByteBuffer buf = java.nio.ByteBuffer.allocate(8 + hdr.length() + na + nb + 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buf.putLong(hdr.length()); buf.put(b(hdr));
        java.util.Random rng = new java.util.Random(5);
        for (int i = 0; i < (na + nb) / 4; ++i) buf.putFloat((float) rng.nextGaussian());
        buf.putFloat(8f);
        Files.write(path, buf.array());
    }
}

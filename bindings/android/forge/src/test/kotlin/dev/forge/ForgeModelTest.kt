package dev.forge

import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Runs on a device/emulator (connectedAndroidTest) or a host JVM with libforge_jni on java.library.path.
 *  Skips when FORGE_TEST_FPK (a model path) is not set. */
class ForgeModelTest {
    @Test fun abi() = assertTrue(ForgeModel.abi >= 4)

    /** ABI 4 surface on shared weights. NEVER RUN (no Kotlin toolchain here); it mirrors tests/jni/JniSmoke.java. */
    @Test fun sharedWeightsAndSampling() {
        val path = System.getenv("FORGE_TEST_FPK"); assumeTrue(path != null && File(path).exists())
        ForgeWeights.open(path!!, device = "cpu").use { w ->
            assertEquals("cpu", w.device)
            val a = w.session(context = 64); val b = w.session()
            val ids = a.tokenize("the weather")
            val fresh = GenerateOptions(ignoreEos = true, reset = true)
            val ref = a.generate(ids, 8, ForgeModel.Sampling(), fresh)
            assertEquals(8, ref.size); assertEquals(StopReason.LENGTH, a.lastStopReason)
            assertTrue(ref.contentEquals(b.generate(ids, 8, ForgeModel.Sampling(), fresh)))
            val hot = ForgeModel.Sampling(temperature = 0.9f, topK = 40, seed = 7, presencePenalty = 0.5f)
            assertTrue(a.generate(ids, 8, hot, GenerateOptions(reset = true)).contentEquals(a.generate(ids, 8, hot, GenerateOptions(reset = true))))
            val yn = a.detokenize(a.generate(ids, 8, hot, GenerateOptions(grammar = "root ::= \"yes\" | \"no\"", reset = true)))
            assertTrue(yn == "yes" || yn == "no"); assertEquals(StopReason.GRAMMAR, a.lastStopReason)
            val text = StringBuilder()
            val streamed = a.stream(ids, 8, options = fresh) { text.append(it); false }
            assertTrue(streamed.contentEquals(ref)); assertEquals(a.detokenize(ref), text.toString())
            a.reset(); a.forward(ids)
            b.loadState(a.saveState())
            assertEquals(ids.size.toLong(), b.position)
            assertTrue(a.forward(intArrayOf(ref[0])).contentEquals(b.forward(intArrayOf(ref[0]))))
            assertTrue(a.stopTokens().contentEquals(intArrayOf(a.eosToken))); assertEquals("generative_sequence", a.kind)
            w.batch(maxActive = 2).use { batch ->
                assertEquals(3, batch.generateMany(listOf(ids, ids, ids), 8, ForgeModel.Sampling(logitBias = mapOf(a.eosToken to -1e9f))).size)
            }
            assertTrue(Forge.jsonSchemaToGrammar("{\"enum\":[1,2]}").startsWith("root ::= "))
            a.close(); b.close()
        }
    }

    @Test fun roundTrip() {
        val path = System.getenv("FORGE_TEST_FPK"); assumeTrue(path != null && File(path).exists())
        ForgeModel.open(path!!).use { m ->
            val ids = m.tokenize("hello", addBos = false)
            assertEquals("hello", m.detokenize(ids))
            val out = m.generate(m.tokenize("hi"), 4)
            assertTrue(out.size in 1..4)
            assertEquals(m.vocabSize.toInt(), m.forward(intArrayOf(out[0])).size)
        }
    }
}

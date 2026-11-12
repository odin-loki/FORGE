# Language bindings

## Concepts
The Python, Kotlin and Swift bindings wrap the C ABI ([capi.md](capi.md), `forge.h`, ABI 4) and share its model:

- **Weights** (`Weights` / `ForgeWeights`): one opened `.fpk`: weights, tokenizer, compiled plans, LoRA adapters, device. Shared by any number of sessions and batches; it may be closed before them.
- **Model** (`Model` / `ForgeModel`): one sequence (KV state and position). Opened from a path it owns private weights; opened from weights it shares them.
- **Batch** (`Batch` / `ForgeBatch`): continuous batching over shared weights.
- One thread at a time per object ([capi.md](capi.md#thread-safety)). Different objects may be used from different threads.

| Binding | Location | Native library | ABI used | Status |
|---|---|---|---|---|
| Python | `python/forge` (ctypes), package `forge-inference` | `libforge` (`forge_c`) | 4 (falls back to the ABI 2 / 3 calls on an older library) | **Tested** by the `python_bindings` ctest on Linux x86_64 (everything in the table below) |
| Kotlin | `bindings/android`, package `dev.forge` | `libforge_jni` + `libforge` | 4 | The JNI bridge (`jni/forge_jni.c`) is compiled (`jni_syntax`) and **run on a host JVM**: every native, under `-Xcheck:jni`, through a Java declaration of the same methods (`jni_smoke`, Linux with a JDK). **The Kotlin sources (`ForgeNative.kt`, `ForgeModel.kt`) and the Gradle build have never been compiled**: no Kotlin compiler or Android SDK here. `jni_decl_matches` checks that the Kotlin externals, the Java mirror and the C functions declare the same names and types |
| Swift | `bindings/apple`, module `Forge` | `FORGE.xcframework` | 4 | **Never compiled** (no Apple SDK or Swift toolchain here; the source says so). Nothing in it has been run |

What "never compiled" means for the Kotlin and Swift files: they were written and reviewed by hand against the C ABI; expect compile errors on first contact with a toolchain. The parts most likely to need attention are the Swift C-interop details (pointer conversions, the `@convention(c)` callbacks, `Unmanaged` lifetimes in `ForgeBatch`) and the Kotlin overloads of `generate` / `generateText`.

**Reset behaviour differs.** Python's `generate` resets the KV state first (`reset=False` continues). Kotlin and Swift continue from the current position by default, as before ABI 4; set `GenerateOptions.reset` or call `reset()` before an unrelated prompt.

**End of generation (changed with ABI 4).** `generate` in all three bindings now goes through `forge_generate_v4`: the token that ends generation (EOS, a stop token) is **not** in the returned ids, and the stop reason is available (`last_stop_reason` / `lastStopReason`). The ABI 1 behaviour (EOS included) is `generateLegacy` in Kotlin and Swift; Python has no legacy call on an ABI 4 library.

## API reference

### Python (`import forge`)
Type-hinted. `forge.version()`, `forge.abi_version()`, `forge.ForgeError` (a `RuntimeError`).

Module functions: `json_schema_to_grammar(schema)` → GBNF text, `grammar_check(text)` (raises on a parse error), `set_deterministic(True | False | None)`, `aot_precompile(path, out=None, seq_lengths=None)` → number of plans written to the sidecar.

`forge.Weights(path, verify=False, device=None, deterministic=None, aot=True)`, a context manager:

| Member | Meaning |
|---|---|
| `device`, `set_device(name)` | The device in use; select one (`"auto"`, `"cpu"`, ...). An unavailable device falls back to `"cpu"`, which is returned |
| `session(context=0, kv_codec=None, kv_offload=False)` | A new `Model` on these weights |
| `load_adapter(name, path)`, `has_adapter(name)` | LoRA adapters (`.fpk` or `.safetensors`) |

`forge.Model(path_or_weights, context=0, verify=False, device=None, kv_codec=None, kv_offload=False)`, a context manager, also closed on garbage collection:

| Member | Meaning |
|---|---|
| `close()`, `reset()`, `rewind(pos)` | |
| `position`, `vocab_size`, `context_size`, `kind`, `manifest` (dict), `weights` | |
| `tokenize(text, add_bos=None)` → `list[int]`, `detokenize(ids)` → `str` | `add_bos` `None`: the model's default |
| `bos_token`, `eos_token`, `token_bytes(id)`, `token_id(str)`, `is_special(id)`, `stop_tokens(chat=True)` | Tokenizer details |
| `chat_template`, `apply_chat_template(messages, tools=None, add_generation_prompt=True, **template_vars)` | The Jinja template and its rendering (messages and tools as Python lists or JSON text) |
| `forward(ids)`, `forward_all(ids)` | Last-row logits / `[len(ids), vocab]` logits as numpy arrays (lists without numpy). Continue from the current position |
| `generate(prompt, max_tokens=32, temperature=0.0, top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0, seed=0, grammar=None, draft=None, spec_k=4, on_token=None, typical_p=1.0, mirostat_tau=0.0, mirostat_eta=0.1, presence_penalty=0.0, frequency_penalty=0.0, repetition_window=64, logit_bias=None, json_schema=None, stop_tokens=None, ignore_eos=False, reset=True)` | `prompt` is a string or token ids. Returns the new ids. `grammar` is GBNF text, `json_schema` a dict or text, `logit_bias` `{token: bias}`, `draft` another `Model`. `on_token(id, text)` is called while generation runs; returning true stops; an exception in it is re-raised. Sets `last_stop_reason` (`"length"`, `"eos"`, `"callback"`, `"grammar"`) |
| `complete(text, **kw)` | Generated text |
| `stream(prompt, **kw)` | A generator that yields text pieces **while generation runs**: the C token callback feeds a queue from a worker thread. Bytes of a character split over tokens are held until it is complete. Closing the generator early stops generation at the next token. Do not use the `Model` for anything else until the generator is finished or closed |
| `chat(messages, tools=None, **kw)` | One assistant turn: template, generation up to an end-of-turn token, text |
| `adapter` (property, assignable) | The selected LoRA adapter (`""` / `None` = base). Assigning clears the KV state when it changes |
| `save_state(n_tokens=-1)` → `bytes`, `load_state(blob)`, `copy_state_from(other, n_tokens=None)`, `state_bytes` | KV state |
| `embed_dim`, `embed(input, pair=None)` | Embedding models: a string or token ids → `[dim]`; a list of strings → `[n, dim]` (a loop with shared setup in the engine) |

`forge.Batch(weights, max_active=8, context=0, prefill_chunk=256, prefix_cache=4, kv_codec=None)`, a context manager: `submit(prompt_ids, max_tokens=32, grammar=None, json_schema=None, stop_tokens=None, adapter=None, priority=0, on_token=None, on_done=None, **sampling)` → id; `step()`, `run()`, `cancel(id)`, `stats()`, `evict_prefix_cache(keep=0)`; `results[id] = (tokens, reason)`; `generate_many(prompts, **kw)` → the tokens of each prompt. Callbacks run on the thread that calls `step` / `run`; an exception in one is re-raised from that call. In `results` the token that ended a sequence (EOS, a stop token) is the last one, as in the C ABI.

Library lookup order: `$FORGE_LIB`; then, for this platform's names (`libforge.so` / `libforge.so.1`; `libforge.dylib`; `forge.dll`), the package directory, `../../build/dev` and `../../build/release` relative to it, and the loader path. Importing raises `ForgeError` if the library's ABI is below 2; features of a newer ABI raise `ForgeError` when used with an older library. Only the Linux names have been loaded here.

### Kotlin (`dev.forge`)
- `Forge.version`, `Forge.abi`, `Forge.jsonSchemaToGrammar(schema)`, `Forge.grammarCheck(grammar)`, `Forge.setDeterministic(Boolean?)`, `Forge.aotPrecompile(path, out, seqLengths)`.
- `ForgeWeights.open(path, verify = false, device = null, deterministic = null, aot = true)` (`AutoCloseable`): `device`, `setDevice(name)`, `session(context = 0, kvCodec = null, kvOffload = false)`, `batch(maxActive = 8, ...)`, `loadAdapter(name, path)`, `hasAdapter(name)`.
- `ForgeModel.open(path, context = 2048, verify = false)` (`AutoCloseable`; private weights): `vocabSize`, `position`, `contextSize`, `kind`, `manifestJson`, `weights()`, `reset()`, `rewind(position)`; `tokenize(text, addBos = true)` (`null`: the model's default), `detokenize(ids)`, `bosToken`, `eosToken`, `isSpecial(id)`, `tokenId(text)`, `tokenBytes(id)`, `stopTokens(chat = true)`, `chatTemplate`, `applyChatTemplate(messagesJson, toolsJson, extraJson, addGenerationPrompt)`; `forward(ids)`, `forwardAll(ids)`.
- `generate(prompt: IntArray, maxNew, sampling = Sampling(), listener = null)` and `generate(prompt, maxNew, sampling, options: GenerateOptions, listener = null)` → `IntArray`; `generateText(...)` → `String`; `generateLegacy(...)` (ABI 1: EOS included); `stream(prompt, maxNew, sampling, options) { text -> stop }`; `chat(messagesJson, maxNew, sampling, toolsJson)`; `lastStopReason`. `TokenListener.onToken(token, text)` returns true to stop.
- `ForgeModel.Sampling(temperature, topK, topP, minP, repetitionPenalty, seed, typicalP, mirostatTau, mirostatEta, presencePenalty, frequencyPenalty, repetitionWindow, logitBias)`; `GenerateOptions(grammar, jsonSchema, stopTokens, ignoreEos, reset, draft, specK)`.
- `adapter` (property), `saveState(nTokens = -1)`, `loadState(blob)`, `copyStateFrom(other, nTokens)`, `stateBytes`; `embedDim`, `embed(text, pair)`, `embed(ids, typeIds)`, `embed(texts)`.
- `ForgeBatch` (`AutoCloseable`): `submit(prompt, maxNew, sampling, grammar, jsonSchema, stopTokens, adapter, priority, listener, onDone)` → id, `step()`, `run()`, `cancel(id)`, `stats()`, `evictPrefixCache(keep)`, `results`, `generateMany(prompts, maxNew, sampling)`. If a listener throws, the rest of that `step` / `run` winds down (running requests are cancelled without their done callbacks) and the exception surfaces from the call.
- Errors throw `IllegalStateException` with the handle's error message. Text crosses JNI as UTF-8 bytes, so characters outside the BMP and tokens that are part of a character are handled (the earlier bridge used JNI's modified UTF-8).

### Swift (`import Forge`)
- `Forge.version`, `Forge.abi`, `Forge.jsonSchemaToGrammar(_:)`, `Forge.grammarCheck(_:)`, `Forge.setDeterministic(_:)`, `Forge.aotPrecompile(path:out:seqLengths:)`.
- `ForgeWeights(path:verify:device: = "metal", deterministic:aot:)`: `device`, `setDevice(_:)`, `session(context:kvCodec:kvOffload:)`, `batch(maxActive:...)`, `loadAdapter(name:path:)`, `hasAdapter(_:)`.
- `ForgeModel(path:context: = 2048, verify: = false, device: = "metal") throws` (private weights; `device` is written to `FORGE_DEVICE`, process-wide, as before). `tokenize(_:addBOS:)`, `detokenize(_:)`, `bosToken`, `eosToken`, `isSpecial(_:)`, `tokenID(_:)`, `tokenBytes(_:)`, `stopTokens(chat:)`, `chatTemplate()`, `applyChatTemplate(messagesJSON:toolsJSON:extraJSON:addGenerationPrompt:)`, `forward(_:)`, `forwardAll(_:)`, `reset()`, `rewind(to:)`; `vocabSize`, `position`, `contextSize`, `kind`, `manifestJSON()`, `weights()`.
- `generate(_:maxNew:sampling:options:onToken:)` takes `[Int32]` (returns ids) or a `String` (returns text); `generateLegacy`, `stream(_:maxNew:sampling:options:onText:)`, `chat(messagesJSON:maxNew:sampling:toolsJSON:)`, `lastStopReason`. `Sampling(...)` and `GenerateOptions(...)` carry the same fields as in Kotlin.
- `adapter`, `setAdapter(_:)`, `saveState(tokens:)` → `Data`, `loadState(_:)`, `copyState(from:tokens:)`, `stateBytes`, `embedDim()`, `embed(_:pair:)`, `embed(tokens:typeIDs:)`, `embed(_: [String])`.
- `ForgeBatch`: `submit(...)`, `step()`, `run()`, `cancel(_:)`, `stats()`, `evictPrefixCache(keep:)`, `results`, `takeResult(_:)`, `generateMany(_:maxNew:sampling:)`. Errors are `ForgeError`.

## Examples

Python (shape of the `python_bindings` test, on `golden/tiny_f32.fpk`):
```python
import forge
with forge.Model("model.fpk", context=256) as m:
    ids = m.generate("Hello", max_tokens=8)
    text = m.complete("Hi", max_tokens=4, temperature=0.7, top_k=40, presence_penalty=0.2)
    yn = m.complete("Answer:", max_tokens=4, grammar='root ::= "yes" | "no"')
    obj = m.complete("JSON:", max_tokens=64, json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]})
    for piece in m.stream("Once upon", max_tokens=32, temperature=0.8):   # arrives while it generates
        print(piece, end="", flush=True)

with forge.Weights("model.fpk", device="auto") as w:
    a, b = w.session(), w.session(context=512)        # two sequences, one set of weights
    b.copy_state_from(a)                               # share a computed prefix
    with forge.Batch(w, max_active=4) as batch:
        outs = batch.generate_many([a.tokenize("Hi"), a.tokenize("Hello")], max_tokens=16)
```

Kotlin (**not compiled here**):
```kotlin
ForgeWeights.open("/sdcard/model.fpk", device = "vulkan").use { w ->
    w.session(context = 2048).use { m ->
        val text = m.generateText("Hello", maxNew = 32, sampling = ForgeModel.Sampling(temperature = 0.7f, topK = 40))
        m.stream(m.tokenize("Once upon"), 64, options = GenerateOptions(reset = true)) { piece -> print(piece); false }
    }
}
```

Swift (**not compiled here**):
```swift
let w = try ForgeWeights(path: "model.fpk", device: "metal")
let m = try w.session(context: 2048)
let out = try m.generate("Hello", maxNew: 32, sampling: Sampling(temperature: 0.7, topK: 40))
try m.stream(try m.tokenize("Once upon"), maxNew: 64, options: GenerateOptions(reset: true)) { print($0, terminator: ""); return false }
```

## Platform notes and limits
- **What ran where.** Python: Linux x86_64, CPython 3.13, against the build tree's `libforge.so`. JNI bridge: Linux x86_64, OpenJDK 21, `-Xcheck:jni`, from Java. Kotlin, Gradle, Swift, XCTest: nothing compiled, nothing run. No binding has run on Android, iOS, macOS or Windows.
- **Not in the bindings**: log-probabilities and tool-call helpers (forge-serve only), vision / audio embedding inputs, a background batch worker. The Kotlin and Swift token listeners get each token's bytes decoded on their own; use `stream` for text that is correct across tokens.
- **Python packaging:** `python/pyproject.toml` declares `forge-inference` 0.1.0 with `libforge.so` as package data and `numpy` as an optional extra. Copy the built library into `python/forge/` before building a wheel.
- **Android** (not run here): `cd bindings/android && gradle :example:assembleDebug`. The AAR builds `forge_jni` through the NDK (minSdk 26, arm64-v8a and x86_64). Device tests: `gradle :forge:connectedAndroidTest` with `FORGE_TEST_FPK` set; the same test runs on a host JVM (`:forge:test`) when `libforge_jni` is on `java.library.path`.
- **Android without Gradle:** `tests/android_emulator.sh apk <build-dir>` builds, installs and launches the example APK and checks the JNI library loaded. Not run here, and not since the natives moved to `dev.forge.ForgeNative` (`tests/mobile_build.sh android` counts the `Java_dev_forge_ForgeNative_*` exports of the NDK build; also not run here).
- **Android hooks:** forward system callbacks to the C++ side (see [os.md](os.md) and [mobile.md](mobile.md)):
  - `ComponentCallbacks2.onTrimMemory` → `forge::os::memory_pressure::instance().on_trim_memory(level)`,
  - battery and thermal state → `power_hooks::report(...)`,
  - a foreground service → `background::set_hooks(...)`,
  - the Android Keystore → `keystore::set_backend(...)`.
- **Apple** (not run here): `tools/build_xcframework.sh` on macOS with Xcode builds `bindings/apple/FORGE.xcframework`; then `swift test` in `bindings/apple` with `FORGE_TEST_FPK` set. The package targets iOS 16.3 / macOS 13.3.

# Language bindings

## Concepts
The Python, Kotlin and Swift bindings all wrap the C ABI described in [capi.md](capi.md) (`forge.h`, ABI version 3), and expose the same model:
1. Open a `.fpk` file.
2. Tokenize the text.
3. Call `generate` with sampling settings and an optional per-token callback. The callback returns true to stop.
4. Close the model.

| Binding | Location | Native library | Status |
|---------|----------|----------------|--------|
| Python | `python/forge` (ctypes) | `libforge.so` | **Tested** by the `python_bindings` ctest |
| Kotlin | `bindings/android`, package `dev.forge` | `libforge_jni.so` (`-DFORGE_JNI=ON`) | The JNI library cross-builds and ctest checks its exports. **The Kotlin and Gradle code itself has not been compiled here** |
| Swift | `bindings/apple`, module `Forge` | XCFramework | **Not compiled** (no Apple SDK available) |

## API reference

### Python (`import forge`)
- `forge.version()`, `forge.abi_version()` and `forge.ForgeError`.
- `forge.Model(path, context=0, verify=False)` is a context manager. Its methods:
  - `close()`, `reset()` and `rewind(pos)`. The properties `position` and `vocab_size` are read-only.
  - `tokenize(text, add_bos=True)` and `detokenize(ids)`.
  - `forward(ids)` and `forward_all(ids)` return numpy arrays.
  - `generate(prompt, max_tokens=32, temperature=0.0, top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0, seed=0, grammar=None, draft=None, spec_k=4, on_token=None, typical_p=1.0, mirostat_tau=0.0, mirostat_eta=0.1)`. It accepts a string or token ids, resets the model state first, and returns the new token ids.
  - `complete(text, **kw)` returns the generated text. `stream(text, **kw)` yields text pieces after generation finishes; use `on_token` if you need true streaming.
- The library is searched for in this order: `$FORGE_LIB`, then `python/forge/libforge.so`, then `build/dev/libforge.so`, then the system loader path.

### Kotlin (`dev.forge.ForgeModel`, `AutoCloseable`)
- `ForgeModel.open(path, context = 2048, verify = false)`, plus `ForgeModel.version` and `ForgeModel.abi`.
- `tokenize(text, addBos = true)`, `detokenize(ids)`, `forward(ids)`, `reset()` and `rewind(position)`.
- The properties `vocabSize` and `position`.
- `generate(prompt: IntArray, maxNew, sampling, listener)` and `generateText(prompt: String, maxNew, sampling, listener)`. A `TokenListener.onToken(token, text)` returns true to stop.
- `Sampling(temperature = 0f, topK = 0, topP = 1f, minP = 0f, repetitionPenalty = 1f, seed = 0)`.
- To use the GPU, set `FORGE_DEVICE=vulkan` before calling `open`.

### Swift (`import Forge`)
- `ForgeModel(path:context: = 2048, verify: = false, device: = "metal") throws`. The `device` value is written to `FORGE_DEVICE`.
- `tokenize(_:addBOS:)`, `detokenize(_:)`, `forward(_:)`, `reset()` and `rewind(to:)`.
- `generate(_:maxNew:sampling:onToken:)` takes either token ids or a `String`.
- `Sampling(temperature:topK:topP:minP:repetitionPenalty:seed:)`.

## Examples

Python. This snippet was run against `golden/tiny_f32.fpk`:
```python
import forge
with forge.Model("model.fpk", context=256) as m:
    ids = m.generate("Hello", max_tokens=8)
    text = m.complete("Hi", max_tokens=4, temperature=0.7, top_k=40)
```

Kotlin (**not compiled here**):
```kotlin
ForgeModel.open("/sdcard/model.fpk", context = 2048).use { m ->
    val text = m.generateText("Hello", maxNew = 32, sampling = ForgeModel.Sampling(temperature = 0.7f, topK = 40))
}
```

Swift (**not compiled here**):
```swift
let m = try ForgeModel(path: "model.fpk", context: 2048, device: "metal")
let out = try m.generate("Hello", maxNew: 32, sampling: Sampling(temperature: 0.7, topK: 40))
```

## Platform notes and limits
- **Android build** (not run here): `cd bindings/android && gradle :example:assembleDebug`. To run the device tests, use `gradle :forge:connectedAndroidTest` with `FORGE_TEST_FPK` set.
- **Android without Gradle:** `tests/android_emulator.sh apk <build-dir>` builds and installs the example APK.
- **Android hooks:** app code should forward system callbacks to the C++ hooks:
  - `ComponentCallbacks2.onTrimMemory` to `forge::os::memory_pressure::instance().on_trim_memory(level)`.
  - Battery and thermal state to `power_hooks::report(...)`.
  - A foreground service to `background::set_hooks(...)`.
  - The Android Keystore to `keystore::set_backend(...)`.

  See [os.md](os.md).
- **Apple:** build the XCFramework with `tools/build_xcframework.sh`, then run `swift test` with `FORGE_TEST_FPK` set. **Neither has been run.**

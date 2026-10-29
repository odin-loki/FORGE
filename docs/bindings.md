# Language bindings

Both wrap the C ABI in `mind/include/forge.h` (version 3; see [`reference.md`](reference.md)).

## Android — Kotlin (`bindings/android`, package `dev.forge`)

```kotlin
ForgeModel.open("/sdcard/model.fpk", context = 2048).use { m ->
    val text = m.generateText("Hello", maxNew = 32, sampling = ForgeModel.Sampling(temperature = 0.7f, topK = 40))
}
```
API: `open(path, context, verify)`, `tokenize`, `detokenize`, `forward` (logits), `generate`/`generateText` with an optional
`TokenListener` (return true to stop), `reset`, `rewind(position)`, `close`, `ForgeModel.abi`.
Native side: `bindings/android/jni/forge_jni.c` → `libforge_jni.so` (CMake `-DFORGE_JNI=ON`).
Build: `cd bindings/android && gradle :example:assembleDebug`; device test `gradle :forge:connectedAndroidTest`
(`FORGE_TEST_FPK`). Status: JNI library cross-builds and exports are checked by ctest; the Kotlin/Gradle project has
not been compiled on this machine.

OS hooks for app glue (C++): `forge::os::memory_pressure::instance().on_trim_memory(level)` from
`ComponentCallbacks2.onTrimMemory`, `power_hooks::report(...)`, `background::set_hooks(...)` for a foreground service,
`keystore::set_backend(...)` for Android Keystore. See [`os.md`](os.md).

## Apple — Swift package (`bindings/apple`, module `Forge`)

```swift
import Forge
let m = try ForgeModel(path: "model.fpk", context: 2048, device: "metal")
let out = try m.generate("Hello", maxNew: 32, sampling: Sampling(temperature: 0.7, topK: 40))
```
API mirrors Kotlin: `tokenize`, `detokenize`, `forward`, `generate` (ids or text, `onToken` callback returns true to stop),
`reset`, `rewind(to:)`. Build the XCFramework with `tools/build_xcframework.sh`, test with `swift test`
(`FORGE_TEST_FPK`). Status: not compiled (no Apple SDK on this machine).

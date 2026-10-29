# lane: CPU features and kernel dispatch

## Concepts
`forge::lane` checks what the CPU supports and what the OS enables (for example AVX-512 state, or AMX permission). It then picks one table of kernel function pointers. That choice is made once, and every later call goes through the table.

## API (`lane/include/forge/lane/dispatch.hpp`)
- `const cpu_features& detect()`: the result is cached after the first call. `summary()` returns a one-line description.
- `const kernels& active()`: the best target for this machine. Set the `FORGE_LANE_TARGET` environment variable to override it.
- `const kernels* by_name(const char*)`: accepts `"scalar"`, `"avx2"` or `"avx512"`. Returns `nullptr` if that target is not built or not usable.
- `kernels` holds `target` plus these pointers: `saxpy`, `dot`, `sum`, `softmax_row`, `rms_norm_row`, the elementwise math functions (exp, log, tanh, sigmoid, silu, gelu, silu_mul, add, sub, mul), `gemm_nt`, `gemm_nn` and `logsumexp`.

## Example
`docs/examples/lane_dispatch.cpp` is built and run as the ctest `doc_example_lane_dispatch`.

```cpp
const forge::lane::kernels& k = forge::lane::active();
k.saxpy(3.f, x.data(), y.data(), x.size());   // y = 3*x + y
float d = k.dot(x.data(), y.data(), x.size());
```

## Platform notes and limits
- Link with `forge_lane_mv`, the static library of compiled kernel variants.
- AMX (x86_64) and SME (aarch64) tile kernels are stubbed out when the compiler cannot build them.
- Only the scalar target exists on non-x86 builds, apart from NEON on aarch64.

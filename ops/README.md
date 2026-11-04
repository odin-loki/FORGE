# ops: one definition per operation

## Concepts
Every tensor operation is declared exactly once, in a `.fspec` file, together with a scalar reference implementation. The other layers are generated from or checked against that declaration:
- The **graph** IR uses the generated `enum class op`.
- **Lowerings** (CPU SIMD, Vulkan and the others) must agree with the reference within the declared tolerance. The oracle in `verify/` checks this.
- **GPU kernels** are generated from the same specs.

## Layout

| Path | Contents |
|------|----------|
| `ops/defs/*.fspec` | Op declarations: `elementwise` (40), `reduction` (11), `movement` (8), `contraction` (3) |
| `ops/include/forge/ops/reference.hpp` | Scalar references, `forge::ops::ref::<name>` |
| `ops/intrinsics/` | Intrinsic database (`x86.json`, built by `build_x86_db.py`) |
| `ops/gen/fspec_gen.py` | Generates `forge/ops/ops.gen.hpp` and `ops.md` |
| `ops/gen/gpu_gen.py` | Generates one GLSL compute shader per kernel, compiled to SPIR-V for the Vulkan backend |
| `ops/gen/intrin_gen.py` | Generates C++ intrinsic wrappers gated by CPU feature. Using an intrinsic the target lacks is a compile error |
| `ops/gen/fit_math.py` | Fits the polynomials behind `simd::erf`, `erfc` and `lgamma`, and writes `simd/include/forge/simd/math_coeffs.hpp` |

## The .fspec format
Each `op` line declares one operation:

```
op <name> class <elementwise|reduction|contraction|data_movement|stateful>
   inputs <n> shape <rule> tol <exact|ulp:N|rel:X|abs:X> fuse <yes|no|epilogue>
   cost <flops per output element> [attrs "<comma list>"] doc "<text>"
```

Example from `ops/defs/elementwise.fspec`:

```
op add      class elementwise  inputs 2  shape broadcast  tol exact     fuse yes  cost 1  doc "y = a + b (numpy broadcasting)"
```

## Generated outputs
The CMake target `forge_ops_gen` runs `fspec_gen.py` at build time. It writes into the build tree:
- `forge/ops/ops.gen.hpp`: `enum class op`, the op info table (class, arity, shape rule, tolerance, fusion, cost and docs), and the binding from each op to `forge::ops::ref::<name>`. **If a reference is missing, the build fails with a compile error.**
- `ops.md`: the generated op reference.

## Adding an op
1. Add an `op` line to the right `.fspec` file.
2. Add `forge::ops::ref::<name>` to `reference.hpp`.
3. Rebuild. The oracle tests in `test_graph` (`oracle_matrix_all_ops`) automatically check every lowering registered for the new op.

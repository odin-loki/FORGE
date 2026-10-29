# spec: one definition per operation

## Concepts
Every tensor operation is declared exactly once, in a `.fspec` file, together with a scalar reference implementation. The other layers are generated from or checked against that declaration:
- The **flow** IR uses the generated `enum class op`.
- **Lowerings** (CPU SIMD, Vulkan and the others) must agree with the reference within the declared tolerance. The oracle in `verify/` checks this.
- **GPU kernels** are generated from the same specs.

## Layout

| Path | Contents |
|------|----------|
| `spec/ops/*.fspec` | Op declarations: `elementwise` (40), `reduction` (11), `movement` (8), `contraction` (3) |
| `spec/include/forge/spec/reference.hpp` | Scalar references, `forge::spec::ref::<name>` |
| `spec/intrinsics/` | Intrinsic database (`x86.json`, built by `build_x86_db.py`) |
| `spec/gen/fspec_gen.py` | Generates `forge/spec/ops.gen.hpp` and `ops.md` |
| `spec/gen/gpu_gen.py` | Generates one GLSL compute shader per kernel, compiled to SPIR-V for the Vulkan backend |
| `spec/gen/intrin_gen.py` | Generates C++ intrinsic wrappers gated by CPU feature. Using an intrinsic the target lacks is a compile error |
| `spec/gen/fit_math.py` | Fits the polynomials behind `lane::erf`, `erfc` and `lgamma`, and writes `lane/include/forge/lane/math_coeffs.hpp` |

## The .fspec format
Each `op` line declares one operation:

```
op <name> class <elementwise|reduction|contraction|data_movement|stateful>
   inputs <n> shape <rule> tol <exact|ulp:N|rel:X|abs:X> fuse <yes|no|epilogue>
   cost <flops per output element> [attrs "<comma list>"] doc "<text>"
```

Example from `spec/ops/elementwise.fspec`:

```
op add      class elementwise  inputs 2  shape broadcast  tol exact     fuse yes  cost 1  doc "y = a + b (numpy broadcasting)"
```

## Generated outputs
The CMake target `forge_spec_gen` runs `fspec_gen.py` at build time. It writes into the build tree:
- `forge/spec/ops.gen.hpp`: `enum class op`, the op info table (class, arity, shape rule, tolerance, fusion, cost and docs), and the binding from each op to `forge::spec::ref::<name>`. **If a reference is missing, the build fails with a compile error.**
- `ops.md`: the generated op reference.

## Adding an op
1. Add an `op` line to the right `.fspec` file.
2. Add `forge::spec::ref::<name>` to `reference.hpp`.
3. Rebuild. The oracle tests in `test_flow` (`oracle_matrix_all_ops`) automatically check every lowering registered for the new op.

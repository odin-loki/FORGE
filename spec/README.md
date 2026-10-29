# spec — one definition per op

Each line of a `.fspec` file declares one operation:

```
op <name> class <elementwise|reduction|contraction|data_movement|stateful>
   inputs <n> shape <rule> tol <exact|ulp:N|rel:X|abs:X> fuse <yes|no|epilogue>
   cost <flops per output element> [attrs "<comma list>"] doc "<text>"
```

`spec/gen/fspec_gen.py` turns the files into:

* `forge/spec/ops.gen.hpp` — `enum class op`, the op info table (class, arity, shape rule,
  tolerance, fusion, cost, docs) and a table binding every op to its scalar reference
  `forge::spec::ref::<name>` in `spec/include/forge/spec/reference.hpp`. A missing reference is a
  compile error.
* `ops.md` — generated op reference documentation.
* The oracle (`verify/`) iterates the generated table, so every new op is tested automatically
  against every lowering registered for it.

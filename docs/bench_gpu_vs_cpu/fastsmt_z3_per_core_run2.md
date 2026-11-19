# z3 per core: 20 formulas, 5 reps, pinned to CPU 63 (taskset), Z3 version 4.8.12 - 64 bit, timeout 10 s; load 93, avail 27.1 GiB

| strategy | median wall s (over formulas of per-formula median) | per-formula max/min over reps (median, worst) | answers | machine before / after |
|---|---|---|---|---|
| default | 0.0455 | 1.30, 2.14 | sat 70, unsat 30 | load 93, avail 27.1 GiB / load 91, avail 27.1 GiB |
| qfbv | 0.0512 | 1.35, 1.79 | sat 70, unsat 30 | load 91, avail 27.1 GiB / load 94, avail 27.7 GiB |
| simplify-bitblast-sat | 0.0582 | 1.30, 1.56 | sat 70, unsat 30 | load 94, avail 27.8 GiB / load 98, avail 27.6 GiB |
| smt | 0.0428 | 1.30, 1.49 | sat 70, unsat 30 | load 98, avail 27.6 GiB / load 104, avail 26.9 GiB |
| solveeqs-bitblast-sat | 0.0534 | 1.26, 2.33 | sat 70, unsat 30 | load 104, avail 26.9 GiB / load 107, avail 26.3 GiB |
| propagate-smt | 0.0522 | 1.19, 1.45 | sat 70, unsat 30 | load 107, avail 26.2 GiB / load 112, avail 26.4 GiB |

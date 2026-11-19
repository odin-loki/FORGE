# z3 per core: 20 formulas, 5 reps, pinned to CPU 63 (taskset), Z3 version 4.8.12 - 64 bit, timeout 10 s; load 74, avail 27.3 GiB

| strategy | median wall s (over formulas of per-formula median) | per-formula max/min over reps (median, worst) | answers | machine before / after |
|---|---|---|---|---|
| default | 0.0564 | 1.23, 1.51 | sat 70, unsat 30 | load 74, avail 27.3 GiB / load 72, avail 27.2 GiB |
| qfbv | 0.0634 | 1.24, 1.42 | sat 70, unsat 30 | load 72, avail 27.2 GiB / load 74, avail 26.0 GiB |
| simplify-bitblast-sat | 0.0626 | 1.22, 1.45 | sat 70, unsat 30 | load 74, avail 26.0 GiB / load 74, avail 27.6 GiB |
| smt | 0.0480 | 1.19, 1.51 | sat 70, unsat 30 | load 74, avail 27.6 GiB / load 76, avail 24.7 GiB |
| solveeqs-bitblast-sat | 0.0666 | 1.19, 1.48 | sat 70, unsat 30 | load 76, avail 24.7 GiB / load 78, avail 26.7 GiB |
| propagate-smt | 0.0524 | 1.22, 1.41 | sat 70, unsat 30 | load 78, avail 26.7 GiB / load 79, avail 26.5 GiB |

# Model compression (`forge::compress`, `forge compress`, `forge import`)

FORGE runs optimised models. `forge import` optimises a float Hugging Face checkpoint while importing it, and
`forge compress` applies any chain of compression techniques to a `.fpk`. Everything runs on FORGE:

- calibration passes are FORGE block graphs over the model's live weights
- the maths uses FORGE's thread pool, SIMD tables and GEMM
- evaluation uses `forge::inference::session`

The code is in `compress/include/forge/compress/` (header-only, CMake target `forge_compress`). The tests are in
`tests/test_compress_persist.cpp` (scratch buffers, checkpoints),
`tests/test_compress.cpp` and `tests/compress_import.cmake`, and the experiment plan is `tools/compress_plan.cpp`
(ForgeScript).

## Commands

```sh
forge import <hf_dir> -o model.fpk                      # optimised by default (data-free pipeline `optimise`)
forge import <hf_dir> -o model.fpk --codec q4_k         # optimised into one codec (rotation + GPTQ + calibration)
forge import <hf_dir> -o model.fpk --no-optimise        # as stored (also FORGE_OPTIMISE=0, or --codec f32|f16|bf16)
forge import <hf_dir> -o model.fpk --compress RECIPE [--calib-data FILE|self]
forge compress model.fpk -o small.fpk --recipe RECIPE [--calib-data FILE|self] [--nsamples 64] [--seqlen 512]
               [--codec C] [--report r.json] [--data DIR --ppl-windows 40 --tasks arc_easy,... --limit 200 --mix 1]
forge eval model.fpk --data DIR [--ppl-windows 40] [--tasks ...] [--limit 200] [--mix 1] [--report r.json]
```

`forge run <hf_dir>` caches an optimised import in the same way.

Calibration data:

- `--calib-data self` (the default) uses text the model writes itself. It is sampled with FORGE's continuous
  batching, so no dataset is needed.
- A `.jsonl` file of `{"text": ...}` documents, or a plain text file, can be given instead.
- `tools/compress_data.py` prepares WikiText-2, C4 and the zero-shot tasks for `forge eval`.

Whenever a recipe needs feedback (`optimise`, `calibrate`, a `kl=` target), `forge compress` first records the
uncompressed model's next-token distributions on separate self-generated probe sequences (`--probes 8`,
`--probe-len 256`).

## Recipes

A recipe is `stage(key=value,...)+stage(...)`. Each stage is calibrated on the model as the earlier stages left it,
layer by layer, with the calibration hidden states propagated through the layers that are already compressed.

| stage | what it does |
|---|---|
| `prune` | unstructured or N:M sparsity. `method=magnitude\|wanda\|ria\|sparsegpt`, `s=`, `nm=2:4`. `refit=mcr`: masked conjugate-gradient least-squares refit. `alloc=owl\|eba`: per-layer / per-projection sparsity. `nm=2:4,alloc=eba,frac=F`: 2:4 only on the cheapest projections |
| `width` | removes FFN neurons. `score=mag\|act\|obs` (structured OBS). Least-squares refit of `w_down`; on sparse models the refit keeps the mask. `alloc=eba` |
| `depth` | removes attention / FFN sublayers or whole layers. `score=bi\|probe` (block influence or measured loss). `replace=linear,rank=r`: closed-form low-rank linear replacement of a removed FFN |
| `experts` | drops MoE experts (`score=freq\|gate\|reap`) |
| `merge` | merges experts. `method=avg`, or `method=ls`: concatenate both experts' neurons, select with OBS, least-squares refit to the pair's gated output |
| `moe2dense` | converts MoE layers to dense FFNs. `width=` (fraction of E·f neurons). `w_down` is solved against the true MoE output. `max_err=` keeps the layers that do not convert well |
| `quant` | `method=rtn\|gptq,codec=C` for any FORGE codec. `method=mixed`: a codec per tensor (Lagrangian knapsack over `ladder=q4_k/q5_k/q6_k/q8_0`), with a KL feedback search `kl=` |
| `rotate` | randomised Hadamard rotation of the residual stream and the value heads. Exact, data-free, done after folding the norm gains. A tied LM head stays tied through a d × d `head_mix` |
| `equalize` | SwiGLU up/down and V/O channel balancing. Exact, data-free |
| `calibrate` | output temperature (folded into `out_norm`) and logit bias, fitted to the reference distributions. Each is kept only if it lowers the held-out KL |
| `optimise` | the default pipeline: `rotate` (dense models), `equalize`, mixed `quant` with KL target `kl=0.005`, `calibrate`. `codec=C` uses one codec instead of the mixed allocation |
| `none` | no change (storage only, e.g. `--codec s24_f16`) |

### Codecs

`s24_f16`, `s24_q8` and `s24_q4` store 2:4-sparse rows: 16 kept values and their 2-bit positions per 32 weights.
They take 36, 22 and 14 bytes per block. The `--codec` picker and the quant stages use them only on tensors that
really are 2:4 sparse; other tensors get the dense codec of the same value format.

Tied embeddings are stored once.

## Evaluation and the context mixer

`forge eval --mix 1` also reports perplexity with an online context mixer, after PAQ and cmix. Six predictors are
mixed:

- the network's distribution
- a sharpened and a flattened copy of it
- a unigram cache and a bigram cache
- a match model

The mixing weights are learned online, one weight set per context (match-length bucket × bigram seen), with Bayesian
updates and fixed-share smoothing. The mixer only uses past tokens, so it is a valid online predictor.

## Long runs: memory and checkpoints

Big buffers live in memory-mapped scratch files (`compress/include/forge/compress/mapped.hpp`), so under memory
pressure the OS writes them back to the disk and drops them instead of killing the job. This covers the model's
tensors, the calibration hidden states, the Gram / Hessian matrices and the reference log-probabilities.

- **Lazy loading.** `Model::load` keeps every tensor in the mapped source file. An f32 tensor is read in place, and
  other codecs are decoded on first access. A tensor gets its own (scratch-backed) copy only when it is first written.
- **Scratch files.** They are named `forge-compress-<pid>-<n>.tmp`. They are created next to the output (or in
  `FORGE_COMPRESS_SCRATCH`) and preallocated, so a full disk fails the allocation instead of crashing later. On POSIX
  they are unlinked at once; on Windows they are deleted when freed or at exit.
- **Thresholds.** `FORGE_COMPRESS_MMAP_MIN` (default 64 MiB; `off` = all in RAM) for compute buffers, and
  `FORGE_COMPRESS_MMAP_WEIGHTS_MIN` (default 1 MiB) for model weights. `FORGE_COMPRESS_LAZY=0` decodes every tensor
  at load.

**Checkpoints.** `forge compress` saves its progress in `--checkpoint DIR` (default `<out>.ckpt/`, `none` = off),
`compress/include/forge/compress/checkpoint.hpp`:

- After every stage it writes the model (`stage<i>.fpk`, f32) and `state.json` (the finished stages and their
  reports, the per-tensor codecs, and the calibration and reference data).
- Inside the sequential stages (`prune`, `width`, `depth replace=linear`, `experts`, `merge`, `moe2dense`, `quant`
  gptq, and the error-curve and GPTQ passes of `quant method=mixed`) it saves the finished layers. This happens every
  `--checkpoint-every N` layers or `--checkpoint-minutes M` (default 10). It writes the layers' tensors, spec and
  codecs, the loop's accumulators, and the hidden states entering the next layer.
- Rerunning the same command resumes from the last checkpoint and writes the same file as an uninterrupted run. The
  job is identified by the input file, the recipe and the calibration options; another job clears the directory.
- The directory is removed when the job (including `--data` evaluation) is done, unless `--keep-checkpoint` is given.
- New sequential stages opt in with `layer_resumed` / `layer_finished` (`compress.hpp`).
- `FORGE_COMPRESS_ABORT_AFTER=<layer>|<phase>:<layer>|stage:<i>` aborts after that point is saved (tests and drills).
  `FORGE_COMPRESS_ABORT_OOM=1` makes it a `std::bad_alloc`.

**Supervision.** `tools/compress_plan_supervise.sh` (POSIX sh) and `.ps1` (Windows) keep the plan running until it
writes `plan.done`. They restart it when it dies and kill it when its heartbeat goes stale. The plan
(`tools/compress_plan.cpp`) does the rest:

- It holds an OS lock (`plan.lock`), so a second instance exits with status 75.
- It rewrites `plan.heartbeat` at least every 30 s.
- It runs every job with `--checkpoint`, so a restarted job resumes mid-way.
- It retries failed jobs with exponential backoff.
- On an out-of-memory kill (exit 137, `std::bad_alloc`) it retries at once with `--nsamples` and `--probes` halved.
- It passes `COMPRESS_MEM_LIMIT` / `FORGE_MEM_LIMIT` on to the jobs.

## Notes

- **Rotation** densifies every matrix that reads or writes the residual stream. It is therefore skipped on sparse
  models.
- **Evaluation** of quantised models runs FORGE's exact quantised kernels only with `FORGE_CPU_INT8=0`. The default
  int8-activation path deviates by up to about 1.6 % (TODO.md).
- **Real-model results** are recorded by `tools/compress_plan.cpp` in the AI-Optimisation repository
  (`results/REPORT.md`).

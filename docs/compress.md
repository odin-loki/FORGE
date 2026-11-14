# Model compression (`forge::compress`, `forge compress`, `forge import`)

FORGE runs optimised models. `forge import` optimises a float Hugging Face checkpoint while importing it, and
`forge compress` applies any chain of compression techniques to a `.fpk`. Everything runs on FORGE:

- calibration passes are FORGE block graphs over the model's live weights
- the maths uses FORGE's thread pool, SIMD tables and GEMM
- evaluation uses `forge::inference::session`

The code is in `compress/include/forge/compress/` (header-only, CMake target `forge_compress`). The tests are in
`tests/test_compress.cpp`, `tests/test_compress_runtime.cpp` and `tests/compress_import.cmake`, and the experiment plan is `tools/compress_plan.cpp`
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
           [--kv-codec int8|fp8|tbq4|tbq3|k8v4] [--draft small.fpk --spec-tokens 64 --spec-prompts 8 --spec-prompt-len 128 --spec-k 4]
forge run model.fpk --prompt TEXT --mix 1                # generation from the context-mixed distribution
```

`forge run <hf_dir>` caches an optimised import in the same way.

Calibration data:

- `--calib-data self` (the default) uses text the model writes itself. It is sampled with FORGE's continuous
  batching, so no dataset is needed.
- `--self-mode jacobi [--jacobi-passes 6]` samples it with full-sequence prefills instead (`jacobi_generate`, a Jacobi
  fixed-point iteration of autoregressive sampling with Gumbel-coupled per-position noise; its fixed point is exactly
  the autoregressive sample). Prefill rows are cheaper than decode steps, but real models settle only about one
  position per pass (qwen3-0.6b, 96 tokens: ~75 % of the positions change in every pass, NLL 8.1 nats/token after
  6 passes against 1.8 for decoded text), so decoding stays the default.
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

The mixing weights are learned online, one weight set per context (match-length bucket × bigram seen; a set used for
the first time starts from the nearest shorter-match set), with Bayesian updates and fixed-share smoothing. The mixer
only uses past tokens, so it is a valid online predictor.

The same models (`mixer.hpp`) also give the full mixed next-token distribution (`StreamMixer`: a dense part for the
network's three components, sparse parts over the tokens the caches have seen). `forge run --mix 1` samples from it
through the sampler's logit filter (temperature, top-k and top-p then apply to it); the prompt trains the weights.

## KV-cache codecs and speculative decoding

- `forge eval --kv-codec C` evaluates with a quantised KV cache (`session_options::kv_codec`); the report records it.
- `forge eval --draft small.fpk` measures greedy speculative decoding (FORGE's `speculative_generate`, `--spec-k`
  drafts per pass, 0 = planner) with that draft against plain greedy decoding of the model, on `--spec-prompts`
  WikiText windows of `--spec-prompt-len` tokens, `--spec-tokens` new tokens each. The report's `speculative` entry
  has the acceptance rate, tokens per pass, decode tokens/s of both and their ratio, and how many outputs were
  identical (greedy verification is lossless).

## Notes

- **Rotation** densifies every matrix that reads or writes the residual stream. It is therefore skipped on sparse
  models.
- **Evaluation** of quantised models runs FORGE's exact quantised kernels only with `FORGE_CPU_INT8=0`. The default
  int8-activation path deviates by up to about 1.6 % (TODO.md).
- **Real-model results** are recorded by `tools/compress_plan.cpp` in the AI-Optimisation repository
  (`results/REPORT.md`).

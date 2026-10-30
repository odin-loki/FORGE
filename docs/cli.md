# The `forge` command-line tool

Source: `apps/forge.cpp`. Run `forge` with no arguments for the built-in usage text.

Options are `--name value`, `--name=value`, or a single-letter `-x value`. Unknown commands print the usage and exit.

## Commands

### `forge info`
Prints the CPU name and feature flags, the lane kernel table in use, and for each grid backend (cpu, cuda, hip, level_zero, vulkan, metal) whether its runtime was found. It ends with physical and available RAM, any cgroup or `RLIMIT_AS` limit, and the host memory budget with its source (see [memory.md](memory.md)).

### `forge import <hf_dir | model.gguf | model.onnx> -o out.fpk [--codec C] [--template NAME]`
Converts a model to `.fpk`.
- `--codec` (default `f32`): `f32 f16 bf16 fp8_e4m3 fp8_e5m2 q8_0 q4_k q4_g32 q8_g32 mxfp8 mxfp6 mxfp4 mxint8 nvfp4 tq2`.
- `--template`: blueprint template from `import/templates/` (`llama`, `mixtral`, `deepseek_v2`); normally chosen from the model's architecture; the import fails with a hint to pass `--template` when none matches. Templates are read from the source tree path compiled into the binary; set `FORGE_TEMPLATES` to use another directory (for example when running an installed binary).

### `forge inspect <model.fpk> [--verify] [--tensors] [--blueprint]`
Prints the manifest, the section table (offset, size, BLAKE3 prefix) and a blueprint summary. `--tensors` lists every tensor, `--blueprint` prints the graph text, `--verify` checks hashes. Works on encrypted files without the key (metadata stays readable).

### `forge run <model> (--prompt TEXT | --tokens 1,2,3) [options]`
`<model>` may be a `.fpk`, a Hugging Face directory, a `.gguf` or an `.onnx` file; non-`.fpk` inputs are imported once into `$FORGE_CACHE` (default `~/.cache/forge`) using `--codec`.

| Option | Default | Meaning |
|--------|---------|---------|
| `-n N` | 32 | New tokens to generate |
| `--temp` | 0 | Temperature (0 = greedy) |
| `--top-k`, `--top-p`, `--min-p` | 0, 1, 0 | Sampling filters (values shown are "off") |
| `--repeat-penalty` | 1 | Repetition penalty |
| `--seed` | 0 | RNG seed |
| `--ctx N` | 0 | Context length; 0 = model maximum |
| `--ids` | off | Print token ids instead of text |
| `--no-bos` | off | Do not prepend the BOS token |
| `--logits FILE` | | Write raw f32 logits |
| `--verify` | off | Check BLAKE3 hashes on open |
| `--grammar FILE.gbnf` | | Constrain output with a GBNF grammar |
| `--draft draft.fpk [--spec-k 4]` | | Speculative decoding with a draft model |

Models whose blueprint session is not `generative_sequence` (for example imported ONNX graphs) run as one-shot inference instead: pass `--input name=file.f32` (raw little-endian f32, repeatable) and optionally `--out PREFIX` to write `PREFIX.<output>.f32`; without `--out` the outputs are printed.

### `forge bench <model.fpk> [-p 32] [-n 32]` / `forge bench --gemm 512`
Measures prefill (`-p` prompt tokens) and decode (`-n` tokens) throughput, or an N×N GEMM.

### `forge dump <model.fpk> <tensor> <out.f32>`
Decodes one tensor to raw f32.

### `forge keygen -o NAME`
Writes an Ed25519 key pair: `NAME.key` (secret seed, keep private) and `NAME.pub`.

### `forge sign <model.fpk> --key NAME.key [-o out.fpk]`
Adds a signature (or appends one to an existing signature section).

### `forge verify <model.fpk> [--pubkey NAME.pub]...`
Checks every section hash and the signatures. Prints `VERIFIED` and exits 0, or prints the failure and exits 1. Without `--pubkey` any valid signature is accepted (integrity only), so an unsigned file fails; with `--pubkey`, at least one valid signature must come from a listed key.

### `forge encrypt <in.fpk> -o out.fpk (--key K.mkey | --new-key K.mkey)`
Seals the tensor data with XChaCha20-Poly1305. `--new-key` creates the key file. Load encrypted models with `FORGE_MODEL_KEY_FILE=K.mkey` or `FORGE_MODEL_KEY=<hex>`.

### `forge rpc-serve [--port 7000] [--bind 127.0.0.1] [--device cpu]`
Exposes a local device to other processes as a remote `rpc:host:port` device (see [flow.md](flow.md)). **No authentication: use only on a trusted network.** (This command exists but is not listed in the built-in usage text.)

## Strict loading
`FORGE_REQUIRE_SIGNED=1` refuses unsigned models; `FORGE_TRUSTED_KEYS=<hex>[,<hex>]` only accepts signatures from those public keys.

## Verified session
These were run against the golden model on Linux x86_64:

```sh
B=build/dev
$B/forge info
$B/forge inspect $B/golden/tiny_f32.fpk
$B/forge run $B/golden/tiny_f32.fpk --prompt "Hello" -n 8
$B/forge bench $B/golden/tiny_f32.fpk -p 8 -n 8
$B/forge keygen -o dk
$B/forge sign $B/golden/tiny_f32.fpk --key dk.key -o s.fpk
$B/forge verify s.fpk --pubkey dk.pub                 # ... s.fpk: VERIFIED
$B/forge encrypt s.fpk -o sealed.fpk --new-key k.mkey
FORGE_MODEL_KEY_FILE=k.mkey $B/forge inspect sealed.fpk
```

## `forge script`

```
forge script <file.cpp> [args...]     # run a script; int main()'s return value is the exit status
forge script -e '<code>' [args...]    # run a snippet
forge script                          # interactive REPL (:help, :type, :dis, :mem, :gc, :vm on|off)
  --tree          tree-walker only (default: compile to verified bytecode)
  --debug         race detector + lock-order / stall deadlock detection
  --timeout MS    wall-clock limit      --max-steps N   instruction budget
  --heap BYTES    managed heap cap      --no-proc / --no-fs / --no-net   capability sandbox
```

Scripts are an interpreted, memory-safe subset of C++23; see [scripting.md](scripting.md).
Arguments after the file are available as `forge::args()`.

# The `forge` command-line tool

Source: `apps/forge.cpp`. Run `forge` with no arguments for the built-in usage text. The usage text is shorter than this page; where they differ, this page follows the code.

## Concepts
- `forge <command> [arguments]`. Commands: `info`, `import`, `requant`, `shard`, `adapter`, `blueprint`, `inspect`, `run`, `aot`, `bench`, `dump`, `keygen`, `sign`, `verify`, `encrypt`, `rpc-serve`, `security`, `script`.
- Options (all commands except `script`) are `--name value`, `--name=value` or `-x value`. Positional arguments can go anywhere.
- These options are flags and never take a value: `--verify`, `--tensors`, `--blueprint`, `--license`, `--allow-unused`, `--ids`, `--no-bos`, `--chat`, `--allow-scripts`, `--script-no-proc`, `--script-no-fs`, `--script-no-net`, `--json`, `--detached`. Any other option takes the next argument as its value, so an option such as `--pure` must be last or written `--pure=1`.
- Unknown options are ignored. An unknown command, or a missing required argument, prints the usage and exits 2. Errors print `forge: error: <message>` and exit 1.
- Wherever a model path is taken by `run` and `bench`, it may be a `.fpk`, a Hugging Face directory, a `.gguf` or an `.onnx` file. Non-`.fpk` inputs are imported once into `$FORGE_CACHE` (default `~/.cache/forge`) with `--codec` (default `f32`) and reused while the source files are unchanged.

## Commands

### `forge info`
Prints the CPU name and feature flags (`cpu`, `features`), the simd kernel table in use (line `lane`), every kernel table this CPU can run, preferred first (line `targets`; pass one to `FORGE_SIMD_TARGET`), and one `grid` line per GPU runtime: `cpu`, `cuda`, `hip`, `level_zero` and `vulkan` show whether the runtime library could be loaded (not whether a device works); `metal` always shows `no`. It ends with physical and available RAM, any cgroup or `RLIMIT_AS` limit, and the host memory budget with its source (see [memory.md](memory.md)).

### `forge import <hf_dir | model.gguf | model.onnx> -o out.fpk [options]`
Converts a model to `.fpk`.

| Option | Meaning |
|--------|---------|
| `--codec C` | Weight codec: `f32 f16 bf16 fp8_e4m3 fp8_e5m2 q8_0 q4_k q5_k q6_k q4_g32 q8_g32 mxfp8 mxfp6 mxfp4 mxint8 nvfp4 tq2 q4_0 q4_1 q5_0 q5_1 q2_k q3_k iq4_nl iq4_xs tq1_0 tq2_0`, the mixed-precision presets `q4_k_m` / `q5_k_m` (Hugging Face inputs), or `keep`. Default `f32` for plain Hugging Face checkpoints. For a GGUF file or an AWQ / GPTQ checkpoint the default is `keep`: the tensors stay in the quantisation they are stored in; another value decodes and re-quantises them and prints a double-quantisation warning ([modelfile.md](modelfile.md#gguf-tensor-types)) |
| `--emb-codec C` | Codec for the token embedding (default `q8_0` when `--codec` is quantised, else the same as `--codec`) |
| `--template NAME` | Blueprint template from `import/templates/`: `llama`, `qwen2`, `mixtral`, `deepseek_v2`, `qwen3`, `qwen3_5`, `qwen3_moe`, `qwen3_5_moe`. Normally chosen from the model's architecture |
| `--allow-unused` | Import although the source has tensors the template does not use. Without it such an import fails and lists them (name and shape). Hugging Face leftovers are left out; GGUF leftovers are kept as `extra.<name>` |
| `--ignore-tensor NAME[,NAME...]` | Source tensors known to be harmless: a whole name, `prefix*` or `*suffix`. `position_ids`, `masked_bias` and `inv_freq` buffers are always ignored |
| `--align 64\|4096\|65536` | File alignment of every tensor (`4k`, `64k` also parse). Default: 4096 when the source weights are at least 64 MiB, else 64; see [format.md](format.md#tensor-alignment) |
| `--shard-size SIZE [--blob-dir DIR]` | Write a sharded model ([format.md](format.md#shards)): the import goes to `out.fpk.unsharded`, is split and removed. `0` = one blob per tensor (models in one store share equal tensors); `2G`, `512M` = packed blobs of at most that size |
| `--calib` | Record statistics of the full-precision weights in a `calib` section, for a later `forge requant` (Hugging Face sources; a GGUF's weights are already quantised, so nothing is recorded) |
| `--recipe forge\|llama` | Mixed precision: `llama` is a port of llama.cpp's per-tensor rules; `forge` (default) adds rules for DeltaNet and attention-output projections |
| `--output-tensor-type T`, `--token-embedding-type T` | Mixed precision: codec for the output head / token embedding |
| `--tensor-type REGEX=TYPE[,REGEX=TYPE...]` | Mixed precision: per-tensor overrides |
| `--pure` | Mixed precision: no per-tensor rules, one codec for all |
| `--bf16-head` | Mixed precision: keep the output head in bf16 |

`forge import` prints the licence it recorded (`license   apache-2.0`, or `(none found in the source)`); see [modelfile.md](modelfile.md#what-an-import-checks-and-records). The mixed-precision options are documented for Hugging Face inputs. Templates are read from `FORGE_TEMPLATES` if set, else from `../share/forge/templates` relative to the executable (an installed binary), else from the source tree compiled into the binary.

### `forge blueprint <model.fpk> --to binary|text [-o out.fpk]` / `forge blueprint <file> --to binary|text|canonical -o <out>`
Stores a model's blueprint in the binary form, or back as text ([format.md](format.md#binary-blueprint)). Without `-o` the model file is rewritten in place. A signature and precompiled plans are dropped. Given a bare blueprint file (`.fbp` text or `.fbpb` binary) it converts that file to `-o`; `canonical` is the fixed-layout text without comments.

### `forge inspect <model.fpk> [--verify] [--tensors] [--blueprint [text|binary|canonical] [-o FILE]] [--license] [--state]`
`--blueprint` (or `--blueprint text`) prints the blueprint as text: the stored text, or the canonical text of a binary section. `--blueprint binary` prints a summary of the binary form (version, size against the text, opset, name, session, the names it calls) and with `-o FILE` writes its bytes; `--blueprint canonical` prints the fixed-layout text.

Prints the manifest, the section table (offset, size, BLAKE3 prefix), a blueprint summary (with a note when its opset is outside the range this build runs) and a `license` line: the licence recorded at import, or `(not recorded)`. `--license` also prints the stored licence text. `--tensors` lists every tensor with its codec and shape, `--blueprint` prints the graph text, `--verify` checks every hash and exits 1 on a mismatch. Works on encrypted files without the key (metadata stays readable).
It also prints the tensor alignment and how many tensors comply, the state slots the file declares (`--state` lists them), every embedded adapter (name, rank, alpha, number of target weights), a summary of the calibration section, and for an encrypted file the scheme, the key id and whether a key was accepted.

### `forge requant <in.fpk> -o out.fpk --codec C [options]`
Re-encodes the float weights of a model file in another codec, without the source checkpoint: rank-2 tensors stored as `f32`, `f16` or `bf16` that are not normalisation weights or LoRA factors. Every other tensor is copied byte for byte (a tensor that is already quantised is not decoded and quantised again). `--codec` and the mixed-precision options are those of `forge import`.

| Option | Meaning |
|--------|---------|
| `--emb-codec C`, `--recipe`, `--output-tensor-type`, `--token-embedding-type`, `--tensor-type`, `--pure`, `--bf16-head` | As for `forge import` |
| `--no-calib` | Do not use calibration statistics |
| `--calib-crest X` | Crest factor (`absmax / rms`) at which the `forge` recipe gives a tensor one step more precision (default 12; two steps at twice that) |
| `--align 64\|4096\|65536` | Tensor alignment of the output (default: the input's) |

With calibration (the default) the statistics come from the input's `calib` section; without one they are computed from the input's `f32` weights. The output carries them. The other sections are copied; a signature and precompiled plans are dropped, an encrypted input needs its key and the output is plain. Files with an `experts` index (stacked MoE experts) are refused: import the source again. See [modelfile.md](modelfile.md#re-quantising-a-model-file).

### `forge shard split <in.fpk> -o out.fpk [--shard-size SIZE] [--blob-dir DIR]` / `join <in.fpk> -o out.fpk` / `list <model.fpk>`
`split` moves the tensors of a single-file model into content-addressed blobs and writes the small main file ([format.md](format.md#shards)); without `--shard-size` (or with `0`) every tensor is its own blob. It prints how many blobs it wrote and how many were already in the store. `join` writes one file again (every blob must be present; each tensor is checked against the index). `list` prints each blob with its size, tensor count and path, or `MISSING`. Both drop a signature and precompiled plans. Set `FORGE_BLOB_DIR` to share one store between models; otherwise blobs live in `blobs/` next to the model.

### `forge adapter list <model.fpk>` / `add <model.fpk> --name N --from lora.fpk [-o out.fpk]` / `remove <model.fpk> --name N [-o out.fpk]`
LoRA adapters carried in the model file. `add` copies the `<weight>.lora_a` / `<weight>.lora_b` tensors of `lora.fpk` (the layout `model::add_adapter_file` reads; `lora_alpha` from its manifest) into the model under the name `N`; `remove` takes one out; without `-o` the file is rewritten in place. Select one with `forge run --adapter N`. Both drop a signature and precompiled plans, refuse an encrypted file, and hold the whole file in memory while they work.

### `forge run <model> (--prompt TEXT | --tokens 1,2,3) [options]`

| Option | Default | Meaning |
|--------|---------|---------|
| `-n N` | 32 | New tokens to generate |
| `--temp T` | 0 | Temperature (0 = greedy) |
| `--top-k`, `--top-p`, `--min-p` | 0, 1, 0 | Sampling filters (the defaults are "off") |
| `--repeat-penalty` | 1 | Repetition penalty |
| `--seed` | 0 | RNG seed |
| `--ctx N` | 0 | Context length; 0 = model maximum |
| `--ids` | off | Print token ids instead of text (always on when the model has no tokenizer) |
| `--no-bos` | off | Do not prepend the BOS token |
| `--chat`, `--system TEXT` | off | Treat `--prompt` as a user message (after an optional system message) and render it with the model's chat template; fails when the file has none. The template writes BOS itself |
| `--logits FILE` | | Write the prefill logits as raw f32 |
| `--verify` | off | Check BLAKE3 hashes on open |
| `--grammar FILE.gbnf` | | Constrain output with a GBNF grammar (needs a tokenizer) |
| `--draft draft.fpk` | | Speculative decoding with a draft model |
| `--lookup [--ngram 3] [--ngram-min 1]` | | Speculative decoding with n-gram (prompt-lookup) drafts |
| `--spec-k N` | 0 | Drafts per pass; 0 = chosen per pass by the planner. Ignored with `--grammar` (plain decoding) |
| `--codec C` | `f32`; `keep` for GGUF and AWQ / GPTQ inputs | Codec for on-the-fly import of non-`.fpk` inputs, as for `forge import` |
| `--adapter NAME` | none | Run with a LoRA adapter carried in the model file (`forge adapter list` shows them) |
| `--allow-unused` | off | On-the-fly import: accept source tensors the template does not use (as `forge import --allow-unused`) |
| `--policy FILE` | | Security policy (see [Security policy](#security-policy)): the model must be readable under `fs.read` and pass `integrity.models` |

Generated text goes to stdout; timing (prompt and new tokens, tok/s, plans compiled and reused) goes to stderr. When plans came precompiled ([`forge aot`](#forge-aot-modelfpk--o-out---aot-mode-plannative-options)) a second line says how many, how many of them ran as native code, and where they came from; `<model>` may also be a compiled model library (`model.so`).

Models whose blueprint session is not `generative_sequence` (for example imported ONNX graphs) run as one-shot inference instead: pass `--input name=file.f32` (raw little-endian f32, repeatable) and optionally `--out PREFIX` to write `PREFIX.<output>.f32`. Without `--out` the outputs are printed. A file named `*.i64` or `*.i32` is read as integers (token ids, masks). A symbolic dimension is worked out from an input's size when it is that input's only unbound one; give the others with `--dim name=N` (the names are in the manifest's `onnx_inputs[].dims`, e.g. `seq`, `dim_batch`).

### `forge embed <model> (--text TEXT [--text-pair TEXT] | --tokens 1,2,3 | --input name=file.f32 ...) [options]`
Runs an embedding model (blueprint `session embedding`: the `bert`, `vit` and `whisper_encoder` templates) once and prints the result vector, or writes it as raw little-endian f32 with `--out file.f32`.

| Option | Meaning |
|---|---|
| `--tokens 1,2,3` | Token ids, including the model's special tokens (`[CLS]` ... `[SEP]`) |
| `--text TEXT` | Tokenise with the model's tokenizer (WordPiece, Unigram, BPE, ...). The tokenizer's post-processor adds the special tokens (`[CLS]` ... `[SEP]`) and sets the token types. `--ids` also prints the ids to stderr |
| `--text-pair TEXT` | With `--text`: a second sequence (`[CLS] A [SEP] B [SEP]`, B in segment 1 for BERT) |
| `--token-types 0,0,1` | Segment ids (default all 0; with `--text`, overrides the tokenizer's) |
| `--mask 1,1,0` | 1 = real token, 0 = padding (default all 1) |
| `--input name=file.f32` | Raw f32 for a declared input, repeatable: `image=...` (`[channels, size, size]`, already resized and normalised), `mel=...` (`[n_mels, frames]`) |
| `--out file.f32` | Write the vector instead of printing it |
| (stderr) | `[aot: N of the plans were precompiled (M native), from ...]` when the model carries precompiled plans that were used ([aot-models.md](aot-models.md)) |
| `--device D`, `--codec C`, `--verify`, `--policy F` | As for `forge run` |

```sh
forge import ./bge-small-en -o bge.fpk
forge embed bge.fpk --text "hello world"
forge embed bge.fpk --tokens 101,7592,2088,102
forge embed vit.fpk --input image=cat.f32 --out cat.emb.f32
```

### `forge aot <model.fpk> [-o out] [--aot-mode plan|native] [options]`
Compiles the model's plans ahead of time. The full description is in [aot-models.md](aot-models.md).

- `--aot-mode plan` (default) stores the plans in the file's `aot` section (in place, or `-o out.fpk`), or in `<model>.fplan` with `--sidecar`. Sessions then skip plan compilation for the precompiled sequence lengths; nothing needs to be passed to `forge run`. `--list` shows the section, `--strip` removes it. A signed input needs `--key NAME.key` (sign the result) or `--drop-signature`.
- `--aot-mode native -o model.so` also generates C++ for the plans and compiles it with the host compiler into a shared library (`--emit shared`, default), a static library (`--emit static`) or an executable (`--emit exe`) that runs the model: `./model --prompt TEXT -n 32`. `forge run model.so` loads a compiled library. `--embed-weights` puts the `.fpk` into the artefact. Compiler and target: `--cxx`, `--target`, `--simd`, `--march`, `--cxxflag`, `--sdk`, `--work`.
- Both: `--shapes 1,8,16` (sequence lengths, default `1,2,4,8,16,32,64`), `--all-rows`, `--kv-codec f32,int8`, `--ctx N`.
- CPU only: `--device` other than `cpu` (or `FORGE_DEVICE`) is an error. These options are flags: `--all-rows`, `--sidecar`, `--strip`, `--list`, `--embed-weights`, `--drop-signature`.

### `forge bench <model> [-p 32] [-n 32] [--warmup 1] [--codec C]` / `forge bench --gemm N`
Measures prefill (`-p` prompt tokens) and decode (`-n` new tokens, always all of them) throughput after `--warmup` untimed runs. With `--gemm N` it instead times an N×N `gemm_nt` with each available simd table (one thread) and with `gpu::prims` (all threads).

On a CUDA or HIP device it also prints driver calls for the prefill and per decode token: kernels launched directly, kernels captured into a CUDA graph (on by default; `FORGE_GRAPHS=0` turns graphs off), graph launches, graph instantiations, kernel-node updates, replayed runs, DtoH/HtoD/DtoD copies and stream syncs. `tools/bench_llm.sh` wraps it for a side-by-side run against llama.cpp.

### `forge dump <model.fpk> <tensor> <out.f32>`
Decodes one tensor to raw little-endian f32.

### `forge keygen -o NAME`
Writes an Ed25519 key pair: `NAME.key` (secret seed, owner read/write only) and `NAME.pub`.

### `forge sign <model.fpk> --key NAME.key [-o out.fpk]`
Adds a signature, or appends one to an existing signature section. Without `-o` the input file is rewritten.

### `forge sign <file> --key NAME.key [-o file.sig] [--name NAME] [--detached]`
Any file that is not a `.fpk` (plugins `.so` / `.dll` / `.dylib`, scripts `.fs` / `.cpp`, ...), or any file with `--detached`: writes the detached signature `<file>.sig` (or `-o`), leaving the file untouched. An existing `.sig` for the same name and content is extended (several signers); one by the same key is replaced. The signature binds the file's base name; `--name` signs it under another name (for a remote script whose `forge::remote` `name` differs). Format: [modelfile.md](modelfile.md#detached-signatures).

### `forge verify <model.fpk> [--pubkey NAME.pub]...`
Checks every section hash and the signatures. Prints `VERIFIED` and exits 0, or prints the failure and exits 1. Without `--pubkey` any valid signature is accepted (integrity only), so an unsigned file fails. With one or more `--pubkey`, at least one valid signature must come from a listed key.

### `forge verify <file> [--sig file.sig] [--pubkey NAME.pub]... [--detached]`
Checks a detached signature: `<file>.sig` (or `--sig`) must name this file's base name and match its BLAKE3 digest and size, and carry a valid signature (from a listed key when `--pubkey` is given). Prints `signature ok key <id>` / `signature BAD key <id>` per signer and `<file>: VERIFIED`, or the reason (`unsigned`, `content does not match the signed digest`, `signature is for 'x', not 'y'`, `not by a trusted key`), exit 0 or 1.

### `forge encrypt <in.fpk> -o out.fpk (--key K.mkey | --new-key K.mkey) [--chunk SIZE] [--single-aead]`
Seals the tensor data with XChaCha20-Poly1305. `--new-key` creates the key file (64 hex digits, mode 0600 where supported). Load encrypted models with `FORGE_MODEL_KEY_FILE=K.mkey` or `FORGE_MODEL_KEY=<hex>`.

- The data is sealed in chunks of `--chunk` plaintext bytes (default `1M`; `4096`, `64k`, `2M`; 64 bytes to 1 GiB), each with its own tag. A model loaded from such a file decrypts a tensor when it is first used, not the file at load; `FORGE_LOCK_WEIGHTS=1` puts the decrypted tensors in locked, dump-excluded memory (best effort: `mlock` can be refused by `RLIMIT_MEMLOCK`).
- `--single-aead` writes the older scheme (one AEAD over all tensor data, decrypted whole into memory at load), for FORGE builds that predate the chunked one.
- `forge inspect` prints the scheme, the key id and whether a key was accepted.
- `forge encrypt` itself holds the whole tensor data in memory while it works.

### `forge security check [--policy FILE] [--json]`
Loads the security policy (`--policy`, `$FORGE_SECURITY_POLICY`, `./forge-security.json`, `<config dir>/forge/forge-security.json`; `$FORGE_SECURITY_PROFILE` picks a base profile) and prints, for each FORGE Security Standard control, its state (`active`, `available`, `emulated`, `missing`, `n/a`, `failed`), backend, level and verdict. It changes nothing. Exits 0 when every required control can be met, 1 when one cannot, 2 when the policy is invalid. With no policy and no profile it reports "no FSS profile active" and exits 0. Details: [security-api.md](security-api.md).

### `forge rpc-serve [options]`
Lends a local device to other processes as a remote `rpc:host:port` device (see [graph.md](graph.md)), and optionally runs remote scripts. Details: [rpc.md](rpc.md).

| Option | Default | Meaning |
|--------|---------|---------|
| `--port N` | 7000 | Listen port |
| `--bind ADDR` | `127.0.0.1` | Listen address. Without authentication on a non-loopback address it prints a warning |
| `--device D` | `cpu` | Device to serve (`FORGE_DEVICE` syntax); falls back to the CPU if unavailable |
| `--token T` | | Accept this bearer token, full access (repeatable); `--token 'allow=device T'` limits it |
| `--token-file F` | | One token per line: `[allow=PERM,...] [ops=...] [max-bytes=N] [script-timeout=MS] <token> [name]`; a line without `allow=` has full access (repeatable). Permissions: `device`, `scripts`, or finer `device.<op class>`, `scripts.run/fs/net/proc` ([rpc.md](rpc.md#authorized_keys-and-per-key-permissions)) |
| `--authorized-keys FILE` | | One key per line: the same options, then `<64-hex key> [comment]`; a line without `allow=` has full access (repeatable) |
| `--pubkey HEX\|FILE` | | Accept this Ed25519 public key (repeatable) |
| `--tls-cert C --tls-key K` | | Serve TLS (needs a build with OpenSSL). Both or neither |
| `--tls-client-ca CA.pem` | | Require a client certificate signed by this CA (mTLS); needs `--tls-cert`/`--tls-key` |
| `--tls-client-map FILE` | | Accept only these client certificates: `[options] sha256:<fingerprint>`, `san:dns\|uri\|email\|ip:<value>` or `cn:<name>` per line; permissions intersect with the token's or key's (repeatable) |
| `--tls-crl PEM` | | Refuse client certificates revoked by these CRLs (repeatable; needs `--tls-client-ca`; re-read when the file changes) |
| `--tls-revoked SHA256\|FILE` | | Refuse the client certificate with this SHA-256 fingerprint, or those listed in a file (repeatable; needs `--tls-client-ca`; files are re-read when they change) |
| `--quiet` | off | No security log on stderr |
| `--allow-scripts` | off | Run scripts sent by authenticated clients |
| `--script-timeout MS` | 60000 | Upper limit per script |
| `--script-mem-mb N` | 2048 | Memory limit per script process |
| `--script-no-proc`, `--script-no-fs`, `--script-no-net` | off | Deny those capabilities to remote scripts |
| `--policy FILE` | | Security policy: the listen address must pass `net_listen`, `--allow-scripts` needs `exec`, and the server calls `lockdown()` after binding, before the first request; script children get the same policy ([rpc.md](rpc.md#security-policy)) |

Without tokens or keys the server is open to anyone who can reach it. Each remote script runs in its own `forge script` child process with the OS sandbox set to `require` (`best-effort` on Windows) when any `--script-no-*` flag is given. Over TLS, token and key authentication are bound to the TLS connection (RFC 9266 `tls-exporter`). Clients take their settings from the `FORGE_RPC_*` variables ([configuration.md](configuration.md#rpc-clients)).

### `forge script`

```
forge script [options] <file.cpp> [args...]   run a script; int main()'s return value is the exit status
forge script [options] -e '<code>' [args...]  run a snippet
forge script [options]                        interactive REPL
forge script --emit <target> <file.cpp> [-o out]   translate to a standalone shell script
forge script --emit-targets                   list the shells that have an emitter
forge script --check [--json] [--strict-check] [--module-dir DIR]... <file.cpp>   static checks only
forge script --aot <out> [--aot-mode native|embed] [--cxx C] [-O<n>] [--target T] <file.cpp>   compile to an executable
```

| Option | Meaning |
|--------|---------|
| `--tree` | Tree-walking interpreter only (default: compile to verified bytecode) |
| `--debug` | Race detector, lock-order and stall deadlock detection |
| `--timeout MS`, `--max-steps N`, `--heap BYTES` | Wall-clock limit, instruction budget, managed heap cap |
| `--mem-limit SIZE` | Hard memory cap (`512M`, `2G`, bytes): managed heap (`OutOfMemoryError`), every child process (OS cap), and on Linux and Windows the `forge script` process itself ([scripting.md](scripting.md), "Resource limits") |
| `--no-proc`, `--no-fs`, `--no-net` | Capability sandbox |
| `--os-sandbox off\|best-effort\|require` | How the capability flags are also enforced by the OS sandbox |
| `--policy FILE` | Security policy: capabilities from the policy, file operations path-scoped to its roots, `lockdown()` before the script runs ([scripting.md](scripting.md), "Security policy") |
| `--no-check` | Skip the static checks that run before a script |
| `--check`, `--json`, `--strict-check`, `--module-dir DIR` | Static checks only. Diagnostics go to stderr, or to stdout as JSON with `--json`. Exit 2 on errors, or on any diagnostic with `--strict-check`. `--module-dir` adds an import directory |
| `--cache-dir DIR`, `--cache-trust HEX`, `--cache-require-signed`, `--cache-sign-seed HEX` | Compiled-script cache: directory, trusted public keys, refuse unsigned entries, signing seed |
| `--ask-model M.fpk`, `--ask-max-tokens N` | Run `forge::ask` and related calls on this model in-process (defaults `$FORGE_ASK_MODEL`, `$FORGE_ASK_MAX_TOKENS` or 256) |
| `--emit TARGET`, `-o FILE` | Translate instead of running. `TARGET` is a shell from `--emit-targets`, `auto` (`$FORGE_SHELL` or the host shell), `android` (mksh), or `ast` |
| `--aot OUT`, `--aot-mode native\|embed\|auto`, `--cxx C`, `-O<level>`, `--target TRIPLE`, `--aot-runtime DIR`, `--aot-keep-source DIR`, `--aot-pch DIR`, `--aot-cxx-flag F`, `--aot-stub FILE`, `--aot-allow-widen` | Compile the script to an executable instead of running it. The capability, limit, sandbox and policy options above are baked into the binary; it accepts them again at run time as leading `--forge-*` arguments, which can only narrow them. See [aot.md](aot.md) |
| `--` | End of options; the next argument is the script file |

- Scripts are an interpreted, memory-safe subset of C++23; see [scripting.md](scripting.md). Arguments after the file are available as `forge::args()`.
- REPL commands: `:help`, `:type`, `:dis`, `:mem`, `:gc`, `:vm on|off`, `:reset`, `:quit`.
- Code drafted by AI calls (`forge::ai_run` / `ai_eval`) runs only after you answer `y` on the terminal.
- `--emit` prints to stdout, or writes `-o FILE` and marks it executable. A construct the target cannot express exits 2 and prints `file:line: emit <target>: <reason>`. See [scripting.md, "Write once, emit anywhere"](scripting.md#write-once-emit-anywhere).
- A bad option or unreadable file prints `forge script: <message>` and exits 2.

## Strict loading
`FORGE_REQUIRE_SIGNED=1` refuses unsigned models; `FORGE_TRUSTED_KEYS=<hex>[,<hex>]` only accepts signatures from those public keys. Both apply to every command that loads a model through `inference`.

## Security policy
`forge run`, `forge bench`, `forge script`, `forge rpc-serve` and `forge-serve` take `--policy FILE`. Without it they use what `forge security check` would: `$FORGE_SECURITY_POLICY`, `./forge-security.json`, `<config dir>/forge/forge-security.json`, or `$FORGE_SECURITY_PROFILE` alone. With none of these (profile `none`) nothing changes. With a policy:

- every model load through `inference` checks `fs.read` for the path and `integrity.models`: `required` needs a valid signature (by one of `integrity.trusted_keys`, if any) and fails with `IntegrityError` / `security::integrity_error` before the blueprint or any tensor is used; `optional` verifies signed files and audits unsigned ones. `FORGE_REQUIRE_SIGNED` / `FORGE_TRUSTED_KEYS` still apply on top;
- `forge script` takes its capabilities from the policy and locks down before running; `integrity.scripts` checks the script file's and its imports' detached `.sig`, and `required` refuses `-e` code ([scripting.md](scripting.md), "Security policy");
- native plugins follow `integrity.plugins` (detached `.sig`) and `integrity.os_code_signing`;
- the servers check the listen address, call `lockdown()` after binding and loading, and audit it ([rpc.md](rpc.md#security-policy), [serve.md](serve.md#security-policy)).

A bad or unreadable policy file is an error (exit 1, or 2 for `forge script`). See [security-api.md](security-api.md).

## Example session
These commands were run against the golden model on Linux x86_64 in an earlier revision; they were not re-run for this revision of the page:

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

`tests/sign_test.py` (ctest `modelfile_signing`) runs `keygen`, `sign`, `verify` and `encrypt` on every test run.

## Platform notes
- `forge encrypt` sets the new key file's mode with `chmod 0600`, which has no effect on Windows.
- `forge rpc-serve` TLS options need a build with OpenSSL (configure prints `FORGE: rpc TLS via OpenSSL ...`).

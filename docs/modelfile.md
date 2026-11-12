# modelfile: the .fpk model container and importers

## Concepts
A `.fpk` file holds everything needed to run a model:
- a JSON **manifest**,
- a **blueprint** (the computation graph as text, `modelfile/blueprint.hpp`),
- a **tokenizer** (Hugging Face style `tokenizer.json`),
- a **tensor index** and the **tensor data**, each tensor in one of the weight codecs of `graph/codecs.hpp`.

Every section and every tensor has a BLAKE3 hash. Files can be **signed** (Ed25519) and their tensor data **encrypted** (XChaCha20-Poly1305), both with the vendored Monocypher. Readers memory-map the file. The byte layout is in [format.md](format.md).

Importers turn Hugging Face checkpoints, GGUF files and ONNX graphs into `.fpk`. Language models get their blueprint from a template in `import/templates/` (`llama`, `qwen3`, `qwen3_5`, `qwen3_moe`, `qwen3_5_moe`, `mixtral`, `deepseek_v2`). MoE templates use the sparse `moe_ffn` op over stacked expert tensors (`layer.N.experts.w_gate|w_up|w_down`, `[n_experts, rows, k]`), indexed in the `experts` section ([format.md](format.md)); HF Mixtral, DeepSeek-V2 and Qwen3-MoE experts are stacked at import.

## API
### `modelfile::writer` (`fpk.hpp`)

| Member | Notes |
|--------|-------|
| `set_manifest(json)`, `set_section(name, bytes)` | |
| `add_tensor(name, shape, span<const float>, codec = "f32")` | Encodes on write with any codec in `graph::codecs::all()`. If a block codec's block size does not divide the element count, the tensor is stored as `f16` |
| `add_raw(name, shape, codec, bytes)` | Stores already-encoded blocks (GGUF blocks, AWQ / GPTQ codes as `q4_g32`) without re-quantising; the byte count must match the codec |
| `write(path)` | Writes to a temporary file and renames it over `path` |

`modelfile::stream_writer(path)` has the same `set_*`/`add_*` calls but writes each tensor to disk as it is added, so peak memory is one tensor. Call `finish()` at the end.

### `modelfile::reader(path, key = nullopt)` (`fpk.hpp`)

| Member | Notes |
|--------|-------|
| `manifest()`, `sections()`, `tensors()`, `entry(name)`, `has_section(name)`, `section_text(name)`, `section_bytes(name)`, `file_size()` | Metadata. `entry` gives `.codec`, `.shape`, `.offset`, `.bytes`, `.blake3` |
| `load(name)` | The tensor decoded to f32 |
| `verify()` | Names of sections and tensors whose hash fails (empty = intact). Tensor hashes are skipped while `locked()` |
| `encrypted()`, `locked()` | `locked()`: encrypted and no key from the constructor, `FORGE_MODEL_KEY` or `FORGE_MODEL_KEY_FILE` |

`modelfile::alignment` is 64; `modelfile::format_version` is 1.

### Others

| Header | Contents |
|--------|----------|
| `tokenizer.hpp` | `bpe_tokenizer`: `encode(text)`, `decode(ids)`, `decode_one(id)`, `token_id(str)`, `vocab_size()`, `bos()`, `eos()`, `add_bos()` (whether a prompt starts with BOS) |
| `sign.hpp` | `generate_keypair()`, `load_secret_key(path)`, `load_public_key(path)`, `sign_file(in, out, key)`, `verify_signature(reader, trusted_keys)` → `signature_report` (`signed_`, `digest_ok`, `trusted`, `valid_keys`, `message`). With no trusted keys, any valid signature counts (integrity only, not provenance) |
| `bundle.hpp` | Ed25519 over a directory: `sign_bundle(root, key)` writes `FORGE-SIGNATURE.json` (path, size, BLAKE3 per file); `verify_bundle(root, trusted)` re-hashes and reports modified, missing and extra files. Also detached `sign_bytes` / `verify_bytes` |
| `blueprint.hpp` | `parse_blueprint(text)`: the blueprint language (params, state slots, `repeat` loops, any spec op by name, composites such as `linear`, `embed`, `heads`, user-defined `op name(args) { ... }`, fixed-size state ops `state[i].conv(...)` / `state[i].delta(...)`) |
| `entropy.hpp` | OS randomness for keys and nonces |

## Importers (`modelfile/include/forge/import/`)

| Source | Function, header | Architectures | Weights |
|--------|------------------|---------------|---------|
| Hugging Face directory (`config.json`, `*.safetensors`, `tokenizer.json`) | `import_hf`, `hf.hpp` (+ `safetensors.hpp`, `hf_quant.hpp`) | `llama` template: Llama, Mistral, Qwen2. `qwen3`: Qwen3 (QK-norm). `qwen3_5`: Qwen3.5 dense text models (Gated DeltaNet + gated full attention, partial RoPE). `mixtral`: Mixtral MoE. `deepseek_v2`: DeepSeek-V2/V3 (absorbed MLA) | Encoded with `--codec`; mixed precision with `q4_k_m` / `q5_k_m`. AWQ and GPTQ checkpoints keep their codes: [AWQ and GPTQ checkpoints](#awq-and-gptq-checkpoints) |
| GGUF v2/v3 | `import_gguf`, `gguf.hpp` | `llama`, `mistral`, `qwen2` → `llama`; `qwen3`; `qwen35` → `qwen3_5`; `qwen3moe` → `qwen3_moe`; `qwen35moe` → `qwen3_5_moe` (untested) | Stored in the type the file uses (`--codec keep`, the default), or re-quantised with `--codec C`: [GGUF tensor types](#gguf-tensor-types) |
| ONNX | `import_onnx`, `onnx.hpp` | Static-shape inference graphs (MLPs, classifiers, encoder blocks) | Initializers become weights |

Importer limits that raise an error:
- HF `deepseek_v2`: no YaRN rope scaling, greedy top-k routing with softmax scoring only.
- HF `qwen3_5` and GGUF `qwen35`: no MoE variants, no full-width RoPE, `hidden_act` must be `silu`, and full-attention layers must repeat every `full_attention_interval` layers.
- GGUF `qwen35`: V heads stored in tiled order (`ssm.v_head_reordered`) are put back in grouped order. Row-wise tensors are permuted as raw blocks. The DeltaNet out projection is decoded, permuted and re-encoded, as q8_0 when the source was quantised.
- ONNX ops: MatMul Gemm Add Sub Mul Div Pow Max Min Relu Sigmoid Tanh Exp Log Sqrt Neg Abs Erf Gelu Softplus Reciprocal Softmax LayerNormalization Transpose Reshape Flatten Identity Constant ReduceMean ReduceSum. One symbolic dimension maps to `seq`. Any other op is rejected by name.

Templates are read from `FORGE_TEMPLATES` if set, else from `import/templates/` in the source tree.

### GGUF tensor types
`gguf_file::type_of` (`gguf.hpp`) is the table. A type whose block layout a FORGE codec shares is copied block for block and decoded at run time by that codec, so the values are the ones the reference dequantiser gives.

| ggml type (id) | Elements / bytes per block | Stored as | Decoder checked against |
|----------------|----------------------------|-----------|-------------------------|
| F32 (0), F16 (1), BF16 (30) | 1 / 4, 2, 2 | `f32`, `f16`, `bf16` | float64 NumPy model (`golden_tiny_llama`, `golden_tiny_qwen3`) |
| Q4_0 (2), Q4_1 (3), Q5_0 (6), Q5_1 (7) | 32 / 18, 20, 22, 24 | `q4_0`, `q4_1`, `q5_0`, `q5_1` | gguf-py `dequantize`, bit for bit: random blocks, and tensors quantised by gguf-py |
| Q8_0 (8) | 32 / 34 | `q8_0` | gguf-py, bit for bit |
| Q2_K (10), Q3_K (11) | 256 / 84, 110 | `q2_k`, `q3_k` | gguf-py `dequantize`, bit for bit: random blocks, and tensors packed by the test's NumPy packer (gguf-py has no K quantiser) |
| Q4_K (12), Q5_K (13), Q6_K (14) | 256 / 144, 176, 210 | `q4_k`, `q5_k`, `q6_k` | gguf-py, bit for bit (random blocks) |
| IQ4_NL (20), IQ4_XS (23) | 32 / 18, 256 / 136 | `iq4_nl`, `iq4_xs` | gguf-py, bit for bit: random blocks, and tensors packed by the test's NumPy packer |
| TQ1_0 (34), TQ2_0 (35) | 256 / 54, 66 | `tq1_0`, `tq2_0` | gguf-py, bit for bit: random blocks, and tensors quantised by gguf-py |
| F64 (28), I8 (24), I16 (25), I32 (26), I64 (27) | 1 / 8, 1, 2, 4, 8 | transcoded to `f32` (I32, I64 and F64 round to the nearest f32) | NumPy cast |
| Q8_K (15) | 256 / 292 | transcoded to `f32` as `d * q` | computed in the test only: gguf-py has no Q8_K dequantiser, and the type is an activation format that model files do not normally contain |
| IQ2_XXS (16), IQ2_XS (17), IQ2_S (22), IQ3_XXS (18), IQ3_S (21), IQ1_S (19), IQ1_M (29) | | not loadable | These need codebook grids (256 to 2048 entries each) that FORGE does not carry; a wrong table would corrupt a model silently, so none is guessed. The import stops with an error that names the type |
| Q8_1 (9), MXFP4 (39) | | not loadable | Q8_1 is an activation format without a reference dequantiser. ggml's MXFP4 layout differs from FORGE's `mxfp4` codec |

- Size checks (`gguf_file::count`, `bytes`): the element count is computed with overflow checks, must be a whole number of blocks, and the blocks must lie inside the file. A file that fails any of them is rejected before a byte of tensor data is read.
- The codecs added for these types run on the CPU through the generic quantised-row path (scalar block decode, SIMD dot). They have no int8 kernel and no device kernel: on a device they follow the f32-copy rule of [graph.md](graph.md#device-execution), CUDA and HIP included. `q8_0`, `q4_k`, `q5_k` and `q6_k` keep their fast paths.
- The manifest records what happened: `gguf_tensor_types` (`{"Q4_0": 9, "F32": 5}`), `transcoded` (`[{from, to, tensors}]`) and, with `--codec`, `codec` and `requantised` (`{codec, emb_codec, tensors, from, double_quantised}`). `forge import` prints the transcoded types and the re-quantisation warning on stderr.

`--codec C` on a GGUF decodes the weight matrices (token embedding, output head, attention and FFN projections, MoE experts) and encodes them as `C`; the token embedding becomes `--emb-codec` (default `q8_0` when `C` is quantised). Norms, routers and the DeltaNet gate, conv and decay tensors keep their type, as do tensors whose rows are not whole blocks of `C` (they become `q8_0` when the row length is a multiple of 32). Re-quantising an already quantised file adds the error of both formats; `forge import` prints a warning that says so. The mixed presets `q4_k_m` / `q5_k_m` are refused for GGUF inputs.

### AWQ and GPTQ checkpoints
A Hugging Face directory whose `config.json` has `quantization_config.quant_method` `awq` or `gptq` stores each quantised Linear as `qweight`, `qzeros`, `scales` (and `g_idx` for GPTQ) instead of `weight`; `W[o][i] = scale[g(i)][o] * (q[i][o] - zero[g(i)][o])`. `hf_quant.hpp` unpacks both layouts:

| | AutoAWQ `WQLinear`, version `gemm` | AutoGPTQ `QuantLinear` |
|---|---|---|
| Bits | 4 | 2, 4 or 8 (3 is refused) |
| `qweight` (int32) | `[in, out/8]`: the eight codes of output rows 8c..8c+7 share a word; code `e` is nibble `{0,4,1,5,2,6,3,7}[e]` | `[in*bits/32, out]`: code `(i, o)` is bits `bits*(i % P)..` of word `[i / P][o]`, `P = 32 / bits` |
| `qzeros` (int32) | `[in/group, out/8]`, same nibble order | `[groups, out*bits/32]`, zero of `(g, o)` at bits `bits*(o % P)..` of word `[g][o / P]`. The original format stores `zero - 1` (read back as `(stored + 1) mod 2^bits`); `checkpoint_format: "gptq_v2"` stores the zero itself |
| `scales` | `[in/group, out]` | `[groups, out]` |
| Groups | `g(i) = i / group_size` | `g_idx[i]` (any assignment, so `desc_act` checkpoints load) |

- **keep** (the default): a Linear at 4 or 8 bits whose groups are runs of whole 32-column blocks (group size a multiple of 32, `g_idx` in order) is stored as `q4_g32` / `q8_g32` blocks with `min = -zero * scale`: the integer codes are kept, nothing is re-quantised. The values differ from the checkpoint's only by the f16 rounding of `min` (at most 2^-11 of `zero * scale`: under 1 % of a quantisation step at 4 bits, up to 12 % at 8 bits) and are identical when `zero * scale` is exact in f16 (symmetric GPTQ). Every other Linear (2 bits, `desc_act` group permutations, row lengths that are not multiples of 32) is decoded to f16. The unquantised tensors are stored as f16, the precision these checkpoints use.
- `--codec C` decodes every quantised Linear to f32 (exactly `scale * (q - zero)`) and encodes it as `C`, with a double-quantisation warning unless `C` is `f32`.
- Refused with an error: AWQ versions other than `gemm`, AWQ without `zero_point`, 3-bit GPTQ, quantised Linears with a non-zero bias. Other `quant_method` values are not interpreted.
- The manifest gets `source_quantisation` (`{method, bits, group_size, linears, stored_as_grouped_codec, mode}`).
- **Verified only against a self-written packer.** The layouts above are implemented from the two exporters' documented formats and tested (`golden_hf_quant`, `test_modelfile`) against checkpoints produced by a NumPy packer written from the same description. No checkpoint written by AutoAWQ, AutoGPTQ or GPTQModel has been imported; if either layout is misread, the import still succeeds and the model computes with wrong weights. Compare a few logits with the exporter's own runtime before relying on such an import.

### Mixed precision (HF only, `quant_recipe.hpp`)
`--codec q4_k_m` or `q5_k_m` chooses a type per tensor. `--recipe llama` is a port of llama.cpp's `llama_tensor_get_type()` rules. `--recipe forge` (default) applies the same rules, then puts Gated DeltaNet in/out projections and `attn_output` at q6_k and the head and token embedding at q8_0 (bf16 with `--bf16-head`). Norms, conv weights, A/dt and the a/b gates stay f32. Overrides: `--output-tensor-type`, `--token-embedding-type`, `--tensor-type REGEX=TYPE[,...]` (first match wins), `--pure`.

### CLI
```sh
forge import <hf_dir | model.gguf | model.onnx> -o out.fpk [--codec C] [--template NAME] [--emb-codec C]
```
Codecs: `f32 f16 bf16 fp8_e4m3 fp8_e5m2 q8_0 q4_k q5_k q6_k q4_g32 q8_g32 mxfp8 mxfp6 mxfp4 mxint8 nvfp4 tq2 q4_0 q4_1 q5_0 q5_1 q2_k q3_k iq4_nl iq4_xs tq1_0 tq2_0`, or `keep`. With a quantised codec, the token embedding defaults to q8_0 (`--emb-codec` overrides). `--codec` defaults to `f32` for plain Hugging Face checkpoints and to `keep` for GGUF files and AWQ / GPTQ checkpoints. `forge run` also accepts an HF directory or `.gguf` directly and caches the import in `~/.cache/forge`.

The importers are tested on small random models built by the test scripts (`verify/golden/tiny_llama.py`, `tiny_moe.py`, `tiny_qwen3.py` (HF and GGUF), `tiny_qwen3_5.py`, `quant_inplace.py`, `gguf_types.py`, `hf_quant.py`, `tests/onnx_test.py`), checked against float64 NumPy or transformers references. `gguf_types.py` imports a model in every loadable GGUF tensor type and compares the tensors with gguf-py's dequantisation and the logits with a forward pass over the dequantised weights. They have not been checked against full-size published checkpoints in the test suite.

## Examples
`docs/examples/modelfile_roundtrip.cpp` (ctest `doc_example_modelfile_roundtrip`):

```cpp
modelfile::writer w;
w.set_manifest(json::object{{"name", "demo"}});
w.add_tensor("w", {16, 16}, a, "f32");
w.add_tensor("wq", {256}, a, "q8_0");      // quantised on write
w.write(path);
modelfile::reader r(path);
bool ok = r.verify().empty();              // empty = every hash matches
auto t = r.load("w");                      // decoded to f32
```

Signing and encryption from the CLI:

```sh
forge keygen -o dk                              # dk.key (secret seed), dk.pub
forge sign model.fpk --key dk.key -o signed.fpk
forge verify signed.fpk --pubkey dk.pub         # prints "signed.fpk: VERIFIED"; exit 1 on failure
forge encrypt signed.fpk -o sealed.fpk --new-key k.mkey
FORGE_MODEL_KEY_FILE=k.mkey forge inspect sealed.fpk --verify
```

Strict loading in inference sessions: `FORGE_REQUIRE_SIGNED=1` refuses unsigned files; `FORGE_TRUSTED_KEYS=<hex>[,<hex>]` also requires a signature from one of those keys.

### Detached signatures

Files that are not `.fpk` (native plugins, scripts, anything else) are signed with a detached signature next to them, `<file>.sig` (FSS INT-2, `security/include/forge/security/integrity.hpp`). The same keys as for models (`forge keygen`):

```sh
forge sign libmyplugin.so --key dk.key          # -> libmyplugin.so.sig (the plugin is not modified)
forge sign job.fs --key dk.key                  # -> job.fs.sig
forge verify job.fs --pubkey dk.pub             # job.fs: VERIFIED; exit 1 on failure
```

`<file>.sig` is one JSON object:

```json
{"format":"forge-detached-sig","version":1,"alg":"ed25519","hash":"blake3-256",
 "file":"job.fs","size":31,"blake3":"<64 hex>",
 "signatures":[{"key_id":"<8 hex>","public_key":"<64 hex>","sig":"<128 hex>"}]}
```

Each signature is Ed25519 (RFC 8032, Monocypher) over the 32-byte digest

```
BLAKE3( "forge-detached-sig-v1" || 0x00 || u64le(len(name)) || name || u64le(size) || BLAKE3(file bytes) )
```

- The base name is signed, so a signed file cannot be swapped for another file signed by the same key (putting `b.so` and `b.so.sig` in place of `a.so` fails: "signature is for 'b.so', not 'a.so'"). The cost: a renamed file must be signed again (or signed with `--name`). Names compare exactly on Linux / BSD and ASCII-case-insensitively on Windows and macOS.
- Verification recomputes the BLAKE3 digest and size from the bytes being loaded and the name from the path being loaded; the `file`, `size` and `blake3` fields are informative.
- Several signers: signing again with another key appends to `signatures` (only while the name and content match); signing again with the same key replaces its entry.
- `key_id` is the first 4 bytes of BLAKE3(public key), as for `.fpk`.
- A `.sig` larger than 64 KiB is rejected.

Under a security policy, `integrity.plugins` and `integrity.scripts` decide whether these signatures are required ([security-api.md](security-api.md#integration)).

## Platform notes and limits
- `FORGE_HUGEPAGES=1` asks for transparent huge pages on mappings of 2 MiB or more.
- In encrypted files only `data` is sealed. The manifest, blueprint, tokenizer, tensor index and section table stay readable, which is why `inspect`, section-hash and signature checks work without the key.
- Signing is cross-checked against pyca/cryptography (`modelfile_signing` ctest).

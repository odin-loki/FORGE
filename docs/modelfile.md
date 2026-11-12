# modelfile: the .fpk model container and importers

## Concepts
A `.fpk` file holds everything needed to run a model:
- a JSON **manifest**,
- a **blueprint** (the computation graph as text, `modelfile/blueprint.hpp`),
- a **tokenizer** (Hugging Face style `tokenizer.json`) and, when the source had one, the model's **chat template**,
- a **tensor index** and the **tensor data**, each tensor in one of the weight codecs of `graph/codecs.hpp`.

Every section and every tensor has a BLAKE3 hash. Files can be **signed** (Ed25519) and their tensor data **encrypted** (XChaCha20-Poly1305), both with the vendored Monocypher. Readers memory-map the file. The byte layout is in [format.md](format.md).

Importers turn Hugging Face checkpoints, GGUF files and ONNX graphs into `.fpk`. Language models get their blueprint from a template in `import/templates/` (`llama`, `qwen3`, `qwen3_5`, `qwen3_moe`, `qwen3_5_moe`, `mixtral`, `deepseek_v2`). MoE templates use the sparse `moe_ffn` op over stacked expert tensors (`layer.N.experts.w_gate|w_up|w_down`, `[n_experts, rows, k]`), indexed in the `experts` section ([format.md](format.md)); HF Mixtral, DeepSeek-V2 and Qwen3-MoE experts are stacked at import.

## API
### `modelfile::writer` (`fpk.hpp`)

| Member | Notes |
|--------|-------|
| `set_manifest(json)`, `set_section(name, bytes)` | |
| `add_tensor(name, shape, span<const float>, codec = "f32")` | Encodes on write with any codec in `graph::codecs::all()`. If a block codec's block size does not divide the element count, the tensor is stored as `f16` |
| `add_raw(name, shape, codec, bytes)` | Stores already-encoded blocks (e.g. GGUF Q4_K) without re-quantising; the byte count must match the codec |
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
| `tokenizer.hpp` | `tokenizer` (`bpe_tokenizer` is the old name for it): see [Tokenizers](#tokenizers) |
| `sign.hpp` | `generate_keypair()`, `load_secret_key(path)`, `load_public_key(path)`, `sign_file(in, out, key)`, `verify_signature(reader, trusted_keys)` → `signature_report` (`signed_`, `digest_ok`, `trusted`, `valid_keys`, `message`). With no trusted keys, any valid signature counts (integrity only, not provenance) |
| `bundle.hpp` | Ed25519 over a directory: `sign_bundle(root, key)` writes `FORGE-SIGNATURE.json` (path, size, BLAKE3 per file); `verify_bundle(root, trusted)` re-hashes and reports modified, missing and extra files. Also detached `sign_bytes` / `verify_bytes` |
| `blueprint.hpp` | `parse_blueprint(text)`: the blueprint language (params, state slots, `repeat` loops, any spec op by name, composites such as `linear`, `embed`, `heads`, user-defined `op name(args) { ... }`, fixed-size state ops `state[i].conv(...)` / `state[i].delta(...)`) |
| `entropy.hpp` | OS randomness for keys and nonces |

## Importers (`modelfile/include/forge/import/`)

| Source | Function, header | Architectures | Weights |
|--------|------------------|---------------|---------|
| Hugging Face directory (`config.json`, `*.safetensors`, `tokenizer.json` or a SentencePiece `tokenizer.model`, `tokenizer_config.json`) | `import_hf`, `hf.hpp` (+ `safetensors.hpp`, `tokenizer_import.hpp`) | `llama` template: Llama, Mistral, Qwen2. `qwen3`: Qwen3 (QK-norm). `qwen3_5`: Qwen3.5 dense text models (Gated DeltaNet + gated full attention, partial RoPE). `mixtral`: Mixtral MoE. `deepseek_v2`: DeepSeek-V2/V3 (absorbed MLA) | Encoded with `--codec`; mixed precision with `q4_k_m` / `q5_k_m` |
| GGUF v2/v3 | `import_gguf`, `gguf.hpp` | `llama`, `mistral`, `qwen2` → `llama`; `qwen3`; `qwen35` → `qwen3_5`; `qwen3moe` → `qwen3_moe`; `qwen35moe` → `qwen3_5_moe` (untested) | Copied verbatim: F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K. `--codec` is not applied |
| ONNX | `import_onnx`, `onnx.hpp` | Static-shape inference graphs (MLPs, classifiers, encoder blocks) | Initializers become weights |

Importer limits that raise an error:
- HF `deepseek_v2`: no YaRN rope scaling, greedy top-k routing with softmax scoring only.
- HF `qwen3_5` and GGUF `qwen35`: no MoE variants, no full-width RoPE, `hidden_act` must be `silu`, and full-attention layers must repeat every `full_attention_interval` layers.
- GGUF `qwen35`: V heads stored in tiled order (`ssm.v_head_reordered`) are put back in grouped order. Row-wise tensors are permuted as raw blocks. The DeltaNet out projection is decoded, permuted and re-encoded, as q8_0 when the source was quantised.
- ONNX ops: MatMul Gemm Add Sub Mul Div Pow Max Min Relu Sigmoid Tanh Exp Log Sqrt Neg Abs Erf Gelu Softplus Reciprocal Softmax LayerNormalization Transpose Reshape Flatten Identity Constant ReduceMean ReduceSum. One symbolic dimension maps to `seq`. Any other op is rejected by name.

Templates are read from `FORGE_TEMPLATES` if set, else from `import/templates/` in the source tree.

## Tokenizers
`modelfile::tokenizer` (`tokenizer.hpp`) reads the `tokenizer` section, a Hugging Face `tokenizer.json`, and runs the `tokenizers` pipeline stage by stage:

| Stage | Supported |
|-------|-----------|
| Added tokens | `special`, `normalized`, `lstrip`, `rstrip`, `single_word`; leftmost match, the longest at that position. Tokens with `normalized: false` are found in the raw text, the others in the normalized text |
| Normalizer | `NFC` `NFD` `NFKC` `NFKD` `Lowercase` `Strip` `StripAccents` `Replace` `Prepend` `Nmt` `BertNormalizer` `Precompiled` (SentencePiece charsmap) `Sequence` |
| Pre-tokenizer | `ByteLevel` (`add_prefix_space`, `use_regex`) `Split` (string or regex, every `behavior`, `invert`) `Whitespace` `WhitespaceSplit` `Punctuation` `Digits` `Metaspace` (`prepend_scheme`, `split`) `BertPreTokenizer` `CharDelimiterSplit` `Sequence` |
| Model | `BPE` (merges, `unk_token`, `fuse_unk`, `byte_fallback`, `ignore_merges`, `continuing_subword_prefix`, `end_of_word_suffix`), `WordPiece`, `Unigram` (Viterbi, `byte_fallback`), `WordLevel`, and the FORGE model type `SentencePieceBPE` (SentencePiece's score-driven BPE; see [format.md](format.md#tokenizer-section)) |
| Post-processor | `TemplateProcessing` (single sequence), `BertProcessing`, `RobertaProcessing`, `ByteLevel`, `Sequence` |
| Decoder | `ByteLevel` `Metaspace` `WordPiece` `BPEDecoder` `ByteFallback` `Fuse` `Strip` `Replace` `Sequence` |

Anything else in a `tokenizer.json` (another component type, a regex the engine cannot compile) raises `unsupported` when the tokenizer is built, with the name of the component; nothing is skipped silently. Patterns are compiled by `forge::os::text::regex` in its Oniguruma dialect, and Unicode classes, normalization forms and grapheme clusters come from the same library (`forge_text`, which `forge_modelfile` links; Unicode 16).

| Member | Notes |
|--------|-------|
| `encode(text)` | Ids without BOS / EOS. Added tokens written in the text (`<|im_start|>`) become their ids. Input that is not valid UTF-8 is read with U+FFFD for the bad bytes |
| `encode(text, true)`, `with_special(ids)` | Also what the model expects around a prompt: BOS / EOS per `add_bos()` / `add_eos()` when the session set them from the manifest, else the `tokenizer.json` post-processor (`[CLS] … [SEP]`, `<s> …`) |
| `decode(ids, skip_special = false)` | The decoder chain over the whole sequence. Bytes of an incomplete UTF-8 character are passed through, not replaced |
| `decode_one(id)` | The bytes one token adds in the middle of a text (streaming, grammar masks). It differs from `decode` where the decoder looks at the position: a SentencePiece word keeps its leading space, WordPiece puts a space before each word-initial piece |
| `token_id(str)`, `id_to_token(id)`, `is_special(id)`, `vocab_size()`, `type()`, `bos()`, `eos()`, `unk()`, `add_bos()`, `add_eos()`, `has_post_processor()` | |
| `normalize(text)`, `pre_tokenize(text)` | The first two stages on their own, for debugging a mismatch |

Sources other than a ready `tokenizer.json` are converted to one at import (`import/tokenizer_import.hpp`):

- **SentencePiece `.model`** (`tokenizer.model`, `spiece.model`, `sentencepiece.bpe.model` in a Hugging Face directory without `tokenizer.json`): read with a small protobuf reader. BPE models become `SentencePieceBPE`, unigram models `Unigram`; the model's normalization (precompiled charsmap, whitespace collapsing, dummy prefix) becomes the normalizer chain, and control / user-defined pieces plus `tokenizer_config.json`'s `added_tokens_decoder` become added tokens.
- **GGUF** `tokenizer.ggml.model`: `gpt2` → byte-level BPE split by the regex of `tokenizer.ggml.pre` (`llama-bpe`, `qwen2`, `deepseek-llm`, `deepseek-coder`, `deepseek-v3`, `gpt-2`, `tekken`, `gpt-4o`, `falcon`, `starcoder`, … as in llama.cpp; an unknown name falls back to GPT-2's with a warning); `llama` → `SentencePieceBPE` over `tokenizer.ggml.scores`; `t5` → `Unigram` (with `precompiled_charsmap`, `remove_extra_whitespaces`); `bert` → `WordPiece`. `tokenizer.ggml.token_type` 3 (control) and 2 (unknown) become special added tokens, 4 (user-defined) plain added tokens, 6 (byte) switches byte fallback on. `add_bos_token` / `add_eos_token` go to the manifest, `add_space_prefix` into the normalizer. A `tokenizer.huggingface.json` key is used as it is. Other models (`rwkv`, …) give a file without a tokenizer, with a warning.

How this is tested, and what is not: `tests/test_tokenizer.cpp` compares ids, normalizer and pre-tokenizer output and decoded text with goldens made by the `tokenizers` 0.23 and `sentencepiece` 0.2 packages (`tests/data/tokenizer/make_fixtures.py`) for small tokenizers trained on a multilingual corpus (ASCII, digit runs, CJK, emoji, combining marks, right-to-left text, mixed whitespace, control characters, invalid UTF-8). It has not been run against the tokenizers of full-size published models. The llama.cpp pre-tokenizer regexes and its BERT conventions were written from memory of llama.cpp and are only checked for `llama-bpe`, `qwen2` and `gpt-2` (against `tokenizers`) and on ASCII text for `bert`. llama.cpp's per-model token attributes (`lstrip` / `rstrip` on some special tokens) are not carried over. Unicode tables are FORGE's (Unicode 16); a reference built against another Unicode version can differ on characters assigned in between.

The **chat template** (`tokenizer_config.json` `chat_template`, `chat_template.jinja`, or GGUF `tokenizer.chat_template`) is stored as the `chat_template` section and rendered by `inference::apply_chat_template` ([inference.md](inference.md#chat-templates-chat_templatehpp)).

### Mixed precision (HF only, `quant_recipe.hpp`)
`--codec q4_k_m` or `q5_k_m` chooses a type per tensor. `--recipe llama` is a port of llama.cpp's `llama_tensor_get_type()` rules. `--recipe forge` (default) applies the same rules, then puts Gated DeltaNet in/out projections and `attn_output` at q6_k and the head and token embedding at q8_0 (bf16 with `--bf16-head`). Norms, conv weights, A/dt and the a/b gates stay f32. Overrides: `--output-tensor-type`, `--token-embedding-type`, `--tensor-type REGEX=TYPE[,...]` (first match wins), `--pure`.

### CLI
```sh
forge import <hf_dir | model.gguf | model.onnx> -o out.fpk [--codec C] [--template NAME] [--emb-codec C]
```
Codecs: `f32 f16 bf16 fp8_e4m3 fp8_e5m2 q8_0 q4_k q5_k q6_k q4_g32 q8_g32 mxfp8 mxfp6 mxfp4 mxint8 nvfp4 tq2`. With a quantised codec, the token embedding defaults to q8_0 (`--emb-codec` overrides). `forge run` also accepts an HF directory or `.gguf` directly and caches the import in `~/.cache/forge`.

The importers are tested on small random models built by the test scripts (`verify/golden/tiny_llama.py`, `tiny_moe.py`, `tiny_qwen3.py` (HF and GGUF), `tiny_qwen3_5.py`, `quant_inplace.py`, `tests/onnx_test.py`), checked against float64 NumPy or transformers references. They have not been checked against full-size published checkpoints in the test suite.

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

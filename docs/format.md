# .fpk file format (version 1)

The authoritative description is the header comment of `modelfile/include/forge/modelfile/fpk.hpp`; this page restates it. All integers are little-endian.

```
[header 64 B] [sections, each aligned to 64 B] [section table]

header : "FORGEPK1" | u32 version | u32 n_sections | u64 table_offset | u64 file_size | 32 B reserved
table  : n_sections x { char name[32]; u64 offset; u64 size; u8 blake3[32]; u8 pad[16] }   (96 B per entry)
```

Readers find sections through the table, so section order on disk does not matter. `writer` writes the other sections in name order, then `data`. `stream_writer` (used by the Hugging Face importer) writes `data` first, then the other sections, and patches the header last.

## Sections

| Name | Required | Contents |
|------|----------|----------|
| `manifest` | yes | JSON. Every importer writes `format` (`"fpk"`), `forge_format_version`, `architecture` and `session`. Optional licence fields, written when the source states them: `license` (identifier, e.g. `apache-2.0`), `license_name`, `license_link`, `license_files` (names of the files whose text is in the `license` section). HF imports add `template`, `codec`, `bos_token_id`, `eos_token_id`, `max_position_embeddings`, and `source_quantisation` for AWQ / GPTQ checkpoints. GGUF imports add `source: "gguf"`, `template`, the same token/position fields, `gguf_tensor_types`, and `transcoded` / `codec` / `requantised` when the import changed a tensor's type ([modelfile.md](modelfile.md#gguf-tensor-types)). Both add `add_bos` (bool) when the source says whether prompts start with BOS (GGUF `tokenizer.ggml.add_bos_token`, HF `tokenizer_config.json` `add_bos_token`); without it the tokenizer's post-processor decides if it has one, else Qwen-family models get no BOS and others do. `add_eos` (bool) likewise for a trailing EOS (`add_eos_token`). ONNX imports add `source: "onnx"`, `producer`, `onnx_opset`, `onnx_inputs`, `onnx_outputs` |
| `blueprint` | yes | Graph text (`modelfile/blueprint.hpp`): name, opset, params, states, inputs, graph, outputs and session kind (e.g. `generative_sequence`). The engine runs opsets 1 to 2 and refuses any other, naming it; opset 2 adds the `rope` frequency-scaling attrs ([modelfile.md](modelfile.md#opsets)) |
| `tokenizer` | text models | Hugging Face style `tokenizer.json`: see [Tokenizer section](#tokenizer-section). HF imports copy it (or convert a SentencePiece `.model`); GGUF imports synthesise one from the GGUF vocabulary |
| `chat_template` | no | The model's Jinja chat template as UTF-8 text, from `tokenizer_config.json` / `chat_template.jinja` / GGUF `tokenizer.chat_template`. Older readers ignore the section; files without it have no template |
| `tensors` | yes | JSON array `[{name, codec, shape, offset, bytes, blake3}]`; `offset` is relative to the start of `data` |
| `data` | yes | Tensor bytes, each tensor aligned to 64 B, in the codec's block layout (`graph/codecs.hpp`) |
| `experts` | no | Minor revision 1 (manifest `forge_format_minor: 1`). JSON `{version, align: 4096, tensors: [{layer, proj (gate/up/down), name, codec, n_experts, offset, file_offset, expert_bytes, aligned}]}` indexing the stacked MoE expert tensors `[n_experts, rows, k]` read by `moe_ffn`. Expert e is bytes `[offset + e * expert_bytes, + expert_bytes)` of `data`; `aligned` means every expert slice starts on a 4 KiB file offset (one direct read per expert, groundwork for expert streaming). Older readers ignore the section |
| `license` | no | The model's licence text as UTF-8, copied at import (Hugging Face `LICENSE*`, `LICENCE*`, `COPYING*`, `NOTICE*` files; an ONNX `doc_string` that mentions a licence). With several files each follows a `===== NAME =====` line. Never sealed; covered by the signature like every other section; `forge sign`, `forge encrypt` and `forge aot` copy it unchanged. No format revision: readers that predate it ignore the section |
| `aot` | no | Precompiled plans written by `forge aot` (binary, see [The aot section](#the-aot-section)). Older readers ignore the section; a reader that cannot use it compiles plans as usual |
| `signature` | no | `{"alg":"ed25519", "scheme":"forge-fpk-v1", "digest", "signatures":[{key_id, public_key, sig}]}` |
| `encryption` | no | `{alg, nonce, mac, key_id}`. `data` is sealed with XChaCha20-Poly1305; the `tensors` JSON is bound as associated data |

## Tokenizer section
The section is the text of a `tokenizer.json`. Two versions exist; readers tell them apart by the keys present.

| Version | Written by | Recognised by | Read as |
|---------|-----------|---------------|---------|
| 1 | GGUF imports before the tokenizer pipeline existed | a `BPE` model with none of the keys `normalizer`, `pre_tokenizer`, `decoder`, `forge_tokenizer_version` | As before: byte-level BPE behind the old ASCII-class approximation of the GPT-2 pre-tokenizer, added tokens taken in list order, raw bytes kept. One difference: an added token that follows a space is now found (the old splitter consumed its first character). Re-import the GGUF to get the pre-tokenizer the model was trained with |
| 2 | HF imports (a verbatim `tokenizer.json`, then as now); GGUF and SentencePiece conversions, which add `"forge_tokenizer_version": 2` | any other file | The full pipeline ([modelfile.md](modelfile.md#tokenizers)) |

A verbatim Hugging Face file imported by an older FORGE is a version 2 section and is now tokenized by its own normalizer and pre-tokenizer, which is where its ids can change from what the old approximation produced. A `forge_tokenizer_version` above 2 is refused.

FORGE's additions to the `tokenizer.json` vocabulary, all optional and only written next to `forge_tokenizer_version`:

- `model.type: "SentencePieceBPE"`: SentencePiece's BPE. `vocab` is `[[piece, score], ...]` as for `Unigram`, with `unk_id` and `byte_fallback`; encoding merges the adjacent pair whose concatenation is the best-scoring piece until none is a piece.
- `model.token_types`: one SentencePiece / GGUF type per piece (1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte), for `SentencePieceBPE` and `Unigram`. Control, unknown and byte pieces are never produced from text; unused pieces merge but are split again.
- `"forge_match": "longest"` on a `Precompiled` normalizer: SentencePiece's own charsmap lookup (longest rule at each position) instead of the `tokenizers` crate's.
- Top-level `bos_id` / `eos_id` (read by version 1 too); the manifest's `bos_token_id` / `eos_token_id` take precedence in a session.

## Codec names
A tensor's codec is the `codec` string of its `tensors` entry, one of the names in `graph/codecs.hpp` ([graph.md](graph.md#weight-codecs)); there are no numeric codec ids in the file. Adding a codec adds a name and does not change `forge_format_version`:
- a reader opens files written before the codec existed unchanged;
- a reader that does not know a codec still opens the file, lists it (`forge inspect`) and checks its hashes and signature, and fails with `unknown codec <name>` when that tensor is loaded or a session is built on it.

`q4_0 q4_1 q5_0 q5_1 q2_k q3_k iq4_nl iq4_xs tq1_0 tq2_0` (the GGUF layouts of the same names) were added after the first files were written; a file that uses them needs a FORGE build that has them.

## Hashing and signing
- Each table entry carries the BLAKE3 hash of its section bytes as stored. Each `tensors` entry carries the hash of its (plaintext) bytes.
- `reader::verify()` checks both. Tensor hashes of an encrypted file can be checked only with the key.
- The signature covers a BLAKE3 digest of the scheme string followed by, for every section except `signature` in name order: name length, name, size and the BLAKE3 of its stored bytes. The digest is recomputed from the bytes on disk; the stored table hashes are not trusted.
- `key_id` is the first 4 bytes (8 hex digits) of the BLAKE3 of the public key.
- Several signatures can be present. `forge sign` on a signed file adds a signature if the digest still matches, and replaces an older signature by the same key.
- A file whose hashes fail `verify()` is not signed (`forge sign` refuses it).

## The aot section
Written by `forge aot` ([aot-models.md](aot-models.md)), read by `graph/include/forge/graph/aot.hpp`, whose header comment is the authoritative description. `str` is a `u32` length followed by the bytes. The same bytes are the content of a sidecar `<model>.fplan` and are embedded in natively compiled models.

```
blob   : "FORGEAOT" | u32 format_version (1) | u32 n_entries | u8 model_key[32] | u32 pipeline_version
         | str forge_version | str simd_target (informational)
         | n_entries x { u8 key[32]; i64 seq; u8 all_rows; str kv_codec; str device; u64 offset; u64 size }
         | entries | u8 blake3[32]                      (BLAKE3 of every byte before it)
entry  : u32 n_nodes x node | u32 n_outputs x { str name; u32 value }
         | u64 arena_bytes; u64 alignment (64); u64 naive_bytes; u32 in_place; u32 aliases
         | n_nodes x { u8 kind (0 arena, 1 external, 2 alias); u64 offset; u64 bytes; u32 root }
         | u32 n_steps x { u8 kind (0 kernel, 1 state_append, 2 state_op); u32 out; str lowering; u8 fuse }
node   : u8 kind (0 input, 1 weight, 2 constant, 3 op, 4 state_append, 6 state_op) | str op_name
         | u32 n_in, u32 in[] | attrs | u32 rank, i64 dim[] | str name | u8 weight_kind (0 none, 1 f32, 2 encoded) | str codec
         | constants only: u64 n, f32[n]
attrs  : u32 n x { str key; u8 type (0 f64, 1 i64 list: u32 n, i64[n], 2 str); value }
```

- `model_key` = BLAKE3 of `"forge-aot-plan"`, the format version, the pipeline version, the FORGE version, BLAKE3(`blueprint` section) and BLAKE3(`tensors` section). An entry's `key` = BLAKE3(`model_key`, device, KV codec, `seq`, `all_rows`). A reader uses an entry only when the key it computes for its own model and request equals the stored one.
- Weights are referenced by tensor name; lowerings by registry name (`lane.gemm_nt`, `cpu.qgemm_nt`, ...). Offsets are in bytes from the start of the plan's arena. Nothing in the blob is an address.
- Readers treat the blob as untrusted: see [aot-models.md](aot-models.md#security-of-precompiled-plans) for what is checked.
- `forge aot` rewrites the file with every other section at its old offset and the `aot` section after them. It drops a `signature` section, which would no longer match.

## Encryption
- Only `data` is sealed. The manifest, blueprint, tokenizer, tensor index and section table stay readable, so `forge inspect`, section-hash checks and signature checks work without the key.
- Readers take the key from the constructor, or from `FORGE_MODEL_KEY` (hex) or `FORGE_MODEL_KEY_FILE`. Without one the reader opens in the `locked()` state, and reading `data` raises an error naming the key id.

## Tools
- `forge inspect <file> [--tensors] [--blueprint] [--verify]` shows the layout.
- `forge aot <file> --list` shows the precompiled plans and whether they would be used; `--strip` removes them.
- `forge dump <file> <tensor> out.f32` extracts one tensor as raw f32.
- `forge keygen`, `forge sign`, `forge verify`, `forge encrypt`: see [modelfile.md](modelfile.md#examples).

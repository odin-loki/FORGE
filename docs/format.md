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
| `manifest` | yes | JSON. Every importer writes `format` (`"fpk"`), `forge_format_version`, `architecture` and `session`. Optional licence fields, written when the source states them: `license` (identifier, e.g. `apache-2.0`), `license_name`, `license_link`, `license_files` (names of the files whose text is in the `license` section). HF imports add `template`, `codec`, `bos_token_id`, `eos_token_id`, `max_position_embeddings`, and `source_quantisation` for AWQ / GPTQ checkpoints. GGUF imports add `source: "gguf"`, `template`, the same token/position fields, `gguf_tensor_types`, and `transcoded` / `codec` / `requantised` when the import changed a tensor's type ([modelfile.md](modelfile.md#gguf-tensor-types)). Both add `eos_token_ids` (a list) when the source names more than the one EOS id that ends generation: HF `generation_config.json` `eos_token_id` (an id or a list; else a list in `config.json`, whose first id then also becomes `eos_token_id`), GGUF `tokenizer.ggml.eos_token_id` / `eot_token_id` / `eom_token_id`; chat front ends stop at any of them. Both add `add_bos` (bool) when the source says whether prompts start with BOS (GGUF `tokenizer.ggml.add_bos_token`, HF `tokenizer_config.json` `add_bos_token`); without it the tokenizer's post-processor decides if it has one, else Qwen-family models get no BOS and others do. `add_eos` (bool) likewise for a trailing EOS (`add_eos_token`). ONNX imports add `source: "onnx"`, `producer`, `onnx_opset`, `onnx_inputs`, `onnx_outputs` (per value: `onnx` name, blueprint `name`, `shape` with -1 for a symbolic dimension, and since the importer tracks several symbolic dimensions `dims` (the same as blueprint expressions: `"seq"`, `"dim_batch"`, `"8"`) and `dtype` (`"float"` or `"int"`); files without `dims` have at most one symbolic dimension, `seq`) |
| `blueprint` | yes | The graph program, as text (`modelfile/blueprint.hpp`) or in the [binary form](#binary-blueprint) (`blueprint_bin.hpp`; the section then starts with `FORGEBPB`): name, opset, params, states, inputs, graph, outputs and session kind (e.g. `generative_sequence`). The engine runs opsets 1 to 4 and refuses any other, naming it; opset 2 adds the `rope` frequency-scaling attrs, opset 4 session parameters, the `diffusion_loop` / `streaming` session kinds and named entry points ([modelfile.md](modelfile.md#opsets)) |
| `tokenizer` | text models | Hugging Face style `tokenizer.json`: see [Tokenizer section](#tokenizer-section). HF imports copy it (or convert a SentencePiece `.model`); GGUF imports synthesise one from the GGUF vocabulary |
| `chat_template` | no | The model's Jinja chat template as UTF-8 text, from `tokenizer_config.json` / `chat_template.jinja` / GGUF `tokenizer.chat_template`. Older readers ignore the section; files without it have no template |
| `tensors` | yes | JSON array `[{name, codec, shape, offset, bytes, blake3}]`; `offset` is relative to the start of `data` |
| `data` | yes | Tensor bytes in the codec's block layout (`graph/codecs.hpp`), each tensor on a file offset that is a multiple of the file's [tensor alignment](#tensor-alignment) (64 B unless the manifest says otherwise) |
| `experts` | no | Minor revision 1 (manifest `forge_format_minor: 1`). JSON `{version, align: 4096, tensors: [{layer, proj (gate/up/down), name, codec, n_experts, offset, file_offset, expert_bytes, aligned}]}` indexing the stacked MoE expert tensors `[n_experts, rows, k]` read by `moe_ffn`. Expert e is bytes `[offset + e * expert_bytes, + expert_bytes)` of `data`; `aligned` means every expert slice starts on a 4 KiB file offset (one direct read per expert, groundwork for expert streaming). Older readers ignore the section |
| `shards` | sharded files | JSON list of the content-addressed blobs that hold the tensor bytes ([Shards](#shards)). With it, `data` is empty and every `tensors` entry has a `shard` field |
| `state` | no | JSON state schema: the session state slots the model keeps between runs ([State section](#state-section)). Written by importers for models that have state; older readers ignore it |
| `adapters` | no | JSON list of the LoRA adapters carried in the file ([Adapters section](#adapters-section)); their factors are ordinary tensors named `adapter.<name>.<weight>.lora_a` / `.lora_b`. Older readers ignore the section and never read those tensors: they run the base model |
| `calib` | no | JSON calibration statistics: per-tensor range, moments and a magnitude histogram of the full-precision weights ([Calib section](#calib-section)). Older readers ignore it |
| `license` | no | The model's licence text as UTF-8, copied at import (Hugging Face `LICENSE*`, `LICENCE*`, `COPYING*`, `NOTICE*` files; an ONNX `doc_string` that mentions a licence). With several files each follows a `===== NAME =====` line. Never sealed; covered by the signature like every other section; `forge sign`, `forge encrypt` and `forge aot` copy it unchanged. No format revision: readers that predate it ignore the section |
| `aot` | no | Precompiled plans written by `forge aot` (binary, see [The aot section](#the-aot-section)). Older readers ignore the section; a reader that cannot use it compiles plans as usual |
| `signature` | no | `{"alg":"ed25519", "scheme":"forge-fpk-v1", "digest", "signatures":[{key_id, public_key, sig}]}` |
| `encryption` | no | JSON header of the sealed `data` section; two schemes, told apart by `alg`: see [Encryption](#encryption) |
| `enc_macs` | with chunked encryption | Binary: one 16-byte Poly1305 tag per sealed chunk, in chunk order |

## Shards
A sharded model is a small `.fpk` (every section except the tensor bytes) plus blob files that are named by their content:

```
model.fpk       shards : {"version": 1, "layout": "per-tensor" | "packed", "shards": [{"blake3": <64 hex>, "bytes": N}, ...]}
                tensors: [{name, codec, shape, bytes, blake3, "shard": <index into shards>, "offset": <within that blob>}, ...]
                data   : empty
<blob dir>/<first two hex digits>/<blake3 hex>.fpkb      raw bytes; the BLAKE3 of the file is its name
```

- **Layouts.** `per-tensor` (the default of `forge shard split` and of `forge import --shard-size 0`): one blob per tensor, named by the tensor's own hash, offset 0. `packed` (`--shard-size 2G`): tensors in file order packed into blobs of at most that size, each on the file's tensor alignment within the blob; a tensor larger than the size gets a blob to itself.
- **Dedup.** A blob is written only if the store does not already hold a file of that name and size. With the per-tensor layout two models written to one store share every tensor whose encoded bytes are equal: a fine-tune that changes one tensor adds one blob. Packed blobs are shared only when a whole pack is identical.
- **Where blobs are looked for**, in order: the directory named by `FORGE_BLOB_DIR`, then `blobs/` next to the model file. `forge shard split` writes to `--blob-dir`, else `FORGE_BLOB_DIR`, else `blobs/` next to its output.
- **Trust.** The index is the root: a signature on the main file covers the `tensors` and `shards` sections, that is every tensor hash and every blob hash, so blobs carry no signature and can be fetched from anywhere. A reader does not hash blobs at open. It maps a blob when a tensor in it is first used and hashes that tensor's bytes against the index before returning them (`integrity_error` on a mismatch); `reader::verify()` (`forge inspect --verify`, `forge verify`, signing) also hashes every blob whole and reports missing ones. Without a signature the hashes still catch corruption, not a rewritten index.
- **Partial downloads.** A missing blob is an error only when one of its tensors is asked for (`not_found`, naming the blob and the directories searched). `forge shard list` and `forge inspect` show which blobs are present.
- **Untrusted names.** A blob name is used only if it is 64 lowercase hex digits, and its path is built with `security::safe_join`, which also refuses a store entry that is a symlink out of the store. Shard indices and offsets are bounds-checked against the `shards` list; a `shard` field without a `shards` section, or a missing one with it, is a format error.
- **Compatibility.** Sharding is opt-in; single-file models are unchanged. The main file's manifest gets `sharded: true` and `forge_format_minor: 3`. A reader that predates shards stops at open with `tensor <name> out of bounds` (its bounds check against the empty `data` section). `forge shard join` turns a sharded model back into one file.
- **What does not combine**: an encrypted file cannot be sharded and a sharded file cannot be encrypted (sealed tensors are not content-addressable; blobs are plain); `forge adapter add / remove` want the joined file; `forge aot --embed-weights` refuses a sharded model. `forge aot` (plan mode), `forge sign`, `forge blueprint` and `forge requant` work on a sharded model; split drops a signature, precompiled plans and an `experts` index.
- Nothing removes blobs that no model references any more; that is left to whoever owns the store.

## Binary blueprint
A blueprint is a program (params, `repeat` loops, composites, expressions over symbolic dimensions), so its binary form encodes that program: the token stream of the text with names interned and numbers typed. `str` is a varint length followed by the bytes; varints are LEB128 (minimal, at most 10 bytes); fixed-size integers are little-endian.

```
"FORGEBPB" | u32 version (1) | u32 opset | str name | str session
| varint n_strings | n_strings x { u8 kind; str text }   kind 0 identifier, 1 string literal, 2 identifier that is called
| varint n_tokens  | tokens                                      (an op, a composite or a built-in)
token : u8 tag
  0x20 .. 0xff  identifier: string index tag - 0x20 (the first 224 strings)
  0x01          identifier + varint string index (224 and up)
  0x02          string literal + varint string index
  0x03          number with an integral value below 2^53 + varint value
  0x04          any other number + f64 (finite, positive)
  0x10 .. 0x1e  { } ( ) [ ] , : = . + - * / ..
```

- **What a round trip keeps**: every token in order: names, strings, operators, brackets, and numbers as values. **What it drops**: comments, spacing and line breaks, and the spelling of a number (`1e-5` and `0.00001` are the same token).
- **Canonical text.** `canonical_blueprint_text()` prints a token sequence in one fixed layout (one declaration or statement per line, two-space indent, numbers as digits or the shortest `%g` form that reads back as the same double). For every text that lexes, `canonical(text)` equals the text recovered from its binary form, byte for byte. `tests/test_modelfile.cpp` checks it for every template in `import/templates` (placeholders bound) and every seed in `fuzz/corpus/blueprint`; `fuzz_blueprint` checks it for every mutated input.
- **Canonical binary.** The decoder accepts exactly one encoding of a token sequence: strings unique, all used and in first-use order; the kind of a string as its uses say; identifiers in the one-byte tag when their index allows; integral values in tag 0x03; `opset`, `name` and `session` equal to what the tokens declare; no trailing bytes. So for every byte string it accepts, encoding the recovered text gives the same bytes.
- **Loading.** A reader turns a binary section back into canonical text and parses that with the one blueprint parser. The decoder is the untrusted-input boundary in front of it: every length, index and count is checked against the bytes present, and an unknown `version` is refused by number.
- **Compatibility.** Importers write text. `forge blueprint model.fpk --to binary` rewrites the section (and sets manifest `blueprint_form: "binary"`, `forge_format_minor: 2`); `--to text` goes back (to the canonical text: the original comments are gone). A FORGE build that predates the binary form cannot parse it: `run`, `inspect` and anything else that reads the blueprint stop with a blueprint parse error (`unexpected character`, raised by the text lexer on the first byte after the magic); `verify` and `sign`, which do not parse it, still work. Files with a text blueprint are untouched by any of this.
- A signature and precompiled plans cover the blueprint bytes, so converting drops both.
- Size: over the 16 templates and 7 fuzz seeds the binary form is 74 % of the canonical text (a blueprint of a few hundred bytes can come out larger than its text: the header and string table are a fixed cost). It is the smaller form on real templates, not a compressed one.

## State section
```
{"version": 1, "slots": [
  {"name": "kv.0.k", "kind": "append",    "shape": [2, 8],  "dtype": "f32"},
  {"name": "conv.1", "kind": "recurrent", "op": "causal_conv", "shape": [3, 32], "dtype": "f32"}]}
```
- `append`: rows are appended per token; `shape` is one logical row (the appended tensor without its sequence axis: `[heads, dim]` for a KV cache). `recurrent`: fixed-size state updated in place by the kernel `op`; `shape` is the whole buffer. `dtype` is `f32` (a session may store appended rows in a KV codec; the schema describes the logical rows).
- Importers derive it from the blueprint (the graph is built once for one token; `modelfile/state_schema.hpp`) and write it when the model has state. Limits a reader enforces: 65536 slots, names up to 256 bytes, rank 1 to 8, dimensions 1 to 1e9, unique names; another `version` is refused.
- A model whose section differs from what its blueprint builds does not open (`... the state section does not describe the state its blueprint builds`). `forge inspect` and `forge verify` still read such a file.
- A file without the section opens as before; the schema is then derived from the blueprint when it is first needed.
- Use: `session::load_prefix` checks every slot of a saved prefix against the schema (declared name, kind, row shape in the session's KV codec, no duplicates, sizes that fit the blob) before it restores anything.

## Adapters section
```
{"version": 1, "adapters": [{"name": "style", "alpha": 16, "rank": 4, "targets": ["layer.0.wq", "layer.0.wv"]}]}
```
- For adapter `N` and target weight `W` (a rank-2 tensor `[out, in]`) the index holds `adapter.N.W.lora_a` (`[rank, in]`) and `adapter.N.W.lora_b` (`[out, rank]`), PEFT layout: `W' = W + alpha / rank * B·A`. They are tensors like any other: hashed, aligned, sealed by `forge encrypt`, covered by the signature.
- Names: letters, digits, `_`, `-`, at most 64. A reader refuses a section with a duplicate name, a target or factor that is not in the index, or factors whose shapes do not fit the target or `rank` (the model then does not open).
- `forge adapter add / remove` rewrite the container (`modelfile/adapters.hpp`). The base tensors keep their offsets; removed factors stay in `data` as dead bytes. Because the tensor index changes, a signature and an `aot` section are dropped, and an encrypted file is refused: embed, then encrypt, then sign.

## Calib section
```
{"version": 1, "source": "weights", "bins": 64,
 "tensors": {"layer.0.wq": {"n": 1024, "min": -0.41, "max": 0.39, "mean": 0.001, "rms": 0.12, "absmax": 0.41, "hist": [ ...64 counts... ]}}}
```
- One entry per weight that an import quantises (rank-2 projection matrices and the token embedding), computed from the f32 source before encoding. `hist` counts `|x|` in `bins` equal bins over `[0, absmax]`; its sum is `n`. Numbers are written with 9 significant digits.
- Written by `forge import --calib` (Hugging Face sources) and by `forge requant` (carried over, or computed when the input still has f32 weights). Read by `forge requant`: see [modelfile.md](modelfile.md#re-quantising-a-model-file).
- These are statistics of the weights. Activation statistics, which need calibration inputs run through the model, are not collected.
- A reader bounds the entry count, name length and histogram length, and refuses non-finite numbers, a histogram that does not sum to `n` and another `version`.

## Tensor alignment
The manifest key `tensor_alignment` (a power of two from 64 to 1048576) says that every tensor of the file starts on a file offset that is a multiple of it. The writers take 64, 4096 or 65536 (`forge import --align`):

| Value | For |
|-------|-----|
| 64 | Compact files. What every file written before the option has, and what an absent key means |
| 4096 | One tensor = whole pages: it can be mapped, read with direct I/O or advised on its own. FORGE_PLAN's default |
| 65536 | The same for 16 KiB and 64 KiB page systems and for device uploads that want large alignment |

- **Default.** `forge import` writes 4096 when the source weights (the `.safetensors` / `.gguf` / `.onnx` files) total at least 64 MiB, else 64. Padding is half the alignment per tensor on average: measured on the 21-tensor golden test model (449 KiB at 64): 472 KiB at 4096 (+5 %) and 1492 KiB at 65536 (3.3 times). Above the threshold the padding at 4096 is at most 4 KiB per tensor against at least 64 MiB of weights. `modelfile::writer` and `stream_writer` default to 64; `set_alignment` changes it.
- **Layout.** `writer` puts `data` on a multiple of the alignment and tensors on multiples of it within `data`. `stream_writer` keeps `data` at file offset 64 and pads each tensor so that its *file* offset is aligned (so `offset` values in the index are 64 short of a multiple). Tools that rewrite a container (`forge sign`, `forge encrypt`) place `data` at an offset with the same remainder modulo the alignment as before; `forge aot` leaves every section where it was. `forge inspect` prints the alignment and how many tensors comply.
- **Compatibility.** No format revision: readers locate tensors by `offset` and never use the key, so older readers load aligned files unchanged, and a file without the key reads as 64. A file whose index does not match the key (rewritten by a tool that predates the option) still loads; `inspect` reports the count.
- The stacked MoE expert tensors keep their own 4096-byte alignment (`experts` section) when the file's is smaller.

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
- `reader::verify()` checks both. Tensor hashes of an encrypted file can be checked only with the key; with the key, `verify()` on a chunk-encrypted file also authenticates every chunk.
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
- Readers take the key from the constructor, or from `FORGE_MODEL_KEY` (hex) or `FORGE_MODEL_KEY_FILE`. Without one the reader opens in the `locked()` state, and reading tensor data raises an error naming the key id.
- `key_id` is the first 8 bytes (16 hex digits) of the BLAKE3 of the key; a reader given another key refuses the file before it decrypts anything.
- The ciphertext has the size and offsets of the plaintext in both schemes, so the `tensors` index is the same before and after sealing.

Two schemes exist. `forge encrypt` writes the chunked one; both are read.

| | Chunked (written by default) | Single AEAD (`forge encrypt --single-aead`, and every file sealed before the chunked scheme) |
|---|---|---|
| `alg` | `xchacha20-poly1305-chunked` | `xchacha20-poly1305` |
| `encryption` JSON | `{alg, scheme: "forge-fpk-enc-v2", nonce (24 B hex), chunk_bytes, chunks, data_bytes, key_id}` | `{alg, nonce (24 B hex), mac (16 B hex), key_id}` |
| Tags | section `enc_macs`: `chunks` x 16 B | `mac` in the header |
| Unit | `data` is cut into chunks of `chunk_bytes` plaintext bytes (the last one shorter); chunk `i` is sealed on its own | all of `data` in one AEAD |
| Nonce | the file nonce with `i` (little-endian u64) XORed into its last 8 bytes | the file nonce |
| Associated data | `"forge-fpk-enc-v2\0"` (17 B), file nonce (24 B), `i`, `chunks`, `data_bytes`, `chunk_bytes` (u64 LE each), BLAKE3 of the `tensors` section (32 B) | `"forge-fpk-enc-v1\0"` followed by the `tensors` JSON |
| When decrypted | per tensor, on first use: only the chunks the tensor lies in | all of `data`, into heap memory, when the file is opened |

What the chunked scheme's associated data rules out: a chunk moved to another position (index and nonce), a file cut short or extended (`chunks`, `data_bytes`; a reader also compares both with the section sizes before it reads anything), a chunk taken from another file or another sealing of the same file (file nonce), a changed chunk size, and an edited tensor index (its hash). A failed chunk raises an integrity error that names the chunk; nothing of the requested tensor is returned.

Compatibility:
- A file sealed with the single-AEAD scheme opens and behaves as before.
- A reader that predates the chunked scheme opens a chunked file without a key as any locked file (`inspect`, hashes, signature); with a key it stops with `unknown encryption xchacha20-poly1305-chunked`. It ignores `enc_macs`.
- `enc_macs` and `encryption` are covered by the signature like every section; `forge sign` and `forge aot` copy them unchanged.

## Tools
- `forge blueprint` converts a model's blueprint (or a bare `.fbp` / `.fbpb` file) between the forms; `forge inspect --blueprint text|binary|canonical [-o FILE]` shows or extracts one.
- `forge inspect <file> [--tensors] [--blueprint] [--verify] [--state]` shows the layout, the alignment, the declared state, embedded adapters and calibration data.
- `forge adapter list|add|remove`, `forge requant`, `forge shard split|join|list`: see [cli.md](cli.md).
- `forge aot <file> --list` shows the precompiled plans and whether they would be used; `--strip` removes them.
- `forge dump <file> <tensor> out.f32` extracts one tensor as raw f32.
- `forge keygen`, `forge sign`, `forge verify`, `forge encrypt`: see [modelfile.md](modelfile.md#examples).

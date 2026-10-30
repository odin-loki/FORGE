# .fpk file format (version 1)

The authoritative description is the header comment of `pack/include/forge/pack/fpk.hpp`; this page restates it. All integers are little-endian.

```
[header 64 B] [sections, each aligned to 64 B] [section table]

header : "FORGEPK1" | u32 version | u32 n_sections | u64 table_offset | u64 file_size | 32 B reserved
table  : n_sections x { char name[32]; u64 offset; u64 size; u8 blake3[32]; u8 pad[16] }   (96 B per entry)
```

## Sections

| Name | Required | Contents |
|------|----------|----------|
| `manifest` | yes | JSON. Imported language models carry `format`, `forge_format_version`, `architecture`, `template`, `codec`, `session`, `bos_token_id`, `eos_token_id`, `max_position_embeddings`. ONNX imports add `onnx_inputs` |
| `blueprint` | yes | Graph text (`pack/blueprint.hpp`): name, opset, session kind (e.g. `generative_sequence`), params, states and ops |
| `tokenizer` | for text models | Hugging Face style BPE JSON |
| `tensors` | yes | JSON array `[{name, codec, shape, offset, bytes, blake3}]`; `offset` is relative to the start of `data` |
| `data` | yes | Tensor bytes, each tensor aligned to 64 B |
| `signature` | no | `{"alg":"ed25519","scheme":"forge-fpk-v1","digest", "signatures":[{key_id, public_key, sig}]}` |
| `encryption` | no | `{alg, nonce, mac, key_id}`. `data` is sealed with XChaCha20-Poly1305; the tensor index is bound as associated data |

Example section table (`forge inspect build/dev/golden/tiny_f32.fpk`): `blueprint` at offset 64, then `manifest`, `tensors`, `tokenizer`, `data`, each starting on a 64-byte boundary.

## Hashing and signing
- Each table entry carries the BLAKE3 hash of its section; each `tensors` entry carries the hash of its bytes.
- The signature covers a domain-separated BLAKE3 digest of the name, size and hash of every non-signature section, in name order. Verifiers recompute section hashes from the bytes on disk and never trust the stored table.
- Several signatures can be appended (`forge sign` on an already-signed file adds one).

## Tools
`forge inspect <file> [--tensors] [--blueprint] [--verify]` shows the layout; `forge dump <file> <tensor> out.f32` extracts one tensor.

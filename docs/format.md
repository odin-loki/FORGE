# .fpk file format (version 1)

This page is taken from the header comment of `pack/include/forge/pack/fpk.hpp`, which is the authoritative description.

```
[header 64 B] [sections, each aligned to 64 B] [section table]
header : "FORGEPK1" | u32 version | u32 n_sections | u64 table_offset | u64 file_size | 32 B reserved
table  : n x { char name[32]; u64 offset; u64 size; u8 blake3[32]; u8 pad[16] }   (96 B per entry)
```

## Sections

| Section | Contents |
|---------|----------|
| `manifest` | JSON. Generated models include `format`, `forge_format_version`, `architecture`, `template`, `codec`, `session`, `bos_token_id`, `eos_token_id` and `max_position_embeddings` |
| `blueprint` | Graph text in the blueprint language (`pack/blueprint.hpp`) |
| `tokenizer` | Hugging Face style BPE JSON |
| `tensors` | JSON array `[{name, codec, shape, offset, bytes, blake3}]`. Each `offset` is relative to `data` |
| `data` | Tensor bytes. Each tensor is aligned to 64 B |
| `signature` | Optional. JSON `{"alg":"ed25519","scheme":"forge-fpk-v1","digest","signatures":[{key_id,public_key,sig}]}` |
| `encryption` | Optional. `{alg, nonce, mac, key_id}`. In an encrypted file, `data` is sealed with XChaCha20-Poly1305 and the tensor index is bound as associated data |

## Hashing and signing
Every section has a BLAKE3 hash in the table, and every tensor has one in `tensors`.

The signature covers a domain-separated BLAKE3 digest built from the name, size and hash of every non-signature section, in name order. A verifier recomputes the section hashes from the bytes on disk; it never trusts the stored table.

Use `forge inspect <file> [--tensors] [--blueprint]` to see a file's layout.

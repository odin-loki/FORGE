# modelfile: the .fpk model container and importers

## Concepts
A `.fpk` file holds everything needed to run a model:
- a JSON **manifest**,
- a **blueprint** (the computation graph as text, `modelfile/blueprint.hpp`),
- a **tokenizer** (Hugging Face style `tokenizer.json`) and, when the source had one, the model's **chat template**,
- a **tensor index** and the **tensor data**, each tensor in one of the weight codecs of `graph/codecs.hpp`.

Every section and every tensor has a BLAKE3 hash. Files can be **signed** (Ed25519) and their tensor data **encrypted** (XChaCha20-Poly1305), both with the vendored Monocypher. Readers memory-map the file. The byte layout is in [format.md](format.md).

Importers turn Hugging Face checkpoints, GGUF files and ONNX graphs into `.fpk`. Models get their blueprint from a template in `import/templates/`: decoders (`llama`, `qwen2`, `qwen3`, `qwen3_5`, `qwen3_moe`, `qwen3_5_moe`, `mixtral`, `deepseek_v2`, `gpt2`, `gemma`, `gemma2`, `gemma3`, `phi3`) and embedding models (`bert`, `vit`, `whisper_encoder`); see [Template families](#template-families-importtemplatesfbp). MoE templates use the sparse `moe_ffn` op over stacked expert tensors (`layer.N.experts.w_gate|w_up|w_down`, `[n_experts, rows, k]`), indexed in the `experts` section ([format.md](format.md)); HF Mixtral, DeepSeek-V2 and Qwen3-MoE experts are stacked at import.

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
| `tensor_bytes(name)` | The encoded bytes of one tensor. A plain file: a view of the mapping. A chunk-encrypted file: the tensor is decrypted on the first call and kept until the reader closes or `release_plaintext()` |
| `chunked_encryption()`, `plaintext_resident()`, `lock_plaintext(bool)`, `release_plaintext()` | Chunk-encrypted files: the scheme in use, bytes of decrypted tensor data held, whether new ones go into locked memory (default `FORGE_LOCK_WEIGHTS`), wipe and free them |
| `data_size()` | Size of the tensor data without reading it |

`modelfile::alignment` is 64; `modelfile::format_version` is 1.

### Others

| Header | Contents |
|--------|----------|
| `tokenizer.hpp` | `tokenizer` (`bpe_tokenizer` is the old name for it): see [Tokenizers](#tokenizers) |
| `sign.hpp` | `generate_keypair()`, `load_secret_key(path)`, `load_public_key(path)`, `sign_file(in, out, key)`, `verify_signature(reader, trusted_keys)` → `signature_report` (`signed_`, `digest_ok`, `trusted`, `valid_keys`, `message`). With no trusted keys, any valid signature counts (integrity only, not provenance) |
| `bundle.hpp` | Ed25519 over a directory: `sign_bundle(root, key)` writes `FORGE-SIGNATURE.json` (path, size, BLAKE3 per file); `verify_bundle(root, trusted)` re-hashes and reports modified, missing and extra files. Also detached `sign_bytes` / `verify_bytes` |
| `blueprint.hpp` | `parse_blueprint(text)`: the blueprint language (params, state slots, `repeat` loops, any spec op by name, composites such as `linear`, `embed`, `heads`, user-defined `op name(args, k = default) { ... }` with keyword arguments (see [Writing a template](#writing-a-template)), fixed-size state ops `state[i].conv(...)` / `state[i].delta(...)`) |
| `entropy.hpp` | OS randomness for keys and nonces |

## Importers (`modelfile/include/forge/import/`)

| Source | Function, header | Architectures | Weights |
|--------|------------------|---------------|---------|
| Hugging Face directory (`config.json`, `*.safetensors`, `tokenizer.json` or a SentencePiece `tokenizer.model`, `tokenizer_config.json`) | `import_hf`, `hf.hpp` (+ `safetensors.hpp`, `hf_quant.hpp`, `tokenizer_import.hpp`, `hf_arch.hpp` for the families under Template families) | `llama` template: Llama, Mistral. `qwen2`: Qwen2 / Qwen2.5 (the llama graph plus biases on the Q, K and V projections). `qwen3`: Qwen3 (QK-norm). `qwen3_5`: Qwen3.5 dense text models (Gated DeltaNet + gated full attention, partial RoPE). `mixtral`: Mixtral MoE. `deepseek_v2`: DeepSeek-V2/V3 (absorbed MLA) | Encoded with `--codec`; mixed precision with `q4_k_m` / `q5_k_m`. AWQ and GPTQ checkpoints keep their codes: [AWQ and GPTQ checkpoints](#awq-and-gptq-checkpoints) |
| GGUF v2/v3 | `import_gguf`, `gguf.hpp` (+ `gguf_arch.hpp`: `gpt2`, `gemma`, `gemma2`, `gemma3`, `phi3`, `bert`) | `llama`, `mistral` → `llama`; `qwen2` → `qwen2` (`attn_q.bias`, `attn_k.bias`, `attn_v.bias`; a `qwen2` file with no bias tensors → `llama`); `qwen3`; `qwen35` → `qwen3_5`; `qwen3moe` → `qwen3_moe`; `qwen35moe` → `qwen3_5_moe` (untested) | Stored in the type the file uses (`--codec keep`, the default), or re-quantised with `--codec C`: [GGUF tensor types](#gguf-tensor-types) |
| ONNX | `import_onnx`, `onnx.hpp` | Static-shape inference graphs (MLPs, classifiers, encoder blocks) | Initializers become weights |

### What an import checks and records
- **Every source tensor is accounted for** (Hugging Face and GGUF). A tensor the template does not consume fails the import with a list of names and shapes, because dropping weights silently gives a model that runs and answers wrongly. `--allow-unused` (`import_options::allow_unused`) imports anyway: Hugging Face leftovers are left out, GGUF leftovers are kept as `extra.<name>`, and both are listed in `import_report::unused`. `--ignore-tensor` (`import_options::ignore_tensors`) names leftovers that are known to be harmless; `position_ids`, `masked_bias` and `inv_freq` buffers and the vision / multi-token-prediction towers of Qwen3.5 checkpoints are always skipped.
- **Every tensor the template reads exists.** A missing source tensor fails with its name. After the last tensor the importer compares `blueprint_weights(bp)` with what it wrote, in both directions, so a `--template` that does not fit the checkpoint is refused.
- **RoPE frequency scaling** is read from `config.json` (`rope_scaling`, or `rope_parameters` in transformers 5; types `linear`, `dynamic`, `yarn`, `llama3`) and from GGUF (`<arch>.rope.scaling.type` `linear` / `yarn` with `.factor`, `.original_context_length`, `.attn_factor`, `.yarn_attn_factor`, `.yarn_log_multiplier`, `.yarn_ext_factor`, `.yarn_beta_fast`, `.yarn_beta_slow`; the llama3 factors that llama.cpp's converter stores as the `rope_freqs.weight` tensor). It becomes attrs of the `rope` op and, for YaRN's attention factor, `attention`'s `scale`. LongRoPE and unknown types are refused. Templates `llama`, `qwen2`, `qwen3`, `qwen3_moe` and `mixtral` take scaling; `deepseek_v2`, `qwen3_5` and `qwen3_5_moe` refuse a model that asks for it.
- **The licence** is recorded: Hugging Face `README.md` front matter (`license`, `license_name`, `license_link`) and the text of `LICENSE*` / `LICENCE*` / `COPYING*` / `NOTICE*` files (up to 1 MiB each); GGUF `general.license`, `general.license.name`, `general.license.link`; ONNX `metadata_props` (`license`, `license_name`, `license_url`) and a `doc_string` that mentions a licence. It goes into the manifest and the `license` section ([format.md](format.md)), `forge inspect` shows it, and signing and encrypting keep it. A source that states no licence gives a file without these fields; the importer does not guess.

### Opsets
A blueprint declares `opset N` (default 1). This engine runs `modelfile::opset_min` to `opset_max` (1 to 3; 2 added RoPE scaling attrs, 3 the spatial and encoder op families, composite defaults / keyword arguments and `session embedding`). `build_graph` and `inference::model` refuse any other opset with an error that names it and the supported range; an op the engine does not know is reported with its name, line and the blueprint's opset. Opset 2 adds the `rope` attrs `scaling`, `factor`, `orig_ctx`, `low_freq`, `high_freq`, `beta_fast`, `beta_slow`, `truncate` and `freq_factors` (`ops/include/forge/ops/rope.hpp`). The importers write opset 2 only for models that scale their RoPE frequencies; every other import stays opset 1 with the same graph as before. Builds older than this check ignore the opset and unknown attrs, so an opset-2 file gives wrong output there.

The declaration is enforced in both directions: a blueprint has to declare at least the opset of the newest thing it uses. The parser and `build_graph` refuse a use below its opset with a `parse_error` that gives the line, the feature and the opset it needs (`blueprint:12: op 'conv2d' needs opset 3, but blueprint "x" declares opset 1 (write `blueprint "x" opset 3`)`), so a file that says `opset 1` cannot carry something an opset-1 engine lacks. What is checked:

| Needs | Feature | Checked in |
|---|---|---|
| opset 2 | the `rope` scaling attrs listed above (`attr_min_opset`) | `build_graph` |
| opset 3 | the 24 ops of the spatial and encoder families (`op_min_opset`: `conv1d`, `conv2d`, `conv_transpose2d`, the pooling and resize ops, `group_norm`, `batch_norm`, `pad`, `quick_gelu`, `mish`, `hard_swish`, `floor`, `attention_bias`, `alibi_bias`, `sinusoid_embed`, `l2_normalize`, `topk`, `topk_indices`, `index_select`, `masked_mean`) | `build_graph` |
| opset 3 | `scalar()`; the number helpers `sqrt floor ceil min max mod` called on numbers | `build_graph` |
| opset 3 | a default value in a composite's signature (`op f(x, eps = 1e-6)`) | `parse_blueprint` |
| opset 3 | a keyword argument to a composite (`f(x, eps: 1e-5)`); keyword attrs of spec ops are opset 1 | `build_graph` |
| opset 3 | `session embedding` | `parse_blueprint` |

Below opset 3, `sqrt(4)` on a number keeps its old meaning (the `sqrt` op on a constant) instead of being folded. A new op or attr gets a line in `op_min_opset` / `attr_min_opset` (`blueprint.hpp`) with the opset that adds it. Every shipped template declares the opset it needs: the families with `session embedding`, composite defaults or opset-3 ops are `opset 3`; the Llama-style templates take `{{opset|1}}` (2 with RoPE scaling); `tests/test_modelfile.cpp` (`blueprint_opset_enforced_per_feature`, `shipped_templates_declare_their_opset`) and the per-family goldens build each of them. A model file written by an earlier build of this session's importers with a too-low opset would be refused at load; none of the shipped templates produced one.

Template placeholders are `{{key}}`, or `{{key|default}}` for optional pieces (`{{rope_args|}}`, `{{attn_args|}}`, `{{opset|1}}`).

### Template families (`import/templates/*.fbp`)

| Template | Session | HF `architectures` / `model_type` | GGUF `general.architecture` | What the template encodes |
|----------|---------|-----------------------------------|-----------------------------|---------------------------|
| `llama` | generative | `LlamaForCausalLM`, `MistralForCausalLM`, `Qwen2ForCausalLM`, `model_type` `llama` (Llama 2/3, TinyLlama, SmolLM, Yi, Zephyr, Mistral 7B) | `llama`, `mistral`, `qwen2` | RMSNorm, RoPE, GQA, SwiGLU. Mistral 7B v0.1's `sliding_window` is not applied |
| `qwen3`, `qwen3_moe`, `qwen3_5`, `qwen3_5_moe`, `mixtral`, `deepseek_v2` | generative | see the importer table above | see above | QK-norm; sparse MoE; Gated DeltaNet hybrid; absorbed MLA |
| `gpt2` | generative | `GPT2LMHeadModel`, `gpt2` | `gpt2` | learned positions, LayerNorm with biases, fused q/k/v projection, Conv1D weights transposed on import, tied head |
| `gemma` | generative | `GemmaForCausalLM`, `gemma` | `gemma` | RMSNorm `(1 + w)` (stored as `1 + w`), embeddings scaled by `sqrt(d)`, GeGLU (GELU-tanh), tied head |
| `gemma2` | generative | `Gemma2ForCausalLM`, `gemma2` | `gemma2` | + pre and post norms, `query_pre_attn_scalar`, attention soft-capping (`attention_bias`), sliding window on every other layer, soft-capped logits. Needs the f32 KV cache |
| `gemma3` | generative | `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration` (text tower only), `gemma3_text` | `gemma3` | + QK-norm, sliding-window pattern with a local RoPE base |
| `phi3` | generative | `Phi3ForCausalLM`, `phi3` (Phi-3 / 3.5 mini, Phi-4-mini) | `phi3` | fused `qkv_proj`, fused `gate_up_proj`, partial RoPE (`partial_rotary_factor`) |
| `bert` | embedding | `BertModel` and heads, `bert`; `RobertaModel`, `XLMRobertaModel`, `roberta`, `xlm-roberta` (MiniLM, BGE, E5, GTE checkpoints use these) | `bert` | token + position + type embeddings, post-LN blocks, key padding mask, CLS or mean pooling (+ L2) taken from `1_Pooling/config.json` / `modules.json`, or `forge_pooling` / `forge_normalize` in `config.json` |
| `vit` | embedding | `ViTModel`; `CLIPVisionModel(WithProjection)`, `CLIPModel` (vision tower); `SiglipVisionModel`, `SiglipModel` (vision tower) | none | conv2d patch embedding, class token (ViT, CLIP), pre-LN blocks; output: LayerNorm of the class token (ViT), that times the visual projection (CLIP), or the attention-pooling head (SigLIP) |
| `whisper_encoder` | embedding | `WhisperForConditionalGeneration`, `WhisperModel`, `whisper` | none | two conv1d + GELU, the checkpoint's sinusoidal positions, pre-LN blocks; output: encoder states `[n_ctx, d_model]`. The decoder is not imported |

`--template NAME` forces a template; `clip_vision`, `siglip_vision` and `roberta` select those variants of `vit` / `bert`.

Not available: Phi-2, Mamba, UNet / DiT denoisers, the Whisper decoder (no template yet); Phi-3 LongRoPE (128k variants) is refused. Tokenizers and chat templates of these families go through the same import calls as the Llama path (`store_hf_tokenizer`, `store_gguf_tokenizer`; see [Tokenizers](#tokenizers)): BERT-family WordPiece and Unigram files, Gemma / Phi-3 SentencePiece BPE, GPT-2 byte-level BPE. Files imported before that keep a non-BPE `tokenizer.json` in a section `tokenizer_json` (manifest `tokenizer_type`); the session still reads that section when the file has no `tokenizer` section, but importers no longer write it.

### Writing a template
A template is a blueprint with `{{key}}` placeholders that the importer fills from `config.json` (numbers, or a string such as an op name or a whole expression). The language (`blueprint.hpp`):

```
blueprint "decoder-lm" opset 3                                   # the newest feature used below (min / mod) is opset 3
params { d_model: {{hidden_size}}  n_layers: {{num_hidden_layers}}  rope: { base: {{rope_theta}} } }   # numbers, nested groups
state kv[n_layers] : kv_cache(heads: n_kv, dim: head_dim)        # per-session state slots
input tokens : f32[seq]                                          # `seq` is bound per run; other dims are params or numbers
op affine(x, w, b) { return linear(x, w) + b }                   # composite op, see below
graph {
  x = embed(W.tok_emb, tokens) * sqrt(d_model)                   # W.<path> names a tensor of the file
  repeat l in 0..n_layers {                                      # unrolled when the graph is built
    local = min(1, mod(l + 1, pattern))                          # numbers are folded: per-layer attrs
    h = rms_norm(x, W.layer[l].attn_norm, eps)                   # any spec op by name; numbers / lists bind to its attrs in order
    kk, vv = kv[l].append(k, v)
    a = attention(q, kk, vv, causal: true, window: window * local)   # or by keyword
  }
}
output logits : f32[1, vocab]
session generative_sequence                                      # or: embedding, one_shot
```

Built-ins: `linear(x, W)` (`x @ W^T`), `embed(W, ids)`, `heads(x, n, d)`, `merge(x, d)`, `last(x, seq)`, `scalar(v)` (a `[1]` constant tensor), and the number helpers `sqrt floor ceil min max mod` (folded at build time when every argument is a number).

**Composite ops.** `op name(a, b, eps = 1e-6) { t = ...  return expr }` at the top level defines an op as a composition of existing ones; it is called like any op, by position or by keyword (`name(x, y)`, `name(x, y, eps: 1e-5)`). Rules:
- Composites are inlined into the graph where they are called, so they cost nothing at run time and the optimiser sees through them.
- Parameters are tensors, weight paths (`W.layer[l].wq`), numbers or lists. A parameter may have a numeric default; parameters without one must come first. A wrong argument count, an unknown or repeated keyword, and a body without `return` are errors that name the op and the line.
- A body sees its parameters, the blueprint's params and the names in scope at the call; its own assignments are local.
- Recursion is forbidden: `parse_blueprint` rejects a composite that reaches itself, directly or through other composites (`composite op 'f' is recursive (f -> g -> f)`), whether or not it is called.
- A spec op or built-in of the same name wins, so a file that defines `op mish(x) { ... }` keeps working on an engine without `mish` and uses the engine's kernel on one that has it.
- Shapes are checked after inlining, by the graph verifier (`node %12 (matmul_t): ...`), not per composite.
- The blueprint text is the canonical form: it is stored verbatim in the `.fpk` and parsed again on load, so there is no separate serialised form to keep in step.

To add an architecture: write the template, map its `architectures` / `model_type` to it and its tensors to the template's `W.` names in `hf_arch.hpp` (one short function per family; see `import_gpt2`), and add a `verify/golden/tiny_<arch>.py` that builds a random tiny checkpoint and compares FORGE with a numpy forward pass.

Importer limits that raise an error:
- HF `deepseek_v2`: no YaRN rope scaling, greedy top-k routing with softmax scoring only.
- HF `qwen3_5` and GGUF `qwen35`: no MoE variants, no full-width RoPE, `hidden_act` must be `silu`, and full-attention layers must repeat every `full_attention_interval` layers.
- GGUF `qwen35`: V heads stored in tiled order (`ssm.v_head_reordered`) are put back in grouped order. Row-wise tensors are permuted as raw blocks. The DeltaNet out projection is decoded, permuted and re-encoded, as q8_0 when the source was quantised.
- ONNX ops: MatMul Gemm Add Sub Mul Div Pow Max Min Relu Sigmoid Tanh Exp Log Sqrt Neg Abs Erf Gelu Softplus Reciprocal Softmax LayerNormalization Transpose Reshape Flatten Identity Constant ReduceMean ReduceSum. One symbolic dimension maps to `seq`. Any other op is rejected by name.

Templates are read from `FORGE_TEMPLATES` if set, else from `../share/forge/templates` relative to the executable when that directory exists (an installed tree), else from `import/templates/` in the source tree.

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
| `encode_typed(text)`, `encode_pair(a, b)` | `{ids, type_ids}` with the special tokens: the post-processor's segment ("token type") id per token. A pair goes through the pair template (`[CLS] A [SEP] B [SEP]` with B in segment 1 for `BertProcessing`, `<s> A </s></s> B </s>` for `RobertaProcessing`, the `pair` list of a `TemplateProcessing`); without one, A then B (segments 0 and 1) inside the BOS / EOS one text gets |
| `encode(text, true)`, `with_special(ids)` | Also what the model expects around a prompt: BOS / EOS per `add_bos()` / `add_eos()` when the session set them from the manifest, else the `tokenizer.json` post-processor (`[CLS] … [SEP]`, `<s> …`) |
| `decode(ids, skip_special = false)` | The decoder chain over the whole sequence. Bytes of an incomplete UTF-8 character are passed through, not replaced |
| `decode_one(id)` | The bytes one token adds in the middle of a text (streaming, grammar masks). It differs from `decode` where the decoder looks at the position: a SentencePiece word keeps its leading space, WordPiece puts a space before each word-initial piece |
| `token_id(str)`, `id_to_token(id)`, `is_special(id)`, `vocab_size()`, `type()`, `bos()`, `eos()`, `unk()`, `add_bos()`, `add_eos()`, `has_post_processor()` | |
| `normalize(text)`, `pre_tokenize(text)` | The first two stages on their own, for debugging a mismatch |

Every importer stores the tokenizer through one call per source format: `store_hf_tokenizer` (`hf.hpp`) for a Hugging Face directory and `store_gguf_tokenizer` (`tokenizer_import.hpp`) for a GGUF, used by the Llama-style importers and by the families of `hf_arch.hpp` / `gguf_arch.hpp` alike. Both build the runtime tokenizer once before writing the section: a `tokenizer.json` this build cannot run (an unknown component) is left out with an import note, and the model then takes token ids, instead of producing a file that fails to open.

Sources other than a ready `tokenizer.json` are converted to one at import (`import/tokenizer_import.hpp`):

- **SentencePiece `.model`** (`tokenizer.model`, `spiece.model`, `sentencepiece.bpe.model` in a Hugging Face directory without `tokenizer.json`): read with a small protobuf reader. BPE models become `SentencePieceBPE`, unigram models `Unigram`; the model's normalization (precompiled charsmap, whitespace collapsing, dummy prefix) becomes the normalizer chain, and control / user-defined pieces plus `tokenizer_config.json`'s `added_tokens_decoder` become added tokens.
- **WordPiece `vocab.txt`** (a BERT checkpoint without `tokenizer.json`): one piece per line; `tokenizer_config.json`'s `do_lower_case`, `strip_accents`, `tokenize_chinese_chars` and `unk_token` / `cls_token` / `sep_token` set the `BertNormalizer` and the `[CLS] … [SEP]` post-processor (`BertTokenizer`'s defaults where the file says nothing).
- **GGUF** `tokenizer.ggml.model`: `gpt2` → byte-level BPE split by the regex of `tokenizer.ggml.pre` (`llama-bpe`, `qwen2`, `deepseek-llm`, `deepseek-coder`, `deepseek-v3`, `gpt-2`, `tekken`, `gpt-4o`, `falcon`, `starcoder`, … as in llama.cpp; an unknown name falls back to GPT-2's with a warning); `llama` → `SentencePieceBPE` over `tokenizer.ggml.scores`; `t5` → `Unigram` (with `precompiled_charsmap`, `remove_extra_whitespaces`); `bert` → `WordPiece` with a `[CLS] … [SEP]` post-processor (ids from `bos_token_id` / `cls_token_id` and `seperator_token_id` / `eos_token_id`, else the pieces named `[CLS]` / `[SEP]`; for `bert` files `add_bos_token` / `add_eos_token` are not copied to the manifest, the post-processor adds both). `tokenizer.ggml.token_type` 3 (control) and 2 (unknown) become special added tokens, 4 (user-defined) plain added tokens, 6 (byte) switches byte fallback on. `add_bos_token` / `add_eos_token` go to the manifest, `add_space_prefix` into the normalizer. A `tokenizer.huggingface.json` key is used as it is. Other models (`rwkv`, …) give a file without a tokenizer, with a warning.

How this is tested, and what is not: `tests/test_tokenizer.cpp` compares ids, normalizer and pre-tokenizer output and decoded text with goldens made by the `tokenizers` 0.23 and `sentencepiece` 0.2 packages (`tests/data/tokenizer/make_fixtures.py`) for small tokenizers trained on a multilingual corpus (ASCII, digit runs, CJK, emoji, combining marks, right-to-left text, mixed whitespace, control characters, invalid UTF-8). End to end (text → ids → model output through `forge run --prompt` / `forge embed --text` on imported tiny models, with the reference ids from the same goldens): `verify/golden/tiny_gpt2.py` (byte-level BPE, HF and GGUF), `tiny_gemma.py` (SentencePiece `tokenizer.model`; a transformers-style SentencePiece-BPE `tokenizer.json`), `tiny_phi3.py` (the same two plus GGUF `llama`, and the chat template from `tokenizer_config.json` / `tokenizer.chat_template` through `--chat`), `tiny_bert.py` (WordPiece from `tokenizer.json`, `vocab.txt` and GGUF `bert`; Unigram; a sentence pair with token types). It has not been run against the tokenizers of full-size published models. The llama.cpp pre-tokenizer regexes and its BERT conventions were written from memory of llama.cpp and are only checked for `llama-bpe`, `qwen2` and `gpt-2` (against `tokenizers`) and on ASCII text for `bert`. llama.cpp's per-model token attributes (`lstrip` / `rstrip` on some special tokens) are not carried over. Unicode tables are FORGE's (Unicode 16); a reference built against another Unicode version can differ on characters assigned in between.

The **chat template** (`tokenizer_config.json` `chat_template`, `chat_template.jinja`, or GGUF `tokenizer.chat_template`) is stored as the `chat_template` section and rendered by `inference::apply_chat_template` ([inference.md](inference.md#chat-templates-chat_templatehpp)).

### Mixed precision (HF only, `quant_recipe.hpp`)
`--codec q4_k_m` or `q5_k_m` chooses a type per tensor. `--recipe llama` is a port of llama.cpp's `llama_tensor_get_type()` rules. `--recipe forge` (default) applies the same rules, then puts Gated DeltaNet in/out projections and `attn_output` at q6_k and the head and token embedding at q8_0 (bf16 with `--bf16-head`). Norms, conv weights, A/dt and the a/b gates stay f32. Overrides: `--output-tensor-type`, `--token-embedding-type`, `--tensor-type REGEX=TYPE[,...]` (first match wins), `--pure`.

### CLI
```sh
forge import <hf_dir | model.gguf | model.onnx> -o out.fpk [--codec C] [--template NAME] [--emb-codec C] [--allow-unused] [--ignore-tensor NAME[,NAME...]]
```
Codecs: `f32 f16 bf16 fp8_e4m3 fp8_e5m2 q8_0 q4_k q5_k q6_k q4_g32 q8_g32 mxfp8 mxfp6 mxfp4 mxint8 nvfp4 tq2 q4_0 q4_1 q5_0 q5_1 q2_k q3_k iq4_nl iq4_xs tq1_0 tq2_0`, or `keep`. With a quantised codec, the token embedding defaults to q8_0 (`--emb-codec` overrides). `--codec` defaults to `f32` for plain Hugging Face checkpoints and to `keep` for GGUF files and AWQ / GPTQ checkpoints. `forge run` also accepts an HF directory or `.gguf` directly and caches the import in `~/.cache/forge` (`--allow-unused` applies to that import too).

The importers are tested on small random models built by the test scripts (`verify/golden/tiny_llama.py`, `tiny_moe.py`, `tiny_qwen2.py` (HF and GGUF), `tiny_rope.py` (RoPE scaling variants, HF and GGUF), `tiny_qwen3.py` (HF and GGUF), `tiny_qwen3_5.py`, `quant_inplace.py`, `gguf_types.py`, `hf_quant.py`, `tests/onnx_test.py`, and for the template families `tiny_gpt2.py`, `tiny_gemma.py`, `tiny_phi3.py`, `tiny_bert.py` (HF and GGUF each), `tiny_vit.py`, `tiny_whisper.py`), checked against float64 NumPy or transformers references. `gguf_types.py` imports a model in every loadable GGUF tensor type and compares the tensors with gguf-py's dequantisation and the logits with a forward pass over the dequantised weights. They have not been checked against full-size published checkpoints in the test suite. The RoPE scaling references are NumPy transcriptions of `transformers/modeling_rope_utils.py`; they have not been run against transformers or llama.cpp themselves.

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
- **Encrypted models and memory.** `forge encrypt` seals the data in chunks ([format.md](format.md#encryption)). A session then decrypts each weight when a plan first binds it and keeps that tensor's plaintext for as long as the model is open, because the kernels read weights in place. What this bounds: nothing is decrypted at open, tensors no plan uses are never decrypted, `reader::load()` (which decodes to f32) works from a scratch copy that is wiped, and the plaintext is not one heap block the size of the file. What it does not: a model that uses all of its weights ends up with all of them decrypted, and those pages are not file-backed, so the OS cannot drop and re-read them as it does for a plain mapped file. Tensors decoded to f32 (a weight in a codec the consuming op cannot read in place, LoRA-merged weights) live in ordinary memory in either case. `section_bytes("data")` on a chunk-encrypted file decrypts the whole section; nothing in FORGE calls it for such a file.
- A file sealed with the older single-AEAD scheme is still decrypted whole, into ordinary heap memory, when it is opened. Re-seal it (`forge encrypt` on the plain file) to get the chunked behaviour.
- Signing is cross-checked against pyca/cryptography (`modelfile_signing` ctest).

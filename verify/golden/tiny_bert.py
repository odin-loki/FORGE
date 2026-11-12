#!/usr/bin/env python3
"""Golden end-to-end test for the bert template and the embedding session: random tiny encoders, a float64 numpy
reference, and `forge embed`.

  bert      BertModel layout, [CLS] pooling, token-type ids, exact GELU; a WordPiece tokenizer.json (kept out of the
            runtime's BPE tokenizer section: ids are passed directly)
  roberta   "roberta." prefix, positions offset by pad_token_id + 1, sentence-transformers mean pooling + Normalize

  1. HF import (f32, q8_0) -> `forge embed --tokens` against the reference, per variant
  2. padding: the sequence followed by pad tokens with --mask must give the embedding of the unpadded sequence
     (masked attention keys + masked mean pooling)
  3. GGUF import (general.architecture bert, pooling_type mean) -> embedding
"""
import json, os
import numpy as np
import golden_util as gu

BASE = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128, vocab_size=300, max_position_embeddings=40,
            layer_norm_eps=1e-12, hidden_act="gelu", position_embedding_type="absolute")
CFGS = {"bert": dict(BASE, architectures=["BertModel"], model_type="bert", type_vocab_size=2, pad_token_id=0),
        "roberta": dict(BASE, architectures=["XLMRobertaModel"], model_type="xlm-roberta", type_vocab_size=1, pad_token_id=1)}

def make_weights(c, seed):
    rng = np.random.default_rng(seed); H, F = c["hidden_size"], c["intermediate_size"]
    ln = lambda: (1 + 0.1 * rng.normal(size=H), 0.1 * rng.normal(size=H))
    lin = lambda o, i: (rng.normal(0, 1 / np.sqrt(i), (o, i)), 0.1 * rng.normal(size=o))
    W = {"embeddings.word_embeddings.weight": rng.normal(0, 1.0, (c["vocab_size"], H)),
         "embeddings.position_embeddings.weight": rng.normal(0, 0.5, (c["max_position_embeddings"], H)),
         "embeddings.token_type_embeddings.weight": rng.normal(0, 0.5, (c["type_vocab_size"], H))}
    W["embeddings.LayerNorm.weight"], W["embeddings.LayerNorm.bias"] = ln()
    for l in range(c["num_hidden_layers"]):
        p = f"encoder.layer.{l}."
        for n, (o, i) in {"attention.self.query": (H, H), "attention.self.key": (H, H), "attention.self.value": (H, H),
                          "attention.output.dense": (H, H), "intermediate.dense": (F, H), "output.dense": (H, F)}.items():
            W[p + n + ".weight"], W[p + n + ".bias"] = lin(o, i)
        W[p + "attention.output.LayerNorm.weight"], W[p + "attention.output.LayerNorm.bias"] = ln()
        W[p + "output.LayerNorm.weight"], W[p + "output.LayerNorm.bias"] = ln()
    return W

def forward(c, W, ids, types, off, pool, l2):
    H, nh, eps = c["hidden_size"], c["num_attention_heads"], c["layer_norm_eps"]; hd = H // nh; s = len(ids)
    aff = lambda x, n: x @ W[n + ".weight"].T + W[n + ".bias"]
    x = W["embeddings.word_embeddings.weight"][ids] + W["embeddings.position_embeddings.weight"][np.arange(s) + off] + W["embeddings.token_type_embeddings.weight"][types]
    x = gu.layer_norm(x, W["embeddings.LayerNorm.weight"], W["embeddings.LayerNorm.bias"], eps)
    for l in range(c["num_hidden_layers"]):
        p = f"encoder.layer.{l}."
        q, k, v = (aff(x, p + "attention.self." + n).reshape(s, nh, hd) for n in ("query", "key", "value"))
        a = aff(gu.attend(q, k, v).reshape(s, H), p + "attention.output.dense")
        x = gu.layer_norm(x + a, W[p + "attention.output.LayerNorm.weight"], W[p + "attention.output.LayerNorm.bias"], eps)
        f = aff(gu.gelu(aff(x, p + "intermediate.dense")), p + "output.dense")
        x = gu.layer_norm(x + f, W[p + "output.LayerNorm.weight"], W[p + "output.LayerNorm.bias"], eps)
    e = x.mean(0) if pool == "mean" else x[0]
    return e / np.linalg.norm(e) if l2 else e

def gguf_tensors(c, W, off):
    T = {"token_embd.weight": W["embeddings.word_embeddings.weight"], "position_embd.weight": W["embeddings.position_embeddings.weight"][off:],   # offset rows dropped
         "token_types.weight": W["embeddings.token_type_embeddings.weight"],
         "token_embd_norm.weight": W["embeddings.LayerNorm.weight"], "token_embd_norm.bias": W["embeddings.LayerNorm.bias"]}
    names = {"attention.self.query": "attn_q", "attention.self.key": "attn_k", "attention.self.value": "attn_v", "attention.output.dense": "attn_output",
             "attention.output.LayerNorm": "attn_output_norm", "intermediate.dense": "ffn_up", "output.dense": "ffn_down", "output.LayerNorm": "layer_output_norm"}
    for k, v in W.items():
        if not k.startswith("encoder.layer."): continue
        _, _, l, rest = k.split(".", 3); mod, kind = rest.rsplit(".", 1)
        T[f"blk.{l}.{names[mod]}.{kind}"] = v
    return T

def main():
    a = gu.args(); ck = gu.Checker(a.forge, a.work)
    ids = [2, 72, 101, 108, 108, 111, 44, 32, 116, 104, 101, 3]; types = [0] * 7 + [1] * 5
    csv = lambda v: ",".join(map(str, v))
    # --- bert: CLS pooling, token types
    c = CFGS["bert"]; hf = os.path.join(a.work, "hf_bert")
    W = gu.save_model(hf, make_weights(c, 301), c, tokenizer="WordPiece")
    ref = forward(c, W, ids, types, 0, "cls", False)
    for codec, cosmin in (("f32", None), ("q8_0", 0.999)):
        fp = ck.imp(hf, f"bert_{codec}.fpk", codec)
        ck.embed(fp, ["--tokens", csv(ids), "--token-types", csv(types)], ref, f"bert cls {codec}", cosmin=cosmin)
    # --- roberta: prefixed names, position offset, mean pooling + L2 normalisation from the sentence-transformers files
    c = CFGS["roberta"]; hf = os.path.join(a.work, "hf_roberta"); off = c["pad_token_id"] + 1
    W = gu.save_model(hf, {"roberta." + k: v for k, v in make_weights(c, 302).items()}, c, tokenizer="WordPiece")
    W = {k[len("roberta."):]: v for k, v in W.items()}
    os.makedirs(os.path.join(hf, "1_Pooling"), exist_ok=True)
    json.dump({"word_embedding_dimension": 64, "pooling_mode_cls_token": False, "pooling_mode_mean_tokens": True}, open(os.path.join(hf, "1_Pooling", "config.json"), "w"))
    json.dump([{"idx": 0, "type": "sentence_transformers.models.Transformer"}, {"idx": 1, "type": "sentence_transformers.models.Pooling"},
               {"idx": 2, "type": "sentence_transformers.models.Normalize"}], open(os.path.join(hf, "modules.json"), "w"))
    ref = forward(c, W, ids, [0] * len(ids), off, "mean", True)
    fp = ck.imp(hf, "roberta_f32.fpk")
    ck.embed(fp, ["--tokens", csv(ids)], ref, "roberta mean+l2 f32")
    ck.ok(abs(np.linalg.norm(ref) - 1) < 1e-9, "roberta: the reference embedding is unit length")
    pad = [c["pad_token_id"]] * 3
    ck.embed(fp, ["--tokens", csv(ids + pad), "--mask", csv([1] * len(ids) + [0] * 3)], ref, "roberta padded + mask == unpadded")
    unmasked = os.path.join(a.work, "unmasked.f32")
    gu.run([a.forge, "embed", fp, "--tokens", csv(ids + pad), "--out", unmasked])
    ck.ok(np.abs(np.fromfile(unmasked, dtype=np.float32) - ref).max() > 1e-2, "roberta: without the mask the padding changes the embedding (the mask is exercised)")
    ck.embed(ck.imp(hf, "roberta_q8_0.fpk", "q8_0"), ["--tokens", csv(ids)], ref, "roberta mean+l2 q8_0", cosmin=0.999)
    # --- GGUF (llama.cpp's bert architecture: pooling_type 1 = mean; normalisation is left to the caller there)
    gp = gu.write_gguf(os.path.join(a.work, "roberta.gguf"), "bert",
                       {"context_length": c["max_position_embeddings"] - off, "embedding_length": c["hidden_size"], "block_count": c["num_hidden_layers"],
                        "feed_forward_length": c["intermediate_size"], "attention.head_count": c["num_attention_heads"],
                        "attention.layer_norm_epsilon": 1e-12, "pooling_type": 1}, gguf_tensors(c, W, off))
    if gp: ck.embed(ck.imp(gp, "roberta_gguf.fpk", None), ["--tokens", csv(ids)], forward(c, W, ids, [0] * len(ids), off, "mean", False), "bert gguf mean F32")
    else: print("[skip] gguf python package not installed")
    ck.done("bert")

if __name__ == "__main__": main()

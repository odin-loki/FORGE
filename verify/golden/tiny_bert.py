#!/usr/bin/env python3
"""Golden end-to-end test for the bert template and the embedding session: random tiny encoders, a float64 numpy
reference, and `forge embed`.

  bert      BertModel layout, [CLS] pooling, token-type ids, exact GELU; a trained WordPiece tokenizer.json
            (tests/data/tokenizer/hf_wordpiece_bert.json: BertNormalizer, BertPreTokenizer, BertProcessing)
  roberta   "roberta." prefix, positions offset by pad_token_id + 1, sentence-transformers mean pooling + Normalize;
            a trained Unigram tokenizer.json with a "$A </s>" template (hf_unigram_t5.json)

  1. HF import (f32, q8_0) -> `forge embed --tokens` against the reference, per variant
  2. padding: the sequence followed by pad tokens with --mask must give the embedding of the unpadded sequence
     (masked attention keys + masked mean pooling)
  3. GGUF import (general.architecture bert, pooling_type mean) -> embedding
  4. text -> ids -> embedding: `forge embed --text` must give the embedding of the ids the `tokenizers` package
     returns for that text with special tokens ([CLS] ... [SEP], from goldens.json) -- from tokenizer.json, from a
     bare vocab.txt, and from the GGUF's tokenizer.ggml.* (model bert); `--text-pair` that of a sentence pair with
     its token types (needs the `tokenizers` package for the reference ids, skipped without it)
"""
import json, os
import numpy as np
import golden_util as gu

BASE = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128, vocab_size=512, max_position_embeddings=64,
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
    W = gu.save_model(hf, make_weights(c, 301), c, tokenizer=None)
    gu.use_tokenizer(hf, "hf_wordpiece_bert.json")
    ref = forward(c, W, ids, types, 0, "cls", False)
    text, tids = gu.tok_golden("hf_wordpiece_bert")
    tref = forward(c, W, tids, [0] * len(tids), 0, "cls", False)
    for codec, cosmin in (("f32", None), ("q8_0", 0.999)):
        fp = ck.imp(hf, f"bert_{codec}.fpk", codec)
        ck.embed(fp, ["--tokens", csv(ids), "--token-types", csv(types)], ref, f"bert cls {codec}", cosmin=cosmin)
        ck.embed(fp, ["--text", text], tref, f"bert --text {codec} (WordPiece, [CLS] + {len(tids) - 2} ids + [SEP])", cosmin=cosmin)
    ck.ok(tids[0] == 2 and tids[-1] == 3, "bert: the reference ids are wrapped in [CLS] / [SEP]")
    fp = os.path.join(a.work, "bert_f32.fpk")
    try:
        from tokenizers import Tokenizer
        e = Tokenizer.from_file(os.path.join(gu.TOK_DIR, "hf_wordpiece_bert.json")).encode("The quick brown fox", "jumps over the lazy dog.")
        ck.ok(1 in e.type_ids and e.type_ids[0] == 0, "bert pair: the reference has both segments")
        ck.embed(fp, ["--text", "The quick brown fox", "--text-pair", "jumps over the lazy dog."], forward(c, W, e.ids, e.type_ids, 0, "cls", False),
                 "bert --text + --text-pair ([CLS] A [SEP] B [SEP], token types 0 / 1)")
    except ImportError: print("[skip] sentence pair: the tokenizers package is not installed")
    # the same vocabulary as a bare vocab.txt (older checkpoints): BertTokenizer's defaults are this fixture's pipeline
    vocab = json.load(open(os.path.join(gu.TOK_DIR, "hf_wordpiece_bert.json"), encoding="utf-8"))["model"]["vocab"]
    os.remove(os.path.join(hf, "tokenizer.json"))
    open(os.path.join(hf, "vocab.txt"), "w", encoding="utf-8").write("".join(t + "\n" for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])))
    json.dump({"do_lower_case": True, "cls_token": "[CLS]", "sep_token": {"content": "[SEP]"}}, open(os.path.join(hf, "tokenizer_config.json"), "w"))
    ck.embed(ck.imp(hf, "bert_vocabtxt.fpk"), ["--text", text], tref, "bert --text from vocab.txt")
    # --- roberta: prefixed names, position offset, mean pooling + L2 normalisation from the sentence-transformers files
    c = CFGS["roberta"]; hf = os.path.join(a.work, "hf_roberta"); off = c["pad_token_id"] + 1
    W = gu.save_model(hf, {"roberta." + k: v for k, v in make_weights(c, 302).items()}, c, tokenizer=None)
    gu.use_tokenizer(hf, "hf_unigram_t5.json")
    W = {k[len("roberta."):]: v for k, v in W.items()}
    os.makedirs(os.path.join(hf, "1_Pooling"), exist_ok=True)
    json.dump({"word_embedding_dimension": 64, "pooling_mode_cls_token": False, "pooling_mode_mean_tokens": True}, open(os.path.join(hf, "1_Pooling", "config.json"), "w"))
    json.dump([{"idx": 0, "type": "sentence_transformers.models.Transformer"}, {"idx": 1, "type": "sentence_transformers.models.Pooling"},
               {"idx": 2, "type": "sentence_transformers.models.Normalize"}], open(os.path.join(hf, "modules.json"), "w"))
    ref = forward(c, W, ids, [0] * len(ids), off, "mean", True)
    fp = ck.imp(hf, "roberta_f32.fpk")
    ck.embed(fp, ["--tokens", csv(ids)], ref, "roberta mean+l2 f32")
    utext, uids = gu.tok_golden("hf_unigram_t5")
    ck.embed(fp, ["--text", utext], forward(c, W, uids, [0] * len(uids), off, "mean", True), f"roberta --text (Unigram, {len(uids) - 1} ids + </s>)")
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
                        "attention.layer_norm_epsilon": 1e-12, "pooling_type": 1}, gguf_tensors(c, W, off), tokenizer="gguf_bert.gguf")
    if gp:
        fg = ck.imp(gp, "roberta_gguf.fpk", None)
        ck.embed(fg, ["--tokens", csv(ids)], forward(c, W, ids, [0] * len(ids), off, "mean", False), "bert gguf mean F32")
        ck.embed(fg, ["--text", text], forward(c, W, tids, [0] * len(tids), off, "mean", False), "bert gguf --text (tokenizer.ggml.model bert -> WordPiece)")
    else: print("[skip] gguf python package not installed")
    ck.done("bert")

if __name__ == "__main__": main()

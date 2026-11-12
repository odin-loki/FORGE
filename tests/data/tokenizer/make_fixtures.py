#!/usr/bin/env python3
"""Regenerates the tokenizer fixtures and goldens in this directory.

    pip install tokenizers sentencepiece gguf protobuf
    python3 tests/data/tokenizer/make_fixtures.py [--check path/to/tokdump] [--legacy path/to/olddump]

Every tokenizer here is tiny (trained on the corpus below, a few hundred pieces) so the files can be checked in.
The expected values come from the reference implementations, not from FORGE:

  hf_*.json    tokenizer.json files built and trained with the Hugging Face `tokenizers` package; goldens are what
               `tokenizers` itself returns (normalize_str, pre_tokenize_str, encode, decode).
  spm_*.model  SentencePiece models trained with the `sentencepiece` package; goldens are SentencePieceProcessor's
               ids. FORGE reads the .model protobuf itself (import/tokenizer_import.hpp).
  gguf_*.gguf  GGUF files holding only tokenizer metadata, written with the `gguf` package from the vocabularies
               above; FORGE's GGUF conversion has to reproduce the goldens of the tokenizer it was made from.
               gguf_tiny_llama.gguf also has the tensors of a one-layer random Llama, for an end-to-end import.
  legacy_v1.json  a tokenizer section as FORGE wrote it before it had the pipeline (model + added_tokens only).
               Its goldens are the output of that older FORGE tokenizer (--legacy: a program built from the old
               header that prints {"ids": [...], "dec": hex} per hex input line); without --legacy they are kept.

goldens.json: {"corpus": [hex...], "fixtures": {name: {"oracle": ..., "cases": [...]}}}. The corpus is hex-encoded
because it contains invalid UTF-8, which the references read with U+FFFD in place of the bad bytes, as FORGE does.
A fixture with "patch" is its "file" with those top-level keys (and keys of "model") replaced.
"""
import json, os, subprocess, sys

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

HERE = os.path.dirname(os.path.abspath(__file__))

CORPUS = [
    b"",
    b" ",
    b"Hello, world!",
    b"  leading and   multiple   spaces  ",
    b"tabs\tand\nnewlines\r\n\n\nend\n",
    b"it's John's, they're we've I'm you'll he'd IT'S DON'T",
    b"1234567890 3.14159 2026-11-12 1,000,000 x86_64 v2.0",
    "Ünïcödé çhàrâctérs: naïve café Ærøskøbing Straße".encode(),
    "decomposed é ä ñ ộ vs é ä ñ".encode(),
    "你好，世界！日本語のテキスト。한국어 텍스트".encode(),
    "emoji 😀 👍🏽 👨‍👩‍👧‍👦 🇯🇵 ❤️ done".encode(),
    "שלום עולם مرحبا بالعالم".encode(),
    "nbsp em ideographic　zwsp​end  line para".encode(),
    b"ctl a\x00b\x01c\x7fd\x1be",
    b"bad utf8 abc\xff\xfe def \xe2\x82 x \xf0\x9f y \xc0\xaf z",
    b"<s>Hello</s> <|im_start|>user\nhi there<|im_end|>\n<|im_start|>assistant\n",
    b"<|endoftext|>text<|endoftext|><|endoftext|> more [CLS] and [SEP] [MASK].",
    b"Pneumonoultramicroscopicsilicovolcanoconiosis antidisestablishmentarianism",
    b"def f(x):\n    return x**2  # comment\n\tif x != 0: print(\"a\\n\")",
    b"https://example.com/path?q=1&r=2#frag foo_bar-baz.qux@example.com",
    "ǅ ß İ ſ ﬁ ① Ⅳ ½ ㍿ ＡＢＣ １２３ ｶﾞ".encode(),
    "नमस्ते दुनिया สวัสดีชาวโลก Ελληνικά Русский".encode(),
    "… — – ‘quotes’ “double” «guillemets» ¿qué? ¡sí! 100% $5.00 €9,99".encode(),
    b"ALLCAPS MiXeD CamelCaseWords snake_case_words kebab-case-words",
    b"The quick brown fox jumps over the lazy dog. The five boxing wizards jump quickly.",
    b"a  b   c    d\n\n\n\ne \t f",
    "１２３４５ ٣٤٥ ²³ ⅔ 12345678".encode(),
    b"word",
    b" word",
    b"word ",
    b"\n",
    "▁already▁meta space ▁ here".encode(),
    b"Hello<mask>world <mask> x<mask>. HELLO World hello",
    b"##hash ##tags [UNK] unaffable unwanted running",
    b"a" + b"xyz" * 40 + b" over one hundred characters in a word",
]

TRAIN = [c.decode("utf-8", "replace") for c in CORPUS] * 3 + [
    "The quick brown fox jumps over the lazy dog.", "the the the and and of of to to in in is is that that it it",
    "running runner runs ran unwanted wanting wanted wants unaffable affable", "Hello hello HELLO world World worlds wording words",
    "1 2 3 4 5 6 7 8 9 0 10 11 12 100 1000 2026", "user assistant system user assistant system hi there",
    "low lower lowest new newer newest wide wider widest", "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z", ". , ! ? ; : ' \" ( ) [ ] { } < > / \\ | @ # $ % ^ & * - _ + = ~ `",
] * 4

LLAMA3_RE = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
QWEN2_RE = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

fixtures = {}   # name -> {"oracle": str, "cases": [...]}
hx = lambda s: s   # reference outputs are valid UTF-8 (only the corpus is hex-encoded)


def hf_cases(tok):
    cases = []
    for raw in CORPUS:
        text = raw.decode("utf-8", "replace")
        norm = tok.normalizer.normalize_str(text) if tok.normalizer is not None else text
        pre = [p for p, _ in tok.pre_tokenizer.pre_tokenize_str(norm)] if tok.pre_tokenizer is not None else ([norm] if norm else [])
        ids = tok.encode(text, add_special_tokens=False).ids
        ids_special = tok.encode(text, add_special_tokens=True).ids
        cases.append({"norm": hx(norm), "pre": [hx(p) for p in pre], "ids": ids, "ids_special": ids_special,
                      "dec": hx(tok.decode(ids, skip_special_tokens=False)), "dec_skip": hx(tok.decode(ids_special, skip_special_tokens=True))})
    return cases


def save_hf(name, tok, base=None):
    """base: name of a fixture this one shares its vocabulary with; only the differing top-level keys (and the
    differing keys of "model") are stored, as "patch", and the test puts the two together."""
    j = json.loads(tok.to_str())
    if base is None:
        json.dump(j, open(os.path.join(HERE, name + ".json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
        fixtures[name] = {"oracle": "tokenizers", "file": name + ".json", "cases": hf_cases(tok)}
    else:
        b = json.load(open(os.path.join(HERE, base + ".json"), encoding="utf-8"))
        patch = {k: v for k, v in j.items() if k != "model" and b.get(k) != v}
        mp = {k: v for k, v in j["model"].items() if b["model"].get(k) != v}
        if mp: patch["model"] = mp
        assert not any(k in mp for k in ("vocab", "merges")), name
        cases = hf_cases(tok)
        if name.startswith("hf_pipe_"): cases = [{k: c[k] for k in (("norm",) if name.startswith("hf_pipe_norm") else ("pre",)) + ("ids",)} for c in cases]
        fixtures[name] = {"oracle": "tokenizers", "file": base + ".json", "patch": patch, "cases": cases}
    return tok


def compose(fx):
    j = json.load(open(os.path.join(HERE, fx["file"]), encoding="utf-8"))
    for k, v in fx.get("patch", {}).items():
        if k == "model": j["model"].update(v)
        else: j[k] = v
    return json.dumps(j, ensure_ascii=False)


def byte_tokens(): return [f"<0x{b:02X}>" for b in range(256)]


def build_hf():
    # 1. GPT-2: byte-level BPE, the ByteLevel pre-tokenizer's own regex
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    t.decoder = decoders.ByteLevel()
    t.post_processor = processors.ByteLevel(trim_offsets=False)
    t.train_from_iterator(TRAIN, trainers.BpeTrainer(vocab_size=520, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    save_hf("hf_bpe_gpt2", t)
    gpt2_model = json.loads(t.to_str())["model"]

    # 2. GPT-2 with add_prefix_space (RoBERTa style) and RobertaProcessing
    t = Tokenizer.from_str(t.to_str())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=True)
    t.add_special_tokens(["<s>", "</s>", AddedToken("<mask>", lstrip=True, special=True)])
    t.post_processor = processors.RobertaProcessing(("</s>", t.token_to_id("</s>")), ("<s>", t.token_to_id("<s>")))
    save_hf("hf_bpe_roberta", t)

    # 3. Llama-3: Split(regex, isolated) + ByteLevel(use_regex=False), ignore_merges, TemplateProcessing BOS
    t = Tokenizer(models.BPE(ignore_merges=True))
    t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(LLAMA3_RE), behavior="isolated", invert=False),
                                               pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    t.decoder = decoders.ByteLevel()
    sp = ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
    t.train_from_iterator(TRAIN, trainers.BpeTrainer(vocab_size=520, special_tokens=sp, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    t.post_processor = processors.Sequence([processors.ByteLevel(trim_offsets=False),
        processors.TemplateProcessing(single="<|begin_of_text|> $A", pair="<|begin_of_text|> $A <|begin_of_text|> $B", special_tokens=[("<|begin_of_text|>", t.token_to_id("<|begin_of_text|>"))])])
    save_hf("hf_bpe_llama3", t)

    # 4. Qwen2: NFC + Split(single digits) + ByteLevel, ChatML added tokens (special and not)
    t = Tokenizer(models.BPE())
    t.normalizer = normalizers.NFC()
    t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(QWEN2_RE), behavior="isolated", invert=False),
                                               pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(TRAIN, trainers.BpeTrainer(vocab_size=520, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    t.add_tokens([AddedToken("<tool_call>", special=False, normalized=False), AddedToken("</tool_call>", special=False, normalized=False)])
    save_hf("hf_bpe_qwen2", t)

    # 5. Llama-2 as transformers converts it: Prepend + Replace normalizer, no pre-tokenizer, byte_fallback, fuse_unk,
    #    decoder Replace + ByteFallback + Fuse + Strip, TemplateProcessing BOS
    base = Tokenizer(models.BPE(unk_token="<unk>"))
    base.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    base.train_from_iterator([s for s in TRAIN if s.isascii()], trainers.BpeTrainer(vocab_size=420, special_tokens=["<unk>", "<s>", "</s>"] + byte_tokens(), show_progress=False))
    j = json.loads(base.to_str())
    j["model"]["byte_fallback"] = True; j["model"]["fuse_unk"] = True
    t = Tokenizer.from_str(json.dumps(j))
    t.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(content=" ", left=1, right=0)])
    t.post_processor = processors.TemplateProcessing(single="<s> $A", pair="<s> $A <s> $B", special_tokens=[("<s>", t.token_to_id("<s>"))])
    save_hf("hf_bpe_llama2", t)

    # 6. the same vocabulary without byte fallback: unknown characters fuse into <unk>; Metaspace pre-tokenizer / decoder
    j = json.loads(t.to_str()); j["model"]["byte_fallback"] = False; j["post_processor"] = None
    for scheme in ("always", "first", "never"):
        for fuse in (True, False):
            j["model"]["fuse_unk"] = fuse
            t = Tokenizer.from_str(json.dumps(j))
            t.normalizer = None
            t.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=scheme, split=(scheme != "never"))
            t.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=scheme, split=(scheme != "never"))
            save_hf(f"hf_bpe_metaspace_{scheme}_{'fuse' if fuse else 'nofuse'}", t, base="hf_bpe_llama2")

    # 7. GPT-1 / CLIP style: end-of-word suffix, lower-casing normalizer chain, Whitespace pre-tokenizer, BPEDecoder
    t = Tokenizer(models.BPE(unk_token="<unk>", end_of_word_suffix="</w>"))
    t.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.decoder = decoders.BPEDecoder(suffix="</w>")
    t.train_from_iterator(TRAIN, trainers.BpeTrainer(vocab_size=500, special_tokens=["<unk>"], end_of_word_suffix="</w>", show_progress=False))
    save_hf("hf_bpe_eow", t)

    # 8. continuing_subword_prefix BPE
    t = Tokenizer(models.BPE(unk_token="<unk>", continuing_subword_prefix="@@"))
    t.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Strip()])
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    t.train_from_iterator(TRAIN, trainers.BpeTrainer(vocab_size=500, special_tokens=["<unk>"], continuing_subword_prefix="@@", show_progress=False))
    save_hf("hf_bpe_prefix", t)

    # 9. BERT uncased: BertNormalizer + BertPreTokenizer + WordPiece + [CLS] $A [SEP] + WordPiece decoder
    t = Tokenizer(models.WordPiece(unk_token="[UNK]", max_input_chars_per_word=100))
    t.normalizer = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True)
    t.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    t.decoder = decoders.WordPiece(prefix="##", cleanup=True)
    t.train_from_iterator(TRAIN, trainers.WordPieceTrainer(vocab_size=500, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], show_progress=False))
    t.post_processor = processors.BertProcessing(("[SEP]", t.token_to_id("[SEP]")), ("[CLS]", t.token_to_id("[CLS]")))
    save_hf("hf_wordpiece_bert", t)
    bert_vocab = json.loads(t.to_str())["model"]["vocab"]

    # 10. cased WordPiece, no accent stripping, TemplateProcessing, no cleanup, other pre-tokenizers
    t = Tokenizer(models.WordPiece(unk_token="[UNK]", max_input_chars_per_word=30))
    t.normalizer = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=False)
    t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation(), pre_tokenizers.Digits(individual_digits=True)])
    t.decoder = decoders.WordPiece(prefix="##", cleanup=False)
    t.train_from_iterator(TRAIN, trainers.WordPieceTrainer(vocab_size=500, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], show_progress=False))
    t.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                                                     special_tokens=[("[CLS]", t.token_to_id("[CLS]")), ("[SEP]", t.token_to_id("[SEP]"))])
    save_hf("hf_wordpiece_cased", t)

    # 11. Unigram (T5 style): Nmt + NFKC + space collapsing, Metaspace, "$A </s>"
    t = Tokenizer(models.Unigram())
    t.normalizer = normalizers.Sequence([normalizers.Nmt(), normalizers.NFKC(), normalizers.Replace(Regex(" {2,}"), " ")])
    t.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    t.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    t.train_from_iterator(TRAIN, trainers.UnigramTrainer(vocab_size=400, special_tokens=["<pad>", "</s>", "<unk>"], unk_token="<unk>", show_progress=False))
    t.post_processor = processors.TemplateProcessing(single="$A </s>", pair="$A </s> $B </s>", special_tokens=[("</s>", t.token_to_id("</s>"))])
    save_hf("hf_unigram_t5", t)

    # 12. Unigram with byte_fallback and no pre-tokenizer (the whole text is one lattice)
    j = json.loads(t.to_str())
    j["model"]["vocab"] += [[b, -20.0] for b in byte_tokens()]
    j["model"]["byte_fallback"] = True
    j["pre_tokenizer"] = None
    j["normalizer"] = {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"}, {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]}
    j["decoder"] = {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"},
                                                     {"type": "Strip", "content": " ", "start": 1, "stop": 0}]}
    j["post_processor"] = None
    save_hf("hf_unigram_bytefallback", Tokenizer.from_str(json.dumps(j)))

    # 13. WordLevel
    t = Tokenizer(models.WordLevel(unk_token="<unk>"))
    t.normalizer = normalizers.Lowercase()
    t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Digits(individual_digits=False)])
    t.train_from_iterator(TRAIN, trainers.WordLevelTrainer(vocab_size=300, special_tokens=["<unk>"], show_progress=False))
    save_hf("hf_wordlevel", t)

    # 14. added tokens: single_word, lstrip, rstrip, normalized (matched in lower-cased text), longest match, non-special
    t = Tokenizer.from_str(json.dumps({**json.loads(Tokenizer(models.BPE()).to_str()), "model": gpt2_model}))
    t.normalizer = normalizers.Lowercase()
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    t.decoder = decoders.ByteLevel()
    t.add_special_tokens([AddedToken("<mask>", lstrip=True, rstrip=False), AddedToken("[SEP]", rstrip=True), AddedToken("[CLS]", single_word=True)])
    t.add_tokens([AddedToken("Hello", single_word=True, normalized=True), AddedToken("World", normalized=True), AddedToken("un", single_word=False, normalized=True),
                  AddedToken("unwanted", normalized=True), AddedToken("<s>", normalized=False), AddedToken("<s>Hello", normalized=False),
                  AddedToken("##", normalized=False, lstrip=True, rstrip=True)])
    save_hf("hf_added_tokens", t)

    # 15. normalizer / pre-tokenizer matrix over one model whose merges were learned across spaces and punctuation
    cross = Tokenizer(models.BPE(unk_token="<unk>"))
    cross.train_from_iterator(TRAIN, trainers.BpeTrainer(vocab_size=450, special_tokens=["<unk>"], show_progress=False))
    cross_json = json.loads(cross.to_str())
    save_hf("hf_pipe_base", cross)
    P, N = pre_tokenizers, normalizers
    variants = {
        "split_str_removed": (None, P.Split(" ", behavior="removed")),
        "split_str_isolated": (None, P.Split(" ", behavior="isolated")),
        "split_str_prev": (None, P.Split(" ", behavior="merged_with_previous")),
        "split_str_next": (None, P.Split(" ", behavior="merged_with_next")),
        "split_str_contiguous": (None, P.Split(" ", behavior="contiguous")),
        "split_re_removed_inv": (None, P.Split(Regex(r"\w+|[^\w\s]+"), behavior="removed", invert=True)),
        "split_re_isolated_inv": (None, P.Split(Regex(r"\s+"), behavior="isolated", invert=True)),
        "split_re_prev": (None, P.Split(Regex(r"[.,!?]+|\d{2}"), behavior="merged_with_previous")),
        "split_re_next": (None, P.Split(Regex(r"[.,!?]+|\d{2}"), behavior="merged_with_next")),
        "split_re_contiguous": (None, P.Split(Regex(r"\p{P}"), behavior="contiguous")),
        "split_re_lookaround": (None, P.Split(Regex(r"(?<=[a-z])(?=[A-Z])|(?i:hello)|^\s+|\s+$"), behavior="isolated")),
        "punct_isolated": (None, P.Punctuation("isolated")),
        "punct_removed": (None, P.Punctuation("removed")),
        "punct_prev": (None, P.Punctuation("merged_with_previous")),
        "punct_next": (None, P.Punctuation("merged_with_next")),
        "punct_contiguous": (None, P.Punctuation("contiguous")),
        "digits_individual": (None, P.Digits(individual_digits=True)),
        "digits_grouped": (None, P.Digits(individual_digits=False)),
        "whitespace": (None, P.Whitespace()),
        "whitespace_split": (None, P.WhitespaceSplit()),
        "bert_pre": (None, P.BertPreTokenizer()),
        "char_delim": (None, P.CharDelimiterSplit("e")),
        "metaspace_nosplit": (None, P.Metaspace(replacement="_", prepend_scheme="first", split=False)),
        "seq_ws_punct_digits": (None, P.Sequence([P.WhitespaceSplit(), P.Punctuation("merged_with_next"), P.Digits(individual_digits=False)])),
        "norm_nfc": (N.NFC(), None), "norm_nfd": (N.NFD(), None), "norm_nfkc": (N.NFKC(), None), "norm_nfkd": (N.NFKD(), None),
        "norm_lower": (N.Lowercase(), None), "norm_strip_accents": (N.Sequence([N.NFD(), N.StripAccents()]), None),
        "norm_strip": (N.Strip(), None), "norm_strip_left": (N.Strip(left=True, right=False), None), "norm_strip_right": (N.Strip(left=False, right=True), None),
        "norm_replace_str": (N.Replace("l", "LL"), None), "norm_replace_re": (N.Replace(Regex(r"\s+"), " "), None),
        "norm_replace_re_class": (N.Replace(Regex(r"[\p{Lu}\d]+(?!\w)"), "#"), None),
        "norm_prepend": (N.Prepend(">> "), None), "norm_nmt": (N.Nmt(), None),
        "norm_bert_default": (N.BertNormalizer(), None),
        "norm_bert_cased_strip": (N.BertNormalizer(clean_text=False, handle_chinese_chars=True, strip_accents=True, lowercase=False), None),
        "norm_bert_lower_keep": (N.BertNormalizer(clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=True), None),
    }
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb
    tsv = os.path.join(os.environ.get("TMPDIR", "/tmp"), "forge_tok_rules.tsv")
    open(tsv, "w").write("".join(f"{a}\t{b}\n" for a, b in CHARSMAP_RULES))
    import io
    buf = io.BytesIO()
    spm.SentencePieceTrainer.train(sentence_iterator=iter(TRAIN), model_writer=buf, minloglevel=2, vocab_size=400, hard_vocab_limit=False, normalization_rule_tsv=tsv)
    mp = pb.ModelProto(); mp.ParseFromString(buf.getvalue())
    variants["norm_precompiled"] = (N.Precompiled(mp.normalizer_spec.precompiled_charsmap), None)
    for name, (norm, pre) in variants.items():
        t = Tokenizer.from_str(json.dumps(cross_json))
        if norm is not None: t.normalizer = norm
        if pre is not None: t.pre_tokenizer = pre
        save_hf("hf_pipe_" + name, t, base="hf_pipe_base")
    return bert_vocab


# ---------------------------------------------------------------------------------------------- SentencePiece
# A small normalization rule table (source code points -> target code points) instead of the 240 KiB nmt_nfkc one:
# the precompiled charsmap SentencePiece builds from it has the same double-array layout.
CHARSMAP_RULES = [("9", "20"), ("A", "20"), ("D", "20"), ("A0", "20"), ("3000", "20"), ("2003", "20"), ("200B", ""), ("FB01", "66 69"), ("2460", "31"),
                  ("BD", "31 2F 32"), ("65 301", "E9"), ("61 308", "E4"), ("1C5", "64 17E"), ("2026", "2E 2E 2E"), ("FF76 FF9E", "30AC")] \
    + [(f"{0xFF21 + i:X}", f"{0x41 + i:X}") for i in range(26)] + [(f"{0xFF10 + i:X}", f"{0x30 + i:X}") for i in range(10)]


def build_spm():
    import io
    import sentencepiece as spm
    out = {}
    tsv = os.path.join(os.environ.get("TMPDIR", "/tmp"), "forge_tok_rules.tsv")
    open(tsv, "w").write("".join(f"{a}\t{b}\n" for a, b in CHARSMAP_RULES))
    def train(name, **kw):
        buf = io.BytesIO()
        spm.SentencePieceTrainer.train(sentence_iterator=iter(TRAIN), model_writer=buf, minloglevel=2, **kw)
        data = buf.getvalue()
        open(os.path.join(HERE, name + ".model"), "wb").write(data)
        sp = spm.SentencePieceProcessor(model_proto=data)
        cases = []
        for raw in CORPUS:
            text = raw.decode("utf-8", "replace")
            ids = sp.encode(text)
            # SentencePiece prints an unknown piece as " \u2047 "; FORGE prints the vocabulary entry
            # ... and a literal "\u2581" in the input is not told apart from a space by either decoder
            cases.append({"ids": ids, "dec": None if sp.unk_id() in ids or "\u2581" in text else hx(sp.decode(ids))})
        # FORGE parses control pieces written out in the text ("<s>"), SentencePiece spells them: those lines are not comparable
        ctl = [sp.id_to_piece(i) for i in range(sp.get_piece_size()) if sp.is_control(i) or sp.is_unknown(i)]
        lines = [i for i, raw in enumerate(CORPUS) if not any(c in raw.decode("utf-8", "replace") for c in ctl)]
        fixtures[name] = {"oracle": "sentencepiece", "file": name + ".model", "lines": lines, "cases": cases, "bos": sp.bos_id(), "eos": sp.eos_id(), "unk": sp.unk_id()}
        out[name] = sp
        return sp
    # Llama-2 / Mistral: BPE, identity normalization, no whitespace collapsing, dummy prefix, byte fallback
    train("spm_bpe_llama", model_type="bpe", vocab_size=560, byte_fallback=True, normalization_rule_name="identity", remove_extra_whitespaces=False,
          add_dummy_prefix=True, character_coverage=1.0, unk_id=0, bos_id=1, eos_id=2, pad_id=-1)
    # BPE without byte fallback: unknown characters become <unk>, runs fused
    train("spm_bpe_unk", model_type="bpe", vocab_size=300, byte_fallback=False, normalization_rule_name="identity", remove_extra_whitespaces=False,
          add_dummy_prefix=True, character_coverage=0.98, unk_id=0, bos_id=1, eos_id=2, pad_id=-1)
    # T5 / XLM-R: Unigram with a precompiled charsmap and whitespace collapsing
    train("spm_unigram_charsmap", model_type="unigram", vocab_size=360, hard_vocab_limit=False, character_coverage=0.995, unk_id=2, bos_id=-1, eos_id=1, pad_id=0,
          normalization_rule_tsv=tsv)
    # Gemma style: Unigram, identity normalization, byte fallback, no dummy prefix
    train("spm_unigram_bytes", model_type="unigram", vocab_size=600, hard_vocab_limit=False, byte_fallback=True, normalization_rule_name="identity", remove_extra_whitespaces=False,
          add_dummy_prefix=False, character_coverage=1.0, unk_id=3, bos_id=2, eos_id=1, pad_id=0, user_defined_symbols=["<start_of_turn>", "<end_of_turn>"])
    return out


# ---------------------------------------------------------------------------------------------- GGUF
def build_gguf(spms, bert_vocab):
    import gguf
    def writer(name):
        w = gguf.GGUFWriter(os.path.join(HERE, name + ".gguf"), "llama")
        return w
    def finish(w):
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    def alias(name, of, **extra):
        if "lines" in fixtures[of]: extra.setdefault("lines", fixtures[of]["lines"])
        fixtures[name] = {"oracle": fixtures[of]["oracle"] + " (goldens of " + of + ")", "file": name + ".gguf", "same_as": of, **extra}

    # gpt2 models with a named pre-tokenizer: same vocabulary and merges as the tokenizer.json fixture
    for name, src, pre in (("gguf_gpt2_llama_bpe", "hf_bpe_llama3", "llama-bpe"), ("gguf_gpt2_qwen2", "hf_bpe_qwen2", "qwen2"), ("gguf_gpt2_default", "hf_bpe_gpt2", "gpt-2")):
        j = json.load(open(os.path.join(HERE, src + ".json"), encoding="utf-8"))
        vocab = dict(j["model"]["vocab"]); added = {a["content"]: a for a in j["added_tokens"]}
        for a in j["added_tokens"]: vocab[a["content"]] = a["id"]
        toks = [None] * len(vocab)
        for t, i in vocab.items(): toks[i] = t
        types = [(3 if added[t]["special"] else 4) if t in added else 1 for t in toks]
        w = writer(name)
        w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre(pre); w.add_token_list(toks); w.add_token_types(types)
        w.add_token_merges([m if isinstance(m, str) else " ".join(m) for m in j["model"]["merges"]])
        if src == "hf_bpe_llama3": w.add_bos_token_id(vocab["<|begin_of_text|>"]); w.add_eos_token_id(vocab["<|eot_id|>"]); w.add_add_bos_token(True)
        else: w.add_add_bos_token(False)
        finish(w)
        # Qwen's tokenizer.json NFC-normalizes; GGUF has no field for that (llama.cpp does not normalize either)
        import unicodedata
        nfc = [i for i, c in enumerate(CORPUS) if unicodedata.is_normalized("NFC", c.decode("utf-8", "replace"))]
        alias(name, src, check=["ids", "dec"] + (["ids_special"] if src != "hf_bpe_gpt2" else []), **({"lines": nfc} if src == "hf_bpe_qwen2" else {}))

    # SentencePiece models: pieces, scores and types as llama.cpp's converter writes them
    for name, src, model in (("gguf_llama_spm", "spm_bpe_llama", "llama"), ("gguf_llama_spm_unk", "spm_bpe_unk", "llama"),
                             ("gguf_t5_unigram", "spm_unigram_charsmap", "t5"), ("gguf_unigram_bytes", "spm_unigram_bytes", "t5")):
        sp = spms[src]
        from sentencepiece import sentencepiece_model_pb2 as pb
        m = pb.ModelProto(); m.ParseFromString(sp.serialized_model_proto())
        toks, scores, types = [], [], []
        for i in range(sp.get_piece_size()):
            toks.append(sp.id_to_piece(i)); scores.append(sp.get_score(i))
            types.append(2 if sp.is_unknown(i) else 3 if sp.is_control(i) else 5 if sp.is_unused(i) else 6 if sp.is_byte(i) else 4 if m.pieces[i].type == 4 else 1)
        w = writer(name)
        w.add_tokenizer_model(model); w.add_token_list(toks); w.add_token_scores(scores); w.add_token_types(types)
        if sp.unk_id() >= 0: w.add_unk_token_id(sp.unk_id())
        if sp.bos_id() >= 0: w.add_bos_token_id(sp.bos_id())
        if sp.eos_id() >= 0: w.add_eos_token_id(sp.eos_id())
        w.add_add_bos_token(sp.bos_id() >= 0); w.add_add_eos_token(model == "t5")
        w.add_add_space_prefix(m.normalizer_spec.add_dummy_prefix)
        if model == "t5":
            w.add_remove_extra_whitespaces(m.normalizer_spec.remove_extra_whitespaces)
            if m.normalizer_spec.precompiled_charsmap: w.add_precompiled_charsmap(m.normalizer_spec.precompiled_charsmap)
        finish(w)
        alias(name, src, check=["ids"])

    # BERT: llama.cpp's converter stores word-initial pieces with a "▁" prefix and continuation pieces bare
    toks = [None] * len(bert_vocab)
    for t, i in bert_vocab.items(): toks[i] = t
    conv = [t if t.startswith("[") and t.endswith("]") else t[2:] if t.startswith("##") else "▁" + t for t in toks]
    types = [3 if t.startswith("[") and t.endswith("]") else 1 for t in toks]
    w = writer("gguf_bert")
    w.add_tokenizer_model("bert"); w.add_token_list(conv); w.add_token_types(types); w.add_token_type_count(2)
    w.add_unk_token_id(bert_vocab["[UNK]"]); w.add_bos_token_id(bert_vocab["[CLS]"]); w.add_eos_token_id(bert_vocab["[SEP]"])
    w.add_sep_token_id(bert_vocab["[SEP]"]); w.add_pad_token_id(bert_vocab["[PAD]"]); w.add_mask_token_id(bert_vocab["[MASK]"])
    finish(w)
    # llama.cpp's WPM tokenizer lower-cases and NFD-normalizes but keeps accents, so only the lines without them
    # are comparable with the BertNormalizer goldens
    plain = [i for i, c in enumerate(CORPUS) if c.isascii() and b"\x00" not in c]
    alias("gguf_bert", "hf_wordpiece_bert", check=["ids", "ids_special"], lines=plain)

    # a whole (random, one-layer) Llama with the SentencePiece vocabulary and a chat template: import_gguf end to end
    import numpy as np
    sp = spms["spm_bpe_llama"]; rng = np.random.default_rng(0); d, V = 8, sp.get_piece_size()
    w = writer("gguf_tiny_llama")
    w.add_block_count(1); w.add_context_length(64); w.add_embedding_length(d); w.add_feed_forward_length(16); w.add_head_count(2); w.add_head_count_kv(1)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_tokenizer_model("llama"); w.add_token_list([sp.id_to_piece(i) for i in range(V)]); w.add_token_scores([sp.get_score(i) for i in range(V)])
    w.add_token_types([2 if sp.is_unknown(i) else 3 if sp.is_control(i) else 6 if sp.is_byte(i) else 1 for i in range(V)])
    w.add_unk_token_id(0); w.add_bos_token_id(1); w.add_eos_token_id(2); w.add_add_bos_token(True); w.add_add_eos_token(False)
    w.add_chat_template("{{ bos_token }}{% for m in messages %}{% if m['role'] == 'user' %}[INST] {{ m['content'] }} [/INST]{% else %}{{ m['content'] }}{{ eos_token }}{% endif %}{% endfor %}")
    f32 = lambda *shape: rng.normal(size=shape).astype(np.float32)
    w.add_tensor("token_embd.weight", f32(V, d)); w.add_tensor("output_norm.weight", np.ones(d, np.float32))
    for n, shape in (("attn_norm", (d,)), ("ffn_norm", (d,)), ("attn_q", (d, d)), ("attn_k", (4, d)), ("attn_v", (4, d)), ("attn_output", (d, d)),
                     ("ffn_gate", (16, d)), ("ffn_up", (16, d)), ("ffn_down", (d, 16))):
        w.add_tensor(f"blk.0.{n}.weight", np.ones(shape, np.float32) if len(shape) == 1 else f32(*shape) * 0.3)
    finish(w)
    alias("gguf_tiny_llama", "spm_bpe_llama", check=["ids"])

    # chat template and misc metadata
    w = writer("gguf_chat_template")
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("qwen2"); w.add_token_list(["a", "b", "<|im_start|>", "<|im_end|>"]); w.add_token_types([1, 1, 3, 3]); w.add_token_merges([])
    w.add_eos_token_id(3); w.add_add_bos_token(False)
    w.add_chat_template("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    finish(w)


def build_legacy(olddump, previous):
    """A version 1 section: what the GGUF importer used to write (id + content for control tokens, nothing else)."""
    j = json.load(open(os.path.join(HERE, "hf_bpe_gpt2.json"), encoding="utf-8"))
    n = len(j["model"]["vocab"])
    v1 = {"added_tokens": [{"id": 0, "content": "<|endoftext|>"}, {"id": n, "content": "<s>"}, {"id": n + 1, "content": "<|im_start|>"}, {"id": n + 2, "content": "<|im_end|>"}],
          "model": {"type": "BPE", "vocab": j["model"]["vocab"], "merges": j["model"]["merges"]}}
    text = json.dumps(v1, ensure_ascii=False, separators=(",", ":"))
    open(os.path.join(HERE, "legacy_v1.json"), "w", encoding="utf-8").write(text)
    if olddump:
        r = subprocess.run([olddump, os.path.join(HERE, "legacy_v1.json")], input="\n".join(c.hex() for c in CORPUS) + "\n", capture_output=True, text=True, check=True)
        cases = [json.loads(l) for l in r.stdout.splitlines()]
        assert len(cases) == len(CORPUS)
        fixtures["legacy_v1"] = {"oracle": "FORGE before the tokenizer pipeline (commit b135941)", "file": "legacy_v1.json", "cases": cases}
    elif previous and "legacy_v1" in previous.get("fixtures", {}) and previous["corpus"] == [c.hex() for c in CORPUS]:
        fixtures["legacy_v1"] = previous["fixtures"]["legacy_v1"]
    else: print("legacy_v1: no goldens (run with --legacy)")


def main():
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    gp = os.path.join(HERE, "goldens.json")
    previous = json.load(open(gp, encoding="utf-8")) if os.path.exists(gp) else None
    bert_vocab = build_hf()
    build_legacy(args.get("--legacy"), previous)
    spms = build_spm()
    build_gguf(spms, bert_vocab)
    with open(os.path.join(HERE, "goldens.json"), "w") as f:   # one fixture per line: readable diffs
        f.write('{"corpus":' + json.dumps([c.hex() for c in CORPUS], separators=(",", ":")) + ',\n"fixtures":{\n')
        f.write(",\n".join(json.dumps(k) + ":" + json.dumps(v, ensure_ascii=False, separators=(",", ":")) for k, v in fixtures.items()))
        f.write("\n}}\n")
    print(len(fixtures), "fixtures,", len(CORPUS), "corpus lines")
    if "--check" in args:   # quick loop against a FORGE build: tokdump <tokenizer.json> < hex lines
        bad = 0
        for name, fx in fixtures.items():
            if fx["oracle"] != "tokenizers": continue
            tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "forge_tok_fixture.json")
            open(tmp, "w", encoding="utf-8").write(compose(fx))
            r = subprocess.run([args["--check"], tmp], input="\n".join(c.hex() for c in CORPUS) + "\n", capture_output=True, text=True)
            if r.returncode: print(name, "FAILED:", r.stderr.strip()); bad += 1; continue
            got = [json.loads(l) for l in r.stdout.splitlines()]
            for i, (g, e) in enumerate(zip(got, fx["cases"])):
                for k in e:
                    want = e[k].encode().hex() if isinstance(e[k], str) else [x.encode().hex() for x in e[k]] if e[k] and isinstance(e[k][0], str) else e[k]
                    if g[k] != want:
                        bad += 1
                        show = lambda v: bytes.fromhex(v).decode("utf-8", "replace") if isinstance(v, str) else [bytes.fromhex(x).decode("utf-8", "replace") for x in v] if v and isinstance(v[0], str) else v
                        print(f"{name} line {i} {k}: {CORPUS[i]!r}\n   want {e[k]!r}\n   got  {show(g[k])!r}")
                        break
        print("mismatches:", bad)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Python binding test: greedy ids against the CLI, forward / forward_all / rewind, grammar, draft model, and the
ABI 4 surface: shared weights, the full sampler chain, true streaming, stop tokens, JSON Schema, adapters, KV state,
embeddings, batching, tokenizer details, context-manager lifetimes.

  python_bindings.py <tiny_f32.fpk> <forge> <draft.fpk> [<bert.fpk> [<work dir>]]
"""
import gc, json, os, subprocess, sys, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
fpk, cli, draft = sys.argv[1], sys.argv[2], sys.argv[3]
bert = sys.argv[4] if len(sys.argv) > 4 else None
work = sys.argv[5] if len(sys.argv) > 5 else None
import forge
assert forge.abi_version() >= 4, forge.abi_version()
m = forge.Model(fpk)
ids = m.tokenize("the weather")
assert ids[0] == 290 and m.detokenize(ids[1:]) == "the weather", (ids, m.detokenize(ids))
want = [int(x) for x in subprocess.run([cli, "run", fpk, "--prompt", "the weather", "-n", "12", "--ids"], capture_output=True, text=True, check=True).stdout.split()]
got = m.generate(ids, max_tokens=12)
assert got == want and 291 not in want, (got, want)
assert m.last_stop_reason == "length"
m.reset(); a = m.forward_all(ids); assert a.shape == (len(ids), m.vocab_size)
m.reset(); l = m.forward(ids); assert np.abs(a[-1] - l).max() < 1e-4
m.rewind(2); assert m.position == 2
d = forge.Model(draft)
assert m.generate(ids, max_tokens=12, draft=d, spec_k=3) == want
seen = []
txt = m.detokenize(m.generate(ids, max_tokens=30, temperature=0.9, seed=4, grammar='root ::= "yes" | "no" | "maybe"', on_token=lambda t, s: seen.append(s) and False))
assert txt in ("yes", "no", "maybe"), txt
assert "".join(seen) == txt and m.last_stop_reason == "grammar"
stopped = m.generate(ids, max_tokens=12, on_token=lambda t, s: True)
assert len(stopped) == 1 and m.last_stop_reason == "callback"
try:
    forge.Model("/nonexistent.fpk"); raise SystemExit("expected error")
except forge.ForgeError as e:
    assert "nonexistent" in str(e)
a = m.generate(ids, max_tokens=10, temperature=0.8, seed=5, typical_p=0.5)
assert a == m.generate(ids, max_tokens=10, temperature=0.8, seed=5, typical_p=0.5) and len(a) == 10
b = m.generate(ids, max_tokens=10, temperature=1.0, seed=5, mirostat_tau=3.0, mirostat_eta=0.2)
assert len(b) == 10
class Boom(Exception): pass
def boom(t, s): raise Boom("from the callback")
try: m.generate(ids, max_tokens=5, on_token=boom); raise SystemExit("expected the callback's exception")
except Boom: pass

# ---- the full sampler chain
pen = m.generate(ids, max_tokens=12, presence_penalty=50.0, frequency_penalty=5.0)
assert len(pen) == 12 and len(set(pen)) == 12, pen                       # huge penalties forbid repeats
assert m.generate(ids, max_tokens=4, logit_bias={want[0]: -1e9})[0] != want[0]
assert m.generate(ids, max_tokens=3, logit_bias={65: 1e9}) == [65, 65, 65]
s1 = m.generate(ids, max_tokens=12, temperature=1.0, top_k=50, top_p=0.9, min_p=0.01, repetition_penalty=1.2, repetition_window=16, seed=11)
assert s1 == m.generate(ids, max_tokens=12, temperature=1.0, top_k=50, top_p=0.9, min_p=0.01, repetition_penalty=1.2, repetition_window=16, seed=11)
assert s1 != m.generate(ids, max_tokens=12, temperature=1.0, top_k=50, top_p=0.9, min_p=0.01, repetition_penalty=1.2, repetition_window=16, seed=12)
# stop tokens end the output and are not returned; EOS likewise (forced here), unless ignore_eos
k = next(i for i in range(1, 12) if want[i] not in want[:i])
assert m.generate(ids, max_tokens=12, stop_tokens=[want[k]]) == want[:k] and m.last_stop_reason == "eos"
assert m.generate(ids, max_tokens=5, logit_bias={m.eos_token: 1e9}) == [] and m.last_stop_reason == "eos"
assert m.generate(ids, max_tokens=3, logit_bias={m.eos_token: 1e9}, ignore_eos=True) == [m.eos_token] * 3
# JSON Schema and grammar helpers
out = m.complete("the", max_tokens=16, temperature=0.9, seed=3, json_schema={"enum": ["red", "green", 7]})
assert json.loads(out) in ("red", "green", 7), out
g = forge.json_schema_to_grammar({"type": "object", "properties": {"a": {"type": "boolean"}}, "required": ["a"]})
assert g.startswith("root ::= ") and forge.grammar_check(g) is None
assert json.loads(m.complete("the", max_tokens=24, grammar=g)) in ({"a": True}, {"a": False})
for bad in (lambda: forge.grammar_check('root ::= "open'), lambda: forge.json_schema_to_grammar("{nope"), lambda: m.generate(ids, grammar="root ::= \"a\"", json_schema={})):
    try: bad(); raise SystemExit("expected ForgeError")
    except forge.ForgeError as e: assert str(e)
# reset=False continues the sequence
first = m.generate(ids, max_tokens=4)
more = m.generate([first[-1]], max_tokens=4, reset=False)
assert first + more == want[:4] + want[4:8], (first, more, want)

# ---- true streaming: pieces arrive while the model is still generating
events = []
def slow(t, s): events.append(("token", time.monotonic())); time.sleep(0.02); return False
pieces = []
for piece in m.stream("the weather", max_tokens=12, on_token=slow):
    events.append(("piece", time.monotonic())); pieces.append(piece)
kinds = [k for k, _ in events]
assert kinds.count("token") == 12 and kinds.index("piece") <= 2, kinds   # the first piece arrived while the second token was being made
assert "".join(pieces) == m.detokenize(want), (pieces, m.detokenize(want))
assert kinds.index("piece") < len(kinds) - 6, "the pieces only arrived at the end"
# closing the generator early stops the generation
it = m.stream(ids, max_tokens=64, ignore_eos=True, on_token=slow); next(it); it.close()
assert m.position < len(ids) + 10, m.position
assert threading.active_count() == 1, "the stream worker thread is gone"
# an error inside generation surfaces from the generator
try: list(m.stream(ids, max_tokens=4, grammar='root ::= "open')); raise SystemExit("expected ForgeError")
except forge.ForgeError: pass
assert m.generate(ids, max_tokens=12) == want, "the model is usable after streams"

# ---- tokenizer details, manifest
assert m.bos_token == 290 and m.eos_token == 291 and m.is_special(291) and not m.is_special(65)
assert m.token_bytes(65) == b"A" and m.token_id("</s>") == 291 and m.token_id("nope-nope") == -1
assert m.stop_tokens() == [291] and m.chat_template == "" and m.kind == "generative_sequence"
assert m.manifest["format"] == "fpk" and m.context_size == 128 and m.manifest["max_position_embeddings"] == 128
try: m.apply_chat_template([{"role": "user", "content": "hi"}]); raise SystemExit("expected ForgeError")
except forge.ForgeError as e: assert "chat template" in str(e)
try: m.token_bytes(10 ** 6); raise SystemExit("expected ForgeError")
except forge.ForgeError as e: assert "vocabulary" in str(e)

# ---- shared weights, sessions, KV state
with forge.Weights(fpk, device="cpu", deterministic=True) as w:
    assert w.device == "cpu" and w.set_device("no-such-device") == "cpu"
    s1, s2 = w.session(context=64), w.session()
    assert s1.context_size == 64 and s2.context_size == 128 and s1.weights is w
    assert s1.generate(ids, max_tokens=12) == want and s2.generate(ids, max_tokens=12, draft=s1, spec_k=2) == want
    s1.reset(); l1 = s1.forward(ids); blob = s1.save_state()
    assert isinstance(blob, bytes) and len(blob) > 16 and s1.state_bytes > 0
    s2.load_state(blob); assert s2.position == len(ids)
    la = s1.forward([want[0]])
    assert np.array_equal(la, s2.forward([want[0]])) and not np.array_equal(la, l1)
    s3 = w.session(); s3.copy_state_from(s1, len(ids)); assert s3.position == len(ids)   # a prefix of s1 (which is one token further)
    assert np.array_equal(s3.forward([want[0]]), la)
    s4 = w.session(); s4.copy_state_from(s1); assert s4.position == s1.position
    for bad in (lambda: s2.load_state(blob[:len(blob) // 2]), lambda: s2.load_state(b"junk"), lambda: s3.copy_state_from(m, 1), lambda: s1.save_state(10 ** 6)):
        try: bad(); raise SystemExit("expected ForgeError")
        except forge.ForgeError as e: assert str(e)
    assert s2.position == 0, "a refused state leaves the session empty"
    # the session's own error, not another handle's
    try: s1.token_bytes(-1); raise SystemExit("expected ForgeError")
    except forge.ForgeError as e: assert "vocabulary" in str(e)

    # ---- LoRA adapters
    assert not w.has_adapter("tuned") and s1.adapter == ""
    if work:
        from safetensors.numpy import save_file
        rng = np.random.default_rng(3)
        lora = os.path.join(work, "py_lora.safetensors")
        save_file({"layer.0.wq.lora_a": rng.normal(size=(2, 64)).astype(np.float32), "layer.0.wq.lora_b": rng.normal(size=(64, 2)).astype(np.float32),
                   "lora_alpha": np.array([8.0], np.float32)}, lora)
        w.load_adapter("tuned", lora); assert w.has_adapter("tuned")
        try: w.load_adapter("tuned", lora); raise SystemExit("expected ForgeError")
        except forge.ForgeError: pass
        s1.adapter = "tuned"; assert s1.adapter == "tuned" and s1.position == 0
        tuned = s1.generate(ids, max_tokens=12); assert tuned != want
        s1.adapter = None; assert s1.adapter == "" and s1.generate(ids, max_tokens=12) == want
        try: s1.adapter = "missing"; raise SystemExit("expected ForgeError")
        except forge.ForgeError as e: assert "missing" in str(e)
    else: tuned = None

    # ---- batching
    with forge.Batch(w, max_active=3) as batch:
        prompts = [m.tokenize(t) for t in ("the weather", "in the", "a cat", "on the mat")]
        assert batch.generate_many(prompts, max_tokens=8, logit_bias={291: -1e9}) == [m.generate(p, max_tokens=8, logit_bias={291: -1e9}) for p in prompts]
        got, toks = {}, []
        r1 = batch.submit(ids, max_tokens=6, on_token=lambda t, s: toks.append(t) and False, on_done=lambda rid, out, why: got.setdefault(rid, (out, why)))
        r2 = batch.submit(ids, max_tokens=12, grammar='root ::= "yes" | "no"', temperature=0.8, seed=2)
        r3 = batch.submit(ids, max_tokens=50, priority=-5)
        r4 = batch.submit(ids, max_tokens=12, adapter="tuned") if tuned else None
        assert batch.cancel(r3) and not batch.cancel(r3) and batch.results[r3] == ([], "cancelled")
        assert batch.step() is True
        batch.run(); assert batch.step() is False
        assert got[r1] == (want[:6], "length") and toks == want[:6] and batch.results[r1][0] == want[:6]
        assert m.detokenize(batch.results[r2][0]) in ("yes", "no") and batch.results[r2][1] == "stop"
        if tuned:   # the scheduler's result includes the EOS that ended it, generate() does not
            res = batch.results[r4][0]; assert res == tuned or (res[-1] == 291 and res[:-1] == tuned), (res, tuned)
        st = batch.stats()
        assert st["completed"] == 4 + 3 + (1 if tuned else 0) and st["failed"] == 0 and st["waiting"] == 0 and st["peak_active"] == 3 and st["prefix_hits"] >= 1, st
        batch.evict_prefix_cache()
        try: batch.submit(ids, adapter="missing"); raise SystemExit("expected ForgeError")
        except forge.ForgeError as e: assert "missing" in str(e)
        try: batch.submit([]); raise SystemExit("expected ForgeError")
        except forge.ForgeError as e: assert "empty prompt" in str(e)
        try: batch.generate_many([[5] * 500], max_tokens=2); raise SystemExit("expected ForgeError")
        except forge.ForgeError as e: assert "context" in str(e)       # the failed request's own message
        def boom_done(rid, out, why): raise Boom("done")
        batch.submit(ids, max_tokens=2, on_done=boom_done)
        try: batch.run(); raise SystemExit("expected the callback's exception")
        except Boom: pass
    # closing the weights (end of the with block) before the sessions is fine
assert s1.generate(ids, max_tokens=12) == want
for s in (s1, s2, s3, s4): s.close()
s1.close()   # twice is harmless

# ---- embeddings
if bert:
    with forge.Model(bert) as e:
        assert e.kind == "embedding" and e.embed_dim > 0
        texts = ["The quick brown fox jumps over the lazy dog.", "Hello, world!", "it's John's"]
        one = e.embed(texts[1]); assert one.shape == (e.embed_dim,) and float(np.abs(one).sum()) > 0
        rows = e.embed(texts); assert rows.shape == (3, e.embed_dim) and np.array_equal(rows[1], one)
        assert np.array_equal(e.embed(e.tokenize(texts[1])), one)            # tokenize() adds [CLS] / [SEP] by default
        assert not np.array_equal(e.embed(e.tokenize(texts[1], add_bos=False)), one)
        ref = os.path.join(work or ".", "py_embed.f32")
        subprocess.run([cli, "embed", bert, "--text", texts[0], "--out", ref], check=True, capture_output=True)
        assert np.abs(np.fromfile(ref, np.float32) - rows[0]).max() < 1e-6
        os.remove(ref)
        pair = e.embed("query", pair="passage"); assert pair.shape == one.shape and not np.array_equal(pair, e.embed("query"))
        for bad in (lambda: e.embed([]), lambda: e.embed(" ".join(["fox"] * 600)), lambda: e.generate([1, 2], max_tokens=2)):
            try: bad(); raise SystemExit("expected ForgeError")
            except forge.ForgeError as err: assert str(err)
    try: m.embed("x"); raise SystemExit("expected ForgeError")
    except forge.ForgeError as err: assert "embedding" in str(err)

# ---- AOT sidecar
if work:
    import shutil
    copy = os.path.join(work, "py_aot.fpk"); shutil.copy(fpk, copy)
    assert forge.aot_precompile(copy, seq_lengths=[1, 3]) == 2 and os.path.exists(copy + ".fplan")
    with forge.Model(copy) as pm: assert pm.generate(ids, max_tokens=12) == want
    os.remove(copy + ".fplan"); os.remove(copy)
# the library names searched per platform (only this platform's can be loaded here)
plat = sys.platform
for name, first in (("win32", "forge.dll"), ("darwin", "libforge.dylib"), ("linux", "libforge.so"), ("freebsd14", "libforge.so")):
    sys.platform = name; assert forge._lib_names()[0] == first, (name, forge._lib_names())
sys.platform = plat
assert forge.set_deterministic(True) is True and forge.set_deterministic(False) is False
forge.set_deterministic(None)

# ---- lifetimes: context managers close, closed objects can be collected, a session keeps its weights alive
with forge.Model(fpk, context=32) as cm:
    assert cm.context_size == 32 and len(cm.generate(ids, max_tokens=3)) == 3
assert cm._h is None
w = forge.Weights(fpk); s = w.session(); w.close(); gc.collect()
assert s.generate(ids, max_tokens=12) == want
s.close(); d.close(); m.close(); gc.collect()
print("python bindings ok:", forge.version(), "abi", forge.abi_version(), "grammar ->", txt)

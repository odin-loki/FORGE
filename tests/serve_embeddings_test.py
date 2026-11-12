#!/usr/bin/env python3
"""forge-serve /v1/embeddings: an embedding model (the tiny BERT of verify/golden/tiny_bert.py, WordPiece tokenizer)
answers with the vectors `forge embed` computes; a generative model answers 400.

  serve_embeddings_test.py <forge-serve> <forge> <bert.fpk> <generative.fpk> <work dir>
"""
import base64, json, os, struct, subprocess, sys, threading, urllib.error, urllib.request
serve, cli, bert, gen, work = sys.argv[1:6]
os.makedirs(work, exist_ok=True)

def start(fpk):
    p = subprocess.Popen([serve, fpk, "--port", "0"], stdout=subprocess.PIPE, text=True)
    line = p.stdout.readline()
    return p, f"http://127.0.0.1:{int(line.rsplit(':', 1)[1].split()[0])}"

def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())

def refused(base, path, body, needle, method="POST"):
    """The request must fail with 400 (405 for a wrong method) and an OpenAI-style error naming `needle`."""
    try:
        if method == "GET": urllib.request.urlopen(base + path, timeout=60)
        else: post(base, path, body)
    except urllib.error.HTTPError as e:
        msg = json.loads(e.read())["error"]["message"]
        assert e.code == (405 if method == "GET" else 400) and needle in msg, (e.code, msg)
        return msg
    raise SystemExit(f"expected an error for {path} {body}")

def cli_embed(args, tag):
    """`forge embed`: the vector, and the token ids it printed to stderr with --ids (None for --tokens)."""
    out = os.path.join(work, tag + ".f32")
    r = subprocess.run([cli, "embed", bert] + args + ["--out", out], capture_output=True, text=True, check=True)
    v = open(out, "rb").read()
    ids = [int(x) for x in r.stderr.split()] if "--ids" in args else None
    return list(struct.unpack(f"<{len(v) // 4}f", v)), ids

def close(a, b): return len(a) == len(b) and max(abs(x - y) for x, y in zip(a, b)) < 1e-6

texts = ["The quick brown fox jumps over the lazy dog.", "Hello, world!", "it's John's"]
want = [cli_embed(["--text", t, "--ids"], f"t{i}") for i, t in enumerate(texts)]
assert all(ids[0] == 2 and ids[-1] == 3 for _, ids in want), "the tokenizer wraps a text in [CLS] / [SEP]"

p, base = start(bert)
try:
    # one string
    r = post(base, "/v1/embeddings", {"input": texts[0], "model": "anything"})
    assert r["object"] == "list" and len(r["data"]) == 1 and r["model"] == os.path.basename(bert), r.keys()
    d = r["data"][0]
    assert d["object"] == "embedding" and d["index"] == 0 and close(d["embedding"], want[0][0]), "string input"
    assert r["usage"] == {"prompt_tokens": len(want[0][1]), "total_tokens": len(want[0][1])}, r["usage"]
    # a list of strings: one vector each, in order; usage sums the tokens
    r = post(base, "/v1/embeddings", {"input": texts, "encoding_format": "float"})
    assert [d["index"] for d in r["data"]] == [0, 1, 2]
    for d, (v, _) in zip(r["data"], want): assert close(d["embedding"], v), d["index"]
    n = sum(len(ids) for _, ids in want)
    assert r["usage"] == {"prompt_tokens": n, "total_tokens": n}, r["usage"]
    # token ids are taken as they are (no special tokens added): one list, and a list of lists
    ids = want[1][1]
    r = post(base, "/v1/embeddings", {"input": ids})
    assert len(r["data"]) == 1 and close(r["data"][0]["embedding"], want[1][0]) and r["usage"]["prompt_tokens"] == len(ids)
    bare, _ = cli_embed(["--tokens", ",".join(map(str, ids[1:-1]))], "bare")
    r = post(base, "/v1/embeddings", {"input": [ids, ids[1:-1]]})
    assert close(r["data"][0]["embedding"], want[1][0]) and close(r["data"][1]["embedding"], bare)
    assert not close(bare, want[1][0]), "[CLS] / [SEP] change the embedding (so the text path really added them)"
    # base64: little-endian f32 bytes of the same vector
    r = post(base, "/v1/embeddings", {"input": texts[1], "encoding_format": "base64"})
    raw = base64.b64decode(r["data"][0]["embedding"])
    assert len(raw) == 4 * len(want[1][0]) and close(list(struct.unpack(f"<{len(raw) // 4}f", raw)), want[1][0])
    # dimensions: only the model's own size
    dim = len(want[0][0])
    assert len(post(base, "/v1/embeddings", {"input": "a", "dimensions": dim})["data"][0]["embedding"]) == dim
    refused(base, "/v1/embeddings", {"input": "a", "dimensions": dim // 2}, "dimensions")
    # concurrent requests give the same vectors
    got = {}
    def worker(i): got[i] = post(base, "/v1/embeddings", {"input": texts[i % 3]})["data"][0]["embedding"]
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(9)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert all(close(got[i], want[i % 3][0]) for i in range(9))
    # batching: many inputs of mixed lengths in one request (run shortest first inside the server, session::embed_batch)
    # come back in input order with the vectors each gives alone; texts and id lists the same way
    order = [0, 2, 1, 1, 0, 2, 2, 0]
    r = post(base, "/v1/embeddings", {"input": [texts[i] for i in order]})
    assert [d["index"] for d in r["data"]] == list(range(len(order))) and all(close(d["embedding"], want[i][0]) for d, i in zip(r["data"], order))
    assert r["usage"]["prompt_tokens"] == sum(len(want[i][1]) for i in order)
    r = post(base, "/v1/embeddings", {"input": [want[i][1] for i in order], "encoding_format": "base64"})
    for d, i in zip(r["data"], order):
        raw = base64.b64decode(d["embedding"]); assert close(list(struct.unpack(f"<{len(raw) // 4}f", raw)), want[i][0])
    # one input that does not fit fails the whole request before anything is computed, naming the entry
    msg = refused(base, "/v1/embeddings", {"input": [texts[0], " ".join(["fox"] * 500), texts[1]]}, "\"input\"[1]")
    m = urllib.request.urlopen(base + "/metrics").read().decode()
    assert "forge_embedding_inputs_total" in m and int(float(next(l.split()[1] for l in m.split("\n") if l.startswith("forge_embedding_inputs_total ")))) >= 3 + 8 + 8, m
    # malformed requests: 400 with a message, and the server keeps serving
    refused(base, "/v1/embeddings", {"model": "x"}, "input")
    refused(base, "/v1/embeddings", {"input": ""}, "empty")
    refused(base, "/v1/embeddings", {"input": []}, "input")
    refused(base, "/v1/embeddings", {"input": ["a", 3]}, "input")
    refused(base, "/v1/embeddings", {"input": [1.5, 2]}, "token ids")
    refused(base, "/v1/embeddings", {"input": [10 ** 9]}, "token ids")
    refused(base, "/v1/embeddings", {"input": "a", "encoding_format": "hex"}, "encoding_format")
    refused(base, "/v1/embeddings", {"input": " ".join(["fox"] * 500)}, "positions")   # longer than the model: an error, not a silent cut
    refused(base, "/v1/embeddings", None, "POST", method="GET")
    # an embedding model has no completions
    refused(base, "/v1/completions", {"prompt": "hi"}, "embedding model")
    refused(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, "/v1/embeddings")
    assert json.loads(urllib.request.urlopen(base + "/health").read())["status"] == "ok" and p.poll() is None
finally:
    p.terminate()

# a generative model: a clear 400, and completions still work
p, base = start(gen)
try:
    msg = refused(base, "/v1/embeddings", {"input": "hello"}, "not an embedding model")
    assert "session embedding" in msg and "generative_sequence" in msg, msg
    assert post(base, "/v1/completions", {"prompt": "the", "max_tokens": 2})["choices"][0]["tokens"] is not None
finally:
    p.terminate()
print("serve_embeddings: PASS")

#!/usr/bin/env python3
"""forge-serve API surface beyond the basics of serve_test.py, on the tiny golden model and a chat variant of it
(ChatML template, <|im_end|> and <tool_call> tokens) built here:

  request fields   stop by token id, priority, logit_bias, seed, n, logprobs / top_logprobs, max_completion_tokens,
                   response_format json_object, stream_options.include_usage, end-of-turn stops of chat models
  tool calling     tool_choice none / auto / required / a function, per-template formats, streamed tool-call deltas
  HTTP             keep-alive, pipelining, chunked request bodies, Expect: 100-continue, size and time limits, CORS,
                   /health, /v1/models, /metrics, OpenAI-shaped errors, graceful shutdown
  LoRA             --lora adapters selected per request, with a prefix cache that is keyed by adapter

  serve_api_test.py <forge-serve> <forge> <golden tiny_f32.fpk> <golden hf dir> <work dir>
"""
import http.client, json, os, shutil, signal, socket, struct, subprocess, sys, threading, time, urllib.error, urllib.request
import numpy as np
serve, cli, fpk, hf, work = sys.argv[1:6]
os.makedirs(work, exist_ok=True)
NAME = os.path.basename(fpk)

def start(model, *flags):
    p = subprocess.Popen([serve, model, "--port", "0", *flags], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = p.stdout.readline()
    assert "forge-serve:" in line, (line, p.stderr.read() if p.poll() is not None else "")
    return p, int(line.split("://127.0.0.1:")[1].split()[0])

class Client:
    def __init__(self, port): self.port = port; self.base = f"http://127.0.0.1:{port}"
    def get(self, path, raw=False):
        r = urllib.request.urlopen(self.base + path, timeout=60)
        return r.read().decode() if raw else json.loads(r.read())
    def post(self, path, body, raw=False):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=120)
        return r.read().decode() if raw else json.loads(r.read())
    def fail(self, path, body, status, needle="", param=None, code=None, method="POST"):
        """The request must fail with `status` and an OpenAI-shaped error; returns (error object, headers)."""
        try:
            if method == "GET": self.get(path)
            else: self.post(path, body)
        except urllib.error.HTTPError as e:
            err = json.loads(e.read())["error"]
            assert e.code == status, (e.code, status, err)
            assert set(err) == {"message", "type", "param", "code"}, err
            assert needle in err["message"], err
            if param is not None: assert err["param"] == param, err
            if code is not None: assert err["code"] == code, err
            return err, e.headers
        raise SystemExit(f"expected HTTP {status} for {path} {body}")
    def stream(self, path, body):
        raw = self.post(path, dict(body, stream=True), raw=True)
        assert raw.rstrip().endswith("data: [DONE]"), raw[-200:]
        return [json.loads(l[6:]) for l in raw.split("\n") if l.startswith("data: ") and l != "data: [DONE]"]

def raw_http(port, data, wait=5.0, parts=None, pause=0.0):
    """Send bytes (or `parts`, with a pause between them) and read until the server closes or goes quiet."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sk:
        for i, part in enumerate(parts or [data]):
            if i: time.sleep(pause)
            sk.sendall(part)
        sk.settimeout(wait)
        out = b""
        try:
            while True:
                b = sk.recv(65536)
                if not b: break
                out += b
        except socket.timeout: pass
        return out

def split_responses(raw):
    """Content-Length framed responses of one connection, in order: [(status, headers, body)]."""
    out = []
    while raw:
        head, _, rest = raw.partition(b"\r\n\r\n")
        lines = head.decode().split("\r\n")
        hd = {l.split(":", 1)[0].lower(): l.split(":", 1)[1].strip() for l in lines[1:]}
        n = int(hd.get("content-length", 0))
        out.append((int(lines[0].split()[1]), hd, rest[:n])); raw = rest[n:]
    return out

# ------------------------------------------------------------------ the chat model: golden weights, ChatML template
chat_hf = os.path.join(work, "chat_hf")
shutil.rmtree(chat_hf, ignore_errors=True); shutil.copytree(hf, chat_hf)
IM_START, IM_END, TOOL_OPEN, TOOL_CLOSE, EOT = 280, 281, 282, 283, 284
tk = json.load(open(os.path.join(chat_hf, "tokenizer.json")))
tk["added_tokens"] += [{"id": i, "content": c, "special": True} for i, c in
                       [(IM_START, "<|im_start|>"), (IM_END, "<|im_end|>"), (TOOL_OPEN, "<tool_call>"), (TOOL_CLOSE, "</tool_call>"), (EOT, "<|eot_id|>")]]
json.dump(tk, open(os.path.join(chat_hf, "tokenizer.json"), "w"))
cfg = json.load(open(os.path.join(chat_hf, "config.json"))); cfg["max_position_embeddings"] = 512
json.dump(cfg, open(os.path.join(chat_hf, "config.json"), "w"))
TEMPLATE = ("{%- if tools %}<|im_start|>system\nT:{% for t in tools %} {{ t.function.name }}{% endfor %} <tool_call></tool_call><|im_end|>\n{% endif -%}"
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
            "{% if m.tool_calls %}{% for c in m.tool_calls %}<tool_call>{{ c.function.arguments | tojson }}</tool_call>{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}{# other families end a turn with <|eot_id|> #}")
json.dump({"chat_template": TEMPLATE}, open(os.path.join(chat_hf, "tokenizer_config.json"), "w"))
# generation_config.json lists what ends generation; the importer records it as the manifest's eos_token_ids
json.dump({"eos_token_id": [291, IM_END]}, open(os.path.join(chat_hf, "generation_config.json"), "w"))
chat_fpk = os.path.join(work, "chat.fpk")
subprocess.run([cli, "import", chat_hf, "-o", chat_fpk], check=True, capture_output=True)
man = subprocess.run([cli, "inspect", chat_fpk], capture_output=True, text=True, check=True).stdout
assert '"eos_token_ids":[291,281]' in man.replace(" ", ""), man

# LoRA adapters as .safetensors: <weight>.lora_a [r, in], <weight>.lora_b [out, r], lora_alpha
from safetensors.numpy import save_file
rng = np.random.default_rng(7)
def adapter(path, weight, out_dim, in_dim, r=4, alpha=8.0):
    save_file({weight + ".lora_a": rng.normal(0, 1, (r, in_dim)).astype(np.float32), weight + ".lora_b": rng.normal(0, 1, (out_dim, r)).astype(np.float32),
               "lora_alpha": np.array([alpha], np.float32)}, path)
    return path
lora_attn = adapter(os.path.join(work, "attn.safetensors"), "layer.0.wq", 64, 64)   # changes the KV rows of every later position
lora_head = adapter(os.path.join(work, "head.safetensors"), "lm_head", 300, 64)

MSG = [{"role": "user", "content": "hi"}]

# ================================================================== 1. request fields (plain golden model)
p, port = start(fpk, "--max-active", "4", "--lora", f"attn={lora_attn}", "--lora", f"head={lora_head}")
try:
    c = Client(port)
    base = c.post("/v1/completions", {"prompt": "the", "max_tokens": 10})["choices"][0]
    T = base["tokens"]; assert len(T) == 10 and base["finish_reason"] == "length"
    # stop as a token id (alone, in a list with strings, and as stop_token_ids): the sequence ends before that token
    k = next(i for i in range(1, 10) if T[i] not in T[:i])
    for body in ({"stop": T[k]}, {"stop": ["\x01never", T[k]]}, {"stop_token_ids": [T[k]]}):
        r = c.post("/v1/completions", dict(body, prompt="the", max_tokens=10))["choices"][0]
        assert r["tokens"] == T[:k] and r["finish_reason"] == "stop", (body, r, T)
    c.fail("/v1/completions", {"prompt": "the", "stop": [300]}, 400, "token ids", param="stop")
    c.fail("/v1/completions", {"prompt": "the", "stop": [1.5]}, 400, "token ids", param="stop")
    # priority is taken (scheduler order is covered in test_inference); a wrong type is a 400 naming the field
    assert c.post("/v1/completions", {"prompt": "the", "max_tokens": 10, "priority": 5})["choices"][0]["tokens"] == T
    c.fail("/v1/completions", {"prompt": "the", "priority": "high"}, 400, "priority", param="priority")
    # logit_bias: -100 bans a token, +100 forces one
    r = c.post("/v1/completions", {"prompt": "the", "max_tokens": 10, "logit_bias": {str(T[0]): -100}})["choices"][0]["tokens"]
    assert T[0] not in r and len(r) == 10, (r, T)
    r = c.post("/v1/completions", {"prompt": "the", "max_tokens": 5, "logit_bias": {"65": 100}})["choices"][0]
    assert r["tokens"] == [65] * 5 and r["text"] == "AAAAA", r
    c.fail("/v1/completions", {"prompt": "the", "logit_bias": {"x": 1}}, 400, "logit_bias", param="logit_bias")
    c.fail("/v1/completions", {"prompt": "the", "logit_bias": {"5": 101}}, 400, "logit_bias", param="logit_bias")
    # seed: the same seed repeats, n choices use seed, seed + 1, ...
    samp = {"prompt": "the", "max_tokens": 12, "temperature": 1.0}
    s5 = c.post("/v1/completions", dict(samp, seed=5))["choices"][0]["tokens"]
    s6 = c.post("/v1/completions", dict(samp, seed=6))["choices"][0]["tokens"]
    s7 = c.post("/v1/completions", dict(samp, seed=7))["choices"][0]["tokens"]
    assert s5 == c.post("/v1/completions", dict(samp, seed=5))["choices"][0]["tokens"] and len({tuple(s5), tuple(s6), tuple(s7)}) > 1
    r = c.post("/v1/completions", dict(samp, seed=5, n=3))
    assert [ch["index"] for ch in r["choices"]] == [0, 1, 2] and [ch["tokens"] for ch in r["choices"]] == [s5, s6, s7], r
    assert r["usage"]["completion_tokens"] == len(s5) + len(s6) + len(s7) and r["usage"]["prompt_tokens"] == 2, r["usage"]
    c.fail("/v1/completions", dict(samp, n=9), 400, "\"n\"", param="n")
    ev = c.stream("/v1/completions", dict(samp, seed=5, n=2))
    for i, want in ((0, s5), (1, s6)):
        got = [t for e in ev for ch in e["choices"] if ch["index"] == i for t in ch.get("tokens", [])]
        assert got == want, (i, got, want)
    # logprobs: completions take a count, chat takes logprobs + top_logprobs
    r = c.post("/v1/completions", {"prompt": "the", "max_tokens": 6, "logprobs": 3})["choices"][0]
    lp = r["logprobs"]
    assert len(lp["tokens"]) == len(lp["token_logprobs"]) == len(lp["top_logprobs"]) == len(lp["text_offset"]) == 6, lp
    assert all(len(t) == 3 for t in lp["top_logprobs"]) and all(x <= 0 for x in lp["token_logprobs"]), lp
    assert all(abs(max(t.values()) - x) < 1e-5 for t, x in zip(lp["top_logprobs"], lp["token_logprobs"])), "greedy picks the most likely token"
    assert lp["text_offset"][0] == 0 and lp["text_offset"] == sorted(lp["text_offset"]) and lp["tokens"][0] == " the", lp
    assert any(t.startswith("bytes:\\x") for t in lp["tokens"]), "a token that is part of a UTF-8 character is labelled by its bytes"
    r = c.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 4, "logprobs": True, "top_logprobs": 2})["choices"][0]
    content = r["logprobs"]["content"]
    assert len(content) == 4 and all(len(e["top_logprobs"]) == 2 and bytes(e["bytes"]).decode(errors="replace") is not None for e in content), content
    assert abs(content[0]["top_logprobs"][0]["logprob"] - content[0]["logprob"]) < 1e-5
    assert "logprobs" not in c.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 2})["choices"][0]
    c.fail("/v1/chat/completions", {"messages": MSG, "top_logprobs": 2}, 400, "logprobs", param="top_logprobs")
    c.fail("/v1/chat/completions", {"messages": MSG, "logprobs": True, "top_logprobs": 21}, 400, "top_logprobs", param="top_logprobs")
    ev = c.stream("/v1/completions", {"prompt": "the", "max_tokens": 6, "logprobs": 1})
    assert sum(len(ch["logprobs"]["tokens"]) for e in ev for ch in e["choices"] if "logprobs" in ch) == 6
    # max_completion_tokens (wins over max_tokens)
    assert len(c.post("/v1/completions", {"prompt": "the", "max_completion_tokens": 3})["choices"][0]["tokens"]) == 3
    assert c.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 9, "max_completion_tokens": 2})["usage"]["completion_tokens"] == 2
    # response_format json_object: any JSON object ('{' and '}' are nudged so that the random model closes it)
    r = c.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 40, "response_format": {"type": "json_object"}, "logit_bias": {"123": 30, "125": 30}})["choices"][0]
    assert r["finish_reason"] == "stop" and isinstance(json.loads(r["message"]["content"]), dict), r
    assert c.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 3, "response_format": {"type": "text"}})["usage"]["completion_tokens"] == 3
    c.fail("/v1/chat/completions", {"messages": MSG, "response_format": {"type": "xml"}}, 400, "response_format", param="response_format")
    c.fail("/v1/chat/completions", {"messages": MSG, "response_format": {"type": "json_schema"}}, 400, "json_schema", param="response_format")
    # stream_options.include_usage: usage null on the chunks, then one event with no choices and the usage
    ev = c.stream("/v1/chat/completions", {"messages": MSG, "max_tokens": 4, "stream_options": {"include_usage": True}})
    assert ev[-1]["choices"] == [] and ev[-1]["usage"]["completion_tokens"] == 4, ev[-1]
    assert ev[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""} and ev[0]["usage"] is None, ev[0]
    assert ev[-2]["choices"][0]["finish_reason"] == "length" and ev[-2]["choices"][0]["delta"] == {}, ev[-2]
    ev2 = c.stream("/v1/chat/completions", {"messages": MSG, "max_tokens": 4})
    assert ev2[-1]["choices"][0]["finish_reason"] == "length" and ev2[-1]["usage"]["completion_tokens"] == 4 and "usage" not in ev2[0]
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in ev2) == c.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 4})["choices"][0]["message"]["content"]
    # a stop string that arrives split over tokens is never streamed in part
    full = base["text"]
    assert full.startswith(" the"), full   # one token; the stop string below begins inside it
    for cut in ("he", "e" + full[4:6]):
        if "\ufffd" in cut: continue       # the stop is matched on the model's bytes, not on the sanitised text
        ev = c.stream("/v1/completions", {"prompt": "the", "max_tokens": 10, "stop": [cut]})
        assert "".join(ch["text"] for e in ev for ch in e["choices"]) == full[:full.find(cut)], (ev, full, cut)
        assert ev[-1]["choices"][0]["finish_reason"] == "stop"
    # a prompt that does not fit the context is refused up front (it used to end every running request)
    c.fail("/v1/completions", {"prompt": [5] * 128}, 400, "context", code="context_length_exceeded")
    c.fail("/v1/completions", {"prompt": "x", "max_tokens": "many"}, 400, "max_tokens", param="max_tokens")
    c.fail("/v1/completions", {}, 400, "prompt", param="prompt")
    c.fail("/v1/chat/completions", {"messages": []}, 400, "messages", param="messages")
    raw = raw_http(port, b"POST /v1/completions HTTP/1.1\r\nContent-Length: 5\r\nConnection: close\r\n\r\n{nope")
    assert raw.startswith(b"HTTP/1.1 400 ") and b"not JSON" in raw, raw

    # ---------------------------------------------------------------- tools on a model with no chat template: generic format
    TOOLS = [{"type": "function", "function": {"name": "wx", "description": "weather", "parameters": {"type": "object", "properties": {"c": {"enum": ["a", "b"]}}, "required": ["c"]}}},
             {"type": "function", "function": {"name": "tm", "parameters": {"type": "object", "properties": {"u": {"type": "boolean"}}, "required": ["u"]}}}]
    def check_call(tc, names=("wx", "tm")):
        assert tc["type"] == "function" and tc["id"].startswith("call_") and tc["function"]["name"] in names, tc
        args = json.loads(tc["function"]["arguments"])
        assert (args in ({"c": "a"}, {"c": "b"})) if tc["function"]["name"] == "wx" else (args in ({"u": True}, {"u": False})), tc
    r = c.post("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "tool_choice": "required", "max_tokens": 80})["choices"][0]
    assert r["finish_reason"] == "tool_calls" and r["message"]["content"] is None and len(r["message"]["tool_calls"]) == 1, r
    check_call(r["message"]["tool_calls"][0])
    r = c.post("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "tm"}}, "max_tokens": 80})["choices"][0]
    check_call(r["message"]["tool_calls"][0], names=("tm",))
    # auto: plain text stays content; a reply that is a call to a declared function becomes tool_calls
    r = c.post("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "max_tokens": 5})["choices"][0]
    assert "tool_calls" not in r["message"] and r["finish_reason"] == "length" and r["message"]["content"], r
    forced = 'root ::= "{\\"name\\": \\"wx\\", \\"arguments\\": {\\"c\\": \\"b\\"}}"'
    r = c.post("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "tool_choice": "auto", "grammar": forced, "max_tokens": 80})["choices"][0]
    assert r["finish_reason"] == "tool_calls" and json.loads(r["message"]["tool_calls"][0]["function"]["arguments"]) == {"c": "b"}, r
    other = 'root ::= "{\\"name\\": \\"zz\\", \\"arguments\\": {}}"'   # not a declared function: ordinary content
    r = c.post("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "grammar": other, "max_tokens": 80})["choices"][0]
    assert "tool_calls" not in r["message"] and json.loads(r["message"]["content"])["name"] == "zz" and r["finish_reason"] == "stop", r
    ev = c.stream("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "grammar": other, "max_tokens": 80})
    assert json.loads("".join(e["choices"][0]["delta"].get("content", "") for e in ev))["name"] == "zz", "held-back text is released as content"
    r = c.post("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "tool_choice": "none", "grammar": forced, "max_tokens": 80})["choices"][0]
    assert "tool_calls" not in r["message"] and r["message"]["content"].startswith('{"name"'), r
    c.fail("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "nope"}}}, 400, "nope", param="tool_choice")
    c.fail("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "tool_choice": "sometimes"}, 400, "tool_choice", param="tool_choice")
    c.fail("/v1/chat/completions", {"messages": MSG, "tool_choice": "required"}, 400, "tools", param="tool_choice")
    c.fail("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "tool_choice": "required", "grammar": forced}, 400, "cannot be combined", param="tool_choice")
    c.fail("/v1/chat/completions", {"messages": MSG, "tools": [{"type": "function"}]}, 400, "function", )
    assert "tool_format" in c.get("/v1/models")["data"][0] and c.get("/v1/models")["data"][0]["tool_format"] == "generic"

    # ---------------------------------------------------------------- LoRA adapters per request
    models = c.get("/v1/models")["data"]
    assert [m["id"] for m in models] == [NAME, NAME + ":attn", NAME + ":head"] and models[1]["adapter"] == "attn" and models[1]["parent"] == NAME, models
    assert c.get("/v1/models/" + NAME + ":head")["id"] == NAME + ":head"
    P = {"prompt": "on the mat the cat", "max_tokens": 8}
    b0 = c.post("/v1/completions", P)                                         # caches the base prefix of this prompt
    a1 = c.post("/v1/completions", dict(P, model=NAME + ":attn"))             # must not reuse it
    a2 = c.post("/v1/completions", dict(P, adapter="attn"))                   # may reuse a1's
    h1 = c.post("/v1/completions", dict(P, lora="head"))
    assert a1["model"] == NAME + ":attn" and b0["model"] == NAME
    tb, ta, th = b0["choices"][0]["tokens"], a1["choices"][0]["tokens"], h1["choices"][0]["tokens"]
    assert ta == a2["choices"][0]["tokens"] and ta != tb and th != tb and th != ta, (tb, ta, th)
    assert c.post("/v1/completions", P)["choices"][0]["tokens"] == tb, "the base result is unchanged after adapter requests"
    assert c.post("/v1/completions", dict(P, model="gpt-4o"))["choices"][0]["tokens"] == tb, "an unrelated model name serves the base weights"
    c.fail("/v1/completions", dict(P, model=NAME + ":nope"), 404, "nope", code="model_not_found")
    c.fail("/v1/completions", dict(P, adapter="nope"), 404, "nope", code="model_not_found")
    # all three at once through the scheduler
    got = {}
    def run(tag, body): got[tag] = c.post("/v1/completions", body)["choices"][0]["tokens"]
    ts = [threading.Thread(target=run, args=(t, b)) for t, b in (("b", P), ("a", dict(P, adapter="attn")), ("h", dict(P, adapter="head")), ("b2", P), ("a2", dict(P, adapter="attn")))]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert got == {"b": tb, "a": ta, "h": th, "b2": tb, "a2": ta}, got
finally:
    p.kill(); p.wait()
# the adapter result without any cached base prefix (a fresh server, adapter request first) is the same
p, port = start(fpk, "--lora", f"attn={lora_attn}")
try:
    assert Client(port).post("/v1/completions", dict(P, adapter="attn"))["choices"][0]["tokens"] == ta, "an adapter request reused KV rows computed with the base weights"
finally:
    p.kill(); p.wait()
r = subprocess.run([serve, fpk, "--port", "0", "--lora", "broken"], capture_output=True, text=True)
assert r.returncode == 1 and "name=file" in r.stderr, r.stderr

# ================================================================== 2. the chat model: end of turn, hermes tool calls
p, port = start(chat_fpk, "--max-active", "4")
try:
    c = Client(port)
    info = c.get("/v1/models")["data"][0]
    assert info["chat_template"] is True and info["tool_format"] == "hermes" and info["context_length"] == 512, info
    # the model "says" <|im_end|> (forced with logit_bias): the turn ends there, the marker is not part of the reply
    r = c.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 8, "logit_bias": {str(IM_END): 100}})
    assert r["choices"][0]["finish_reason"] == "stop" and r["choices"][0]["message"]["content"] == "" and r["usage"]["completion_tokens"] == 0, r
    # <|eot_id|> is not in generation_config.json; it ends chat turns because the chat template writes it
    r = c.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 8, "logit_bias": {str(EOT): 100}})
    assert r["choices"][0]["finish_reason"] == "stop" and r["usage"]["completion_tokens"] == 0, r
    # raw completions stop at the manifest's ids only
    r = c.post("/v1/completions", {"prompt": "hi", "max_tokens": 4, "logit_bias": {str(IM_END): 100}})["choices"][0]
    assert r["finish_reason"] == "stop" and r["tokens"] == [], r
    r = c.post("/v1/completions", {"prompt": "hi", "max_tokens": 4, "logit_bias": {str(EOT): 100}})["choices"][0]
    assert r["finish_reason"] == "length" and r["tokens"] == [EOT] * 4 and r["text"] == "<|eot_id|>" * 4, r
    plain = c.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 6})
    assert plain["choices"][0]["finish_reason"] == "length" and plain["usage"]["completion_tokens"] == 6
    # the prompt is the rendered template
    rendered = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    t = c.post("/v1/completions", {"prompt": rendered, "add_bos": False, "max_tokens": 6})
    assert t["usage"]["prompt_tokens"] == plain["usage"]["prompt_tokens"] and t["choices"][0]["text"] == plain["choices"][0]["message"]["content"], (t, plain)
    # `forge run --chat` stops at the end-of-turn token too
    def ids(*a): return subprocess.run([cli, "run", chat_fpk, *a, "--ids"], capture_output=True, text=True, check=True).stdout.split()
    assert ids("--chat", "--prompt", "hi", "-n", "6") == [str(x) for x in t["choices"][0]["tokens"]]

    # tool_choice required: the grammar writes the model's own syntax, the reply comes back as tool_calls
    r = c.post("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "tool_choice": "required", "max_tokens": 120})
    ch = r["choices"][0]
    assert ch["finish_reason"] == "tool_calls" and ch["message"]["content"] is None and len(ch["message"]["tool_calls"]) == 1, r
    check_call(ch["message"]["tool_calls"][0])
    assert r["usage"]["prompt_tokens"] > plain["usage"]["prompt_tokens"], "the template was given the tools"
    r = c.post("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "wx"}}, "max_tokens": 120})["choices"][0]
    check_call(r["message"]["tool_calls"][0], names=("wx",))
    # required, streamed: a header delta (id, name), argument deltas that add up to the JSON, finish_reason tool_calls
    ev = c.stream("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "tool_choice": "required", "max_tokens": 120})
    deltas = [tc for e in ev for chx in e["choices"] for tc in chx["delta"].get("tool_calls", [])]
    assert deltas[0]["index"] == 0 and deltas[0]["id"].startswith("call_") and deltas[0]["type"] == "function" and deltas[0]["function"]["arguments"] == "", deltas[0]
    assert all("id" not in d and "name" not in d["function"] for d in deltas[1:]) and len(deltas) > 2, deltas
    check_call({"type": "function", "id": deltas[0]["id"], "function": {"name": deltas[0]["function"]["name"], "arguments": "".join(d["function"]["arguments"] for d in deltas)}})
    assert ev[-1]["choices"][0]["finish_reason"] == "tool_calls" and not any(e["choices"][0]["delta"].get("content") for e in ev), ev
    # auto: text, then a call in the model's syntax (written here by a grammar); two calls; an unfinished one is text
    def hermes(*calls, lead=""):
        body = "".join('"<tool_call>\\n{\\"name\\": \\"%s\\", \\"arguments\\": %s}\\n</tool_call>" ' % (n, a.replace('"', '\\"')) for n, a in calls)
        return 'root ::= %s%s' % ('"%s" ' % lead if lead else "", body)
    r = c.post("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "grammar": hermes(("wx", '{"c": "a"}'), lead="ok "), "max_tokens": 120})["choices"][0]
    assert r["finish_reason"] == "tool_calls" and r["message"]["content"] == "ok" and json.loads(r["message"]["tool_calls"][0]["function"]["arguments"]) == {"c": "a"}, r
    r = c.post("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "grammar": hermes(("wx", '{"c": "a"}'), ("tm", '{"u": true}')), "max_tokens": 160})["choices"][0]
    assert [t["function"]["name"] for t in r["message"]["tool_calls"]] == ["wx", "tm"] and len({t["id"] for t in r["message"]["tool_calls"]}) == 2, r
    ev = c.stream("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "grammar": hermes(("wx", '{"c": "a"}'), ("tm", '{"u": true}'), lead="ok "), "max_tokens": 160})
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in ev).strip() == "ok"
    deltas = [tc for e in ev for tc in e["choices"][0]["delta"].get("tool_calls", [])]
    for i, (name, args) in enumerate((("wx", {"c": "a"}), ("tm", {"u": True}))):
        mine = [d for d in deltas if d["index"] == i]
        assert mine[0]["function"]["name"] == name and json.loads("".join(d["function"]["arguments"] for d in mine)) == args, mine
    r = c.post("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "grammar": hermes(("wx", '{"c": "a"}')), "max_tokens": 12})["choices"][0]
    assert "tool_calls" not in r["message"] and r["finish_reason"] == "length" and r["message"]["content"].startswith("<tool_call>"), r
    r = c.post("/v1/chat/completions", {"messages": MSG, "tools": TOOLS, "tool_choice": "none", "grammar": hermes(("wx", '{"c": "a"}')), "max_tokens": 120})["choices"][0]
    assert "tool_calls" not in r["message"] and "<tool_call>" in r["message"]["content"], r
    # a conversation that carries an earlier call (arguments as the JSON string clients send) and its result
    conv = MSG + [{"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "wx", "arguments": "{\"c\": \"a\"}"}}]},
                  {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}]
    r = c.post("/v1/chat/completions", {"messages": conv, "tools": TOOLS, "max_tokens": 3})
    assert r["usage"]["completion_tokens"] == 3 and r["usage"]["prompt_tokens"] > plain["usage"]["prompt_tokens"] + 10, r["usage"]
    assert "forge_tool_calls_total" in c.get("/metrics", raw=True)
finally:
    p.kill(); p.wait()

# ================================================================== 3. HTTP
p, port = start(fpk, "--max-header-bytes", "2048", "--max-body-bytes", "4096", "--request-timeout", "2", "--keep-alive-timeout", "3")
try:
    c = Client(port)
    body = json.dumps({"prompt": "the", "max_tokens": 4}).encode()
    want = c.post("/v1/completions", {"prompt": "the", "max_tokens": 4})["choices"][0]["tokens"]
    m0 = c.get("/metrics", raw=True)
    conns = lambda m: int(float(next(l.split()[1] for l in m.split("\n") if l.startswith("forge_http_connections_total "))))
    # keep-alive: several requests (JSON, an error, a stream) on one connection
    h = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    for _ in range(3):
        h.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
        r = h.getresponse(); assert r.status == 200 and r.getheader("Connection") == "keep-alive", r.getheaders()
        assert json.loads(r.read())["choices"][0]["tokens"] == want
    h.request("GET", "/nope"); r = h.getresponse(); assert r.status == 404 and json.loads(r.read())["error"]["type"] == "not_found_error"
    h.request("POST", "/v1/completions", json.dumps({"prompt": "the", "max_tokens": 4, "stream": True}).encode())
    r = h.getresponse(); assert r.status == 200 and r.getheader("Transfer-Encoding") == "chunked" and r.read().decode().rstrip().endswith("data: [DONE]")
    h.request("GET", "/health"); r = h.getresponse(); assert r.status == 200 and json.loads(r.read())["status"] == "ok"
    h.close()
    assert conns(c.get("/metrics", raw=True)) == conns(m0) + 2, "six requests shared one connection (+1 for this metrics request)"
    # pipelining: two requests in one write, answered in order
    rs = split_responses(raw_http(port, b"GET /health HTTP/1.1\r\nHost: x\r\n\r\nPOST /v1/completions HTTP/1.1\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s" % (len(body), body)))
    assert [s for s, _, _ in rs] == [200, 200] and json.loads(rs[0][2])["status"] == "ok" and json.loads(rs[1][2])["choices"][0]["tokens"] == want, rs
    assert rs[0][1]["connection"] == "keep-alive" and rs[1][1]["connection"] == "close"
    # chunked request body (chunk extension, trailer), bytes arriving in pieces
    chunked = b"POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n" + \
              b"5;ext=1\r\n" + body[:5] + b"\r\n" + (b"%x\r\n" % (len(body) - 5)) + body[5:] + b"\r\n0\r\nX-Trailer: 1\r\n\r\n"
    for parts in ([chunked], [chunked[:60], chunked[60:95], chunked[95:]], [bytes([b]) for b in chunked]):
        rs = split_responses(raw_http(port, None, parts=parts, pause=0.01 if len(parts) < 10 else 0))
        assert rs[0][0] == 200 and json.loads(rs[0][2])["choices"][0]["tokens"] == want, rs
    bad = lambda req: split_responses(raw_http(port, req))[0]
    for req, status in ((b"POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n", 400),
                        (b"POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nabcdeXX", 400),
                        (b"POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\nContent-Length: 3\r\n\r\n0\r\n\r\n", 400),
                        (b"POST /v1/completions HTTP/1.1\r\nContent-Length: 3\r\nContent-Length: 4\r\n\r\nabcd", 400),
                        (b"POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: gzip\r\n\r\n", 501),
                        (b"POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n1001\r\n", 413),
                        (b"POST /v1/completions HTTP/1.1\r\nContent-Length: 4097\r\n\r\n", 413),
                        (b"GET /health HTTP/1.1\r\nX-Pad: " + b"a" * 3000 + b"\r\n\r\n", 431),
                        (b"GET /health HTTP/2.0\r\n\r\n", 505),
                        (b"GET /health HTTP/1.1\r\nBad Header: x\r\n\r\n", 400),
                        (b"GET /health HTTP/1.1\r\nExpect: 200-ok\r\n\r\n", 417),
                        (b"nonsense\r\n\r\n", 400)):
        s, hd, b = bad(req)
        assert s == status and hd["connection"] == "close" and json.loads(b)["error"]["message"], (req[:60], s, b)
    # Expect: 100-continue: the interim response comes before the body is sent
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sk:
        sk.sendall(b"POST /v1/completions HTTP/1.1\r\nContent-Length: %d\r\nExpect: 100-continue\r\nConnection: close\r\n\r\n" % len(body))
        assert sk.recv(4096) == b"HTTP/1.1 100 Continue\r\n\r\n"
        sk.sendall(body)
        out = b""
        while True:
            b = sk.recv(65536)
            if not b: break
            out += b
        assert json.loads(split_responses(out)[0][2])["choices"][0]["tokens"] == want
    # time limits: a request that stalls gets 408 after --request-timeout; an idle kept connection is closed quietly
    t0 = time.time(); raw = raw_http(port, b"POST /v1/completions HTTP/1.1\r\nContent-Length: 50\r\n\r\n{", wait=10)
    assert raw.startswith(b"HTTP/1.1 408 ") and 1.5 < time.time() - t0 < 8, (raw, time.time() - t0)
    t0 = time.time(); raw = raw_http(port, b"GET /health HTTP/1.1\r\n\r\n", wait=10)
    assert len(split_responses(raw)) == 1 and 2 < time.time() - t0 < 9, (raw, time.time() - t0)
    # HTTP/1.0: no keep-alive unless asked, streams end with the connection
    s, hd, b = bad(b"GET /health HTTP/1.0\r\n\r\n"); assert s == 200 and hd["connection"] == "close"
    raw = raw_http(port, b"POST /v1/completions HTTP/1.0\r\nContent-Length: %d\r\n\r\n%s" % (len(body) + 16, body[:-1] + b', "stream": true}'))
    assert b"Transfer-Encoding" not in raw and raw.rstrip().endswith(b"data: [DONE]"), raw
    # CORS (default: any origin), preflight, methods
    r = urllib.request.urlopen(c.base + "/health"); assert r.headers["Access-Control-Allow-Origin"] == "*"
    s, hd, b = bad(b"OPTIONS /v1/chat/completions HTTP/1.1\r\nOrigin: http://x\r\nConnection: close\r\n\r\n")
    assert s == 204 and hd["access-control-allow-origin"] == "*" and "Authorization" in hd["access-control-allow-headers"], hd
    err, hd = c.fail("/v1/completions", None, 405, "POST", method="GET"); assert "POST" in hd["Allow"]
    err, hd = c.fail("/v1/models/zzz", None, 404, "zzz", code="model_not_found", method="GET")
    try: c.post("/health", {}); raise SystemExit("expected 405")
    except urllib.error.HTTPError as e: assert e.code == 405 and "GET" in e.headers["Allow"]
    # /health, /v1/models, /metrics
    hl = c.get("/health"); assert hl == {"status": "ok", "model": NAME, "kind": "generative", "queue_waiting": 0, "queue_active": 0}, hl
    assert c.get("/healthz")["status"] == "ok"
    md = c.get("/v1/models"); one = md["data"][0]
    assert md["object"] == "list" and one["id"] == NAME and one["object"] == "model" and one["owned_by"] == "forge" and one["context_length"] == 128 and one["vocab_size"] == 300 \
        and one["kind"] == "generative" and one["chat_template"] is False and isinstance(one["created"], (int, float)), one
    assert c.get("/v1/models/" + NAME) == one
    r = urllib.request.urlopen(c.base + "/metrics"); m = r.read().decode()
    assert r.headers["Content-Type"].startswith("text/plain")
    series = [l.split()[0].split("{")[0] for l in m.split("\n") if l and not l.startswith("#")]
    typed = {l.split()[2] for l in m.split("\n") if l.startswith("# TYPE ")}
    for name in ("forge_steps", "forge_completed", "forge_failed_total", "forge_queue_waiting", "forge_http_connections_total", "forge_http_connections_active",
                 "forge_http_request_errors_total", "forge_prompt_tokens_total", "forge_completion_tokens_total", "forge_auth_failures_total", "forge_draining"):
        assert name in series and name in typed, name
    assert 'forge_http_responses_total{code="200"}' in m and 'forge_http_responses_total{code="413"} 2' in m and 'forge_http_responses_total{code="408"} 1' in m, m
    tok_total = lambda m, n: int(float(next(l.split()[1] for l in m.split("\n") if l.startswith(n + " "))))
    assert tok_total(m, "forge_completion_tokens_total") >= 4 * 9
finally:
    p.kill(); p.wait()

p, port = start(fpk, "--cors", "https://app.example")
try:
    r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
    assert r.headers["Access-Control-Allow-Origin"] == "https://app.example" and r.headers["Vary"] == "Origin"
finally:
    p.kill(); p.wait()
p, port = start(fpk, "--cors", "off", "--max-connections", "2")
try:
    r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health"); assert r.headers["Access-Control-Allow-Origin"] is None
    r.read(); r.close()
    # the connection limit: two idle connections fill it, the third is turned away with 503. The server may not
    # have noticed the connection above close yet (it still counts against the limit), so retry until both fit.
    for attempt in range(50):
        held = [socket.create_connection(("127.0.0.1", port), timeout=10) for _ in range(2)]
        replies = []
        for sk in held: sk.sendall(b"GET /health HTTP/1.1\r\n\r\n"); replies.append(sk.recv(4096))
        if all(x.startswith(b"HTTP/1.1 200") for x in replies): break
        for sk in held: sk.close()
        time.sleep(0.1)
    else:
        raise AssertionError(replies)
    assert raw_http(port, b"GET /health HTTP/1.1\r\n\r\n").startswith(b"HTTP/1.1 503 ")
    for sk in held: sk.close()
    for _ in range(100):
        try:
            if json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read())["status"] == "ok": break
        except Exception: time.sleep(0.05)
    else: raise SystemExit("the server did not free the closed connections")
finally:
    p.kill(); p.wait()

# graceful shutdown: SIGTERM while a stream runs; the stream completes, idle connections close, the exit code is 0
if os.name == "posix":
    p, port = start(fpk, "--shutdown-timeout", "20")
    idle = socket.create_connection(("127.0.0.1", port), timeout=20)
    idle.sendall(b"GET /health HTTP/1.1\r\n\r\n"); assert idle.recv(4096).startswith(b"HTTP/1.1 200")
    sk = socket.create_connection(("127.0.0.1", port), timeout=30)
    body = json.dumps({"prompt": "the", "max_tokens": 120, "stream": True}).encode()
    sk.sendall(b"POST /v1/completions HTTP/1.1\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
    first = sk.recv(256); assert first.startswith(b"HTTP/1.1 200")
    p.send_signal(signal.SIGTERM)
    out = first
    while True:
        b = sk.recv(65536)
        if not b: break
        out += b
    assert out.count(b"data: {") >= 100 and b"data: [DONE]" in out, (len(out), out[-200:])
    assert idle.recv(4096) == b"", "the idle kept-alive connection is closed during the drain"
    assert p.wait(timeout=30) == 0, p.stderr.read()
    assert "shutting down" in p.stdout.read()
    try: socket.create_connection(("127.0.0.1", port), timeout=2); raise SystemExit("still listening after shutdown")
    except OSError: pass
print("serve api ok")

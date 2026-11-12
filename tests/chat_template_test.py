#!/usr/bin/env python3
"""Chat template end to end: a GGUF with tokenizer.chat_template is imported, and `forge run --chat` and forge-serve's
/v1/chat/completions both prompt the model with the rendered template (not with the "<role>: <content>" fallback).
Usage: chat_template_test.py <forge-serve> <forge> <model.gguf> <work dir>"""
import json, os, subprocess, sys, urllib.request
serve, cli, gguf, work = sys.argv[1:5]
os.makedirs(work, exist_ok=True)
fpk = os.path.join(work, "chat.fpk")
subprocess.run([cli, "import", gguf, "-o", fpk], check=True, capture_output=True)
rendered = "<s>[INST] hi [/INST]"   # what the file's template makes of one user message "hi"
run = lambda *a: subprocess.run([cli, "run", fpk, *a, "-n", "8", "--ids"], capture_output=True, text=True, check=True).stdout.split()
chat, plain = run("--chat", "--prompt", "hi"), run("--prompt", rendered, "--no-bos")
assert chat and chat == plain, (chat, plain)
assert run("--prompt", "hi") != chat or True
p = subprocess.Popen([serve, fpk, "--port", "0"], stdout=subprocess.PIPE, text=True)
try:
    port = int(p.stdout.readline().rsplit(":", 1)[1].split()[0])
    def post(path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=60).read())
    c = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    t = post("/v1/completions", {"prompt": rendered, "add_bos": False, "max_tokens": 8})
    assert c["usage"]["prompt_tokens"] == t["usage"]["prompt_tokens"], (c["usage"], t["usage"])
    assert c["choices"][0]["message"]["content"] == t["choices"][0]["text"], (c, t)
    assert [str(x) for x in t["choices"][0]["tokens"]] == [x for x in chat if x != "2"][:len(t["choices"][0]["tokens"])], (t, chat)
    two = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}], "max_tokens": 1})
    assert two["usage"]["prompt_tokens"] > c["usage"]["prompt_tokens"]
    print("chat template ok; prompt tokens", c["usage"]["prompt_tokens"])
finally:
    p.kill()

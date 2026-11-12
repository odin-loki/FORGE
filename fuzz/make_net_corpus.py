#!/usr/bin/env python3
"""Regenerate the seed corpus of the network-input fuzz targets (serve_http, proxy_http, rpc, json_schema).
Standard library only. Usage: make_net_corpus.py   (writes fuzz/corpus/<target>/*; regress_* files are kept)

The policy and signature seeds come from fuzz/make_sig_corpus.cpp (see docs/testing.md, "Fuzzing")."""
import json, os, struct
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")

def put(target, name, data):
    d = os.path.join(here, target); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "wb") as f: f.write(data if isinstance(data, bytes) else data.encode())

# ---- HTTP: byte 0 is the recv() chunk size selector of the harness (200+ = everything at once)
body = json.dumps({"prompt": "hi", "max_tokens": 4, "stream": False})
reqs = {
    "get_health": "GET /health HTTP/1.1\r\nHost: x\r\n\r\n",
    "get_models_query": "GET /v1/models?x=1 HTTP/1.1\r\nHost: x\r\nAccept: */*\r\n\r\n",
    "post_completion": f"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n{body}",
    "options": "OPTIONS /v1/chat/completions HTTP/1.1\r\nOrigin: http://a\r\n\r\n",
    "post_short_body": "POST /v1/completions HTTP/1.1\r\ncontent-length: 50\r\n\r\n{\"prompt\":",
    "no_colon_header": "GET / HTTP/1.1\r\nnocolon\r\nA:  b\r\n\r\ntrailing",
}
for n, r in reqs.items():
    put("serve_http", n, b"\xc8" + r.encode()); put("serve_http", n + "_bytewise", b"\x00" + r.encode())
    put("proxy_http", n, b"\xc8" + r.encode())
chunked = "POST /v1/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\nX-Session-Id: s1\r\n\r\n5\r\nhello\r\n10\r\n0123456789abcdef\r\n0\r\n\r\n"
put("proxy_http", "post_chunked", b"\xc8" + chunked.encode())
put("proxy_http", "post_chunked_split", b"\x02" + chunked.encode())
put("proxy_http", "keepalive_pipeline", b"\xc8" + (reqs["get_health"] + reqs["post_completion"] + "GET /x HTTP/1.0\r\n\r\n").encode())
put("proxy_http", "connection_close", b"\x07GET / HTTP/1.1\r\nConnection: close\r\n\r\nGET /second HTTP/1.1\r\n\r\n")
put("proxy_http", "upstream_response", b"\xc8HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\nok")
put("proxy_http", "upstream_chunked", b"\x03HTTP/1.1 503 Service Unavailable\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n")

# ---- RPC: byte 0 % 7 picks the decoder (see fuzz_rpc.cpp)
u32 = lambda v: struct.pack("<I", v)
i64 = lambda v: struct.pack("<q", v)
s = lambda t: u32(len(t)) + (t if isinstance(t, bytes) else t.encode())
shape = lambda *d: u32(len(d)) + b"".join(i64(x) for x in d)
f32 = lambda *v: struct.pack("<%df" % len(v), *v)
def frpc(op, attrs, ins, out):
    b = b"FRPC" + s(op) + u32(len(attrs))
    for k, v in attrs:
        if isinstance(v, float): b += s(k) + u32(0) + struct.pack("<d", v)
        elif isinstance(v, list): b += s(k) + u32(1) + u32(len(v)) + b"".join(i64(x) for x in v)
        else: b += s(k) + u32(2) + s(v)
    b += u32(len(ins))
    for sh, data in ins: b += shape(*sh) + f32(*data)
    return b + shape(*out)
frames = {
    "add": frpc("add", [], [((2, 2), (1, 2, 3, 4)), ((2, 2), (5, 6, 7, 8))], (2, 2)),
    "matmul_t": frpc("matmul_t", [], [((1, 3), (1, 2, 3)), ((2, 3), (1, 0, 0, 0, 1, 0))], (1, 2)),
    "softmax_axis": frpc("softmax", [("axis", -1.0)], [((2, 3), (1, 2, 3, 4, 5, 6))], (2, 3)),
    "reshape": frpc("reshape", [("shape", [3, 2])], [((2, 3), (1, 2, 3, 4, 5, 6))], (3, 2)),
    "transpose_perm": frpc("transpose", [("perm", [1, 0])], [((2, 3), (1, 2, 3, 4, 5, 6))], (3, 2)),
    "gather": frpc("gather", [], [((3, 2), (1, 2, 3, 4, 5, 6)), ((2,), (0, 2))], (2, 2)),
    "attention": frpc("attention", [("causal", 1.0)], [((2, 1, 2), (1, 2, 3, 4)), ((2, 1, 2), (1, 0, 0, 1)), ((2, 1, 2), (5, 6, 7, 8))], (2, 1, 2)),
    "unknown_op": frpc("nope", [("k", "v")], [((1,), (1,))], (1,)),
    "two_frames": frpc("neg", [], [((2,), (1, -1))], (2,)) + frpc("exp", [], [((1,), (0,))], (1,)),
}
script = b"FRPS" + s("std::println(1);") + s("a.fs") + u32(1) + s("arg") + u32(1000)
signed = b"FRPZ" + s("std::println(1);") + s("a.fs") + u32(0) + u32(0) + s('{"format":"forge-detached-sig"}')
for n, f in frames.items():
    put("rpc", "plain_" + n, b"\x04" + f); put("rpc", "authed_" + n, b"\x05" + f)
put("rpc", "authed_script", b"\x05" + script); put("rpc", "authed_signed_script", b"\x05" + signed)
put("rpc", "authed_script_then_op", b"\x05" + script + frames["add"])
put("rpc", "plain_script_refused", b"\x04" + script)
key = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
put("rpc", "hello_token", b"\x06FRPH" + u32(1) + s("tok") + frames["add"])
put("rpc", "hello_bad_token", b"\x06FRPH" + u32(1) + s("nope"))
put("rpc", "hello_key", b"\x06FRPH" + u32(2) + bytes.fromhex(key) + bytes(64))
put("rpc", "hello_bound_token_no_tls", b"\x06FRPH" + u32(3) + bytes(64))
put("rpc", "hello_unknown_method", b"\x06FRPH" + u32(9))
put("rpc", "no_hello", b"\x06" + frames["add"])
put("rpc", "authorized_keys", b"\x00# keys\nallow=device " + key.encode() + b" laptop\n" + key.upper().encode() + b"\nallow=device,scripts " + key.encode() + b"\n")
put("rpc", "tokens", b"\x01s3cret\nallow=scripts abc= name with blanks\n# c\n\nallow=all t\n")
put("rpc", "client_map", b"\x02allow=device sha256:" + ":".join(key[i:i + 2] for i in range(0, 64, 2)).encode() + b" a\ncn:build-host b\n")
put("rpc", "perms", b"\x03device,scripts,all")

# ---- JSON Schema -> GBNF
schemas = {
    "object": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3}}, "required": ["name"]},
    "enum_const": {"anyOf": [{"enum": ["a", 1, None, True, {"k": [1.5]}]}, {"const": "x\"y\\zé"}]},
    "refs": {"$defs": {"node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/node"}, "v": {"type": ["number", "null"]}}}}, "$ref": "#/$defs/node"},
    "definitions": {"definitions": {"a": {"oneOf": [{"type": "boolean"}, {"type": "null"}]}}, "type": "array", "items": {"$ref": "#/definitions/a"}, "maxItems": 0},
    "any": {},
    "true": True,
    "array_bounds": {"type": "array", "minItems": 2, "maxItems": 4, "items": {"type": "array", "minItems": 0, "maxItems": 2}},
    "nested": {"type": "object", "properties": {"a": {"type": "object", "properties": {"b": {"type": "object"}}, "required": ["b"]}, "c": {}}},
}
for n, v in schemas.items(): put("json_schema", n + ".json", json.dumps(v))
for t in ("serve_http", "proxy_http", "rpc", "json_schema"): print(t, len(os.listdir(os.path.join(here, t))))

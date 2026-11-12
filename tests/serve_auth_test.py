#!/usr/bin/env python3
"""forge-serve authentication and transport security:

  API keys   --api-key / --api-key-file: 401 without or with a wrong key, /health and CORS preflight stay open,
             per-key requests-per-minute limits (429 + Retry-After) and token quotas (429 insufficient_quota)
  TLS        --tls-cert / --tls-key (HTTPS), --tls-client-ca (mutual TLS). Certificates are made with the `openssl`
             command; without it, or when forge-serve was built without OpenSSL (FORGE_TLS=OFF), that part is skipped
             and the plain-HTTP-only behaviour is checked instead (the TLS flags are refused with a clear message).

  serve_auth_test.py <forge-serve> <golden tiny_f32.fpk> <work dir>
"""
import http.client, json, os, shutil, socket, ssl, subprocess, sys, urllib.error, urllib.request
serve, fpk, work = sys.argv[1:4]
os.makedirs(work, exist_ok=True)

def start(*flags):
    p = subprocess.Popen([serve, fpk, "--port", "0", *flags], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = p.stdout.readline()
    assert "forge-serve:" in line, (line, p.stderr.read() if p.poll() is not None else "")
    return p, int(line.split("://127.0.0.1:")[1].split()[0]), line

def call(port, path, body=None, key=None, ctx=None, scheme="http"):
    """(status, json, headers)"""
    hd = {"Content-Type": "application/json"}
    if key is not None: hd["Authorization"] = "Bearer " + key
    req = urllib.request.Request(f"{scheme}://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(), headers=hd)
    handlers = [urllib.request.ProxyHandler({})] + ([urllib.request.HTTPSHandler(context=ctx)] if ctx else [])   # never through a proxy
    try:
        r = urllib.request.build_opener(*handlers).open(req, timeout=60)
        raw = r.read()
        return r.status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode()), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), e.headers

BODY = {"prompt": "the", "max_tokens": 4}

# ------------------------------------------------------------------ API keys
keyfile = os.path.join(work, "keys.txt")
with open(keyfile, "w") as f:
    f.write("# name and limits per key\n"
            "sk-alpha-0123456789 alice\n"
            "rpm=2 sk-slow-0123456789 bob\n"
            "quota=12 sk-quota-0123456789 carol\n")
os.chmod(keyfile, 0o600)
p, port, _ = start("--api-key", "sk-cli-0123456789", "--api-key-file", keyfile)
try:
    s, j, h = call(port, "/v1/completions", BODY)
    assert s == 401 and j["error"]["code"] == "invalid_api_key" and j["error"]["type"] == "authentication_error" and h["WWW-Authenticate"] == "Bearer", (s, j)
    for wrong in ("sk-alpha-012345678", "sk-alpha-0123456789x", "", "x" * 5000):
        s, j, _ = call(port, "/v1/completions", BODY, key=wrong); assert s == 401, (wrong[:20], s, j)
    for path in ("/v1/models", "/metrics"):
        assert call(port, path)[0] == 401 and call(port, path, key="sk-alpha-0123456789")[0] == 200, path
    assert call(port, "/health")[0] == 200, "liveness needs no key"
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sk:   # a browser's preflight carries no credentials
        sk.sendall(b"OPTIONS /v1/completions HTTP/1.1\r\nConnection: close\r\n\r\n"); assert sk.recv(4096).startswith(b"HTTP/1.1 204 ")
    for key in ("sk-alpha-0123456789", "sk-cli-0123456789"):
        s, j, _ = call(port, "/v1/completions", BODY, key=key); assert s == 200 and len(j["choices"][0]["tokens"]) == 4, (s, j)
    # the scheme word is case-insensitive
    h = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    h.request("POST", "/v1/completions", json.dumps(BODY), {"Authorization": "bearer sk-alpha-0123456789"}); assert h.getresponse().status == 200
    h.close()
    # rpm=2: the third request inside the minute is refused with Retry-After, other keys are not affected
    assert [call(port, "/v1/completions", BODY, key="sk-slow-0123456789")[0] for _ in range(2)] == [200, 200]
    s, j, h = call(port, "/v1/completions", BODY, key="sk-slow-0123456789")
    assert s == 429 and j["error"]["code"] == "rate_limit_exceeded" and j["error"]["type"] == "rate_limit_error" and int(h["Retry-After"]) >= 1 and "bob" in j["error"]["message"], (s, j)
    assert call(port, "/v1/models", key="sk-slow-0123456789")[0] == 200, "only generation requests are counted"
    assert call(port, "/v1/completions", BODY, key="sk-alpha-0123456789")[0] == 200
    # quota=12 tokens: each request spends 2 prompt + 4 completion tokens; the third finds the quota used up
    assert [call(port, "/v1/completions", BODY, key="sk-quota-0123456789")[0] for _ in range(2)] == [200, 200]
    s, j, _ = call(port, "/v1/completions", BODY, key="sk-quota-0123456789")
    assert s == 429 and j["error"]["code"] == "insufficient_quota" and "carol" in j["error"]["message"], (s, j)
    m = call(port, "/metrics", key="sk-alpha-0123456789")[1]
    val = lambda n: int(float(next(l.split()[1] for l in m.split("\n") if l.startswith(n + " "))))
    assert val("forge_auth_failures_total") == 7 and val("forge_rate_limited_total") == 2, m
    assert 'forge_http_responses_total{code="401"} 7' in m and 'forge_http_responses_total{code="429"} 2' in m, m
finally:
    p.kill(); p.wait()
# --rate-limit without keys: one bucket for everybody
p, port, _ = start("--rate-limit", "1")
try:
    assert call(port, "/v1/completions", BODY)[0] == 200 and call(port, "/v1/completions", BODY)[0] == 429
finally:
    p.kill(); p.wait()
for flags, needle in ((["--api-key", "short"], "8 characters"), (["--api-key-file", os.path.join(work, "missing.txt")], "api-key-file")):
    r = subprocess.run([serve, fpk, "--port", "0", *flags], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and needle in r.stderr, (flags, r.stderr)
print("serve auth: API keys ok")

# ------------------------------------------------------------------ TLS
def openssl(*a): subprocess.run(["openssl", *a], check=True, capture_output=True)
have_openssl = shutil.which("openssl") is not None
probe = subprocess.run([serve, fpk, "--port", "0", "--tls-cert", os.path.join(work, "none.pem"), "--tls-key", os.path.join(work, "none.key")], capture_output=True, text=True, timeout=60)
assert probe.returncode == 1, probe
if "built without TLS" in probe.stderr:
    # FORGE_TLS=OFF (or no OpenSSL at build time): plain HTTP only, and the flags say so instead of being ignored
    p, port, line = start()
    try: assert "http://" in line and call(port, "/health")[0] == 200
    finally: p.kill(); p.wait()
    print("serve auth: built without TLS, the TLS flags are refused; HTTPS part skipped")
    sys.exit(0)
assert "certificate" in probe.stderr, probe.stderr
r = subprocess.run([serve, fpk, "--port", "0", "--tls-cert", "x.pem"], capture_output=True, text=True, timeout=60)
assert r.returncode == 1 and "--tls-cert and --tls-key" in r.stderr, r.stderr
if not have_openssl:
    print("serve auth: no openssl command to make certificates; HTTPS part skipped")
    sys.exit(0)
d = lambda n: os.path.join(work, n)
ext = d("san.cnf"); open(ext, "w").write("subjectAltName=IP:127.0.0.1,DNS:localhost\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid\n")
openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", d("ca.key"), "-out", d("ca.pem"), "-days", "2", "-subj", "/CN=forge test CA",
        "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign")
for name in ("server", "client", "stranger"):
    openssl("req", "-newkey", "rsa:2048", "-nodes", "-keyout", d(name + ".key"), "-out", d(name + ".csr"), "-subj", f"/CN={name}")
    if name == "stranger": openssl("x509", "-req", "-in", d(name + ".csr"), "-signkey", d(name + ".key"), "-out", d(name + ".pem"), "-days", "2")
    else: openssl("x509", "-req", "-in", d(name + ".csr"), "-CA", d("ca.pem"), "-CAkey", d("ca.key"), "-CAcreateserial", "-out", d(name + ".pem"), "-days", "2", "-extfile", ext)
    os.chmod(d(name + ".key"), 0o600)

def client_ctx(cert=None):
    ctx = ssl.create_default_context(cafile=d("ca.pem"))
    if cert: ctx.load_cert_chain(d(cert + ".pem"), d(cert + ".key"))
    return ctx

p, port, line = start("--tls-cert", d("server.pem"), "--tls-key", d("server.key"), "--api-key", "sk-tls-0123456789")
try:
    assert "https://127.0.0.1" in line, line
    ctx = client_ctx()
    assert call(port, "/health", ctx=ctx, scheme="https")[1]["status"] == "ok"
    assert call(port, "/v1/completions", BODY, ctx=ctx, scheme="https")[0] == 401
    s, j, _ = call(port, "/v1/completions", BODY, key="sk-tls-0123456789", ctx=ctx, scheme="https"); assert s == 200 and len(j["choices"][0]["tokens"]) == 4
    # keep-alive and a stream over one TLS connection
    h = http.client.HTTPSConnection("127.0.0.1", port, timeout=60, context=ctx)
    for body in (BODY, dict(BODY, stream=True), BODY):
        h.request("POST", "/v1/completions", json.dumps(body), {"Authorization": "Bearer sk-tls-0123456789"})
        r = h.getresponse(); raw = r.read().decode()
        assert r.status == 200 and (raw.rstrip().endswith("data: [DONE]") if body.get("stream") else json.loads(raw)["usage"]["completion_tokens"] == 4), raw
    h.close()
    assert ssl.create_default_context(cafile=d("ca.pem")).wrap_socket(socket.create_connection(("127.0.0.1", port), timeout=10), server_hostname="127.0.0.1").version() in ("TLSv1.2", "TLSv1.3")
    # plain HTTP on the TLS port gets no HTTP answer; a client that does not trust the certificate refuses it
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sk:
        sk.sendall(b"GET /health HTTP/1.1\r\n\r\n"); sk.settimeout(15)
        try: got = sk.recv(4096)
        except OSError: got = b""
        assert not got.startswith(b"HTTP/"), got
    try: call(port, "/health", ctx=ssl.create_default_context(), scheme="https"); raise SystemExit("an untrusted certificate was accepted")
    except urllib.error.URLError as e: assert "CERTIFICATE_VERIFY_FAILED" in str(e), e
    assert call(port, "/health", ctx=ctx, scheme="https")[0] == 200, "the server survives failed handshakes"
finally:
    p.kill(); p.wait()

# mutual TLS: a client certificate signed by --tls-client-ca is required
p, port, _ = start("--tls-cert", d("server.pem"), "--tls-key", d("server.key"), "--tls-client-ca", d("ca.pem"))
try:
    assert call(port, "/health", ctx=client_ctx("client"), scheme="https")[1]["status"] == "ok"
    s, j, _ = call(port, "/v1/completions", BODY, ctx=client_ctx("client"), scheme="https"); assert s == 200, (s, j)
    for cert in (None, "stranger"):
        try: call(port, "/health", ctx=client_ctx(cert), scheme="https"); raise SystemExit(f"mTLS accepted a client with certificate {cert}")
        except (urllib.error.URLError, OSError, http.client.HTTPException): pass
    assert call(port, "/health", ctx=client_ctx("client"), scheme="https")[0] == 200
finally:
    p.kill(); p.wait()
print("serve auth: TLS and mTLS ok")

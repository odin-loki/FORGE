# forge rpc-serve: authentication, TLS and remote scripts

`forge rpc-serve` lends a machine's devices (and, optionally, its script runtime) to other FORGE
processes. Code: `graph/include/forge/graph/rpc.hpp`, `graph/include/forge/graph/rpc_secure.hpp`.

## Authentication

Either or both, checked before the first frame:

- **Tokens** — `--token T` / `--token-file F` (repeatable). Compared in constant time. A token grants
  every permission.
- **Ed25519 keys** — reuses FORGE's `.fpk` signing (vendored Monocypher, same key files as
  `forge keygen`). `--authorized-keys FILE` (format below) or `--pubkey HEX|FILE` (full access).
  The server sends a 32-byte CSPRNG nonce; the client signs `"forge-rpc-auth-v1" || nonce || pubkey`
  (over TLS: `"forge-rpc-auth-v2" || nonce || pubkey || channel binding`, see below).

With no tokens or keys the server is unauthenticated (legacy) and warns if bound beyond loopback
(unless it requires client certificates).

### authorized_keys and per-key permissions

One key per line; blank lines and lines starting with `#` are ignored. Like ssh, a line may start
with options before the key; anything after the key is a comment:

```
# [options] <64-hex ed25519 public key> [comment]
d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a alice        # no options: full access
allow=device 3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c gpu-borrower
allow=scripts fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025 ci
```

| Option | Meaning |
|---|---|
| `allow=PERM[,PERM...]` | The key may only do these. Repeated `allow=` options add up. `allow=` with no value authenticates but permits nothing. |

| Permission | Request it allows |
|---|---|
| `device` | `FRPC` frames: running ops on the server's device (`rpc:host:port` devices, `pipe:` stages) |
| `scripts` | `FRPS` frames: remote scripts (`forge::remote`); the server must also run with `--allow-scripts` |
| `all` | both (the default for a line without options, so old files keep working) |

Unknown options or permissions, or a line without a 64-hex key, make `rpc-serve` refuse to start
(fail closed). A refused request gets the error `permission denied: ...` and the connection stays
usable; `forge::remote` raises it as `PermissionError`.

## TLS

`--tls-cert cert.pem --tls-key key.pem` (TLS ≥ 1.2, OpenSSL). Clients must verify the server with a
CA file (chain + hostname check) or a pinned SHA-256 certificate fingerprint; there is no silent
"trust anything". Build with OpenSSL found (`-DOPENSSL_ROOT_DIR=…` if it is outside the toolchain
sysroot); `-DFORGE_TLS=OFF` builds without it. TLS and plain builds of `rpc.hpp` live in different
inline namespaces, so they can be mixed in one program.

### Client certificates (mTLS)

`--tls-client-ca ca.pem` (needs `--tls-cert/--tls-key`) makes the server demand a client certificate
that chains to `ca.pem`; the TLS handshake fails without one. This is in addition to token / key
authentication, not a replacement: remote scripts still need a token or key. Clients present a
certificate with `FORGE_RPC_TLS_CERT` + `FORGE_RPC_TLS_KEY` (or `forge::remote` options `tls_cert` /
`tls_key`).

### Channel binding

Over TLS, token and key authentication are bound to the TLS connection, so an auth exchange captured
or relayed by a man in the middle (for example one holding a certificate the client wrongly trusts)
does not work on any other TLS connection:

- the binding is RFC 9266 `tls-exporter` (32 bytes, label `EXPORTER-Channel-Binding`, no context) on
  TLS 1.3 and on TLS 1.2 with extended master secret, otherwise RFC 5929 `tls-unique`;
- keys sign `"forge-rpc-auth-v2" || nonce || pubkey || binding` (method 4);
- tokens are never sent: the client sends `BLAKE2b-512(key = BLAKE2b-512(token), "forge-rpc-token-v2" || binding)`
  (method 3), which the server recomputes for each configured token.

A TLS server refuses the unbound methods (1 = token, 2 = key) with "TLS connections need
channel-bound authentication", so clients built before binding cannot authenticate over TLS; a plain
TCP server refuses the bound ones. Without TLS the handshake is unchanged.

### Client settings

Client settings (device `rpc:host:port` and `forge::remote`) come from the environment:
`FORGE_RPC_TOKEN`, `FORGE_RPC_KEY` (secret key file), `FORGE_RPC_TLS=1`, `FORGE_RPC_CA`,
`FORGE_RPC_FINGERPRINT`, `FORGE_RPC_SERVER_NAME`, `FORGE_RPC_TLS_CERT` / `FORGE_RPC_TLS_KEY` (client
certificate and its key, PEM; setting the certificate turns TLS on).

## Remote scripts

`--allow-scripts` (requires authentication) accepts `FRPS` frames and runs each one in its own child
process (`forge script --rpc-child`, source on stdin) with `--script-timeout MS` (default 60000), a
`--script-mem-mb N` memory cap (default 2048) and optional `--script-no-proc/--script-no-fs/--script-no-net`.
When any `--script-no-*` flag is set the child applies the OS sandbox with `require` (best-effort on
Windows; see docs/scripting.md), so a sandbox failure fails the script instead of running it unconfined. From a script:

```cpp
auto r = forge::remote("box2:7000", "std::println(\"{}\", os.hostname());",
                       {{"key", "~/.forge/me.key"}, {"fingerprint", "ab12…"}, {"args", std::vector<std::string>{"x"}}});
if (r["ok"]) std::print("{}", r["output"]); else std::println(stderr, "{}: {}", r["error_type"], r["error"]);
```

Wrong credentials and refused permissions raise `PermissionError`; network failures raise `IOError`.
Options: `args`, `timeout_ms`, `token`, `key`, `tls`, `ca`, `fingerprint`, `server_name`, `tls_cert`,
`tls_key`, `name`.

`forge::remote_call(node, fn, {args...}, opts)` sends a function instead of a whole script: its source and the
plain values it captures travel as a generated script over the same path (same child, sandbox and limits), and the
call returns `fn(args...)`'s value. See docs/scripting.md, "Remote execution".

```sh
forge keygen -o me                                     # me.key / me.pub
forge rpc-serve --bind 0.0.0.0 --authorized-keys keys.txt --tls-cert c.pem --tls-key k.pem \
    --tls-client-ca clients-ca.pem --allow-scripts
FORGE_RPC_KEY=me.key FORGE_RPC_CA=c.pem FORGE_RPC_TLS_CERT=me-client.pem FORGE_RPC_TLS_KEY=me-client.key \
    FORGE_DEVICE=rpc:box2:7000 forge run model.fpk --prompt hi
```

Not yet: per-token permissions (tokens always have full access), mapping client-certificate
identities to permissions, and shipping lambdas (only source strings travel).

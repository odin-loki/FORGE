# rpc: remote devices, authentication, TLS and remote scripts

`forge rpc-serve` lends a machine's devices, and optionally a script runtime, to other FORGE processes. Code: `graph/include/forge/graph/rpc.hpp` (protocol, client, server) and `rpc_secure.hpp` (TLS and authentication). Tests: `tests/test_rpc.cpp` (ctests `test_rpc` and `test_rpc_tls`).

## Concepts
- **Remote device.** A client uses the device `rpc:host:port` (for example `FORGE_DEVICE=rpc:box2:7000`), or a pipeline stage such as `pipe:rpc:box2:7000,cpu` (see [graph.md](graph.md)). Each op is sent as an `FRPC` frame with its inputs; the server runs it on its own device registry and returns the output. Every op is still checked by the oracle on the client. If the connection fails, the client falls back to the CPU.
- **Remote script.** An `FRPS` frame carries ForgeScript source; the server runs it in a sandboxed child process and returns `{ok, error_type, error, output}`. Clients are `forge::remote` and `forge::remote_call` ([scripting.md](scripting.md)).
- **Security.** Optional TLS (OpenSSL), then optional authentication with tokens or Ed25519 keys, checked before the first frame. Keys can be limited to device ops or scripts.

## `forge rpc-serve` reference

```
forge rpc-serve [--port 7000] [--bind 127.0.0.1] [--device cpu|vulkan|cuda:0|...]
    [--token T | --token-file F]... [--authorized-keys FILE | --pubkey HEX|FILE]...
    [--tls-cert C --tls-key K [--tls-client-ca CA]]
    [--allow-scripts [--script-timeout MS] [--script-mem-mb N] [--script-no-proc] [--script-no-fs] [--script-no-net]]
```

| Flag | Default | Effect |
|---|---|---|
| `--port`, `--bind` | 7000, `127.0.0.1` | Listen address |
| `--device` | `cpu` | Device that runs the ops. An unavailable device falls back to `cpu` with a warning |
| `--token T`, `--token-file F` | — | Accepted bearer tokens (repeatable). A file's whitespace is stripped. Tokens have full access |
| `--authorized-keys FILE` | — | Ed25519 public keys with optional permissions (format below; repeatable) |
| `--pubkey HEX\|FILE` | — | One public key (64 hex, or a `.pub` file from `forge keygen`) with full access (repeatable) |
| `--tls-cert`, `--tls-key` | — | Serve TLS only (TLS ≥ 1.2). Must be given together |
| `--tls-client-ca CA` | — | mTLS: require a client certificate that chains to `CA`. Needs `--tls-cert`/`--tls-key` |
| `--allow-scripts` | off | Accept `FRPS` frames. Requires tokens or keys; the server refuses to start without them |
| `--script-timeout MS` | 60000 | Per-script limit; a client may ask for less, not more |
| `--script-mem-mb N` | 2048 | Memory limit of the script child |
| `--script-no-proc`, `--script-no-fs`, `--script-no-net` | off | Deny those capabilities in the child (see Remote scripts) |

With no tokens or keys the server accepts anyone (legacy mode). It then warns if bound to anything other than `127.0.0.1`, `::1` or `localhost`, unless it requires client certificates.

## Authentication
Client and server exchange an `FRPH` hello naming a method:

| Method | Transport | Exchange |
|---|---|---|
| 1 token | plain TCP | Client sends the token; compared in constant time |
| 2 key | plain TCP | Client sends its public key, server sends a 32-byte CSPRNG nonce, client signs `"forge-rpc-auth-v1" ‖ nonce ‖ pubkey` |
| 3 bound token | TLS | Client sends `BLAKE2b-512(key = BLAKE2b-512(token), "forge-rpc-token-v2" ‖ cb)`; the token itself never crosses the wire |
| 4 bound key | TLS | As 2, signing `"forge-rpc-auth-v2" ‖ nonce ‖ pubkey ‖ cb` |

Keys use FORGE's `.fpk` signing code (vendored Monocypher) and the same files as `forge keygen` (secret = 64-hex seed, public = 64 hex). A client with both a key and a token uses the key.

### authorized_keys and per-key permissions
One key per line; blank lines and lines starting with `#` are ignored. As in ssh, options may come before the key; the rest of the line is a comment:

```
# [options] <64-hex ed25519 public key> [comment]
d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a alice        # no options: full access
allow=device 3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c gpu-borrower
allow=scripts fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025 ci
```

| Option | Meaning |
|---|---|
| `allow=PERM[,PERM...]` | The key may only do these. Repeated `allow=` options add up. `allow=` with no value authenticates but permits nothing |

| Permission | Allows |
|---|---|
| `device` | `FRPC` frames: ops on the server's device (`rpc:` devices, `pipe:` stages) |
| `scripts` | `FRPS` frames: remote scripts; the server must also run with `--allow-scripts` |
| `all` | Both. The default for a line without options, so older files keep working |

Unknown options or permissions, or a line without a 64-hex key, make `rpc-serve` refuse to start (fail closed). A refused request gets `permission denied: ...` and the connection stays usable; `forge::remote` raises it as `PermissionError`.

## TLS
`--tls-cert cert.pem --tls-key key.pem` (TLS ≥ 1.2). Clients must verify the server with a CA file (chain plus host-name check) or a pinned SHA-256 certificate fingerprint; there is no "trust anything" setting in the environment or `forge::remote` options.

Build: OpenSSL is found by CMake (`-DOPENSSL_ROOT_DIR=…` if it is outside the toolchain sysroot); `-DFORGE_TLS=OFF` builds without it. Without TLS support, a client that asks for TLS fails with "this build has no TLS". TLS and plain builds of `rpc.hpp` use different inline namespaces, so both can be linked into one program.

### Client certificates (mTLS)
`--tls-client-ca ca.pem` makes the TLS handshake fail unless the client presents a certificate that chains to `ca.pem`. This is in addition to token or key authentication, not a replacement: remote scripts still need a token or key. Clients present a certificate with `FORGE_RPC_TLS_CERT` and `FORGE_RPC_TLS_KEY`, or the `forge::remote` options `tls_cert` / `tls_key`.

### Channel binding
Over TLS, token and key authentication are bound to the TLS connection (methods 3 and 4), so an exchange captured or relayed by a man in the middle (for example one holding a certificate the client wrongly trusts) fails on any other connection. The binding `cb` is RFC 9266 `tls-exporter` (32 bytes, label `EXPORTER-Channel-Binding`, no context) on TLS 1.3 and on TLS 1.2 with extended master secret, otherwise RFC 5929 `tls-unique`.

A TLS server refuses methods 1 and 2 ("TLS connections need channel-bound authentication"), so **clients built before channel binding cannot authenticate to a TLS server**. A plain TCP server refuses methods 3 and 4. Without TLS the handshake is unchanged.

## Client settings
Device clients (`rpc:host:port`) read the environment. `forge::remote` reads it too, and its options override it.

| Variable | `forge::remote` option | Meaning |
|---|---|---|
| `FORGE_RPC_TOKEN` | `token` | Bearer token |
| `FORGE_RPC_KEY` | `key` | Secret key file (`forge keygen`) |
| `FORGE_RPC_TLS=1` | `tls` | Use TLS |
| `FORGE_RPC_CA` | `ca` | CA file for server verification (turns TLS on) |
| `FORGE_RPC_FINGERPRINT` | `fingerprint` | SHA-256 of the server certificate, hex (turns TLS on) |
| `FORGE_RPC_SERVER_NAME` | `server_name` | Name checked against the certificate (default: the host) |
| `FORGE_RPC_TLS_CERT`, `FORGE_RPC_TLS_KEY` | `tls_cert`, `tls_key` | Client certificate and key, PEM (the certificate turns TLS on) |

## Remote scripts
With `--allow-scripts`, each `FRPS` frame runs in its own child process (`forge script --rpc-child`, source on stdin) with the timeout and memory limit above, at most 32 MiB of output, and `FORGE_RPC_TOKEN`, `FORGE_RPC_KEY`, `FORGE_RPC_TLS_KEY` and `FORGE_SCRIPT_CACHE_SIGN_SEED` removed from its environment. When any `--script-no-*` flag is set, the child applies the OS sandbox with `require` on POSIX, so a sandbox failure fails the script instead of running it unconfined. On Windows it is `best-effort` (job object; the interpreter enforces the rest); see [scripting.md](scripting.md).

```cpp
auto r = forge::remote("box2:7000", "std::println(\"{}\", os.hostname());",
                       {{"key", "~/.forge/me.key"}, {"fingerprint", "ab12…"}, {"args", std::vector<std::string>{"x"}}});
if (r["ok"]) std::print("{}", r["output"]); else std::println(stderr, "{}: {}", r["error_type"], r["error"]);
```

Options: `args`, `timeout_ms`, `name`, plus the credential options in the table above. Failed authentication and refused permissions raise `PermissionError`; network failures raise `IOError`.

`forge::remote_call(node, fn, {args...}, opts)` sends a function instead of a whole script: its source and the plain values it captures travel as a generated script over the same path (same child, sandbox and limits), and the call returns `fn(args...)`. See [scripting.md](scripting.md), "Remote execution".

## Example

```sh
forge keygen -o me                                     # me.key / me.pub
forge rpc-serve --bind 0.0.0.0 --authorized-keys keys.txt --tls-cert c.pem --tls-key k.pem \
    --tls-client-ca clients-ca.pem --allow-scripts
FORGE_RPC_KEY=me.key FORGE_RPC_CA=c.pem FORGE_RPC_TLS_CERT=me-client.pem FORGE_RPC_TLS_KEY=me-client.key \
    FORGE_DEVICE=rpc:box2:7000 forge run model.fpk --prompt hi
```

## Limits
- Not optimised: weights travel with every op (no remote residency), and device ops on one server run one at a time.
- Tokens always have full access; there are no per-token permissions.
- Client-certificate identities are not mapped to permissions; mTLS only gates the connection.
- Only source travels for scripts and `remote_call`; compiled lambdas are not shipped.
- `test_rpc_tls` needs OpenSSL and its CLI, and exits 77 (skipped) without them.

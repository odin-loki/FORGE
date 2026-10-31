# forge rpc-serve: authentication, TLS and remote scripts

`forge rpc-serve` lends a machine's devices (and, optionally, its script runtime) to other FORGE
processes. Code: `flow/include/forge/flow/rpc.hpp`, `flow/include/forge/flow/rpc_secure.hpp`.

## Authentication

Either or both, checked before the first frame:

- **Tokens** — `--token T` / `--token-file F` (repeatable). Compared in constant time.
- **Ed25519 keys** — reuses FORGE's `.fpk` signing (vendored Monocypher, same key files as
  `forge keygen`). `--authorized-keys FILE` (one 64-hex public key per line, `#` comments) or
  `--pubkey HEX|FILE`. The server sends a 32-byte CSPRNG nonce; the client signs
  `"forge-rpc-auth-v1" || nonce || pubkey`.

With no tokens or keys the server is unauthenticated (legacy) and warns if bound beyond loopback.

## TLS

`--tls-cert cert.pem --tls-key key.pem` (TLS ≥ 1.2, OpenSSL). Clients must verify the server with a
CA file (chain + hostname check) or a pinned SHA-256 certificate fingerprint; there is no silent
"trust anything". Build with OpenSSL found (`-DOPENSSL_ROOT_DIR=…` if it is outside the toolchain
sysroot); `-DFORGE_TLS=OFF` builds without it. TLS and plain builds of `rpc.hpp` live in different
inline namespaces, so they can be mixed in one program.

Client settings (device `rpc:host:port` and `forge::remote`) come from the environment:
`FORGE_RPC_TOKEN`, `FORGE_RPC_KEY` (secret key file), `FORGE_RPC_TLS=1`, `FORGE_RPC_CA`,
`FORGE_RPC_FINGERPRINT`, `FORGE_RPC_SERVER_NAME`.

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

Wrong credentials raise `PermissionError`; network failures raise `IOError`.

```sh
forge keygen -o me                                     # me.key / me.pub
forge rpc-serve --bind 0.0.0.0 --authorized-keys keys.txt --tls-cert c.pem --tls-key k.pem --allow-scripts
```

Not yet: TLS channel binding of the key signature, client certificates, per-key permissions, and
shipping lambdas (only source strings travel).

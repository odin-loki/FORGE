# Security

## Reporting
Report vulnerabilities privately to the author (Odin Loch) rather than in a public issue.

## Threat model
- **Model files are untrusted input.** The `.fpk` reader bounds-checks every section and tensor against the file size before use, and the parsers are fuzzed (ctest label `fuzz`). `forge inspect --verify` / `model::open(path, true)` re-hash every tensor (BLAKE3).
- **Signatures.** `forge sign` / `forge verify` use Ed25519 over the file's section digests. `forge verify` without `--pubkey` still requires a valid signature, so an unsigned file fails. Pass `--pubkey` to pin the signer; otherwise any valid key is accepted.
- **Encryption.** `forge encrypt` uses XChaCha20-Poly1305 for tensor data. Keep keys out of the repository and process environment where possible.
- **Network.** There is no telemetry. The runtime opens sockets only when you run `forge rpc-serve` or `forge-serve` (server or proxy). Neither has authentication or TLS: the RPC protocol is unauthenticated and `forge-serve` binds to 127.0.0.1 by default. Do not expose them on untrusted networks without a reverse proxy or firewall.
- **Plugins and dlopen.** GPU backends dlopen system driver libraries (libvulkan, libcuda, libamdhip64, OpenCL, Level Zero). `FORGE_LIB` lets the Python binding load a library from any path; treat it as trusted configuration.

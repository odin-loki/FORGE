# Releasing FORGE

## Concepts
`tools/release.py` drives a release that is reproducible and signed. `SOURCE_DATE_EPOCH` is taken from the release commit, so rebuilding the same commit produces identical archives and an identical SBOM (as long as `FORGE_REPRODUCIBLE=ON`, the default).

## Commands (`tools/release.py --help`)

| Command | Effect |
|---------|--------|
| `version` | Prints the version from `CMakeLists.txt` (currently `0.1.0`) |
| `bump X.Y.Z` | Sets the project VERSION, plus the Android and Swift metadata. `libforge` follows `${PROJECT_VERSION}` |
| `changelog [--since TAG]` | Adds a new section at the top of `CHANGELOG.md`, built from the git history |
| `build [--preset release] [--skip-tests] [--sign-key PEM] [--out DIR]` | Runs configure, build (`-j` up to 4) and `ctest -LE slow`, then CPack binary and source archives, the SPDX SBOM (`tools/sbom.py`), `SHA256SUMS` and optional detached Ed25519 `.sig` files. Output goes to `dist/<version>/` |
| `verify DIR [--pub PUB]` | Re-checks `SHA256SUMS`, and the signatures if a public key is given |
| `tag` | Creates an annotated `vX.Y.Z` tag. It never pushes the tag |

## Procedure
1. `tools/release.py bump 1.0.0`
2. `tools/release.py changelog`, then edit `CHANGELOG.md` by hand and commit.
3. `tools/release.py build --sign-key release-ed25519.pem`
4. `tools/release.py verify dist/1.0.0 --pub release-ed25519.pub`
5. `tools/release.py tag`, then push the tag yourself.

To generate a signing key:
```sh
openssl genpkey -algorithm ed25519 -out release-ed25519.pem
openssl pkey -in release-ed25519.pem -pubout -out release-ed25519.pub
```

## Platform notes and limits
- To build for other platforms, pass `--preset windows-mingw`, `android-arm64` or `macos-osxcross`.
- `cmake --workflow --preset release` runs configure, build and test in a single command.
- `ci/repro_check.sh` checks that rebuilds are bit-identical.
- **None of the release commands except `version` were run while writing this page.**

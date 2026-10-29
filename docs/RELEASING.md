# Releasing FORGE

1. `tools/release.py bump 1.0.0` — sets the project VERSION (and the libforge VERSION via `${PROJECT_VERSION}`).
2. `tools/release.py changelog` — prepends a grouped section to `CHANGELOG.md` from commits since the last tag. Edit it by hand.
3. Commit, then `tools/release.py build --sign-key release-ed25519.pem` — configures the `release` preset, builds, runs
   `ctest -LE slow`, runs CPack (TGZ + DEB on Linux, ZIP on Windows, plus a source archive), writes the SPDX SBOM
   (`tools/sbom.py`), `SHA256SUMS`, and detached Ed25519 `.sig` files. Output: `dist/<version>/`.
   `SOURCE_DATE_EPOCH` is taken from the release commit so archives and the SBOM are reproducible.
4. `tools/release.py verify dist/<version> --pub release-ed25519.pub` — re-checks hashes and signatures.
5. `tools/release.py tag` — annotated `vX.Y.Z` tag (it never pushes; push the tag yourself).

Other platforms use the same flow with `--preset windows-mingw`, `android-arm64`, or `macos-osxcross`
(see `CMakePresets.json`). There is also a single-command `cmake --workflow --preset release`.
Key generation: `openssl genpkey -algorithm ed25519 -out release-ed25519.pem && openssl pkey -in release-ed25519.pem -pubout -out release-ed25519.pub`.

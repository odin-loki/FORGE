#!/usr/bin/env python3
"""Generate an SBOM for FORGE: the project, its vendored third-party code, and a SHA-256 per source file,
as SPDX 2.3 JSON (default) or CycloneDX 1.5 JSON. Deterministic for a given tree (no timestamps unless
SOURCE_DATE_EPOCH is set, serial number derived from the commit), so the SBOM itself is reproducible.

usage: tools/sbom.py [--format spdx|cyclonedx] [--out forge.spdx.json] [--root .]
"""
import argparse, datetime, hashlib, json, os, re, subprocess, sys, uuid

VENDORED = [  # name, version, path, licence (SPDX expression), supplier, download
    ("monocypher", "4.0.2", "third_party/monocypher", "BSD-2-Clause OR CC0-1.0", "Organization: Loup Vaillant", "https://monocypher.org"),
    ("level-zero", "1.20", "third_party/level-zero", "MIT", "Organization: Intel Corporation", "https://github.com/oneapi-src/level-zero"),
]
FORGE_LICENCE = "AGPL-3.0-or-later OR LicenseRef-FORGE-Commercial"
REPO = "https://github.com/odin-loki/FORGE"
SRC_EXT = (".hpp", ".h", ".cpp", ".c", ".py", ".cmake", ".txt", ".json", ".fspec", ".fbp", ".glsl", ".cu", ".md", ".yml")

def files(root):
    try:
        out = subprocess.run(["git", "-C", root, "ls-files"], check=True, capture_output=True, text=True).stdout.split("\n")
    except Exception:
        out = [os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/") for d, _, fs in os.walk(root)
               if "/build" not in d.replace(os.sep, "/") and "/.git" not in d.replace(os.sep, "/") for f in fs]
    return sorted(f for f in out if f and f.endswith(SRC_EXT) and os.path.isfile(os.path.join(root, f)))

def sha(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f: h.update(f.read())
    return h.hexdigest()

def spdx_id(s): return "SPDXRef-" + "".join(c if c.isalnum() or c in ".-" else "-" for c in s)

def project_version(root):
    try: return re.search(r"project\(FORGE VERSION ([0-9.]+)", open(os.path.join(root, "CMakeLists.txt")).read()).group(1)
    except Exception: return "0"

def cyclonedx(root, rev, created, owner):
    """CycloneDX 1.5: forge as metadata.component, one library component per vendored package, every source
    file as a nested component of type "file" with its SHA-1 and SHA-256."""
    def file_components(pkg):
        return [{"type": "file", "bom-ref": "file:" + f, "name": f,
                 "hashes": [{"alg": "SHA-1", "content": sha(os.path.join(root, f), "sha1")},
                            {"alg": "SHA-256", "content": sha(os.path.join(root, f), "sha256")}]} for f in owner.get(pkg, [])]
    ver = project_version(root)
    forge_ref = "pkg:github/odin-loki/FORGE@" + rev
    main = {"type": "application", "bom-ref": forge_ref, "name": "forge", "version": ver, "purl": forge_ref,
            "supplier": {"name": "Odin Loch"}, "copyright": "Copyright (c) 2026 Odin Loch (Imortek)",
            "licenses": [{"expression": FORGE_LICENCE}],
            "externalReferences": [{"type": "vcs", "url": REPO}],
            "properties": [{"name": "forge:git-revision", "value": rev}],
            "components": file_components("forge")}
    comps, deps = [], []
    for n, v, p, lic, sup, dl in VENDORED:
        ref = "pkg:generic/" + n + "@" + v
        comps.append({"type": "library", "bom-ref": ref, "name": n, "version": v, "purl": ref, "scope": "required",
                      "supplier": {"name": sup.split(": ", 1)[-1]}, "licenses": [{"expression": lic}],
                      "externalReferences": [{"type": "website", "url": dl}],
                      "properties": [{"name": "forge:vendored-path", "value": p}],
                      "components": file_components(n)})
        deps.append({"ref": ref, "dependsOn": []})
    return {"bomFormat": "CycloneDX", "specVersion": "1.5", "version": 1,
            "serialNumber": "urn:uuid:" + str(uuid.uuid5(uuid.NAMESPACE_URL, REPO + "/cyclonedx/" + rev)),
            "metadata": {"timestamp": created, "tools": {"components": [{"type": "application", "name": "forge-tools-sbom.py"}]},
                         "component": main, "licenses": [{"license": {"id": "CC0-1.0"}}]},
            "components": comps,
            "dependencies": [{"ref": forge_ref, "dependsOn": sorted(d["ref"] for d in deps)}] + deps}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="-"); ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--format", choices=("spdx", "cyclonedx"), default="spdx")
    a = ap.parse_args(); root = a.root
    try: rev = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except Exception: rev = "unknown"
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    created = datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fl = files(root)
    doc = {"spdxVersion": "SPDX-2.3", "dataLicense": "CC0-1.0", "SPDXID": "SPDXRef-DOCUMENT", "name": "forge-" + rev[:12],
           "documentNamespace": REPO + "/spdx/" + rev,
           "creationInfo": {"created": created, "creators": ["Tool: forge-tools-sbom.py"]},
           "packages": [], "files": [], "relationships": []}
    vend_paths = {p: n for n, _, p, *_ in VENDORED}
    owner = {}
    for f in fl:
        pkg = next((n for p, n in vend_paths.items() if f.startswith(p + "/")), "forge")
        owner.setdefault(pkg, []).append(f)
        doc["files"].append({"fileName": "./" + f, "SPDXID": spdx_id("File-" + f),
                             "checksums": [{"algorithm": "SHA1", "checksumValue": sha(os.path.join(root, f), "sha1")},
                                           {"algorithm": "SHA256", "checksumValue": sha(os.path.join(root, f), "sha256")}],
                             "licenseConcluded": next((v[3] for v in VENDORED if v[0] == pkg), "AGPL-3.0-or-later"), "copyrightText": "NOASSERTION"})
    if a.format == "cyclonedx": doc = cyclonedx(root, rev, created, owner)
    else: spdx_fill(doc, root, rev, owner)
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if a.out == "-": sys.stdout.write(text)
    else:
        with open(a.out, "w") as f: f.write(text)

def spdx_fill(doc, root, rev, owner):
    def verif(pkg):  # SPDX package verification code: sha1 over sorted file sha1s
        s = sorted(sha(os.path.join(root, f), "sha1") for f in owner.get(pkg, []))
        return hashlib.sha1("".join(s).encode()).hexdigest()
    doc["packages"].append({"name": "forge", "SPDXID": "SPDXRef-Package-forge", "versionInfo": rev, "supplier": "Person: Odin Loch",
                            "downloadLocation": "git+" + REPO + "@" + rev, "filesAnalyzed": True,
                            "packageVerificationCode": {"packageVerificationCodeValue": verif("forge")},
                            "licenseConcluded": FORGE_LICENCE, "licenseDeclared": FORGE_LICENCE,
                            "copyrightText": "Copyright (c) 2026 Odin Loch (Imortek)"})
    doc["relationships"].append({"spdxElementId": "SPDXRef-DOCUMENT", "relationshipType": "DESCRIBES", "relatedSpdxElement": "SPDXRef-Package-forge"})
    for n, v, p, lic, sup, dl in VENDORED:
        doc["packages"].append({"name": n, "SPDXID": spdx_id("Package-" + n), "versionInfo": v, "supplier": sup, "downloadLocation": dl,
                                "filesAnalyzed": True, "packageVerificationCode": {"packageVerificationCodeValue": verif(n)},
                                "licenseConcluded": lic, "licenseDeclared": lic, "copyrightText": "NOASSERTION"})
        doc["relationships"].append({"spdxElementId": "SPDXRef-Package-forge", "relationshipType": "CONTAINS", "relatedSpdxElement": spdx_id("Package-" + n)})
    for pkg, fs in owner.items():
        for f in fs: doc["relationships"].append({"spdxElementId": spdx_id("Package-" + pkg), "relationshipType": "CONTAINS", "relatedSpdxElement": spdx_id("File-" + f)})
    doc["hasExtractedLicensingInfos"] = [{"licenseId": "LicenseRef-FORGE-Commercial", "name": "FORGE commercial licence", "extractedText": "See COMMERCIAL-LICENCE.md"}]

if __name__ == "__main__": main()

"""SBOM check: SPDX 2.3 and CycloneDX 1.5 structure, every vendored file attributed, deterministic output."""
import json, re, subprocess, sys
root, py = sys.argv[1], sys.executable
run = lambda *fmt: subprocess.run([py, root + "/tools/sbom.py", "--root", root, *fmt], check=True, capture_output=True, text=True).stdout
a, b = run(), run()
assert a == b, "SBOM not deterministic"
assert a == run("--format", "spdx"), "spdx is the default format"
d = json.loads(a)
assert d["spdxVersion"] == "SPDX-2.3" and d["dataLicense"] == "CC0-1.0"
names = {p["name"]: p for p in d["packages"]}
assert {"forge", "monocypher", "level-zero"} <= set(names), names.keys()
ids = {p["SPDXID"] for p in d["packages"]} | {f["SPDXID"] for f in d["files"]} | {"SPDXRef-DOCUMENT"}
assert all(r["spdxElementId"] in ids and r["relatedSpdxElement"] in ids for r in d["relationships"])
mono = [f for f in d["files"] if "third_party/monocypher/" in f["fileName"]]
assert mono and all(f["licenseConcluded"] == "BSD-2-Clause OR CC0-1.0" for f in mono)
ze = [f for f in d["files"] if "third_party/level-zero/" in f["fileName"]]
assert ze and all(f["licenseConcluded"] == "MIT" for f in ze)
assert any(f["fileName"] == "./FORGE_PLAN.md" for f in d["files"])
# data and code derived from third-party sources (NOTICE): a package each, the files attributed with the source's licence
derived = {"unicode-ucd": ("Unicode-3.0", ["os/src/text/ucd_data.inc"]), "unicode-cldr": ("Unicode-3.0", ["os/src/text/cldr_data.inc"]),
           "whatwg-encoding-indexes": ("NOASSERTION", ["os/src/text/encoding_data.inc"]),
           "llvm-clang-intrinsic-headers": ("Apache-2.0 WITH LLVM-exception", ["ops/intrinsics/x86.json", "ops/intrinsics/ppc.json"]),
           "llvm-clang-arm-intrinsic-headers": ("Apache-2.0 WITH LLVM-exception AND MIT", ["ops/intrinsics/arm.json"]),
           "rvv-intrinsic-doc": ("NOASSERTION", ["ops/intrinsics/riscv.json", "ops/intrinsics/riscv_policy.json"]),
           "llama.cpp": ("(AGPL-3.0-or-later OR LicenseRef-FORGE-Commercial) AND MIT", ["modelfile/include/forge/import/quant_recipe.hpp"])}
assert set(derived) <= set(names), set(derived) - set(names)
by_file = {f["fileName"][2:]: f for f in d["files"]}
contains = {(r["spdxElementId"], r["relatedSpdxElement"]) for r in d["relationships"] if r["relationshipType"] == "CONTAINS"}
for n, (lic, fs) in derived.items():
    assert names[n]["licenseConcluded"] == lic and names[n]["comment"], n
    assert ("SPDXRef-Package-forge", names[n]["SPDXID"]) in contains, n
    for f in fs:
        assert by_file[f]["licenseConcluded"] == lic, f
        assert (names[n]["SPDXID"], by_file[f]["SPDXID"]) in contains and ("SPDXRef-Package-forge", by_file[f]["SPDXID"]) not in contains, f
# every intrinsic database is attributed to one of them, and NOTICE names every package's files
import glob, os
dbs = {os.path.relpath(p, root).replace(os.sep, "/") for p in glob.glob(root + "/ops/intrinsics/*.json")}
assert dbs and all(by_file[f]["licenseConcluded"] != "AGPL-3.0-or-later" for f in dbs), "an intrinsic database without a source attribution"
notice = open(root + "/NOTICE", encoding="utf-8").read()
for needle in ("Unicode License V3", "UNICODE LICENSE V3", "Apache-2.0 WITH LLVM-exception", "rvv-intrinsic-doc", "NOT ESTABLISHED", "ucd_data.inc", "cldr_data.inc",
               "encoding_data.inc", "quant_recipe.hpp", "tests/data/tokenizer", "tests/data/chat_template", "third_party/monocypher", "ze_api.h"):
    assert needle in notice, f"NOTICE does not mention {needle}"

# CycloneDX 1.5 JSON: the required properties and value formats of the 1.5 schema, checked by hand (no schema
# validator is assumed to be installed).
ca, cb = run("--format", "cyclonedx"), run("--format", "cyclonedx")
assert ca == cb, "CycloneDX SBOM not deterministic"
c = json.loads(ca)
assert c["bomFormat"] == "CycloneDX" and c["specVersion"] == "1.5" and c["version"] == 1
assert re.fullmatch(r"urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", c["serialNumber"]), c["serialNumber"]
assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", c["metadata"]["timestamp"])
main = c["metadata"]["component"]
assert main["type"] == "application" and main["name"] == "forge" and re.fullmatch(r"\d+\.\d+\.\d+", main["version"])
assert main["licenses"] == [{"expression": "AGPL-3.0-or-later OR LicenseRef-FORGE-Commercial"}]
comps = {x["name"]: x for x in c["components"]}
assert set(comps) == {"monocypher", "level-zero"} | set(derived), comps.keys()
for n, (lic, fs) in derived.items():
    assert comps[n].get("licenses") == (None if lic == "NOASSERTION" else [{"expression": lic}]), n
    assert set(fs) <= {k["name"] for k in comps[n]["components"]} and comps[n]["description"], n
assert comps["monocypher"]["version"] == "4.0.2" and comps["monocypher"]["licenses"] == [{"expression": "BSD-2-Clause OR CC0-1.0"}]
assert comps["level-zero"]["licenses"] == [{"expression": "MIT"}]
def walk(x):
    yield x
    for k in x.get("components", []): yield from walk(k)
every = [k for x in [main, *c["components"]] for k in walk(x)]
refs = [x["bom-ref"] for x in every]
assert len(refs) == len(set(refs)), "bom-ref values must be unique"
for x in every:
    assert x["type"] in ("application", "library", "file") and x["name"]
    for h in x.get("hashes", []):
        assert h["alg"] in ("SHA-1", "SHA-256") and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", h["content"])
cfiles = {x["name"]: x for x in every if x["type"] == "file"}
# the same file set and the same SHA-256 per file as the SPDX document
assert set(cfiles) == {f["fileName"][2:] for f in d["files"]}
spdx_sha = {f["fileName"][2:]: next(k["checksumValue"] for k in f["checksums"] if k["algorithm"] == "SHA256") for f in d["files"]}
assert all(next(h["content"] for h in x["hashes"] if h["alg"] == "SHA-256") == spdx_sha[n] for n, x in cfiles.items())
assert {k["name"] for k in comps["monocypher"]["components"]} == {f["fileName"][2:] for f in mono}
deps = {x["ref"]: x["dependsOn"] for x in c["dependencies"]}
assert set(deps) <= set(refs) and all(r in refs for v in deps.values() for r in v)
assert sorted(deps[main["bom-ref"]]) == sorted(x["bom-ref"] for x in c["components"])
print(f"sbom ok: spdx {len(d['files'])} files, {len(d['packages'])} packages; cyclonedx {len(cfiles)} files, {len(c['components'])} components")

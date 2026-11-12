#!/usr/bin/env python3
"""Checks that README.md, the other top-level guides and docs/*.md point at things that exist.

  links       every relative Markdown link `[text](path#anchor)` resolves to a file or directory of the repository,
              and its `#anchor`, when the target is a Markdown file, to a heading of that file (GitHub's slug rules)
  repo paths  every backticked token that is written as a repository path (`tests/test_aot.cpp`, `docs/gpu.md`,
              `ops/gen/gpu_gen.py`) names an existing file or directory. A header may be written the short way the
              guides use, relative to its module's include directory (`graph/aot.hpp` for
              graph/include/forge/graph/aot.hpp, `import/hf.hpp` for modelfile/include/forge/import/hf.hpp).

  index       every docs/*.md is linked from docs/index.md

Not checked: URLs with a scheme, paths with a placeholder or pattern (`<isa>`, `*`, `{`, `...`), paths under build
directories, and docs/reference.md (generated; `docs_up_to_date` covers it).

usage: tools/check_docs_links.py [--root DIR] [--list]      exit 1 when something does not resolve
"""
import argparse, os, re, sys

TOP = ("core", "os", "ops", "simd", "gpu", "graph", "modelfile", "inference", "loadbalancer", "script", "support", "verify", "security", "apps", "import",
       "python", "bindings", "examples", "tests", "fuzz", "docs", "cmake", "ci", "tools", "perf", "third_party", ".github")
PATH_RE = re.compile(r"^(?:%s)/[A-Za-z0-9_./+-]+$" % "|".join(re.escape(t) for t in TOP))
SKIP_FILES = {"docs/reference.md"}
# Paths the guides name on purpose although the repository does not hold them, with the reason.
NOT_IN_TREE = {
    "bindings/apple/FORGE.xcframework": "built by tools/build_xcframework.sh",
    "tests/data/ucd": "the Unicode conformance files, downloaded by cmake/fetch_ucd.cmake (git-ignored)",
    "ci/sanitizers": "where a sanitizer suppression would go; there are none, so the directory does not exist",
}

def slug(heading):
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces to hyphens."""
    h = re.sub(r"`", "", heading.strip())
    h = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", h)   # a link in a heading contributes its text
    h = h.lower()
    h = "".join(c for c in h if c.isalnum() or c in " -_")
    return h.replace(" ", "-")

def anchors(path, cache={}):
    if path not in cache:
        out, seen, fence = set(), {}, False
        for line in open(path, encoding="utf-8"):
            if line.lstrip().startswith("```"): fence = not fence; continue
            m = None if fence else re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
            if m:
                s = slug(m.group(2)); n = seen.get(s, 0); seen[s] = n + 1
                out.add(s if n == 0 else f"{s}-{n}")
            for a in re.findall(r"<a\s+(?:name|id)=\"([^\"]+)\"", line): out.add(a)
        cache[path] = out
    return cache[path]

def candidates(tok):
    """Where a backticked path may live: as written, or under its module's include directory."""
    yield tok
    top, _, rest = tok.partition("/")
    if top == "import":
        yield "modelfile/include/forge/import/" + rest
        yield "import/" + rest
    yield f"{top}/include/forge/{top}/{rest}"
    yield f"{top}/include/forge/{rest}"
    yield f"{top}/include/{rest}"
    yield f"{top}/src/{rest}"
    for mod in ("support", "os", "core", "security"):   # support/include/forge/<name>/..., os/include/forge/os/...
        yield f"{mod}/include/forge/{tok}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--list", action="store_true", help="print every checked reference")
    a = ap.parse_args(); root = a.root
    files = [f for f in sorted(os.listdir(root)) if f.endswith(".md")] + ["docs/" + f for f in sorted(os.listdir(os.path.join(root, "docs"))) if f.endswith(".md")]
    for extra in ("ops/README.md", "verify/README.md", "examples/README.md"):
        if os.path.exists(os.path.join(root, extra)): files.append(extra)
    bad, n_links, n_paths = [], 0, 0
    for rel in files:
        if rel in SKIP_FILES: continue
        base = os.path.dirname(os.path.join(root, rel)); fence = False
        for ln, line in enumerate(open(os.path.join(root, rel), encoding="utf-8"), 1):
            if line.lstrip().startswith("```"): fence = not fence; continue
            if fence: continue
            # links, outside code spans
            for m in re.finditer(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", re.sub(r"`[^`]*`", "``", line)):
                target = m.group(1).strip("<>")
                if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target): continue   # http:, mailto:, ...
                path, _, frag = target.partition("#")
                dest = os.path.normpath(os.path.join(base, path)) if path else os.path.join(root, rel)
                n_links += 1
                if a.list: print(f"{rel}:{ln}: link {target}")
                if not os.path.exists(dest): bad.append(f"{rel}:{ln}: link target does not exist: {target}"); continue
                if frag and dest.endswith(".md") and frag.lower() not in anchors(dest):
                    bad.append(f"{rel}:{ln}: no heading for anchor #{frag} in {os.path.relpath(dest, root)}")
            # backticked repository paths
            for tok in re.findall(r"`([^`\s]+)`", line):
                tok = tok.rstrip(".,;:)").lstrip("(")
                tok = re.sub(r":\d+(-\d+)?$", "", tok)              # file.cpp:123
                if not PATH_RE.match(tok): continue
                if any(c in tok for c in "<>*{}$") or "..." in tok or "/build/" in "/" + tok or tok.startswith("docs/wip/"): continue
                last = tok.rstrip("/").rsplit("/", 1)[-1]
                if "." not in last and not os.path.isdir(os.path.join(root, tok.rstrip("/"))):
                    # not a file name: only a directory is expected to exist; anything else (`graph/attention` as prose) is skipped
                    if tok.count("/") == 1 and not tok.endswith("/"): continue
                if tok.rstrip("/") in NOT_IN_TREE: continue
                n_paths += 1
                if a.list: print(f"{rel}:{ln}: path {tok}")
                if not any(os.path.exists(os.path.join(root, c.rstrip("/"))) for c in candidates(tok)):
                    bad.append(f"{rel}:{ln}: path does not exist: {tok}")
    # every guide is reachable from the index
    index = open(os.path.join(root, "docs", "index.md"), encoding="utf-8").read()
    for rel in files:
        name = os.path.basename(rel)
        if rel.startswith("docs/") and name != "index.md" and f"]({name}" not in index:
            bad.append(f"docs/index.md: no link to {rel}")
    for b in bad: print(b)
    print(f"docs links: {len(files) - len(SKIP_FILES & set(files))} files, {n_links} relative links, {n_paths} repository paths, {len(bad)} unresolved")
    return 1 if bad else 0

if __name__ == "__main__": sys.exit(main())

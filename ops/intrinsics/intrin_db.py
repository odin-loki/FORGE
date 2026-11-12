#!/usr/bin/env python3
"""Shared machinery for the Tier R database builders (build_<isa>_db.py).

A database is derived from the compiler's own intrinsic headers, never typed in:

  functions   clang -Xclang -ast-dump gives every FunctionDecl with its exact signature and target attribute
  macros      clang -E -dM gives the definitions; parameter types are read from the casts in the fully
              expanded body, the return type is deduced by the compiler (auto) and read back from the AST
  features    taken from the target attribute, or found by compiling the call with a minimal feature set and
              reading the "needs target feature" diagnostic
  validation  every candidate is compiled (parse + code generation) against the real target headers; whatever
              fails is dropped with a logged reason. Older/other compilers are then tried and the result is
              recorded per entry ("cc"), so the generator can gate instead of drop.

Nothing here is needed at build time: the checked-in JSON is the build input."""
import json, os, re, subprocess, sys, tempfile, itertools, collections

CLANG = os.environ.get("FORGE_RAW_CLANG", "clang++-23")
CLANG_OLD = os.environ.get("FORGE_RAW_CLANG_OLD") or "no-older-clang"   # optional: an older clang++ to record what it rejects
BASE = ["-std=c++20", "-ffreestanding", "-fno-color-diagnostics", "-fno-caret-diagnostics"]


def have(tool):
    from shutil import which
    return which(tool) is not None


def run(cmd, src, suffix=".cpp", want_stdout=False):
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as f:
        f.write(src); path = f.name
    try:
        p = subprocess.run(cmd + [path], capture_output=True, text=True, errors="replace")
        return (p.stdout if want_stdout else p.stderr), p.returncode, path
    finally:
        os.unlink(path)


def resource_dir(cc=None):
    return subprocess.run([cc or CLANG, "-print-resource-dir"], capture_output=True, text=True).stdout.strip() + "/include"


def cc_version(cc):
    return subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0].strip()


# ---------------------------------------------------------------- AST: functions
_FN = re.compile(r"^([ |`]*)[|`]-FunctionDecl 0x[0-9a-f]+ (?:prev 0x[0-9a-f]+ )?<[^>]*> \S+ (?:(?:implicit|used|referenced|invalid|constexpr|consteval) )*"
                 r"(\S+) '([^']*)'")
_AT = re.compile(r"^([ |`]*)[|`]-(\w+Attr) 0x[0-9a-f]+ <[^>]*>(?: Implicit)?(?: Inherited)? ?(.*)$")


def split_fn_type(t):
    """'R (P1, P2) quals' -> (R, [P...], quals)."""
    depth = 0; close = None
    # the parameter list is the last top-level parenthesised group
    i = len(t) - 1
    while i >= 0 and t[i] != ")": i -= 1
    close = i
    while i >= 0:
        if t[i] == ")": depth += 1
        elif t[i] == "(":
            depth -= 1
            if depth == 0: break
        i -= 1
    ret = t[:i].strip(); inner = t[i + 1:close]; quals = t[close + 1:].strip()
    ps, cur, d = [], "", 0
    for ch in inner:
        if ch in "(<[": d += 1
        elif ch in ")>]": d -= 1
        if ch == "," and d == 0: ps.append(cur.strip()); cur = ""
        else: cur += ch
    if cur.strip(): ps.append(cur.strip())
    if ps == ["void"]: ps = []
    return ret, ps, quals


def ast_functions(flags, src, cc=None, name_re=None):
    """Every function declared by `src`: [{name, ret, params, quals, attrs:{AttrKind: text}}] in order."""
    cmd = [cc or CLANG] + BASE + flags + ["-fsyntax-only", "-Xclang", "-ast-dump", "-w"]
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write(src); path = f.name
    out = []
    try:
        p1 = subprocess.Popen(cmd + [path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p2 = subprocess.Popen(["grep", "-aE", r"^[ |`]{0,8}[|`]-(FunctionDecl|[A-Za-z]+Attr) "], stdin=p1.stdout,
                              stdout=subprocess.PIPE, text=True, errors="replace")
        cur = None; cur_depth = -1
        for line in p2.stdout:
            m = _FN.match(line)
            if m:
                name = m.group(2)
                cur = None
                if name_re is None or name_re.match(name):
                    ret, ps, quals = split_fn_type(m.group(3))
                    cur = {"name": name, "ret": ret, "params": ps, "quals": quals, "attrs": {}}
                    cur_depth = len(m.group(1)); out.append(cur)
                continue
            m = _AT.match(line)
            if m and cur is not None and len(m.group(1)) == cur_depth + 2:
                cur["attrs"][m.group(2)] = m.group(3).strip().strip('"')
        p2.wait(); p1.wait()
    finally:
        os.unlink(path)
    return out


# ---------------------------------------------------------------- preprocessor: macros
def macros(flags, src, cc=None, name_re=None):
    """{name: (params or None, body)} for every macro visible after `src`."""
    out, _, _ = run([cc or CLANG] + BASE + flags + ["-E", "-dM", "-w"], src, want_stdout=True)
    res = {}
    for line in out.splitlines():
        m = re.match(r"#define (\w+)(\(([^)]*)\))? ?(.*)$", line)
        if not m: continue
        if name_re is not None and not name_re.match(m.group(1)): continue
        params = None if m.group(2) is None else [p.strip() for p in m.group(3).split(",") if p.strip()]
        res[m.group(1)] = (params, m.group(4))
    return res


def expand(flags, src, lines, cc=None):
    """Fully macro-expand each of `lines` (after including `src`); returns the list of expansions."""
    mark = "@@FORGE_RAW@@"
    text = src + "\n" + "\n".join(f"{mark} {i} {l}" for i, l in enumerate(lines)) + "\n"
    out, _, _ = run([cc or CLANG] + BASE + flags + ["-E", "-P", "-w"], text, want_stdout=True)
    res = [""] * len(lines)
    body = out.split(mark)
    for chunk in body[1:]:
        chunk = chunk.strip()
        m = re.match(r"(\d+)\s(.*)$", chunk, re.S)
        if m: res[int(m.group(1))] = " ".join(m.group(2).split())
    return res


def header_index(names, incdir=None, only=None):
    """{name: header file} by scanning the resource headers for the definition site."""
    incdir = incdir or resource_dir()
    want = set(names); res = {}
    files = sorted(f for f in os.listdir(incdir) if f.endswith(".h") and (only is None or only(f)))
    pat = re.compile(r"^(?:#\s*define\s+)?(\w+)\s*\(", re.M)
    weak = {}
    for f in files:
        text = open(os.path.join(incdir, f), errors="replace").read()
        for m in pat.finditer(text):
            n = m.group(1)
            if n in want and n not in res: res[n] = f
        for n in re.findall(r"\b(\w+)\s*\(", text):
            if n in want and n not in weak: weak[n] = f
    for n in want:
        if n not in res and n in weak: res[n] = weak[n]
    return res


# ---------------------------------------------------------------- probing
_ERR = re.compile(r"^(?:[^:\n]*):(\d+):(?:(\d+):)? (?:fatal )?error: (.*)$", re.M)


def base_flags(cc):
    """GCC: hosted (its <mm_malloc.h> needs malloc; the cross toolchains carry a libc) and its own diagnostics switches."""
    if "clang" in os.path.basename(cc): return BASE
    return [b for b in BASE if b not in ("-ffreestanding", "-fno-color-diagnostics", "-fno-caret-diagnostics")] + ["-fno-diagnostics-show-caret"]


_DIAG = re.compile(r"^([^:\n]+):(\d+):(?:(\d+):)? (fatal )?(error|note): (.*)$")
_FLAGS_OK = {}


def flags_ok(cc, flags, prelude):
    """Does the compiler accept these flags and the prelude at all? (cached)"""
    key = (cc, tuple(flags), prelude)
    if key not in _FLAGS_OK:
        base = base_flags(cc)
        _, rc, _ = run([cc] + base + list(flags) + ["-w", "-fsyntax-only"], prelude + "\n")
        _FLAGS_OK[key] = rc == 0
    return _FLAGS_OK[key]


def probe(flags, prelude, snippets, cc=None, codegen=False, chunk=3000, extra=(), cols=False, warn=("-w",)):
    """Compile one-line snippets; returns {index: [error messages]} for those that failed.

    Each snippet is on its own line so a diagnostic's line number identifies it. With codegen=True the
    snippets that parse are also lowered to assembly, which is where target-feature errors appear."""
    cc = cc or CLANG
    is_gcc = "clang" not in os.path.basename(cc)
    bad = {}
    if not flags_ok(cc, list(flags) + list(extra), prelude):
        raise RuntimeError(f"{cc} rejects flags/prelude: {' '.join(flags)}")
    idx = list(range(len(snippets)))
    npre = prelude.count("\n") + 1
    lim = ["-fmax-errors=0"] if is_gcc else ["-ferror-limit=0"]
    base = base_flags(cc)

    def attempt(live, stage):
        """True when the stage ran to completion (clean, or with every failure attributed to a snippet)."""
        # One namespace = one top-level declaration: clang stops generating code for further top-level
        # declarations after the first error, which would surface a single target-feature error per run.
        src = prelude + "\nnamespace forge_probe {\n" + "\n".join(snippets[i] for i in live) + "\n}\n"
        err, rc, path = run([cc] + base + list(flags) + list(extra) + lim + list(warn) + stage, src)
        found = False
        pending = None      # an error reported inside a header: the next note in our file says which snippet
        for line in err.splitlines():
            m = _DIAG.match(line)
            if not m: continue
            mine = m.group(1) == path
            if m.group(5) == "error" and not mine: pending = m.group(6); continue
            if not mine: continue
            if m.group(5) == "error": msg = m.group(6)
            elif pending is not None: msg = pending
            else: continue
            pending = None
            ln = int(m.group(2)) - npre - 2
            if 0 <= ln < len(live):
                bad.setdefault(live[ln], []).append((int(m.group(3) or 0), msg) if cols else msg); found = True
        if rc != 0 and not found:
            # nothing attributable (a diagnostic inside a header, or a compiler crash): bisect to the culprit
            if len(live) == 1:
                tail = [l for l in err.splitlines() if "error" in l.lower()]
                msg = "compiler crash or unattributable error: " + (tail[0][:160] if tail else "?")
                bad.setdefault(live[0], []).append((0, msg) if cols else msg)
            else:
                h = len(live) // 2
                run_stage(live[:h], stage); run_stage(live[h:], stage)
            return True
        return rc == 0

    def run_stage(live, stage):
        # a codegen error hides the functions after it only within one function, but be safe: loop until clean
        for _ in range(40):
            live = [i for i in live if i not in bad]
            if not live or attempt(live, stage) or stage == ["-fsyntax-only"]: return
        for i in live: bad.setdefault(i, []).append((0, "code generation did not come out clean") if cols else "code generation did not come out clean")

    for s in range(0, len(idx), chunk):
        part = idx[s:s + chunk]
        for stage in (["-fsyntax-only"], ["-S", "-o", "/dev/null"]) if codegen else (["-fsyntax-only"],):
            run_stage(part, stage)
    return bad


def deduce_returns(flags, prelude, defs, cc=None):
    """defs: {fn_name: 'auto fn_name(params) { return expr; }'} -> {fn_name: return type as the compiler spells it}."""
    src = prelude + "\n" + "\n".join(defs.values()) + "\n"
    fns = ast_functions(flags, src, cc=cc, name_re=re.compile(r"^forge_p_\d+"))
    return {f["name"]: f["ret"] for f in fns}


def feature_alternatives(expr):
    """clang's required-feature expression ("a,b|c", "(sve,(sve2|sme))|sme": ',' = and, '|' = or, evaluated left
    to right, parentheses group) as a list of alternatives, each a list of features."""
    pos = 0

    def parse():
        nonlocal pos
        cur = None      # DNF so far: list of sets
        op = None
        while pos < len(expr):
            ch = expr[pos]
            if ch == ")": break
            if ch in ",|": op = ch; pos += 1; continue
            if ch == " ": pos += 1; continue
            if ch == "(":
                pos += 1; term = parse(); pos += 1
            else:
                m = re.match(r"[\w.+\-]+", expr[pos:]); pos += len(m.group(0)); term = [frozenset([m.group(0)])]
            if cur is None: cur = term
            elif op == "|": cur = cur + term
            else: cur = [a | b for a in cur for b in term]
        return cur or [frozenset()]
    out = []
    for alt in parse():
        if alt not in out: out.append(alt)
    return [sorted(a) for a in out]


# ---------------------------------------------------------------- snippets
def is_imm(t):
    """An immediate operand is spelled as a const by-value parameter: "const int", "const svpattern"."""
    return t.startswith("const ") and "*" not in t and "&" not in t


def call_snippet(e, n, callee=None, attrs="", want_cols=False):
    """A function with the entry's exact signature that calls the intrinsic (the body of the wrapper)."""
    args, call, k = [], [], 0
    imm = e.get("imm", [])
    for i, t in enumerate(e["params"]):
        if is_imm(t):
            v = str(imm[k] if k < len(imm) else 0); k += 1
            call.append(v if t == "const int" else f"({t[6:]}){v}")
        else:
            args.append(f"{t} a{i}"); call.append(f"a{i}")
    r = "" if e["ret"] == "void" else "return "
    head = f"{e['ret']} forge_p_{n}({', '.join(args)}) {attrs} {{ {r}{callee or e['name']}("
    if want_cols:
        cols, c = [], len(head) + 1
        for a in call: cols.append((c, c + len(a))); c += len(a) + 2
        return head + ", ".join(call) + "); }", cols
    return head + ", ".join(call) + "); }"


_RANGE = re.compile(r"outside the valid range \[(-?\d+), (-?\d+)\]")
_CONST = re.compile(r"constant integer|constant expression|must be a constant|not a constant|integral constant|must be an immediate|must be constant|no matching function for call")


def resolve_immediates(flags, prelude, entries, int_like, attrs_of=lambda e: "", warn=("-w",), rounds=20, enum_like=lambda t: False, budget=120):
    """Find which integer parameters the compiler only accepts as constants, and a valid value for each.

    Breadth-first over assignments, starting with every integer parameter passed at run time. A "must be a
    constant" diagnostic branches on which run-time parameter becomes an immediate (the diagnostic points at
    the call, not the argument); a range diagnostic moves the immediate that holds the rejected value into
    the range. The first assignment that compiles has the fewest immediates. Parameters already spelled
    "const T" stay immediates. Returns {index: reason} for the entries that never compile."""
    def initial(e):
        st, k = {}, 0
        for j, t in enumerate(e["params"]):
            if is_imm(t): st[j] = e.get("imm", [])[k] if k < len(e.get("imm", [])) else 0; k += 1
            elif int_like(t): st[j] = None
        return st

    def view(e, st):
        ps, imm = [], []
        for j, t in enumerate(e["params"]):
            if st.get(j) is not None:
                ps.append(t if is_imm(t) else ("const " + t.replace("enum ", "") if enum_like(t) else "const int")); imm.append(st[j])
            else: ps.append(t)
        return dict(e, params=ps, imm=imm)

    def key(st): return tuple(sorted(st.items(), key=lambda kv: kv[0]))

    frontier = {i: [initial(e)] for i, e in enumerate(entries)}
    seen = {i: {key(frontier[i][0])} for i in frontier}
    result, last = {}, {}
    for _ in range(rounds):
        batch = [(i, st) for i, sts in frontier.items() for st in sts]
        if not batch: break
        sn = [call_snippet(view(entries[i], st), n, attrs=attrs_of(entries[i]), want_cols=True) for n, (i, st) in enumerate(batch)]
        bad = probe(flags, prelude, [t for t, _ in sn], cols=True, warn=warn)
        nxt = collections.defaultdict(list)
        for n, (i, st) in enumerate(batch):
            if i in result: continue
            if n not in bad: result[i] = st; continue
            last[i] = bad[n][0][1]
            kids = []
            for col, msg in bad[n]:
                at = next((a for a, (lo, hi) in enumerate(sn[n][1]) if lo <= col < hi), None)
                at = at if at in st else None
                r = _RANGE.search(msg)
                if r:
                    lo, hi = int(r.group(1)), int(r.group(2))
                    v = re.search(r"argument value (-?\d+)", msg)
                    js = [at] if at is not None and st[at] is not None else [j for j, x in st.items() if x is not None and (v is None or x == int(v.group(1)))]
                    for j in js:
                        for nv in (lo, hi):
                            if nv != st[j]: kids.append({**st, j: nv}); break
                    continue
                if _CONST.search(msg):
                    js = [at] if at is not None and st[at] is None else [j for j, x in st.items() if x is None]
                    kids += [{**st, j: 0} for j in js]
                if any(x is not None for x in st.values()) and re.search(r"argument|invalid|immediate|rounding|multiple|index|lane|no matching function", msg):
                    named = [int(v) for v in re.findall(r"(?<![\w.])(\d+)(?![\w.])", msg)][:6]      # "should be the value 90 or 270"
                    for j, x in st.items():
                        if x is None: continue
                        for nv in named:
                            if nv != x: kids.append({**st, j: nv})
                        for nv in (1, 2, 4, 8, 16, 3, 32, 64):
                            if nv > x: kids.append({**st, j: nv}); break
            for kid in kids:
                if key(kid) not in seen[i] and len(seen[i]) < budget: seen[i].add(key(kid)); nxt[i].append(kid)
        frontier = {i: sts for i, sts in nxt.items() if i not in result}
    failed = {}
    for i, e in enumerate(entries):
        if i not in result: failed[i] = last.get(i, "no valid immediate found"); continue
        v = view(e, result[i]); e["params"] = v["params"]
        if v["imm"]: e["imm"] = v["imm"]
        elif "imm" in e: del e["imm"]
    return failed


# ---------------------------------------------------------------- output
def write_db(path, meta, entries, dropped):
    """One entry per line: diffs stay readable and the file stays a fraction of indent=1 JSON."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("{\n")
        for k, v in meta.items():
            f.write(f" {json.dumps(k)}: {json.dumps(v)},\n")
        f.write(' "intrinsics": [\n')
        f.write(",\n".join("  " + json.dumps(e, separators=(",", ":")) for e in entries))
        f.write("\n ]\n}\n")
    if dropped is None: return
    with open(os.path.splitext(path)[0] + ".dropped.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Candidates left out of {os.path.basename(path)} by its build script, with the reason.\n")
        by = collections.Counter(r.split(":")[0] for _, r in dropped)
        for k, c in sorted(by.items()): f.write(f"# {c:6d}  {k}\n")
        for n, r in sorted(set(dropped)): f.write(f"{n}\t{r}\n")
    print(f"{os.path.basename(path)}: {len(entries)} intrinsics, {len(set(n for n, _ in dropped))} dropped")


def classify(entries, snippets_of, compilers, codegen=False):
    """Record which other compilers accept each entry. compilers: [(key, cc, flags_of(features))]."""
    for key, cc, flags_of in compilers:
        if not have(cc): print(f"  ({cc} not installed: '{key}' not recorded)"); continue
        groups = collections.defaultdict(list)
        for i, e in enumerate(entries): groups[tuple(e["features"])].append(i)
        for feats, ids in groups.items():
            prelude, fl = flags_of(feats)
            if fl is None:
                for i in ids: entries[i].setdefault("no", []).append(key)
                continue
            try:
                bad = probe(fl, prelude, [snippets_of(entries[i], i) for i in ids], cc=cc, codegen=codegen)
            except RuntimeError as ex:
                print(f"  ({key}: {ex}: {len(ids)} entries marked unavailable)")
                bad = {k: ["header"] for k in range(len(ids))}
            for k in bad: entries[ids[k]].setdefault("no", []).append(key)

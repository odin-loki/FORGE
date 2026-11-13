#!/usr/bin/env python3
"""Structure-aware seeds for fuzz_aot (precompiled plan blobs, graph/include/forge/graph/aot.hpp).

A plan blob is a directory of entries; each entry is a graph (nodes with op, inputs, attributes, shape), its outputs,
a memory plan (one buffer slot per node) and the step list. Almost every byte is cross-checked against the others
(input ids against node order, slot sizes against shapes, arena offsets against the arena, kernels against ops), so
a byte-level mutation of the one valid seed is nearly always rejected by the first count or bounds check and the
checks behind it (check_memory, check_state, shape inference, kernel lookup) are never reached.

This script decodes the valid seed that test_aot writes (fuzz/corpus/aot/seed.fplan), changes ONE field at a time in
the decoded form and encodes the result again with a consistent directory, so each seed is well-formed up to the one
field under test. It also writes a few small plans of its own. Standard library only.

    aot_seeds.py [corpus-dir]        (default: fuzz/corpus/aot next to this file; writes gen_*.fplan there)
    aot_seeds.py --dump FILE         print a decoded blob
"""
import copy, os, struct, sys

MAGIC = b"FORGEAOT"
# Ops for the small plans this script writes itself (names as in ops::info; the kernel of op X is taken from the
# seed when the seed uses X, else guessed as "lane.X": a wrong guess is still a seed for the kernel lookup).
BINARY = [b"add", b"sub", b"mul", b"div", b"maximum", b"minimum", b"pow", b"matmul", b"matmul_t", b"silu_mul"]
OPS = BINARY + [b"neg", b"abs", b"sqrt", b"exp", b"tanh", b"sigmoid", b"silu", b"gelu", b"relu", b"softmax", b"rms_norm", b"layer_norm",
                b"reduce_sum", b"argmax", b"transpose", b"reshape", b"concat", b"slice", b"gather", b"rope", b"cumsum", b"l2_normalize",
                b"conv1d", b"max_pool2d", b"pad", b"topk"]


class R:
    def __init__(self, b): self.b, self.at = b, 0
    def take(self, n):
        if n > len(self.b) - self.at: raise ValueError("truncated")
        v = self.b[self.at:self.at + n]; self.at += n; return v
    def u8(self): return self.take(1)[0]
    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def i64(self): return struct.unpack("<q", self.take(8))[0]
    def f64(self): return struct.unpack("<d", self.take(8))[0]
    def str(self): return self.take(self.u32())


class W:
    def __init__(self): self.b = bytearray()
    def u8(self, v): self.b.append(v & 0xff)
    def u32(self, v): self.b += struct.pack("<I", v & 0xffffffff)
    def u64(self, v): self.b += struct.pack("<Q", v & 0xffffffffffffffff)
    def i64(self, v): self.b += struct.pack("<q", v) if -2**63 <= v < 2**63 else struct.pack("<Q", v & 0xffffffffffffffff)
    def f64(self, v): self.b += struct.pack("<d", v)
    def str(self, s): self.u32(len(s)); self.b += s
    def raw(self, b): self.b += b


def dec_entry(b):
    r = R(b); e = {"nodes": [], "outputs": [], "slots": [], "steps": []}
    for _ in range(r.u32()):
        n = {"kind": r.u8(), "op": r.str()}
        n["inputs"] = [r.u32() for _ in range(r.u32())]
        n["attrs"] = []
        for _ in range(r.u32()):
            k = r.str(); t = r.u8()
            v = r.f64() if t == 0 else [r.i64() for _ in range(r.u32())] if t == 1 else r.str()
            n["attrs"].append([k, t, v])
        n["shape"] = [r.i64() for _ in range(r.u32())]
        n["name"] = r.str(); n["wk"] = r.u8(); n["codec"] = r.str()
        n["const"] = r.take(4 * r.u64()) if n["kind"] == 2 else None   # node_kind::constant
        e["nodes"].append(n)
    for _ in range(r.u32()): e["outputs"].append([r.str(), r.u32()])
    e["arena"], e["align"], e["naive"], e["in_place"], e["aliases"] = r.u64(), r.u64(), r.u64(), r.u32(), r.u32()
    for _ in e["nodes"]: e["slots"].append([r.u8(), r.u64(), r.u64(), r.u32()])   # kind, offset, bytes, root
    for _ in range(r.u32()): e["steps"].append([r.u8(), r.u32(), r.str(), r.u8()])   # kind, out, kernel, fuse
    if r.at != len(b): raise ValueError("trailing bytes")
    return e


def enc_entry(e):
    w = W(); w.u32(e.get("n_nodes", len(e["nodes"])))
    for n in e["nodes"]:
        w.u8(n["kind"]); w.str(n["op"]); w.u32(len(n["inputs"]))
        for i in n["inputs"]: w.u32(i)
        w.u32(len(n["attrs"]))
        for k, t, v in n["attrs"]:
            w.str(k); w.u8(t)
            if t == 0: w.f64(v)
            elif t == 1:
                w.u32(len(v))
                for x in v: w.i64(x)
            else: w.str(v)
        w.u32(len(n["shape"]))
        for d in n["shape"]: w.i64(d)
        w.str(n["name"]); w.u8(n["wk"]); w.str(n["codec"])
        if n["const"] is not None: w.u64(n.get("const_n", len(n["const"]) // 4)); w.raw(n["const"])
    w.u32(len(e["outputs"]))
    for nm, v in e["outputs"]: w.str(nm); w.u32(v)
    w.u64(e["arena"]); w.u64(e["align"]); w.u64(e["naive"]); w.u32(e["in_place"]); w.u32(e["aliases"])
    for k, off, nb, root in e["slots"]: w.u8(k); w.u64(off); w.u64(nb); w.u32(root)
    w.u32(len(e["steps"]))
    for k, out, low, fuse in e["steps"]: w.u8(k); w.u32(out); w.str(low); w.u8(fuse)
    return bytes(w.b)


def dec_blob(b):
    if b[:8] != MAGIC: raise ValueError("not a plan blob")
    r = R(b[:-32]); r.at = 8
    h = {"version": r.u32()}
    n = r.u32()
    h["model"], h["pipeline"], h["forge"], h["simd"] = r.take(32), r.u32(), r.str(), r.str()
    h["entries"] = []
    for _ in range(n):
        t = {"key": r.take(32), "seq": r.i64(), "all_rows": r.u8(), "kv": r.str(), "device": r.str()}
        off, size = r.u64(), r.u64()
        t["entry"] = dec_entry(b[off:off + size]); h["entries"].append(t)
    h["hash"] = b[-32:]
    return h


def enc_blob(h):
    """Directory offsets and sizes are recomputed; the trailing hash is kept as it was (fuzz_aot does not require it)."""
    w = W(); w.raw(MAGIC); w.u32(h["version"]); w.u32(len(h["entries"])); w.raw(h["model"]); w.u32(h["pipeline"]); w.str(h["forge"]); w.str(h["simd"])
    bodies = [t.get("raw") if t.get("raw") is not None else enc_entry(t["entry"]) for t in h["entries"]]
    off = len(w.b) + sum(32 + 8 + 1 + 4 + len(t["kv"]) + 4 + len(t["device"]) + 16 for t in h["entries"])
    for t, body in zip(h["entries"], bodies):
        w.raw(t["key"]); w.i64(t["seq"]); w.u8(t["all_rows"]); w.str(t["kv"]); w.str(t["device"])
        w.u64(t.get("offset", off)); w.u64(t.get("size", len(body))); off += len(body)
    for body in bodies: w.raw(body)
    w.raw(h["hash"])
    return bytes(w.b)


def numel(shape):
    n = 1
    for d in shape: n *= d
    return n


def small_plan(base, nodes, steps, outputs):
    """A plan of this script's own: every op node gets its own 64-aligned arena buffer, inputs included."""
    e = {"nodes": [], "outputs": outputs, "slots": [], "steps": steps, "align": 64, "in_place": 0, "aliases": 0}
    off = 0
    for kind, op, inputs, shape, name, attrs in nodes:
        const = struct.pack("<%df" % numel(shape), *([0.5] * numel(shape))) if kind == 2 else None
        e["nodes"].append({"kind": kind, "op": op, "inputs": inputs, "attrs": attrs, "shape": shape, "name": name,
                           "wk": 1 if kind == 1 else 0, "codec": b"", "const": const})
        nb = numel(shape) * 4
        if kind in (1, 2, 4): e["slots"].append([1, 0, nb, 0xffffffff])   # weight / constant / state_append: external
        else: e["slots"].append([0, off, nb, 0xffffffff]); off += (nb + 63) // 64 * 64
    e["arena"] = off; e["naive"] = off
    h = copy.deepcopy(base); t = h["entries"][0]; t["entry"] = e; h["entries"] = [t]
    return h


def variants(base):
    """(name, blob dict) pairs: one field of the first entry changed per variant."""
    out = []
    def var(name, fn):
        h = copy.deepcopy(base)
        try: fn(h, h["entries"][0]["entry"])
        except (IndexError, KeyError): return
        out.append((name, h))
    e0 = base["entries"][0]["entry"]
    N = len(e0["nodes"])
    ops = sorted({n["op"] for n in e0["nodes"] if n["op"]})
    kernels = sorted({s[2] for s in e0["steps"] if s[2]})
    op_nodes = [i for i, n in enumerate(e0["nodes"]) if n["kind"] == 3]
    arena = [i for i, s in enumerate(e0["slots"]) if s[0] == 0]
    alias = [i for i, s in enumerate(e0["slots"]) if s[0] == 2]   # buffer_slot::kind_t: arena 0, external 1, alias 2
    ext = [i for i, s in enumerate(e0["slots"]) if s[0] == 1]
    first, mid, last = (op_nodes + [0])[0], (op_nodes + [0])[len(op_nodes) // 2], (op_nodes + [0])[-1]

    # ---- graph
    for tag, i in (("first", first), ("mid", mid), ("last", last)):
        var("node_%s_shape_zero" % tag, lambda h, e, i=i: e["nodes"][i]["shape"].__setitem__(0, 0))
        var("node_%s_shape_grow" % tag, lambda h, e, i=i: e["nodes"][i]["shape"].__setitem__(-1, e["nodes"][i]["shape"][-1] + 1))
        var("node_%s_shape_rank" % tag, lambda h, e, i=i: e["nodes"][i]["shape"].append(1))
        var("node_%s_no_inputs" % tag, lambda h, e, i=i: e["nodes"][i]["inputs"].clear())
        var("node_%s_input_dup" % tag, lambda h, e, i=i: e["nodes"][i]["inputs"].append(e["nodes"][i]["inputs"][0]))
        var("node_%s_input_swap" % tag, lambda h, e, i=i: e["nodes"][i]["inputs"].reverse())
        var("node_%s_input_zero" % tag, lambda h, e, i=i: e["nodes"][i]["inputs"].__setitem__(0, 0))
        var("node_%s_no_attrs" % tag, lambda h, e, i=i: e["nodes"][i]["attrs"].clear())
        for k, op in enumerate(ops[:6]):
            if op != e0["nodes"][i]["op"]: var("node_%s_op_%s" % (tag, op.decode("ascii", "replace")), lambda h, e, i=i, op=op: e["nodes"][i].__setitem__("op", op))
    var("node_kind_state_op", lambda h, e: e["nodes"][mid].update(kind=6, name=b"slot"))
    var("node_kind_state_append", lambda h, e: e["nodes"][mid].update(kind=4, name=b"slot", inputs=e["nodes"][mid]["inputs"][:1]))
    var("node_kind_input", lambda h, e: e["nodes"][mid].update(kind=0, op=b"", inputs=[]))
    var("shape_huge", lambda h, e: e["nodes"][last].__setitem__("shape", [1 << 20, 1 << 20]))
    var("shape_negative", lambda h, e: e["nodes"][last]["shape"].__setitem__(0, -1))
    var("shape_scalar", lambda h, e: e["nodes"][last].__setitem__("shape", []))
    for i, n in enumerate(e0["nodes"]):
        for j, (k, t, v) in enumerate(n["attrs"]):
            if t == 1 and len(out) < 400:
                var("attr_%d_%s_neg" % (i, k.decode("ascii", "replace")), lambda h, e, i=i, j=j: e["nodes"][i]["attrs"][j].__setitem__(2, [-x - 1 for x in e["nodes"][i]["attrs"][j][2]]))
                var("attr_%d_%s_big" % (i, k.decode("ascii", "replace")), lambda h, e, i=i, j=j: e["nodes"][i]["attrs"][j].__setitem__(2, [1 << 40 for _ in e["nodes"][i]["attrs"][j][2]]))
                var("attr_%d_%s_empty" % (i, k.decode("ascii", "replace")), lambda h, e, i=i, j=j: e["nodes"][i]["attrs"][j].__setitem__(2, []))
            elif t == 0:
                var("attr_%d_%s_nan" % (i, k.decode("ascii", "replace")), lambda h, e, i=i, j=j: e["nodes"][i]["attrs"][j].__setitem__(2, float("nan")))
                var("attr_%d_%s_huge" % (i, k.decode("ascii", "replace")), lambda h, e, i=i, j=j: e["nodes"][i]["attrs"][j].__setitem__(2, 1e300))
                var("attr_%d_%s_as_text" % (i, k.decode("ascii", "replace")), lambda h, e, i=i, j=j: e["nodes"][i]["attrs"][j].__setitem__(slice(1, 3), [2, b"x"]))
    # ---- outputs
    var("output_none", lambda h, e: e["outputs"].clear())
    var("output_twice", lambda h, e: e["outputs"].append(list(e["outputs"][0])))
    var("output_of_input", lambda h, e: e["outputs"][0].__setitem__(1, 0))
    var("output_of_mid", lambda h, e: e["outputs"][0].__setitem__(1, mid))
    # ---- memory plan
    for tag, lst in (("arena", arena), ("alias", alias), ("ext", ext)):
        if not lst: continue
        i = lst[len(lst) // 2]
        for k in range(3):
            if k != e0["slots"][i][0]: var("slot_%s_kind_%d" % (tag, k), lambda h, e, i=i, k=k: e["slots"][i].__setitem__(0, k))
        var("slot_%s_offset_0" % tag, lambda h, e, i=i: e["slots"][i].__setitem__(1, 0))
        var("slot_%s_offset_unaligned" % tag, lambda h, e, i=i: e["slots"][i].__setitem__(1, e["slots"][i][1] + 4))
        var("slot_%s_offset_end" % tag, lambda h, e, i=i: e["slots"][i].__setitem__(1, e["arena"]))
        var("slot_%s_offset_past" % tag, lambda h, e, i=i: e["slots"][i].__setitem__(1, e["arena"] + 64))
        var("slot_%s_offset_wrap" % tag, lambda h, e, i=i: e["slots"][i].__setitem__(1, (1 << 64) - 64))
        var("slot_%s_bytes_less" % tag, lambda h, e, i=i: e["slots"][i].__setitem__(2, e["slots"][i][2] - 4))
        var("slot_%s_bytes_wrap" % tag, lambda h, e, i=i: e["slots"][i].__setitem__(2, (1 << 64) - 4))
        var("slot_%s_root_self" % tag, lambda h, e, i=i: e["slots"][i].__setitem__(3, i))
        var("slot_%s_root_0" % tag, lambda h, e, i=i: e["slots"][i].__setitem__(3, 0))
        var("slot_%s_root_prev" % tag, lambda h, e, i=i: e["slots"][i].__setitem__(3, max(i - 1, 0)))
    if len(arena) > 1:   # two live buffers at one offset
        var("slots_overlap", lambda h, e: e["slots"][arena[1]].__setitem__(1, e["slots"][arena[0]][1]))
        var("slots_all_at_0", lambda h, e: [e["slots"][i].__setitem__(1, 0) for i in arena])
    var("arena_zero", lambda h, e: e.__setitem__("arena", 0))
    var("arena_minus_64", lambda h, e: e.__setitem__("arena", max(e["arena"] - 64, 0)))
    var("arena_huge", lambda h, e: e.__setitem__("arena", 1 << 39))
    var("align_1", lambda h, e: e.__setitem__("align", 1))
    var("align_0", lambda h, e: e.__setitem__("align", 0))
    var("slots_one_missing", lambda h, e: e["slots"].pop())
    # ---- steps
    S = len(e0["steps"])
    var("steps_none", lambda h, e: e["steps"].clear())
    var("steps_drop_first", lambda h, e: e["steps"].pop(0))
    var("steps_drop_last", lambda h, e: e["steps"].pop())
    var("steps_reversed", lambda h, e: e["steps"].reverse())
    var("steps_twice", lambda h, e: e["steps"].extend(copy.deepcopy(e["steps"])))
    for tag, i in (("first", 0), ("mid", S // 2), ("last", S - 1)):
        var("step_%s_out_0" % tag, lambda h, e, i=i: e["steps"][i].__setitem__(1, 0))
        var("step_%s_out_last" % tag, lambda h, e, i=i: e["steps"][i].__setitem__(1, N - 1))
        var("step_%s_no_kernel" % tag, lambda h, e, i=i: e["steps"][i].__setitem__(2, b""))
        for k in range(3):
            if k != e0["steps"][i][0]: var("step_%s_kind_%d" % (tag, k), lambda h, e, i=i, k=k: e["steps"][i].__setitem__(0, k))
        for f in (1, 2, 3, 255):
            if f != e0["steps"][i][3]: var("step_%s_fuse_%d" % (tag, f), lambda h, e, i=i, f=f: e["steps"][i].__setitem__(3, f))
        for kn in kernels[:8]:
            if kn != e0["steps"][i][2]: var("step_%s_kernel_%s" % (tag, kn.decode("ascii", "replace")), lambda h, e, i=i, kn=kn: e["steps"][i].__setitem__(2, kn))
    # ---- directory
    var("dir_two_entries", lambda h, e: h["entries"].append(copy.deepcopy(h["entries"][0])))
    var("dir_entry_empty", lambda h, e: h["entries"][0].__setitem__("raw", b""))
    var("dir_entry_count_only", lambda h, e: h["entries"][0].__setitem__("raw", struct.pack("<I", 0) * 2 + struct.pack("<QQQII", 0, 64, 0, 0, 0) + struct.pack("<I", 0)))
    var("dir_size_short", lambda h, e: h["entries"][0].__setitem__("size", 40))
    var("dir_offset_0", lambda h, e: h["entries"][0].__setitem__("offset", 0))
    var("dir_node_count_plus", lambda h, e: e.__setitem__("n_nodes", N + 1))
    var("dir_node_count_minus", lambda h, e: e.__setitem__("n_nodes", N - 1))
    # ---- plans of this script's own (kernel names taken from the seed)
    kernel_of = {n["op"]: s[2] for s in e0["steps"] for n in [e0["nodes"][s[1]]] if s[2] and n["op"]}
    def kern(op): return kernel_of.get(op, b"lane." + op)
    w88 = [1, b"", [], [8, 8], b"w", []]
    for op in OPS:
        nm = op.decode("ascii")
        if op in BINARY:   # x (1x8) with a weight (8x8), and a chain x -> op(x, x) -> op(., x)
            nodes = [[0, b"", [], [1, 8], b"x", []], w88, [3, op, [0, 1], [1, 8], b"", []]]
            out.append(("own_%s_x_w" % nm, small_plan(base, nodes, [[0, 2, kern(op), 0]], [[b"y", 2]])))
            nodes = [[0, b"", [], [1, 8], b"x", []], [3, op, [0, 0], [1, 8], b"", []], [3, op, [1, 0], [1, 8], b"", []]]
            out.append(("own_%s_chain" % nm, small_plan(base, nodes, [[0, 1, kern(op), 0], [0, 2, kern(op), 0]], [[b"y", 2]])))
        else:
            nodes = [[0, b"", [], [1, 8], b"x", []], [3, op, [0], [1, 8], b"", []], [3, op, [1], [1, 8], b"", []]]
            out.append(("own_%s_chain" % nm, small_plan(base, nodes, [[0, 1, kern(op), 0], [0, 2, kern(op), 0]], [[b"y", 2]])))
    nodes = [[0, b"", [], [1, 8], b"x", []], [2, b"", [], [1, 8], b"c", []], [4, b"", [0], [1, 8], b"kv", []], [6, b"", [0, 2], [1, 8], b"kv2", []]]
    out.append(("own_state", small_plan(base, nodes, [[1, 2, b"", 0], [2, 3, b"", 0]], [[b"y", 3]])))
    return out


def main(argv):
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus", "aot")
    if len(argv) > 2 and argv[1] == "--dump":
        import pprint
        pprint.pprint(dec_blob(open(argv[2], "rb").read()), width=160)
        return 0
    if len(argv) > 1: here = argv[1]
    seed = open(os.path.join(here, "seed.fplan"), "rb").read()
    base = dec_blob(seed)
    if enc_blob(base) != seed: raise SystemExit("aot_seeds.py: the seed does not survive decode + encode (format changed?)")
    for f in os.listdir(here):
        if f.startswith("gen_"): os.remove(os.path.join(here, f))
    seen = {seed}
    n = 0
    for name, h in variants(base):
        b = enc_blob(h)
        if b in seen: continue
        seen.add(b); n += 1
        open(os.path.join(here, "gen_%s.fplan" % "".join(c if c.isalnum() or c in "_-" else "_" for c in name)), "wb").write(b)
    print("aot_seeds.py: %d seeds in %s" % (n, here))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

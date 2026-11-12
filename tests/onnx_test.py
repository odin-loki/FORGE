#!/usr/bin/env python3
"""ONNX import: build graphs with the `onnx` package, import them with `forge import`, run them with
`forge run --input`, and compare every output with onnx's reference evaluator (onnx.reference.ReferenceEvaluator)
and, when it is installed, with onnxruntime. The first two cases also keep their hand-written numpy references.
Also: external tensor data (and what a hostile location does), integer inputs, several symbolic dimensions, shape
subgraphs that must be folded, and the error text for what the importer refuses."""
import os, subprocess, sys, tempfile, math, shutil
import numpy as np
import onnx
from onnx import helper as H, TensorProto as TP, numpy_helper as NH
from onnx.reference import ReferenceEvaluator
try:
    import onnxruntime as ort
except Exception:
    ort = None
forge = sys.argv[1]
runner = os.environ.get("FORGE_RUNNER", "").split()
d = tempfile.mkdtemp()
rng = np.random.default_rng(7)
cases = 0
def run(*a, ok=True):
    r = subprocess.run(runner + [forge, *a], capture_output=True, text=True, errors="replace")
    if (r.returncode == 0) != ok: sys.exit(f"unexpected exit {r.returncode}: {a}\n{r.stdout}{r.stderr}")
    return r.stdout + r.stderr
def W(name, *shape, scale=0.5): return NH.from_array((rng.standard_normal(shape) * scale).astype(np.float32), name)
def C(name, arr, dtype=np.int64): return NH.from_array(np.array(arr, dtype), name)
def vi(name, shape, t=TP.FLOAT): return H.make_tensor_value_info(name, t, shape)
def model_of(graph, opset): return H.make_model(graph, opset_imports=[H.make_opsetid("", opset)], producer_name="forge-test", ir_version=10)

def run_forge(name, path, feeds, outputs, dims=()):
    out = os.path.join(d, name + ".fpk")
    run("import", path, "-o", out)
    args = []
    for k, v in feeds.items():
        if v.dtype.kind in "iub":   # integer and bool inputs go in as int64 files
            f = os.path.join(d, f"{name}.{k}.i64"); v.astype("<i8").tofile(f)
        else:
            f = os.path.join(d, f"{name}.{k}.bin"); v.astype("<f4").tofile(f)
        args += ["--input", f"{k}={f}"]
    for kv in dims: args += ["--dim", kv]
    run("run", out, *args, "--out", os.path.join(d, name))
    return [np.fromfile(os.path.join(d, f"{name}.out_{''.join(c if c.isalnum() else '_' for c in o)}.f32"), "<f4") for o in outputs]

def check(name, graph, feeds, ref=None, opset=17, tol=1e-4, dims=(), oracle=True):
    """Import + run, then compare with the reference evaluator, onnxruntime, and `ref` (numpy) when given."""
    global cases
    m = onnx.shape_inference.infer_shapes(model_of(graph, opset))   # fills in the output shapes left as None above
    onnx.checker.check_model(m)
    p = os.path.join(d, name + ".onnx"); onnx.save(m, p)
    outs = [o.name for o in graph.output]
    got = run_forge(name, p, feeds, outs, dims)
    oracles = {}
    if oracle:
        oracles["onnx reference"] = ReferenceEvaluator(m).run(None, feeds)
        if ort is not None:
            try: oracles["onnxruntime"] = ort.InferenceSession(p, providers=["CPUExecutionProvider"]).run(None, feeds)
            except Exception as e: print(f"  (onnxruntime did not run {name}: {str(e).splitlines()[0][:100]})")
    if ref is not None: oracles["numpy"] = ref
    if not oracles: sys.exit(f"{name}: nothing to compare with")
    worst = 0.0
    for who, vals in oracles.items():
        for o, g, r in zip(outs, got, vals):
            r = np.asarray(r, dtype=np.float64).ravel()
            err = float(np.max(np.abs(g - r))) if g.size == r.size and g.size else (0.0 if g.size == r.size else math.inf)
            worst = max(worst, err)
            if err > tol: sys.exit(f"{name}: output {o} differs from {who}: max err {err:.3e} ({g.size} vs {r.size} values)")
    cases += 1
    print(f"{name}: {len(outs)} output(s), max err {worst:.2e} vs {', '.join(oracles)}")

def refused(name, graph, *needles, opset=17):
    global cases
    p = os.path.join(d, name + ".onnx"); onnx.save(model_of(graph, opset), p)
    msg = run("import", p, "-o", os.path.join(d, name + ".fpk"), ok=False)
    for s in needles:
        if s not in msg: sys.exit(f"{name}: the error does not mention '{s}': {msg.strip()}")
    cases += 1
    print(f"{name}: refused: {msg.strip().splitlines()[-1][:150]}")

# ---- 1. MLP classifier with a symbolic batch dim: Gemm(transB) Relu MatMul Add Mul(scalar) Softmax, plus Tanh/Sigmoid branch
w1, b1, w2, b2 = W("w1", 16, 8), W("b1", 16), W("w2", 16, 4), W("b2", 4)
half = NH.from_array(np.array(0.5, np.float32), "half")
g = H.make_graph([
    H.make_node("Gemm", ["x", "w1", "b1"], ["h"], transB=1),
    H.make_node("Relu", ["h"], ["hr"]),
    H.make_node("MatMul", ["hr", "w2"], ["z"]),
    H.make_node("Add", ["z", "b2"], ["z2"]),
    H.make_node("Mul", ["z2", "half"], ["z3"]),
    H.make_node("Softmax", ["z3"], ["probs"], axis=-1),
    H.make_node("Tanh", ["z2"], ["t"]), H.make_node("Sigmoid", ["t"], ["sg"]),
], "mlp", [vi("x", ["batch", 8])], [vi("probs", ["batch", 4]), vi("sg", ["batch", 4])], [w1, b1, w2, b2, half])
for batch in (3, 5):
    x = rng.standard_normal((batch, 8)).astype(np.float32)
    A = lambda t: NH.to_array(t)
    z2 = np.maximum(x @ A(w1).T + A(b1), 0) @ A(w2) + A(b2)
    e = np.exp(z2 * 0.5 - (z2 * 0.5).max(-1, keepdims=True)); probs = e / e.sum(-1, keepdims=True)
    check(f"mlp_b{batch}", g, {"x": x}, [probs, 1 / (1 + np.exp(-np.tanh(z2)))])

# ---- 2. Encoder-style block: LayerNormalization Transpose Reshape, erf-GELU, ReduceMean, Sub, Div, Constant, Flatten, Sqrt
gam, bet = W("gamma", 8), W("beta", 8)
nodes = [
    H.make_node("LayerNormalization", ["x", "gamma", "beta"], ["ln"], axis=-1, epsilon=1e-5),
    H.make_node("Transpose", ["ln"], ["tr"], perm=[1, 0, 2]),
    H.make_node("Reshape", ["tr", "shape"], ["r"]),
    H.make_node("Constant", [], ["rt2"], value=NH.from_array(np.array(math.sqrt(2), np.float32))),
    H.make_node("Div", ["r", "rt2"], ["u"]), H.make_node("Erf", ["u"], ["eu"]),
    H.make_node("Constant", [], ["one"], value=NH.from_array(np.array([1.0], np.float32))),
    H.make_node("Add", ["eu", "one"], ["eu1"]), H.make_node("Mul", ["r", "eu1"], ["g2"]),
    H.make_node("Constant", [], ["hf"], value=NH.from_array(np.array(0.5, np.float32))),
    H.make_node("Mul", ["g2", "hf"], ["gelu"]),
    H.make_node("ReduceMean", ["gelu"], ["mu"], axes=[-1], keepdims=1),
    H.make_node("Sub", ["gelu", "mu"], ["c"]),
    H.make_node("Mul", ["c", "c"], ["c2"]), H.make_node("Abs", ["c2"], ["ac"]), H.make_node("Sqrt", ["ac"], ["out"]),
    H.make_node("Flatten", ["out"], ["flat"], axis=0),
]
g = H.make_graph(nodes, "enc", [vi("x", [2, 3, 8])], [vi("flat", [1, 48])], [gam, bet, C("shape", [3, 16])])
x = rng.standard_normal((2, 3, 8)).astype(np.float32)
mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
ln = (x - mu) / np.sqrt(var + 1e-5) * NH.to_array(gam) + NH.to_array(bet)
r = ln.transpose(1, 0, 2).reshape(3, 16)
gl = 0.5 * r * (1 + np.vectorize(math.erf)(r / math.sqrt(2)))
check("encoder", g, {"x": x}, [np.abs(gl - gl.mean(-1, keepdims=True)).reshape(1, 48)], opset=17, tol=2e-4)

# ---- 3. a small CNN: Conv (pads, strides) BatchNormalization Relu MaxPool, depthwise + dilated Conv, GlobalAveragePool, Flatten, Gemm, Softmax
bn = [NH.from_array((1 + 0.2 * rng.standard_normal(8)).astype(np.float32), "bn_w"), W("bn_b", 8, scale=0.2), W("bn_m", 8, scale=0.2),
      NH.from_array((0.5 + rng.random(8)).astype(np.float32), "bn_v")]
g = H.make_graph([
    H.make_node("Conv", ["x", "cw", "cb"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2]),
    H.make_node("BatchNormalization", ["c1", "bn_w", "bn_b", "bn_m", "bn_v"], ["n1"], epsilon=1e-3),
    H.make_node("Relu", ["n1"], ["r1"]),
    H.make_node("MaxPool", ["r1"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
    H.make_node("Conv", ["p1", "dw"], ["c2"], kernel_shape=[3, 3], pads=[2, 2, 2, 2], dilations=[2, 2], group=8),
    H.make_node("Conv", ["c2", "pw", "pb"], ["c3"], kernel_shape=[1, 1]),
    H.make_node("GlobalAveragePool", ["c3"], ["gap"]),
    H.make_node("Flatten", ["gap"], ["fl"]),
    H.make_node("Gemm", ["fl", "fc", "fb"], ["logits"], transB=1),
    H.make_node("Softmax", ["logits"], ["probs"]),
], "cnn", [vi("x", [1, 3, 16, 16])], [vi("probs", [1, 5]), vi("c3", [1, 6, 4, 4])],
   [W("cw", 8, 3, 3, 3), W("cb", 8), *bn, W("dw", 8, 1, 3, 3), W("pw", 6, 8, 1, 1), W("pb", 6), W("fc", 5, 6), W("fb", 5)])
check("cnn", g, {"x": rng.standard_normal((1, 3, 16, 16)).astype(np.float32)})

# ---- 4. convolution and pooling variants: 1-d conv, asymmetric pads, auto_pad, average pools, transposed conv, instance / group norm
g = H.make_graph([
    H.make_node("Conv", ["s", "w1d", "b1d"], ["c1d"], kernel_shape=[5], pads=[2, 1], strides=[2]),
    H.make_node("MaxPool", ["c1d"], ["m1d"], kernel_shape=[3], strides=[2], pads=[1, 1]),
    H.make_node("Conv", ["x", "wa"], ["ca"], kernel_shape=[3, 2], pads=[0, 1, 2, 0]),
    H.make_node("Conv", ["x", "ws"], ["cs"], kernel_shape=[3, 3], strides=[2, 2], auto_pad="SAME_UPPER"),
    H.make_node("AveragePool", ["x"], ["ap0"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]),
    H.make_node("AveragePool", ["x"], ["ap1"], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1], count_include_pad=1),
    H.make_node("AveragePool", ["x"], ["ap2"], kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 1, 1], count_include_pad=1),
    H.make_node("MaxPool", ["x"], ["mp2"], kernel_shape=[3, 3], strides=[2, 2], pads=[0, 0, 1, 1]),
    H.make_node("GlobalMaxPool", ["x"], ["gmp"]),
    H.make_node("ConvTranspose", ["x", "wt", "bt"], ["ct"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1], output_padding=[1, 1]),
    H.make_node("InstanceNormalization", ["x", "in_w", "in_b"], ["inorm"], epsilon=1e-5),
    H.make_node("GroupNormalization", ["x", "gn_w", "gn_b"], ["gnorm"], num_groups=2, epsilon=1e-5),
], "convs", [vi("x", [1, 4, 9, 10]), vi("s", [1, 3, 21])],
   [vi("m1d", [1, 6, 5]), vi("ca", [1, 5, 9, 10]), vi("cs", [1, 5, 5, 5]), vi("ap0", [1, 4, 5, 5]), vi("ap1", [1, 4, 9, 10]), vi("ap2", [1, 4, 5, 5]),
    vi("mp2", [1, 4, 4, 5]), vi("gmp", [1, 4, 1, 1]), vi("ct", [1, 3, 18, 20]), vi("inorm", [1, 4, 9, 10]), vi("gnorm", [1, 4, 9, 10])],
   [W("w1d", 6, 3, 5), W("b1d", 6), W("wa", 5, 4, 3, 2), W("ws", 5, 4, 3, 3), W("wt", 4, 3, 3, 3), W("bt", 3), W("in_w", 4), W("in_b", 4), W("gn_w", 4), W("gn_b", 4)])
check("convs", g, {"x": rng.standard_normal((1, 4, 9, 10)).astype(np.float32), "s": rng.standard_normal((1, 3, 21)).astype(np.float32)}, opset=21)

# ---- 5. Resize / Pad / activations
g = H.make_graph([
    H.make_node("Resize", ["x", "", "sc"], ["up_n"], mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor"),
    H.make_node("Resize", ["x", "", "sc"], ["up_h"], mode="nearest"),
    H.make_node("Resize", ["x", "", "sc23"], ["up_l"], mode="linear", coordinate_transformation_mode="half_pixel"),
    H.make_node("Resize", ["x", "", "", "sizes"], ["up_a"], mode="linear", coordinate_transformation_mode="align_corners"),
    H.make_node("Pad", ["x", "pads", "pv"], ["pad_c"], mode="constant"),
    H.make_node("Pad", ["x", "pads"], ["pad_r"], mode="reflect"),
    H.make_node("Pad", ["x", "pads"], ["pad_e"], mode="edge"),
    H.make_node("LeakyRelu", ["x"], ["lrelu"], alpha=0.1), H.make_node("Elu", ["x"], ["elu"], alpha=1.5), H.make_node("Selu", ["x"], ["selu"]),
    H.make_node("Celu", ["x"], ["celu"], alpha=2.0), H.make_node("HardSigmoid", ["x"], ["hsig"]), H.make_node("HardSwish", ["x"], ["hsw"]),
    H.make_node("PRelu", ["x", "slope"], ["prelu"]), H.make_node("Clip", ["x", "lo", "hi"], ["clip"]), H.make_node("Softsign", ["x"], ["ssign"]),
    H.make_node("Ceil", ["x"], ["ceil"]), H.make_node("Floor", ["x"], ["floor"]), H.make_node("Mish", ["x"], ["mish"]), H.make_node("Gelu", ["x"], ["gelu"]),
    H.make_node("Softplus", ["x"], ["sp"]), H.make_node("Reciprocal", ["sp"], ["recip"]),
], "resize", [vi("x", [1, 2, 4, 5])],
   [vi(n, None) for n in ["up_n", "up_h", "up_l", "up_a", "pad_c", "pad_r", "pad_e", "lrelu", "elu", "selu", "celu", "hsig", "hsw", "prelu", "clip", "ssign", "ceil", "floor", "mish", "gelu", "recip"]],
   [C("sc", [1, 1, 2, 2], np.float32), C("sc23", [1, 1, 2, 3], np.float32), C("sizes", [1, 2, 8, 15]), C("pads", [0, 0, 1, 2, 0, 0, 2, 1]), C("pv", 0.25, np.float32),
    W("slope", 2, 1, 1), C("lo", -0.5, np.float32), C("hi", 0.7, np.float32)])
check("resize_pad_act", g, {"x": rng.standard_normal((1, 2, 4, 5)).astype(np.float32)}, opset=20, tol=2e-4)

# ---- 6. data movement: Concat Slice Split Squeeze Unsqueeze Where Expand Gather(const) Transpose, reductions, ArgMax, variadic ops, Softmax on a middle axis
g = H.make_graph([
    H.make_node("Concat", ["x", "y", "x"], ["cat"], axis=1),
    H.make_node("Slice", ["cat", "st", "en", "ax"], ["sl"]),
    H.make_node("Split", ["x", "parts"], ["sp0", "sp1"], axis=2),
    H.make_node("Unsqueeze", ["sp0", "ax0"], ["us"]), H.make_node("Squeeze", ["us", "ax0"], ["sq"]),
    H.make_node("Where", ["mask", "x", "y"], ["wh"]),
    H.make_node("Expand", ["row", "eshape"], ["ex"]), H.make_node("Add", ["ex", "x"], ["exx"]),
    H.make_node("Gather", ["x", "gi"], ["ga1"], axis=2), H.make_node("Gather", ["x", "gs"], ["ga0"], axis=1),
    H.make_node("Transpose", ["x"], ["tr"], perm=[2, 0, 1]),
    H.make_node("ReduceSum", ["x", "axes12"], ["rs"], keepdims=0), H.make_node("ReduceMax", ["x"], ["rmax"], keepdims=1),
    H.make_node("ReduceMin", ["x", "axes0"], ["rmin"], keepdims=0), H.make_node("ReduceL2", ["x", "axes12"], ["rl2"], keepdims=1),
    H.make_node("ReduceMean", ["x"], ["rmean"], keepdims=0), H.make_node("ReduceLogSumExp", ["x", "axes0"], ["rlse"], keepdims=0),
    H.make_node("ArgMax", ["x"], ["am"], axis=2, keepdims=0), H.make_node("Cast", ["am"], ["amf"], to=TP.FLOAT),
    H.make_node("Sum", ["x", "y", "x"], ["sum3"]), H.make_node("Mean", ["x", "y"], ["mean2"]), H.make_node("Max", ["x", "y", "row"], ["max3"]), H.make_node("Min", ["x", "y"], ["min2"]),
    H.make_node("Softmax", ["x"], ["sm1"], axis=1), H.make_node("LogSoftmax", ["x"], ["lsm"], axis=-1),
    H.make_node("Pow", ["sm1", "two"], ["pw"]),
], "move", [vi("x", [2, 3, 4]), vi("y", [2, 3, 4]), vi("mask", [2, 3, 4], TP.BOOL)],
   [vi(n, None) for n in ["sl", "sp1", "sq", "wh", "exx", "ga1", "ga0", "tr", "rs", "rmax", "rmin", "rl2", "rmean", "rlse", "amf", "sum3", "mean2", "max3", "min2", "sm1", "lsm", "pw"]],
   [C("st", [1, -3]), C("en", [8, 2 ** 40]), C("ax", [1, 2]), C("parts", [1, 3]), C("ax0", [0]), W("row", 1, 4), C("eshape", [2, 3, 4]), C("gi", [3, 0, -2]), C("gs", 1),
    C("axes12", [1, 2]), C("axes0", [0]), C("two", 2.0, np.float32)])
x = rng.standard_normal((2, 3, 4)).astype(np.float32)
check("movement", g, {"x": x, "y": rng.standard_normal((2, 3, 4)).astype(np.float32), "mask": rng.random((2, 3, 4)) > 0.5}, opset=18)

# ---- 7. shape subgraphs with two symbolic dimensions: Shape -> Gather -> Unsqueeze -> Concat -> Reshape (head split and merge),
#         Shape -> Slice -> Concat, ConstantOfShape and Expand to a shape that depends on them
g = H.make_graph([
    H.make_node("Shape", ["x"], ["shp"]),
    H.make_node("Gather", ["shp", "i0"], ["b"]), H.make_node("Gather", ["shp", "i1"], ["s"]),
    H.make_node("Unsqueeze", ["b", "ax0"], ["b1"]), H.make_node("Unsqueeze", ["s", "ax0"], ["s1"]),
    H.make_node("Concat", ["b1", "s1", "heads"], ["split_shape"], axis=0),
    H.make_node("Reshape", ["x", "split_shape"], ["xh"]),                      # [b, s, 2, 4]
    H.make_node("Transpose", ["xh"], ["xt"], perm=[0, 2, 1, 3]),               # [b, 2, s, 4]
    H.make_node("MatMul", ["xt", "wq"], ["q"]),
    H.make_node("Transpose", ["q"], ["qt"], perm=[0, 2, 1, 3]),
    H.make_node("Shape", ["x"], ["shp2"]), H.make_node("Slice", ["shp2", "z0", "z2", "z0"], ["bs"]),
    H.make_node("Concat", ["bs", "m1"], ["merge_shape"], axis=0),
    H.make_node("Reshape", ["qt", "merge_shape"], ["merged"]),                 # [b, s, 8]
    H.make_node("Mul", ["b", "s"], ["bxs"]), H.make_node("Unsqueeze", ["bxs", "ax0"], ["bxs1"]),
    H.make_node("Concat", ["bxs1", "m1"], ["flat_shape"], axis=0),
    H.make_node("Reshape", ["merged", "flat_shape"], ["flat"]),                # [b * s, 8]
    H.make_node("Concat", ["b1", "s1", "one"], ["ones_shape"], axis=0),
    H.make_node("ConstantOfShape", ["ones_shape"], ["fill"], value=NH.from_array(np.array([0.5], np.float32))),   # [b, s, 1]
    H.make_node("Add", ["merged", "fill"], ["shifted"]),
    H.make_node("Expand", ["bias", "split_shape"], ["bias_x"]),                # [b, s, 2, 4]
    H.make_node("Add", ["xh", "bias_x"], ["xb"]),
    H.make_node("ReduceMean", ["x"], ["pooled"], axes=[1], keepdims=0),        # [b, 8]
], "shapes", [vi("x", ["batch", "seq_len", 8])],
   [vi("flat", None), vi("shifted", None), vi("xb", None), vi("pooled", None)],
   [C("i0", 0), C("i1", 1), C("ax0", [0]), C("heads", [2, 4]), W("wq", 4, 4), C("z0", [0]), C("z2", [2]), C("m1", [-1]), C("one", [1]), W("bias", 1, 4)])
for b, s in ((2, 3), (1, 5)):
    check(f"shapes_b{b}_s{s}", g, {"x": rng.standard_normal((b, s, 8)).astype(np.float32)}, opset=17, dims=(f"dim_batch={b}",))

# ---- 8. integer inputs: token ids -> Gather on an embedding table; an int mask -> Cast -> Mul; positions by Slice of a table
g = H.make_graph([
    H.make_node("Gather", ["emb", "ids"], ["tok"]),                            # [seq, 16]
    H.make_node("Cast", ["mask"], ["mf"], to=TP.FLOAT), H.make_node("Unsqueeze", ["mf", "axm1"], ["m1"]),
    H.make_node("Mul", ["tok", "m1"], ["tokm"]),
    H.make_node("ReduceSum", ["tokm", "ax0"], ["summed"], keepdims=1),
    H.make_node("ReduceSum", ["mf", "ax0"], ["cnt"], keepdims=1), H.make_node("Div", ["summed", "cnt"], ["meanp"]),
    H.make_node("MatMul", ["meanp", "head"], ["logits"]),
    H.make_node("Gather", ["emb3", "ids"], ["tok3"]),                          # a rank-3 table: [seq, 2, 3]
], "tokens", [vi("ids", ["seq"], TP.INT64), vi("mask", ["seq"], TP.INT32)], [vi("logits", [1, 5]), vi("tok3", None), vi("tok", None)],
   [W("emb", 50, 16), W("head", 16, 5), C("axm1", [-1]), C("ax0", [0]), W("emb3", 50, 2, 3)])
for n in (4, 7):
    ids = rng.integers(0, 50, size=n).astype(np.int64); mask = (rng.random(n) > 0.3).astype(np.int32); mask[0] = 1
    check(f"tokens_{n}", g, {"ids": ids, "mask": mask})

# ---- 9. constants computed at import: Range, Cast, arithmetic on shapes of constants, Where with a constant condition
g = H.make_graph([
    H.make_node("Range", ["r0", "r4", "r1"], ["idx"]),                         # [0, 1, 2, 3]
    H.make_node("Mul", ["idx", "r2"], ["idx2"]),                               # [0, 2, 4, 6]
    H.make_node("Gather", ["x", "idx2"], ["picked"], axis=1),
    H.make_node("Shape", ["table"], ["tshape"]), H.make_node("ReduceProd", ["tshape"], ["tsize"], keepdims=1),
    H.make_node("Concat", ["m1", "tsize"], ["rs"], axis=0), H.make_node("Reshape", ["tile_src", "rs"], ["reshaped"]),
    H.make_node("Greater", ["r4", "r1"], ["cond"]), H.make_node("Where", ["cond", "ca", "cb"], ["chosen"]),
    H.make_node("Add", ["picked", "chosen"], ["out"]),
    H.make_node("Sqrt", ["nine"], ["three"]), H.make_node("Mul", ["reshaped", "three"], ["scaled"]),
], "folds", [vi("x", [2, 8]), vi("tile_src", [2, 6])], [vi("out", None), vi("scaled", None)],
   [C("r0", 0), C("r4", 4), C("r1", 1), C("r2", 2), W("table", 2, 3), C("m1", [-1]), C("ca", [10.0, 20.0, 30.0, 40.0], np.float32), C("cb", [1.0, 2.0, 3.0, 4.0], np.float32), C("nine", 9.0, np.float32)])
check("folds", g, {"x": rng.standard_normal((2, 8)).astype(np.float32), "tile_src": rng.standard_normal((2, 6)).astype(np.float32)})

# ---- 10. external tensor data: weights in a file next to the model
ext_dir = os.path.join(d, "ext"); os.makedirs(ext_dir)
g = H.make_graph([H.make_node("MatMul", ["x", "big"], ["h"]), H.make_node("Add", ["h", "bias"], ["y"])], "ext",
                 [vi("x", ["n", 32])], [vi("y", ["n", 24])], [W("big", 32, 24), W("bias", 24)])
m = model_of(g, 17); p = os.path.join(ext_dir, "ext.onnx")
onnx.save(m, p, save_as_external_data=True, all_tensors_to_one_file=True, location="ext_weights.bin", size_threshold=0)
if not os.path.exists(os.path.join(ext_dir, "ext_weights.bin")) or os.path.getsize(p) > 2000: sys.exit("external data was not written as expected")
x = rng.standard_normal((3, 32)).astype(np.float32)
got = run_forge("ext", p, {"x": x}, ["y"])[0]
want = ReferenceEvaluator(p).run(None, {"x": x})[0].ravel()
if np.max(np.abs(got - want)) > 1e-4: sys.exit("ext: mismatch against the reference evaluator")
cases += 1; print(f"external data: max err {np.max(np.abs(got - want)):.2e} vs onnx reference")
def ext_variant(name, edit):
    mm = onnx.load(p, load_external_data=False)
    for t in mm.graph.initializer:
        if t.name == "big": edit(t)
    q = os.path.join(ext_dir, name + ".onnx"); onnx.save(mm, q)
    return run("import", q, "-o", os.path.join(d, name + ".fpk"), ok=False)
def set_kv(t, k, v):
    for e in t.external_data:
        if e.key == k: e.value = v
secret = os.path.join(d, "secret.bin"); open(secret, "wb").write(b"\0" * 4096)
msgs = [ext_variant("ext_up", lambda t: set_kv(t, "location", "../secret.bin")),
        ext_variant("ext_abs", lambda t: set_kv(t, "location", secret)),
        ext_variant("ext_off", lambda t: set_kv(t, "offset", str(10 ** 9))),
        ext_variant("ext_len", lambda t: set_kv(t, "length", "16")),
        ext_variant("ext_none", lambda t: set_kv(t, "location", "no_such_file.bin"))]
for msg, needle in zip(msgs, ["not a path inside the model directory", "not a path inside the model directory", "past the end", "its shape and type need", "cannot read external data"]):
    if needle not in msg or "big" not in msg: sys.exit(f"external data: expected '{needle}' in: {msg.strip()}")
if os.name != "nt":   # a symlink in the model directory that points outside it is not followed
    os.symlink(secret, os.path.join(ext_dir, "link.bin"))
    msg = ext_variant("ext_link", lambda t: set_kv(t, "location", "link.bin"))
    if "not a path inside the model directory" not in msg: sys.exit("external data: a symlink out of the model directory was followed: " + msg)
shutil.copy(os.path.join(ext_dir, "ext_weights.bin"), os.path.join(ext_dir, "short.bin"))
with open(os.path.join(ext_dir, "short.bin"), "r+b") as f: f.truncate(1000)
msg = ext_variant("ext_short", lambda t: set_kv(t, "location", "short.bin"))
if "does not fit" not in msg and "past the end" not in msg: sys.exit("external data: a truncated file was accepted: " + msg)
cases += 1; print("external data: traversal, absolute path, symlink, bad offset / length, missing and truncated files are refused")

# ---- 11. what the importer refuses, and how it says so: the operator and the opset are named
X4 = [vi("x", [2, 3, 4, 4])]
refused("no_einsum", H.make_graph([H.make_node("Einsum", ["x"], ["y"], equation="nchw->nc")], "g", X4, [vi("y", None)]), "Einsum", "opset 17")
refused("conv_batch2", H.make_graph([H.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3])], "g", X4, [vi("y", None)], [W("w", 2, 3, 3, 3)]), "Conv", "batch size 1", "opset 17")
refused("less_runtime", H.make_graph([H.make_node("Less", ["x", "x"], ["y"])], "g", X4, [vi("y", None, TP.BOOL)]), "Less", "opset 17")
refused("cast_trunc", H.make_graph([H.make_node("Cast", ["x"], ["y"], to=TP.INT64)], "g", X4, [vi("y", None, TP.INT64)]), "Cast", "truncation")
refused("resize_cubic", H.make_graph([H.make_node("Resize", ["x", "", "sc"], ["y"], mode="cubic")], "g", [vi("x", [1, 3, 4, 4])], [vi("y", None)], [C("sc", [1, 1, 2, 2], np.float32)]), "Resize", "cubic")
refused("sym_concat", H.make_graph([H.make_node("Concat", ["x", "x"], ["y"], axis=0)], "g", [vi("x", ["n", 4])], [vi("y", None)]), "Concat", "symbolic")
refused("sym_offset", H.make_graph([H.make_node("Shape", ["x"], ["s"]), H.make_node("Gather", ["s", "i0"], ["n"]), H.make_node("Sub", ["n", "one"], ["n1"]),
                                    H.make_node("Unsqueeze", ["n1", "ax0"], ["n1v"]), H.make_node("Concat", ["n1v", "m1"], ["sh"], axis=0), H.make_node("Reshape", ["x", "sh"], ["y"])],
                                   "g", [vi("x", ["n", 4])], [vi("y", None)], [C("i0", 0), C("one", 1), C("ax0", [0]), C("m1", [-1])]), "Sub", "symbolic dimension")
refused("old_softmax", H.make_graph([H.make_node("Softmax", ["x"], ["y"], axis=1)], "g", X4, [vi("y", None)]), "Softmax", "opset 11", opset=11)
if os.environ.get("FORGE_ONNX_SEEDS"):   # refresh fuzz/corpus/onnx: FORGE_ONNX_SEEDS=fuzz/corpus/onnx python3 tests/onnx_test.py <forge>
    for n in ("cnn", "convs", "resize_pad_act", "movement", "shapes_b2_s3", "tokens_4", "folds"): shutil.copy(os.path.join(d, n + ".onnx"), os.path.join(os.environ["FORGE_ONNX_SEEDS"], n.split("_b")[0].replace("_4", "") + ".onnx"))
    shutil.copy(os.path.join(ext_dir, "ext.onnx"), os.path.join(os.environ["FORGE_ONNX_SEEDS"], "external.onnx"))
print(f"onnx import OK: {cases} cases{'' if ort is not None else ' (onnxruntime not installed: reference evaluator only)'}")
shutil.rmtree(d, ignore_errors=True)

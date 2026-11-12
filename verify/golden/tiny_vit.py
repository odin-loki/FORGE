#!/usr/bin/env python3
"""Golden end-to-end test for the vit template (image-embedding session): random tiny vision encoders, a float64
numpy reference, and `forge embed --input image=...`.

  vit      ViTModel layout: conv patch embedding with bias, class token, exact GELU; output = LayerNorm(class token)
  clip     CLIP vision tower: no patch bias, class token, pre-LayerNorm, QuickGELU; output = post-LayerNorm(class
           token) @ visual_projection^T
  siglip   SigLIP vision tower: no class token, GELU-tanh; output = attention-pooling head (a learned probe
           cross-attends over the patch tokens, then an MLP)

The image is 3 x 16 x 16 with 4 x 4 patches (16 patch tokens). HF import in f32 and q8_0 per variant.
"""
import os
import numpy as np
import golden_util as gu

D, L, NH, F, IMG, P, C = 32, 2, 4, 64, 16, 4, 3
NP = (IMG // P) ** 2
BASE = dict(hidden_size=D, num_hidden_layers=L, num_attention_heads=NH, intermediate_size=F, image_size=IMG, patch_size=P, num_channels=C)
CFGS = {"vit": dict(BASE, architectures=["ViTModel"], model_type="vit", hidden_act="gelu", layer_norm_eps=1e-12),
        "clip": dict(BASE, architectures=["CLIPVisionModelWithProjection"], model_type="clip_vision_model", hidden_act="quick_gelu", layer_norm_eps=1e-5, projection_dim=24),
        # SigLIP checkpoints nest the tower under vision_config
        "siglip": dict(architectures=["SiglipModel"], model_type="siglip", vision_config=dict(BASE, model_type="siglip_vision_model", hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6))}
ACT = {"vit": gu.gelu, "clip": gu.quick_gelu, "siglip": gu.gelu_tanh}
EPS = {"vit": 1e-12, "clip": 1e-5, "siglip": 1e-6}

def make_weights(kind, seed):
    """Weights under neutral names (used by the reference) and the checkpoint's own names (written to safetensors)."""
    rng = np.random.default_rng(seed)
    ln = lambda: (1 + 0.1 * rng.normal(size=D), 0.1 * rng.normal(size=D))
    lin = lambda o, i: (rng.normal(0, 1 / np.sqrt(i), (o, i)), 0.1 * rng.normal(size=o))
    N = {"patch_w": rng.normal(0, 0.2, (D, C, P, P)), "patch_b": np.zeros(D) if kind == "clip" else 0.1 * rng.normal(size=D),
         "pos": rng.normal(0, 0.5, (NP + (kind != "siglip"), D)), "post": ln()}
    if kind != "siglip": N["cls"] = rng.normal(0, 0.5, D)
    if kind == "clip": N["pre"] = ln(); N["proj"] = rng.normal(0, 1 / np.sqrt(D), (24, D))
    for l in range(L):
        N[f"{l}.n1"], N[f"{l}.n2"] = ln(), ln()
        for n, (o, i) in {"q": (D, D), "k": (D, D), "v": (D, D), "o": (D, D), "in": (F, D), "out": (D, F)}.items(): N[f"{l}.{n}"] = lin(o, i)
    if kind == "siglip":
        N["probe"] = rng.normal(0, 0.5, D); N["h.in_proj"] = lin(3 * D, D); N["h.o"] = lin(D, D); N["h.n"] = ln(); N["h.in"] = lin(F, D); N["h.out"] = lin(D, F)
    H = {}
    def wb(name, t): H[name + ".weight"], H[name + ".bias"] = t
    if kind == "vit":
        H["embeddings.patch_embeddings.projection.weight"] = N["patch_w"]; H["embeddings.patch_embeddings.projection.bias"] = N["patch_b"]
        H["embeddings.cls_token"] = N["cls"].reshape(1, 1, D); H["embeddings.position_embeddings"] = N["pos"].reshape(1, NP + 1, D)
        wb("layernorm", N["post"])
        for l in range(L):
            p = f"encoder.layer.{l}."
            wb(p + "layernorm_before", N[f"{l}.n1"]); wb(p + "layernorm_after", N[f"{l}.n2"])
            wb(p + "attention.attention.query", N[f"{l}.q"]); wb(p + "attention.attention.key", N[f"{l}.k"]); wb(p + "attention.attention.value", N[f"{l}.v"])
            wb(p + "attention.output.dense", N[f"{l}.o"]); wb(p + "intermediate.dense", N[f"{l}.in"]); wb(p + "output.dense", N[f"{l}.out"])
    else:
        v = "vision_model."
        H[v + "embeddings.patch_embedding.weight"] = N["patch_w"]; H[v + "embeddings.position_embedding.weight"] = N["pos"]
        if kind == "clip": H[v + "embeddings.class_embedding"] = N["cls"]; wb(v + "pre_layrnorm", N["pre"]); H["visual_projection.weight"] = N["proj"]
        else: H[v + "embeddings.patch_embedding.bias"] = N["patch_b"]
        wb(v + "post_layernorm", N["post"])
        for l in range(L):
            p = f"{v}encoder.layers.{l}."
            wb(p + "layer_norm1", N[f"{l}.n1"]); wb(p + "layer_norm2", N[f"{l}.n2"])
            wb(p + "self_attn.q_proj", N[f"{l}.q"]); wb(p + "self_attn.k_proj", N[f"{l}.k"]); wb(p + "self_attn.v_proj", N[f"{l}.v"])
            wb(p + "self_attn.out_proj", N[f"{l}.o"]); wb(p + "mlp.fc1", N[f"{l}.in"]); wb(p + "mlp.fc2", N[f"{l}.out"])
        if kind == "siglip":
            H[v + "head.probe"] = N["probe"].reshape(1, 1, D); H[v + "head.attention.in_proj_weight"], H[v + "head.attention.in_proj_bias"] = N["h.in_proj"]
            wb(v + "head.attention.out_proj", N["h.o"]); wb(v + "head.layernorm", N["h.n"]); wb(v + "head.mlp.fc1", N["h.in"]); wb(v + "head.mlp.fc2", N["h.out"])
    return N, H

def forward(kind, N, img):
    act, eps = ACT[kind], EPS[kind]
    f64 = lambda t: tuple(np.float32(x).astype(np.float64) for x in t) if isinstance(t, tuple) else np.float32(t).astype(np.float64)
    N = {k: f64(v) for k, v in N.items()}   # the checkpoint holds f32
    aff = lambda x, t: x @ t[0].T + t[1]
    ln = lambda x, t: gu.layer_norm(x, t[0], t[1], eps)
    g = IMG // P   # patches in row-major (gy, gx) order
    patches = img.reshape(C, g, P, g, P).transpose(1, 3, 0, 2, 4).reshape(NP, C * P * P)
    x = patches @ N["patch_w"].reshape(D, -1).T + N["patch_b"]
    if kind != "siglip": x = np.concatenate([N["cls"][None], x], 0)
    x = x + N["pos"]
    if kind == "clip": x = ln(x, N["pre"])
    s = x.shape[0]; hd = D // NH
    for l in range(L):
        h = ln(x, N[f"{l}.n1"])
        q, k, v = (aff(h, N[f"{l}.{n}"]).reshape(s, NH, hd) for n in "qkv")
        x = x + aff(gu.attend(q, k, v).reshape(s, D), N[f"{l}.o"])
        x = x + aff(act(aff(ln(x, N[f"{l}.n2"]), N[f"{l}.in"])), N[f"{l}.out"])
    if kind == "vit": return ln(x[0], N["post"])
    if kind == "clip": return ln(x[0], N["post"]) @ N["proj"].T
    x = ln(x, N["post"])
    wi, bi = N["h.in_proj"]
    q = (N["probe"][None] @ wi[:D].T + bi[:D]).reshape(1, NH, hd)
    k = (x @ wi[D:2 * D].T + bi[D:2 * D]).reshape(s, NH, hd); v = (x @ wi[2 * D:].T + bi[2 * D:]).reshape(s, NH, hd)
    y = aff(gu.attend(q, k, v).reshape(1, D), N["h.o"])
    y = y + aff(act(aff(ln(y, N["h.n"]), N["h.in"])), N["h.out"])
    return y[0]

def main():
    a = gu.args(); ck = gu.Checker(a.forge, a.work)
    img = np.random.default_rng(5).normal(0, 1.0, (C, IMG, IMG)).astype(np.float32)
    imgf = os.path.join(a.work, "image.f32"); img.tofile(imgf)
    for i, kind in enumerate(("vit", "clip", "siglip")):
        hf = os.path.join(a.work, "hf_" + kind)
        N, H = make_weights(kind, 400 + i)
        gu.save_model(hf, H, CFGS[kind], tokenizer=None)
        ref = forward(kind, N, img.astype(np.float64))
        ck.embed(ck.imp(hf, f"{kind}_f32.fpk"), ["--input", "image=" + imgf], ref, f"{kind} f32 ({ref.shape[0]}-d)")
        ck.embed(ck.imp(hf, f"{kind}_q8_0.fpk", "q8_0"), ["--input", "image=" + imgf], ref, f"{kind} q8_0", cosmin=0.999)
    ck.done("vit")

if __name__ == "__main__": main()

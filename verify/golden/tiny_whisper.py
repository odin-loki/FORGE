#!/usr/bin/env python3
"""Golden end-to-end test for the whisper_encoder template: a random tiny Whisper encoder (two conv1d + GELU, the
sinusoidal position table, pre-LayerNorm blocks whose key projection has no bias), a float64 numpy reference, and
`forge embed --input mel=...`. The decoder weights of the checkpoint are ignored by the importer (encoder only).

  1. HF import (f32, q8_0) -> encoder states [n_ctx, d_model] against the reference
  2. the checkpoint's position table equals the sinusoid_embed op's formula (shift 1), checked in numpy here and
     against the op's reference in tests/test_ops_ext.cpp
"""
import os
import numpy as np
import golden_util as gu

CFG = dict(architectures=["WhisperForConditionalGeneration"], model_type="whisper", d_model=32, encoder_layers=2, encoder_attention_heads=4,
           encoder_ffn_dim=64, decoder_layers=1, decoder_attention_heads=4, decoder_ffn_dim=64, num_mel_bins=8, max_source_positions=10,
           max_target_positions=16, vocab_size=300, activation_function="gelu", bos_token_id=290, eos_token_id=291)

def sinusoids(length, channels, max_timescale=10000.0):   # openai/whisper model.py
    inc = np.log(max_timescale) / (channels // 2 - 1)
    t = np.arange(length)[:, None] * np.exp(-inc * np.arange(channels // 2))[None, :]
    return np.concatenate([np.sin(t), np.cos(t)], 1)

def make_weights(seed=500):
    rng = np.random.default_rng(seed); D, F, M = CFG["d_model"], CFG["encoder_ffn_dim"], CFG["num_mel_bins"]
    ln = lambda: (1 + 0.1 * rng.normal(size=D), 0.1 * rng.normal(size=D))
    e = "model.encoder."
    W = {e + "conv1.weight": rng.normal(0, 0.3, (D, M, 3)), e + "conv1.bias": 0.1 * rng.normal(size=D),
         e + "conv2.weight": rng.normal(0, 0.1, (D, D, 3)), e + "conv2.bias": 0.1 * rng.normal(size=D),
         e + "embed_positions.weight": sinusoids(CFG["max_source_positions"], D),
         "model.decoder.embed_tokens.weight": rng.normal(size=(CFG["vocab_size"], D))}   # decoder tensors are present and must be ignored
    W[e + "layer_norm.weight"], W[e + "layer_norm.bias"] = ln()
    for l in range(CFG["encoder_layers"]):
        p = f"{e}layers.{l}."
        W[p + "self_attn_layer_norm.weight"], W[p + "self_attn_layer_norm.bias"] = ln()
        W[p + "final_layer_norm.weight"], W[p + "final_layer_norm.bias"] = ln()
        for n, (o, i) in {"self_attn.q_proj": (D, D), "self_attn.k_proj": (D, D), "self_attn.v_proj": (D, D), "self_attn.out_proj": (D, D), "fc1": (F, D), "fc2": (D, F)}.items():
            W[p + n + ".weight"] = rng.normal(0, 1 / np.sqrt(i), (o, i))
            if n != "self_attn.k_proj": W[p + n + ".bias"] = 0.1 * rng.normal(size=o)
    return W

def conv1d(x, w, b, stride, pad):   # x [c, l], w [o, c, k]
    xp = np.pad(x, ((0, 0), (pad, pad))); k = w.shape[2]; n = (xp.shape[1] - k) // stride + 1
    cols = np.stack([xp[:, t * stride:t * stride + k] for t in range(n)])   # [n, c, k]
    return np.einsum("tck,ock->ot", cols, w) + b[:, None]

def forward(W, mel):
    D, nh = CFG["d_model"], CFG["encoder_attention_heads"]; hd = D // nh; e = "model.encoder."
    ln = lambda x, n: gu.layer_norm(x, W[n + ".weight"], W[n + ".bias"], 1e-5)
    aff = lambda x, n: x @ W[n + ".weight"].T + (W[n + ".bias"] if n + ".bias" in W else 0.0)
    x = gu.gelu(conv1d(mel, W[e + "conv1.weight"], W[e + "conv1.bias"], 1, 1))
    x = gu.gelu(conv1d(x, W[e + "conv2.weight"], W[e + "conv2.bias"], 2, 1)).T + W[e + "embed_positions.weight"]
    s = x.shape[0]
    for l in range(CFG["encoder_layers"]):
        p = f"{e}layers.{l}."
        h = ln(x, p + "self_attn_layer_norm")
        q, k, v = (aff(h, p + "self_attn." + n).reshape(s, nh, hd) for n in ("q_proj", "k_proj", "v_proj"))
        x = x + aff(gu.attend(q, k, v).reshape(s, D), p + "self_attn.out_proj")
        x = x + aff(gu.gelu(aff(ln(x, p + "final_layer_norm"), p + "fc1")), p + "fc2")
    return ln(x, e + "layer_norm")

def main():
    a = gu.args(); ck = gu.Checker(a.forge, a.work)
    hf = os.path.join(a.work, "hf")
    W = gu.save_model(hf, make_weights(), CFG)
    frames = 2 * CFG["max_source_positions"]
    mel = np.random.default_rng(6).normal(0, 1.0, (CFG["num_mel_bins"], frames)).astype(np.float32)
    melf = os.path.join(a.work, "mel.f32"); mel.tofile(melf)
    ref = forward(W, mel.astype(np.float64))
    ck.ok(ref.shape == (CFG["max_source_positions"], CFG["d_model"]), f"reference encoder states are {ref.shape}")
    ck.embed(ck.imp(hf, "whisper_f32.fpk"), ["--input", "mel=" + melf], ref, "whisper encoder f32")
    ck.embed(ck.imp(hf, "whisper_q8_0.fpk", "q8_0"), ["--input", "mel=" + melf], ref, "whisper encoder q8_0", cosmin=0.999)
    # the op's formula: f_i = max_period^(-i / (dim / 2 - shift)) with shift 1, [sin | cos]
    D = CFG["d_model"]; i = np.arange(D // 2); f = 10000.0 ** (-i / (D // 2 - 1)); t = np.arange(CFG["max_source_positions"])[:, None] * f[None]
    ck.ok(np.abs(np.concatenate([np.sin(t), np.cos(t)], 1) - sinusoids(CFG["max_source_positions"], D)).max() < 1e-12, "sinusoid_embed(shift 1) formula equals Whisper's position table")
    ck.done("whisper")

if __name__ == "__main__": main()

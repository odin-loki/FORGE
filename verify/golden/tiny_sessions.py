#!/usr/bin/env python3
"""Golden end-to-end test for the session kinds a model file chooses (FORGE_PLAN E.3): `session diffusion_loop` and
`session streaming`. Each model is a blueprint directory (blueprint.fbp + model.safetensors), imported with
`forge import`, run with `forge denoise` / `forge stream`, and compared with a numpy reference written here from
the definitions (not from FORGE's code):

  diffusion  a tiny denoiser MLP with a class-conditioning entry point; DDIM over a scaled-linear / linear / cosine
             schedule with epsilon / v / sample prediction, classifier-free guidance, and the flow-matching Euler
             sampler. The initial latent is given (--init), so the whole trajectory is deterministic.
  streaming  a per-frame projection followed by a causal convolution over frames whose window is the model's state;
             the reference frames the signal offline. Different input chunk sizes must give the same bytes.
"""
import json, os, subprocess, sys
import numpy as np
from safetensors.numpy import save_file
import golden_util as gu

DEN_BP = """blueprint "tiny-denoiser" opset 4
params { n: 8  h: 16  classes: 5  e: 4 }
input x : f32[n]
input t : f32[1]
input cond : f32[e]
graph {
  xr = reshape(x, shape: [1, n])
  hid = tanh(linear(xr, W.w1) + scale(t, alpha: 0.001) * W.wt + linear(reshape(cond, shape: [1, e]), W.wc) + W.b1)
  eps = reshape(linear(hid, W.w2) + W.b2, shape: [n])
}
output eps : f32[n]
entry encode {
  input label : f32[1]
  output cond : f32[e]
  graph { cond = reshape(embed(W.emb, label), shape: [e]) }
}
session diffusion_loop(steps: %(T)d, schedule: %(schedule)s, beta_start: %(b0)g, beta_end: %(b1)g, prediction: %(pred)s,
                       sampler: %(sampler)s, inference_steps: %(n)d, guidance: %(g)g, latent: x, timestep: t, encoder: encode)
"""
STREAM_BP = """blueprint "tiny-stream" opset 4
params { window: 16  hop: 8  d: 6  taps: 3 }
state conv[1] : conv_window
input frame : f32[window]
graph {
  f = linear(reshape(frame, shape: [1, window]), W.proj) + W.bias
  y = tanh(conv[0].conv(f, W.cw))
}
output y : f32[1, d]
session streaming(window: window, hop: hop, max_lag: 64)
"""

def write_dir(d, bp, W):
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "blueprint.fbp"), "w").write(bp)
    save_file({k: v.astype(np.float32) for k, v in W.items()}, os.path.join(d, "model.safetensors"))

# ---- diffusion reference -------------------------------------------------------------------------------------
def alphas_cumprod(T, schedule, b0, b1):
    t = np.arange(T, dtype=np.float64)
    if schedule == "linear": beta = b0 + (b1 - b0) * t / max(1, T - 1)
    elif schedule == "scaled_linear": beta = (np.sqrt(b0) + (np.sqrt(b1) - np.sqrt(b0)) * t / max(1, T - 1)) ** 2
    else:   # cosine (Nichol & Dhariwal), betas clipped at 0.999
        f = lambda u: np.cos((u / T + 0.008) / 1.008 * np.pi / 2) ** 2
        beta = np.minimum(0.999, 1 - f(t + 1) / f(t))
    return np.cumprod(1 - beta)

def denoiser(W, x, t, cond):
    W = {k: v.astype(np.float64) for k, v in W.items()}
    hid = np.tanh(W["w1"] @ x + 0.001 * t * W["wt"] + W["wc"] @ cond + W["b1"])
    return W["w2"] @ hid + W["b2"]

def sample(W, x, cfg, label, ulabel, steps=None, guidance=None):
    """The loop of a deterministic sampler: DDIM with eta 0 on `trailing` timesteps, or Euler on the flow ODE."""
    T, n = cfg["T"], steps or cfg["n"]; g = cfg["g"] if guidance is None else guidance
    emb = W["emb"].astype(np.float64)
    def guided(x, t):
        c = denoiser(W, x, t, emb[label])
        if g == 1: return c
        u = denoiser(W, x, t, emb[ulabel])
        return u + g * (c - u)
    x = x.astype(np.float64)
    if cfg["sampler"] == "euler_flow":   # x_t = (1 - t) x0 + t noise, the model predicts v = noise - x0; t runs 1 -> 0
        for i in range(n):
            t, tn = 1.0 - i / n, 1.0 - (i + 1) / n
            x = x + (tn - t) * guided(x, t * 1000)
        return x
    ab_all = alphas_cumprod(T, cfg["schedule"], cfg["b0"], cfg["b1"])
    ts = [int(np.floor(T - 1 - i * T / n + 0.5)) for i in range(n)]   # trailing spacing, high noise first
    for i, t in enumerate(ts):
        ab = ab_all[t]; abp = ab_all[ts[i + 1]] if i + 1 < n else 1.0
        m = guided(x, t)
        if cfg["pred"] == "epsilon": eps = m; x0 = (x - np.sqrt(1 - ab) * eps) / np.sqrt(ab)
        elif cfg["pred"] == "sample": x0 = m; eps = (x - np.sqrt(ab) * x0) / np.sqrt(1 - ab)
        else: x0 = np.sqrt(ab) * x - np.sqrt(1 - ab) * m; eps = np.sqrt(1 - ab) * x + np.sqrt(ab) * m   # v-prediction
        x = np.sqrt(abp) * x0 + np.sqrt(1 - abp) * eps                # DDIM, eta = 0
    return x

def main():
    a = gu.args(); fails = 0
    rng = np.random.default_rng(2024)
    W = {"w1": rng.normal(0, 0.4, (16, 8)), "wt": rng.normal(0, 0.4, 16), "wc": rng.normal(0, 0.4, (16, 4)), "b1": rng.normal(0, 0.2, 16),
         "w2": rng.normal(0, 0.4, (8, 16)), "b2": rng.normal(0, 0.2, 8), "emb": rng.normal(0, 1.0, (5, 4))}
    W = {k: v.astype(np.float32) for k, v in W.items()}
    x0 = rng.normal(size=8).astype(np.float32)
    f32 = lambda name, v: (lambda p: (np.asarray(v, dtype="<f4").tofile(p), p)[1])(os.path.join(a.work, name))
    init = f32("den_init.f32", x0)
    variants = [dict(T=1000, schedule="scaled_linear", b0=0.00085, b1=0.012, pred="epsilon", sampler="ddim", n=12, g=1),
                dict(T=1000, schedule="linear", b0=0.0001, b1=0.02, pred="v", sampler="ddim", n=9, g=1),
                dict(T=200, schedule="cosine", b0=0.0001, b1=0.02, pred="sample", sampler="ddim", n=10, g=1),
                dict(T=1000, schedule="scaled_linear", b0=0.00085, b1=0.012, pred="epsilon", sampler="ddim", n=8, g=3.5),
                dict(T=1000, schedule="scaled_linear", b0=0.00085, b1=0.012, pred="epsilon", sampler="euler_flow", n=16, g=1)]
    for vi, cfg in enumerate(variants):
        d = os.path.join(a.work, f"den{vi}"); fpk = os.path.join(a.work, f"den{vi}.fpk")
        write_dir(d, DEN_BP % cfg, W)
        gu.run([a.forge, "import", d, "-o", fpk])
        info = gu.run([a.forge, "inspect", fpk])
        ok_meta = "session diffusion_loop" in info and "opset 4" in info
        out = os.path.join(a.work, f"den{vi}.out.f32")
        gu.run([a.forge, "denoise", fpk, "--init", init, "--input", "label=" + f32("lab3.f32", [3]), "--uncond", "label=" + f32("lab1.f32", [1]), "--out", out])
        got = np.fromfile(out, "<f4"); ref = sample(W, x0, cfg, 3, 1)
        err = np.abs(got - ref).max(); tol = 2e-4 * max(1.0, np.abs(ref).max())
        ok = ok_meta and got.size == 8 and err < tol
        print(f"[{'ok' if ok else 'FAIL'}] diffusion_loop {cfg['schedule']}/{cfg['pred']}/{cfg['sampler']}, {cfg['n']} steps, guidance {cfg['g']}: max|err| {err:.2e} (latent magnitude {np.abs(ref).max():.2f})")
        fails += not ok
    # the command line overrides what the file declares: steps, guidance, sampler
    fpk = os.path.join(a.work, "den0.fpk"); out = os.path.join(a.work, "den_over.f32"); cfg = dict(variants[0])
    gu.run([a.forge, "denoise", fpk, "--init", init, "--input", "label=" + os.path.join(a.work, "lab3.f32"), "--steps", "5", "--guidance", "2", "--out", out])
    ref = sample(W, x0, cfg, 3, 0, steps=5, guidance=2)     # no --uncond: the unconditional branch is the encoder on zeros (class 0)
    err = np.abs(np.fromfile(out, "<f4") - ref).max(); ok = err < 2e-4 * max(1.0, np.abs(ref).max())
    print(f"[{'ok' if ok else 'FAIL'}] --steps 5 --guidance 2 override the file: max|err| {err:.2e}"); fails += not ok
    # seeded noise: the same seed gives the same latent, another seed another one; DDPM is seeded too
    def den(*extra):
        o = os.path.join(a.work, "den_seed.f32"); gu.run([a.forge, "denoise", fpk, "--out", o, *extra]); return np.fromfile(o, "<f4")
    s1, s1b, s2 = den("--seed", "7"), den("--seed", "7"), den("--seed", "8")
    p1, p1b = den("--seed", "7", "--sampler", "ddpm"), den("--seed", "7", "--sampler", "ddpm")
    ok = np.array_equal(s1, s1b) and not np.array_equal(s1, s2) and np.array_equal(p1, p1b) and not np.array_equal(p1, s1) and np.isfinite(p1).all()
    print(f"[{'ok' if ok else 'FAIL'}] seeded noise and the DDPM sampler are reproducible per seed"); fails += not ok
    # a diffusion model is not a language model, and the other way round
    r = subprocess.run(os.environ.get("FORGE_RUNNER", "").split() + [a.forge, "stream", fpk, "--input", init], capture_output=True, text=True, errors="replace")
    ok = r.returncode != 0 and "not a streaming model" in r.stdout + r.stderr
    print(f"[{'ok' if ok else 'FAIL'}] forge stream refuses a diffusion model"); fails += not ok

    # ---- streaming ------------------------------------------------------------------------------------------
    S = {"proj": rng.normal(0, 0.3, (6, 16)).astype(np.float32), "bias": rng.normal(0, 0.2, 6).astype(np.float32), "cw": rng.normal(0, 0.5, (6, 3)).astype(np.float32)}
    d = os.path.join(a.work, "stream"); fpk = os.path.join(a.work, "stream.fpk")
    write_dir(d, STREAM_BP, S); gu.run([a.forge, "import", d, "-o", fpk])
    info = gu.run([a.forge, "inspect", fpk, "--state"])
    ok_meta = "session streaming" in info and "1 recurrent" in info and "causal_conv" in info
    sig = rng.normal(size=8 * 57 + 16 + 5).astype(np.float32); sp = f32("stream_in.f32", sig)
    n = (sig.size - 16) // 8 + 1
    frames = np.stack([sig[i * 8: i * 8 + 16] for i in range(n)]).astype(np.float64)
    feat = frames @ S["proj"].astype(np.float64).T + S["bias"]                       # [n, 6]
    hist = np.concatenate([np.zeros((2, 6)), feat])                                  # the window before the first frame is zeros
    ref = np.tanh(sum(S["cw"][:, j].astype(np.float64) * hist[j: j + n] for j in range(3)))   # y[t, c] = sum_j cw[c, j] f[t - 2 + j, c]
    outs = []
    for chunk in ("1", "8", "13", "200", str(sig.size)):
        o = os.path.join(a.work, f"stream_out_{chunk}.f32")
        r = subprocess.run(os.environ.get("FORGE_RUNNER", "").split() + [a.forge, "stream", fpk, "--input", sp, "--chunk", chunk, "--out", o], capture_output=True, text=True, errors="replace")
        if r.returncode != 0: print("FAILED: forge stream", r.stdout, r.stderr); sys.exit(1)
        outs.append(np.fromfile(o, "<f4")); ok_meta = ok_meta and f"{n} frames, 0 dropped" in r.stderr
    err = np.abs(outs[0] - ref.ravel()).max() if outs[0].size == ref.size else np.inf
    ok = ok_meta and err < 1e-5 and all(np.array_equal(outs[0], o) for o in outs[1:])
    print(f"[{'ok' if ok else 'FAIL'}] streaming: {n} frames of 16 / hop 8, causal conv state carried across frames: max|err| {err:.2e}; chunk sizes 1, 8, 13, 200 and all-at-once give the same bytes")
    fails += not ok
    print("golden:", "PASS" if fails == 0 else f"{fails} FAILURE(S)")
    sys.exit(1 if fails else 0)

if __name__ == "__main__": main()

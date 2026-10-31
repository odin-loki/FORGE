#!/usr/bin/env python3
"""Golden end-to-end test for the qwen3_5 template (Gated DeltaNet + gated full attention, partial RoPE):
a random tiny Qwen3.5 text model built with transformers, imported with `forge import`.

  1. prefill logits (FORGE recurrent delta rule vs transformers' chunked delta rule)
  2. greedy decode with state (conv window, recurrent state, KV cache) vs transformers generate()
  3. the same through the multimodal wrapper layout (model.language_model.*, text_config)
Needs torch + transformers >= 5 (qwen3_5); skips otherwise.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny_llama as tl

TOK = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
           vocab_size=300, rms_norm_eps=1e-6, rope_theta=1000000.0, max_position_embeddings=128, bos_token_id=290,
           eos_token_id=291, tie_word_embeddings=True)
CFG = dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=8, num_attention_heads=4,
           num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-6, max_position_embeddings=128, tie_word_embeddings=False,
           linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
           linear_num_value_heads=4, full_attention_interval=4, hidden_act="silu", bos_token_id=290, eos_token_id=291,
           rope_parameters=dict(rope_type="default", rope_theta=1000000.0, partial_rotary_factor=0.25,
                                mrope_section=[2, 1, 1], mrope_interleaved=True))

def make_model(d, seed=97):
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    saved = dict(tl.CFG); tl.CFG.update(TOK)
    try: tl.make_model(d)  # tokenizer.json (the llama weights are overwritten below)
    finally: tl.CFG.clear(); tl.CFG.update(saved)
    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(**CFG); cfg._attn_implementation = "eager"
    m = Qwen3_5ForCausalLM(cfg).float().eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_parameters():   # break the zero/one inits so every term matters
            if n.endswith("norm.weight"): p.add_(0.2 * torch.randn(p.shape, generator=g))
            elif n.endswith("A_log"): p.copy_(torch.log(torch.rand(p.shape, generator=g) * 4 + 0.05))
            elif n.endswith("dt_bias"): p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif p.dim() >= 2: p.normal_(0, 0.6 / np.sqrt(p.shape[-1]), generator=g)
    for f in os.listdir(d):
        if f.endswith(".safetensors"): os.remove(os.path.join(d, f))
    m.save_pretrained(d, safe_serialization=True)
    return m

def wrap_multimodal(src, dst):
    """Rewrite a text checkpoint into the Qwen3_5ForConditionalGeneration layout (text_config + language_model)."""
    from safetensors.numpy import load_file, save_file
    os.makedirs(dst, exist_ok=True)
    W = {}
    for f in os.listdir(src):
        if f.endswith(".safetensors"): W.update(load_file(os.path.join(src, f)))
    W = {(k.replace("model.", "model.language_model.", 1) if k.startswith("model.") else k): v for k, v in W.items()}
    W["model.visual.dummy.weight"] = np.zeros((2, 2), np.float32)   # ignored by the importer
    save_file(W, os.path.join(dst, "model.safetensors"))
    tc = json.load(open(os.path.join(src, "config.json"))); tc.pop("architectures", None); tc["model_type"] = "qwen3_5_text"
    json.dump(dict(architectures=["Qwen3_5ForConditionalGeneration"], model_type="qwen3_5", text_config=tc,
                   tie_word_embeddings=False), open(os.path.join(dst, "config.json"), "w"), indent=1)
    for f in ("tokenizer.json",):
        open(os.path.join(dst, f), "w").write(open(os.path.join(src, f)).read())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--forge", required=True); ap.add_argument("--work", default="build/golden_qwen3_5")
    a = ap.parse_args(); run = tl.run
    try:
        import torch
        from transformers import Qwen3_5ForCausalLM  # noqa: F401
    except ImportError:
        print("[skip] torch / transformers with qwen3_5 not installed"); print("golden qwen3_5: SKIP"); return
    hf = os.path.join(a.work, "hf"); m = make_model(hf)
    prompt = [290, 72, 101, 108, 108, 111, 44, 32, 116, 104, 101, 32, 119, 111, 114, 108, 100, 33, 32, 84]
    with torch.no_grad():
        ref = m(torch.tensor([prompt])).logits[0, -1].double().numpy()
        n_gen = 24
        out = m.generate(torch.tensor([prompt]), max_new_tokens=n_gen, do_sample=False, eos_token_id=None, pad_token_id=0)
        ref_gen = out[0, len(prompt):].tolist()
    fails = 0
    def check(fp, label, tol=None, cosmin=None, gen_n=1, ref=ref):
        nonlocal fails
        lf = fp + ".logits.f32"
        gen = run([a.forge, "run", fp, "--tokens", ",".join(map(str, prompt)), "-n", str(gen_n), "--ids", "--logits", lf])
        got = np.fromfile(lf, dtype=np.float32)
        err = np.abs(got - ref).max(); cos = got @ ref / np.linalg.norm(got) / np.linalg.norm(ref)
        ok = err < tol if tol else cos > cosmin
        print(f"[{'ok' if ok else 'FAIL'}] {label}: prefill logits max|err| {err:.2e} (|logit| max {np.abs(ref).max():.1f}), cosine {cos:.6f}")
        fails += not ok
        return gen
    def greedy(gen, label):
        nonlocal fails
        fg = [int(t) for t in gen.split()]; same = fg == ref_gen
        if not same:   # tolerate a flip only at a near-tie
            i = next((i for i in range(min(len(fg), len(ref_gen))) if fg[i] != ref_gen[i]), min(len(fg), len(ref_gen)))
            with torch.no_grad():
                lg = np.sort(m(torch.tensor([prompt + ref_gen[:i]])).logits[0, -1].numpy())
            same = i == len(ref_gen) or lg[-1] - lg[-2] < 1e-3
        print(f"[{'ok' if same else 'FAIL'}] {label}: greedy decode matches transformers: forge {fg[:8]}... ref {ref_gen[:8]}...")
        fails += not same
    out = os.path.join(a.work, "qwen3_5_f32.fpk")
    run([a.forge, "import", hf, "-o", out, "--codec", "f32"])
    greedy(check(out, "hf f32", 1e-3, None, n_gen), "hf f32")
    # q8_0: this random tiny model amplifies weight rounding (cosine ~0.97 vs the f32 model), so compare against
    # transformers running the same q8_0-rounded weights (block 32, scale = amax / 127, round half to even)
    import copy
    mq = copy.deepcopy(m)
    with torch.no_grad():
        for n, p in mq.named_parameters():
            if p.dim() == 2 and "embed_tokens" not in n and "in_proj_a" not in n and "in_proj_b" not in n:
                w = p.reshape(-1, 32); sc = w.abs().amax(1, keepdim=True) / 127; sc[sc == 0] = 1
                p.copy_((torch.round(w / sc) * sc).reshape(p.shape))
        ref_q = mq(torch.tensor([prompt])).logits[0, -1].double().numpy()
    out = os.path.join(a.work, "qwen3_5_q8_0.fpk")
    run([a.forge, "import", hf, "-o", out, "--codec", "q8_0"])
    check(out, "hf q8_0 (vs q8_0-rounded transformers)", None, 0.9999, 1, ref_q)
    mm = os.path.join(a.work, "hf_mm"); wrap_multimodal(hf, mm)
    out = os.path.join(a.work, "qwen3_5_mm.fpk")
    run([a.forge, "import", mm, "-o", out])
    greedy(check(out, "multimodal layout f32", 1e-3, None, n_gen), "multimodal layout f32")
    print("golden qwen3_5:", "PASS" if fails == 0 else f"{fails} FAILURE(S)")
    sys.exit(1 if fails else 0)

if __name__ == "__main__": main()

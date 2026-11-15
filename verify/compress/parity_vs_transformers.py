#!/usr/bin/env python3
"""Oracle test: FORGE vs Hugging Face transformers on the same token ids.

transformers is used here ONLY as an independent reference to prove FORGE runs the published checkpoints
correctly; no framework computation depends on it.

usage: parity_vs_transformers.py HF_DIR MODEL.fpk [--trust-remote-code]
Checks: tokenizer ids identical; per-position logits max|diff|; mean NLL difference; greedy-argmax agreement.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("FORGE_PY", "/home/user/FORGE/python"))
import forge  # noqa: E402

TEXT = ("The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest "
        "structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, "
        "the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world.")


def nll(logits: np.ndarray, ids: np.ndarray) -> float:
    l = logits[:-1].astype(np.float64)
    m = l.max(-1, keepdims=True)
    lse = (m[:, 0] + np.log(np.exp(l - m).sum(-1)))
    return float((lse - l[np.arange(len(ids) - 1), ids[1:]]).mean())


def main() -> int:
    hf_dir, fpk = sys.argv[1], sys.argv[2]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_dir)
    hf_ids = tok(TEXT)["input_ids"]

    fm = forge.Model(fpk)
    f_ids = list(fm.tokenize(TEXT))
    print(f"tokens: hf={len(hf_ids)} forge={len(f_ids)} identical={hf_ids == f_ids}")
    if hf_ids != f_ids:
        for i, (a, b) in enumerate(zip(hf_ids, f_ids)):
            if a != b:
                print(f"  first mismatch at {i}: hf={hf_ids[i:i+5]} forge={f_ids[i:i+5]}")
                break

    ids = np.array(hf_ids, dtype=np.int64)
    fm.reset()
    fl = np.asarray(fm.forward_all([int(x) for x in ids]), dtype=np.float32).reshape(len(ids), -1)

    model = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        hl = model(torch.tensor(ids)[None]).logits[0].float().numpy()
    v = min(fl.shape[1], hl.shape[1])
    d = np.abs(fl[:, :v] - hl[:, :v])
    agree = float((fl[:, :v].argmax(-1) == hl[:, :v].argmax(-1)).mean())
    n_f, n_h = nll(fl[:, :v], ids), nll(hl[:, :v], ids)
    print(f"logits max|diff|={d.max():.4g} mean|diff|={d.mean():.4g} scale={np.abs(hl).max():.3g}")
    print(f"NLL forge={n_f:.5f} hf={n_h:.5f} (ppl {np.exp(n_f):.3f} vs {np.exp(n_h):.3f}); argmax agree={agree:.4f}")
    ok = abs(n_f - n_h) < 1e-3 and agree > 0.98
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

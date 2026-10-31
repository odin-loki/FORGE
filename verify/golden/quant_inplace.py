#!/usr/bin/env python3
"""Quantised weights are read in place (matmul_t on encoded rows): logits must match the decode-everything path
(FORGE_DECODE_WEIGHTS=1) for every codec, on a tiny random Qwen3 model."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny_qwen3 as tq, tiny_llama as tl

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--forge", required=True); ap.add_argument("--work", default="build/golden_quant_inplace")
    a = ap.parse_args()
    tq.CFG.update(hidden_size=256, intermediate_size=512, head_dim=64)  # k divisible by the 256-element k-quant blocks
    hf = os.path.join(a.work, "hf"); tq.make_model(hf)
    prompt = "290,72,101,108,108,111,44,32,116"
    fails = 0
    for codec in ("f32", "f16", "q8_0", "q4_k", "q8_g32"):
        fp = os.path.join(a.work, f"m_{codec}.fpk")
        tl.run([a.forge, "import", hf, "-o", fp, "--codec", codec])
        outs = {}
        for mode in ("0", "1"):
            env = dict(os.environ, FORGE_DECODE_WEIGHTS=mode); lf = fp + f".{mode}.f32"
            import subprocess
            ids = subprocess.run([a.forge, "run", fp, "--tokens", prompt, "-n", "8", "--ids", "--logits", lf], env=env,
                                 capture_output=True, text=True, check=True).stdout.split()
            outs[mode] = (np.fromfile(lf, dtype=np.float32), ids)
        (l0, i0), (l1, i1) = outs["0"], outs["1"]
        err = float(np.abs(l0 - l1).max()) / max(float(np.abs(l1).max()), 1e-9)
        ok = err < 1e-4 and i0 == i1
        print(f"[{'ok' if ok else 'FAIL'}] {codec}: in-place vs decoded logits rel err {err:.1e}, tokens {'equal' if i0 == i1 else 'DIFFER'}")
        fails += not ok
    print("quant in-place:", "PASS" if not fails else f"{fails} FAILURE(S)"); sys.exit(1 if fails else 0)

if __name__ == "__main__": main()

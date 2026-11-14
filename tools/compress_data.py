#!/usr/bin/env python3
"""Download evaluation / calibration data and write it as plain text / JSONL for forge compress / forge eval.

Only data plumbing lives here (download + prompt formatting, following lm-evaluation-harness conventions);
tokenisation, model execution and every number are computed by FORGE in `forge compress` / `forge eval`.

Outputs (in OUT_DIR):
  wikitext2_test.txt        WikiText-2 raw test split, joined with "\n\n" (lm-eval wikitext convention)
  wikitext2_train.txt       WikiText-2 raw train split (alternate calibration source)
  c4_calib.jsonl            {"text": ...} documents from the C4 en validation shard (calibration, as Wanda/SparseGPT)
  tasks/<name>.jsonl        {"ctxs": [...], "conts": [...], "label": int, "norm": bool}
                            ctxs has one entry (shared context) or one per choice (WinoGrande partial scoring).
usage: prepare_data.py OUT_DIR
"""
import io
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

API = "https://datasets-server.huggingface.co/parquet?dataset={}&config={}"


def parquet_urls(dataset: str, config: str, split: str):
    with urllib.request.urlopen(API.format(dataset, config)) as r:
        d = json.load(r)
    urls = [f["url"] for f in d["parquet_files"] if f["split"] == split]
    if not urls:
        raise SystemExit(f"no parquet for {dataset}/{config}/{split}")
    return urls


def read_table(dataset, config, split, max_files=1):
    rows = []
    for u in parquet_urls(dataset, config, split)[:max_files]:
        with urllib.request.urlopen(u) as r:
            rows.extend(pq.read_table(io.BytesIO(r.read())).to_pylist())
    return rows


def hellaswag_pre(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def write_jsonl(path: Path, items):
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{path.name}: {len(items)}")


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks").mkdir(exist_ok=True)

    for split in ("test", "train"):
        rows = read_table("Salesforce/wikitext", "wikitext-2-raw-v1", split)
        (out / f"wikitext2_{split}.txt").write_text("\n\n".join(r["text"] for r in rows))
        print(f"wikitext2_{split}: {len(rows)} lines")

    rows = read_table("allenai/c4", "en", "validation")
    rnd = random.Random(0)
    rnd.shuffle(rows)
    write_jsonl(out / "c4_calib.jsonl", [{"text": r["text"]} for r in rows[:4000]])

    tasks = {}
    for cfg, name in (("ARC-Easy", "arc_easy"), ("ARC-Challenge", "arc_challenge")):
        items = []
        for r in read_table("allenai/ai2_arc", cfg, "test"):
            labels = list(r["choices"]["label"])
            if r["answerKey"] not in labels:
                continue
            items.append({"ctxs": [f"Question: {r['question']}\nAnswer:"],
                          "conts": [" " + t for t in r["choices"]["text"]],
                          "label": labels.index(r["answerKey"]), "norm": True})
        tasks[name] = items

    items = []
    for r in read_table("Rowan/hellaswag", "default", "validation"):
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        items.append({"ctxs": [hellaswag_pre(r["activity_label"] + ": " + ctx)],
                      "conts": [" " + hellaswag_pre(e) for e in r["endings"]],
                      "label": int(r["label"]), "norm": True})
    tasks["hellaswag"] = items

    items = []
    for r in read_table("baber/piqa", "default", "validation"):
        items.append({"ctxs": [f"Question: {r['goal']}\nAnswer:"], "conts": [" " + r["sol1"], " " + r["sol2"]],
                      "label": int(r["label"]), "norm": False})
    tasks["piqa"] = items

    items = []
    for r in read_table("allenai/winogrande", "winogrande_xl", "validation"):
        s = r["sentence"]
        i = s.index("_")
        cont = " " + s[i + 1:].strip()
        items.append({"ctxs": [s[:i] + r["option1"], s[:i] + r["option2"]], "conts": [cont, cont],
                      "label": int(r["answer"]) - 1, "norm": False})
    tasks["winogrande"] = items

    items = []
    for r in read_table("google/boolq", "default", "validation"):
        items.append({"ctxs": [f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:"], "conts": [" no", " yes"],
                      "label": int(bool(r["answer"])), "norm": False})
    tasks["boolq"] = items

    items = []
    for r in read_table("allenai/openbookqa", "main", "test"):
        labels = list(r["choices"]["label"])
        items.append({"ctxs": [r["question_stem"]], "conts": [" " + t for t in r["choices"]["text"]],
                      "label": labels.index(r["answerKey"]), "norm": True})
    tasks["openbookqa"] = items

    for name, items in tasks.items():
        rnd = random.Random(1234)
        rnd.shuffle(items)   # fixed order; the evaluator takes the first N
        write_jsonl(out / "tasks" / f"{name}.jsonl", items)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))

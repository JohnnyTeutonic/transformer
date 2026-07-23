#!/usr/bin/env python3
"""Verify an --export-hf directory loads in real HuggingFace Transformers and
generates identically to this repo's own inference — i.e. that the RoPE Q/K
permute (interleaved -> rotate_half) is correct.

Usage:
  # compare against a saved engine logits row (TINYLLAMA_LOGITS_DUMP output):
  verify_hf_export.py <hf_dir> --prompt "user: how are you ? assistant:" \
      --engine-logits eng.tsv
  # or just sanity-print HF's top predictions (no reference):
  verify_hf_export.py <hf_dir> --prompt "user: how are you ? assistant:"

The permute is CORRECT when argmax + top-5 match the engine and the logit
vectors correlate ~1.0; WRONG shows near-zero correlation and a different
argmax (HF's rotate_half applied to un-permuted weights is garbage).
"""
import argparse, json, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--engine-logits", default=None,
                    help="TSV whose first line is the engine's last-position logits")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    vocab = json.load(open(f"{a.hf_dir}/vocab.json"))
    inv = {v: k for k, v in vocab.items()}
    ids = []
    for w in a.prompt.split():
        if w not in vocab:
            sys.exit(f"token not in vocab: {w!r}")
        ids.append(vocab[w])

    model = AutoModelForCausalLM.from_pretrained(a.hf_dir, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0, -1].numpy()

    top = np.argsort(-logits)[:5]
    print("HF top-5:", [(int(t), inv.get(int(t), "?")) for t in top])

    if not a.engine_logits:
        return
    en = np.array([float(x) for x in open(a.engine_logits).readline().split()])
    n = min(len(logits), len(en))
    hf, en = logits[:n], en[:n]
    ok = (int(hf.argmax()) == int(en.argmax())
          and len(set(np.argsort(-hf)[:5].tolist()) & set(np.argsort(-en)[:5].tolist())) >= 4
          and float(np.corrcoef(hf, en)[0, 1]) > 0.9)
    print(f"engine argmax {int(en.argmax())} ({inv.get(int(en.argmax()),'?')!r}); "
          f"corr {float(np.corrcoef(hf, en)[0,1]):.4f}")
    print("PERMUTE", "CORRECT" if ok else "WRONG")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

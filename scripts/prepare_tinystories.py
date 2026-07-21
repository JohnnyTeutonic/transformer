"""Prepare TinyStories for transformer_cpp's train_wikitext data contract.

Writes, into --out (default data/tinystories-txt):
    train-0.txt / train-1.txt          one story per line (newlines -> spaces)
    train-0-small.txt / train-1-small.txt   truncated variants for --small runs
    validation.txt                     validation split

The trainer builds its word-level vocabulary from these files directly, so no
tokenizer artifacts are needed. Run from the repo root:

    python scripts/prepare_tinystories.py [--stories-per-file 300000]

Works locally (Windows) and on Colab: uses `datasets` when available, else
falls back to huggingface_hub + pandas parquet download.
"""
import argparse
import os


def iter_stories(split):
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split=split)
        for t in ds["text"]:
            yield t
        return
    except ImportError:
        pass
    # Fallback: download the split's parquet shards via huggingface_hub
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = [f for f in api.list_repo_files("roneneldan/TinyStories", repo_type="dataset")
             if f.endswith(".parquet") and split.split("[")[0] in f]
    if not files:
        raise RuntimeError(f"no parquet shards found for split {split}")
    for fname in sorted(files):
        local = hf_hub_download("roneneldan/TinyStories", fname, repo_type="dataset")
        for t in pd.read_parquet(local)["text"]:
            yield t


def clean(story: str) -> str:
    return " ".join(str(story).split())


# Story separator: a single whitespace-delimited word for the word-level
# tokenizer. The trainer's packing loader concatenates lines, so without it
# stories butt together and the model has no learnable story boundary (and
# nothing to emit to stop generation). The GGUF exporter prefers this token
# as EOS when present, so inference stops at story end.
EOT = "<|endoftext|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tinystories-txt")
    ap.add_argument("--stories-per-file", type=int, default=300_000,
                    help="stories in each of train-0.txt / train-1.txt")
    ap.add_argument("--small-stories", type=int, default=5_000,
                    help="stories in each *-small.txt smoke-test file")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    n_train = 2 * args.stories_per_file
    paths = {k: os.path.join(args.out, k) for k in
             ("train-0.txt", "train-1.txt", "train-0-small.txt",
              "train-1-small.txt", "validation.txt")}

    written = 0
    f0 = open(paths["train-0.txt"], "w", encoding="utf-8")
    f1 = open(paths["train-1.txt"], "w", encoding="utf-8")
    s0 = open(paths["train-0-small.txt"], "w", encoding="utf-8")
    s1 = open(paths["train-1-small.txt"], "w", encoding="utf-8")
    for story in iter_stories("train"):
        line = clean(story)
        if not line:
            continue
        tgt = f0 if written < args.stories_per_file else f1
        tgt.write(line + " " + EOT + "\n")
        if written < args.small_stories:
            s0.write(line + " " + EOT + "\n")
        elif written < 2 * args.small_stories:
            s1.write(line + " " + EOT + "\n")
        written += 1
        if written % 100_000 == 0:
            print(f"  train: {written}/{n_train} stories", flush=True)
        if written >= n_train:
            break
    for f in (f0, f1, s0, s1):
        f.close()
    print(f"train: {written} stories", flush=True)

    with open(paths["validation.txt"], "w", encoding="utf-8") as fv:
        n_val = 0
        for story in iter_stories("validation"):
            line = clean(story)
            if line:
                fv.write(line + " " + EOT + "\n")
                n_val += 1
    print(f"validation: {n_val} stories", flush=True)

    for name, p in paths.items():
        print(f"  {name}: {os.path.getsize(p)/1e6:.1f} MB")


if __name__ == "__main__":
    main()

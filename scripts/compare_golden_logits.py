#!/usr/bin/env python3
"""Golden-batch GGUF-parity comparator (see golden_batch_test.cpp epigraph:
A LYING METRIC MIMICS EVERY OTHER FAILURE MODE).

Trainer side:
  build_wsl/golden_batch_test --ckpt latest_tcpp.ckpt \
      --vocab-file /tmp/tinychat-txt/train-0.txt \
      --vocab-file /tmp/tinychat-txt/train-1.txt \
      --mid --seq-len 256 \
      --prompt "user: how are you ? assistant:" ... --dump trainer_logits.tsv

Engine side (one call PER prompt, fresh dump file, first line = the logits
that sample the first reply token, i.e. the trainer's last-position row):
  TINYLLAMA_LOGITS_DUMP=eng.tsv tinyllama model.gguf model.gguf 4 prompt \
      "<same prompt>" --max-tokens 1 --top-k 1 --n-gpu-layers 0 --raw-prompt

Then:
  compare_golden_logits.py trainer_logits.tsv eng_prompt0.tsv eng_prompt1.tsv ...
(engine files given in the same order as the trainer's --prompt flags)

Reports per-prompt max|delta|, argmax agreement, and top-5 rank overlap.
Argmax agreement with small deltas => stacks agree (any anomaly is the
model itself). Diverging logits => train/deploy divergence localized to the
engine forward — bisect with the engine's [CPU_FWD] layer logs.
"""

import sys


def read_trainer(path):
    rows = []
    with open(path) as f:
        for line in f:
            prompt, vec = line.rstrip("\n").split("\t")
            rows.append((prompt, [float(x) for x in vec.split()]))
    return rows


def read_engine_first_row(path):
    with open(path) as f:
        line = f.readline().strip()
    return [float(x) for x in line.split()]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    trainer = read_trainer(sys.argv[1])
    engine_files = sys.argv[2:]
    if len(engine_files) != len(trainer):
        sys.exit(f"{len(trainer)} trainer rows but {len(engine_files)} engine files")

    all_ok = True
    for (prompt, tv), ef in zip(trainer, engine_files):
        ev = read_engine_first_row(ef)
        n = min(len(tv), len(ev))
        if len(tv) != len(ev):
            print(f"  (vocab size mismatch: trainer {len(tv)} vs engine {len(ev)})")
        deltas = [abs(a - b) for a, b in zip(tv[:n], ev[:n])]
        maxd = max(deltas)
        t_arg = max(range(n), key=lambda i: tv[i])
        e_arg = max(range(n), key=lambda i: ev[i])
        t_top5 = set(sorted(range(n), key=lambda i: -tv[i])[:5])
        e_top5 = set(sorted(range(n), key=lambda i: -ev[i])[:5])
        agree = t_arg == e_arg
        ok = agree and maxd < 0.05
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] max|d|={maxd:.5f} "
              f"argmax {'==' if agree else f'{t_arg}!={e_arg}'} "
              f"top5-overlap {len(t_top5 & e_top5)}/5  :: {prompt[:50]!r}")
    print("PARITY:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

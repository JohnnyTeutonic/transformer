"""VM boot: chat v7a — MID model (4L/256h), seq 256, dialogmix corpus.

Enterprise-era lane A. DailyDialog + EmpatheticDialogues (~38k dialogues,
everyday + emotional registers), context doubled to 256 words, first
chat-capable core (~3.2M core params, 8x tiny). All 2026-07 fixes included:
FFN cache invalidation, format-3 checkpoints, hashes, batched-judging probe.
Batch 32 (halved: seq 256 doubles per-row cost; keeps logits buffer and step
time in T4 comfort).
"""
import subprocess

subprocess.Popen(
    "cd /content && nohup python colab_train_tinystories.py "
    "--mid --seq-len 256 --dataset dialogmix --steps 6000 --epochs 30 "
    "--batch-size 32 --export both --lr 6e-4 --cosine-decay "
    "> /content/tcpp_train.log 2>&1 & echo $!",
    shell=True, executable="/bin/bash")
print("LAUNCHED", flush=True)

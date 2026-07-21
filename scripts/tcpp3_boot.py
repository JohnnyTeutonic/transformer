"""VM boot: chat coherence v3 — full-Adam trainer (2026-07-13 ~03:30).

v1 (2e-4) and v2 (2e-3) both plateaued at loss ~5.4-7: the trainer ran a
franken-optimizer (Adam on the LM head only, SGD-momentum on the core,
embeddings NEVER updated — the backward chain dropped the layer-0 gradient).
This build gives every module Adam, wires the token-embedding update into
the backward pass, and updates final_ln (also previously frozen).
CPU smoke test: loss 2.39 -> 0.60 in 60 steps on a toy corpus.
LR 6e-4 is a true Adam-scale rate (nanoGPT territory for this size).
"""
import subprocess

subprocess.Popen(
    "cd /content && nohup python colab_train_tinystories.py "
    "--tiny --dataset instruct --steps 6000 --export both --lr 6e-4 "
    "> /content/tcpp_train.log 2>&1 & echo $!",
    shell=True, executable="/bin/bash")
print("LAUNCHED", flush=True)

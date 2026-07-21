"""VM boot: chat coherence v4 — full-Adam + cosine LR decay (2026-07-13 ~05:10).

v3 (Adam, constant 6e-4) proved the optimizer port: loss 6.9 -> 3.36 by step
~2900 (v1/v2 never broke 4.1). But constant LR destabilized late — loss rose
back to ~5.3 in epoch 2, and the rolling checkpoint kept only the degraded
weights. v4 = same recipe + cosine decay to 10%, so late training anneals
instead of oscillating and the final export IS the best model.
"""
import subprocess

subprocess.Popen(
    "cd /content && nohup python colab_train_tinystories.py "
    "--tiny --dataset instruct --steps 6000 --export both --lr 6e-4 --cosine-decay "
    "> /content/tcpp_train.log 2>&1 & echo $!",
    shell=True, executable="/bin/bash")
print("LAUNCHED", flush=True)

"""VM boot: chat coherence run v2 — LR 2e-3 (10x trainer default).

The v1 chat run plateaued at loss ~5.4-6.0 from step 500 onward: the
SGD-momentum update at LR 2e-4 with per-tensor grad clipping (norm 1.0)
bounds per-step movement so tightly the model can only fit unigram/bigram
statistics in 6000 steps. Clipping also bounds the risk of the higher LR.
"""
import subprocess

subprocess.Popen(
    "cd /content && nohup python colab_train_tinystories.py "
    "--tiny --dataset instruct --steps 6000 --export both --lr 2e-3 "
    "> /content/tcpp_train.log 2>&1 & echo $!",
    shell=True, executable="/bin/bash")
print("LAUNCHED", flush=True)

"""VM boot: chat v7b — MID model (4L/256h), seq 256, FIXED instruct corpus.

Enterprise-era lane B. TinyStories-Instruct with the aggregation fix
(fields + story now share one example; words:/summary: conditioning is
finally learnable). Same mid config and fixes as lane A.
"""
import subprocess

subprocess.Popen(
    "cd /content && nohup python colab_train_tinystories.py "
    "--mid --seq-len 256 --dataset instruct --steps 6000 "
    "--batch-size 32 --export both --lr 6e-4 --cosine-decay "
    "> /content/tcpp_train.log 2>&1 & echo $!",
    shell=True, executable="/bin/bash")
print("LAUNCHED", flush=True)

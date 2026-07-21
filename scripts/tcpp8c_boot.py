"""VM boot: chat v8c — TinyChat backbone + DailyDialog variety."""
import subprocess

subprocess.Popen(
    "cd /content && nohup python colab_train_tinystories.py "
    "--mid --seq-len 256 --dataset tinychatmix --steps 8000 --epochs 30 --doc-aligned --batch-size 32 --export both --lr 6e-4 --cosine-decay "
    "> /content/tcpp_train.log 2>&1 & echo $!",
    shell=True, executable="/bin/bash")
print("LAUNCHED", flush=True)

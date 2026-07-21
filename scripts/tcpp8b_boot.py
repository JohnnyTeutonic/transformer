"""VM boot: chat v8b — coherence via capacity-matched synthetic TinyChat (145-word vocab, consistent QA semantics)."""
import subprocess

subprocess.Popen(
    "cd /content && nohup python colab_train_tinystories.py "
    "--mid --seq-len 256 --dataset tinychat --steps 12000 --epochs 40 --doc-aligned --batch-size 32 --export both --lr 6e-4 --cosine-decay "
    "> /content/tcpp_train.log 2>&1 & echo $!",
    shell=True, executable="/bin/bash")
print("LAUNCHED", flush=True)

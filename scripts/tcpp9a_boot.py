"""VM boot: chat v9a — assistant-only loss arm. Identical to 8b (TinyChat,
12k steps, doc-aligned) except --assistant-loss: loss masked on user-turn
tokens and role markers, so every update optimizes P(answer|question).
Full-CE control is chat8b itself (same data, same config, same steps).
Compare via eval_tinychat_generalisation.py rows."""
import subprocess

subprocess.Popen(
    "cd /content && nohup python colab_train_tinystories.py "
    "--mid --seq-len 256 --dataset tinychat --steps 12000 --epochs 40 --doc-aligned --assistant-loss --batch-size 32 --export both --lr 6e-4 --cosine-decay "
    "> /content/tcpp_train.log 2>&1 & echo $!",
    shell=True, executable="/bin/bash")
print("LAUNCHED", flush=True)

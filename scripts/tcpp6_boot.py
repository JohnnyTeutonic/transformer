"""VM boot: chat v6 — REAL dialogue (DailyDialog) on the FFN-fixed trainer.

First conversational (user:/assistant:) run. DailyDialog is small (13k
dialogues ~ a few thousand packed sequences), so one epoch is only a few
hundred steps: run many epochs with cosine decay to 3000 steps total.
Vocabulary is richer than TinyStories — expect some <unk>; judged
acceptable for the first chat-format model.
"""
import subprocess

subprocess.Popen(
    "cd /content && nohup python colab_train_tinystories.py "
    "--tiny --dataset dailydialog --steps 3000 --epochs 40 --export both "
    "--lr 6e-4 --cosine-decay "
    "> /content/tcpp_train.log 2>&1 & echo $!",
    shell=True, executable="/bin/bash")
print("LAUNCHED", flush=True)

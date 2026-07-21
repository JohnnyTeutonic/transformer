"""VM boot: chat v5 — eos-separated packing + loss ignore-index (2026-07-17).

v4 carried the full-Adam + cosine-decay recipe but its training stream had no
document boundaries: prepare_chat_data.py packed dialogues back-to-back with
no separator, so termination was unlearnable and dialogues bled into each
other. v5 source adds (1) a <|endoftext|> token after every dialogue (the
GGUF exporter prefers it as EOS, so the served model stops at reply end) and
(2) an ignore-index in the CE loss: target 0 (PAD/<unk>) now contributes no
loss/gradient — this also stops every row's unfilled last position from being
trained toward <unk>. Recipe otherwise identical to v4 (tiny 2L/128h,
TinyStories-Instruct, 6000 steps, lr 6e-4, cosine decay).
"""
import subprocess

subprocess.Popen(
    "cd /content && nohup python colab_train_tinystories.py "
    "--tiny --dataset instruct --steps 6000 --export both --lr 6e-4 --cosine-decay "
    "> /content/tcpp_train.log 2>&1 & echo $!",
    shell=True, executable="/bin/bash")
print("LAUNCHED", flush=True)

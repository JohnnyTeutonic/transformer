"""VM boot: one-shot sub-op divergence diagnostic (2026-07-19).

Restores the frozen step-3000 checkpoint (pushed as restore_tcpp.ckpt),
forces the resume past the probe, and runs ONLY the probe forwards with
TCPP_LAYER_TRACE on (--steps == ckpt step -> zero training). The [SUBOP]
lines are the CUDA build's per-sub-op activation L2s on the exact state the
CPU build measures locally; the first diverging column names the broken op.
"""
import subprocess

subprocess.Popen(
    "cd /content && nohup env TCPP_FORCE_RESUME=1 TCPP_LAYER_TRACE=1 "
    "python colab_train_tinystories.py "
    "--tiny --dataset instruct --steps 3000 --export gguf "
    "> /content/tcpp_train.log 2>&1 & echo $!",
    shell=True, executable="/bin/bash")
print("LAUNCHED", flush=True)

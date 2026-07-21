"""VM-side launcher (A100 lane): PLAIN TinyStories finisher — resumes the
step-5750 checkpoint relayed from the reclaimed run."""
import subprocess
subprocess.Popen(
    "cd /content && nohup python colab_train_tinystories.py --tiny --dataset plain "
    "--steps 6000 --export both > /content/tcpp_train.log 2>&1 & echo $!",
    shell=True, executable="/bin/bash")
print("LAUNCHED", flush=True)

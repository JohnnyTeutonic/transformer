"""VM-side launcher (T4 lane): CHAT model — TinyStories-Instruct, tiny config."""
import subprocess
subprocess.Popen(
    "cd /content && nohup python colab_train_tinystories.py --tiny --dataset instruct "
    "--steps 6000 --export both > /content/tcpp_train.log 2>&1 & echo $!",
    shell=True, executable="/bin/bash")
print("LAUNCHED", flush=True)

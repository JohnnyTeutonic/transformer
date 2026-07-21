"""Package the transformer_cpp SOURCE tree for a Colab build.

Produces transformer_cpp_src.zip (a few MB): src/, include/, third_party/,
scripts/, config/, CMakeLists.txt — no build dirs, no data, no checkpoints.

    python scripts/make_colab_src_zip.py     # writes ../transformer_cpp_src.zip
"""
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "transformer_cpp_src.zip")
INCLUDE_DIRS = ("src", "include", "third_party", "scripts", "config")
INCLUDE_FILES = ("CMakeLists.txt", ".clang-format")
EXCLUDE_EXT = (".obj", ".lib", ".exe", ".pdb", ".zip", ".ckpt", ".gguf",
               ".safetensors", ".log")

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for d in INCLUDE_DIRS:
        for root, _, files in os.walk(os.path.join(ROOT, d)):
            if "__pycache__" in root:
                continue
            for f in files:
                if f.endswith(EXCLUDE_EXT):
                    continue
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, ROOT))
    for f in INCLUDE_FILES:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            z.write(p, f)
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")

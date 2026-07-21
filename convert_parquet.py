import pandas as pd
import os

# Paths
downloads = os.path.expanduser("~/Downloads")
output_dir = "data/wikitext-103-txt"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Convert train files (2 parts -> train-0.txt, train-1.txt)
print("Converting train files...")
for i, fname in enumerate(["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]):
    path = os.path.join(downloads, fname)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        out_path = os.path.join(output_dir, f"train-{i}.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            for text in df['text']:
                f.write(str(text) + '\n')
        print(f"  {fname} -> {out_path} ({len(df)} rows)")
    else:
        print(f"  WARNING: {fname} not found")

# Convert validation file
print("Converting validation file...")
val_path = os.path.join(downloads, "validation-00000-of-00001.parquet")
if os.path.exists(val_path):
    df = pd.read_parquet(val_path)
    out_path = os.path.join(output_dir, "valid.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        for text in df['text']:
            f.write(str(text) + '\n')
    print(f"  validation -> {out_path} ({len(df)} rows)")

# Convert test file
print("Converting test file...")
test_path = os.path.join(downloads, "test-00000-of-00001.parquet")
if os.path.exists(test_path):
    df = pd.read_parquet(test_path)
    out_path = os.path.join(output_dir, "test.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        for text in df['text']:
            f.write(str(text) + '\n')
    print(f"  test -> {out_path} ({len(df)} rows)")

print("\nDone! Files in:", output_dir)
print("Contents:")
for f in os.listdir(output_dir):
    size = os.path.getsize(os.path.join(output_dir, f))
    print(f"  {f}: {size/1024/1024:.1f} MB")

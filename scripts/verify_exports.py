"""Validate transformer_cpp exports without torch.

safetensors: parse the 8-byte header length + JSON header with stdlib, check
every tensor's data_offsets are contiguous, in-bounds, and match dtype*shape;
scan a sample of each tensor for NaN/Inf.
gguf: check magic/version and count tensors from the header.

    python scripts/verify_exports.py build/toy_ts.safetensors build/toy_ts.gguf
"""
import json
import math
import struct
import sys


def verify_safetensors(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        data_start = 8 + hlen
        f.seek(0, 2)
        fsize = f.tell()
        meta = header.pop("__metadata__", {})
        expected = 0
        bad = []
        for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            b0, b1 = info["data_offsets"]
            n = 1
            for d in info["shape"]:
                n *= d
            if info["dtype"] != "F32" or b1 - b0 != 4 * n or b0 != expected:
                bad.append(name)
            expected = b1
            # NaN/Inf scan on a prefix sample
            f.seek(data_start + b0)
            sample = f.read(min(4 * n, 4 * 4096))
            floats = struct.unpack(f"<{len(sample)//4}f", sample)
            if any(math.isnan(x) or math.isinf(x) for x in floats):
                bad.append(name + " (nan/inf)")
        ok = not bad and data_start + expected == fsize
        print(f"[safetensors] {path}: {len(header)} tensors, "
              f"{expected/1e6:.1f} MB data, metadata keys={sorted(meta)[:4]}..., "
              f"{'OK' if ok else 'PROBLEMS: ' + str(bad[:5])}")
        return ok


def verify_gguf(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        (version,) = struct.unpack("<I", f.read(4))
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_meta,) = struct.unpack("<Q", f.read(8))
    ok = magic == b"GGUF"
    print(f"[gguf] {path}: magic={magic!r} v{version}, {n_tensors} tensors, "
          f"{n_meta} metadata entries, {'OK' if ok else 'BAD MAGIC'}")
    return ok


if __name__ == "__main__":
    results = []
    for p in sys.argv[1:]:
        results.append(verify_safetensors(p) if p.endswith(".safetensors") else verify_gguf(p))
    sys.exit(0 if all(results) else 1)

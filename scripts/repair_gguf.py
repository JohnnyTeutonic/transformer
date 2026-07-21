"""Repair transformer_cpp GGUFs for spec-conformant readers (tinyllama.cpp).

Two defects in transformer_cpp's writer (2026-07-13):
  1. The tensor-data section is NOT padded to general.alignment (32): readers
     mmap data at align_up(header_end, 32), so every tensor was read shifted
     (11 bytes for tonight's model) -> garbage weights -> NaN logits.
  2. Special-token metadata claims bos=2/eos=3/unk=1/pad=0, but the word-level
     vocab is <unk>=0, '|'(sep)=1, real words from id 2 - so "the"/"and" were
     treated as BOS/EOS (dropped from decode; EOS stops generation).

Repair: rewrite as header (with patched special ids) + zero padding to the
32-byte boundary + unmodified tensor data. Tensor offsets are relative to the
data-section start, so they remain valid.

    python repair_gguf.py in.gguf out.gguf
"""
import struct
import sys

ALIGN = 32
PATCH = {"tokenizer.ggml.bos_token_id": 1,       # '|' sep; never injected
         "tokenizer.ggml.eos_token_id": 1,       # '|' = document boundary
         "tokenizer.ggml.unknown_token_id": 0,   # '<unk>'
         "tokenizer.ggml.padding_token_id": 0}
SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def main():
    src, dst = sys.argv[1], sys.argv[2]
    data = bytearray(open(src, "rb").read())
    pos = 4  # after magic
    assert bytes(data[:4]) == b"GGUF"
    _, n_tensors, n_kv = struct.unpack_from("<IQQ", data, pos)
    pos += 20

    def read_str():
        nonlocal pos
        n = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        s = bytes(data[pos:pos + n]).decode("utf-8", "replace")
        pos += n
        return s

    def skip_value(t):
        nonlocal pos
        if t in SCALAR_SIZE:
            pos += SCALAR_SIZE[t]
        elif t == 8:
            read_str()
        elif t == 9:
            et = struct.unpack_from("<I", data, pos)[0]
            n = struct.unpack_from("<Q", data, pos + 4)[0]
            pos += 12
            for _ in range(n):
                skip_value(et)
        else:
            raise ValueError(t)

    patched = []
    for _ in range(n_kv):
        key = read_str()
        t = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if key in PATCH and t == 4:                     # u32
            struct.pack_into("<I", data, pos, PATCH[key])
            patched.append(key)
        skip_value(t)

    for _ in range(n_tensors):
        read_str()
        nd = struct.unpack_from("<I", data, pos)[0]
        pos += 4 + 8 * nd + 12                          # ne + type + offset

    header_end = pos
    pad = (ALIGN - header_end % ALIGN) % ALIGN
    with open(dst, "wb") as f:
        f.write(data[:header_end])
        f.write(b"\x00" * pad)
        f.write(data[header_end:])
    print(f"patched {patched}; header_end={header_end}, inserted {pad} pad "
          f"bytes -> data starts at {header_end + pad} (mult of {ALIGN})")


if __name__ == "__main__":
    main()

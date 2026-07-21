#!/usr/bin/env python3
"""Extract the embedded vocabulary of a GGUF (tokenizer.ggml.tokens) to a
text file, one word per line, in id order.

Needed because the trainer's vocabulary construction sorts by frequency with
NO tie-break over an unordered_map (tiktoken_tokenizer.cpp): equal-frequency
words receive platform-dependent ids, so a checkpoint's vocab can only be
reproduced from the GGUF exported by the SAME run. Feed the output to
golden_batch_test --vocab-list for train/deploy parity tests.

Usage: extract_gguf_vocab.py model.gguf vocab.txt
"""

import struct
import sys

SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def main():
    src, dst = sys.argv[1], sys.argv[2]
    data = open(src, "rb").read()
    pos = 4
    assert data[:4] == b"GGUF"
    _, n_tensors, n_kv = struct.unpack_from("<IQQ", data, pos)
    pos += 20

    def read_str():
        nonlocal pos
        n = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        s = data[pos:pos + n].decode("utf-8", "replace")
        pos += n
        return s

    def read_value(t, collect=False):
        nonlocal pos
        if t in SCALAR_SIZE:
            pos += SCALAR_SIZE[t]
            return None
        if t == 8:
            return read_str()
        if t == 9:
            et = struct.unpack_from("<I", data, pos)[0]
            n = struct.unpack_from("<Q", data, pos + 4)[0]
            pos += 12
            items = []
            for _ in range(n):
                v = read_value(et, collect)
                if collect:
                    items.append(v)
            return items if collect else None
        raise ValueError(t)

    tokens = None
    for _ in range(n_kv):
        key = read_str()
        t = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if key == "tokenizer.ggml.tokens":
            tokens = read_value(t, collect=True)
        else:
            read_value(t)

    if tokens is None:
        sys.exit("no tokenizer.ggml.tokens in GGUF")
    with open(dst, "w", encoding="utf-8") as f:
        for w in tokens:
            f.write(w + "\n")
    print(f"wrote {len(tokens)} tokens -> {dst}")


if __name__ == "__main__":
    main()

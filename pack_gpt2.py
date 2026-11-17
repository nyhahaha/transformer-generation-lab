"""Pack the pinned openai-community/gpt2 checkpoint for browser-side inference."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

SOURCE = "https://huggingface.co/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
OPENAI_SOURCE = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/"
WIDTH, DEPTH, HEADS, CONTEXT, VOCAB = 768, 12, 12, 1024, 50257


def read_checkpoint(path):
    with path.open("rb") as checkpoint:
        length = struct.unpack("<Q", checkpoint.read(8))[0]
        header = json.loads(checkpoint.read(length))

    def read(key, shape):
        entry = header.get("transformer." + key) or header[key]
        if tuple(entry["shape"]) != shape or entry["dtype"] != "F32":
            raise ValueError(f"Unexpected {key}: {entry['shape']} {entry['dtype']}")
        return np.memmap(path, mode="r", offset=8 + length + entry["data_offsets"][0],
                         dtype="<f4", shape=shape)

    return read


class Shard:
    def __init__(self, directory, name):
        self.path = directory / f"{name}.bin"
        self.file = self.path.open("wb")
        self.tensors = {}

    def offset(self):
        padding = -self.file.tell() % 4
        self.file.write(b"\0" * padding)
        return self.file.tell()

    def float(self, name, values):
        array = np.asarray(values, dtype="<f4", order="C")
        self.tensors[name] = dict(kind="float", offset=self.offset(), shape=list(array.shape))
        self.file.write(memoryview(array).cast("B"))

    def quantized(self, name, values):
        rows, cols = values.shape
        info = dict(kind="int8", offset=self.offset(), shape=[rows, cols])
        scale = np.empty(rows, dtype="<f4")
        for start in range(0, rows, 256):
            end = min(start + 256, rows)
            chunk = np.asarray(values[start:end], dtype=np.float32)
            factors = np.maximum(np.max(np.abs(chunk), axis=1), 1e-12) / 127
            encoded = np.clip(np.rint(chunk / factors[:, None]), -127, 127).astype(np.int8, order="C")
            self.file.write(memoryview(encoded).cast("B"))
            scale[start:end] = factors
        info["scaleOffset"] = self.offset()
        self.file.write(memoryview(scale).cast("B"))
        self.tensors[name] = info

    def finish(self):
        self.file.close()
        with self.path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        return dict(file=self.path.name, bytes=self.path.stat().st_size, sha256=digest, tensors=self.tensors)


def pack(checkpoint, directory):
    if checkpoint.suffix == ".index":
        from read_openai_checkpoint import read_openai_checkpoint
        data = checkpoint.with_name(checkpoint.stem + ".data-00000-of-00001")
        read = read_openai_checkpoint(checkpoint, data)
        source = OPENAI_SOURCE
    else:
        read = read_checkpoint(checkpoint)
        source = SOURCE
    directory.mkdir(parents=True, exist_ok=True)
    shards = []
    embedding = read("wte.weight", (VOCAB, WIDTH))
    shard = Shard(directory, "embeddings")
    shard.quantized("wte", embedding)
    shard.float("wpe", read("wpe.weight", (CONTEXT, WIDTH)))
    for field in ("weight", "bias"):
        shard.float(f"ln_f.{field}", read(f"ln_f.{field}", (WIDTH,)))
    shards.append(shard.finish())

    for i in range(DEPTH):
        prefix = f"h.{i}"
        shard = Shard(directory, f"block-{i:02d}")
        for norm in ("ln_1", "ln_2"):
            for field in ("weight", "bias"):
                key = f"{prefix}.{norm}.{field}"
                shard.float(key, read(key, (WIDTH,)))
        for part, inputs, outputs in (("attn.c_attn", WIDTH, 3 * WIDTH), ("attn.c_proj", WIDTH, WIDTH),
                                      ("mlp.c_fc", WIDTH, 4 * WIDTH), ("mlp.c_proj", 4 * WIDTH, WIDTH)):
            key = f"{prefix}.{part}"
            shard.quantized(key + ".weight", read(key + ".weight", (inputs, outputs)).T)
            shard.float(key + ".bias", read(key + ".bias", (outputs,)))
        shards.append(shard.finish())
    manifest = dict(model="OpenAI GPT-2 Small", source=source, format="gpt2-row-int8-v1",
                    width=WIDTH, depth=DEPTH, heads=HEADS, vocab=VOCAB, context=CONTEXT,
                    totalBytes=sum(shard["bytes"] for shard in shards), shards=shards)
    (directory / "weights.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Packed {manifest['totalBytes']:,} bytes from {checkpoint.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pack(args.checkpoint, args.output)

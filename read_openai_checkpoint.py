"""Read the uncompressed single-shard TensorFlow bundle used by OpenAI GPT-2."""
import struct

import numpy as np


def varint(data, position):
    value = 0
    for shift in range(0, 70, 7):
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, position
    raise ValueError("Invalid varint")


def block_handle(data, position):
    offset, position = varint(data, position)
    length, position = varint(data, position)
    return offset, length, position


def entries(table, offset, length):
    if table[offset + length] != 0:
        raise ValueError("Expected an uncompressed TensorFlow index block")
    data = table[offset:offset + length]
    restart_count = struct.unpack_from("<I", data, len(data) - 4)[0]
    limit = len(data) - 4 * (restart_count + 1)
    position = 0
    key = b""
    while position < limit:
        shared, position = varint(data, position)
        unshared, position = varint(data, position)
        size, position = varint(data, position)
        key = key[:shared] + data[position:position + unshared]
        position += unshared
        yield key, data[position:position + size]
        position += size


def proto_fields(data):
    position = 0
    while position < len(data):
        tag, position = varint(data, position)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, position = varint(data, position)
        elif wire == 2:
            size, position = varint(data, position)
            value = data[position:position + size]
            position += size
        elif wire == 5:
            value = struct.unpack_from("<I", data, position)[0]
            position += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield field, value


def tensorflow_name(key):
    parts = key.split(".")
    if parts[0] == "h":
        parts[:2] = [parts[0] + parts[1]]
    if parts[-1] in ("weight", "bias"):
        is_norm = parts[-2] in ("ln_1", "ln_2", "ln_f")
        parts[-1] = ("g" if is_norm else "w") if parts[-1] == "weight" else "b"
    if parts[0] in ("wte", "wpe"):
        parts.pop()
    return "model/" + "/".join(parts)


def read_openai_checkpoint(index, data):
    table = index.read_bytes()
    if table[-8:] != bytes.fromhex("57 fb 80 8b 24 75 47 db"):
        raise ValueError("Unexpected TensorFlow checkpoint signature")
    _, _, cursor = block_handle(table, len(table) - 48)
    offset, size, _ = block_handle(table, cursor)
    metadata = {}
    for _, reference in entries(table, offset, size):
        offset, size, _ = block_handle(reference, 0)
        metadata.update((key.decode(), dict(proto_fields(value))) for key, value in entries(table, offset, size) if key)

    def read(key, shape):
        info = metadata[tensorflow_name(key)]
        dims = tuple(dict(proto_fields(dim))[1] for field, dim in proto_fields(info[2]) if field == 2)
        if dims != shape and dims != (1, *shape):
            raise ValueError(f"Unexpected TensorFlow shape for {key}: {dims}")
        expected = int(np.prod(shape)) * 4
        offset = info.get(4, 0)
        if info[1] != 1 or info.get(3, 0) != 0 or info[5] != expected or offset + expected > data.stat().st_size:
            raise ValueError(f"Unexpected TensorFlow dtype, shard or length for {key}")
        return np.memmap(data, mode="r", offset=offset, dtype="<f4", shape=shape)

    return read

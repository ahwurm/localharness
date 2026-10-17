"""Shared GGUF fixture builder for the window-autosense tests (D-32).

An importable helper module, like tests/unit/launch_support.py: its name does not start with `test_`,
so pytest collects nothing from it. A test module imports what it needs by name:

    from tests.unit.gguf_support import Raw, gguf_bytes, long_string_array, write_gguf

- `gguf_bytes(kv, ...)`: the bytes of a GGUF whose KV section holds `kv` (insertion order), no tensors.
- `write_gguf(path, kv, **kw)`: the same, written to `path`.
- `gguf_bytes(..., tensors=[(name, dims, type, offset)], alignment=32, data_bytes=0)`: also the tensor infos, zero
  padding to the alignment and `data_bytes` of zeros; `n_tensors` still overrides the written count.
- `write_recorded(directory, fixture)`: sparse shards reproducing a recorded header subset and tensor table.
- `long_string_array(n)`: n distinct 6-byte strings, the shape of a tokenizer's `tokens` array.
- `Raw(type_id, payload)`: a value written exactly as given, for hostile headers (a declared-huge
  string length, an array declaring millions of elements, a type id nobody defined).

Python type -> GGUF type: bool 7; int 4 (u32), 10 (u64 past 2**32), 5 (i32 when negative); float 6
(f32); str 8; bytes 8 (the raw bytes, for invalid UTF-8); list 9 (element type from the first item).
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import NamedTuple

from localharness.provider.gguf_header import MAX_STRING

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "gguf_tables"
_PAD_OVERHEAD = 8 + len("tests.pad.00") + 4 + 8
_PACK = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Raw(NamedTuple):
    """A KV value written as-is: the type id and the exact payload bytes after it."""

    type_id: int
    payload: bytes


def _string(data: str | bytes) -> bytes:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return struct.pack("<Q", len(raw)) + raw


def _scalar(value: object) -> tuple[int, bytes]:
    if isinstance(value, Raw):
        return value.type_id, value.payload
    if isinstance(value, bool):
        return 7, struct.pack("<?", value)
    if isinstance(value, int):
        if 0 <= value < 2**32:
            return 4, struct.pack("<I", value)
        return (10, struct.pack("<Q", value)) if value >= 0 else (5, struct.pack("<i", value))
    if isinstance(value, float):
        return 6, struct.pack("<f", value)
    if isinstance(value, (str, bytes)):
        return 8, _string(value)
    raise TypeError(f"cannot encode {value!r} as a GGUF value")


def _value(value: object) -> tuple[int, bytes]:
    if not isinstance(value, list):
        return _scalar(value)
    elem = _scalar(value[0])[0] if value else 4
    body = b"".join(_scalar(v)[1] for v in value)
    return 9, struct.pack("<I", elem) + struct.pack("<Q", len(value)) + body


def _infos(tensors: list[tuple]) -> bytes:
    out = []
    for name, dims, type_id, offset in tensors:
        out += [_string(name), struct.pack("<I", len(dims)), *(struct.pack("<Q", d) for d in dims),
                struct.pack("<I", type_id), struct.pack("<Q", offset)]
    return b"".join(out)


def _align(n: int, alignment: int) -> int:
    return -(-n // alignment) * alignment


def _header(kv: dict[str, object], *, version: int = 3, n_tensors: int = 0, magic: bytes = b"GGUF",
            n_kv: int | None = None) -> bytes:
    out = [magic, struct.pack("<I", version), struct.pack("<Q", n_tensors),
           struct.pack("<Q", len(kv) if n_kv is None else n_kv)]
    for key, value in kv.items():
        type_id, payload = _value(value)
        out += [_string(key), struct.pack("<I", type_id), payload]
    return b"".join(out)


def gguf_bytes(kv: dict[str, object], *, version: int = 3, n_tensors: int | None = None,
               magic: bytes = b"GGUF", n_kv: int | None = None, tensors: list[tuple] | None = None,
               alignment: int = 32, data_bytes: int = 0) -> bytes:
    """The bytes of a GGUF holding `kv`; `n_kv` overrides the written pair count (hostile counts)."""
    if alignment != 32:
        kv = {"general.alignment": alignment, **kv}
    count = len(tensors or []) if n_tensors is None else n_tensors
    head = _header(kv, version=version, n_tensors=count, magic=magic, n_kv=n_kv)
    if tensors is None:
        return head
    head += _infos(tensors)
    return head.ljust(_align(len(head), alignment), b"\0") + bytes(data_bytes)


def write_gguf(path: Path, kv: dict[str, object], **kw) -> Path:
    path.write_bytes(gguf_bytes(kv, **kw))
    return path


def long_string_array(n: int, token: str = "tok_00") -> list[str]:
    """n distinct strings, each as many bytes as `token` (6 by default), like a tokenizer's `tokens`
    array: hex indexes, zero padded (so unique for n up to 16**len(token))."""
    return [f"{i:0{len(token)}x}" for i in range(n)]


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _packed(type_id: int, value: object) -> bytes:
    if type_id == 8:
        return _string(value)
    if type_id == 9:
        elem, items = value
        return struct.pack("<IQ", elem, len(items)) + b"".join(_packed(elem, x) for x in items)
    return struct.pack(_PACK[type_id], value)


def _recorded_kv(kv: list) -> dict[str, object]:
    return {key: Raw(type_id, _packed(type_id, value)) for key, type_id, value in kv}


def write_recorded(directory: Path, fixture: dict) -> list[Path]:
    """Each recorded shard as a sparse file: recorded KV, hole-payload padding KVs up to the data start, infos."""
    paths = []
    for shard in fixture["shards"]:
        kv, infos = _recorded_kv(shard["kv"]), _infos(shard["tensors"])
        pad = shard["data_start"] - len(_header(kv)) - len(infos)
        chunks = -(-pad // (MAX_STRING + _PAD_OVERHEAD)) if pad >= _PAD_OVERHEAD else 0
        body = pad - chunks * _PAD_OVERHEAD
        sizes = [body // chunks + (i < body % chunks) for i in range(chunks)] if chunks else []
        path = directory / shard["file"]
        with open(path, "wb") as f:
            f.write(_header(kv, n_tensors=len(shard["tensors"]), n_kv=len(kv) + len(sizes)))
            for i, n in enumerate(sizes):
                f.write(_string(f"tests.pad.{i:02d}") + struct.pack("<I", 8) + struct.pack("<Q", n))
                f.seek(n, os.SEEK_CUR)
            f.write(infos)
        os.truncate(path, shard["file_bytes"])
        paths.append(path)
    return paths


def write_sparse(path: Path, size: int) -> Path:
    path.write_bytes(b"")
    os.truncate(path, size)
    return path

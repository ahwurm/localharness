"""Shared GGUF fixture builder for the window-autosense tests (D-32).

An importable helper module, like tests/unit/launch_support.py: its name does not start with `test_`,
so pytest collects nothing from it. A test module imports what it needs by name:

    from tests.unit.gguf_support import Raw, gguf_bytes, long_string_array, write_gguf

- `gguf_bytes(kv, ...)`: the bytes of a GGUF whose KV section holds `kv` (insertion order), no tensors.
- `write_gguf(path, kv, **kw)`: the same, written to `path`.
- `long_string_array(n)`: n distinct 6-byte strings, the shape of a tokenizer's `tokens` array.
- `Raw(type_id, payload)`: a value written exactly as given, for hostile headers (a declared-huge
  string length, an array declaring millions of elements, a type id nobody defined).

Python type -> GGUF type: bool 7; int 4 (u32), 10 (u64 past 2**32), 5 (i32 when negative); float 6
(f32); str 8; bytes 8 (the raw bytes, for invalid UTF-8); list 9 (element type from the first item).
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import NamedTuple


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


def gguf_bytes(kv: dict[str, object], *, version: int = 3, n_tensors: int = 0,
               magic: bytes = b"GGUF", n_kv: int | None = None) -> bytes:
    """The bytes of a GGUF holding `kv`; `n_kv` overrides the written pair count (hostile counts)."""
    out = [magic, struct.pack("<I", version), struct.pack("<Q", n_tensors),
           struct.pack("<Q", len(kv) if n_kv is None else n_kv)]
    for key, value in kv.items():
        type_id, payload = _value(value)
        out += [_string(key), struct.pack("<I", type_id), payload]
    return b"".join(out)


def write_gguf(path: Path, kv: dict[str, object], **kw) -> Path:
    path.write_bytes(gguf_bytes(kv, **kw))
    return path


def long_string_array(n: int, token: str = "tok_00") -> list[str]:
    """n distinct strings, each as many bytes as `token` (6 by default), like a tokenizer's `tokens`
    array: hex indexes, zero padded (so unique for n up to 16**len(token))."""
    return [f"{i:0{len(token)}x}" for i in range(n)]

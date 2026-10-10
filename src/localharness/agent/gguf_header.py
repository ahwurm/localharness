"""A bounded reader for the KV header of a GGUF file — what the window fit needs, and nothing else.

The context-window fit (block count, KV heads, head size) and the trained-window probe (context
length) need a handful of metadata values from a model file that may be a hostile or merely odd
thing: a FIFO, a device symlink, a header declaring four billion elements. `read_header(path, want)`
reads only the KV section — never the tensor infos, never tensor data — and stops the moment every
wanted name is resolved. Every count and length is checked BEFORE it is acted on (`MAX_KV`,
`MAX_STRING`, `MAX_ARRAY_ELEMENTS`) and the bytes read or skipped are charged against
`HEADER_BYTE_BUDGET`, so a crafted header can only produce a `GgufHeaderError`, never a hang or a
large allocation. The file is opened with `open_regular`, which refuses anything but a regular file.

Pure standard library (struct, os): no llama-cpp-python, no new dependency.
Layout (little-endian): b"GGUF", u32 version (2 or 3), u64 n_tensors, u64 n_kv, then n_kv pairs of
string key (u64 length + bytes), u32 value type, value. Wanted names are suffixes resolved against
`general.architecture`: `general.<name>`, `<arch>.<name>` or `<arch>.attention.<name>`.
"""
from __future__ import annotations

import os
import struct
from collections.abc import Iterable
from typing import Any

from localharness.core.private_files import NotRegularFile, open_regular

__all__ = ["GgufHeaderError", "NotRegularFile", "read_header", "MAGIC", "MAX_KV", "MAX_STRING",
           "MAX_ARRAY_ELEMENTS", "HEADER_BYTE_BUDGET"]

MAGIC = b"GGUF"
MAX_KV = 4096
MAX_STRING = 1 << 20
MAX_ARRAY_ELEMENTS = 1_048_576
HEADER_BYTE_BUDGET = 16 << 20
_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q",
            12: "<d"}
_STRING, _ARRAY = 8, 9
_ARCH_KEY = "general.architecture"


class GgufHeaderError(ValueError):
    """The file is not a GGUF this reader will read (bad magic, unsupported version, over a bound)."""


class _Cursor:
    """The file plus the running byte count: every read or skip is charged before it happens."""

    def __init__(self, f) -> None:
        self.f, self.used = f, 0

    def _charge(self, n: int) -> None:
        if self.used + n > HEADER_BYTE_BUDGET:
            raise GgufHeaderError("header exceeds byte budget")
        self.used += n

    def read(self, n: int) -> bytes:
        self._charge(n)
        data = self.f.read(n)
        if len(data) != n:
            raise GgufHeaderError("truncated")
        return data

    def skip(self, n: int) -> None:
        self._charge(n)
        self.f.seek(n, os.SEEK_CUR)

    def unpack(self, fmt: str) -> Any:
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self, keep: bool = True) -> str | None:
        n = self.unpack("<Q")
        if n > MAX_STRING:
            raise GgufHeaderError("string too long")
        if not keep:
            self.skip(n)
            return None
        return self.read(n).decode("utf-8", errors="replace")


def _value(c: _Cursor, type_id: int, keep: bool) -> Any:
    """Read one value (or step over it when `keep` is False and return None)."""
    if type_id in _FORMATS:
        if keep:
            return c.unpack(_FORMATS[type_id])
        return c.skip(_SIZES[type_id])
    if type_id == _STRING:
        return c.string(keep)
    if type_id != _ARRAY:
        raise GgufHeaderError("unknown value type")
    elem, count = c.unpack("<I"), c.unpack("<Q")
    if count > MAX_ARRAY_ELEMENTS:
        raise GgufHeaderError("array too large")
    if elem in _SIZES:
        if not keep:
            return c.skip(count * _SIZES[elem])
        return [v for (v,) in struct.iter_unpack(_FORMATS[elem], c.read(count * _SIZES[elem]))]
    if elem == _STRING:
        items = [c.string(keep) for _ in range(count)]
        return items if keep else None
    raise GgufHeaderError("unknown value type")


def _forms(arch: str, name: str) -> tuple[str, str, str]:
    return f"general.{name}", f"{arch}.{name}", f"{arch}.attention.{name}"


def _wanted_key(key: str, wanted: set[str], arch: str | None) -> bool:
    suffix = key.rsplit(".", 1)[-1]
    if suffix not in wanted:
        return False
    return arch is None or key in _forms(arch, suffix)  # until the architecture is known, buffer


def read_header(path: str | os.PathLike[str], want: Iterable[str]) -> dict[str, Any]:
    """The wanted metadata values: {name: value}, whatever subset the file holds.

    Raises NotRegularFile (an OSError) for a FIFO, device or directory, other OSErrors as the open
    raises them, and GgufHeaderError for a malformed or over-bound header."""
    wanted = set(want)
    with os.fdopen(open_regular(path), "rb") as f:
        c = _Cursor(f)
        if c.read(len(MAGIC)) != MAGIC:
            raise GgufHeaderError("bad magic")
        version = c.unpack("<I")
        if version not in (2, 3):
            raise GgufHeaderError(f"unsupported GGUF version {version}")
        c.unpack("<Q")  # n_tensors: the tensor infos are never read
        n_kv = c.unpack("<Q")
        if n_kv > MAX_KV:
            raise GgufHeaderError("too many KV pairs")
        if not wanted:
            return {}
        found: dict[str, Any] = {}
        seen: dict[str, Any] = {}  # candidate keys, buffered until general.architecture is known
        arch: str | None = None
        for _ in range(n_kv):
            key = c.string() or ""
            keep = key == _ARCH_KEY or _wanted_key(key, wanted, arch)
            value = _value(c, c.unpack("<I"), keep)
            if key == _ARCH_KEY and isinstance(value, str):
                arch = value
            if keep:
                seen[key] = value
            if arch is not None:
                for name in wanted - found.keys():
                    for candidate in _forms(arch, name):
                        if candidate in seen:
                            found[name] = seen[candidate]
                            break
                if found.keys() == wanted:
                    return found
        return found

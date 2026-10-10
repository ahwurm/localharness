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
string key (u64 length + bytes), u32 value type, value. A dotless wanted name is a suffix resolved
against `general.architecture`: `general.<name>`, `<arch>.<name>` or `<arch>.attention.<name>`. A
wanted name that contains a `.` is an exact key (`split.no`, `clip.vision.projection_dim`), matched in
any position whatever the architecture, and returned under its full name.
"""
from __future__ import annotations

import os
import re
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localharness.core.private_files import NotRegularFile, open_regular

__all__ = ["GgufHeaderError", "NotRegularFile", "read_header", "MAGIC", "MAX_KV", "MAX_STRING",
           "MAX_ARRAY_ELEMENTS", "HEADER_BYTE_BUDGET", "DISCOVERY_MAX_DEPTH", "DISCOVERY_MAX_FILES",
           "DISCOVERY_MAX_ENTRIES", "DISCOVERY_KEYS", "DiscoveredModel", "Discovery", "walk_ggufs",
           "discover", "preset_name", "unique_preset_name", "rank_models", "served_model"]

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
        exact = {name for name in wanted if "." in name}  # D-47: a dotted name is an exact key
        suffixes = wanted - exact
        found: dict[str, Any] = {}
        seen: dict[str, Any] = {}  # candidate keys, buffered until general.architecture is known
        arch: str | None = None
        for _ in range(n_kv):
            key = c.string() or ""
            is_exact = key in exact
            keep = key == _ARCH_KEY or is_exact or _wanted_key(key, suffixes, arch)
            value = _value(c, c.unpack("<I"), keep)
            if key == _ARCH_KEY and isinstance(value, str):
                arch = value
            if is_exact:
                found[key] = value
            elif keep:
                seen[key] = value
            if arch is not None:
                for name in suffixes - found.keys():
                    for candidate in _forms(arch, name):
                        if candidate in seen:
                            found[name] = seen[candidate]
                            break
            if found.keys() == wanted:
                return found
        return found


# ---- discovery (D-01..D-05, D-07, D-47, D-48): what `init` finds under the model roots --------------
# Pure functions beside the one reader: a bounded walk, a typing pass that reads every candidate once,
# pairing of projectors with models, preset naming and ranking. Nothing here writes or launches.

DISCOVERY_MAX_DEPTH = 6        # directory levels below a root that are listed
DISCOVERY_MAX_FILES = 256      # `.gguf` candidates returned
DISCOVERY_MAX_ENTRIES = 20000  # directory entries visited in all
# The dotted names are exact keys (D-47); the rest resolve against the architecture. `discover` adds
# `autosense.FIT_KEYS` (imported lazily, to keep this module import-light).
DISCOVERY_KEYS = frozenset({"type", "architecture", "embedding_length", "context_length", "split.no",
                            "split.count", "clip.vision.projection_dim"})

_SHARD_STEM = re.compile(r"-(\d{5})-of-(\d{5})$")
_PRESET_BAD = re.compile(r"[^A-Za-z0-9._+-]")
_PROJECTOR_LEAD = re.compile(r"^mmproj[-_.]", re.IGNORECASE)


@dataclass(frozen=True)
class DiscoveredModel:
    """One preset-worthy GGUF: the path as written (a split model's is its first shard), its resolved
    path, the D-05 name (not yet unique), the typed header values discovery used, the file's mtime and
    the projector paired with it, if any."""

    path: Path
    real: str
    name: str
    arch: str | None
    embedding: int | None
    context_length: int | None
    mtime: float
    mmproj: Path | None


@dataclass(frozen=True)
class Discovery:
    """`models` in path order; `projectors` are the projectors no model claimed."""

    models: tuple[DiscoveredModel, ...]
    projectors: tuple[Path, ...]


def _stem(path: str | os.PathLike[str]) -> str:
    """The basename without `.gguf` (any case) and without a `-NNNNN-of-NNNNN` split suffix."""
    name = os.path.basename(os.fspath(path))
    if name.lower().endswith(".gguf"):
        name = name[:-len(".gguf")]
    return _SHARD_STEM.sub("", name)


def preset_name(path: str | os.PathLike[str]) -> str:
    """The D-05 preset name: the file stem, every character outside [A-Za-z0-9._+-] made `-`."""
    return _PRESET_BAD.sub("-", _stem(path)) or "model"


def unique_preset_name(name: str, taken: Iterable[str]) -> str:
    """`name`, or the first of `name-2`, `name-3`, … that is not in `taken`."""
    used = set(taken)
    candidate, n = name, 1
    while candidate in used:
        n += 1
        candidate = f"{name}-{n}"
    return candidate


def walk_ggufs(roots: Iterable[str | os.PathLike[str]], *, max_depth: int = DISCOVERY_MAX_DEPTH,
               max_files: int = DISCOVERY_MAX_FILES,
               max_entries: int = DISCOVERY_MAX_ENTRIES) -> list[Path]:
    """The `.gguf` candidates under `roots`, in walk order (entries sorted by name)."""
    found: list[Path] = []
    seen: set[str] = set()
    entries = 0
    for entry_path in _scan(roots, max_depth):
        entries += 1
        if entries > max_entries:
            break
        if entry_path is not None and entry_path not in seen:
            seen.add(entry_path)
            found.append(Path(entry_path))
            if len(found) >= max_files:
                break
    return found


def _scan(roots: Iterable[str | os.PathLike[str]], max_depth: int):
    """Yield one item per directory entry visited: the candidate path, or None."""
    real_roots: list[str] = []
    for root in roots:
        real = os.path.realpath(os.path.expanduser(os.fspath(root)))
        if real not in real_roots and os.path.isdir(real):
            real_roots.append(real)
    for root in real_roots:
        yield from _scan_dir(root, 0, max_depth, real_roots)


def _scan_dir(directory: str, depth: int, max_depth: int, real_roots: list[str]):
    try:
        with os.scandir(directory) as it:
            entries = sorted(it, key=lambda e: e.name)
    except OSError:
        return
    for entry in entries:
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
            candidate = entry.name.lower().endswith(".gguf") and (
                entry.is_file(follow_symlinks=False)
                or (entry.is_symlink() and _inside(os.path.realpath(entry.path), real_roots)))
        except OSError:
            continue
        yield entry.path if candidate else None
        if is_dir and depth < max_depth:
            yield from _scan_dir(entry.path, depth + 1, max_depth, real_roots)


def _inside(real: str, real_roots: Sequence[str]) -> bool:
    for root in real_roots:
        try:
            if os.path.commonpath([real, root]) == root:
                return True
        except ValueError:
            continue
    return False


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def discover(paths: Iterable[str | os.PathLike[str]],
             read: Callable[..., Mapping[str, Any]] = read_header) -> Discovery:
    """Type every candidate with one read each, admit the models and pair them with projectors.

    A projector is `general.type == "mmproj"` or architecture `clip`; a model has `general.type`
    absent or `model`; any other type (`adapter`, `imatrix`, …) is skipped, as is a header the reader
    refuses (GgufHeaderError, NotRegularFile, any OSError)."""
    from localharness.provider.autosense import FIT_KEYS

    keys = DISCOVERY_KEYS | FIT_KEYS
    models: list[dict[str, Any]] = []
    projectors: list[dict[str, Any]] = []
    for raw in sorted({os.fspath(p) for p in paths}):
        try:
            header = read(raw, keys)
            mtime = os.stat(raw).st_mtime
        except (GgufHeaderError, OSError):
            continue
        kind, arch = header.get("type"), header.get("architecture")
        item = {"path": Path(raw), "arch": arch if isinstance(arch, str) else None, "mtime": mtime,
                "dim": _int(header.get("clip.vision.projection_dim")),
                "embedding": _int(header.get("embedding_length")),
                "context_length": _int(header.get("context_length"))}
        if kind == "mmproj" or arch == "clip":
            projectors.append(item)
        elif kind is None or kind == "model":
            models.append(item)
    paired = _pair(models, projectors)
    found = tuple(
        DiscoveredModel(path=m["path"], real=os.path.realpath(m["path"]), name=preset_name(m["path"]),
                        arch=m["arch"], embedding=m["embedding"], context_length=m["context_length"],
                        mtime=m["mtime"], mmproj=paired.get(m["path"]))
        for m in models)
    claimed = set(paired.values())
    return Discovery(found, tuple(p["path"] for p in projectors if p["path"] not in claimed))


def _pair(models: list[dict[str, Any]], projectors: list[dict[str, Any]]) -> dict[Path, Path]:
    """model path -> projector path. Dimension match first; Task 2 adds the prefix tie-break."""
    paired: dict[Path, Path] = {}
    for proj in projectors:
        same = [m for m in models if m["path"].parent == proj["path"].parent
                and proj["dim"] is not None and m["embedding"] == proj["dim"]
                and m["path"] not in paired]
        if len(same) == 1 and sum(1 for p in projectors if p["dim"] == proj["dim"]
                                  and p["path"].parent == proj["path"].parent) == 1:
            paired[same[0]["path"]] = proj["path"]
    return paired


def rank_models(models: Iterable[DiscoveredModel], *, unit_model: str | None,
                ledger_paths: Iterable[str], fits: Mapping[str, int | None]) -> list[DiscoveredModel]:
    """D-07, pure: the unit's model, then a model whose path the ledger recorded, then one that fits
    (above zero), then the newest, then the path string."""
    recorded = set(ledger_paths)

    def key(m: DiscoveredModel) -> tuple:
        names = (m.real, str(m.path))
        return (unit_model is None or unit_model not in names,
                not any(n in recorded for n in names),
                (fits.get(str(m.path)) or 0) <= 0,
                -m.mtime, str(m.path))

    return sorted(models, key=key)

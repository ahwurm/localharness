"""A bounded reader for the KV header of a GGUF file — what the window fit needs, and nothing else.

The context-window fit (block count, KV heads, head size) and the trained-window probe (context
length) need a handful of metadata values from a model file that may be a hostile or merely odd
thing: a FIFO, a device symlink, a header declaring four billion elements. `read_header(path, want)`
reads only the KV section and stops the moment every
wanted name is resolved. Every count and length is checked BEFORE it is acted on (`MAX_KV`,
`MAX_STRING`, `MAX_ARRAY_ELEMENTS`) and the bytes read or skipped are charged against
`HEADER_BYTE_BUDGET`, so a crafted header can only produce a `GgufHeaderError`, never a hang or a
large allocation. `read_tensor_table` also reads the tensor infos (never tensor data) under `MAX_TENSORS`,
`MAX_TENSOR_NAME` and `MAX_DIMS`. The file is opened with `open_regular`, which refuses anything but a regular file.

Pure standard library (struct, os): no llama-cpp-python, no new dependency.
Layout (little-endian): b"GGUF", u32 version (2 or 3), u64 n_tensors, u64 n_kv, then n_kv pairs of
string key (u64 length + bytes), u32 value type, value. A dotless wanted name is a suffix resolved
against `general.architecture`: `general.<name>`, `<arch>.<name>` or `<arch>.attention.<name>`. A
wanted name that contains a `.` is an exact key (`split.no`, `clip.vision.projection_dim`), matched in
any position whatever the architecture, and returned under its full name; once the architecture is known
it also matches `<arch>.attention.<name>` (`indexer.key_length`).
"""
from __future__ import annotations

import os
import re
import stat
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

from localharness.core.private_files import NotRegularFile, open_regular

__all__ = ["GgufHeaderError", "NotRegularFile", "read_header", "read_header_from", "FileKind", "model_of", "MAGIC", "MAX_KV", "MAX_STRING",
           "MAX_ARRAY_ELEMENTS", "HEADER_BYTE_BUDGET", "DISCOVERY_MAX_DEPTH", "DISCOVERY_MAX_FILES",
           "DISCOVERY_MAX_ENTRIES", "DISCOVERY_KEYS", "DiscoveredModel", "Discovery", "walk_ggufs",
           "discover", "preset_name", "unique_preset_name", "rank_models", "served_model", "TensorEntry",
           "TensorTable", "read_tensor_table", "shard_paths", "MAX_TENSORS", "MAX_TENSOR_NAME", "MAX_DIMS",
           "classify", "kv_layer_ids", "kv_layers", "indexer_layers", "cross_checked"]

MAGIC = b"GGUF"
MAX_KV = 4096
MAX_STRING = 1 << 20
MAX_ARRAY_ELEMENTS = 1_048_576
HEADER_BYTE_BUDGET = 16 << 20
MAX_TENSORS = 65_536
MAX_TENSOR_NAME = 256
MAX_DIMS = 4
DEFAULT_ALIGNMENT = 32
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

    def string(self, keep: bool = True, limit: int = MAX_STRING) -> str | None:
        n = self.unpack("<Q")
        if n > limit:
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
    with os.fdopen(open_regular(path), "rb") as f:
        return read_header_from(f, want)


def _preamble(c: _Cursor) -> tuple[int, int]:
    """Check magic and version; return (n_tensors, n_kv) with the KV count bounded."""
    if c.read(len(MAGIC)) != MAGIC:
        raise GgufHeaderError("bad magic")
    version = c.unpack("<I")
    if version not in (2, 3):
        raise GgufHeaderError(f"unsupported GGUF version {version}")
    n_tensors, n_kv = c.unpack("<Q"), c.unpack("<Q")
    if n_kv > MAX_KV:
        raise GgufHeaderError("too many KV pairs")
    return n_tensors, n_kv


def read_header_from(f, want: Iterable[str]) -> dict[str, Any]:
    """`read_header` over an already-open binary file positioned at its start."""
    wanted = set(want)
    c = _Cursor(f)
    _, n_kv = _preamble(c)
    if not wanted:
        return {}
    exact = {name for name in wanted if "." in name}  # a dotted name is an exact key
    suffixes = wanted - exact
    found: dict[str, Any] = {}
    seen: dict[str, Any] = {}  # candidate keys, buffered until general.architecture is known
    arch: str | None = None
    for _ in range(n_kv):
        key = c.string() or ""
        is_exact = key in exact
        keep = (key == _ARCH_KEY or is_exact or _wanted_key(key, suffixes, arch)
                or any(key.endswith(f".attention.{name}") for name in exact))
        value = _value(c, c.unpack("<I"), keep)
        if key == _ARCH_KEY and isinstance(value, str):
            arch = value
        if is_exact:
            found[key] = value
        elif keep:
            seen[key] = value
        if arch is not None:
            for name in wanted - found.keys():
                for candidate in _forms(arch, name) if name in suffixes else (f"{arch}.attention.{name}",):
                    if candidate in seen:
                        found[name] = seen[candidate]
                        break
        if found.keys() == wanted:
            return found
    return found


class TensorEntry(NamedTuple):
    name: str
    bytes: int


@dataclass(frozen=True)
class TensorTable:
    entries: tuple[TensorEntry, ...]
    data_bytes: int


_SHARD_NAME = re.compile(r"^(?P<prefix>.*)-\d{5}-of-(?P<n>\d{5})\.gguf$")


def shard_paths(path: str | os.PathLike[str]) -> list[Path]:
    """The model's files in shard order (`<prefix>-00001-of-0000N.gguf` ..), else the path alone."""
    p = Path(path)
    shard = _SHARD_NAME.match(p.name)
    if shard is None or int(shard["n"]) < 1:
        return [p]
    return [p.with_name(f"{shard['prefix']}-{i:05d}-of-{shard['n']}.gguf") for i in range(1, int(shard["n"]) + 1)]


def read_tensor_table(path: str | os.PathLike[str]) -> TensorTable:
    """The tensors of the model and its shards with the bytes each holds; GgufHeaderError when a shard is not priceable."""
    tables = [_shard_table(p) for p in shard_paths(path)]
    return TensorTable(tuple(e for entries, _ in tables for e in entries), sum(d for _, d in tables))


def _shard_table(path: Path) -> tuple[list[TensorEntry], int]:
    with os.fdopen(open_regular(path), "rb") as f:
        size = os.fstat(f.fileno()).st_size
        c = _Cursor(f)
        n_tensors, n_kv = _preamble(c)
        if n_tensors > MAX_TENSORS:
            raise GgufHeaderError("too many tensors")
        alignment = DEFAULT_ALIGNMENT
        for _ in range(n_kv):
            key = c.string() or ""
            value = _value(c, c.unpack("<I"), key == "general.alignment")
            if key == "general.alignment":
                if not isinstance(value, int) or value < 1:
                    raise GgufHeaderError("bad alignment")
                alignment = value
        infos: list[tuple[str, int]] = []
        for _ in range(n_tensors):
            name = c.string(limit=MAX_TENSOR_NAME) or ""
            dims = c.unpack("<I")
            if dims > MAX_DIMS:
                raise GgufHeaderError("too many tensor dimensions")
            c.read(8 * dims)
            c.unpack("<I")
            offset = c.unpack("<Q")
            if offset % alignment:
                raise GgufHeaderError("tensor offset not aligned")
            if infos and offset <= infos[-1][1]:
                raise GgufHeaderError("tensor offsets not increasing")
            infos.append((name, offset))
        data_bytes = size - -(-f.tell() // alignment) * alignment
    if data_bytes < 0 or any(offset >= data_bytes for _, offset in infos):
        raise GgufHeaderError("tensor offset past the end of the data")
    ends = [offset for _, offset in infos[1:]] + [data_bytes]
    return [TensorEntry(name, end - offset) for (name, offset), end in zip(infos, ends)], data_bytes


_INPUT_TENSORS = frozenset({"token_embd", "per_layer_token_embd", "pos_embd", "token_types"})
_BLOCK_TENSOR = re.compile(r"^blk\.(\d+)\.(.*)$")


def classify(name: str) -> str:
    """Where llama.cpp keeps a tensor: "input" (read row by row through mmap), "experts" (`_exps`) or "gpu"."""
    if name.split(".", 1)[0] in _INPUT_TENSORS:
        return "input"
    return "experts" if "_exps" in name else "gpu"


def _blocks(entries: Iterable[TensorEntry]) -> dict[int, list[str]]:
    blocks: dict[int, list[str]] = {}
    for e in entries:
        m = _BLOCK_TENSOR.match(e.name)
        if m:
            blocks.setdefault(int(m[1]), []).append(m[2])
    return blocks


def _holders(entries: Iterable[TensorEntry]) -> list[int]:
    return sorted(n for n, names in _blocks(entries).items()
                  if any(t.startswith("attn_k.") for t in names)
                  or (any(t.startswith("attn_qkv.") for t in names) and not any(t.startswith("ssm_") for t in names)))


def kv_layer_ids(entries: Iterable[TensorEntry], header: Mapping[str, Any]) -> tuple[int, ...]:
    """The trunk blocks holding K, ascending; the trunk ends at `block_count - nextn_predict_layers`."""
    trunk = header.get("block_count", 0) - header.get("nextn_predict_layers", 0)
    return tuple(n for n in _holders(entries) if n < trunk)


def kv_layers(entries: Iterable[TensorEntry], header: Mapping[str, Any]) -> tuple[int, int]:
    """(main, draft): the blocks holding K, split at `block_count - nextn_predict_layers`."""
    entries = tuple(entries)
    main = len(kv_layer_ids(entries, header))
    return main, len(_holders(entries)) - main


def indexer_layers(entries: Iterable[TensorEntry]) -> int:
    """The blocks holding a `blk.N.indexer.k_proj` tensor."""
    return sum(any(t.startswith("indexer.k_proj.") for t in names) for names in _blocks(entries).values())


def cross_checked(sums: Mapping[str, int], main: int, header: Mapping[str, Any], data_bytes: int) -> bool:
    """Whether the table agrees with the header on attention layers, experts and the input tables' share."""
    blocks = header.get("block_count")
    if blocks is None or not 1 <= main <= blocks:
        return False
    interval = header.get("full_attention_interval")
    if interval is not None and main != (blocks - header.get("nextn_predict_layers", 0)) // interval:
        return False
    return (sums.get("experts", 0) == 0 or "expert_count" in header) and 2 * sums.get("input", 0) <= data_bytes


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


class FileKind(str, Enum):
    MODEL = "model"
    PROJECTOR = "projector"
    SHARD = "shard"
    OTHER = "other"
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class DiscoveredModel:
    """One preset-worthy GGUF: the path as written (a split model's is its first shard), its resolved
    path, the preset name (not yet unique), the typed header values discovery used, the file's mtime,
    the projector paired with it, if any, and its MTP layer count (None without valid MTP layers)."""

    path: Path
    real: str
    name: str
    arch: str | None
    embedding: int | None
    context_length: int | None
    mtime: float
    mmproj: Path | None
    mtp: int | None = None


@dataclass(frozen=True)
class Discovery:
    """`models` in path order; `projectors` are the projectors no model claimed; `kinds` holds one
    `FileKind` per input path string."""

    models: tuple[DiscoveredModel, ...]
    projectors: tuple[Path, ...]
    kinds: Mapping[str, FileKind] = field(default_factory=dict)


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
    absent or `model`; any other type (`adapter`, `imatrix`, …) is `other`, a later shard or an
    incomplete split is `shard`, and a header the reader refuses (GgufHeaderError, NotRegularFile,
    any OSError) is `unreadable`; only models and projectors are returned."""
    from localharness.provider.autosense import FIT_KEYS

    keys = DISCOVERY_KEYS | FIT_KEYS
    models: list[dict[str, Any]] = []
    projectors: list[dict[str, Any]] = []
    kinds: dict[str, FileKind] = {}
    for raw in sorted({os.fspath(p) for p in paths}):
        try:
            header = read(raw, keys)
            mtime = os.stat(raw).st_mtime
        except (GgufHeaderError, OSError):
            kinds[raw] = FileKind.UNREADABLE
            continue
        kinds[raw] = kind = _kind(raw, header)
        if kind is FileKind.PROJECTOR:
            projectors.append(_item(raw, header, mtime))
        elif kind is FileKind.MODEL:
            models.append(_item(raw, header, mtime))
    paired = _pair(models, projectors)
    found = tuple(_model(m, paired.get(m["path"])) for m in models)
    claimed = set(paired.values())
    return Discovery(found, tuple(p["path"] for p in projectors if p["path"] not in claimed), kinds)


def model_of(path: str | os.PathLike[str], header: Mapping[str, Any], *,
             mtime: float) -> DiscoveredModel | None:
    """The model a header read with `discover`'s keys describes, unpaired, or None when it is not one."""
    raw = os.fspath(path)
    return _model(_item(raw, header, mtime), None) if _kind(raw, header) is FileKind.MODEL else None


def _kind(raw: str, header: Mapping[str, Any]) -> FileKind:
    kind, arch = header.get("type"), header.get("architecture")
    if kind == "mmproj" or arch == "clip":
        return FileKind.PROJECTOR
    if kind is None or kind == "model":
        admitted = _admitted(raw, _int(header.get("split.no")), _int(header.get("split.count")))
        return FileKind.MODEL if admitted else FileKind.SHARD
    return FileKind.OTHER


def _item(raw: str, header: Mapping[str, Any], mtime: float) -> dict[str, Any]:
    from localharness.provider.tuning import mtp_layers

    arch = header.get("architecture")
    return {"path": Path(raw), "arch": arch if isinstance(arch, str) else None, "mtime": mtime,
            "dim": _int(header.get("clip.vision.projection_dim")),
            "embedding": _int(header.get("embedding_length")),
            "context_length": _int(header.get("context_length")), "mtp": mtp_layers(header)}


def _model(item: dict[str, Any], mmproj: Path | None) -> DiscoveredModel:
    return DiscoveredModel(path=item["path"], real=os.path.realpath(item["path"]),
                           name=preset_name(item["path"]), arch=item["arch"], embedding=item["embedding"],
                           context_length=item["context_length"], mtime=item["mtime"], mmproj=mmproj,
                           mtp=item["mtp"])


def _admitted(path: str, split_no: int | None, split_count: int | None) -> bool:
    """D-03: a model without split keys is admitted as itself; `split.no` above zero is a later shard,
    never a preset; `split.no == 0` only when every `-0000k-of-N` sibling is a regular file (stat only,
    never opened). A count of one is a whole model."""
    if split_no is None:
        return True
    if split_no != 0:
        return False
    if split_count is None or split_count < 1:
        return False
    if split_count == 1:
        return True
    name = os.path.basename(path)
    if not name.lower().endswith(".gguf"):
        return False
    stem = _SHARD_STEM.search(name[:-len(".gguf")])
    if stem is None or int(stem.group(2)) != split_count or int(stem.group(1)) != 1:
        return False
    prefix, ext = name[:stem.start()], name[-len(".gguf"):]
    directory = os.path.dirname(path)
    for k in range(1, split_count + 1):
        sibling = os.path.join(directory, f"{prefix}-{k:05d}-of-{split_count:05d}{ext}")
        try:
            if not stat.S_ISREG(os.stat(sibling).st_mode):
                return False
        except OSError:
            return False
    return True


def _lcp(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _pair(models: list[dict[str, Any]], projectors: list[dict[str, Any]]) -> dict[Path, Path]:
    """model path -> projector path, per directory (D-04, D-48): the dimension must match
    (`clip.vision.projection_dim == embedding_length`); a projector claims the one model its
    mmproj-stripped stem shares the strictly longest prefix with (LCP > 0), a model claimed twice goes
    to the strictly longer prefix and a tie pairs neither; then, repeated until stable, the last unpaired
    projector and model of one dimension pair. Sorted throughout, so the input order never matters."""
    by_dir: dict[Path, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for m in sorted(models, key=lambda m: str(m["path"])):
        by_dir.setdefault(m["path"].parent, ([], []))[0].append(m)
    for p in sorted(projectors, key=lambda p: str(p["path"])):
        if p["dim"] is not None and p["path"].parent in by_dir:
            by_dir[p["path"].parent][1].append(p)
    paired: dict[Path, Path] = {}
    for dir_models, dir_projectors in by_dir.values():
        paired.update(_pair_directory(dir_models, dir_projectors))
    return paired


def _pair_directory(models: list[dict[str, Any]], projectors: list[dict[str, Any]]) -> dict[Path, Path]:
    def stem_of(p: dict[str, Any]) -> str:
        return _PROJECTOR_LEAD.sub("", _stem(p["path"]), count=1)

    def fits(p: dict[str, Any], m: dict[str, Any]) -> bool:
        return m["embedding"] is not None and m["embedding"] == p["dim"]

    claims: dict[Path, list[tuple[int, Path]]] = {}  # model path -> [(lcp, projector path)]
    for p in projectors:
        scored = [(_lcp(stem_of(p), _stem(m["path"])), m["path"]) for m in models if fits(p, m)]
        top = max((n for n, _ in scored), default=0)
        winners = [path for n, path in scored if n == top]
        if top > 0 and len(winners) == 1:
            claims.setdefault(winners[0], []).append((top, p["path"]))
    result: dict[Path, Path] = {}
    for model_path, claimants in claims.items():
        best = max(n for n, _ in claimants)
        leaders = [path for n, path in claimants if n == best]
        if len(leaders) == 1:
            result[model_path] = leaders[0]
    while True:
        free_models = [m for m in models if m["path"] not in result]
        free_projectors = [p for p in projectors if p["path"] not in set(result.values())]
        added = False
        for dim in sorted({p["dim"] for p in free_projectors}):
            ps = [p for p in free_projectors if p["dim"] == dim]
            ms = [m for m in free_models if m["embedding"] == dim]
            if len(ps) == 1 and len(ms) == 1:
                result[ms[0]["path"]] = ps[0]["path"]
                added = True
        if not added:
            return result


def rank_models(models: Iterable[DiscoveredModel], *, unit_model: str | None,
                ledger_paths: Iterable[str], fits: Mapping[str, int | None],
                gpu_bytes: Mapping[str, int | None] | None = None) -> list[DiscoveredModel]:
    """Pure: the unit's model, a ledger-recorded path, a fit above zero, the larger window, fewer GPU-resident bytes (unknown last), the newest, the path."""
    recorded = set(ledger_paths)
    gpu = gpu_bytes or {}

    def key(m: DiscoveredModel) -> tuple:
        names = (m.real, str(m.path))
        window, size = fits.get(str(m.path)) or 0, gpu.get(str(m.path))
        return (unit_model is None or unit_model not in names,
                not any(n in recorded for n in names),
                window <= 0, -window,
                size is None, size or 0,
                -m.mtime, str(m.path))

    return sorted(models, key=key)


_SHARD_FILE = re.compile(r"^(?P<head>.*)-\d{5}(?P<of>-of-\d{5}\.gguf)$", re.IGNORECASE)


def served_model(paths: Iterable[str | os.PathLike[str]],
                 read: Callable[..., Mapping[str, Any]] = read_header) -> str | None:
    """D-52: the first of `paths` that types as a model, or None. A model is a GGUF (the reader checks
    the magic) whose `general.type` is absent or `model`, whose architecture is not `clip` and whose
    `split.no` is absent or 0. A shard-pattern name is read as its `-00001-of-` sibling and that is the
    path returned; a header the reader refuses skips the candidate. The import (Plan 07) types
    `procfs.ProcRecord.mapped_ggufs` with it; `procfs` itself parses no GGUF."""
    for raw in paths:
        path = os.fspath(raw)
        shard = _SHARD_FILE.match(os.path.basename(path))
        if shard:
            path = os.path.join(os.path.dirname(path), f"{shard['head']}-00001{shard['of']}")
        try:
            header = read(path, {"type", "architecture", "split.no"})
        except (GgufHeaderError, OSError):
            continue
        if (header.get("type") in (None, "model") and header.get("architecture") != "clip"
                and header.get("split.no") in (None, 0)):
            return path
    return None

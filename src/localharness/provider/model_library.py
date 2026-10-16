"""The model library's pure core: what the model roots hold that no record names, and the file that remembers it."""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from localharness.agent import gguf_header
from localharness.agent.gguf_header import (
    DISCOVERY_MAX_FILES,
    DiscoveredModel,
    Discovery,
    FileKind,
    GgufHeaderError,
    discover,
    unique_preset_name,
    walk_ggufs,
)
from localharness.config.models import EndpointRef, ManagedServerConfig
from localharness.config.overlay import OverlayBusy, overlay_lock
from localharness.core.private_files import atomic_write_text, open_regular
from localharness.provider import procfs

__all__ = ["LIBRARY_FILE", "LIBRARY_VERSION", "LIBRARY_CAP", "Entry", "Library", "Shown", "Found",
           "library_path", "scan", "peers_for", "free_ports"]

LIBRARY_FILE = "model-library.json"
LIBRARY_VERSION = 1
LIBRARY_CAP = 1 << 20
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class Entry:
    first_seen: str
    size: int
    verdict: FileKind
    declined: bool = False


def _entry(key: Any, value: Any) -> Entry | None:
    if not (isinstance(key, str) and os.path.isabs(key) and "\0" not in key and isinstance(value, dict)):
        return None
    first_seen, size, declined = value.get("first_seen"), value.get("size"), value.get("declined", False)
    try:
        verdict = FileKind(value.get("verdict"))
        date.fromisoformat(first_seen)
    except (ValueError, TypeError):
        return None
    if not (_DATE.fullmatch(first_seen) and type(size) is int and size >= 0 and type(declined) is bool):
        return None
    return Entry(first_seen, size, verdict, declined)


@dataclass(frozen=True)
class Library:
    """The file as loaded: `entries` by realpath; `missing` or `corrupt` give no entries at all."""

    entries: Mapping[str, Entry] = field(default_factory=dict)
    missing: bool = False
    corrupt: bool = False

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Library:
        try:
            with os.fdopen(open_regular(path), "rb") as fh:
                raw = fh.read(LIBRARY_CAP + 1)
        except FileNotFoundError:
            return cls(missing=True)
        except OSError:
            return cls(corrupt=True)
        try:
            data = json.loads(raw) if len(raw) <= LIBRARY_CAP else None
        except (ValueError, RecursionError):
            data = None
        entries = data.get("entries") if isinstance(data, dict) and data.get("version") == LIBRARY_VERSION \
            and type(data["version"]) is int else None
        if not isinstance(entries, dict) or len(entries) > DISCOVERY_MAX_FILES:
            return cls(corrupt=True)
        return cls({key: e for key, value in entries.items() if (e := _entry(key, value))})

    def unrecorded(self, recorded: Collection[str]) -> Library:
        """The entries no record names: the one test of recorded-ness the scan, the first run and the counts share."""
        return replace(self, entries={k: e for k, e in self.entries.items() if k not in recorded})

    def marked(self, *, declined: Iterable[str] = (), dropped: Iterable[str] = ()) -> Library:
        gone, no = set(dropped), set(declined)
        return replace(self, entries={k: replace(e, declined=True) if k in no else e
                                      for k, e in self.entries.items() if k not in gone})

    def save(self, path: str | os.PathLike[str]) -> bool:
        """Write the entries whose files still exist under the overlay lock; False when the lock is busy."""
        path = Path(path)
        kept = {key: {"first_seen": e.first_seen, "size": e.size, "verdict": e.verdict.value,
                      "declined": e.declined}
                for key, e in sorted(self.entries.items()) if os.path.lexists(key)}
        text = json.dumps({"version": LIBRARY_VERSION, "entries": kept}, indent=1, sort_keys=True)
        try:
            with overlay_lock(path.parent):
                atomic_write_text(path, text)
        except OverlayBusy:
            return False
        return True


@dataclass(frozen=True)
class Shown:
    """One file for the person: `model` is a fresh header read, None for a file that is not shown."""

    path: Path
    real: str
    model: DiscoveredModel | None
    size: int
    first_seen: str
    declined: bool


@dataclass(frozen=True)
class Found:
    new: tuple[Shown, ...]
    seen: tuple[Shown, ...]
    declined: tuple[Shown, ...]
    library: Library = field(default_factory=Library)
    baselined: bool = False
    changed: bool = False


def library_path(global_dir: str | os.PathLike[str]) -> Path:
    return Path(global_dir) / LIBRARY_FILE


def _once(read: Callable[..., Mapping[str, Any]]) -> Callable[..., Mapping[str, Any]]:
    cache: dict[str, Any] = {}

    def reader(path: str, keys: Iterable[str]) -> Mapping[str, Any]:
        if path not in cache:
            try:
                cache[path] = read(path, keys)
            except (GgufHeaderError, OSError) as exc:
                cache[path] = exc
        if isinstance(cache[path], BaseException):
            raise cache[path]
        return cache[path]

    return reader


def _typed(known: Mapping[str, Entry], found: Mapping[str, tuple[Path, int]],
           scope: str) -> Discovery:
    """One typing pass over the files that need it, then the known projectors beside a model it found."""
    need = [path for real, (path, size) in found.items()
            if (old := known.get(real)) is None or old.verdict is FileKind.UNREADABLE or old.size != size
            or (scope == "all" and old.verdict is FileKind.MODEL)]
    if not need:
        return Discovery((), ())
    read = _once(gguf_header.read_header)
    typed = discover(need, read=read)
    homes = {m.path.parent for m in typed.models}
    partners = [path for real, (path, _) in found.items()
                if path.parent in homes and path not in need and known[real].verdict is FileKind.PROJECTOR]
    return discover([*need, *partners], read=read) if partners else typed


def scan(roots: Iterable[str | os.PathLike[str]], recorded: Collection[str],
         library_path: str | os.PathLike[str], *, scope: Literal["new", "all"],
         rank: Callable[[list[DiscoveredModel]], Sequence[DiscoveredModel]],
         first_run: bool = False, today: date | None = None) -> Found:
    """What the roots hold that `recorded` does not name: new is a first sighting in the library, never an mtime."""
    if scope not in ("new", "all"):
        raise ValueError(f"scope is 'new' or 'all', not {scope!r}")
    lib = Library.load(library_path)
    baselined = (lib.missing or lib.corrupt) and not first_run
    stamp = (today or datetime.now(timezone.utc).date()).isoformat()
    found: dict[str, tuple[Path, int]] = {}
    for path in walk_ggufs(roots):
        real = os.path.realpath(path)
        if real in recorded or real in found:
            continue
        try:
            found[real] = (path, os.stat(path).st_size)
        except OSError:
            continue
    entries = dict(lib.unrecorded(recorded).entries)
    typed = _typed(entries, found, scope)
    models = {str(m.path): m for m in typed.models}
    new: list[Shown] = []
    seen: list[Shown] = []
    declined: list[Shown] = []
    for real, (path, size) in found.items():
        old, key = entries.get(real), str(path)
        first = False
        if key in typed.kinds:
            kind = typed.kinds[key]
            if kind is FileKind.MODEL and old is not None and old.verdict is FileKind.MODEL:
                entry = Entry(old.first_seen, size, kind, old.declined)
            elif kind is FileKind.MODEL:
                entry, first = Entry(stamp, size, kind), True
            else:
                entry = Entry(old.first_seen if old else stamp, size, kind)
        else:
            entry = old
        entries[real] = entry
        if entry.verdict is not FileKind.MODEL:
            continue
        is_new = first and not baselined
        shown = Shown(path, real, models.get(key) if scope == "all" or is_new else None, size,
                      entry.first_seen, entry.declined)
        (new if is_new else declined if entry.declined else seen).append(shown)
    ranked = ()
    if new:
        by_path = {str(s.path): s for s in new}
        ranked = tuple(by_path[str(m.path)] for m in rank([s.model for s in new]))
    return Found(ranked, tuple(seen), tuple(declined), Library(entries), baselined,
                 lib.missing or lib.corrupt or entries != lib.entries)


def free_ports(start: int, count: int, *, taken: Collection[int]) -> list[int]:
    """The next `count` ports from `start` upward that no record uses (`taken`) and nothing listens on."""
    ports: list[int] = []
    port = start
    while len(ports) < count and port <= 65535:
        if port not in taken and not procfs.listeners(port):
            ports.append(port)
        port += 1
    return ports


def peers_for(models: Sequence[DiscoveredModel], *, start: int, taken_names: Collection[str],
              taken_ports: Collection[int]) -> list[EndpointRef]:
    """One launchable peer per model: a GPU llama.cpp record on the next free port from `start`, named as the
    preset, carrying only the model, its projector, its served name and its port."""
    ports = free_ports(start, len(models), taken=taken_ports)
    names = set(taken_names)
    peers: list[EndpointRef] = []
    for model, port in zip(models, ports):
        name = unique_preset_name(model.name, names)
        names.add(name)
        peers.append(EndpointRef(
            name=name,
            base_url=f"http://127.0.0.1:{port}/v1",
            provider_type="llamacpp",
            gpu=True,
            lifecycle=ManagedServerConfig(
                runtime="llamacpp", model=str(model.path),
                mmproj=str(model.mmproj) if model.mmproj else None,
                served_name=name, port=port, gpu=True),
        ))
    return peers

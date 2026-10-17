"""The model library's pure core: what the model roots hold that no record names, and the file that remembers it."""
from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from localharness.provider import gguf_header
from localharness.provider.gguf_header import (
    DISCOVERY_MAX_DEPTH,
    DISCOVERY_MAX_ENTRIES,
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
from localharness.config.overlay import overlay_lock
from localharness.core.private_files import NotRegularFile, atomic_write_text, open_regular
from localharness.provider import procfs
from localharness.provider.tuning import MODEL_CONFIG_CAP

__all__ = ["LIBRARY_FILE", "LIBRARY_VERSION", "LIBRARY_CAP", "Entry", "Library", "Shown", "Found", "Checkpoint",
           "library_path", "scan", "peers_for", "free_ports", "checkpoints", "snapshots", "snapshot_of"]

LIBRARY_FILE = "model-library.json"
LIBRARY_VERSION = 1
LIBRARY_CAP = 1 << 20
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?")
_TEXT_GENERATION = ("ForCausalLM", "ForConditionalGeneration")
_EMBEDDING_FILES = ("modules.json", "config_sentence_transformers.json")
_UNLOADED = "its snapshot holds an entry the harness does not load"
_NO_WEIGHTS = "its snapshot holds no safetensors weights of a text-generation model"


def _repo_ok(repo: object) -> bool:
    return isinstance(repo, str) and len(repo) <= 96 and _REPO.fullmatch(repo) is not None


def _recorded_by(recorded: Collection[str], real: str, repo: str | None) -> bool:
    return real in recorded or (repo is not None and repo in recorded)


@dataclass(frozen=True)
class Checkpoint:
    """A cached Hugging Face repo: its chosen snapshot directory (and that directory's realpath) and what it holds."""

    repo_id: str
    snapshot: Path
    real: str
    size: int
    kind: str = "vllm"
    refusal: str | None = None
    ggufs: tuple[Path, ...] = ()


@dataclass(frozen=True)
class Entry:
    first_seen: str
    size: int
    verdict: FileKind
    declined: bool = False
    repo: str | None = None


def _entry(key: Any, value: Any) -> Entry | None:
    if not (isinstance(key, str) and os.path.isabs(key) and "\0" not in key and isinstance(value, dict)):
        return None
    first_seen, size, declined = value.get("first_seen"), value.get("size"), value.get("declined", False)
    repo = value.get("repo")
    try:
        verdict = FileKind(value.get("verdict"))
        date.fromisoformat(first_seen)
    except (ValueError, TypeError):
        return None
    if not (_DATE.fullmatch(first_seen) and type(size) is int and size >= 0 and type(declined) is bool
            and (repo is None or _repo_ok(repo))):
        return None
    return Entry(first_seen, size, verdict, declined, repo)


@dataclass(frozen=True)
class Library:
    """The file as loaded: `entries` by realpath; `missing`, `corrupt` or `unreadable` give no entries at all."""

    entries: Mapping[str, Entry] = field(default_factory=dict)
    missing: bool = False
    corrupt: bool = False
    unreadable: bool = False
    gone: frozenset[str] = frozenset()

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Library:
        try:
            with os.fdopen(open_regular(path), "rb") as fh:
                raw = fh.read(LIBRARY_CAP + 1)
        except FileNotFoundError:
            return cls(missing=True)
        except NotRegularFile:
            return cls(corrupt=True)
        except OSError:
            return cls(unreadable=True)
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
        kept = {k: e for k, e in self.entries.items() if not _recorded_by(recorded, k, e.repo)}
        return replace(self, entries=kept, gone=self.gone | (self.entries.keys() - kept.keys()))

    def marked(self, *, declined: Iterable[str] = (), dropped: Iterable[str] = ()) -> Library:
        gone, no = set(dropped), set(declined)
        return replace(self, entries={k: replace(e, declined=True) if k in no else e
                                      for k, e in self.entries.items() if k not in gone}, gone=self.gone | gone)

    def save(self, path: str | os.PathLike[str], *, merge: bool = False) -> bool:
        """Write the entries whose files exist under the overlay lock; `merge` lays them over the file as reloaded there."""
        if self.unreadable:
            return False
        path = Path(path)
        try:
            with overlay_lock(path.parent):
                entries = self.entries
                if merge:
                    disk = Library.load(path)
                    if disk.unreadable:
                        return False
                    entries = {**{k: e for k, e in disk.entries.items() if k not in self.gone},
                               **{k: replace(e, declined=e.declined or (k in disk.entries and disk.entries[k].declined))
                                  for k, e in self.entries.items()}}
                kept = {key: {"first_seen": e.first_seen, "size": e.size, "verdict": e.verdict.value,
                              "declined": e.declined, **({"repo": e.repo} if e.repo else {})}
                        for key, e in sorted(entries.items()) if os.path.lexists(key)}
                atomic_write_text(path, json.dumps({"version": LIBRARY_VERSION, "entries": kept}, indent=1,
                                                   sort_keys=True))
        except OSError:
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
    checkpoint: Checkpoint | None = None


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
         first_run: bool = False, today: date | None = None, checkpoints: Sequence[Checkpoint] = (),
         extra: Sequence[Path] = ()) -> Found:
    """What the roots hold that `recorded` does not name: new is a first sighting in the library, never an mtime."""
    if scope not in ("new", "all"):
        raise ValueError(f"scope is 'new' or 'all', not {scope!r}")
    lib = Library.load(library_path)
    if lib.unreadable:
        return Found((), (), (), lib)
    baselined = (lib.missing or lib.corrupt) and not first_run
    stamp = (today or datetime.now(timezone.utc).date()).isoformat()
    found: dict[str, tuple[Path, int]] = {}
    for path in (*walk_ggufs(roots), *extra):
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
    new_cp: list[Shown] = []
    seen_cp: list[Shown] = []
    declined_cp: list[Shown] = []
    for cp in sorted(checkpoints, key=lambda c: c.repo_id):
        if _recorded_by(recorded, cp.real, cp.repo_id):
            continue
        old = entries.get(cp.real)
        entry = Entry(old.first_seen if old else stamp, cp.size, FileKind.MODEL, old.declined if old else False,
                      cp.repo_id if _repo_ok(cp.repo_id) else None)
        entries[cp.real] = entry
        is_new = old is None and not baselined
        shown = Shown(cp.snapshot, cp.real, None, cp.size, entry.first_seen, entry.declined,
                      cp if scope == "all" or is_new else None)
        (new_cp if is_new else declined_cp if entry.declined else seen_cp).append(shown)
    return Found((*ranked, *new_cp), (*seen, *seen_cp), (*declined, *declined_cp),
                 Library(entries, gone=frozenset(lib.entries.keys() - entries.keys())), baselined,
                 lib.missing or lib.corrupt or entries != lib.entries)


def _cache_repos() -> tuple[Any, ...]:
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        return tuple(scan_cache_dir().repos)
    except CacheNotFound:
        return ()


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        with os.fdopen(open_regular(path), "rb") as fh:
            raw = fh.read(MODEL_CONFIG_CAP + 1)
        data = json.loads(raw) if len(raw) <= MODEL_CONFIG_CAP else None
    except (OSError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _generates_text(snapshot: Path) -> bool:
    architectures = (_json_object(snapshot / "config.json") or {}).get("architectures")
    return isinstance(architectures, list) and any(
        isinstance(a, str) and a.endswith(_TEXT_GENERATION) for a in architectures)


def _walk_ok(snapshot: Path) -> tuple[bool, list[str]]:
    """Every entry by lstat: a file, a directory, or a link to a regular file inside this repo's blobs."""
    blobs = os.path.realpath(snapshot.parent.parent / "blobs")
    names: list[str] = []

    def walk(directory: Path, depth: int) -> bool:
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError:
            return False
        for entry in entries:
            names.append(Path(entry.path).relative_to(snapshot).as_posix())
            if len(names) > DISCOVERY_MAX_ENTRIES:
                return False
            if entry.is_symlink():
                try:
                    inside = os.path.commonpath([os.path.realpath(entry.path), blobs]) == blobs
                    ok = inside and stat.S_ISREG(os.stat(entry.path).st_mode)
                except (OSError, ValueError):
                    ok = False
            elif entry.is_dir(follow_symlinks=False):
                ok = depth < DISCOVERY_MAX_DEPTH and walk(Path(entry.path), depth + 1)
            else:
                ok = entry.is_file(follow_symlinks=False)
            if not ok:
                return False
        return True

    return walk(snapshot, 1), names


def _snapshot_kind(snapshot: Path, names: Collection[str]) -> tuple[str | None, tuple[Path, ...]]:
    """gguf when it holds a `.gguf`; vllm when it holds safetensors and a text-generation config; else None."""
    ggufs = tuple(snapshot / n for n in names if n.lower().endswith(".gguf"))
    if ggufs:
        return "gguf", ggufs
    if any(n in _EMBEDDING_FILES or any(part.endswith("_Pooling") for part in n.split("/")) for n in names):
        return None, ()
    if _generates_text(snapshot) and any(n.endswith(".safetensors") for n in names):
        return "vllm", ()
    return None, ()


def _checkpoint(repo: Any, *, refuse_unloadable: bool = False) -> Checkpoint | None:
    rev = max(repo.revisions, key=lambda r: ("main" in r.refs, r.last_modified), default=None)
    if rev is None or not _repo_ok(repo.repo_id):
        return None
    snapshot = Path(rev.snapshot_path)
    found = Checkpoint(repo.repo_id, snapshot, os.path.realpath(snapshot), repo.size_on_disk)
    walked, names = _walk_ok(snapshot)
    if not walked:
        return replace(found, refusal=_UNLOADED) if refuse_unloadable else None
    kind, ggufs = _snapshot_kind(snapshot, names)
    if kind is None:
        return replace(found, kind="", refusal=_NO_WEIGHTS) if refuse_unloadable else None
    remote = kind == "vllm" and any("auto_map" in (_json_object(snapshot / n) or {})
                                    for n in ("config.json", "tokenizer_config.json"))
    return replace(found, kind=kind, ggufs=ggufs, refusal="needs trust_remote_code" if remote else None)


def _model_repos() -> list[Any]:
    return sorted((r for r in _cache_repos() if r.repo_type == "model" and r.size_on_disk > 0),
                  key=lambda r: r.repo_id)


def snapshots(*, limit: int = DISCOVERY_MAX_FILES) -> tuple[Checkpoint, ...]:
    """The cached repos in repo-id order that hold a model, one snapshot each: the `main` revision, else the latest."""
    try:
        repos = _model_repos()[:limit]
    except Exception:  # noqa: BLE001 — a cache that cannot be listed holds nothing
        return ()
    return tuple(c for r in repos if (c := _checkpoint(r)))


def snapshot_of(repo_id: str) -> Checkpoint | None:
    """The one repo's snapshot, resolved at the call; raises when the cache cannot be listed."""
    return next((c for r in _model_repos() if r.repo_id == repo_id and (c := _checkpoint(r, refuse_unloadable=True))), None)


def checkpoints(*, limit: int = DISCOVERY_MAX_FILES) -> tuple[Checkpoint, ...]:
    """The snapshots a vLLM launch can offer: kind vllm and no refusal."""
    return tuple(c for c in snapshots(limit=limit) if c.kind == "vllm" and c.refusal is None)


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
    """One launchable peer per model: a GPU llama.cpp record on the next free port from `start`, named as the preset."""
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

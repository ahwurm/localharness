"""The task flow's filer: the no-follow walk to an agent's task folders and every read and write of a task doc."""
from __future__ import annotations

import errno
import hashlib
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from localharness.core.caps import Caps
from localharness.core.private_files import PRIVATE_DIR_MODE, NotRegularFile, atomic_write_text, open_regular
from localharness.orchestrator.task_flow.record import Record

TRUNCATION_MARKER = "\n[truncated]\n"
SESSION_DOCS = ("jtbd.md", "reply-review.md")
_BOUND = Caps.DOC + len(TRUNCATION_MARKER.encode())
_DIR = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
_FOLDER = re.compile(r"t[0-9]+")
_TASK_DOC = re.compile(r"(?:(?:decisions|research|plan|step-[1-9][0-9]*)(?:-review)?|review)\.md")
_ATTEMPTS = 8


class Refused(Exception):
    """A path the walk will not follow, or a doc the filer will not touch: one line for the task call."""


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def valid_component(name: str) -> bool:
    return bool(name) and "/" not in name and os.sep not in name and "\x00" not in name and name not in (".", "..")


def _open_dir(path: str | os.PathLike[str], nofollow: bool, dir_fd: int | None = None) -> int:
    try:
        return os.open(path, _DIR | (getattr(os, "O_NOFOLLOW", 0) if nofollow else 0), dir_fd=dir_fd)
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.ENOTDIR):
            raise Refused(f"{Path(path).name} is a link or not a folder; the filer does not follow it") from e
        raise


def _walk(root: Path, workspace_layer: bool, parts: tuple[str, ...], create: bool) -> int:
    """An open descriptor for `root/parts`, one component at a time. The root and `agents` follow a link
    only on the machine layer; every other component refuses one."""
    fd = _open_dir(root, workspace_layer)
    for i, name in enumerate(parts):
        try:
            if create:
                try:
                    os.mkdir(name, PRIVATE_DIR_MODE, dir_fd=fd)
                except FileExistsError:
                    pass
            nxt = _open_dir(name, workspace_layer or i > 0, fd)
        finally:
            os.close(fd)
        fd = nxt
    return fd


def _listing(fd: int) -> list[str]:
    return os.listdir(fd)


def _highest(fd: int) -> int:
    return max((int(n[1:]) for n in _listing(fd) if _FOLDER.fullmatch(n)), default=0)


def _folder(task: int, name: str) -> str:
    folder = f"t{task}"
    if not _FOLDER.fullmatch(folder) or not _TASK_DOC.fullmatch(name):
        raise Refused(f"{folder}/{name} is not a task doc")
    return folder


def _session(name: str) -> str:
    if name not in SESSION_DOCS:
        raise Refused(f"{name} is not a session doc")
    return name


def _cut(text: str) -> tuple[str, bool]:
    raw = text.encode()
    return (text, False) if len(raw) <= Caps.DOC else (raw[: Caps.DOC].decode(errors="ignore") + TRUNCATION_MARKER, True)


def _slurp(fd: int, name: str) -> str | None:
    """The doc `name` inside `fd`, or None when it is missing or longer than a doc can be."""
    try:
        with os.fdopen(open_regular(name, dir_fd=fd), "rb") as fh:
            raw = fh.read(_BOUND + 1)
    except FileNotFoundError:
        return None
    except NotRegularFile as e:
        raise Refused(f"{name} is not a regular file") from e
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.ENOTDIR):
            raise Refused(f"{name} is a link; the filer does not follow it") from e
        raise
    if len(raw) > _BOUND:
        return None
    try:
        return raw.decode()
    except UnicodeDecodeError as e:
        raise Refused(f"{name} is not UTF-8") from e


@dataclass
class Docs:
    state_dir: Path
    workspace_layer: bool
    agent: str
    adopted: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if not valid_component(self.agent):
            raise Refused(f"{self.agent!r} is not one folder name")

    @contextmanager
    def _dir(self, *parts: str, create: bool):
        fd = _walk(self.state_dir, self.workspace_layer, ("agents", self.agent, *parts), create)
        try:
            yield fd
        finally:
            os.close(fd)

    def new_task(self) -> int:
        with self._dir("tasks", create=True) as fd:
            n = _highest(fd) + 1
            for _ in range(_ATTEMPTS):
                try:
                    os.mkdir(f"t{n}", PRIVATE_DIR_MODE, dir_fd=fd)
                except FileExistsError:
                    n = max(n, _highest(fd)) + 1
                    continue
                self.adopted.add(n)
                return n
        raise Refused("no free task folder")

    def write(self, task: int, name: str, text: str) -> tuple[str, bool]:
        folder = _folder(task, name)
        if task not in self.adopted:
            raise Refused(f"{folder} was not made by this session; it is read-only")
        cut, truncated = _cut(text)
        with self._dir("tasks", folder, create=False) as fd:
            atomic_write_text(name, cut, dir_fd=fd)
        return digest(cut), truncated

    def read(self, task: int, name: str) -> str | None:
        folder = _folder(task, name)
        try:
            with self._dir("tasks", folder, create=False) as fd:
                return _slurp(fd, name)
        except FileNotFoundError:
            return None

    def read_session(self, name: str) -> str:
        try:
            with self._dir("sessions", create=False) as fd:
                return _slurp(fd, _session(name)) or ""
        except FileNotFoundError:
            return ""

    def write_session(self, name: str, text: str) -> None:
        with self._dir("sessions", create=True) as fd:
            atomic_write_text(_session(name), _cut(text)[0], dir_fd=fd)

    def reset_sessions(self) -> None:
        for name in SESSION_DOCS:
            self.write_session(name, "")

    def path(self, task: int, name: str) -> Path:
        return self.state_dir / "agents" / self.agent / "tasks" / f"t{task}" / name

    def verified(self, record: Record, key: tuple[str, str | None]) -> bool:
        """A whole doc's current text still has the digest it was filed with."""
        row = record.filed.get(key)
        try:
            text = self.read(record.id, key[0])
        except Refused:
            return False
        return bool(row and text is not None and digest(text) == row.digest)

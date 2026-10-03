"""Owner-only files and directories, from the moment they exist — best effort.

The config directory holds keys, the phone token, grants, the trust store and every session log, so
it is 0700 and the files in it are 0600. Creating with the mode (never write-then-chmod) leaves no
moment in which another account on the machine could read a file. Best effort everywhere: a
filesystem that cannot express modes (Windows, some mounts), a read-only mount, or a directory
another account owns is left as it is and the call raises nothing — tightening a file must never
stop the harness from working (orchestrator ruling R16); doctor reports what is still open."""
from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Optional

PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700
OPEN_TO_OTHERS = 0o077  # any permission bit for the group or for other accounts


def private_opener(path: str, flags: int) -> int:
    """For open()/anyio.open_file(opener=...): a file this call creates is 0600."""
    return os.open(path, flags, PRIVATE_FILE_MODE)


def write_private_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path`: created 0600, or an existing file set to 0600 before a byte is
    written (the mode change is best effort; the write itself raises as a write would)."""
    with open(path, "wb", opener=private_opener) as fh:
        if hasattr(os, "fchmod"):  # an existing file keeps its old mode through O_CREAT
            try:
                os.fchmod(fh.fileno(), PRIVATE_FILE_MODE)
            except OSError:
                pass
        fh.write(data)


def touch_private(path: Path) -> None:
    """Create `path` empty at 0600 when it does not exist, so the write that follows lands in an
    owner-only file. An existing file is left alone. Any OSError is swallowed: the real write after
    it reports a real problem exactly as it did before."""
    try:
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, PRIVATE_FILE_MODE))
    except OSError:
        pass


def ensure_private_dir(path: Path) -> Optional[int]:
    """Create `path` at 0700 (missing parents at the default mode — `mkdir(mode=, parents=True)`
    does not make the leaf private on its own), or tighten an existing directory that others can
    reach to 0700. Returns the mode it had when it tightened one, else None — including when the
    change was refused (read-only, owned by another account): OSError is swallowed."""
    try:
        path.mkdir(mode=PRIVATE_DIR_MODE, parents=True)
        return None
    except FileExistsError:
        pass
    except OSError:
        return None
    if os.name == "nt":  # mode bits are not what guards a folder there; nothing to tighten
        return None
    try:
        before = os.stat(path)
        if not stat.S_ISDIR(before.st_mode) or not before.st_mode & OPEN_TO_OTHERS:
            return None
        os.chmod(path, PRIVATE_DIR_MODE)
        if os.stat(path).st_mode & OPEN_TO_OTHERS:  # a mount that ignores modes: nothing changed
            return None
    except OSError:
        return None
    return stat.S_IMODE(before.st_mode)


def readable_by_others(root: Path, limit: int = 10_000) -> list[Path]:
    """Files under `root` whose mode lets a group or other account read them (mode & 0o044),
    walking at most `limit` entries; unreadable entries and symlinks are skipped. Folders are not
    listed: a folder holds names, not contents, and below an owner-only root nobody else can
    reach one."""
    found: list[Path] = []
    seen = 0
    for top, dirs, files in os.walk(root):
        seen += len(dirs)
        for name in files:
            seen += 1
            if seen > limit:
                return found
            path = Path(top) / name
            try:
                st = os.lstat(path)
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode) and st.st_mode & 0o044:
                found.append(path)
        if seen >= limit:
            break
    return found

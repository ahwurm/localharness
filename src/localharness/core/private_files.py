"""Owner-only files and directories, from the moment they exist — best effort.

The config directory holds keys, the phone token, grants, the trust store and every session log, so
it is 0700 and the files in it are 0600. Creating with the mode (never write-then-chmod) leaves no
moment in which another account on the machine could read a file. Best effort everywhere: a
filesystem that cannot express modes (Windows, some mounts), a read-only mount, or a directory
another account owns is left as it is and the call raises nothing — tightening a file must never
stop the harness from working (orchestrator ruling R16); doctor reports what is still open."""
from __future__ import annotations

import errno
import logging
import os
import re
import secrets
import stat
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700
OPEN_TO_OTHERS = 0o077  # any permission bit for the group or for other accounts


class NotRegularFile(OSError):
    """The path names something other than a regular file (a FIFO, a device, a directory)."""


def open_regular(path: str | os.PathLike[str], *, dir_fd: int | None = None) -> int:
    """The one opener for a file the harness will READ and whose type it does not trust.

    A plain `open()` on a FIFO blocks before any check can run, and a symlink to a device hands back
    a stream with no end. `O_NONBLOCK` makes the open of a FIFO return at once (a regular file
    ignores the flag, so reads stay ordinary), `O_NOCTTY` keeps a terminal device from becoming the
    controlling terminal, and `fstat` on the descriptor — not a `stat` of the name, which could be
    swapped between the check and the open — decides: anything but a regular file is closed and
    refused with `NotRegularFile`. Callers wrap the returned descriptor with `os.fdopen(fd, "rb")`.
    A missing or unreadable path raises its own OSError, left as it is.

    With `dir_fd` the path is a name inside that directory and the open adds `O_NOFOLLOW`, so a link
    planted at the name is refused (ELOOP) rather than followed."""
    if os.name == "nt":  # no FIFOs, no controlling terminal: only the type check below applies
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    else:
        flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY | os.O_CLOEXEC
    if dir_fd is not None:
        fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), dir_fd=dir_fd)
    else:
        fd = os.open(path, flags)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise NotRegularFile(errno.EINVAL, "not a regular file", str(path))
    return fd


def open_private_dir(path: str | os.PathLike[str], *, create: bool = True) -> int:
    """A descriptor for the directory `path`, opened `O_DIRECTORY | O_NOFOLLOW`: a symlink at `path`
    is refused (ELOOP), never followed. With `create` a missing directory is made 0700 first
    (missing parents at the default mode). ELOOP, ENOTDIR and ENOENT propagate as OSError for the
    caller to read as "no directory"."""
    if create:
        try:
            Path(path).mkdir(mode=PRIVATE_DIR_MODE, parents=True, exist_ok=True)
        except FileExistsError:  # a non-directory (or a dangling link) is there; the open below says which
            pass
    flags = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_CLOEXEC", 0))
    return os.open(path, flags)


def read_private_text(name: str, *, dir_fd: int, cap: int, pattern: "re.Pattern[str]") -> Optional[str]:
    """The stripped text of the small file `name` inside the directory `dir_fd`, or None.

    The file is opened through `open_regular(dir_fd=)` (no link followed, a FIFO never blocks) and at
    most `cap + 1` bytes are read: more than `cap` bytes, bytes that are not UTF-8, any OSError, or a
    stripped text that does not `pattern.fullmatch` all read as None."""
    try:
        with os.fdopen(open_regular(name, dir_fd=dir_fd), "rb") as fh:
            raw = fh.read(cap + 1)
    except OSError:
        return None
    if len(raw) > cap:
        return None
    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    return text if pattern.fullmatch(text) else None


def refuse_unwritable(path: Path) -> None:
    """Raise PermissionError when `path` (a symlink's target) exists and may not be written, or
    its directory may not be entered and written, as `write_text` would refuse it. Replacing a
    read-only file through its directory would quietly override a read-only bit the owner set, and
    a caller that keeps an undo (migrate's backup) asks this BEFORE making one, so a file that will
    be refused never leaves a copy behind."""
    path = Path(path).expanduser()
    target = Path(os.path.realpath(path)) if path.is_symlink() else path
    if target.exists() and not os.access(target, os.W_OK):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(target))
    parent = target.parent
    if parent.exists() and not os.access(parent, os.W_OK | os.X_OK):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(parent))


def _restrict_tmp(path: Path) -> None:
    """Best-effort chmod of a temp file to :data:`PRIVATE_FILE_MODE` (POSIX only; never raises)."""
    if os.name != "posix":
        return
    try:
        os.chmod(path, PRIVATE_FILE_MODE)
    except OSError:
        log.debug("could not restrict %s to %o", path, PRIVATE_FILE_MODE, exc_info=True)


def _fsync_dir(directory: Path) -> None:
    """fsync `directory` so a rename into it survives a power cut. POSIX only, best effort: the
    file itself is already whole, and some filesystems refuse a directory fsync."""
    if os.name != "posix":
        return
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        log.debug("could not fsync directory %s", directory, exc_info=True)
    finally:
        os.close(fd)


def atomic_write_text(path: Path | str, text: str, *, refuse_read_only: bool = False,
                      follow_links: bool = False, dir_fd: int | None = None) -> None:
    """Replace the file `path` names with `text`, atomically. The one writer every small file the
    harness owns goes through (IN-21, D-28): a same-directory temp file (a cross-filesystem temp
    file would defeat `os.replace`'s atomicity on container/NFS mounts), fsynced, restricted to
    :data:`PRIVATE_FILE_MODE` BEFORE the rename, then `os.replace`d over the target, and the
    directory fsynced after it (best effort). A crash, Ctrl-C or a full disk leaves the old file or
    the new one, never a truncated one, and never a stranded temp file.

    By default a symlink at `path` is REPLACED in place by a regular file, and the file it pointed
    at is never touched (CR-31). A project's `.localharness/` comes from a cloned repository, and
    git keeps symlinks: following one would let a repository redirect a `--workspace` write into
    the machine overlay, or into any file the user can write, and create directories wherever it
    points. Only migrate passes `follow_links=True`: it rewrites files in the machine's own config
    dir that it has just read, and a symlinked one keeps its link. `refuse_read_only` refuses a
    file (or directory) the caller may not write (:func:`refuse_unwritable`) rather than replacing
    it through the directory.

    With `dir_fd` (an open directory, see :func:`open_private_dir`) `path` is a bare file name
    inside that directory: the temp file is created `O_EXCL | O_NOFOLLOW` in it, 0600, and renamed
    over the name by descriptor, so a link planted at the name or a directory swapped for a link
    after the open is never followed. `refuse_read_only` and `follow_links` do not apply to a
    directory-relative write and raise ValueError, as does a `path` holding a separator."""
    if dir_fd is not None:
        _atomic_write_in_dir(str(path), text, refuse_read_only, follow_links, dir_fd)
        return
    path = Path(path).expanduser()
    if follow_links:
        target = Path(os.path.realpath(path)) if path.is_symlink() else path
    else:
        target = path
        target.parent.mkdir(parents=True, exist_ok=True)
    if refuse_read_only:
        refuse_unwritable(target)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(tmp.name)
    try:
        with tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        # Tightened BEFORE the rename, so the file at `path` is never briefly group-readable — and
        # stated rather than inherited: `NamedTemporaryFile` happens to create at 0600 today, which
        # is not a guarantee this module should be resting an api_key on.
        _restrict_tmp(tmp_path)
        os.replace(str(tmp_path), str(target))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    _fsync_dir(target.parent)


def _atomic_write_in_dir(name: str, text: str, refuse_read_only: bool, follow_links: bool,
                         dir_fd: int) -> None:
    """The `dir_fd` form of :func:`atomic_write_text`."""
    if refuse_read_only or follow_links:
        raise ValueError("a directory-relative write neither follows links nor checks read-only bits")
    if not name or "/" in name or os.sep in name or name in (".", ".."):
        raise ValueError(f"dir_fd write needs a bare file name, not {name!r}")
    tmp = f".{name}.{secrets.token_hex(6)}.tmp"
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_CLOEXEC", 0))
    fd = os.open(tmp, flags, PRIVATE_FILE_MODE, dir_fd=dir_fd)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            if hasattr(os, "fchmod"):
                try:
                    os.fchmod(fh.fileno(), PRIVATE_FILE_MODE)
                except OSError:
                    log.debug("could not restrict %s to %o", tmp, PRIVATE_FILE_MODE, exc_info=True)
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except BaseException:
        try:
            os.unlink(tmp, dir_fd=dir_fd)
        except OSError:
            pass
        raise
    try:
        os.fsync(dir_fd)
    except OSError:
        log.debug("could not fsync the directory holding %s", name, exc_info=True)


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

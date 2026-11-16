"""A session's run folder: the agent's scratch, deleted when the session ends."""
from __future__ import annotations

import contextlib
import errno
import functools
import hmac
import logging
import os
import re
import secrets
import socket
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

from localharness.core.private_files import (
    _DIR_FLAGS, OPEN_TO_OTHERS, PRIVATE_DIR_MODE, PRIVATE_FILE_MODE, atomic_write_text,
    empty_dir, in_background, open_private_dir, read_private_text, remove_dir,
)

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

SCRATCH_SENTENCE = (
    "Scratch is for files only you need; it is deleted when the task is cleared or replaced, or the "
    "session ends, so do not delete it yourself. Anything you hand the person goes in the project; "
    "name scratch files to the person by file name."
)
REMOVING_LINE = "Removing scratch… (Ctrl+C leaves it for the next start)"
END_WAIT_S = 2.0
PRIVATE_END_WAIT_S = 30.0
_SLOW_AFTER_S = 1.0
_TOLD_VISIT_CAP = 2000
_ID = re.compile(r"[0-9a-f]{12}")
_DELETING = re.compile(r"\.deleting-[0-9a-f]{12}")
_MARKER = re.compile(r"[0-9a-f]{32}")


@dataclass
class RunFolder:
    path: Path
    fd: int | None = None
    parent: int | None = None
    held: dict = field(default_factory=dict)
    ident: os.stat_result | None = None
    private: bool = False
    _mu: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _ender: "Future[tuple[str, ...]] | None" = field(default=None, repr=False)

    @property
    def scratch(self) -> Path:
        return self.path / "scratch"

    @property
    def state(self) -> Path:
        return self.path / "state"

    @property
    def ending(self) -> bool:
        return self._ender is not None

    def command_env(self, workspace_root: str | None = None) -> dict[str, str]:
        if scratch_of(self, workspace_root) is None:
            return {}
        return {"LOCALHARNESS_SCRATCH": str(self.scratch), "TMPDIR": str(self.path / "t")}


_current: RunFolder | None = None


def current() -> RunFolder | None:
    return _current


def shadow(store: str) -> dict:
    """What the current run holds for `store`, by key."""
    run = _current
    return {k: v for (s, k), v in list(run.held.items()) if s == store} if run else {}


def hold(store: str, key: object, value: object, scratch: bool = False) -> bool:
    """Keep a store write in the current run's memory; False when the store must write to disk."""
    run = _current
    if run is None or not (run.private or scratch):
        return False
    run.held[(store, key)] = value
    return True


def scratch_of(run: RunFolder | None, workspace_root: str | None) -> Path | None:
    return run.scratch if run and not workspace_root else None


@functools.cache
def marker() -> str:
    return hmac.new(_machine_id().encode(), b"localharness run", "sha256").digest()[:16].hex()


def _machine_id() -> str:
    try:
        return Path("/etc/machine-id").read_text(encoding="ascii").strip() or socket.gethostname()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                                 capture_output=True, text=True, timeout=2).stdout
            if m := re.search(r'"IOPlatformUUID" = "([^"]+)"', out):
                return m.group(1)
        except (OSError, subprocess.SubprocessError):
            pass
    return socket.gethostname()


def _native() -> bool:
    return fcntl is not None and hasattr(os, "O_NOFOLLOW") and os.open in os.supports_dir_fd


def _lock(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def begin(config_dir: Path, private: bool = False) -> RunFolder | None:
    """Make this session's run folder, or None when there is no config yet or it cannot be made."""
    global _current
    config_dir = Path(config_dir)
    if not (config_dir / "config.yaml").exists():
        return None
    try:
        run = _begin(config_dir, private) if _native() else _fallback(private)
    except OSError as e:
        where = config_dir / "run" if e.errno == errno.ENOSPC else "the run folder"
        log.warning("no scratch this session: %s: %s", where, e.strerror or e)
        return None
    _current = run
    return run


def _begin(config_dir: Path, private: bool) -> RunFolder:
    try:
        parent = open_private_dir(config_dir / "run")
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.ENOTDIR):
            return _fallback(private)
        raise
    try:
        st = os.fstat(parent)
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid():
            run = _fallback(private)
        else:
            if st.st_mode & OPEN_TO_OTHERS:
                os.fchmod(parent, PRIVATE_DIR_MODE)
            _gitignore(parent)
            _sweep_briefly(parent)
            run = _make(parent, config_dir.resolve() / "run", private)
        if run.parent is None:
            os.close(parent)
        return run
    except BaseException:
        os.close(parent)
        raise


def _gitignore(parent: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(".gitignore", flags, PRIVATE_FILE_MODE, dir_fd=parent)
    except FileExistsError:
        return
    with os.fdopen(fd, "w") as fh:
        fh.write("*\n")


def _sweep_briefly(parent: int) -> None:
    dup = os.dup(parent)

    def run() -> None:
        try:
            sweep(dup)
        finally:
            os.close(dup)

    try:
        in_background(run).result(timeout=_SLOW_AFTER_S)
    except Exception:  # noqa: BLE001 - a slow or failing sweep never holds the start; it keeps going
        log.debug("run folder sweep", exc_info=True)


def _make(parent: int, run_dir: Path, private: bool) -> RunFolder:
    for _ in range(8):
        name = secrets.token_hex(6)
        try:
            os.mkdir(name, PRIVATE_DIR_MODE, dir_fd=parent)
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError(errno.EEXIST, "no free run folder name")
    fd = os.open(name, _DIR_FLAGS, dir_fd=parent)
    try:
        try:
            _lock(fd)
            enforced = _enforced(name, parent)
        except OSError:
            enforced = False
        if enforced:
            atomic_write_text(".marker", marker(), dir_fd=fd)
            os.mkdir("scratch", PRIVATE_DIR_MODE, dir_fd=fd)
            os.mkdir("t", PRIVATE_DIR_MODE, dir_fd=fd)
            if private:
                os.mkdir("state", PRIVATE_DIR_MODE, dir_fd=fd)
            return RunFolder(run_dir / name, fd, parent, private=private)
    except BaseException:
        _discard(name, parent, fd)
        raise
    _discard(name, parent, fd)
    return _fallback(private)


def _enforced(name: str, parent: int) -> bool:
    """A second open of the folder must not be able to take the lock the first holds."""
    probe = os.open(name, _DIR_FLAGS, dir_fd=parent)
    try:
        _lock(probe)
    except BlockingIOError:
        return True
    finally:
        os.close(probe)
    return False


def _discard(name: str, parent: int, fd: int) -> None:
    try:
        remove_dir(name, parent)
    except OSError:
        log.debug("run folder cleanup", exc_info=True)
    os.close(fd)


def _fallback(private: bool = False) -> RunFolder:
    path = Path(tempfile.mkdtemp(prefix="localharness-run-", dir="/tmp" if sys.platform == "darwin" else None))
    fd = None
    if hasattr(os, "O_NOFOLLOW") and os.open in os.supports_dir_fd:
        fd = os.open(path, _DIR_FLAGS)
        st = os.fstat(fd)
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid():
            os.close(fd)
            raise PermissionError(errno.EACCES, "temp folder is not ours", str(path))
        os.mkdir("scratch", PRIVATE_DIR_MODE, dir_fd=fd)
        os.mkdir("t", PRIVATE_DIR_MODE, dir_fd=fd)
        if private:
            os.mkdir("state", PRIVATE_DIR_MODE, dir_fd=fd)
    else:
        (path / "scratch").mkdir()
        (path / "t").mkdir()
        if private:
            (path / "state").mkdir()
    return RunFolder(path, fd, None, ident=os.lstat(path), private=private)


def end(run: RunFolder | None, say: Callable[[str], None] = lambda line: None) -> None:
    """Remove this session's run folder, waiting about two seconds (thirty when private); the next start finishes a slower one."""
    global _current
    if run is None:
        return
    if _current is run:
        _current = None
    with run._mu:
        if run._ender is None:
            run._ender = in_background(_finish, run)
        ender = run._ender
    try:
        try:
            left = ender.result(timeout=_SLOW_AFTER_S)
        except TimeoutError:
            say(REMOVING_LINE)
            left = ender.result(timeout=(PRIVATE_END_WAIT_S if run.private else END_WAIT_S) - _SLOW_AFTER_S)
    except TimeoutError:
        return
    except Exception:  # noqa: BLE001 - a folder that would not go is left for the next start
        log.warning("run folder not removed", exc_info=True)
        return
    if left:
        from localharness.channels.base import one_line
        say(one_line(f"Scratch not fully removed: {left[0]} stays until the next start"))


def _finish(run: RunFolder) -> tuple[str, ...]:
    """Empty the folder while holding its lock, then unlink its name."""
    try:
        if run.fd is None:
            if run.ident is not None and os.path.samestat(os.lstat(run.path), run.ident):
                return remove_dir(run.path.name, run.path.parent).left
            return ()
        name = run.path.name
        if run.parent is not None:
            gone = f".deleting-{name}"
            os.rename(name, gone, src_dir_fd=run.parent, dst_dir_fd=run.parent)
            name = gone
        left = remove_dir("state", run.fd).left + empty_dir(run.fd, name).left
        if not left:
            if run.parent is not None:
                os.rmdir(name, dir_fd=run.parent)
            else:
                os.rmdir(run.path)
        return left
    finally:
        for fd in (run.fd, run.parent):
            if fd is not None:
                os.close(fd)


def sweep(where: "int | Path") -> None:
    """Remove the run folders nobody holds that are this machine's or half made; leave the rest. `where` is an open `run/` or the config folder."""
    if not isinstance(where, int):
        if not _native():
            return
        try:
            fd = open_private_dir(Path(where) / "run", create=False)
        except OSError:
            return
        try:
            if os.fstat(fd).st_uid == os.geteuid():
                sweep(fd)
        finally:
            os.close(fd)
        return
    parent = where
    for name in os.listdir(parent):
        deleting = _DELETING.fullmatch(name)
        if not (deleting or _ID.fullmatch(name)):
            continue
        try:
            _sweep_one(name, parent, bool(deleting))
        except OSError:
            log.debug("run folder sweep: %s", name, exc_info=True)


def _sweep_one(name: str, parent: int, deleting: bool) -> None:
    if not stat.S_ISDIR(os.lstat(name, dir_fd=parent).st_mode):
        os.unlink(name, dir_fd=parent)
        return
    fd = os.open(name, _DIR_FLAGS, dir_fd=parent)
    try:
        _lock(fd)
        if deleting or _classify(fd) != "foreign":
            remove_dir(name, parent)
    except BlockingIOError:
        pass
    finally:
        os.close(fd)


def _classify(fd: int) -> str:
    """ours: this machine's marker; empty: a begin cut short; foreign: anything else."""
    mark = read_private_text(".marker", dir_fd=fd, cap=64, pattern=_MARKER)
    if mark == marker():
        return "ours"
    if mark is None and not _has(fd, "scratch"):
        return "empty"
    return "foreign"


def _has(fd: int, name: str) -> bool:
    try:
        os.lstat(name, dir_fd=fd)
        return True
    except OSError:
        return False


def fallback_cause(config_dir: Path) -> str | None:
    """Why sessions cannot use `<config dir>/run`, for doctor; None when they can."""
    if not _native():
        return None
    run = Path(config_dir) / "run"
    try:
        st = os.lstat(run)
    except OSError:
        return None
    if stat.S_ISLNK(st.st_mode):
        return "a link"
    if st.st_uid != os.geteuid():
        return "another account's"
    try:
        first, second = os.open(run, _DIR_FLAGS), os.open(run, _DIR_FLAGS)
    except OSError:
        return None
    try:
        _lock(first)
        try:
            _lock(second)
        except BlockingIOError:
            return None
        return "on a folder without enforced locks"
    except OSError:
        return None
    finally:
        os.close(first)
        os.close(second)


def foreign_count(config_dir: Path) -> int:
    """How many run folders another machine's marker, or none beside a scratch, leave alone."""
    try:
        parent = open_private_dir(Path(config_dir) / "run", create=False)
    except OSError:
        return 0
    found = 0
    try:
        for name in os.listdir(parent):
            if not _ID.fullmatch(name):
                continue
            try:
                fd = os.open(name, _DIR_FLAGS, dir_fd=parent)
            except OSError:
                continue
            try:
                found += _classify(fd) == "foreign"
            finally:
                os.close(fd)
    finally:
        os.close(parent)
    return found



def told(scratch: Path, since: float, reply: str) -> tuple[str, ...]:
    """Files under scratch made since `since` whose names the reply holds, as sorted posix paths relative to scratch."""
    found, stack, seen = [], [""], 0
    while stack and seen < _TOLD_VISIT_CAP:
        rel = stack.pop()
        with contextlib.suppress(OSError), os.scandir(scratch / rel) as entries:
            for entry in entries:
                seen += 1
                if seen > _TOLD_VISIT_CAP:
                    break
                name = f"{rel}/{entry.name}" if rel else entry.name
                if entry.is_dir(follow_symlinks=False):
                    stack.append(name)
                elif entry.name in reply and entry.stat(follow_symlinks=False).st_mtime >= since:
                    found.append(name)
    return tuple(sorted(found))

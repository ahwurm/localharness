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

from localharness.config.paths import RUN_DIR_NAME
from localharness.core.caps import Caps, capped
from localharness.core.private_files import (
    _DIR_FLAGS, OPEN_TO_OTHERS, PRIVATE_DIR_MODE, PRIVATE_FILE_MODE, atomic_write_text,
    empty_dir, in_background, open_private_dir, read_private_text, remove_dir,
)
from localharness.core.sanitize import one_line

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
PRIVATE_REMOVING_LINE = "Removing this private session… (Ctrl+C leaves it for the next start)"
LAYOUT = SCRATCH_DIR, TMP_DIR, STATE_DIR = "scratch", "t", "state"
SCRATCH_VAR, TMP_VAR = "LOCALHARNESS_SCRATCH", "TMPDIR"
COMMIT_MARKER = "LOCALHARNESS_COMMIT"
END_WAIT_S = 2.0
LINKED, FOREIGN, UNLOCKED = "a link", "another account's", "on a folder without enforced locks"
PRIVATE_LINKED_RUN_LINE = "{path} is a link or another account's; remove it, or use --config-dir <local folder>."
PRIVATE_UNLOCKED_LINE = "{path} does not enforce file locks; use --config-dir <local folder>."
PRIVATE_NEEDS_POSIX_LINE = "private sessions need POSIX file locks; use WSL on Windows."
PRIVATE_NO_RUN_LINE = "no run folder for a private session; use --config-dir <local folder>."
PRIVATE_END_WAIT_S = 30.0
_SLOW_AFTER_S = 1.0
_TOLD_VISIT_CAP = 2000
_ID = re.compile(r"[0-9a-f]{12}")
_DELETING = re.compile(r"\.deleting-[0-9a-f]{12}")
_MARKER = re.compile(r"[0-9a-f]{32}")
KEPT_NAMES = 6


@dataclass
class Written:
    turn: int
    announced: bool = False
    shown: bool = False


@dataclass
class RunFolder:
    path: Path
    fd: int | None = None
    parent: int | None = None
    held: dict = field(default_factory=dict)
    ident: os.stat_result | None = None
    private: bool = False
    turn: int = 0
    written: dict = field(default_factory=dict)
    told: set = field(default_factory=set)
    stated: bool = False
    _mu: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _ender: "Future[tuple[str, ...]] | None" = field(default=None, repr=False)

    @property
    def scratch(self) -> Path:
        return self.path / SCRATCH_DIR

    @property
    def state(self) -> Path:
        return self.path / STATE_DIR


_current: RunFolder | None = None


def current() -> RunFolder | None:
    return _current


def is_private() -> bool:
    return bool(_current and _current.private)


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


def note_written(run: RunFolder, paths, shown: bool) -> None:
    """Record the paths a call wrote, each with the turn that first wrote it; the first write wins."""
    for path in paths:
        run.written.setdefault(str(path), Written(run.turn, shown=shown))


def _stays(run: RunFolder) -> list[tuple[str, Written]]:
    """The written files still on disk outside the run folder."""
    own = (str(run.path), str(run.path.resolve()))
    return [(p, w) for p, w in list(run.written.items()) if not p.startswith(own) and os.path.lexists(p)]


def _inside(path: str, root) -> str | None:
    for base in (Path(root), Path(root).resolve()) if root else ():
        with contextlib.suppress(ValueError):
            return str(Path(path).relative_to(base))
    return None


def _named(names: list[str]) -> list[str]:
    return [one_line(n)[:Caps.FINDING] for n in capped(names, KEPT_NAMES)]


def kept_line(run: RunFolder, root) -> str:
    """What this turn's new files say about what stays: files outside the project, and project files a tool row has not shown, every one until the first statement."""
    made, outside = [], []
    for path, w in _stays(run):
        if w.turn != run.turn or w.announced:
            continue
        if (rel := _inside(path, root)) is None:
            outside.append(path)
        elif not w.shown or not run.stated:
            made.append(rel)
    if not (made or outside):
        return ""
    rows = [", ".join(_named(made))] * bool(made) + [f"also wrote outside the project: {', '.join(_named(outside))}"] * bool(outside)
    return "\n".join(rows + ["at least these; commands may leave more"] * (not run.stated))


def kept_names(run: RunFolder, root) -> tuple[str, ...]:
    """Every written file still on disk outside the run folder, project-relative where it can be, capped."""
    return tuple(_named([_inside(p, root) or p for p, _ in _stays(run)]))


def end_kept_turn(run: RunFolder, root) -> None:
    run.stated = run.stated or bool(kept_line(run, root))
    for w in run.written.values():
        if w.turn == run.turn:
            w.announced = True


def goes(run: RunFolder) -> tuple[tuple[str, str], ...]:
    """Files that go with the session: scratch files a reply named, pictures, saved pages, task docs."""
    from localharness.agent.taint import DOWNLOADS
    from localharness.core.artifacts import ARTIFACTS_DIR_NAME
    from localharness.orchestrator.task_flow.filer import task_doc
    found = [("file", n.rsplit("/", 1)[-1]) for n in sorted(run.told) if os.path.lexists(run.scratch / n)]
    state = run.state
    for kind, pattern in (("file", f"{ARTIFACTS_DIR_NAME}/*/*"), ("saved page", "/".join((*DOWNLOADS, "*"))),
                          ("task doc", str(task_doc(Path(), "*", "*", "**/*")))):
        found += [(kind, p.name) for p in sorted(state.glob(pattern)) if p.is_file() and not p.is_symlink()]
    return tuple(found)


def scratch_of(run: RunFolder | None, workspace_root: str | None) -> Path | None:
    inside = run and (not workspace_root or run.scratch.is_relative_to(Path(workspace_root).expanduser().resolve()))
    return run.scratch if inside else None


def command_env(run: RunFolder | None, workspace_root: str | None = None) -> dict[str, str]:
    """What every command the agent runs carries: the commit marker, and the run's folders when it has scratch."""
    env = {COMMIT_MARKER: "private" if run and run.private else "1"}
    if (scratch := scratch_of(run, workspace_root)) is None:
        return env
    return {**env, SCRATCH_VAR: str(scratch), TMP_VAR: str(run.path / TMP_DIR)}


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


def native() -> bool:
    return fcntl is not None and hasattr(os, "O_NOFOLLOW") and os.open in os.supports_dir_fd


def _lock(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def begin(config_dir: Path, private: bool = False, home: Path | None = None) -> RunFolder | None:
    """Make this session's run folder in `home` (a project's `.localharness/`) when it can hold it, else in the config folder, else in the temp folder; None with no config yet or when none can be made."""
    global _current
    config_dir = Path(config_dir)
    if not (config_dir / "config.yaml").exists():
        return None
    try:
        run = None
        if native():
            if home is not None:
                try:
                    run = _begin(Path(home), private, legacy=config_dir.resolve())
                except OSError:
                    log.debug("project run folder", exc_info=True)
            run = run or _begin(config_dir.resolve(), private)
        run = run or _fallback(private)
    except OSError as e:
        where = config_dir / RUN_DIR_NAME if e.errno == errno.ENOSPC else "the run folder"
        log.warning("no scratch this session: %s: %s", where, e.strerror or e)
        return None
    _current = run
    return run


def _ours(fd: int) -> bool:
    st = os.fstat(fd)
    return stat.S_ISDIR(st.st_mode) and st.st_uid == os.geteuid()


def _open_run(home: Path, create: bool = True) -> int | None:
    """An open `run/` inside `home`, or None when `home` or `run/` is not a folder of this account; no link is followed. The config folder is passed resolved, since a person may have linked it; a project's `.localharness/` as found, since a link there is refused."""
    top = open_private_dir(home, create=create)
    try:
        if not _ours(top):
            return None
        if create:
            with contextlib.suppress(FileExistsError):
                os.mkdir(RUN_DIR_NAME, PRIVATE_DIR_MODE, dir_fd=top)
        fd = os.open(RUN_DIR_NAME, _DIR_FLAGS, dir_fd=top)
    finally:
        os.close(top)
    if _ours(fd):
        return fd
    os.close(fd)
    return None


def _begin(home: Path, private: bool, legacy: Path | None = None) -> RunFolder | None:
    try:
        parent = _open_run(home)
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.ENOTDIR):
            return None
        raise
    if parent is None:
        return None
    try:
        if os.fstat(parent).st_mode & OPEN_TO_OTHERS:
            os.fchmod(parent, PRIVATE_DIR_MODE)
        _gitignore(parent)
        _sweep_briefly(parent, legacy)
        run = _make(parent, home.resolve() / RUN_DIR_NAME, private)
    except BaseException:
        os.close(parent)
        raise
    if run is None:
        os.close(parent)
    return run


def _gitignore(parent: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(".gitignore", flags, PRIVATE_FILE_MODE, dir_fd=parent)
    except FileExistsError:
        return
    with os.fdopen(fd, "w") as fh:
        fh.write("*\n")


def _sweep_briefly(parent: int, legacy: Path | None = None) -> None:
    dup = os.dup(parent)

    def run() -> None:
        try:
            sweep(dup)
            if legacy is not None:
                sweep(legacy)
        finally:
            os.close(dup)

    try:
        in_background(run).result(timeout=_SLOW_AFTER_S)
    except Exception:  # noqa: BLE001 - a slow or failing sweep never holds the start; it keeps going
        log.debug("run folder sweep", exc_info=True)


def _make(parent: int, run_dir: Path, private: bool) -> RunFolder | None:
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
            atomic_write_text(".gitignore", "*\n", dir_fd=fd)
            _lay_out(fd, private)
            return RunFolder(run_dir / name, fd, parent, private=private)
    except BaseException:
        _discard(name, parent, fd)
        raise
    _discard(name, parent, fd)
    return None


def _lay_out(where: int | Path, private: bool) -> None:
    for name in LAYOUT if private else LAYOUT[:2]:
        if isinstance(where, int):
            os.mkdir(name, PRIVATE_DIR_MODE, dir_fd=where)
        else:
            (where / name).mkdir()


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
    path = Path(tempfile.mkdtemp(prefix="localharness-run-", dir="/tmp" if sys.platform == "darwin" else None)).resolve()
    fd = None
    if hasattr(os, "O_NOFOLLOW") and os.open in os.supports_dir_fd:
        fd = os.open(path, _DIR_FLAGS)
        st = os.fstat(fd)
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid():
            os.close(fd)
            raise PermissionError(errno.EACCES, "temp folder is not ours", str(path))
    _lay_out(path if fd is None else fd, private)
    return RunFolder(path, fd, None, ident=os.lstat(path), private=private)


def end(run: RunFolder | None, say: Callable[[str], None] = lambda line: None) -> str:
    """Remove this session's run folder, waiting about two seconds (thirty when private); the next start finishes a slower one. Returns the line naming what is left, or ''."""
    global _current
    if run is None:
        return ""
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
            say(PRIVATE_REMOVING_LINE if run.private else REMOVING_LINE)
            left = ender.result(timeout=(PRIVATE_END_WAIT_S if run.private else END_WAIT_S) - _SLOW_AFTER_S)
    except TimeoutError:
        gone = run.path.with_name(f".deleting-{run.path.name}") if run.parent is not None else run.path
        left = (str(gone),) if run.private else ()
    except Exception:  # noqa: BLE001 - a folder that would not go is left for the next start
        log.warning("run folder not removed", exc_info=True)
        return ""
    if not left:
        return ""
    shown = _shown(run.path.parent / left[0])
    line = one_line(f"Private session not fully removed: {shown} stays until localharness next starts" if run.private
                    else f"Scratch not fully removed: {shown} stays until the next start")
    say(line)
    return line


def _shown(path: Path) -> str:
    try:
        return _inside(str(path), Path.cwd()) or str(path)
    except OSError:
        return str(path)


def _finish(run: RunFolder) -> tuple[str, ...]:
    """Empty the folder while holding its lock, then unlink its name."""
    try:
        if run.fd is None:
            if run.ident is not None and os.path.samestat(os.lstat(run.path), run.ident):
                return remove_dir(run.path.name, run.path.parent)
            return ()
        name = run.path.name
        if run.parent is not None:
            gone = f".deleting-{name}"
            os.rename(name, gone, src_dir_fd=run.parent, dst_dir_fd=run.parent)
            name = gone
        left = tuple(f"{name}/{p}" for p in remove_dir(STATE_DIR, run.fd)) + empty_dir(run.fd, name)
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
    """Remove the run folders nobody holds that are this machine's or half made; leave the rest. `where` is an open `run/` or a folder holding one (a project's `.localharness/`, or the resolved config folder)."""
    if not isinstance(where, int):
        if not native():
            return
        try:
            fd = _open_run(Path(where), create=False)
        except OSError:
            return
        if fd is None:
            return
        try:
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
    if mark is None and not _has(fd, SCRATCH_DIR):
        return "empty"
    return "foreign"


def _has(fd: int, name: str) -> bool:
    try:
        os.lstat(name, dir_fd=fd)
        return True
    except OSError:
        return False


def fallback_cause(home: Path) -> str | None:
    """Why a session cannot keep its run folder in `home`, for doctor; None when it can."""
    if not native():
        return None
    run = Path(home) / RUN_DIR_NAME
    for place in (Path(home), run):
        try:
            st = os.lstat(place)
        except OSError:
            return None
        if stat.S_ISLNK(st.st_mode):
            return LINKED
        if st.st_uid != os.geteuid():
            return FOREIGN
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
        return UNLOCKED
    except OSError:
        return None
    finally:
        os.close(first)
        os.close(second)


def private_run_problem(run: RunFolder | None, cfg: Path) -> str | None:
    """Why a private session cannot run in this run folder, or None; the caller ends a refused run."""
    if run is None:
        return PRIVATE_NO_RUN_LINE
    if run.parent is not None:
        return None
    return (PRIVATE_LINKED_RUN_LINE if fallback_cause(Path(cfg).resolve()) in (LINKED, FOREIGN)
            else PRIVATE_UNLOCKED_LINE if native() else PRIVATE_NEEDS_POSIX_LINE).format(path=Path(cfg) / RUN_DIR_NAME)


def foreign_count(home: Path) -> int:
    """How many run folders another machine's marker, or none beside a scratch, leave alone."""
    try:
        parent = _open_run(Path(home), create=False)
    except OSError:
        return 0
    if parent is None:
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

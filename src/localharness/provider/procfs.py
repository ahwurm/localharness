"""Every `/proc` read of the init import and of `/model`: who listens on a port, which process holds
that socket, and one capped, uid-checked record of that process.

What it reads: `/proc/net/tcp` and `/proc/net/tcp6` (LISTEN rows only), the `fd` links of the
caller's own processes, and for one process `stat`, `cgroup`, `cmdline`, `environ` and `maps`, plus
the `exe` and `cwd` link texts. What it refuses: a process of another uid, a command line or an
environment past its cap, an environment it may not read, a pid that was reused while it was read.
A refusal is a value (`ProcRefusal`), never an exception: the caller turns it into the one
"attached only" line. Nothing here parses a GGUF, speaks HTTP or starts a process.

`read_process` opens `/proc/<pid>` once as a directory fd and reads every file through that fd, so
the uid it checked is the uid of the directory it reads, whatever happens to the pid number meanwhile;
the start time is read before and after as the second guard. `exe` and `cwd` are only ever
`readlink`-ed, never opened. `PROC_ROOT` is read at call time, so a test points it at a tmp tree.

Pure standard library (os, re, dataclasses); the one cross-module use is the start-time parser the
pidfile code in `provider.server` already owns. `pwd` is not imported at module level and
the directory-open flag is read with `getattr`, so the module still imports where those are absent.
"""
from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from localharness.provider import server

__all__ = [
    "PROC_ROOT", "NET_TCP_CAP", "PID_SCAN_CAP", "FD_SCAN_CAP", "FD_TOTAL_CAP", "PROC_ENTRY_CAP",
    "CMDLINE_CAP", "ENVIRON_CAP",
    "STAT_CAP", "CGROUP_CAP", "MAPS_LINE_CAP", "MAPPED_PATHS_CAP",
    "Listener", "ProcRecord", "ProcRefusal", "listeners", "reachable_first", "pid_for_inode",
    "read_process", "parent_pid", "is_self_or_descendant",
]

PROC_ROOT = "/proc"
"""The /proc mount. Read through `os.path.join(PROC_ROOT, ...)` at call time, never captured."""

NET_TCP_CAP = 4 << 20
PID_SCAN_CAP = 4096
"""Most of THIS user's processes `pid_for_inode` enters; another user's pids are not counted against it."""
FD_SCAN_CAP = 16384
"""Most fd links read of ONE process; a process past it is passed over, the scan goes on."""
FD_TOTAL_CAP = 1 << 20
"""Most fd links read across all of this user's processes."""
PROC_ENTRY_CAP = 1 << 20
"""Most numeric `/proc` entries looked at, whoever owns them."""
CMDLINE_CAP = 256 * 1024
ENVIRON_CAP = 256 * 1024
STAT_CAP = 4 * 1024
CGROUP_CAP = 16 * 1024
MAPS_LINE_CAP = 200_000
MAPPED_PATHS_CAP = 32

_LISTEN = "0A"
_UNIT_NAME = re.compile(r"[A-Za-z0-9:_.\\@-]+\.service")
_UNIT_NAME_MAX = 255
_MAPS_READ_LIMIT = 16384
_READ_CHUNK = 65536


@dataclass(frozen=True)
class Listener:
    """One LISTEN row: the socket inode, the uid /proc/net/tcp names as its owner and the local address
    it is bound to (`""` when the row's address could not be read; not part of equality)."""

    inode: int
    uid: int
    local_ip: str = field(default="", compare=False)

    @property
    def reachable_on_loopback(self) -> bool:
        """True for a socket bound to loopback or to every address (or whose address is unknown): the
        rows a connection to 127.0.0.1 can have reached. A socket bound to another interface's address
        on the same port is a different server."""
        if not self.local_ip:
            return True
        try:
            ip = ipaddress.ip_address(self.local_ip)
        except ValueError:
            return True
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        return ip.is_loopback or ip.is_unspecified


@dataclass(frozen=True)
class ProcRecord:
    """What `/proc` says about one process of this user. `environ=None` and `cwd=None` mean
    "unknown" (the sources that cannot know them); `unit` is the systemd user unit or None;
    `mapped_ggufs` are the candidate model paths from `maps`, in maps order, not yet typed."""

    pid: int
    uid: int
    argv: tuple[str, ...]
    exe: str | None
    cwd: str | None
    start_time: int
    unit: str | None
    environ: Mapping[str, str] | None
    mapped_ggufs: tuple[str, ...]


@dataclass(frozen=True)
class ProcRefusal:
    """Why a process was not read. `reason` is the text the init sentence carries verbatim
    (`<who> <reason> — attached only, no server record written`); `unit` is the unit the cgroup
    named when it was already read."""

    reason: str
    unit: str | None = None


def _proc_path(*parts: str) -> str:
    return os.path.join(PROC_ROOT, *parts)


# ------------------------------------------------------------------------------- listeners


def _net_tcp_rows(name: str) -> list[list[str]]:
    """The data rows of `<PROC_ROOT>/net/<name>` as split fields; a missing or unreadable file is
    no rows, a partial last line is dropped, and at most `NET_TCP_CAP` bytes are read."""
    try:
        with open(_proc_path("net", name), "rb") as fh:
            raw = fh.read(NET_TCP_CAP)
    except OSError:
        return []
    lines = raw.split(b"\n")
    lines.pop()  # the text after the last newline is empty or a partial line
    return [line.decode("ascii", "replace").split() for line in lines[1:]]


def _decode_ip(text: str) -> str:
    """The address of a `/proc/net/tcp{,6}` row's hex `local_address` (host-endian 32-bit words, so
    little-endian on the machines this runs on), or `""` for anything else."""
    try:
        raw = bytes.fromhex(text)
    except ValueError:
        return ""
    if len(raw) == 4:
        return str(ipaddress.IPv4Address(raw[::-1]))
    if len(raw) == 16:
        return str(ipaddress.IPv6Address(b"".join(raw[i:i + 4][::-1] for i in range(0, 16, 4))))
    return ""


def reachable_first(found: list[Listener]) -> list[Listener]:
    """`found` with the rows a loopback connection can have reached ahead of the others (stable)."""
    return sorted(found, key=lambda lst: not lst.reachable_on_loopback)


def listeners(port: int) -> list[Listener]:
    """The LISTEN sockets on local `port`, `net/tcp` rows first then `net/tcp6`, in file order.
    A row with inode 0 (a socket on its way out) never matches."""
    found: list[Listener] = []
    for name in ("tcp", "tcp6"):
        for fields in _net_tcp_rows(name):
            try:
                if fields[3] != _LISTEN or fields[9] == "0":
                    continue
                address, _, local_port = fields[1].rpartition(":")
                if int(local_port, 16) != port:
                    continue
                found.append(Listener(int(fields[9]), int(fields[7]), _decode_ip(address)))
            except (IndexError, ValueError):
                continue
    return found


def pid_for_inode(inode: int) -> int | None:
    """The pid of this user's process that holds socket `inode`, or None. Only `/proc/<pid>` of
    this uid is entered. At most `PID_SCAN_CAP` of this user's pids, `FD_SCAN_CAP` links of one process
    and `FD_TOTAL_CAP` links in all are looked at (another user's pids cost nothing against the first;
    a process past the second is passed over, not the whole scan); a cap hit is `None`."""
    wanted = f"socket:[{inode}]"
    me = os.getuid()
    entries = pids = total_links = 0
    try:
        top = os.scandir(_proc_path())
    except OSError:
        return None
    with top:
        for entry in top:
            if not entry.name.isdigit():
                continue
            entries += 1
            if entries > PROC_ENTRY_CAP:
                return None
            try:
                if entry.stat(follow_symlinks=False).st_uid != me:
                    continue
                pids += 1  # only this user's processes count against the cap
                if pids > PID_SCAN_CAP:
                    return None
                with os.scandir(os.path.join(entry.path, "fd")) as fds:
                    links = 0
                    for fd_entry in fds:
                        links += 1
                        total_links += 1
                        if total_links > FD_TOTAL_CAP:
                            return None
                        if links > FD_SCAN_CAP:
                            break  # this process alone holds more than the cap: not read further
                        try:
                            if os.readlink(fd_entry.path) == wanted:
                                return int(entry.name)
                        except OSError:
                            continue
            except OSError:
                continue
    return None


def parent_pid(pid: int) -> int | None:
    """The parent pid in `/proc/<pid>/stat` (field 4), or None when it cannot be read exactly."""
    try:
        with open(_proc_path(str(pid), "stat"), "rb") as fh:
            raw = fh.read(STAT_CAP + 1)
        if len(raw) > STAT_CAP:
            return None
        return int(raw.rsplit(b")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return None


def is_self_or_descendant(pid: int, ancestor: int, *, depth: int = 16) -> bool:
    """True when `pid` is `ancestor` or reaches it by at most `depth` parent links. A pid whose parent
    cannot be read ends the walk with False."""
    for _ in range(depth + 1):
        if pid == ancestor:
            return True
        parent = parent_pid(pid)
        if parent is None or parent <= 1 or parent == pid:
            return False
        pid = parent
    return False


# ------------------------------------------------------------------------------- one process


def _read_at(name: str, dir_fd: int, cap: int) -> bytes:
    """Up to `cap + 1` bytes of file `name` under directory fd `dir_fd` (one byte past the cap is
    how a caller knows it was exceeded). The one seam every per-pid file read goes through."""
    fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC, dir_fd=dir_fd)
    try:
        chunks: list[bytes] = []
        total = 0
        while total <= cap:
            chunk = os.read(fd, min(_READ_CHUNK, cap + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _unit_from_cgroup(raw: bytes) -> str | None:
    """The systemd user unit of a `0::<path>` cgroup line, or None. The unit is trusted only as the
    last component below `user@<uid>.service`, in systemd's own charset, within 255 bytes: an
    agent can make a sub-cgroup of any name, so anything else is not shown as a unit."""
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.startswith("0::"):
            continue
        parts = [p for p in line[3:].split("/") if p]
        if len(parts) < 2 or f"user@{os.getuid()}.service" not in parts[:-1]:
            return None
        name = parts[-1]
        if len(name.encode("utf-8")) > _UNIT_NAME_MAX or not _UNIT_NAME.fullmatch(name):
            return None
        return name
    return None


def _split_nul(raw: bytes) -> list[bytes]:
    parts = raw.split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()
    return parts


def _parse_environ(raw: bytes) -> Mapping[str, str]:
    env: dict[str, str] = {}
    for item in _split_nul(raw):
        name, sep, value = os.fsdecode(item).partition("=")
        if sep and name:
            env.setdefault(name, value)  # getenv answers with the first entry
    return MappingProxyType(env)


def _mapped_candidates(dir_fd: int) -> tuple[str, ...]:
    """The distinct `maps` pathnames, in maps order, that look like a model file: they end `.gguf`
    in any case, or sit in a directory named `blobs` (a Hugging Face hub download). Lines are
    streamed and counted, so `MAPS_LINE_CAP` bounds the work; a cap hit or an unreadable `maps`
    keeps what was found and is never an error. Whether a candidate IS a model is not decided here."""
    try:
        fd = os.open("maps", os.O_RDONLY | os.O_CLOEXEC, dir_fd=dir_fd)
    except OSError:
        return ()
    found: list[str] = []
    try:
        with os.fdopen(fd, "rb") as fh:
            for _ in range(MAPS_LINE_CAP):
                line = fh.readline(_MAPS_READ_LIMIT)
                if not line:
                    break
                if not line.endswith(b"\n") and len(line) >= _MAPS_READ_LIMIT:
                    continue  # an over-long fragment is not a maps line
                fields = line.rstrip(b"\n").split(None, 5)
                if len(fields) < 6:
                    continue
                path = os.fsdecode(fields[5])
                if path.endswith(" (deleted)"):
                    continue
                if not (path.lower().endswith(".gguf")
                        or os.path.basename(os.path.dirname(path)) == "blobs"):
                    continue
                if path in found:
                    continue
                found.append(path)
                if len(found) >= MAPPED_PATHS_CAP:
                    break
    except OSError:
        pass
    return tuple(found)


def _link(name: str, dir_fd: int) -> str | None:
    try:
        return os.fsdecode(os.readlink(name, dir_fd=dir_fd))
    except OSError:
        return None


def _start_time(dir_fd: int) -> int | None:
    try:
        raw = _read_at("stat", dir_fd, STAT_CAP)
    except OSError:
        return None
    if len(raw) > STAT_CAP:
        return None
    return server.start_time_from_stat(raw)


def read_process(pid: int) -> ProcRecord | ProcRefusal:
    """The `/proc` record of this user's process `pid`, or the named reason it was not read.

    One directory fd for the whole read; the start time is read before the other files and again
    after them, so a pid reused meanwhile is a refusal and never a mixed record. Never raises for a
    vanished or unreadable process."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_CLOEXEC
    try:
        fd = os.open(_proc_path(str(pid)), flags)
    except OSError:
        return ProcRefusal("is gone")
    try:
        return _read_process_at(pid, fd)
    finally:
        os.close(fd)


def _read_process_at(pid: int, fd: int) -> ProcRecord | ProcRefusal:
    try:
        owner = os.fstat(fd).st_uid
    except OSError:
        return ProcRefusal("is gone")
    if owner != os.getuid():
        return ProcRefusal("is owned by another uid")

    start = _start_time(fd)
    if start is None:
        return ProcRefusal("is gone")

    unit: str | None = None
    try:
        cgroup = _read_at("cgroup", fd, CGROUP_CAP)
        if len(cgroup) <= CGROUP_CAP:
            unit = _unit_from_cgroup(cgroup)
    except OSError:
        pass  # no unit is shown; the pid and the command line still are

    try:
        cmdline = _read_at("cmdline", fd, CMDLINE_CAP)
    except OSError:
        return ProcRefusal("is gone", unit)
    if len(cmdline) > CMDLINE_CAP:
        return ProcRefusal(f"has its command line over {CMDLINE_CAP} bytes", unit)

    try:
        environ_raw = _read_at("environ", fd, ENVIRON_CAP)
    except PermissionError:
        return ProcRefusal("has an unreadable environment", unit)
    except OSError:
        return ProcRefusal("is gone", unit)
    if len(environ_raw) > ENVIRON_CAP:
        return ProcRefusal(f"has its environment over {ENVIRON_CAP} bytes", unit)

    mapped = _mapped_candidates(fd)
    exe = _link("exe", fd)
    cwd = _link("cwd", fd)

    if _start_time(fd) != start:
        return ProcRefusal("pid reused during the read", unit)

    return ProcRecord(
        pid=pid,
        uid=os.getuid(),
        argv=tuple(os.fsdecode(a) for a in _split_nul(cmdline)),
        exe=exe,
        cwd=cwd,
        start_time=start,
        unit=unit,
        environ=_parse_environ(environ_raw),
        mapped_ggufs=mapped,
    )

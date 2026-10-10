"""A fake `/proc` tree for the procfs tests: `net/tcp{,6}` rows and per-pid files under a tmp root.

`fake_proc` builds an empty tree and points `procfs.PROC_ROOT` at it, so no test reads the real
`/proc` of anything. The pid directories are owned by the test user, so `os.getuid()` matches their
`st_uid`; an "other uid" case patches `procfs.os.getuid`.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

from localharness.provider import procfs

_NET_HEADER = ("  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  "
               "timeout inode\n")


def stat_bytes(pid: int, start: int, *, comm: str = "llama-server", state: str = "S") -> bytes:
    """A `/proc/<pid>/stat` line with `start` as field 22: after the last ")" the tokens begin at
    field 3, so the state is index 0 and the start time index 19 (17 fields lie between)."""
    fields = [state, "1", *(["0"] * 17), str(start), "0", "0"]
    return f"{pid} ({comm}) {' '.join(fields)}\n".encode()


class FakeProc:
    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "net").mkdir(parents=True, exist_ok=True)
        self._rows: dict[str, list[str]] = {"tcp": [], "tcp6": []}
        self._flush()

    def _flush(self) -> None:
        for family, rows in self._rows.items():
            (self.root / "net" / family).write_text(_NET_HEADER + "".join(rows), encoding="ascii")

    def listen(self, port: int, inode: int, uid: int, *, family: str = "tcp", state: str = "0A",
               ip: str = "0100007F") -> None:
        """Append one well-formed row. `ip` is the little-endian hex address (default 127.0.0.1);
        a tcp6 row gets the 32-digit loopback unless `ip` says otherwise."""
        if family == "tcp6" and len(ip) == 8:
            ip = "00000000000000000000000001000000"
        index = len(self._rows[family])
        row = (f"{index:4d}: {ip}:{port:04X} {'0' * len(ip)}:0000 {state} 00000000:00000000 "
               f"00:00000000 00000000 {uid:5d}        0 {inode} 1 0000000000000000 100 0 0 10 0\n")
        self._rows[family].append(row)
        self._flush()

    def raw_net(self, family: str, text: str) -> None:
        """Replace `net/<family>` with `text` verbatim (a header-only, partial or hostile file)."""
        (self.root / "net" / family).write_bytes(text.encode("ascii"))

    def process(self, pid: int, *, argv: Iterable[str] = ("llama-server",),
                environ: Mapping[str, str] | None = None, cgroup: str | None = None,
                maps: Iterable[str] = (), exe: str | None = None, cwd: str | None = None,
                fds: Mapping[int, int] | None = None, start_time: int = 777,
                comm: str = "llama-server", stat: bytes | None = None,
                cmdline: bytes | None = None, environ_raw: bytes | None = None,
                maps_text: str | None = None) -> Path:
        """Write `<root>/<pid>/` and return it. `fds` maps an fd number to a socket inode;
        `maps` are bare pathnames (rendered as maps lines); the `*_raw`/`*_text`/`stat`/`cmdline`
        knobs write the file verbatim."""
        d = self.root / str(pid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "stat").write_bytes(stat if stat is not None else stat_bytes(pid, start_time, comm=comm))
        (d / "cmdline").write_bytes(
            cmdline if cmdline is not None else b"".join(a.encode() + b"\0" for a in argv))
        if environ_raw is not None:
            (d / "environ").write_bytes(environ_raw)
        else:
            (d / "environ").write_bytes(
                b"".join(f"{k}={v}".encode() + b"\0" for k, v in (environ or {}).items()))
        if cgroup is not None:
            (d / "cgroup").write_text(f"0::{cgroup}\n", encoding="utf-8")
        else:
            (d / "cgroup").write_text("", encoding="utf-8")
        if maps_text is not None:
            (d / "maps").write_text(maps_text, encoding="utf-8")
        else:
            lines = [f"7f00{i:08x}-7f01{i:08x} r--s 00000000 08:01 {1000 + i}"
                     f"                      {path}\n" for i, path in enumerate(maps)]
            (d / "maps").write_text("".join(lines), encoding="utf-8")
        if exe is not None:
            (d / "exe").symlink_to(exe)
        if cwd is not None:
            (d / "cwd").symlink_to(cwd)
        if fds:
            (d / "fd").mkdir(exist_ok=True)
            for number, inode in fds.items():
                (d / "fd" / str(number)).symlink_to(f"socket:[{inode}]")
        return d


@pytest.fixture
def fake_proc(tmp_path: Path, monkeypatch) -> FakeProc:
    root = tmp_path / "proc"
    root.mkdir()
    monkeypatch.setattr(procfs, "PROC_ROOT", str(root))
    return FakeProc(root)


def real_uid() -> int:
    return os.getuid()

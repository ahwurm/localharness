"""Who else is driving this agent right now (WEBCH-29).

**The hazard this makes visible.** `history.jsonl` and `compact.md` are appended without a lock.
One process is fine. TWO processes on the same agent in the same workspace — `localharness web`
serving a phone while `localharness start` runs in a terminal, which is a habit, not an exotic
case — interleave their appends, and an append above `PIPE_BUF` can tear. Nothing today notices.

**It warns; it never refuses.** Owner ruling: a second session keeps working. This is a presence
REGISTRY, not a mutex, and the name "advisory lock" in the PRD describes its effect rather than
its mechanism — said plainly here so nobody later reads `lock` and expects exclusion.

**Why a directory of files and not one file.** Each process owns exactly one file named after its
pid, so two starting at once cannot lose each other's write, and no locking is needed to read.
Nothing is deleted on exit: liveness is decided by asking the OS whether the pid is still there,
which is also the right answer after a SIGKILL, a crash or a pulled plug — a cleanup that only
runs on a graceful exit would leave exactly the stale entries it was written to prevent.

**The known weakness, and why it is acceptable.** A recycled pid can make a dead session look
live. The cost of being wrong is one extra line of warning text, not a refused start-up, and that
asymmetry is the whole reason this is advisory: a check that cannot hurt you is allowed to be
cheap.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import structlog

log = structlog.get_logger(__name__)

PRESENCE_DIR_NAME = "live-sessions"
"""Under the GLOBAL config dir: one machine, one register. A workspace-local one could not see
the sibling session it exists to notice."""


@dataclass(frozen=True)
class LiveSession:
    """One other process, as it described itself when it started."""

    pid: int
    agent: str
    channel: str
    workspace: str
    session_id: str
    started_at: float

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.started_at)

    def describe(self) -> str:
        minutes = int(self.age_s // 60)
        when = f"{minutes} min ago" if minutes else "just now"
        return f"{self.channel} (pid {self.pid}) in {self.workspace}, started {when}"


def presence_dir(config_dir: Optional[str | Path] = None) -> Path:
    from localharness.config.paths import global_config_dir

    return global_config_dir(config_dir) / PRESENCE_DIR_NAME


def _alive(pid: int) -> bool:
    """Is that process still there? Signal 0 asks without delivering anything."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists and belongs to somebody else. Alive for our purposes.
        return True
    except (OSError, ValueError):
        return False
    return True


def _key(workspace: str | Path) -> str:
    """REALPATH, like the trust and grant stores. A symlinked project path is the same project,
    and two sessions reaching the same `history.jsonl` by different names is precisely the case
    this must catch."""
    try:
        return str(Path(workspace).resolve())
    except (OSError, RuntimeError, ValueError):
        return str(workspace)


def live_sessions(
    config_dir: Optional[str | Path] = None, *, agent: Optional[str] = None,
    workspace: Optional[str | Path] = None, exclude_pid: Optional[int] = None,
) -> list[LiveSession]:
    """Everything currently registered, pruning anything whose process is gone.

    Filtered on (agent, realpath workspace) when both are given: two projects that happen to use
    the same agent name write to different files and are not in each other's way, so warning
    about them would be noise — and noise is how a real warning gets ignored.
    """
    directory = presence_dir(config_dir)
    want = _key(workspace) if workspace is not None else None
    found: list[LiveSession] = []
    try:
        entries = sorted(directory.glob("*.json"))
    except OSError:
        return []
    for path in entries:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            session = LiveSession(
                pid=int(row["pid"]), agent=row.get("agent", ""), channel=row.get("channel", ""),
                workspace=row.get("workspace", ""), session_id=row.get("session_id", ""),
                started_at=float(row.get("started_at", 0.0)),
            )
        except (OSError, ValueError, KeyError, TypeError):
            _discard(path)
            continue
        if not _alive(session.pid):
            _discard(path)
            continue
        if exclude_pid is not None and session.pid == exclude_pid:
            continue
        if agent is not None and session.agent != agent:
            continue
        if want is not None and _key(session.workspace) != want:
            continue
        found.append(session)
    return found


def register(
    config_dir: Optional[str | Path] = None, *, agent: str, channel: str, session_id: str,
    workspace: str | Path, pid: Optional[int] = None,
) -> list[LiveSession]:
    """Announce this session and return the OTHERS already driving the same agent here.

    Registering happens even when the list comes back empty, and that is the half that makes the
    next process's warning possible: a registry only one side writes to tells nobody anything.
    """
    mine = os.getpid() if pid is None else pid
    others = live_sessions(config_dir, agent=agent, workspace=workspace, exclude_pid=mine)
    directory = presence_dir(config_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{mine}.json").write_text(json.dumps({
            "pid": mine,
            "agent": agent,
            "channel": channel,
            "workspace": str(workspace),
            "session_id": session_id,
            "started_at": time.time(),
        }), encoding="utf-8")
    except OSError:
        # A registry that cannot be written is a lost warning, never a failed start-up.
        log.warning("session_presence_unwritable", path=str(directory), exc_info=True)
    return others


def release(config_dir: Optional[str | Path] = None, *, pid: Optional[int] = None) -> None:
    """Remove this session's entry on a graceful exit. Optional by design — `live_sessions`
    prunes by pid liveness, so forgetting this (or being killed before it runs) costs nothing."""
    _discard(presence_dir(config_dir) / f"{os.getpid() if pid is None else pid}.json")


def _discard(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


WARNING = (
    "ANOTHER SESSION IS LIVE on agent `{agent}` in this workspace:\n"
    "{others}\n"
    "Both keep working — this is a warning, not a refusal. But they share this agent's "
    "history.jsonl and compact.md, and those appends are NOT locked: two writers can interleave "
    "and a long one can tear. Close one, or accept the risk knowingly."
)


def warning(others: list[LiveSession], *, agent: str) -> str:
    """The loud line. It NAMES the other session, because "something else is running" sends a
    person hunting through `ps` for the thing this function already knew."""
    listed = "\n".join(f"  - {other.describe()}" for other in others)
    return WARNING.format(agent=agent, others=listed)


def summary(others: list[LiveSession]) -> list[dict[str, Any]]:
    """The same facts as data, for `GET /api/health` — so a phone that connected later can still
    see a co-tenant it was never present to be warned about."""
    return [
        {"pid": o.pid, "channel": o.channel, "workspace": o.workspace, "agent": o.agent,
         "session_id": o.session_id, "age_s": round(o.age_s, 1)}
        for o in others
    ]

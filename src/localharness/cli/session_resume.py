"""What a session carries into its rebuild — in memory across a `/plugins` restart, on disk
across a sleep.

`Resume` is plain data on purpose: the same handle that a terminal session passes to its own
rebuild is what `localharness mobile` writes beside the session log when the idle session goes to
sleep, and reads back when the next message wakes it. One format, one builder path (`_start_async`'s
`resume=`), two carriers.

The file holds the model-side conversation verbatim, so it is created owner-only like the session
log that holds the same words. It exists exactly while a thread is asleep: written at sleep, taken
(read and removed) at wake, and put back only when the wake failed before a session was bound —
a stale copy must never be mistaken for the thread, because a thread that woke and moved on has
turns the copy lacks.

The folder it sits in is one a cloned repo can ship and the agent's own write tool can reach
(SECURITY.md), so the file carries conversation TEXT — at the trust level of the session log and
compact.md beside it — and never authority: typed-ahead lines are not in it (a line read from disk
would run as typed, slash commands included), and the permission mode it records is applied by the
session builder only when it is no looser than the configured one. It also records the folder the
server was serving: a thread wakes only where it slept.

The ContentStore is not in the file. A disk resume starts with an empty store, so a tool result the
sleeping session had evicted comes back as its stub and the model re-fetches it if it needs it.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from localharness.core.private_files import write_private_bytes

log = logging.getLogger(__name__)

ASLEEP_FILE = "asleep.json"
"""The sleeping thread, in the agent's `sessions/` folder beside the logs (`*.jsonl`, so no log
scan ever lists it)."""

ASLEEP_FORMAT = 1
SLEEP_ACTION: tuple[str, str] = ("sleep", "")
"""`Resume.action` for a thread that slept, next to `("enable" | "disable", plugin)`."""


@dataclass(frozen=True)
class Resume:
    """What a session carries into its rebuild after `/plugins enable|disable` (in memory) or a
    sleep (from disk) — plain data, so one builder path serves both."""
    action: tuple[str, str]          # ("enable" | "disable", plugin name) or SLEEP_ACTION
    agent_name: str                  # the agent this sitting ran; the rebuild shows no picker
    conversation: tuple[dict, ...]   # AgentLoop's exact model-side messages
    prior_context: str               # the prior-session context folded into the system prompt
    eviction_store: Any              # the ContentStore holding evicted tool-result bodies; None from disk
    queued: tuple[str, ...]          # lines typed ahead, still waiting in the REPL's queue
    gate_mode: str                   # /mode as the person left it
    previous_sitting_id: str         # logged beside the new one; memory needs a fresh id
    failed_check: str = ""           # from the step: its check's first row that failed
    skipped_check: str = ""          # from the step: its first skipped row, when none failed
    step_stopped: bool = False       # the step was stopped (Ctrl-C, Ctrl-D, a refusal or an error)

    @property
    def slept(self) -> bool:
        return self.action == SLEEP_ACTION


@dataclass(frozen=True)
class Restart:
    """`_start_async`'s answer when the REPL ended for `/plugins enable|disable` (start_app runs the
    plugin's step on the plain terminal, then rebuilds from `resume`) or for a sleep (the mobile
    runner has already written `resume` to disk)."""
    action: tuple[str, str]
    resume: Resume


def asleep_path(session_dir: Path) -> Path:
    return session_dir / ASLEEP_FILE


def write_asleep(path: Path, resume: Resume, *, workspace: str) -> None:
    """Persist the thread, owner-only and atomically: a reader finds the old file or the new one,
    never half of either. `workspace` is the folder the server serves (its working directory),
    which is where the thread wakes. ASCII-escaped on purpose: a lone surrogate in a tool result
    must not be the reason a sleep fails and a thread is lost."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "format": ASLEEP_FORMAT,
        "slept_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workspace": workspace,
        "agent_name": resume.agent_name,
        "previous_sitting_id": resume.previous_sitting_id,
        "gate_mode": resume.gate_mode,
        "prior_context": resume.prior_context,
        "conversation": list(resume.conversation),
    }
    tmp = path.with_name(path.name + ".tmp")
    write_private_bytes(tmp, json.dumps(record, ensure_ascii=True).encode("utf-8"))
    os.replace(tmp, path)


def take_asleep(path: Path, *, workspace: str) -> Resume | None:
    """The sleeping thread, removed from disk as it is read — or None: nothing is asleep here.

    A thread that slept serving another folder is left where it is, with a line in the log: it
    wakes in its own folder, not in whichever one started a server next (the agent's folder is
    shared when no project has its own). A file that cannot be read as a thread is moved aside
    (`asleep.json.corrupt`) and logged, and the session starts fresh: a guess at a conversation is
    worse than none, and leaving the file in place would fail every wake the same way."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        resume, slept_in = _parse(json.loads(raw))
    except (ValueError, TypeError, KeyError) as exc:
        aside = path.with_name(path.name + ".corrupt")
        log.warning("the sleeping thread at %s could not be read (%s); moved to %s and starting fresh",
                    path, exc, aside)
        os.replace(path, aside)
        return None
    if slept_in != workspace:
        log.info("the thread asleep at %s belongs to %s, not %s; left in place", path, slept_in, workspace)
        return None
    path.unlink(missing_ok=True)
    return resume


def sleeping_sitting(path: Path) -> str | None:
    """The sitting id of the thread asleep at `path`, or None — never raises: the drawer asks this
    about every delete, and a missing or unreadable file is simply "nothing is asleep"."""
    try:
        _, _ = _parse(json.loads(path.read_bytes()))
        return json.loads(path.read_bytes())["previous_sitting_id"]
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _parse(record: Any) -> tuple[Resume, str]:
    """The handle and the folder it slept in. `queued` is not read even if present: a line from
    disk would run as typed."""
    if not isinstance(record, dict) or record.get("format") != ASLEEP_FORMAT:
        raise ValueError(f"not an asleep record of format {ASLEEP_FORMAT}")
    conversation = record["conversation"]
    if not isinstance(conversation, list) or not all(
            isinstance(m, dict) and isinstance(m.get("role"), str) for m in conversation):
        raise ValueError("conversation is not a list of role-bearing messages")
    fields = {k: record[k] for k in ("agent_name", "previous_sitting_id", "gate_mode", "prior_context")}
    workspace = record["workspace"]
    if not all(isinstance(v, str) for v in (*fields.values(), workspace)):
        raise ValueError("a text field is not text")
    return Resume(action=SLEEP_ACTION, conversation=tuple(conversation), eviction_store=None,
                  queued=(), **fields), workspace

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


def write_asleep(path: Path, resume: Resume) -> None:
    """Persist the thread, owner-only and atomically: a reader finds the old file or the new one,
    never half of either."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "format": ASLEEP_FORMAT,
        "slept_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent_name": resume.agent_name,
        "previous_sitting_id": resume.previous_sitting_id,
        "gate_mode": resume.gate_mode,
        "prior_context": resume.prior_context,
        "queued": list(resume.queued),
        "conversation": list(resume.conversation),
    }
    tmp = path.with_name(path.name + ".tmp")
    write_private_bytes(tmp, json.dumps(record, ensure_ascii=False).encode("utf-8"))
    os.replace(tmp, path)


def take_asleep(path: Path) -> Resume | None:
    """The sleeping thread, removed from disk as it is read — or None: nothing is asleep.

    A file that cannot be read as a thread is moved aside (`asleep.json.corrupt`) and logged, and
    the session starts fresh: a guess at a conversation is worse than none, and leaving the file in
    place would fail every wake the same way."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        resume = _parse(json.loads(raw))
    except (ValueError, TypeError, KeyError) as exc:
        aside = path.with_name(path.name + ".corrupt")
        log.warning("the sleeping thread at %s could not be read (%s); moved to %s and starting fresh",
                    path, exc, aside)
        os.replace(path, aside)
        return None
    path.unlink(missing_ok=True)
    return resume


def _parse(record: Any) -> Resume:
    if not isinstance(record, dict) or record.get("format") != ASLEEP_FORMAT:
        raise ValueError(f"not an asleep record of format {ASLEEP_FORMAT}")
    conversation = record["conversation"]
    if not isinstance(conversation, list) or not all(
            isinstance(m, dict) and isinstance(m.get("role"), str) for m in conversation):
        raise ValueError("conversation is not a list of role-bearing messages")
    queued = record.get("queued", [])
    if not isinstance(queued, list) or not all(isinstance(q, str) for q in queued):
        raise ValueError("queued is not a list of lines")
    fields = {k: record[k] for k in ("agent_name", "previous_sitting_id", "gate_mode", "prior_context")}
    if not all(isinstance(v, str) for v in fields.values()):
        raise ValueError("a text field is not text")
    return Resume(action=SLEEP_ACTION, conversation=tuple(conversation), eviction_store=None,
                  queued=tuple(queued), **fields)

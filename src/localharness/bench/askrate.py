"""The ask-rate report: how often the permission gate would interrupt a human.

Implements PRD §3.6 (`.planning/2026-09-11-zed-acp-and-permission-spine-prd.md`) over the trace
files the harness already writes, and supersedes the one-off replay script
`.planning/research/zed-acp-permission-gate-sim-20260911.py` that produced the PRD §5 numbers.

Two sources, in this order:

* **Bus events.** When a corpus carries `PermissionAsked` / `PermissionResolved`, the report is
  a count, not a model: prompts per session and the decision each one got.
* **Replay.** When it does not — every trace written before v0.14 — the `Action` events are
  replayed through the real `agent.verdict.evaluate` with an empty in-memory grant lookup that
  remembers each key it asks about, which is the ask-once-per-key rule of PRD §3.3. The answer
  is then what the gate WOULD have asked, not what a human saw.

Prompt fatigue is itself a security failure (PRD §3.6): people rubber-stamp a gate that asks
about the wrong things, so the rate is a first-class metric and this report is the regression
tool for any classifier change.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

from localharness.agent.gate_types import (
    UNGRANTABLE_CLASSES,
    GateSettings,
    Grant,
    ToolMeta,
    Verdict,
)
from localharness.agent.verdict import GateContext, derive_boundary, evaluate

# ------------------------------------------------------------------ constants

PROMPT_BUCKETS: tuple[tuple[str, int, Optional[int]], ...] = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-5", 2, 5),
    (">5", 6, None),
)
"""Per-session prompt buckets, as `(label, low, high_inclusive_or_None)`.

The boundaries are PRD §5's own reporting shape ("sessions with 1 / 2-5 / >5 prompts"), kept
identical so a run of this report is comparable against the 384-session baseline table rather
than against a differently-bucketed number.
"""

FIRST_N_SESSIONS_DEFAULT = 50
"""Where the convergence split falls by default (PRD §5: "zero-prompt rate, first 50 sessions"
82.0 % vs "remaining 334" 91.0 %). The split measures whether a workspace's command vocabulary
is settling — the first sessions pay the first-exposure prompts, later ones should not.
"""

TOP_KEYS_REPORTED = 15
"""How many grant keys the report lists, in first-ask order (PRD §5: "first 15, in first-time-ask
order"). The list is the answer to "what did the gate interrupt me about", so it is ordered by
when each key was first asked, never by frequency — a key is asked exactly once."""

PERCENT_SCALE = 100.0
"""Fraction → percent. Named because every SLO in this file is stated as a percentage (PRD §3.6)
and the report prints percentages next to them: a bare ``100.0 * x / y`` reads as a threshold at a
glance, which is exactly the thing it must not be confused with."""

ZERO_PROMPT_SLO_PCT = 90.0
"""PRD §3.6 SLO: ≥90 % of sessions see zero prompts once a workspace is warm. Printed next to the
measured rate so a regression is visible in the report itself."""

MEDIAN_PROMPTS_SLO = 0
"""PRD §3.6 SLO: the median session sees no prompt at all."""

FRESH_WORKSPACE_PROMPTS_SLO = 3
"""PRD §3.6 SLO: the first session in a fresh workspace pays at most three prompts."""

REPLAY_PROVENANCE = "replay"
"""Channel and session id stamped on the in-memory grants the replay hands back. They never reach
the durable store (`config/grants.py`) — the replay is read-only by construction — but `Grant`
requires provenance (PRD §3.3), so it gets provenance that says what it is."""

BUILTIN_TOOL_GROUPS: Mapping[str, str] = {
    "read": "fs.read", "glob": "fs.read", "grep": "fs.read", "load_document": "fs.read",
    "chunk": "fs.read", "tool_result_get": "fs.read",
    "write": "fs.write", "edit": "fs.write",
    "bash_exec": "shell",
    "python_exec": "code", "cruncher_exec": "code",
    "agent": "delegate",
    "web_search": "web", "web_fetch": "web", "web_page_query": "web",
    "memory_search": "memory", "memory_get": "memory", "remember": "memory",
}
"""`ToolSchema.group` for each builtin (A3's taxonomy, PRD §6), read off the builtin schemas.

A trace records a tool's NAME, not its schema, so the replay has to reconstruct the `ToolMeta`
the live gate gets from the registry. Names the live gate knows by name (`write`, `bash_exec`,
`agent`, …) do not depend on this map; it matters for the rest.
"""

MCP_NAME_SEPARATOR = "__"
"""How an MCP tool's name is built: `f"{server_name}__{tool}"` (`tools/mcp.py:63`). A trace name
carrying it is replayed as that server's MCP tool — the honest guess, and the only one available
from a name alone."""

TOOL_CALL_EVENT = ("Action", "tool_call")
"""`(event_type, action_type)` of a dispatched tool call in a trace (`core/events.Action`)."""

PERMISSION_ASKED_EVENT = "PermissionAsked"
PERMISSION_RESOLVED_EVENT = "PermissionResolved"
"""The two v0.14 bus events (PRD §3.6). Their presence switches the report from replay to count."""

DECISION_KINDS: tuple[str, ...] = ("allow_once", "allow_always", "reject_once", "reject_always")
"""The four `DecisionKind` values every channel maps its UI onto (`agent/gate_types.py`)."""


# ----------------------------------------------------------------- trace input

@dataclass(frozen=True)
class TraceSession:
    """One session's bus-event JSONL file, in the order the events were written."""

    session_id: str
    path: Path
    first_timestamp: str
    events: tuple[dict, ...]


def read_events(path: Path) -> tuple[dict, ...]:
    """Every JSON object in a bus-event JSONL file; unparseable lines are skipped."""
    events: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                events.append(record)
    return tuple(events)


def load_sessions(traces_dir: Path) -> list[TraceSession]:
    """Every `*.jsonl` under `traces_dir`, chronological by first event timestamp (PRD §5).

    Chronological order is load-bearing for the replay: the ask-once-per-key rule only converges
    if sessions are replayed in the order they happened, and the first-N split is meaningless
    otherwise. A file with no events at all is still a session — a session that asked nothing is
    exactly what the SLO is about.
    """
    sessions: list[TraceSession] = []
    for path in sorted(traces_dir.rglob("*.jsonl")):
        events = read_events(path)
        stamps = [e.get("timestamp", "") for e in events if e.get("timestamp")]
        sessions.append(
            TraceSession(
                session_id=path.stem,
                path=path,
                first_timestamp=min(stamps) if stamps else "",
                events=events,
            )
        )
    sessions.sort(key=lambda s: (s.first_timestamp == "", s.first_timestamp, s.session_id))
    return sessions


def tool_calls(session: TraceSession) -> Iterable[tuple[str, dict]]:
    """`(tool_name, tool_params)` of every dispatched tool call, in order."""
    event_type, action_type = TOOL_CALL_EVENT
    for event in session.events:
        if event.get("event_type") != event_type or event.get("action_type") != action_type:
            continue
        name = event.get("tool_name")
        if not isinstance(name, str) or not name:
            continue
        params = event.get("tool_params")
        yield name, params if isinstance(params, dict) else {}


def tool_meta_for(tool_name: str) -> ToolMeta:
    """The `ToolMeta` the live gate would hold for this tool, reconstructed from its name.

    MCP tools are `server__tool` (`tools/mcp.py:63`) and are marked destructive by their schema,
    so they replay as the `mcp` class. A name this function does not recognize lands at group
    `other`, which the verdict treats as the ALLOW tier — an honest undercount for a plugin tool
    in an old trace, and the reason the report prints how many names it did not know.
    """
    group = BUILTIN_TOOL_GROUPS.get(tool_name)
    if group is not None:
        return ToolMeta(group=group)
    if MCP_NAME_SEPARATOR in tool_name:
        server = tool_name.split(MCP_NAME_SEPARATOR, 1)[0]
        return ToolMeta(group=f"mcp/{server}", is_mcp=True, mcp_server=server, destructive=True)
    return ToolMeta()


# --------------------------------------------------------------------- report

@dataclass(frozen=True)
class SessionStats:
    session_id: str
    tool_calls: int
    prompts: int
    ungrantable_prompts: int


@dataclass(frozen=True)
class AskRateReport:
    source: str
    """`events` (counted from PermissionAsked/Resolved) or `replay` (derived from Action)."""

    traces_dir: Path
    sessions: tuple[SessionStats, ...]
    first_ask_keys: tuple[str, ...]
    destructive: tuple[tuple[str, int], ...]
    """`(signature, prompts)` for the ungrantable classes, which ask every time."""

    decisions: Mapping[str, int]
    first_n: int
    workspace: Optional[Path] = None
    boundary: Optional[Path] = None
    unknown_tools: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def total_prompts(self) -> int:
        return sum(s.prompts for s in self.sessions)

    @property
    def total_calls(self) -> int:
        return sum(s.tool_calls for s in self.sessions)


def bucket_of(prompts: int) -> str:
    """The PRD §5 bucket a session's prompt count falls in."""
    for label, low, high in PROMPT_BUCKETS:
        if prompts >= low and (high is None or prompts <= high):
            return label
    return PROMPT_BUCKETS[-1][0]


def _zero_rate(sessions: Iterable[SessionStats]) -> tuple[int, int, float]:
    """`(zero-prompt sessions, sessions, percent)` — the SLO's own numerator and denominator."""
    rows = list(sessions)
    zero = sum(1 for s in rows if s.prompts == 0)
    return zero, len(rows), (PERCENT_SCALE * zero / len(rows) if rows else 0.0)


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def has_permission_events(sessions: Iterable[TraceSession]) -> bool:
    """Whether this corpus carries v0.14's gate events (PRD §3.6) or has to be replayed."""
    return any(
        event.get("event_type") == PERMISSION_ASKED_EVENT
        for session in sessions
        for event in session.events
    )


def summarize_from_events(sessions: Iterable[TraceSession], first_n: int, traces_dir: Path) -> AskRateReport:
    """Count what a human actually saw (PRD §3.6).

    Timeouts are deliberately NOT reported as their own number: a timeout resolves as
    `reject_once` (PRD §3.5) and the trace records no flag distinguishing it from a human who
    typed "no", so only `decision` values are counted. Claiming a timeout count here would be
    inventing one.
    """
    stats: list[SessionStats] = []
    decisions: Counter[str] = Counter()
    first_ask_keys: list[str] = []
    seen_keys: set[str] = set()
    destructive: Counter[str] = Counter()

    for session in sessions:
        prompts = 0
        ungrantable = 0
        for event in session.events:
            if event.get("event_type") == PERMISSION_ASKED_EVENT:
                prompts += 1
                klass = event.get("klass") or ""
                key = event.get("key")
                if klass in UNGRANTABLE_CLASSES:
                    ungrantable += 1
                    destructive[f"{klass}: {key or event.get('tool_name') or '?'}"] += 1
                elif isinstance(key, str) and key and key not in seen_keys:
                    seen_keys.add(key)
                    first_ask_keys.append(key)
            elif event.get("event_type") == PERMISSION_RESOLVED_EVENT:
                decision = event.get("decision")
                if isinstance(decision, str):
                    decisions[decision] += 1
        stats.append(
            SessionStats(
                session_id=session.session_id,
                tool_calls=sum(1 for _ in tool_calls(session)),
                prompts=prompts,
                ungrantable_prompts=ungrantable,
            )
        )

    return AskRateReport(
        source="events",
        traces_dir=traces_dir,
        sessions=tuple(stats),
        first_ask_keys=tuple(first_ask_keys),
        destructive=tuple(destructive.most_common()),
        decisions=dict(decisions),
        first_n=first_n,
        notes=(
            "Counted from PermissionAsked / PermissionResolved — what a human was actually asked.",
            "Timeouts are not separable: a timeout resolves as reject_once and carries no flag, "
            "so only decision values are counted.",
        ),
    )


def summarize_from_replay(
    sessions: Iterable[TraceSession],
    *,
    workspace: Path,
    first_n: int,
    traces_dir: Path,
    settings: Optional[GateSettings] = None,
) -> AskRateReport:
    """Replay `Action` events through the real verdict with an empty grant store (PRD §3.6, §5).

    One workspace is assumed for the whole corpus — the boundary derived from `workspace` — and
    one grant set spans every session, in chronological order: that is the ask-once-per-key rule
    of PRD §3.3 measured over a corpus, and it is what makes the first-N split meaningful.

    Honest limits, printed with the report: the DENY tier is not replayed (it needs the config
    that was live at the time), so a call that today's deny patterns would refuse is counted as
    a prompt here; the channel is assumed to have a review surface (the terminal case, PRD §3.5),
    so in-workspace edits do not ask; and tool names the group map does not know land in the
    ALLOW tier.
    """
    gate_settings = settings if settings is not None else GateSettings()
    workspace = Path(workspace).expanduser().resolve()
    boundary = derive_boundary(workspace, None, None, Path.home())

    granted: dict[tuple[str, str], Grant] = {}

    def lookup(_workspace: Path, klass: str, key: str) -> Optional[Grant]:
        """A replay grant answers ONE class (PRD §3.3): keys are unique only within a class."""
        return granted.get((klass, key))

    ctx = GateContext(
        boundary=boundary,
        workspace=workspace,
        grants=lookup,
        mode="guarded",
        can_ask=True,
        has_review_surface=True,
        deny=None,
    )

    stats: list[SessionStats] = []
    first_ask_keys: list[str] = []
    destructive: Counter[str] = Counter()
    unknown: set[str] = set()

    for session in sessions:
        prompts = 0
        ungrantable = 0
        calls = 0
        for tool_name, params in tool_calls(session):
            calls += 1
            if tool_name not in BUILTIN_TOOL_GROUPS and MCP_NAME_SEPARATOR not in tool_name:
                unknown.add(tool_name)
            result = evaluate(tool_name, params, tool_meta_for(tool_name), ctx, gate_settings)
            if result.verdict is not Verdict.ASK or result.request is None:
                continue
            request = result.request
            prompts += 1
            if not request.grantable:
                ungrantable += 1
                destructive[f"{request.klass}: {request.key or tool_name}"] += 1
                continue
            key = request.key
            if isinstance(key, str) and key:
                granted[(request.klass, key)] = Grant(
                    key=key,
                    klass=request.klass,
                    granted_at=REPLAY_PROVENANCE,
                    channel=REPLAY_PROVENANCE,
                    session_id=REPLAY_PROVENANCE,
                    workspace=str(workspace),
                )
                first_ask_keys.append(key)
        stats.append(
            SessionStats(
                session_id=session.session_id,
                tool_calls=calls,
                prompts=prompts,
                ungrantable_prompts=ungrantable,
            )
        )

    notes = [
        "Replayed: these traces predate the gate, so this is what it WOULD have asked, not what "
        "anyone saw.",
        "The DENY tier is not replayed (it needs the config that was live then), so a call today's "
        "deny patterns would refuse still counts as a prompt here.",
        "The channel is assumed to have a review surface (the terminal case), so in-workspace "
        "edits do not ask.",
    ]
    if boundary is None:
        notes.append(
            f"No workspace boundary for {workspace}: it is your home directory or above, so every "
            "write-shaped call asks ungrantably (PRD §3.1)."
        )
    return AskRateReport(
        source="replay",
        traces_dir=traces_dir,
        sessions=tuple(stats),
        first_ask_keys=tuple(first_ask_keys),
        destructive=tuple(destructive.most_common()),
        decisions={},
        first_n=first_n,
        workspace=workspace,
        boundary=boundary,
        unknown_tools=tuple(sorted(unknown)),
        notes=tuple(notes),
    )


def build_report(
    traces_dir: Path,
    *,
    workspace: Optional[Path] = None,
    first_n: int = FIRST_N_SESSIONS_DEFAULT,
    settings: Optional[GateSettings] = None,
) -> AskRateReport:
    """The ask-rate report for a trace directory (PRD §3.6): counted if it can be, replayed if not."""
    sessions = load_sessions(Path(traces_dir))
    if has_permission_events(sessions):
        return summarize_from_events(sessions, first_n=first_n, traces_dir=Path(traces_dir))
    return summarize_from_replay(
        sessions,
        workspace=Path(workspace) if workspace is not None else Path.cwd(),
        first_n=first_n,
        traces_dir=Path(traces_dir),
        settings=settings,
    )


# --------------------------------------------------------------------- render

def render(report: AskRateReport) -> str:
    """The plain-text report (PRD §3.6). Every number here came from the run; none is estimated."""
    lines: list[str] = []
    lines.append(f"ask-rate report — {report.traces_dir}")
    lines.append(f"source: {report.source}")
    if report.source == "replay":
        lines.append(f"workspace: {report.workspace}")
        lines.append(f"boundary:  {report.boundary if report.boundary is not None else '(none)'}")
    lines.append("")

    total_calls = report.total_calls
    total_prompts = report.total_prompts
    share = (PERCENT_SCALE * total_prompts / total_calls) if total_calls else 0.0
    lines.append(f"sessions:   {len(report.sessions)}")
    lines.append(f"tool calls: {total_calls}")
    lines.append(f"prompts:    {total_prompts}  ({share:.1f}% of tool calls)")
    if report.decisions:
        lines.append("decisions:  " + ", ".join(
            f"{kind}={report.decisions.get(kind, 0)}" for kind in DECISION_KINDS
        ))
        other = {k: v for k, v in report.decisions.items() if k not in DECISION_KINDS}
        if other:
            lines.append("            other: " + ", ".join(f"{k}={v}" for k, v in sorted(other.items())))
    lines.append("")

    counts = Counter(bucket_of(s.prompts) for s in report.sessions)
    lines.append("prompts per session")
    for label, _low, _high in PROMPT_BUCKETS:
        lines.append(f"  {label:>3}: {counts.get(label, 0)}")
    zero, total, zero_pct = _zero_rate(report.sessions)
    median = _median([s.prompts for s in report.sessions])
    lines.append(f"  zero-prompt sessions: {zero}/{total} ({zero_pct:.1f}%; SLO ≥{ZERO_PROMPT_SLO_PCT:.0f}%)")
    lines.append(f"  median prompts: {median:g} (SLO {MEDIAN_PROMPTS_SLO})")
    if report.sessions:
        first = report.sessions[0]
        lines.append(
            f"  first session in the corpus: {first.prompts} prompt(s) "
            f"(fresh-workspace SLO ≤{FRESH_WORKSPACE_PROMPTS_SLO})"
        )
    lines.append("")

    head = report.sessions[: report.first_n]
    tail = report.sessions[report.first_n:]
    h_zero, h_total, h_pct = _zero_rate(head)
    t_zero, t_total, t_pct = _zero_rate(tail)
    lines.append(f"convergence — first {report.first_n} sessions vs the rest")
    lines.append(f"  first {report.first_n:>4}: prompts={sum(s.prompts for s in head):5d}  zero {h_zero}/{h_total} ({h_pct:.1f}%)")
    lines.append(f"  remaining {t_total:>4}: prompts={sum(s.prompts for s in tail):5d}  zero {t_zero}/{t_total} ({t_pct:.1f}%)")
    lines.append("")

    lines.append(f"grant keys, first-ask order (top {TOP_KEYS_REPORTED} of {len(report.first_ask_keys)})")
    if report.first_ask_keys:
        for key in report.first_ask_keys[:TOP_KEYS_REPORTED]:
            lines.append(f"  {key}")
    else:
        lines.append("  (none)")
    lines.append("")

    ungrantable_total = sum(count for _sig, count in report.destructive)
    lines.append(f"ungrantable prompts — asked every time ({ungrantable_total})")
    if report.destructive:
        for signature, count in report.destructive:
            lines.append(f"  {count:5d}  {signature}")
    else:
        lines.append("  (none)")

    if report.unknown_tools:
        lines.append("")
        lines.append(
            f"tool names with no known group ({len(report.unknown_tools)}), classified as ALLOW: "
            + ", ".join(report.unknown_tools)
        )

    if report.notes:
        lines.append("")
        lines.append("caveats")
        for note in report.notes:
            lines.append(f"  - {note}")
    return "\n".join(lines)

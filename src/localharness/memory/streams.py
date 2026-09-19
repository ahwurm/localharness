"""Streams, logged whole (memory spec, role 1) — the reader over the EventBus ledgers.

The harness already logs every published event whole (`core/bus.py` appends the full
pydantic dump to `agents/<agent>/sessions/<session_id>.jsonl` before delivery). Those
ledgers ARE the ground-truth stream; this module only READS them, assembling turn
windows on the harness's own execution boundary (TurnStarted → TurnCompleted/Failed)
— the system's structure, never a hand-picked alphabet. Subagent child sessions are
separate files in the same directory and are read as first-class streams.

Incremental digestion: the caller holds per-file byte offsets (the store's
`digest_marks`). A window is emitted only once it is CLOSED (its turn completed,
failed, or a next turn started); the mark advances only past emitted windows, so a
turn still being written is simply re-read on the next pass. The amount digested is
the only clock — nothing here reads wall time into the mechanism.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_CLOSERS = frozenset({"TurnCompleted", "TurnFailed"})


@dataclass
class TurnWindow:
    """One turn's whole flow, rendered to text for the resonance probe."""
    session_id: str
    turn_index: int
    parts: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(p for p in self.parts if p)


def _event_text(ev: dict) -> str:
    """The content that flowed in one event — the texts, not a field-picked summary.

    UserMessage/Action/Observation carry the conversational and tool stream; other
    event types are control structure with no content payload and render nothing.
    """
    et = ev.get("event_type", "")
    if et == "UserMessage":
        return str(ev.get("content") or "")
    if et == "Action":
        atype = ev.get("action_type", "")
        if atype == "llm_response":
            return str(ev.get("content") or "")
        if atype == "tool_call":
            params = ev.get("tool_params")
            ptxt = (json.dumps(params, ensure_ascii=False)
                    if isinstance(params, (dict, list)) else str(params or ""))
            return f"{ev.get('tool_name') or ''} {ptxt}".strip()
        return ""
    if et == "Observation":
        return str(ev.get("output") or "")
    if et == "TurnStarted":
        return str(ev.get("task_summary") or "")
    return ""


def read_new_windows(
    sessions_dir: Path,
    marks: dict[str, int],
    *,
    max_windows: int | None = None,
) -> tuple[list[TurnWindow], dict[str, int]]:
    """New CLOSED turn windows across every session ledger, plus advanced marks.

    `marks` maps ledger filename -> byte offset already digested. Only complete
    lines are parsed; only closed windows are emitted; the returned mark for a file
    points just past the last event of its last closed window (so an in-flight turn
    is re-read whole next time). Files absent from `marks` start at 0.

    `max_windows` bounds one call (a pass's work guardrail): reading stops once the
    bound is reached and the marks advance only past what was emitted — the rest is
    simply the next pass's stream.
    """
    windows: list[TurnWindow] = []
    new_marks = dict(marks)
    if not sessions_dir.is_dir():
        return windows, new_marks

    for path in sorted(sessions_dir.glob("*.jsonl")):
        if max_windows is not None and len(windows) >= max_windows:
            break
        name = path.name
        start = marks.get(name, 0)
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size <= start:
            continue
        try:
            with open(path, "rb") as fh:
                fh.seek(start)
                raw = fh.read()
        except OSError:
            continue

        turn_count_before = marks.get(f"{name}#turns", 0)
        cur: TurnWindow | None = None
        cur_index = turn_count_before
        consumed = 0          # bytes (from `start`) covered by EMITTED windows
        pos = 0               # bytes scanned (complete lines only)
        emitted_here = 0

        for line in raw.split(b"\n")[:-1]:  # last element is a partial line or b""
            line_len = len(line) + 1
            pos += line_len
            text = line.strip()
            if not text:
                continue
            try:
                ev = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(ev, dict):
                continue
            et = ev.get("event_type", "")
            if et == "TurnStarted":
                if cur is not None and cur.parts:
                    # previous turn never emitted a closer — the next turn closes it
                    windows.append(cur)
                    emitted_here += 1
                    consumed = pos - line_len
                    if max_windows is not None and len(windows) >= max_windows:
                        cur = None
                        break
                cur = TurnWindow(session_id=path.stem, turn_index=cur_index)
                cur_index += 1
                t = _event_text(ev)
                if t:
                    cur.parts.append(t)
            elif cur is None:
                continue  # pre-turn breadcrumbs stay unread until a turn opens
            elif et in _CLOSERS:
                if cur.parts:
                    windows.append(cur)
                    emitted_here += 1
                consumed = pos
                cur = None
                if max_windows is not None and len(windows) >= max_windows:
                    break
            else:
                t = _event_text(ev)
                if t:
                    cur.parts.append(t)

        if emitted_here:
            new_marks[name] = start + consumed
            new_marks[f"{name}#turns"] = turn_count_before + emitted_here

    return windows, new_marks

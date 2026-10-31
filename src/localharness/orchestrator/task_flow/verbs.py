"""The task flow's renderer: a LINES row filled with a channel's levers and printed through notice()."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from localharness.channels.base import notice
from localharness.orchestrator.task_flow.record import LINES, Unit, _fill, printed_bound, unit_minutes

Levers = tuple[str, str]


def _put(text: str, fields: dict[str, Any]) -> str:
    step = fields.get("step")
    if step is not None:
        n, m = step if isinstance(step, tuple) else (step, "")
        text = text.replace("N/M", f"{n}/{m}").replace("step N", f"step {n}")
    if fields.get("n") == 1:
        text = text.replace("<n> blockers", "<n> blocker")
    return _fill(text, fields)


def _ranges(steps: tuple[int, ...]) -> str:
    runs: list[list[int]] = []
    for n in sorted(steps):
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return ("steps " if len(steps) > 1 else "step ") + ", ".join(f"{a}-{b}" if b > a else str(a) for a, b in runs)


def render(key: str, levers: Levers, **fields: Any) -> str:
    """The `LINES` row as the finished line a channel prints, or empty for a row only the model reads."""
    line = LINES[key]
    if line.model and not line.commands:
        return ""
    fields = {"stop": levers[0], "cancel": levers[1], **fields}
    state, commands = line.state, list(line.commands)
    if key == "pass" and fields.get("stage") == "plan":
        state += f", {_ranges(fields['reopened'])} reopened" if fields.get("reopened") else ""
        commands += ["plan <change> (revises)", *(["accept decisions (unreviewed)"] * bool(fields.get("decisions_unreviewed")))]
    return notice(fields.get("task", "task"), _put(state, fields), [_put(c, fields) for c in commands], _put(line.tail, fields))


def bound(flow: Any, unit: Unit, sensed: tuple[str, ...] = ()) -> int:
    return printed_bound(unit_minutes(unit, flow.limits, sensed), flow.cap.remaining())


def cap_minutes(flow: Any) -> int:
    return printed_bound(flow.limits.flow_cap_minutes, flow.cap.unspent())


def progress(flow: Any, subject: str, state: str, commands: tuple[str, ...], tail: str, levers: Levers) -> str:
    """The emit sink's line: the levers filled in, and on the turn's first `<stop>` row `typing waits` and the cap said once."""
    spoken: list[str] = []
    if "<stop>" in commands:
        armed = any(c.startswith("cap ") for c in tail.split(" · "))
        if not armed and "typing" not in flow.said:
            spoken.append("typing waits")
        if not armed and "cap" not in flow.said:
            spoken.append(f"cap {cap_minutes(flow)} min")
        flow.said |= {"cap"} if armed else {"typing", "cap"}
    fill = {"stop": levers[0], "cancel": levers[1]}
    return notice(subject, state, [_fill(c, fill) for c in commands], " · ".join(p for p in (*spoken, tail) if p))


@dataclass(frozen=True)
class Reply:
    lines: tuple[str, ...] = ()
    turn: tuple[str, str] | None = None


def task_command(flow: Any, args: str, *, turn: int, running: bool, levers: Levers, phone: bool = False) -> Reply:
    return Reply()

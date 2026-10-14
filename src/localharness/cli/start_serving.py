"""What `start` decides once it knows what serves (spec 20 §8.2, §1 item 3, §9.3).

The selected record comes from the ladder (`model_ops.resolve_serving`); the sensor
(`model_ops.sense_serving`) says which other launch records answer. From those this module decides
attach / attach to another / swap / launch (`decide`), lands an attach (`settle_attach`), tells
whether a launch that is "not answering" is a launch still loading (`liveness_verdict`), waits a
bounded time for the autostart unit (`wait_starting`), closes a boot retry on attach
(`boot_clause`) and runs the swap under the launch lock with a redacting console reporter
(`run_swap`).

It never imports `start_cmd` (which imports it): callers pass the clause text and the printers in.
Nothing here starts, stops or signals a process except through `lifecycle.swap`, and nothing here
runs a service-manager command: the unit's state comes from `unit_runtime_state`, a seam that
answers None in this release (D-14, D-32).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from localharness.cli import model_ops
from localharness.cli.model_ops import (
    RUNG_ACTIVE,
    Serving,
    ServingRecord,
    ServingSense,
)
from localharness.provider import units

# ------------------------------------------------------------------------------------------ strings

RESUME_CLAUSE = "your last /model choice"
ATTACH_LINE = "Serving {name} ({clause})"
OTHER_SERVING_LINE = "serving {serving}; your last choice was {selected} — /model {selected} swaps"
NOT_LAUNCHABLE_LINE = (
    "{name} ({base_url}) is not answering and the harness does not launch it — starting on the primary")
STARTING_LINE = "autostart is starting {name} (pid {pid}) — waiting up to {seconds}s"
STARTING_GAVE_UP_LINE = "autostart's server (pid {pid}) did not answer within {seconds}s — not touched"
BOOT_RETRY_CLAUSE = "retried at boot: {parts} · {undo}: off {keeps}"
SWAP_LINE = "stopping {incumbent}…, starting {target} ({inside}model load can take several minutes)..."
START_TARGET_LINE = "starting {target} ({inside}model load can take several minutes)..."
STARTING_WAIT_SECONDS = 600

_ATTACH_KINDS = ("attach", "attach_other")


def unit_runtime_state() -> units.UnitRuntimeState | None:
    """The autostart unit's live ``InvocationID`` and ``NRestarts``, or None. None in this release: no
    code path of this phase asks a service manager anything (D-32); the autostart phase implements it.
    Tests patch this name."""
    return None


# ----------------------------------------------------------------------------------------- the decision


@dataclass(frozen=True)
class Decision:
    """What `start` does about what serves. ``kind``: ``"attach"`` (the selected record answers),
    ``"attach_other"`` (another record answers and nothing explicit asked otherwise), ``"swap"``
    (another record answers and the selection is explicit) or ``"launch"`` (nothing answers).
    ``record`` is the record the kind acts on (for ``attach_other`` the one that answers);
    ``incumbent`` the answering record a swap stops; ``lines`` the clauses to print (at most one);
    ``rung`` the serving rung of an ``attach`` (None for ``attach_other``)."""
    kind: str
    record: ServingRecord
    incumbent: ServingRecord | None
    lines: tuple[str, ...]
    rung: str | None


def selected_answers(
    probe_ok: bool, selected: ServingRecord, sense: ServingSense, *, shared: bool
) -> bool:
    """Whether the selected record is what answers at its address. ``probe_ok`` is start's own probe
    of that address; when the address is shared with a probed record (``shared``), the sensor must
    also attribute it to the selected record (R-13-28)."""
    if not probe_ok:
        return False
    if not shared:
        return True
    return any(r.prefix == selected.prefix for r in sense.answering)


def decide(
    serving: Serving, *, selected_answers: bool, sense: ServingSense, last_served: str | None
) -> Decision:
    """Pick the action (D-13, D-15, D-16). The two clauses sit on separate branches, so one event
    prints at most one: the resume clause only on an attach to the selected record that the
    ``active_endpoint`` rung named and the ledger's ``last_served`` is not; the attach-to-another
    clause only when ``last_served`` is not the record that answers."""
    record = serving.record
    if selected_answers:
        lines: tuple[str, ...] = ()
        if serving.rung == RUNG_ACTIVE and last_served != record.name:
            lines = (ATTACH_LINE.format(name=record.name, clause=RESUME_CLAUSE),)
        return Decision("attach", record, None, lines, serving.rung)
    answering = next((r for r in sense.answering if r.prefix != record.prefix), None)
    if answering is not None:
        if serving.explicit:
            return Decision("swap", record, answering, (), serving.rung)
        lines = ()
        if last_served != answering.name:
            lines = (OTHER_SERVING_LINE.format(serving=answering.name, selected=record.name),)
        return Decision("attach_other", answering, None, lines, None)
    return Decision("launch", record, None, (), serving.rung)


async def settle_attach(harness: Any, decision: Decision, *, config_dir: Any = None) -> None:
    """Land an attach (R-13-29): ``last_served`` is the answering record's, ``pending_preset`` goes
    only on the ``pending_preset`` rung and no selection is persisted. Both attach kinds go through
    ``persist_landed``; there is no separate attach writer. Raises what the overlay writers raise."""
    if decision.kind not in _ATTACH_KINDS:
        raise ValueError(f"settle_attach is for an attach, not {decision.kind!r}")
    await model_ops.persist_landed(
        harness, decision.record, outcome="attach", rung=decision.rung, config_dir=config_dir)

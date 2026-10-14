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

import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from localharness.cli import model_ops, workspace
from localharness.cli.model_ops import (
    RUNG_ACTIVE,
    Serving,
    ServingRecord,
    ServingSense,
)
from localharness.provider import lifecycle, procfs, speed_stats, units
from localharness.provider.autosense import WindowReport

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


# ------------------------------------------------------------------------------------------ liveness


@dataclass(frozen=True)
class LivenessVerdict:
    """Why nothing answers. ``kind``: ``"none"`` (no launch is live), ``"this"`` (a verified live
    launch of the selected record: still loading), ``"other"`` (a verified live launch of another
    record, ``record``: the swap's incumbent), ``"starting"`` (the autostart unit's own
    ``localharness serve``), ``"foreign"`` (something that is no harness launch holds the selected
    record's port; ``line`` names it). ``pid`` is the process the verdict is about."""
    kind: str
    pid: int | None
    record: ServingRecord | None
    line: str | None


def starting_seconds(unit_state: Any = None) -> int:
    """How long `start` waits for the autostart unit: the unit state's ``timeout_start_seconds`` when
    it holds a positive number, else ``STARTING_WAIT_SECONDS``."""
    bound = getattr(unit_state, "timeout_start_seconds", None)
    if isinstance(bound, (int, float)) and not isinstance(bound, bool) and bound > 0:
        return int(bound)
    return STARTING_WAIT_SECONDS


def _port_of(base_url: str) -> int | None:
    try:
        return urlsplit(base_url).port
    except ValueError:
        return None


def _launchable_others(harness: Any, trust: Any, selected: ServingRecord) -> tuple[ServingRecord, ...]:
    return tuple(
        r for r in model_ops.serving_records(harness)
        if r.prefix != selected.prefix and r.spec is not None and not model_ops.record_withheld(trust, r))


def liveness_verdict(
    harness: Any, trust: Any, selected: ServingRecord, config_dir: Any, *, main_pid: int | None = None
) -> LivenessVerdict:
    """The test behind "nothing answers" (D-18, R-13-30). The pidfile is a claim: a launch counts
    only when its process is verified against a record's own identity, so a planted pidfile can never
    make `start` wait or stop anything.

    A ``localharness serve`` process of the harness unit (its cgroup and ``main_pid``) is "starting".
    Otherwise a verified live launch of ``selected`` is "this", of another launchable record
    "other". With no verified launch, a listener on the selected record's loopback port that
    ``workspace.port_holder`` can name is "foreign"; else "none"."""
    started = procfs.read_launch(config_dir, main_pid=main_pid)
    if started is not None and started.starting:
        return LivenessVerdict(
            "starting", started.pid, selected,
            STARTING_LINE.format(name=workspace.visible_text(selected.name), pid=started.pid,
                                 seconds=starting_seconds(unit_runtime_state())))
    for record, kind in [(selected, "this"), *((r, "other") for r in _launchable_others(harness, trust, selected))]:
        if record.spec is None:
            continue
        identity = lifecycle.record_identity(record.spec)
        if identity is None:
            continue
        launch = procfs.read_launch(config_dir, expect=identity, main_pid=main_pid)
        if launch is not None and launch.verified:
            return LivenessVerdict(kind, launch.pid, record, None)
    port = _port_of(selected.base_url)
    if port is not None and model_ops.is_loopback(selected.base_url) and procfs.listeners(port):
        holder = workspace.port_holder(port)
        if holder is not None:
            line = workspace.incumbent_sentence(
                holder, port=port, form="imperative", served=selected.served, target=selected.name)
            return LivenessVerdict("foreign", holder.pid, None, line)
    return LivenessVerdict("none", None, None, None)


async def wait_starting(probe: Callable[[], Any], *, seconds: float, poll_seconds: float = 3.0) -> bool:
    """Poll ``probe`` (a callable returning a bool, or an awaitable of one) until it is true or
    ``seconds`` have passed; True when it turned true. A probe that raises counts as not yet. It only
    waits: nothing is signalled, so a process that never answers is named by the caller and left
    alone."""
    began = time.monotonic()
    waited = 0.0
    while True:
        try:
            got = probe()
            if inspect.isawaitable(got):
                got = await got
        except Exception:  # noqa: BLE001 — a failed look is "not yet"
            got = False
        if got:
            return True
        if waited >= seconds:
            return False
        step = min(poll_seconds, seconds - waited)
        await asyncio.sleep(step)
        waited = max(waited + step, time.monotonic() - began)


# ---------------------------------------------------------------------------------------- the boot clause


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def boot_clause(
    config_dir: Any, record: ServingRecord, *, autosense: Any, unit_state: Any,
    served_window: int | None, header: Any = None,
) -> str | None:
    """Close a boot retry on attach (D-14, D-32, R-13-31). With a boot marker on disk and the unit's
    runtime state, ``confirm_launch(marker=)`` runs with the verified process, and the clause comes
    back only for a ``cause: death`` marker when the ledger held no ``retry`` for this record before.
    The clause is built from the RECORDED retry, never from the marker's own claims. No marker, no
    unit state (None in this release: the marker is left for the run it belongs to), no launch spec
    or nothing dropped or halved: None."""
    marker = units.read_marker(config_dir)
    if marker is None or unit_state is None or record.spec is None:
        return None
    key = lifecycle.record_key(record.spec)
    ledger = speed_stats.default_speed_stats_path(config_dir)
    before = speed_stats.last_retry(ledger, key)
    process = None
    launch = procfs.read_launch(
        config_dir, expect=lifecycle.record_identity(record.spec),
        main_pid=getattr(unit_state, "main_pid", None))
    if launch is not None and launch.verified:
        got = procfs.read_process(launch.pid)
        process = None if isinstance(got, procfs.ProcRefusal) else got
    live = lifecycle.LiveEndpoint(
        base_url=record.base_url, served_models=[], handle=None,
        window=WindowReport(served=served_window, trained=None, fits=None, free_bytes=None,
                            total_bytes=None, weights_bytes=None, asked=None, mode="not-computed",
                            attached=True))
    lifecycle.confirm_launch(
        config_dir, key, marker=marker, live=live, process=process, unit=unit_state, spec=record.spec,
        autosense=autosense, header=header)
    after = speed_stats.last_retry(ledger, key)
    if before is not None or after is None or marker.cause != "death":
        return None
    parts: list[str] = []
    if "draft-mtp" in after.dropped:
        parts.append("mtp off")
    if _plain_int(marker.planned_context) and _plain_int(after.full_fit) \
            and marker.planned_context < after.full_fit:
        parts.append("-c halved")
    if not parts:
        return None
    return BOOT_RETRY_CLAUSE.format(
        parts=", ".join(parts), undo=f"{record.prefix}.speculative",
        keeps="keeps the full window" if "-c halved" in parts else "keeps drafting pinned off")


# ----------------------------------------------------------------------------------------------- the swap


class ConsoleSwapReporter:
    """What `start` says while a swap runs (a ``lifecycle.SwapReporter``). ``line`` prints a failure
    sentence masked by ``model_ops.redact_for_channel`` through ``print_err`` (called as
    ``print_err(text, markup=False, soft_wrap=True)``, so a rich console's ``print`` is one); the
    other events print nothing: the launch line is the caller's, said through ``on_plan``. ``redact``
    is the same masking, bound, for the caller to hand ``run_swap``."""

    def __init__(self, print_err: Callable[..., Any], harness: Any, key_file_key: Any = None) -> None:
        self._print_err = print_err
        self._harness = harness
        self._key = key_file_key

    def redact(self, text: str) -> str:
        return model_ops.redact_for_channel(text, self._harness, self._key)

    async def freed(self) -> None:
        return None

    async def launched(self, live: Any) -> None:
        return None

    async def restored(self, spec: Any) -> None:
        return None

    async def line(self, text: str) -> None:
        self._print_err(self.redact(text), markup=False, soft_wrap=True)


def swap_launch_line(*, incumbent: str | None, target: str, inside: str) -> str:
    """The launch line of a swap, or of a plain start when nothing is stopped (``incumbent`` None).
    ``inside`` is the plan's tuning text with its trailing separator (or empty)."""
    if incumbent is None:
        return START_TARGET_LINE.format(target=target, inside=inside)
    return SWAP_LINE.format(incumbent=incumbent, target=target, inside=inside)


async def run_swap(
    harness: Any,
    incumbent: ServingRecord | None,
    target: Serving,
    *,
    config_dir: Path,
    autosense: Any,
    reporter: Any,
    redact: Callable[[str], str],
    on_plan: Callable[[Any], None] | None = None,
    on_poll: Callable[[float], None] | None = None,
    on_wait: Callable[[], Any] | None = None,
) -> lifecycle.SwapOutcome:
    """Stop ``incumbent`` (a record whose server runs, or None) and launch ``target`` (the trust-
    resolved ``Serving``), holding the launch lock around the whole swap (D-21). ``redact`` is handed
    to the swap so neither its lines nor its outcome's error carry a secret (D-22); the stop is the
    verified one every strategy does. A record with no launch record is ``ValueError`` from
    ``heavy_for``, before anything is stopped."""
    target_heavy = model_ops.heavy_for(target, harness)
    incumbent_heavy = (
        model_ops.incumbent_heavy((incumbent.spec, incumbent.base_url), harness)
        if incumbent is not None and incumbent.spec is not None else None)
    async with lifecycle.launch_lock(config_dir, on_wait=on_wait):
        return await lifecycle.swap(
            incumbent_heavy, target_heavy, config_dir=config_dir, autosense=autosense,
            reporter=reporter, redact=redact, on_poll=on_poll, on_plan=on_plan)

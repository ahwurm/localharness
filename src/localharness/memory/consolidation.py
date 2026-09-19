"""Dreaming — idle consolidation as the memory spec defines it (2026-09-18, owner-ruled).

Dreaming is when the model reads its own streams deeply. The pass replays the NEW
event-ledger windows since the last digest (the amount digested is the only clock —
no wall-time term exists in the mechanism), and for each window distributes that
moment's one unit of attention over the stored traces by resonance in the model's
own representation space (memory/resonance.py). Traces that resonate gain standing;
everything else loses ground only RELATIVELY — forgetting is the shadow cast by
learning, and an empty pass forgets nothing.

The pass, in order:
  1. embed-backfill — any active fact without a vector (owner edits, restores,
     pre-v10 rows) gets one; a changed embedding model re-embeds everything
     (vectors from different models are not comparable).
  2. digest — new closed turn windows -> resonance shares -> standing, committed
     atomically with the ledger offsets that cover them.
  3. bind — a window whose attention concentrated on several traces (above-uniform
     shares) is a binding observation: the co-fired set becomes / strengthens a
     named group, and the model names it (labels are for human legibility only,
     never mechanism). Skipped silently without an LLM.
  4. fold + settle — staged read-counters fold into the base columns; writer
     paid/lost tallies recompute from the store's own tables (bets settle).
  5. forget — gated by agent.memory.archival.enabled (default OFF): score by the
     one salience currency, read the proven-useful line out of the store, archive
     below it. Nothing is deleted; restore is one verb.

Machine-safety properties carried over unchanged from the previous consolidation:
in-harness only (no daemon), cooperatively cancellable the instant a user turn
arrives (the serial inference gate is never held against the user), every LLM look
budget-capped, every step exception-isolated.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from localharness.config.models import MemoryArchivalConfig, MemoryConsolidationConfig
    from localharness.core.bus import EventBus, SubscriptionHandle
    from localharness.memory.resonance import ResonanceEngine
    from localharness.memory.salience import Salience
    from localharness.memory.sqlite import MemoryStore

log = logging.getLogger(__name__)

_WATERMARK_KEY = "consolidation/last_run"
_EMBED_MODEL_KEY = "resonance/embed_model"


@dataclass
class ConsolidationReport:
    """One dreaming pass's ledger."""
    embedded_backfill: int = 0     # facts given a vector this pass
    reembedded_all: bool = False   # embedding model changed -> full re-embed happened
    windows_digested: int = 0      # new stream windows integrated
    standing_touched: int = 0      # facts that received any share this pass
    folded: int = 0                # staged read-counters folded
    groups_observed: int = 0       # binding observations (new or strengthened)
    groups_named: int = 0          # groups the model named this pass
    writers_settled: int = 0       # writers whose paid/lost tallies were recomputed
    archived: int = 0
    archive_line: float | None = None
    cancelled: bool = False
    duration_s: float = 0.0

    def archive_report_line(self) -> str:
        """The owner-visible one-liner for this pass's forgetting half."""
        from localharness.memory.salience import format_pass_line
        return format_pass_line(self.archived, self.archive_line)


# ---------------------------------------------------------------------------
# Archival verbs — scorer-agnostic plumbing (kept from rung 1; the scorer behind
# them is now the resonance salience in memory/salience.py).
# ---------------------------------------------------------------------------


@dataclass
class ArchiveRun:
    """One archival run's outcome — the shape both the dreaming step and
    `localharness memory archive [--dry-run]` report from (one code path, two triggers)."""
    line: float | None
    active_before: int
    anchors: int
    candidates: list["Salience"] = field(default_factory=list)  # below the line, unpinned
    pinned_below_line: int = 0   # the hard floor's catch (0 under the anchor-floor line)
    moved: int = 0               # rows actually moved (0 on a dry run)
    refused: int = 0             # rows the store declined (unfolded reads, or raced away)
    vacuumed: bool = False
    dry_run: bool = False

    def report_line(self) -> str:
        from localharness.memory.salience import format_pass_line
        return format_pass_line(self.moved, self.line)


async def archive_dormant_facts(
    store: "MemoryStore",
    *,
    dry_run: bool = False,
    now: int | None = None,
    cancel: Optional[asyncio.Event] = None,
) -> ArchiveRun:
    """Score every ACTIVE fact, read the line out of the store, and move what sits below it.

    THE one implementation — the idle dreaming step and the CLI verb both call this,
    so a dry-run's preview and a real pass can never diverge. Superseded rows are not
    touched; nothing is deleted, ever.
    """
    from localharness.memory.salience import (
        archive_line, score_facts, select_archivable, vacuum_warranted,
    )
    from localharness.memory.sqlite import ARCHIVE_SURFACE_FLOOR_LINE, _row_to_fact

    assert store._db is not None
    async with store._db.execute(
        f"SELECT {store._FACT_COLS} FROM facts WHERE agent_id = ? AND status = 'active'",
        (store._agent_id,),
    ) as cur:
        facts = [_row_to_fact(r) for r in await cur.fetchall()]
    # Rows read since the last fold carry evidence the Fact projection cannot show (the
    # staged counters are not part of it). They anchor the line like any other recall —
    # see score_facts(also_anchor=...). Empty inside a pass; not empty from the CLI.
    async with store._db.execute(
        "SELECT id FROM facts WHERE agent_id = ? AND status = 'active' "
        "AND access_count_staged > 0",
        (store._agent_id,),
    ) as cur:
        staged_reads = [r[0] for r in await cur.fetchall()]

    scored = score_facts(facts, also_anchor=staged_reads)
    line = archive_line(scored)
    candidates = select_archivable(scored, line)
    run = ArchiveRun(
        line=line, active_before=len(scored),
        anchors=sum(1 for s in scored if s.anchor),
        candidates=candidates,
        pinned_below_line=sum(1 for s in scored if line is not None and s.s < line and s.pinned),
        dry_run=dry_run,
    )
    if dry_run or not candidates:
        return run

    for cand in candidates:
        if cancel is not None and cancel.is_set():
            break   # a user turn is waiting; moves already committed stand
        moved = await store.archive_fact(
            cand.fact_id, surface=ARCHIVE_SURFACE_FLOOR_LINE,
            s_at_archive=cand.s, line_at_archive=line,
        )
        if moved:
            run.moved += 1
        else:
            run.refused += 1
    if vacuum_warranted(run.moved, run.active_before):
        await store.vacuum()
        run.vacuumed = True
    return run


# ---------------------------------------------------------------------------
# List-driven archival — the harness takes a LIST of fact ids and stays out of the
# judging; every safety rail is re-checked HERE, at execution time.
# ---------------------------------------------------------------------------

# Why a row was not moved. Named, because these strings are the report the owner reads.
SKIP_UNKNOWN = "no such fact in this store"
SKIP_NOT_ACTIVE = "not active (already superseded or archived)"
SKIP_OWNER_TOUCHED = "your own hand (pinned)"
SKIP_RECALLED = "recalled before (anchor)"
SKIP_UNFOLDED_READS = "read since the last fold"
SKIP_REFUSED_AT_MOVE = "changed underfoot during the run"
SKIP_UNPARSEABLE = "unparseable row"

# The list's field separator and comment marker. Trailing fields are IGNORED by design:
# the list is produced by an external scorer whose columns WILL change, and a consumer that
# broke on an extra column would turn every scorer change into a harness change.
_LIST_SEPARATOR = "|"
_LIST_COMMENT = "#"


@dataclass(frozen=True)
class SkippedFact:
    """One row the run declined, with the reason in the owner's words. `fact_id` is None
    for a row that never parsed into an id (then `line_no` and `raw` locate it in the file)."""
    reason: str
    fact_id: int | None = None
    key: str = ""
    line_no: int | None = None
    raw: str = ""


def parse_archive_list(text: str) -> tuple[list[int], list[SkippedFact]]:
    """Parse an external condemned-id list: one row per fact, `|`-separated, FIRST field is
    the fact id. Blank lines and `#` comments are skipped silently; trailing fields are
    ignored; ids repeat harmlessly (first occurrence wins, file order preserved).

    Returns (ids, unparseable rows). A garbage row is REPORTED, never fatal — one bad line
    in a 10k-row list must not cost the other 9,999 their run, and it must not pass in
    silence either.
    """
    ids: list[int] = []
    seen: set[int] = set()
    bad: list[SkippedFact] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        row = raw.strip()
        if not row or row.startswith(_LIST_COMMENT):
            continue
        head = row.split(_LIST_SEPARATOR, 1)[0].strip()
        try:
            fact_id = int(head)
        except ValueError:
            bad.append(SkippedFact(reason=SKIP_UNPARSEABLE, line_no=line_no, raw=row))
            continue
        if fact_id not in seen:
            seen.add(fact_id)
            ids.append(fact_id)
    return ids, bad


@dataclass
class ArchiveListRun:
    """One list-driven run's outcome — same reporting shape as ArchiveRun, minus the line
    (there is none: the list is the authority, not a threshold this harness computed)."""
    listed: int = 0
    active_before: int = 0
    candidates: list["Salience"] = field(default_factory=list)
    skipped: list[SkippedFact] = field(default_factory=list)
    moved: int = 0
    vacuumed: bool = False
    dry_run: bool = False


async def archive_listed_facts(
    store: "MemoryStore",
    ids: list[int],
    *,
    dry_run: bool = False,
    now: int | None = None,
    unparseable: list[SkippedFact] | None = None,
) -> ArchiveListRun:
    """Archive exactly the listed ids — through every rail, re-checked at execution time.

    The rails do not care what the list says. A listed id is skipped (and REPORTED, never
    raised) when it names no row here, names a row that is not active, names a fact the
    owner touched, names a fact that was ever recalled, or names a fact read since the last
    fold. The store's own move verb then re-checks the last two on its own, so a fact that
    becomes live between the scan and the move is still refused.
    """
    from localharness.memory.salience import is_pinned, score_fact, vacuum_warranted
    from localharness.memory.sqlite import ARCHIVE_SURFACE_CONSENSUS_LIST, _row_to_fact

    assert store._db is not None
    run = ArchiveListRun(listed=len(ids), dry_run=dry_run,
                         skipped=list(unparseable or []))
    async with store._db.execute(
        "SELECT COUNT(*) FROM facts WHERE agent_id = ? AND status = 'active'",
        (store._agent_id,),
    ) as cur:
        (run.active_before,) = await cur.fetchone()

    for fact_id in ids:
        async with store._db.execute(
            f"SELECT {store._FACT_COLS}, access_count_staged FROM facts "
            "WHERE agent_id = ? AND id = ?",
            (store._agent_id, fact_id),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            run.skipped.append(SkippedFact(reason=SKIP_UNKNOWN, fact_id=fact_id))
            continue
        cells = tuple(row)
        fact, staged = _row_to_fact(cells[:-1]), cells[-1]
        if fact.status != "active":
            run.skipped.append(
                SkippedFact(reason=SKIP_NOT_ACTIVE, fact_id=fact_id, key=fact.key))
            continue
        if is_pinned(fact):
            run.skipped.append(
                SkippedFact(reason=SKIP_OWNER_TOUCHED, fact_id=fact_id, key=fact.key))
            continue
        if (fact.access_count or 0) > 0:
            run.skipped.append(
                SkippedFact(reason=SKIP_RECALLED, fact_id=fact_id, key=fact.key))
            continue
        if (staged or 0) > 0:
            run.skipped.append(
                SkippedFact(reason=SKIP_UNFOLDED_READS, fact_id=fact_id, key=fact.key))
            continue

        scored = score_fact(fact)
        run.candidates.append(scored)
        if dry_run:
            continue
        if await store.archive_fact(fact_id, surface=ARCHIVE_SURFACE_CONSENSUS_LIST,
                                    s_at_archive=scored.s, line_at_archive=None):
            run.moved += 1
        else:
            run.candidates.pop()
            run.skipped.append(
                SkippedFact(reason=SKIP_REFUSED_AT_MOVE, fact_id=fact_id, key=fact.key))
    if not dry_run and vacuum_warranted(run.moved, run.active_before):
        await store.vacuum()
        run.vacuumed = True
    return run


# ---------------------------------------------------------------------------
# The dreaming pass
# ---------------------------------------------------------------------------


class ConsolidationPass:
    """One dreaming run. Construct fresh per run; `cancel()` at any time — the pass
    exits at its next per-window / per-step check and everything already committed
    stands."""

    def __init__(
        self,
        store: "MemoryStore",
        cfg: "MemoryConsolidationConfig",
        *,
        engine: Optional["ResonanceEngine"] = None,
        llm: Any = None,
        archival: Optional["MemoryArchivalConfig"] = None,
    ) -> None:
        self._store = store
        self._cfg = cfg
        self._engine = engine
        self._llm = llm
        self._archival = archival
        self._cancel = asyncio.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    async def run(self) -> ConsolidationReport:
        report = ConsolidationReport()
        started = time.monotonic()
        steps: list[Callable[[ConsolidationReport], Awaitable[None]]] = [
            self._step_embed_backfill,
            self._step_digest,
            self._step_fold_and_settle,
            self._step_forget,
        ]
        for step in steps:
            if self.cancelled:
                break
            try:
                await step(report)
            except Exception:
                log.exception("dreaming step %s failed (isolated)", step.__name__)
        await _set_meta(self._store, _WATERMARK_KEY, str(int(time.time())))
        report.cancelled = self.cancelled
        report.duration_s = time.monotonic() - started
        return report

    async def _embed(self, texts: list[str]):
        assert self._engine is not None
        return await asyncio.to_thread(self._engine.embed_docs, texts)

    async def _step_embed_backfill(self, report: ConsolidationReport) -> None:
        """Every active fact gets a vector; a changed embedding model re-embeds ALL
        (vectors from different models are not comparable — the store records which
        model embedded it). No engine wired -> the step reports and returns; nothing
        pretends to embed."""
        from localharness.memory import resonance as _res

        if self._engine is None:
            log.warning("dreaming: no resonance engine wired — embed/digest skipped")
            return
        stored_model = await _get_meta(self._store, _EMBED_MODEL_KEY)
        if stored_model is not None and stored_model != self._engine.model_name:
            ids = await self._store.all_embedded_fact_ids()
            for fid in ids:
                await self._store.set_fact_embedding(fid, None)
            report.reembedded_all = True
            log.info("dreaming: embedding model changed (%s -> %s); re-embedding %d facts",
                     stored_model, self._engine.model_name, len(ids))
        while not self.cancelled:
            batch = await self._store.facts_missing_embedding(limit=64)
            if not batch:
                break
            vecs = await self._embed([f"{f.key}: {f.value}" for f in batch])
            for f, v in zip(batch, vecs):
                await self._store.set_fact_embedding(f.id, _res.pack(v))
                report.embedded_backfill += 1
        await _set_meta(self._store, _EMBED_MODEL_KEY, self._engine.model_name)

    async def _step_digest(self, report: ConsolidationReport) -> None:
        """Replay the new stream and let the present meet the past: each window's one
        unit of attention distributes over the stored traces by resonance share.
        Standing deltas and ledger marks commit atomically per pass. Windows whose
        attention concentrated on >=2 traces become binding observations."""
        from localharness.memory import resonance as _res
        from localharness.memory.streams import read_new_windows

        if self._engine is None:
            return
        sessions_dir = self._store._agent_dir / "sessions"
        marks = await self._store.get_digest_marks()
        windows, new_marks = read_new_windows(
            sessions_dir, marks, max_windows=self._cfg.iteration_cap
        )
        if not windows:
            return
        id_blobs = await self._store.active_embedded()
        deltas: dict[int, float] = {}
        bindings: list[list[int]] = []
        digested = 0
        for w in windows:
            if self.cancelled:
                break
            text = w.text()
            if not text.strip():
                digested += 1
                continue
            vec = (await self._embed([text]))[0]
            shares = _res.shares(vec, id_blobs)
            for fid, share in shares.items():
                deltas[fid] = deltas.get(fid, 0.0) + share
            if len(id_blobs) >= 2 and shares:
                # Above-uniform share = this moment's attention CONCENTRATED here.
                uniform = 1.0 / len(id_blobs)
                cofired = sorted(fid for fid, sh in shares.items() if sh > uniform)
                if len(cofired) >= 2:
                    bindings.append(cofired)
            digested += 1
        if self.cancelled and digested < len(windows):
            # Marks may only cover what was integrated: re-derive them for the digested
            # prefix by re-reading with the smaller bound (cheap; file IO only).
            _, new_marks = read_new_windows(sessions_dir, marks, max_windows=digested)
        await self._store.apply_digest(deltas, new_marks)
        report.windows_digested = digested
        report.standing_touched = len(deltas)
        await self._step_bind(report, bindings)

    async def _step_bind(self, report: ConsolidationReport, bindings: list[list[int]]) -> None:
        """Binding observations become named groups. The model names them (one
        budget-capped, cancellable look per new group); an unnamed group waits for a
        later pass — labels are legibility, never mechanism, so nothing blocks on them."""
        from localharness.memory.idle_llm import complete_cancellable

        for members in bindings:
            if self.cancelled:
                return
            gid = await self._store.upsert_group(members)
            report.groups_observed += 1
            if self._llm is None:
                continue
            groups = {g["id"]: g for g in await self._store.list_groups()}
            g = groups.get(gid)
            if g is None or g["label"]:
                continue
            facts = await self._store.get_facts_by_ids(members)
            listing = "\n".join(f"- {f.key}: {f.value}" for f in facts)
            answer = await complete_cancellable(
                self._llm,
                "These memories fired together during one experience:\n"
                f"{listing}\n"
                "If they form one coherent topic, answer with a short name for it "
                "(2-4 words). If they do not, answer exactly NONE.",
                self._cancel,
            )
            label = (answer or "").strip().splitlines()[0].strip() if answer else ""
            if label and label.upper() != "NONE":
                await self._store.set_group_label(gid, label)
                report.groups_named += 1

    async def _step_fold_and_settle(self, report: ConsolidationReport) -> None:
        """Fold staged read-counters (the one moment reads may reorder the injected
        block), then settle the bet ledger: paid/lost recomputed from the store's own
        tables — uniform statistics, no incremental drift."""
        report.folded = await self._store.fold_staged_access()
        report.writers_settled = await self._store.settle_writer_outcomes()

    async def _step_forget(self, report: ConsolidationReport) -> None:
        """The forgetting half — gated OFF by default (agent.memory.archival.enabled);
        the owner flips it after watching a dry run. One implementation shared with the
        CLI verb."""
        if self._archival is None or not getattr(self._archival, "enabled", False):
            return
        run = await archive_dormant_facts(self._store, cancel=self._cancel)
        report.archived = run.moved
        report.archive_line = run.line


# ---------------------------------------------------------------------------
# Scheduler — trigger + cancellation owner (unchanged machine-safety shape)
# ---------------------------------------------------------------------------


class ConsolidationScheduler:
    """The trigger + cancellation owner. No daemon exists on this box — the scheduler
    lives inside the harness process: a staleness check at session start plus an
    in-session idle timer; any user activity cancels a running pass instantly and
    resets the timer."""

    def __init__(
        self,
        store: "MemoryStore",
        bus: "EventBus",
        agent_id: str,
        cfg: "MemoryConsolidationConfig",
        *,
        engine: Optional["ResonanceEngine"] = None,
        llm: Any = None,
        archival: Optional["MemoryArchivalConfig"] = None,
    ) -> None:
        self._store = store
        self._bus = bus
        self._agent_id = agent_id
        self._cfg = cfg
        self._engine = engine
        self._llm = llm
        self._archival = archival   # agent.memory.archival — None reads as OFF
        self._handles: list["SubscriptionHandle"] = []
        self._running: Optional[ConsolidationPass] = None
        self._run_task: Optional[asyncio.Task] = None
        self._timer_task: Optional[asyncio.Task] = None
        self._last_activity = time.monotonic()
        self._turn_in_flight = False  # #78: an agent turn is mid-flight (defers/cancels passes)
        self.last_report: Optional[ConsolidationReport] = None

    async def start(self) -> None:
        if not self._cfg.enabled:
            return
        from localharness.core.events import UserMessage, TurnStarted, TurnCompleted, TurnFailed
        self._handles.append(
            self._bus.subscribe(UserMessage, self._on_user_activity)
        )
        # #78: track the agent turn in flight. TurnStarted -> (TurnCompleted|TurnFailed) is a
        # guaranteed 1:1 bracket per turn. Filter to THIS agent_id: nested subagent turns ride
        # the same bus with their own agent_id inside the root turn's bracket.
        self._handles.append(
            self._bus.subscribe(TurnStarted, self._on_turn_started, agent_id=self._agent_id)
        )
        self._handles.append(
            self._bus.subscribe(TurnCompleted, self._on_turn_ended, agent_id=self._agent_id)
        )
        self._handles.append(
            self._bus.subscribe(TurnFailed, self._on_turn_ended, agent_id=self._agent_id)
        )
        if await self.should_run():
            self.launch()
        self._timer_task = asyncio.create_task(self._idle_timer_loop())

    async def stop(self) -> None:
        for h in self._handles:
            self._bus.unsubscribe(h)
        self._handles.clear()
        if self._timer_task is not None:
            self._timer_task.cancel()
            self._timer_task = None
        self.cancel_running()
        if self._run_task is not None:
            try:
                # Bounded: a pathologically slow step must not stall process shutdown;
                # the cancel above makes steps exit at their next check.
                await asyncio.wait_for(self._run_task, timeout=15.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                self._run_task.cancel()

    async def _on_user_activity(self, event: Any) -> None:
        """A user turn arrived: the box is NOT idle. Cancel any in-flight pass (it will
        release the inference gate) and reset the idle clock."""
        self._last_activity = time.monotonic()
        self.cancel_running()

    async def _on_turn_started(self, event: Any) -> None:
        """An agent turn is mid-flight (#78): defer STARTING a pass (via the launch()
        guard) and cancel any RUNNING one. Does NOT reset the idle clock: idle is
        measured from turn END so a long turn can't look idle mid-flight."""
        self._turn_in_flight = True
        self.cancel_running()

    async def _on_turn_ended(self, event: Any) -> None:
        """Turn finished: clear the in-flight gate and reset the idle clock so
        idle_minutes counts from turn END (#78)."""
        self._turn_in_flight = False
        self._last_activity = time.monotonic()

    def cancel_running(self) -> None:
        if self._running is not None:
            self._running.cancel()

    def launch(self) -> None:
        """Fire a pass as a background task (idempotent while one is running; deferred
        while an agent turn is in flight — #78). Bench/eval never call this (they run
        ConsolidationPass directly)."""
        if self._turn_in_flight:
            return
        if self._run_task is not None and not self._run_task.done():
            return
        self._running = ConsolidationPass(
            self._store, self._cfg, engine=self._engine, llm=self._llm,
            archival=self._archival,
        )
        self._run_task = asyncio.create_task(self._run_and_record())

    async def _emit_status(self, *, started: bool) -> None:
        """Fire-and-forget dreaming-dot signal for the interactive REPL (#20). A bus
        fault is swallowed — the pass must never break on a status dot."""
        from localharness.core.events import ConsolidationFinished, ConsolidationStarted
        event = (ConsolidationStarted if started else ConsolidationFinished)(agent_id=self._agent_id)
        try:
            await self._bus.publish(event)
        except Exception:
            log.debug("consolidation status emit failed (non-fatal)", exc_info=True)

    async def _run_and_record(self) -> None:
        await self._emit_status(started=True)
        try:
            assert self._running is not None
            self.last_report = await self._running.run()
            if self.last_report.cancelled:
                log.info("dreaming pass cancelled by user activity")
            else:
                log.info(
                    "dreaming: embedded=%d windows=%d touched=%d folded=%d groups=%d/%d settled=%d%s",
                    self.last_report.embedded_backfill, self.last_report.windows_digested,
                    self.last_report.standing_touched, self.last_report.folded,
                    self.last_report.groups_named, self.last_report.groups_observed,
                    self.last_report.writers_settled,
                    # The forgetting half is owner-visible or it did not happen.
                    f" — {self.last_report.archive_report_line()}"
                    if self.last_report.archived else "",
                )
        except Exception:
            log.exception("dreaming pass crashed (non-fatal)")
        finally:
            self._running = None
            await self._emit_status(started=False)

    async def should_run(self) -> bool:
        """Session-start staleness: run when the watermark is old AND there is work."""
        if not self._cfg.enabled:
            return False
        raw = await _get_meta(self._store, _WATERMARK_KEY)
        stale = True
        if raw is not None:
            stale = (time.time() - int(raw)) > self._cfg.staleness_hours * 3600
        if not stale:
            return False
        return await self._has_work()

    async def _has_work(self) -> bool:
        """Work = undigested stream (ledger bytes past the marks), a fact without a
        vector, or staged reads awaiting the fold. Cheap: two SELECTs and a stat walk."""
        assert self._store._db is not None
        async with self._store._db.execute(
            "SELECT EXISTS(SELECT 1 FROM facts WHERE agent_id = :a AND access_count_staged > 0), "
            "EXISTS(SELECT 1 FROM facts WHERE agent_id = :a AND status = 'active' "
            "       AND embedding IS NULL)",
            {"a": self._agent_id},
        ) as cur:
            staged, unembedded = await cur.fetchone()
        if staged or unembedded:
            return True
        marks = await self._store.get_digest_marks()
        sessions_dir = self._store._agent_dir / "sessions"
        if sessions_dir.is_dir():
            for path in sessions_dir.glob("*.jsonl"):
                try:
                    if path.stat().st_size > marks.get(path.name, 0):
                        return True
                except OSError:
                    continue
        return False

    async def _idle_timer_loop(self) -> None:
        """In-session idle trigger: no user activity for idle_minutes → launch a pass.
        The body is exception-guarded: one transient _has_work error must not silently
        kill idle dreaming for the rest of the session."""
        interval = max(5.0, self._cfg.idle_minutes * 60 / 4)
        fired_for_this_idle = False
        while True:
            await asyncio.sleep(interval)
            try:
                idle_s = time.monotonic() - self._last_activity
                if idle_s >= self._cfg.idle_minutes * 60:
                    if not fired_for_this_idle:
                        fired_for_this_idle = True
                        if await self._has_work():
                            self.launch()
                else:
                    fired_for_this_idle = False
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("idle-timer check failed (non-fatal; timer continues)")


# ---------------------------------------------------------------------------
# Tiny agent-scoped KV (watermark + embed-model identity). The schema-v10 migration
# creates the same table; the idempotent DDL here keeps pre-v10 callers working.
# ---------------------------------------------------------------------------

async def _ensure_meta(store: "MemoryStore") -> None:
    assert store._db is not None
    await store._db.execute(
        "CREATE TABLE IF NOT EXISTS meta ("
        "agent_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, "
        "PRIMARY KEY (agent_id, key)) WITHOUT ROWID"
    )


async def _get_meta(store: "MemoryStore", key: str) -> str | None:
    await _ensure_meta(store)
    assert store._db is not None
    async with store._db.execute(
        "SELECT value FROM meta WHERE agent_id = ? AND key = ?", (store._agent_id, key)
    ) as cur:
        row = await cur.fetchone()
    return row[0] if row else None


async def _set_meta(store: "MemoryStore", key: str, value: str) -> None:
    await _ensure_meta(store)
    assert store._db is not None
    await store._db.execute(
        "INSERT INTO meta (agent_id, key, value) VALUES (?, ?, ?) "
        "ON CONFLICT(agent_id, key) DO UPDATE SET value = excluded.value",
        (store._agent_id, key, value),
    )
    await store._db.commit()

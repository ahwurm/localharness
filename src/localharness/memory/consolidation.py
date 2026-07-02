"""Idle-time consolidation (CONS-01..06) — the CLS slow-integrate half of the memory system.

Cognitive frame: the hippocampus captures fast, sparse episodes during the day (here: the
event bus writing sessions + the WriteGate's candidates); the cortex integrates them into
durable knowledge preferentially during idle periods, by REPLAY. The reasons the brain
does this split are this box's reasons: integration is expensive (9.5 tok/s) and must
never block behavior.

Owner rulings + critic fixes baked in structurally:
- IN-HARNESS feature, default-on / config-off (`agent.memory.consolidation.*` registry
  axes) — NOT an OS cron job; no daemon assumed. v1 triggers: a session-start staleness
  check + an in-session idle timer (CONS-01; critic BLOCKER 4).
- COOPERATIVELY CANCELLABLE (CONS-02; critic BLOCKER 2): the serial `_inference_gate`
  (provider/client.py) is non-preemptive and held to the last token — built after a real
  dual-process box freeze — so any user turn cancels the in-flight pass (including its
  LLM generation task) instead of making the user wait behind it.
- SOFT capacity cap (CONS-05; critic BLOCKER 3): admission NEVER blocks inline; this pass
  trims the active tier back under the bound — merge/dedup/demote, silent deletion never.
- Guardrails non-optional (CONS-04): hard iteration cap, dedup-before-generate,
  verify-against-leaf. LLM-judged deletion does not exist here at all.
- Quality proxies (CONS-06; critic MAJOR 2): promote-then-superseded churn rate + a
  promotion-sample hook (the dispatch layer can pipe samples to Discord for passive
  owner spot-check) — fire counters alone can't see silent corruption.

The deterministic core (fold, candidate promotion on cross-episode recurrence, decay,
cap trim, proxies) runs with NO model at all. The LLM session-replay claim-extractor is
a seam (`llm=` param): implemented + cancellable + guarded, wired OFF by default until
its output quality is iterated live (owner: don't pause for slow items).
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from localharness.config.models import MemoryConsolidationConfig
    from localharness.core.bus import EventBus, SubscriptionHandle
    from localharness.memory.sqlite import MemoryStore

log = logging.getLogger(__name__)

_WATERMARK_KEY = "consolidation/last_run"
_DEMOTED_RS = 0.15   # below the 0.2 index gate: out of the injected block, still searchable
_PROMOTED_CONFIDENCE = 0.8  # above the 0.7 injection threshold: promoted facts surface


@dataclass
class ConsolidationReport:
    folded: int = 0
    promoted: int = 0
    decayed: int = 0
    demoted: int = 0
    replayed_claims: int = 0
    active_over_cap: int = 0
    churn_rate: float = 0.0
    cancelled: bool = False
    duration_s: float = 0.0
    promoted_keys: list[str] = field(default_factory=list)


class ConsolidationPass:
    """One consolidation run. Construct fresh per run; `cancel()` at any time."""

    def __init__(
        self,
        store: "MemoryStore",
        cfg: "MemoryConsolidationConfig",
        *,
        llm: Any = None,
        on_promotion_sample: Optional[Callable[[list[Any]], Awaitable[None] | None]] = None,
    ) -> None:
        self._store = store
        self._cfg = cfg
        self._llm = llm
        self._on_promotion_sample = on_promotion_sample
        self._cancel = asyncio.Event()
        # Same-run promotion grace (critic BLOCKER 2): cap-trim must never demote what
        # this very pass just promoted (fresh records have zero access history — the
        # lowest slow-score in any mature store, i.e. first trim victims).
        self._promoted_ids_this_run: set[int] = set()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # ------------------------------------------------------------------
    # The run
    # ------------------------------------------------------------------

    async def run(self) -> ConsolidationReport:
        t0 = time.monotonic()
        report = ConsolidationReport()
        steps = (
            self._step_fold,
            self._step_promote_recurring,
            self._step_replay_llm,
            self._step_decay,
            self._step_cap_trim,
            self._step_proxies,
        )
        for step in steps:
            if self._cancel.is_set():
                report.cancelled = True
                break
            try:
                await step(report)
            except asyncio.CancelledError:
                report.cancelled = True
                break
            except Exception:
                log.exception("consolidation step %s failed (non-fatal)", step.__name__)
        if not report.cancelled:
            await self._set_watermark(int(time.time()))
        report.duration_s = time.monotonic() - t0
        return report

    # -- 1. fold staged read-counters (RANK-04 boundary) -----------------

    async def _step_fold(self, report: ConsolidationReport) -> None:
        report.folded = await self._store.fold_staged_access()

    # -- 2. promote gate candidates that RECUR across episodes (CONS-03) --

    async def _step_promote_recurring(self, report: ConsolidationReport) -> None:
        """Cross-episode recurrence is the promotion warrant (Tse 2007 schema-consistent
        fast track, translated): the same lesson captured from ≥2 distinct sessions
        graduates from candidate (below the injection threshold) to durable fact
        (above it), linked `derived_from` its source candidates."""
        # Direct candidate query (Phase-31 critic M3): the relevance-ranked query_facts
        # crowded backlog candidates out of the iteration_cap window as the store
        # accumulated well-used facts — the guardrail silently became a starvation
        # bound. Oldest-first over the pending set is what "bounded work on candidates"
        # actually means.
        from localharness.memory.sqlite import _row_to_fact

        assert self._store._db is not None
        async with self._store._db.execute(
            f"SELECT {self._store._FACT_COLS} FROM facts "
            "WHERE agent_id = ? AND status = 'active' "
            "AND tags LIKE '%\"pending_consolidation\"%' "
            "ORDER BY created_at ASC, id ASC LIMIT ?",
            (self._store._agent_id, self._cfg.iteration_cap),
        ) as cur:
            candidates = [_row_to_fact(r) for r in await cur.fetchall()]
        groups: dict[tuple[str, str], list] = {}
        for c in candidates:
            parts = c.key.split("/")  # gate/<tier>/<tool>[/<hash>]
            if len(parts) < 3 or parts[0] != "gate":
                continue
            groups.setdefault((parts[1], parts[2]), []).append(c)

        for (tier, tool), members in groups.items():
            if self._cancel.is_set():
                return
            provenances = {m.provenance for m in members if m.provenance}
            key = f"learned/{tool}/{tier}"
            existing = await self._store.get_fact(key)
            # Promotion warrant: recurrence (≥2 distinct episodes), OR an existing
            # promoted record (a single fresh episode is schema-consistent evidence —
            # Tse 2007 fast track), OR a SALIENT flag (Phase-31 critic M1: the APPROACH
            # §C "or carries a salience flag" route — the gate marks stuck-recoveries
            # salient; one occurrence is warrant enough). Novelty candidates carry
            # neither and by design never promote (telemetry tier).
            salient = any("salient" in m.tags for m in members)
            if existing is None and len(provenances) < 2 and not salient:
                continue
            # verify-against-leaf (CONS-04): the merged record is composed ONLY of
            # verbatim candidate bodies (and the prior record's own bullets).
            new_bodies = sorted({m.value for m in members if m.value})
            if not new_bodies:
                continue
            prev_n = 0
            prior_bullets: list[str] = []
            if existing is not None:
                prior_bullets = [ln[2:] for ln in existing.value.splitlines() if ln.startswith("- ")]
                if existing.provenance.startswith("consolidated:"):
                    try:
                        prev_n = int(existing.provenance.split(":")[1].split("-")[0])
                    except (ValueError, IndexError):
                        prev_n = 0
            total_n = prev_n + len(provenances)
            # Newest evidence first, cap 5, and say what was dropped (critic M2: the
            # old alphabetical [:5] silently discarded episodes while the count climbed).
            seen_b: set[str] = set()
            bullets = [b for b in new_bodies + prior_bullets
                       if not (b in seen_b or seen_b.add(b))]
            shown = bullets[:5]
            dropped = len(bullets) - len(shown)
            merged = (
                f"Recurring ({total_n} episodes): {tier} on `{tool}`.\n"
                + "\n".join(f"- {b}" for b in shown)
                + (f"\n- … (+{dropped} earlier example(s) consolidated away)" if dropped else "")
            )
            promoted = await self._store.store_fact(
                key=key,
                value=merged,
                tags=["consolidated", f"tier:{tier}"],
                confidence=_PROMOTED_CONFIDENCE,
                source="consolidation",
                provenance=f"consolidated:{total_n}-episodes",
            )
            self._promoted_ids_this_run.add(promoted.id)
            for m in members:
                try:
                    await self._store.add_edge(promoted.id, m.id, "derived_from")
                except Exception:
                    pass
                await self._untag_candidate(m.id, m.tags)
            report.promoted += 1
            report.promoted_keys.append(promoted.key)

    async def _untag_candidate(self, fact_id: int, tags: list[str]) -> None:
        """Candidate consumed: leaves the pending set and the index-eligible set (its
        content lives on in the promoted record + the derived_from edge)."""
        new_tags = json.dumps([t for t in tags if t != "pending_consolidation"])
        assert self._store._db is not None
        await self._store._db.execute(
            "UPDATE facts SET tags = ?, retrieval_strength = MIN(retrieval_strength, ?) WHERE id = ?",
            (new_tags, _DEMOTED_RS, fact_id),
        )
        await self._store._db.commit()

    # -- 3. LLM session replay (seam; cancellable; OFF unless llm wired) --

    async def _step_replay_llm(self, report: ConsolidationReport) -> None:
        if self._llm is None:
            return
        claims = await self._replay_sessions()
        report.replayed_claims = len(claims)

    async def _replay_sessions(self) -> list[str]:
        """Extract claim candidates from recent history via the (cancellable) LLM.
        Guardrails: iteration cap; dedup-before-generate (skip claims already stored);
        verify-against-leaf (a claim must share a ≥6-char token with the source text)."""
        from localharness.memory.sqlite import FactQuery

        history = await self._store.get_history(limit=200)
        if not history:
            return []
        corpus = "\n".join(
            str(r.get("content", ""))[:400] for r in history if r.get("content")
        )[:8000]
        if not corpus.strip():
            return []
        prompt = (
            "Extract at most 5 durable, self-contained lessons from this agent transcript "
            "(one per line, no numbering; only things worth knowing in FUTURE sessions):\n\n"
            + corpus
        )
        raw = await self._cancellable_complete(prompt)
        if raw is None:
            return []
        stored: list[str] = []
        for line in raw.splitlines():
            claim = line.strip(" -•\t")
            if not claim or len(stored) >= min(5, self._cfg.iteration_cap):
                continue
            # verify-against-leaf (critic M4: any-single-token was trivially passed by
            # confabulations sharing one common word like "contains"): a MAJORITY of
            # the claim's ≥6-char tokens must appear verbatim in the source transcript.
            tokens = [t for t in claim.split() if len(t) >= 6]
            if tokens:
                matched = sum(1 for t in tokens if t in corpus)
                if matched * 2 < len(tokens):  # < 50% grounded → confabulation risk
                    continue
            # dedup-before-generate: skip if an equivalent fact already exists.
            existing = await self._store.query_facts(FactQuery(text=claim[:60], limit=1))
            if existing and existing[0].value.strip() == claim:
                continue
            await self._store.store_fact(
                key=f"replay/{abs(hash(claim)) % 10**10}",
                value=claim,
                tags=["replay", "pending_consolidation"],
                confidence=0.6,
                source="consolidation_replay",
            )
            stored.append(claim)
        return stored

    async def _cancellable_complete(self, prompt: str) -> str | None:
        """Race the generation against the cancel event — a user turn must never wait
        behind a consolidation generation (the serial inference gate is non-preemptive
        and held to the last token; CONS-02)."""
        gen_task = asyncio.ensure_future(self._complete(prompt))
        cancel_task = asyncio.ensure_future(self._cancel.wait())
        done, _ = await asyncio.wait(
            {gen_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if gen_task in done:
            cancel_task.cancel()
            try:
                return await gen_task
            except Exception:
                log.exception("consolidation replay generation failed (non-fatal)")
                return None
        gen_task.cancel()  # releases the inference gate's slot
        try:
            await gen_task
        except (asyncio.CancelledError, Exception):
            pass
        return None

    async def _complete(self, prompt: str) -> str:
        result = self._llm.complete(prompt)
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            return await result
        return result

    # -- 4. retrieval-strength decay (RANK-03's time axis) ----------------

    async def _step_decay(self, report: ConsolidationReport) -> None:
        """Accessibility decays with disuse (per half-life); trust (confidence) never
        does — you don't lose the childhood phone number, you lose the ability to
        summon it. Floor at 0.05: facts fade from the index, never from the store."""
        assert self._store._db is not None
        now = int(time.time())
        half_life_s = max(self._cfg.decay_half_life_days, 0.01) * 86400
        async with self._store._db.execute(
            "SELECT id, retrieval_strength, COALESCE(last_accessed_at, updated_at) "
            "FROM facts WHERE agent_id = ? AND status = 'active' AND retrieval_strength > 0.05",
            (self._store._agent_id,),
        ) as cur:
            rows = await cur.fetchall()
        updates = []
        for fact_id, rs, last in rows:
            idle_s = max(0, now - (last or now))
            if idle_s < 86400:  # under a day idle: no decay churn
                continue
            new_rs = max(0.05, rs * 0.5 ** (idle_s / half_life_s))
            if abs(new_rs - rs) >= 0.01:
                updates.append((new_rs, fact_id))
        if updates:
            await self._store._db.executemany(
                "UPDATE facts SET retrieval_strength = ? WHERE id = ?", updates
            )
            await self._store._db.commit()
        report.decayed = len(updates)

    # -- 5. SOFT-cap trim (CONS-05) ---------------------------------------

    async def _step_cap_trim(self, report: ConsolidationReport) -> None:
        """The capacity bound is enforced HERE, at the consolidation boundary — never at
        admission (critic BLOCKER 3: hard-cap + background-only was a contradiction).
        Trim = demote lowest-scoring actives below the index gate. Nothing is deleted."""
        assert self._store._db is not None
        cap = self._cfg.max_active_facts
        now = int(time.time())
        async with self._store._db.execute(
            "SELECT COUNT(*) FROM facts WHERE agent_id = ? AND status = 'active' "
            "AND retrieval_strength >= 0.2",
            (self._store._agent_id,),
        ) as cur:
            (active,) = await cur.fetchone()
        report.active_over_cap = max(0, active - cap)
        if active <= cap:
            return
        # Exclusion (critic BLOCKER 2): never demote a record promoted in THIS run —
        # promote-then-self-demote under ordinary over-cap conditions pitted two
        # success criteria against each other. Deterministic tiebreakers (critic minor)
        # so victim selection is stable, not SQLite-implementation-defined.
        grace = self._promoted_ids_this_run or {-1}
        grace_marks = ",".join("?" * len(grace))
        async with self._store._db.execute(
            "SELECT id FROM facts WHERE agent_id = ? AND status = 'active' "
            "AND retrieval_strength >= 0.2 "
            f"AND id NOT IN ({grace_marks}) "
            "ORDER BY lh_slow_score(importance, access_count, last_accessed_at, updated_at, ?) ASC, "
            "updated_at ASC, id ASC "
            "LIMIT ?",
            (self._store._agent_id, *grace, now, active - cap),
        ) as cur:
            victims = [r[0] for r in await cur.fetchall()]
        if victims:
            await self._store._db.executemany(
                "UPDATE facts SET retrieval_strength = ? WHERE id = ?",
                [(_DEMOTED_RS, v) for v in victims],
            )
            await self._store._db.commit()
        report.demoted = len(victims)

    # -- 6. quality proxies (CONS-06) --------------------------------------

    async def _step_proxies(self, report: ConsolidationReport) -> None:
        assert self._store._db is not None
        cutoff = int(time.time()) - 14 * 86400
        async with self._store._db.execute(
            "SELECT COUNT(*), SUM(CASE WHEN status = 'superseded' THEN 1 ELSE 0 END) "
            "FROM facts WHERE agent_id = ? AND source = 'consolidation' AND created_at >= ?",
            (self._store._agent_id, cutoff),
        ) as cur:
            total, churned = await cur.fetchone()
        report.churn_rate = (churned or 0) / total if total else 0.0

        if report.promoted_keys and self._on_promotion_sample is not None:
            sample_keys = random.sample(
                report.promoted_keys, min(3, len(report.promoted_keys))
            )
            sample = [await self._store.get_fact(k) for k in sample_keys]
            try:
                result = self._on_promotion_sample([f for f in sample if f is not None])
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception("promotion-sample hook failed (non-fatal)")

    # -- watermark ---------------------------------------------------------

    async def _set_watermark(self, ts: int) -> None:
        await _set_meta(self._store, _WATERMARK_KEY, str(ts))


class ConsolidationScheduler:
    """CONS-01/02: the trigger + cancellation owner. No daemon exists on this box —
    the scheduler lives inside the harness process: a staleness check at session start
    plus an in-session idle timer; any user activity cancels a running pass instantly
    and resets the timer."""

    def __init__(
        self,
        store: "MemoryStore",
        bus: "EventBus",
        agent_id: str,
        cfg: "MemoryConsolidationConfig",
        *,
        llm: Any = None,
        on_promotion_sample: Optional[Callable[[list[Any]], Awaitable[None] | None]] = None,
    ) -> None:
        self._store = store
        self._bus = bus
        self._agent_id = agent_id
        self._cfg = cfg
        self._llm = llm
        self._on_promotion_sample = on_promotion_sample
        self._handles: list["SubscriptionHandle"] = []
        self._running: Optional[ConsolidationPass] = None
        self._run_task: Optional[asyncio.Task] = None
        self._timer_task: Optional[asyncio.Task] = None
        self._last_activity = time.monotonic()
        self.last_report: Optional[ConsolidationReport] = None

    async def start(self) -> None:
        if not self._cfg.enabled:
            return
        from localharness.core.events import UserMessage
        self._handles.append(
            self._bus.subscribe(UserMessage, self._on_user_activity)
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
                # Bounded (critic minor 3): a pathologically slow step must not stall
                # process shutdown; the cancel above makes steps exit at their next check.
                await asyncio.wait_for(self._run_task, timeout=15.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                self._run_task.cancel()

    async def _on_user_activity(self, event: Any) -> None:
        """A user turn arrived: the box is NOT idle. Cancel any in-flight pass (it will
        release the inference gate) and reset the idle clock."""
        self._last_activity = time.monotonic()
        self.cancel_running()

    def cancel_running(self) -> None:
        if self._running is not None:
            self._running.cancel()

    def launch(self) -> None:
        """Fire a pass as a background task (idempotent while one is running)."""
        if self._run_task is not None and not self._run_task.done():
            return
        self._running = ConsolidationPass(
            self._store, self._cfg, llm=self._llm,
            on_promotion_sample=self._on_promotion_sample,
        )
        self._run_task = asyncio.create_task(self._run_and_record())

    async def _run_and_record(self) -> None:
        try:
            assert self._running is not None
            self.last_report = await self._running.run()
            if self.last_report.cancelled:
                log.info("consolidation pass cancelled by user activity")
            else:
                log.info(
                    "consolidation: folded=%d promoted=%d decayed=%d demoted=%d churn=%.2f",
                    self.last_report.folded, self.last_report.promoted,
                    self.last_report.decayed, self.last_report.demoted,
                    self.last_report.churn_rate,
                )
        except Exception:
            log.exception("consolidation pass crashed (non-fatal)")
        finally:
            self._running = None

    async def should_run(self) -> bool:
        """Session-start staleness: run when the watermark is old AND there is work
        (pending candidates, staged reads, or an over-cap active tier)."""
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
        assert self._store._db is not None
        q = (
            "SELECT EXISTS(SELECT 1 FROM facts WHERE agent_id = :a AND access_count_staged > 0), "
            "EXISTS(SELECT 1 FROM facts WHERE agent_id = :a AND status = 'active' "
            "       AND tags LIKE '%\"pending_consolidation\"%'), "
            "(SELECT COUNT(*) FROM facts WHERE agent_id = :a AND status = 'active' "
            "       AND retrieval_strength >= 0.2)"
        )
        async with self._store._db.execute(q, {"a": self._agent_id}) as cur:
            staged, pending, active = await cur.fetchone()
        return bool(staged or pending or active > self._cfg.max_active_facts)

    async def _idle_timer_loop(self) -> None:
        """In-session idle trigger: no user activity for idle_minutes → launch a pass.
        The body is exception-guarded (critic M5): one transient _has_work error must
        not silently kill idle consolidation for the rest of the session."""
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
# Tiny agent-scoped KV (watermark) — idempotent DDL, no schema-version bump
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

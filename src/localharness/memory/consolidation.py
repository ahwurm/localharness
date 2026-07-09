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
cap trim, proxies) runs with NO model at all. The LLM passes (`llm=` param) are the idle
extractors — mining (the primary semantic feeder, MOVE 2), the chapter-writer, and
reconciliation — each cancellable + guarded and config-gated. (The old session-replay seam
that wrote unreachable `replay/*` keys was retired into mining in MOVE 2.)

Deliberate (whole-milestone critic m4): decay / cap-trim / untag write score columns
via raw UPDATEs, bypassing store_fact's read-back-verify — these mutate derived ranking
state, never claimed content; untag includes `tags` in its SET list so the narrowed
facts_au trigger keeps FTS consistent, and the others don't touch indexed columns.
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
_CLASSIFY_BACKFILL_CAP = 10  # tag-graph F4: max untagged pool atoms bucket-filed per idle cycle


@dataclass
class ConsolidationReport:
    folded: int = 0
    promoted: int = 0
    decayed: int = 0
    demoted: int = 0
    replayed_claims: int = 0
    schemas_written: int = 0   # Phase 36 SEMA-02/03: chapters written this pass
    reconciled: int = 0        # Phase 36 PGATE-03: correction_pending rows resolved this pass
    mined: int = 0             # Phase 36 PGATE-03: transcript facts mined this pass
    tags_proposed: int = 0     # Tag-graph discovery: new candidate child tags this pass
    tags_incorporated: int = 0 # Tag-graph discovery: candidates promoted to active this pass
    tags_pruned: int = 0       # Tag-graph discovery: stale candidates retired this pass
    tags_backfilled: int = 0   # Tag-graph F4: pool atoms bucket-filed by the idle classify step
    embedder_used: str = ""    # Tag-graph F7: the embedder class discovery ran with (forensics)
    active_over_cap: int = 0
    churn_rate: float = 0.0
    cancelled: bool = False
    duration_s: float = 0.0
    promoted_keys: list[str] = field(default_factory=list)
    # Run-2 ruling 4 (observability): EVERY chapter-writer attempt (written or rejected, with
    # its reason + grounding fields) — 'no chapter written' must leave a forensic trail.
    schema_attempts: list[dict] = field(default_factory=list)
    # FIX 2c (run-3): the RAW miner completion per chunk (pre-parse) — run-3's were unrecoverable,
    # making the shadow-duplicate root-cause inferential. A forensic trail for the supersede path.
    mining_completions: list[dict] = field(default_factory=list)
    # STAGE 1 (extraction science plan): coverage/residue — committed records the miner processed,
    # how many sourced a written atom, and the uncited rest (recall observability; the eval
    # persists the residue per run for the cross-run intersection that gates any repair build).
    mined_records_seen: int = 0
    mined_records_cited: int = 0
    mining_residue: list[dict] = field(default_factory=list)


class ConsolidationPass:
    """One consolidation run. Construct fresh per run; `cancel()` at any time."""

    def __init__(
        self,
        store: "MemoryStore",
        cfg: "MemoryConsolidationConfig",
        *,
        llm: Any = None,
        embedder: Any = None,
        on_promotion_sample: Optional[Callable[[list[Any]], Awaitable[None] | None]] = None,
    ) -> None:
        self._store = store
        self._cfg = cfg
        self._llm = llm
        self._embedder = embedder
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
            # MOVE 2: mining is the primary semantic feeder and runs BEFORE clustering so the
            # sem/ atoms it writes are available to the SAME pass's chapter-writer (else a
            # freshly-mined atom waits a whole pass to be grouped). The orphaned replay seam
            # (unreachable replay/* writes) is retired — mining is now the one idle extractor.
            self._step_mine,            # Phase 36 / MOVE 2: typed-atom transcript mining
            self._step_classify_untagged,  # Tag-graph F4: file pool atoms lacking a bucket tag
            self._step_discover_tags,   # Tag-graph: discover NEW child tags before clustering reads them
            self._step_write_schemas,   # Phase 36: chapter-writer clusters sem/ atoms (llm+config-gated)
            self._step_reconcile,       # Phase 36: correction-queue reconciliation
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

        # EXCLUDE disputed/correction facts (critic MAJOR 2): a correction_phrase supersede
        # writes the disputed row back onto the ORIGINAL gate/ key with tier:correction_pending,
        # which would otherwise group + promote to 0.8 (into the injected block) here — the
        # exact "<0.7 until Phase 36" violation. The tier tag catches BOTH the gate/-keyed
        # disputed supersede rows and the correction/quarantine/ facts in one predicate.
        assert self._store._db is not None
        async with self._store._db.execute(
            f"SELECT {self._store._FACT_COLS} FROM facts "
            "WHERE agent_id = ? AND status = 'active' "
            "AND tags LIKE '%\"pending_consolidation\"%' "
            "AND tags NOT LIKE '%\"tier:correction_pending\"%' "
            "ORDER BY created_at ASC, id ASC LIMIT ?",
            (self._store._agent_id, self._cfg.iteration_cap),
        ) as cur:
            candidates = [_row_to_fact(r) for r in await cur.fetchall()]
        # Grouping (whole-milestone critic B1): recurrence = the SAME LESSON across
        # episodes, keyed by the gate's content hash — grouping by (tier, tool) alone
        # merged two unrelated one-off errors on one tool into a fabricated "recurring"
        # record (false positive) while the genuinely-recurring case self-superseded to
        # one provenance and never promoted (false negative).
        groups: dict[tuple[str, str, str], list] = {}
        for c in candidates:
            parts = c.key.split("/")  # gate/<tier>/<tool>[/<lesson>[/<session>]]
            if len(parts) < 3 or parts[0] != "gate":
                continue
            lesson = parts[3] if len(parts) > 3 else ""
            groups.setdefault((parts[1], parts[2], lesson), []).append(c)

        for (tier, tool, lesson), members in groups.items():
            if self._cancel.is_set():
                return
            provenances = {m.provenance for m in members if m.provenance}
            key = f"learned/{tool}/{tier}" + (f"/{lesson}" if lesson else "")
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
            # PAYLOAD-FIRST, nothing before it (live test 2026-07-03: the
            # "Recurring (N episodes): tier — " prefix pushed the payload past the
            # index-line truncation — chat #3 saw bookkeeping ending in "File not
            # found: /home/…", no filename, no resolution, and fumbled with the
            # lesson nominally in context). Recurrence bookkeeping rides as a
            # SUFFIX; the same rule the dogfood forced on gate captures one layer
            # down. Every layer that touches lesson text repeats this rule.
            # 135 = the discriminating payload budget: an absolute path (~50) plus
            # error head plus resolution head must ALL fit, and 135 + the ~41-char
            # recurrence suffix stays inside the index render's 180-char line.
            lesson_preview = " ".join(shown[0].split())[:135] if shown else ""
            plural = "s" if total_n != 1 else ""
            merged = (
                f"{lesson_preview} [recurring: {total_n} episode{plural}, {tier}]\n"
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

    # -- 3. Phase-36 idle LLM passes (mine / chapter-writer / reconcile) --
    # The orphaned replay seam (MOVE 2) is RETIRED: it was a near-duplicate idle extractor that
    # wrote unreachable `replay/*` keys (neither promotion nor clustering could consume them).
    # Mining is now the ONE idle extractor. Each pass below mirrors the same gating (return
    # immediately when self._llm is None) AND its own config axis, delegating to a Wave-2 sibling
    # (all never-raise + cancellable via the shared idle_llm path) and threading self._cancel so a
    # user turn stops them mid-look. Because they early-return with no LLM, every existing llm=None
    # test sees identical behavior — the deterministic core is byte-unchanged.

    async def _step_write_schemas(self, report: ConsolidationReport) -> None:
        if self._llm is None or not getattr(self._cfg, "schema_writer_enabled", False):
            return
        from localharness.memory.chapter_writer import write_cluster_schemas
        written = await write_cluster_schemas(
            self._store, self._llm, self._cancel,
            min_sessions=self._cfg.cluster_min_sessions,
            write_budget=self._cfg.schema_write_budget,
            depth_cap=self._cfg.schema_depth_cap,
            attempts_log=report.schema_attempts,  # ruling 4: every attempt observable
        )
        report.schemas_written = len(written)

    async def _step_classify_untagged(self, report: ConsolidationReport) -> None:
        """Tag-graph F4: file pool-visible atoms that LACK a bucket tag through the SAME two-step
        classifier mining uses at mint time. remember()-sourced facts are pool members but never
        pass through mine_transcript, so without this step they could never co-tag-edge (Stage B)
        or enter discovery (which requires a bucket tag); it also catches pre-existing untagged
        atoms. Bounded per cycle (_CLASSIFY_BACKFILL_CAP); provenance='backfill' keeps mint vs
        idle filing distinguishable in forensics. Same gating as mint filing."""
        if self._llm is None or not getattr(self._cfg, "mint_tagging_enabled", False):
            return
        from localharness.memory.clustering import _load_pool
        from localharness.memory.tag_classify import file_atom_tags
        assert self._store._db is not None
        async with self._store._db.execute(
            "SELECT DISTINCT a.atom_id FROM atom_tags a JOIN tags t ON t.id = a.tag_id "
            "WHERE t.agent_id = ? AND t.parent_id IS NULL", (self._store._agent_id,),
        ) as cur:
            bucketed = {r[0] for r in await cur.fetchall()}
        todo = [f for f in await _load_pool(self._store)
                if f.node_kind != "schema" and f.id not in bucketed][:_CLASSIFY_BACKFILL_CAP]
        for f in todo:
            if self._cancel.is_set():
                return
            topic = f.key.split("/")[1] if f.key.startswith("sem/") else f.key
            bucket, _child = await file_atom_tags(
                self._store, self._llm, self._cancel,
                atom_id=f.id, topic=topic, claim=f.value, provenance="backfill")
            if bucket is not None:
                report.tags_backfilled += 1

    async def _step_discover_tags(self, report: ConsolidationReport) -> None:
        """Tag-graph discovery (Stage C): propose/incorporate/prune child tags over bucket-only
        atoms BEFORE the chapter-writer runs, so a just-incorporated tag forms co-tag edges in the
        SAME pass. Gated on the LLM (llm=None -> inert, deterministic core byte-unchanged) and its
        config axis. The embedder falls back to the dep-free default when none was injected; the
        class actually used is recorded on the report (F7 — run-9 forensics must tell MiniLM from
        the HashingEmbedder fallback)."""
        if self._llm is None or not getattr(self._cfg, "tag_discovery_enabled", False):
            return
        from localharness.memory.discovery import discover_tags
        embedder = self._embedder
        if embedder is None:
            from localharness.memory.embeddings import default_embedder
            embedder = default_embedder()
        report.embedder_used = type(embedder).__name__
        r = await discover_tags(self._store, self._llm, self._cancel, embedder=embedder)
        report.tags_proposed = len(r.proposed)
        report.tags_incorporated = len(r.incorporated)
        report.tags_pruned = len(r.pruned)

    async def _step_reconcile(self, report: ConsolidationReport) -> None:
        if self._llm is None or not getattr(self._cfg, "reconcile_enabled", False):
            return
        from localharness.memory.reconciliation import reconcile_corrections
        r = await reconcile_corrections(
            self._store, self._llm, self._cancel, ttl_looks=self._cfg.reconcile_ttl_looks
        )
        # Every disposition is a resolved queue row (36-05's 7-field, shape-aware counters):
        # confirm (shape b) + confirm-corrected/retire (shape a) + revert-restore/clear + undecided.
        report.reconciled = (
            r.confirmed + r.confirmed_corrected + r.retired
            + r.reverted_restored + r.reverted_cleared + r.undecided
        )

    async def _step_mine(self, report: ConsolidationReport) -> None:
        if self._llm is None or not getattr(self._cfg, "mining_enabled", False):
            return
        from localharness.memory.mining import mine_transcript
        m = await mine_transcript(
            self._store, self._llm, self._cancel, write_budget=self._cfg.mining_write_budget,
            corpus_char_cap=self._cfg.mining_corpus_char_cap,  # FIX 3b chunk size
            known_atoms_cap=self._cfg.mining_known_atoms_cap,  # FIX 3
            # FIX 4: conversational surface only (no tool read-backs)
            operative_message_types=self._cfg.mining_operative_message_types,
            completions_log=report.mining_completions,  # FIX 2c: persist raw completions
            file_tags=getattr(self._cfg, "mint_tagging_enabled", True),  # M1 mint-time filing
        )
        report.mined = m.written
        report.mined_records_seen = m.records_seen      # STAGE 1 coverage
        report.mined_records_cited = m.records_cited
        report.mining_residue = m.residue

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
        embedder: Any = None,
        on_promotion_sample: Optional[Callable[[list[Any]], Awaitable[None] | None]] = None,
    ) -> None:
        self._store = store
        self._bus = bus
        self._agent_id = agent_id
        self._cfg = cfg
        self._llm = llm
        self._embedder = embedder
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
            self._store, self._cfg, llm=self._llm, embedder=self._embedder,
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
        # Scope the pending probe to gate/-keyed candidates (critic minor 1): predgate/ +
        # correction/quarantine/ telemetry carry pending_consolidation forever (they never
        # match the gate/ promotion prefix, so _untag_candidate never clears them), which
        # otherwise pins _has_work True every sitting and defeats the staleness optimization.
        # gate/ is exactly the set _step_promote_recurring promotes — WriteGate's convention.
        # Disputed rows are likewise excluded (critic re-verdict residual): a correction
        # supersede on a staged gate/ candidate keeps the gate/ key but carries
        # tier:correction_pending, which promotion skips — so it can never be untagged and
        # must not count as work either. Mirrors _step_promote_recurring's exclusion.
        # Phase-36 DELIBERATE premise change (the 4th clause): reconciliation is the ONLY
        # consumer that CLEARS tier:correction_pending (promote-recurring excludes it), so when
        # reconcile is enabled an active quarantined correction IS work — the idle scheduler
        # must fire the consumer instead of optimizing the pass away. When reconcile is disabled
        # the old behavior holds EXACTLY (Pitfall 3's staleness optimization intact for non-36
        # configs). surprising_failure rows stay EXCLUDED here on purpose: the chapter-writer
        # drains them piggybacking on real-work passes (36-04), never re-pinning this probe.
        q = (
            "SELECT EXISTS(SELECT 1 FROM facts WHERE agent_id = :a AND access_count_staged > 0), "
            "EXISTS(SELECT 1 FROM facts WHERE agent_id = :a AND status = 'active' "
            "       AND tags LIKE '%\"pending_consolidation\"%' AND key LIKE 'gate/%' "
            "       AND tags NOT LIKE '%\"tier:correction_pending\"%'), "
            "(SELECT COUNT(*) FROM facts WHERE agent_id = :a AND status = 'active' "
            "       AND retrieval_strength >= 0.2), "
            "EXISTS(SELECT 1 FROM facts WHERE agent_id = :a AND status = 'active' "
            "       AND tags LIKE '%\"tier:correction_pending\"%')"
        )
        async with self._store._db.execute(q, {"a": self._agent_id}) as cur:
            staged, pending, active, corrections = await cur.fetchone()
        reconcile_work = corrections and getattr(self._cfg, "reconcile_enabled", False)
        return bool(staged or pending or reconcile_work or active > self._cfg.max_active_facts)

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

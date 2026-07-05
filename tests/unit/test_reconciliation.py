"""Phase-36 reconciliation (36-05) — the idle tri-outcome model-look that clears the
`tier:correction_pending` quarantine, resolving BOTH gate shapes so no row re-surfaces.

Everything runs against a FAKE LLM (machine-safety: the suite never launches a live
prefill). Task 1 locks the shape-aware CONFIRM + the UNDECIDED TTL drain + clean cancel;
Task 2 locks REVERT (restore shape a / clear shape b) and the stacked-dispute /
no-antecedent traps (Pitfall 5 + BLOCKER 1).
"""
import asyncio
from pathlib import Path

from localharness.memory.consolidation import _get_meta, _set_meta
from localharness.memory.reconciliation import (
    _DISPUTE_MARKER,
    _QUARANTINE_PREFIX,
    ReconcileReport,
    reconcile_corrections,
)
from localharness.memory.sqlite import FactQuery, MemoryStore

AGENT = "reconcile-agent"
# The exact tag set the live gate stamps on BOTH dispute shapes (predictive_write_gate.py:232/245).
_PENDING_TAGS = ["correction", "tier:correction_pending", "pending_consolidation"]


def make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(agent_id=AGENT, division_id="", org_id="", base_dir=str(tmp_path))


class _FakeLLM:
    """Returns a fixed one-line disposition — the model-look under test is deterministic
    (mirrors test_idle_llm._FakeLLM; each test seeds exactly one queued fact)."""

    def __init__(self, answer: str):
        self.answer = answer
        self.calls = 0

    async def complete(self, prompt: str) -> str:
        self.calls += 1
        return self.answer


async def _seed_shape_b(store: MemoryStore, *, preview: str = "i got super duper sun burnt today",
                        provenance: str = "evt-b") -> str:
    """The MEASURED-MAJORITY shape: a fresh `correction/quarantine/{h8}` quarantine-only fact
    (the user's own words, prefix-tagged, NO antecedent) — exactly predictive_write_gate.py:242-249."""
    key = "correction/quarantine/deadbeef"
    await store.store_fact(
        key=key, value=f"{_QUARANTINE_PREFIX}{preview}",
        tags=list(_PENDING_TAGS), confidence=0.65,
        source="predictive_write_gate", provenance=provenance,
    )
    return key


async def _seed_shape_a(store: MemoryStore, *, pre_value: str, key: str = "profile/city",
                        provenance: str = "evt-a", ctx_msg: str | None = None) -> str:
    """The supersede-WRAP shape: a pre-dispute antecedent, then the exact gate wrap over it
    (predictive_write_gate.py:228-236). Optionally seed the user_signals correction text the
    reconciler dereferences by provenance == event_id."""
    await store.store_fact(key=key, value=pre_value, confidence=0.8)  # pre-dispute antecedent
    await store.store_fact(
        key=key, value=f"{_DISPUTE_MARKER} {pre_value}",
        tags=list(_PENDING_TAGS), confidence=0.6,
        source="predictive_write_gate", provenance=provenance,
    )
    if ctx_msg is not None:
        await store.record_user_signal(
            session_id="s1", ts=0, signal_type="correction", trigger_family="correction_phrase",
            matched_text="actually", user_message=ctx_msg, corrected_turn_summary=None,
            event_id=provenance,
        )
    return key


# ---------------------------------------------------------------------------
# Task 1 — shape-aware CONFIRM, UNDECIDED TTL drain, clean cancel
# ---------------------------------------------------------------------------


async def test_shape_b_confirm_strips_prefix_and_keeps_audit_row(tmp_path: Path):
    # Shape (b): the quarantined row IS the user's own statement -> CONFIRM settles it with the
    # quarantine prefix STRIPPED; the prefixed audit row survives in get_fact_history.
    store = make_store(tmp_path)
    await store.open()
    try:
        key = await _seed_shape_b(store, preview="i got super duper sun burnt today")
        report = await reconcile_corrections(store, _FakeLLM("CONFIRM"), asyncio.Event())
        active = await store.get_fact(key)
        assert active is not None
        assert active.value == "i got super duper sun burnt today"  # prefix STRIPPED
        assert not active.value.startswith(_QUARANTINE_PREFIX)
        assert "tier:correction_pending" not in active.tags         # left the queue
        assert "tier:reconcile_confirmed" in active.tags
        history = await store.get_fact_history(key)
        assert any(v.value.startswith(_QUARANTINE_PREFIX) for v in history)  # audit row survives
        assert report.confirmed == 1
    finally:
        await store.close()


async def test_shape_a_confirm_writes_grounded_corrected_value_never_suspect(tmp_path: Path):
    # Shape (a): CONFIRM means the correction was RIGHT -> the wrapped suspect was WRONG. Write
    # the model-derived, GROUNDED corrected value; the stripped suspect text is NEVER re-asserted.
    store = make_store(tmp_path)
    await store.open()
    try:
        key = await _seed_shape_a(
            store, pre_value="The capital of France is Marseille", provenance="evt-city",
            ctx_msg="no, the capital of France is Paris, not Marseille",
        )
        llm = _FakeLLM("CONFIRM: The capital of France is Paris")
        report = await reconcile_corrections(store, llm, asyncio.Event())
        active = await store.get_fact(key)
        assert active is not None
        assert active.value == "The capital of France is Paris"      # corrected value
        assert active.value != "The capital of France is Marseille"  # NOT the suspect
        assert _DISPUTE_MARKER not in active.value                   # never re-wrapped
        assert "tier:correction_pending" not in active.tags
        assert "tier:reconcile_confirmed" in active.tags
        assert report.confirmed_corrected == 1
        assert report.retired == 0
    finally:
        await store.close()


async def test_shape_a_confirm_bare_retires_suspect_never_reasserts(tmp_path: Path):
    # Shape (a) CONFIRM with NO derivable value -> RETIRE the suspect (retag out + demote rs);
    # the stripped suspect text must NEVER become the settled active value.
    store = make_store(tmp_path)
    await store.open()
    try:
        key = await _seed_shape_a(store, pre_value="The capital of France is Marseille",
                                  provenance="evt-city2")
        report = await reconcile_corrections(store, _FakeLLM("CONFIRM"), asyncio.Event())
        active = await store.get_fact(key)
        assert active is not None
        assert active.value != "The capital of France is Marseille"  # suspect NOT settled clean
        assert active.value.startswith(_DISPUTE_MARKER)              # still marked, demoted
        assert "tier:reconcile_retired" in active.tags
        assert "tier:correction_pending" not in active.tags
        assert active.retrieval_strength == 0.15                     # out of retrieval competition
        assert report.retired == 1
        assert report.confirmed_corrected == 0
    finally:
        await store.close()


async def test_shape_a_confirm_ungrounded_value_retires_not_writes(tmp_path: Path):
    # KILL discipline: a hallucinated correction (no token derivable from the context) must NOT
    # settle — it RETIRES the suspect rather than asserting invented text as fact.
    store = make_store(tmp_path)
    await store.open()
    try:
        key = await _seed_shape_a(store, pre_value="The database uses Postgres",
                                  provenance="evt-db", ctx_msg="no it's not Postgres")
        llm = _FakeLLM("CONFIRM: The elephant migrated overseas yesterday")  # ungrounded confab
        report = await reconcile_corrections(store, llm, asyncio.Event())
        active = await store.get_fact(key)
        assert active is not None
        assert "elephant" not in active.value                        # confab NEVER written
        assert "tier:reconcile_retired" in active.tags
        assert report.retired == 1
        assert report.confirmed_corrected == 0
    finally:
        await store.close()


async def test_undecided_drains_at_ttl(tmp_path: Path):
    # UNDECIDED bounds the queue: pre-seed the per-key look counter to ttl_looks-1 so this look
    # trips the TTL and DRAINS the row permanently via a raw retag (tier:reconcile_stale).
    store = make_store(tmp_path)
    await store.open()
    try:
        key = await _seed_shape_b(store)
        await _set_meta(store, f"reconcile/looks/{key}", "2")  # ttl_looks default 3 -> next look drains
        report = await reconcile_corrections(store, _FakeLLM("UNDECIDED"), asyncio.Event())
        active = await store.get_fact(key)
        assert active is not None
        assert "tier:reconcile_stale" in active.tags
        assert "tier:correction_pending" not in active.tags          # drained out of the queue
        assert report.undecided == 1
        assert await _get_meta(store, f"reconcile/looks/{key}") == "3"
    finally:
        await store.close()


async def test_undecided_below_ttl_stays_quarantined(tmp_path: Path):
    # Below the TTL, an undecidable fact STAYS quarantined (bump the counter, leave the tags).
    store = make_store(tmp_path)
    await store.open()
    try:
        key = await _seed_shape_b(store)
        report = await reconcile_corrections(store, _FakeLLM("UNDECIDED"), asyncio.Event())
        active = await store.get_fact(key)
        assert active is not None
        assert "tier:correction_pending" in active.tags              # still quarantined (1 < 3)
        assert "tier:reconcile_stale" not in active.tags
        assert report.undecided == 1
        assert await _get_meta(store, f"reconcile/looks/{key}") == "1"
    finally:
        await store.close()


async def test_set_cancel_event_stops_loop_without_hang(tmp_path: Path):
    # A set cancel_event stops the loop promptly (no hang) and dispositions nothing.
    store = make_store(tmp_path)
    await store.open()
    try:
        key = await _seed_shape_b(store)
        cancel = asyncio.Event()
        cancel.set()
        report = await asyncio.wait_for(
            reconcile_corrections(store, _FakeLLM("CONFIRM"), cancel), timeout=3.0
        )
        assert isinstance(report, ReconcileReport)
        assert report.cancelled is True
        assert report.confirmed == 0
        active = await store.get_fact(key)
        assert active is not None and "tier:correction_pending" in active.tags  # untouched
    finally:
        await store.close()

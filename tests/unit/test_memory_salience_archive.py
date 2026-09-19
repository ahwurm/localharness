"""Salience + cold archive under the resonance rebuild.

What is under test, in the order it matters:

1. **One salience currency**: S = ln(1 + standing) + truth log-odds + declared stakes.
   No clock term — an idle store's scores do not move.
2. **The line is read out of the store, never configured**: the minimum S over facts
   the store can PROVE were worth keeping (ever recalled, or the owner's own hand).
   No anchors -> no line -> nothing archives (cold start).
3. **The owner's hand is a hard floor**, independent of whichever line rule is in force.
4. **Archive is a MOVE, not a delete**: the row leaves `facts` (and the FTS haystack
   with it), and restore brings it back byte-identical — full physical row.
5. **The list-driven verb re-checks every rail at execution time.**
"""
from __future__ import annotations

import math
import time
from pathlib import Path

from localharness.memory.consolidation import (
    SKIP_NOT_ACTIVE,
    SKIP_OWNER_TOUCHED,
    SKIP_RECALLED,
    SKIP_UNKNOWN,
    SKIP_UNPARSEABLE,
    archive_dormant_facts,
    archive_listed_facts,
    parse_archive_list,
)
from localharness.memory.salience import (
    archive_line,
    format_pass_line,
    is_anchor,
    is_pinned,
    score_fact,
    score_facts,
    select_archivable,
    truth_log_odds,
    vacuum_warranted,
)
from localharness.memory.sqlite import (
    ARCHIVE_STAMP_PREFIX,
    USER_EDIT_PROVENANCE_PREFIX,
    FactQuery,
    MemoryStore,
    _logit,
)


def make_store(tmp_path: Path, agent: str = "orchestrator") -> MemoryStore:
    return MemoryStore(agent_id=agent, division_id="default", org_id="default",
                       base_dir=str(tmp_path))


async def _set(store: MemoryStore, key: str, **cols) -> int:
    fact = await store.get_fact(key)
    assert fact is not None
    sets = ", ".join(f"{c} = ?" for c in cols)
    await store._db.execute(
        f"UPDATE facts SET {sets} WHERE id = ?", (*cols.values(), fact.id)
    )
    await store._db.commit()
    return fact.id


# --------------------------------------------------------------------- scoring

async def test_salience_is_standing_plus_truth_plus_stakes(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        f = await store.store_fact("k", "v", source="w", importance=0.25)
        await _set(store, "k", standing=3.0, truth_logodds=0.7)
        fact = await store.get_fact("k")
        s = score_fact(fact)
        assert s.need == math.log1p(3.0)
        assert s.truth == 0.7
        assert s.stakes == 0.25
        assert s.s == s.need + s.truth + s.stakes
        assert f.standing == 0.0  # birth standing is zero — earned, not granted
    finally:
        await store.close()


async def test_legacy_row_truth_derives_from_confidence(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("k", "v", confidence=0.8, source="w")
        await _set(store, "k", truth_logodds=None)
        fact = await store.get_fact("k")
        assert fact.truth_logodds is None
        assert truth_log_odds(fact) == _logit(0.8)
    finally:
        await store.close()


async def test_no_clock_term_scores_are_time_invariant(tmp_path):
    """The rebuild's law: an idle store does not rot. Backdating a fact by a year
    changes nothing about its salience."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("k", "v", source="w")
        fid = await _set(store, "k", standing=1.0)
        before = score_fact(await store.get_fact("k")).s
        year_ago = int(time.time()) - 365 * 86400
        await store._db.execute(
            "UPDATE facts SET created_at = ?, updated_at = ?, last_accessed_at = ? WHERE id = ?",
            (year_ago, year_ago, year_ago, fid),
        )
        await store._db.commit()
        after = score_fact(await store.get_fact("k")).s
        assert before == after
    finally:
        await store.close()


# --------------------------------------------------------------------- anchors + line

def test_line_is_min_over_anchors_and_none_when_cold():
    assert archive_line([]) is None


async def test_cold_store_archives_nothing(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        for i in range(5):
            await store.store_fact(f"k{i}", f"v{i}", source="w")
        run = await archive_dormant_facts(store)
        assert run.line is None and run.moved == 0
        assert await store.count_archived() == 0
    finally:
        await store.close()


async def test_recalled_fact_anchors_and_junk_below_line_archives(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("useful", "recalled fact", source="w")
        await _set(store, "useful", access_count=2, standing=2.0)
        await store.store_fact("junk", "never recalled", source="w")
        # junk sits below the anchor: standing 0, same truth, no stakes
        facts = [await store.get_fact("useful"), await store.get_fact("junk")]
        scored = score_facts(facts)
        line = archive_line(scored)
        assert line == score_fact(facts[0]).s
        condemned = select_archivable(scored, line)
        assert [c.key for c in condemned] == ["junk"]

        run = await archive_dormant_facts(store)
        assert run.moved == 1
        assert await store.get_fact("junk") is None
        assert await store.get_fact("useful") is not None
    finally:
        await store.close()


async def test_owner_hand_is_pinned_at_any_score(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("mine", "remembered by owner", source="remember")
        await store.store_fact(
            "edited", "owner edited this", source="w",
            provenance=f"{USER_EDIT_PROVENANCE_PREFIX}123;web",
        )
        for key in ("mine", "edited"):
            fact = await store.get_fact(key)
            assert is_pinned(fact) and is_anchor(fact)
        # even handed a line above them, select_archivable refuses pins
        scored = score_facts([await store.get_fact("mine"), await store.get_fact("edited")])
        condemned = select_archivable(scored, line=999.0)
        assert condemned == []
    finally:
        await store.close()


def test_vacuum_bar_is_moved_greater_than_kept():
    assert vacuum_warranted(6, 10) is True      # moved 6, kept 4
    assert vacuum_warranted(5, 10) is False     # moved 5, kept 5 — tie keeps freelist
    assert vacuum_warranted(0, 10) is False


def test_format_pass_line_cold_and_hot():
    assert "S=none" in format_pass_line(0, None)
    assert "S=1.25" in format_pass_line(3, 1.25)


# --------------------------------------------------------------------- move + restore

async def test_archive_is_a_verified_move_and_restore_is_byte_identical(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("gone", "the body", source="w", tags=["t1"])
        fid = await _set(store, "gone", standing=0.5)
        row_before = await store.get_fact("gone")

        assert await store.archive_fact(fid, surface="rung1-floor-line",
                                        s_at_archive=-1.0, line_at_archive=0.0)
        assert await store.get_fact("gone") is None
        # FTS haystack no longer sees it
        assert await store.query_facts(FactQuery(text="body")) == []
        archived = await store.list_archived()
        assert [f.id for f in archived] == [fid]

        assert await store.restore_fact(fid)
        row_after = await store.get_fact("gone")
        assert row_after == row_before
        assert await store.query_facts(FactQuery(text="body")) != []
    finally:
        await store.close()


async def test_unfolded_reads_refuse_the_move(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("hot", "just read", source="w")
        await store.touch_staged(["hot"])
        fact = await store.get_fact("hot")
        assert not await store.archive_fact(fact.id, surface="rung1-floor-line",
                                            s_at_archive=0.0, line_at_archive=None)
        assert await store.get_fact("hot") is not None
    finally:
        await store.close()


# --------------------------------------------------------------------- list-driven rails

def test_parse_archive_list_ids_comments_and_garbage():
    ids, bad = parse_archive_list("# comment\n12|key|stuff\n\nnope|x\n12\n13\n")
    assert ids == [12, 13]
    assert len(bad) == 1 and bad[0].reason == SKIP_UNPARSEABLE


async def test_listed_run_recheck_rails(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("plain", "archivable", source="w")
        await store.store_fact("pinned", "owner's own", source="remember")
        await store.store_fact("recalled", "used before", source="w")
        await _set(store, "recalled", access_count=1)
        plain = await store.get_fact("plain")
        pinned = await store.get_fact("pinned")
        recalled = await store.get_fact("recalled")

        run = await archive_listed_facts(
            store, [plain.id, pinned.id, recalled.id, 99_999]
        )
        assert run.moved == 1
        reasons = {sk.fact_id: sk.reason for sk in run.skipped}
        assert reasons[pinned.id] == SKIP_OWNER_TOUCHED
        assert reasons[recalled.id] == SKIP_RECALLED
        assert reasons[99_999] == SKIP_UNKNOWN

        # a second run on the moved id: the row has LEFT facts (it lives in the
        # archive), so the rail reports unknown-here — and a superseded row
        # reports not-active.
        run2 = await archive_listed_facts(store, [plain.id])
        assert run2.moved == 0
        assert run2.skipped[0].reason == SKIP_UNKNOWN
        await store.store_fact("recalled", "new version", source="w")
        old = (await store.get_fact_history("recalled"))[-1]
        run3 = await archive_listed_facts(store, [old.id])
        assert run3.moved == 0
        assert run3.skipped[0].reason == SKIP_NOT_ACTIVE
    finally:
        await store.close()


async def test_dry_run_moves_nothing(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("stay", "content", source="w")
        fact = await store.get_fact("stay")
        run = await archive_listed_facts(store, [fact.id], dry_run=True)
        assert run.moved == 0 and len(run.candidates) == 1
        assert await store.get_fact("stay") is not None
        assert await store.count_archived() == 0
    finally:
        await store.close()


async def test_archive_stamp_carries_epoch_and_surface(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("s", "v", source="w")
        fact = await store.get_fact("s")
        await store.archive_fact(fact.id, surface="consensus-list",
                                 s_at_archive=0.0, line_at_archive=None)
        async with store._db.execute(
            "SELECT archive_rung, archived_at FROM facts_archive WHERE id = ?", (fact.id,)
        ) as cur:
            rung, at = await cur.fetchone()
        assert rung == f"{ARCHIVE_STAMP_PREFIX}{at};consensus-list"
    finally:
        await store.close()

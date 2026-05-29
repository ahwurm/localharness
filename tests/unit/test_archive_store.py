"""ARCH-01 — ArchiveStore persistence contract (Phase 15 Wave 0 RED stubs).

Each test is an xfail(strict=False) stub: it encodes the intended behavior with a
real assertion so it goes RED→GREEN as the ArchiveStore implementation lands
(write/get/reopen/FK-RESTRICT/migration/event-emit/approval-history land in 15-02).

Module-level import is guarded so collection never breaks before the module ships;
once it exists the import resolves and the xfail markers govern pass/fail.
"""
import sqlite3

import pytest

try:
    from localharness.autoresearch.archive import ArchiveStore, ArchiveEntry, ArchiveQuery  # noqa: F401
except ImportError:
    pytest.skip("ArchiveStore not yet implemented (15-02)", allow_module_level=True)


@pytest.mark.xfail(strict=False, reason="impl lands in 15-02")
async def test_write_read_round_trip_all_fields(archive_store, seeded_archive):
    """All 12 fields (incl. nested per-fixture dict + diff JSON blob) round-trip exactly."""
    import json

    [written] = await seeded_archive(
        archive_store,
        [
            dict(
                id="rt-1",
                parent_id=None,
                component="agents.main.system_prompt",
                diff=json.dumps({"before": "old", "after": "new"}),
                train_score=0.81,
                train_scores_per_fixture={"fx1": 0.9, "fx2": 0.7},
                holdout_score=0.79,
                p_value=0.012,
                cost=1.25,
                ts=1700000000,
                approved_by="human:alice",
                status="promoted",
            )
        ],
    )
    got = await archive_store.get(written.id)
    assert got is not None
    assert got.id == "rt-1"
    assert got.parent_id is None
    assert got.component == "agents.main.system_prompt"
    assert json.loads(got.diff) == {"before": "old", "after": "new"}
    assert got.train_score == 0.81
    assert got.train_scores_per_fixture == {"fx1": 0.9, "fx2": 0.7}
    assert got.holdout_score == 0.79
    assert got.p_value == 0.012
    assert got.cost == 1.25
    assert got.ts == 1700000000
    assert got.approved_by == "human:alice"
    assert got.status == "promoted"


@pytest.mark.xfail(strict=False, reason="impl lands in 15-02")
async def test_survives_reopen(tmp_path):
    """A row written by one store is readable after closing and reopening a NEW store on the same path."""
    import json

    path = tmp_path / ".localharness" / "archive.db"
    store = ArchiveStore(path)
    await store.open()
    await store.write(
        ArchiveEntry(
            id="persist-1",
            parent_id=None,
            component="agents.main.system_prompt",
            diff=json.dumps({"before": "a", "after": "b"}),
            train_score=0.5,
            train_scores_per_fixture=None,
            holdout_score=None,
            p_value=None,
            cost=None,
            ts=1700000000,
            approved_by=None,
            status="in_flight",
        )
    )
    await store.close()

    reopened = ArchiveStore(path)
    await reopened.open()
    got = await reopened.get("persist-1")
    assert got is not None and got.id == "persist-1"
    assert await reopened.integrity_check() == []
    await reopened.close()


@pytest.mark.xfail(strict=False, reason="impl lands in 15-02")
async def test_parent_delete_restricted(archive_store, seeded_archive):
    """Deleting a row that is a lineage parent raises IntegrityError (ON DELETE RESTRICT)."""
    await seeded_archive(
        archive_store,
        [
            dict(id="parent-a", parent_id=None),
            dict(id="child-b", parent_id="parent-a"),
        ],
    )
    with pytest.raises(sqlite3.IntegrityError):
        await archive_store._db.execute("DELETE FROM mutations WHERE id='parent-a'")
        await archive_store._db.commit()


@pytest.mark.xfail(strict=False, reason="impl lands in 15-02")
async def test_user_version_set_on_open(tmp_path):
    """PRAGMA user_version is 1 after open and stays 1 across a reopen (idempotent migration)."""
    path = tmp_path / ".localharness" / "archive.db"
    store = ArchiveStore(path)
    await store.open()
    async with store._db.execute("PRAGMA user_version") as cur:
        (version,) = await cur.fetchone()
    assert version == 1
    await store.close()

    reopened = ArchiveStore(path)
    await reopened.open()
    async with reopened._db.execute("PRAGMA user_version") as cur:
        (version2,) = await cur.fetchone()
    assert version2 == 1
    await reopened.close()


@pytest.mark.xfail(strict=False, reason="impl lands in 15-02")
async def test_write_emits_event(tmp_path, bus):
    """write() publishes a MutationArchived event whose mutation_id matches the written row id."""
    import json

    from localharness.core.events import MutationArchived

    received = []

    async def _handler(event):
        received.append(event)

    bus.subscribe(MutationArchived, _handler)

    store = ArchiveStore(tmp_path / ".localharness" / "archive.db", bus=bus)
    await store.open()
    written = await store.write(
        ArchiveEntry(
            id="evt-1",
            parent_id=None,
            component="agents.main.system_prompt",
            diff=json.dumps({"before": "a", "after": "b"}),
            train_score=None,
            train_scores_per_fixture=None,
            holdout_score=None,
            p_value=None,
            cost=None,
            ts=1700000000,
            approved_by=None,
            status="in_flight",
        )
    )
    # publish() delivers to subscribers inline (bus.py:_deliver), so the
    # handler has already run by the time write() returns.
    assert len(received) == 1
    assert received[0].mutation_id == written.id
    await store.close()


@pytest.mark.xfail(strict=False, reason="impl lands in 15-02")
async def test_approval_appends_history(archive_store, seeded_archive):
    """add_approval twice updates mutations.approved_by to the latest approver and appends 2 history rows."""
    [entry] = await seeded_archive(archive_store, [dict(id="appr-1", status="in_flight")])
    await archive_store.add_approval(entry.id, "human:alice", "looks good")
    await archive_store.add_approval(entry.id, "human:bob", "ship it")

    got = await archive_store.get(entry.id)
    assert got.approved_by == "human:bob"
    async with archive_store._db.execute(
        "SELECT COUNT(*) FROM mutation_approvals WHERE mutation_id=?", (entry.id,)
    ) as cur:
        (count,) = await cur.fetchone()
    assert count == 2


# ---------------------------------------------------------------------------
# Phase 17 Wave 0 — update_verdict (impl lands 17-02)
#
# The experiment runner is the metric writer: it flips an in_flight row to
# promoted/train_rejected/holdout_rejected and fills train/holdout/p/cost on the
# rows the `propose --archive` seam created with null scores.
# ---------------------------------------------------------------------------


async def test_update_verdict(archive_store, seeded_archive):
    """update_verdict flips status + fills every score; returned entry AND a fresh get() agree."""
    await seeded_archive(archive_store, [dict(id="uv-1", status="in_flight")])
    returned = await archive_store.update_verdict(
        "uv-1",
        status="promoted",
        train_score=0.82,
        train_scores_per_fixture={"01_pure_qa": 0.9},
        holdout_score=0.80,
        p_value=0.01,
        cost=1234.0,
    )
    refetched = await archive_store.get("uv-1")
    for got in (returned, refetched):
        assert got.status == "promoted"
        assert got.train_score == 0.82
        assert got.train_scores_per_fixture == {"01_pure_qa": 0.9}
        assert got.holdout_score == 0.80
        assert got.p_value == 0.01
        assert got.cost == 1234.0


async def test_update_verdict_republishes_event(tmp_path, bus, seeded_archive):
    """update_verdict re-publishes MutationArchived with the NEW status + filled train_score."""
    from localharness.core.events import MutationArchived

    received = []

    async def _handler(event):
        received.append(event)

    bus.subscribe(MutationArchived, _handler)
    store = ArchiveStore(tmp_path / ".localharness" / "archive.db", bus=bus)
    await store.open()
    await seeded_archive(store, [dict(id="uv-evt", status="in_flight")])
    received.clear()  # drop the write() event; we assert on the update re-publish only
    await store.update_verdict("uv-evt", status="promoted", train_score=0.77, p_value=0.02)
    promoted = [e for e in received if e.status == "promoted"]
    assert len(promoted) == 1
    assert promoted[0].mutation_id == "uv-evt"
    assert promoted[0].train_score == 0.77
    await store.close()


async def test_update_verdict_preserves_integrity(archive_store, seeded_archive):
    """integrity_check stays clean ([]) after an update_verdict write-back."""
    await seeded_archive(archive_store, [dict(id="uv-int", status="in_flight")])
    await archive_store.update_verdict("uv-int", status="promoted", train_score=0.9, p_value=0.01)
    assert await archive_store.integrity_check() == []


# ---------------------------------------------------------------------------
# Phase 18 Wave 0 — new loop statuses (adopted / held / adoption_rejected)
#
# status is a free-text TEXT column (no CHECK constraint, no migration), so the
# orchestrator's three new lifecycle statuses round-trip immediately. This test is
# NOT xfail: it must pass the moment it lands (the schema already supports it).
# ---------------------------------------------------------------------------


async def test_new_loop_statuses(archive_store, seeded_archive):
    """The three Phase-18 statuses adopted/held/adoption_rejected each round-trip via update_verdict (no migration error)."""
    await seeded_archive(archive_store, [dict(id="loop-status", status="promoted")])

    for status in ("adopted", "held", "adoption_rejected"):
        returned = await archive_store.update_verdict("loop-status", status=status)
        assert returned.status == status
        refetched = await archive_store.get("loop-status")
        assert refetched.status == status  # the literal persists across a fresh read

    assert await archive_store.integrity_check() == []  # no migration corruption

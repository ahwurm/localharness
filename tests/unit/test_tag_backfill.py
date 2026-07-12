"""TAGG-04 (Plan 36.2-03, wave 2) — the one-shot tag backfill migration, fake-LLM unit proof.

Existing stores hold `sem/` atoms minted before tag filing (untagged on the graph axis). The
Plan-01 re-key keys fold/supersede on the CHILD tag, so a NEW correction/fold that REFERENCES a
legacy untagged atom cannot tag-match it until it is backfilled. `backfill_tags` files a
validated child (+bucket) tag onto every active untagged `sem/` atom — append-only
(provenance='backfill'), supersede-not-overwrite (facts byte-untouched), backup-then-write,
idempotent, bounded revert. These four tests lock that migrate-safety contract off the machine
(no vLLM; the classifier axis itself is covered by test_tag_graph)."""
import sqlite3
import sys
from pathlib import Path

import pytest

from localharness.memory.sqlite import MemoryStore
from localharness.memory.tag_classify import _BUCKET_MARKER, _CHILD_MARKER

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import migrate_tag_backfill as mtb  # noqa: E402


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(agent_id="backfill-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


class _ClassifierLLM:
    """Prompt-aware fake (mirrors test_tag_graph / test_tag_grouping): a fixed bucket for the
    bucket pick, a fixed child for the child pick, keyed on the classify prompt markers."""

    def __init__(self, bucket: str = "project", child: str = "ops"):
        self.bucket, self.child = bucket, child

    async def complete(self, prompt: str) -> str:
        if _BUCKET_MARKER in prompt:
            return self.bucket
        if _CHILD_MARKER in prompt:
            return self.child
        return ""


async def _seed_untagged(store, key, value):
    """One active `sem/` fact atom with ZERO atom_tags rows — a pre-migration legacy atom."""
    return await store.store_fact(key=key, value=value, tags=["sem", "pending_consolidation"],
                                  confidence=0.65, provenance="seed", node_kind="fact")


async def _count_backfill_since(store, since_ts):
    async with store._db.execute(
        "SELECT COUNT(*) FROM atom_tags WHERE provenance='backfill' AND ts >= ?", (since_ts,)
    ) as cur:
        return (await cur.fetchone())[0]


@pytest.mark.asyncio
async def test_backfill_idempotent(store):
    """First pass files a child (+bucket) on every untagged sem/ atom; a SECOND pass tags 0 —
    an atom already carrying an edge-eligible child tag is excluded from the target set."""
    await _seed_untagged(store, "sem/gpu-ops/aaaa1111", "vLLM server listens on port 8081")
    await _seed_untagged(store, "sem/subagents/dddd4444", "subagents are read-only by default")
    llm = _ClassifierLLM(bucket="project", child="ops")

    r1 = await mtb.backfill_tags(store, llm, backup=False)
    assert r1["tagged"] == 2 and r1["processed"] == 2          # each untagged atom got a child tag

    r2 = await mtb.backfill_tags(store, llm, backup=False)
    assert r2["tagged"] == 0 and r2["processed"] == 0          # idempotent: nothing left untagged


@pytest.mark.asyncio
async def test_backfill_backup_written(store):
    """backup=True writes an on-disk `<db>.backup-pre-tagbackfill-<ts>` BEFORE any atom_tags
    write, and its bytes equal the pre-migration db — a faithful, restorable snapshot (proven by
    reopening it: it holds the seeded atoms and zero backfill rows)."""
    await _seed_untagged(store, "sem/gpu-ops/aaaa1111", "vLLM server listens on port 8081")
    await _seed_untagged(store, "sem/kyoto-trip/bbbb2222", "one-week Kyoto trip in November")
    # Fold the WAL into the main db so the pre-image is complete and byte-stable.
    async with store._db.execute("PRAGMA wal_checkpoint(TRUNCATE)") as cur:
        await cur.fetchall()
    pre = Path(store._db_path).read_bytes()

    report = await mtb.backfill_tags(store, _ClassifierLLM(bucket="project", child="ops"), backup=True)

    assert report["backup"] and "backup-pre-tagbackfill" in report["backup"]
    assert Path(report["backup"]).exists()
    assert Path(report["backup"]).read_bytes() == pre          # copied before the first write
    # The snapshot is a REAL restore point, not an empty shell:
    con = sqlite3.connect(report["backup"])
    try:
        assert con.execute("SELECT COUNT(*) FROM facts WHERE key LIKE 'sem/%'").fetchone()[0] == 2
        assert con.execute("SELECT COUNT(*) FROM atom_tags WHERE provenance='backfill'").fetchone()[0] == 0
    finally:
        con.close()


@pytest.mark.asyncio
async def test_pre_migration_row_readable(store):
    """The migration ONLY adds atom_tags rows — a fact row is byte-untouched (key/value/status/id
    unchanged) and fully queryable before and after; the append-only child tag is the only delta."""
    f = await _seed_untagged(store, "sem/markets/cccc3333", "user follows HBM memory makers")
    before = await store.get_fact(f.key)

    await mtb.backfill_tags(store, _ClassifierLLM(bucket="personal", child="preferences"), backup=False)

    after = await store.get_fact(f.key)
    assert (after.key, after.value, after.status, after.id) == \
           (before.key, before.value, before.status, before.id)
    assert "preferences" in {t.name for t in await store.tags_for_atom(f.id)}   # append-only add


@pytest.mark.asyncio
async def test_revert_bounded(store):
    """revert_backfill deletes EXACTLY the provenance='backfill' rows added at/after since_ts and
    nothing else: a pre-existing 'mint' child tag survives, an OLDER backfill row (ts < since_ts,
    a prior migration) survives, and this run's fresh atoms return to fully untagged."""
    ops = await store.get_tag("ops")
    travel = await store.get_tag("travel")

    # A) a mint child-tag row that must survive (its atom already has a child -> not a target)
    mint_atom = await _seed_untagged(store, "sem/health/eeee5555", "training for a 10k in September")
    await store.add_atom_tag(mint_atom.id, ops.id, "mint")
    # B) an OLDER backfill row from a prior migration (ts=1000, well before this run's start_ts)
    old_atom = await _seed_untagged(store, "sem/travel/ffff6666", "JR pass not worth it for one Tokyo day")
    await store._db.execute(
        "INSERT INTO atom_tags (atom_id, tag_id, provenance, ts) VALUES (?, ?, 'backfill', 1000)",
        (old_atom.id, travel.id))
    await store._db.commit()
    # C) two fresh untagged atoms this run must tag (and revert must undo)
    fresh1 = await _seed_untagged(store, "sem/gpu-ops/1111aaaa", "KV cache spills to CPU stall")
    await _seed_untagged(store, "sem/gpu-ops/2222bbbb", "Q4_K_M is the best quant for a 27B model")

    report = await mtb.backfill_tags(store, _ClassifierLLM(bucket="project", child="ops"), backup=False)
    assert report["tagged"] == 2 and report["processed"] == 2       # only the two fresh atoms
    assert await _count_backfill_since(store, report["start_ts"]) == 4   # bucket+child x2 fresh

    deleted = await mtb.revert_backfill(store, report["start_ts"])
    assert deleted == 4
    assert await _count_backfill_since(store, report["start_ts"]) == 0   # this run's rows gone
    assert {t.name for t in await store.tags_for_atom(mint_atom.id)} == {"ops"}    # mint intact
    assert {t.name for t in await store.tags_for_atom(old_atom.id)} == {"travel"}  # old backfill intact
    assert await store.tags_for_atom(fresh1.id) == []                              # fresh atom restored

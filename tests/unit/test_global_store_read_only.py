"""The recall router is structurally READ-ONLY toward the store it does not own (v0.13 blocker B1).

`recall_scope: global` points a PROJECT session's reads at the machine-global memory. The released
claim is absolute — "no value of this knob makes a project session write global memory" — and it
was false in two places at once:

* `touch_staged` / `record_activation_trace` routed through `_read_store()`, so a `memory_search`
  in a scope:global project session bumped staged read-counters and appended activation traces in
  ANOTHER project's database (the reviewer's runtime repro: `access_count_staged` 0 -> 1, traces
  0 -> 1);
* `ensure_global()` opened the twin with the OWNER's `open()`, which performs the Phase-33.1 legacy
  `default` -> `orchestrator` directory rename and seeds the tag spine — writes made against a tree
  this session merely reads.

The proof here is ROW-LEVEL, not file-level: SQLite in WAL mode leaves `memory.db`'s bytes
identical while the rows change (that is exactly how the leak survived MEMS-03's byte proof), so
every snapshot below reads the tables back through a fresh connection.

The honest remainder, asserted rather than hidden: a reader open still mkdirs the agent directory,
creates `memory.db` when absent and applies pending SCHEMA migrations — none of which is separable
from being able to query at all. What it leaves alone is the row-level state a session can observe:
facts, tags, activation traces and staged counters.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from localharness.memory.router import ORIGIN_GLOBAL, ORIGIN_WORKSPACE, SCOPE_BOTH, RecallRouter
from localharness.memory.sqlite import FactQuery, MemoryStore

AGENT = "orchestrator"          # the root name, so the legacy-rename migration is in play
GLOBAL_KEY = "org_rule"
GLOBAL_VALUE = "the machine-global rule"


def make_store(base: Path, agent: str = AGENT) -> MemoryStore:
    return MemoryStore(
        agent_id=agent, division_id="default", org_id="default",
        base_dir=str(base), global_base_dir=str(base),
    )


def db_path(base: Path, agent: str = AGENT) -> Path:
    return base / "agents" / agent / "memory.db"


def row_state(db: Path) -> dict[str, list[tuple]]:
    """Every row of every table, as data. A dict comparison names the table that moved."""
    assert db.exists(), f"no database at {db}"
    con = sqlite3.connect(str(db))
    try:
        tables = [
            r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        # Sorted by repr, not by rowid: some of these tables are FTS shadow tables with no
        # rowid at all, and the comparison only needs to be order-independent.
        return {
            t: sorted(con.execute(f'SELECT * FROM "{t}"').fetchall(), key=repr) for t in tables
        }
    finally:
        con.close()


def staged(db: Path, key: str) -> int:
    """`access_count_staged` is a COLUMN, not a `Fact` field (it is folded into `access_count`
    at consolidation), so the only way to read it is the way the leak was found: in SQL."""
    con = sqlite3.connect(str(db))
    try:
        row = con.execute("SELECT access_count_staged FROM facts WHERE key = ?", (key,)).fetchone()
    finally:
        con.close()
    assert row is not None, f"no fact keyed {key!r} in {db}"
    return row[0]


@pytest.fixture
async def seeded_global(tmp_path: Path) -> Path:
    """A real machine-global memory with real content, closed before the session exists."""
    base = tmp_path / "home" / ".localharness"
    s = make_store(base)
    await s.open()
    await s.store_fact(key=GLOBAL_KEY, value=GLOBAL_VALUE, tags=["rules"], confidence=0.9)
    await s.close()
    return base


@pytest.fixture
async def primary(tmp_path: Path) -> MemoryStore:
    base = tmp_path / "proj" / ".localharness"
    s = make_store(base)
    await s.open()
    await s.store_fact(key="ws_rule", value="this project's own rule", tags=["rules"],
                       confidence=0.9)
    yield s
    await s.close()


async def _a_memory_search_turn(router: RecallRouter) -> list:
    """Exactly what `MemorySearchTool` does on a model-issued search: the read, then the two
    enrichment writes it looks up with `getattr` (memory_tools.py)."""
    hits = await router.query_facts(FactQuery(text="rule", min_confidence=0.0, limit=5))
    await router.touch_staged([f.key for f in hits])
    await router.record_activation_trace(
        stimulus="what is the rule?", fired_ids=[f.id for f in hits],
        injected_ids=[f.id for f in hits], session_id="ws-session-1", source="memory_search",
    )
    return hits


# --------------------------------------------------------------- the leak itself


async def test_a_scope_global_project_session_leaves_the_global_rows_untouched(
    primary: MemoryStore, seeded_global: Path
) -> None:
    """The reviewer's repro, as a test. A workspace session reading global memory writes nothing
    into it — staged counters, activation traces, tags and fact rows all identical."""
    before = row_state(db_path(seeded_global))

    router = RecallRouter(primary, make_store(seeded_global), scope=ORIGIN_GLOBAL)
    hits = await _a_memory_search_turn(router)
    await router.get_fact(GLOBAL_KEY)
    await router.load_context(index_mode=True, max_session_history=0)
    await router.close()

    # The reads really happened — otherwise "nothing changed" passes for the wrong reason.
    assert [f.key for f in hits] == [GLOBAL_KEY], f"the global store was never read: {hits}"
    after = row_state(db_path(seeded_global))
    assert after == before, "a read-only session mutated the machine-global store"


async def test_scope_both_leaves_the_global_rows_untouched(
    primary: MemoryStore, seeded_global: Path
) -> None:
    """`both` reads BOTH stores every turn; only one of them belongs to this session."""
    before = row_state(db_path(seeded_global))

    router = RecallRouter(primary, make_store(seeded_global), scope=SCOPE_BOTH)
    ctx = await router.load_context(index_mode=True, max_session_history=0)
    hits = await _a_memory_search_turn(router)
    await router.close()

    assert GLOBAL_VALUE in ctx.agent_memory_md, "the global block never rendered"
    assert {f.key for f in hits} == {"ws_rule", GLOBAL_KEY}, f"both stores were not read: {hits}"
    assert row_state(db_path(seeded_global)) == before, "a merged read mutated the global store"


async def test_the_primary_still_gets_its_enrichment_writes(
    primary: MemoryStore, seeded_global: Path
) -> None:
    """The other direction, so the no-op above cannot pass by disabling the feature: the store
    this session OWNS still records the staged bump and the activation trace."""
    router = RecallRouter(primary, make_store(seeded_global), scope=ORIGIN_WORKSPACE)

    await _a_memory_search_turn(router)
    await router.close()

    assert staged(db_path(primary.base_dir), "ws_rule") == 1, \
        "the primary's staged counter never moved"
    assert [t.source for t in await primary.recent_activation_traces()] == ["memory_search"]


async def test_both_mode_still_stages_on_the_primary(
    primary: MemoryStore, seeded_global: Path
) -> None:
    """`both` writes its staged bumps to the primary — the store that owns the row. Only the
    activation trace is skipped there (its id list spans two databases; Pitfall 7)."""
    router = RecallRouter(primary, make_store(seeded_global), scope=SCOPE_BOTH)

    await _a_memory_search_turn(router)
    await router.close()

    assert staged(db_path(primary.base_dir), "ws_rule") == 1, \
        "both-mode dropped the primary's staged bump"


# --------------------------------------------------------------- the open path


async def test_a_reader_open_does_not_seed_the_tag_spine(
    primary: MemoryStore, tmp_path: Path
) -> None:
    """`open()` seeds the tag hierarchy — an INSERT per seeded bucket and child. Seeding a
    database this session does not own is the store owner's write, not the reader's.

    The honest remainder is asserted too: the database FILE is created (a read cannot happen
    through a missing file), it is simply left with no rows of ours in it.
    """
    virgin = tmp_path / "fresh-home" / ".localharness"
    router = RecallRouter(primary, make_store(virgin), scope=ORIGIN_GLOBAL)

    assert await router.query_facts(FactQuery(text="anything", min_confidence=0.0)) == []
    await router.close()

    assert db_path(virgin).exists(), "the reader open never connected at all"
    state = row_state(db_path(virgin))
    assert state["tags"] == [], f"a reader open seeded the tag spine: {state['tags'][:3]}"
    assert state["facts"] == []


async def test_a_reader_open_neither_adopts_nor_orphans_the_legacy_default_tree(
    primary: MemoryStore, tmp_path: Path
) -> None:
    """Phase 33.1 adopts a pre-rename `agents/default/` tree the first time the store opens as
    `orchestrator` — a RENAME of another install's memory directory. A session that only reads
    that tree does not get to make that call.

    But skipping the rename is not free, and this is the trap the fix had to walk around: the
    open still mkdirs `agents/orchestrator/`, and the adoption REFUSES whenever the destination
    exists. A reader that skipped the rename and created the directory anyway would orphan that
    tree permanently — worse than the write it was avoiding. So the reader refuses to open at
    all in that state, and the owner's next open still adopts. Both directions asserted.
    """
    from localharness.memory.sqlite import LegacyStoreAwaitingAdoption

    base = tmp_path / "legacy-home" / ".localharness"
    legacy = make_store(base, agent="default")
    await legacy.open()
    await legacy.store_fact(key="legacy_rule", value="from before the rename", confidence=0.9)
    await legacy.close()

    router = RecallRouter(primary, make_store(base), scope=ORIGIN_GLOBAL)
    with pytest.raises(LegacyStoreAwaitingAdoption):
        await router.query_facts(FactQuery(text="rule", min_confidence=0.0))
    await router.close()

    assert (base / "agents" / "default").is_dir(), "a reader open renamed the legacy tree"
    assert not (base / "agents" / AGENT).exists(), \
        "the reader created the directory that makes the owner's adoption refuse forever"

    # And the owner's own open still adopts it — the skip is scoped to readers, not removed.
    owner = make_store(base)
    await owner.open()
    await owner.close()
    assert not (base / "agents" / "default").exists(), "the owner open stopped adopting"
    assert (base / "agents" / AGENT).is_dir()

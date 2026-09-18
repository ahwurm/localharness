"""Salience + cold archive — rung 1 of the memory forgetting redesign.

What is under test, in the order it matters:

1. **The line is read out of the store, never configured.** It is the minimum salience
   over the facts the store can PROVE were worth keeping (ever recalled, or the owner's
   own hand). No anchors -> no line -> nothing archives (cold start).
2. **The owner's hand is a hard floor**, independent of whichever line rule is in force.
3. **Archive is a MOVE, not a delete**: the row leaves `facts` (and the FTS haystack with
   it), and restore brings it back byte-identical — full physical row, every column.
4. **The automatic step is OFF by default**; the explicit CLI verb works regardless, and
   `--dry-run` is provably inert.

Mutation-tested (documented in the report, re-run by hand from byte snapshots):
`archive_line` min->max, the pin filter dropped from `select_archivable`, and
`vacuum_warranted`'s `moved > kept` flipped — each reddens a named test below.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config.models import MemoryArchivalConfig, MemoryConsolidationConfig
from localharness.memory.consolidation import (
    SKIP_NOT_ACTIVE,
    SKIP_OWNER_TOUCHED,
    SKIP_RECALLED,
    SKIP_UNFOLDED_READS,
    SKIP_UNKNOWN,
    SKIP_UNPARSEABLE,
    ConsolidationPass,
    archive_dormant_facts,
    archive_listed_facts,
    parse_archive_list,
)
from localharness.memory.salience import (
    CONFIDENCE_LOGIT_MAX,
    CONFIDENCE_LOGIT_MIN,
    archive_line,
    format_pass_line,
    is_anchor,
    is_pinned,
    need_now,
    score_facts,
    select_archivable,
    truth_log_odds,
    vacuum_warranted,
)
from localharness.memory.sqlite import (
    ARCHIVE_STAMP_PREFIX,
    ARCHIVE_SURFACE_CONSENSUS_LIST,
    ARCHIVE_SURFACE_FLOOR_LINE,
    USER_EDIT_PROVENANCE_PREFIX,
    FactQuery,
    MemoryStore,
    _base_activation,
)

runner = CliRunner()
DAY = 86400


def make_store(tmp_path: Path, agent: str = "orchestrator") -> MemoryStore:
    return MemoryStore(agent_id=agent, division_id="default", org_id="default",
                       base_dir=str(tmp_path))


async def _age(store: MemoryStore, key: str, *, days: int, access_count: int = 0) -> int:
    """Backdate a fact and set its folded read counter — the two inputs `need` reads.
    Returns the fact id."""
    fact = await store.get_fact(key)
    assert fact is not None
    then = int(time.time()) - days * DAY
    await store._db.execute(
        "UPDATE facts SET updated_at = ?, created_at = ?, last_accessed_at = ?, "
        "access_count = ? WHERE id = ?",
        (then, then, then if access_count else None, access_count, fact.id),
    )
    await store._db.commit()
    return fact.id


async def _seed_measured_shape(store: MemoryStore) -> dict[str, int]:
    """A miniature of the MEASURED live store (2026-09-18 dry run): a couple of anchors
    carrying real evidence, and a pile of old never-recalled mined rows."""
    ids: dict[str, int] = {}
    await store.store_fact("profile/city", "the owner lives in Denver", confidence=0.9,
                           source="remember")
    ids["profile/city"] = await _age(store, "profile/city", days=40)

    await store.store_fact("learned/vllm/resolved_error", "restart vllm after a GPU lock",
                           tags=["tier:resolved_error"], confidence=0.8, source="consolidation")
    ids["learned/vllm/resolved_error"] = await _age(
        store, "learned/vllm/resolved_error", days=20, access_count=5)

    for i in range(6):
        key = f"mined/dead-{i}"
        await store.store_fact(key, f"a mined observation number {i} about the harness",
                               tags=["sem"], confidence=0.65, source="transcript_mining")
        ids[key] = await _age(store, key, days=30 + i)
    return ids


# ---------------------------------------------------------------------------
# 1. The three terms, and that `need` is the store's OWN formula
# ---------------------------------------------------------------------------

async def test_need_is_the_stores_own_activation_not_a_second_curve(tmp_path: Path):
    """One forgetting curve in this codebase. `need` must be `_base_activation` itself —
    a re-derivation here would be a second, silently-diverging decay law."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("k/a", "v", confidence=0.65)
        await _age(store, "k/a", days=9, access_count=3)
        fact = await store.get_fact("k/a")
        now = int(time.time())
        assert need_now(fact, now) == _base_activation(
            fact.access_count, fact.last_accessed_at, fact.updated_at, now,
            day_granularity=True,
        )
    finally:
        await store.close()


def test_truth_is_log_odds_and_the_clamp_only_guards_the_poles():
    import math
    assert truth_log_odds(0.5) == 0.0
    assert truth_log_odds(0.65) == math.log(0.65 / 0.35)
    # The clamp is a DOMAIN guard: the poles must stay finite, nothing in between moves.
    assert truth_log_odds(0.0) == truth_log_odds(CONFIDENCE_LOGIT_MIN)
    assert truth_log_odds(1.0) == truth_log_odds(CONFIDENCE_LOGIT_MAX)
    assert math.isfinite(truth_log_odds(0.0)) and math.isfinite(truth_log_odds(1.0))
    # The live store's whole measured range (0.5-0.9) is untouched by it.
    for c in (0.5, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9):
        assert truth_log_odds(c) == math.log(c / (1 - c))


async def test_salience_sums_need_truth_and_stakes(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("tier/schema", "a chapter", tags=["tier:schema"],
                               confidence=0.8, source="consolidation")
        fact = await store.get_fact("tier/schema")
        now = int(time.time())
        (s,) = score_facts([fact], now)
        assert s.s == s.need + s.truth + s.stakes
        assert s.stakes == fact.importance > 0.0     # stakes is the stored column, as-is
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# 2. The line: floor of the anchors, computed every time
# ---------------------------------------------------------------------------

async def test_line_is_the_floor_of_the_anchor_set(tmp_path: Path):
    """MUTATION TARGET: flip `min` to `max` in archive_line and this reddens — with a max
    line, proven-useful facts sit below their own line and would archive."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await _seed_measured_shape(store)
        facts = await store.query_facts(FactQuery(limit=500, min_confidence=0.0))
        scored = score_facts(facts, int(time.time()))
        anchors = [s for s in scored if s.anchor]
        assert len(anchors) == 2                       # 1 remembered + 1 recalled
        line = archive_line(scored)
        assert line == min(a.s for a in anchors)
        # The property that matters: NO anchor is ever below the line it defines.
        assert all(a.s >= line for a in anchors)
        below = select_archivable(scored, line)
        assert {c.key for c in below} == {f"mined/dead-{i}" for i in range(6)}
    finally:
        await store.close()


async def test_cold_start_has_no_line_and_archives_nothing(tmp_path: Path):
    """Nothing recalled, nothing owner-touched -> no evidence -> no line -> no-op. The
    store must not invent a threshold on a store it knows nothing about."""
    store = make_store(tmp_path)
    await store.open()
    try:
        for i in range(4):
            await store.store_fact(f"mined/cold-{i}", f"observation {i}", confidence=0.65,
                                   source="transcript_mining")
            await _age(store, f"mined/cold-{i}", days=90)
        facts = await store.query_facts(FactQuery(limit=500, min_confidence=0.0))
        scored = score_facts(facts, int(time.time()))
        assert archive_line(scored) is None
        assert select_archivable(scored, None) == []
        run = await archive_dormant_facts(store)
        assert run.line is None and run.moved == 0
        assert await store.count_archived() == 0
    finally:
        await store.close()


async def test_a_junk_recall_only_lowers_the_line(tmp_path: Path):
    """The failure direction is safe by construction: anything that becomes an anchor can
    only drag the floor DOWN, i.e. archive less. There is no input that archives more."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await _seed_measured_shape(store)
        facts = await store.query_facts(FactQuery(limit=500, min_confidence=0.0))
        before = archive_line(score_facts(facts, int(time.time())))
        await _age(store, "mined/dead-5", days=35, access_count=1)   # junk gets recalled once
        facts = await store.query_facts(FactQuery(limit=500, min_confidence=0.0))
        after = archive_line(score_facts(facts, int(time.time())))
        assert after < before
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# 3. Pins — the owner's hand is a hard floor
# ---------------------------------------------------------------------------

async def test_the_owners_hand_is_recognised_in_all_three_shapes(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("p/remember", "v", source="remember")
        await store.store_fact("p/edited", "v", source="transcript_mining",
                               provenance=f"{USER_EDIT_PROVENANCE_PREFIX}1758000000;cli")
        await store.store_fact("p/schema", "v", tags=["tier:schema"], source="consolidation")
        await store.store_fact("p/mined", "v", source="transcript_mining")
        for key, expected in (("p/remember", True), ("p/edited", True),
                              ("p/schema", True), ("p/mined", False)):
            fact = await store.get_fact(key)
            assert is_pinned(fact) is expected, key
            assert is_anchor(fact) is expected, key   # pins anchor the line too
    finally:
        await store.close()


async def test_pins_survive_a_line_drawn_above_them(tmp_path: Path):
    """MUTATION TARGET: drop `and not s.pinned` from select_archivable and this reddens.

    Under today's anchor-floor line this filter is redundant (a pin IS an anchor, so it
    cannot sit below the anchors' minimum) — which is exactly why it is tested against a
    line the caller supplies. Rung 4's calibrated cost-ratio line is not derived from the
    anchor set, and the owner's hand must outrank whichever line is in force.
    """
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("p/remember", "the owner said so", source="remember")
        await _age(store, "p/remember", days=365)          # ancient and never recalled
        await store.store_fact("mined/x", "a mined line", confidence=0.65,
                               source="transcript_mining")
        facts = await store.query_facts(FactQuery(limit=500, min_confidence=0.0))
        scored = score_facts(facts, int(time.time()))
        pin = next(s for s in scored if s.key == "p/remember")
        forced_line = pin.s + 1.0                          # a line ABOVE the owner's fact
        selected = select_archivable(scored, forced_line)
        assert pin.key not in {s.key for s in selected}
        assert "mined/x" in {s.key for s in selected}      # the rest of the sweep still runs
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# 4. The move: transactional, verified, FTS-correct, byte-identical on the way back
# ---------------------------------------------------------------------------

async def _full_row(store: MemoryStore, fact_id: int):
    cols = ", ".join(store._ARCHIVE_ROW_COLS)
    async with store._db.execute(f"SELECT {cols} FROM facts WHERE id = ?", (fact_id,)) as cur:
        row = await cur.fetchone()
    return tuple(row) if row is not None else None


async def test_archive_then_restore_round_trips_every_column(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("ops/vllm-port", "the vllm server listens on port 8081",
                               tags=["ops", "sem"], confidence=0.65,
                               source="transcript_mining", provenance="sess-7")
        fact_id = await _age(store, "ops/vllm-port", days=30)
        before = await _full_row(store, fact_id)

        assert await store.archive_fact(fact_id, surface=ARCHIVE_SURFACE_FLOOR_LINE,
                                        s_at_archive=-1.4, line_at_archive=-0.56) is True
        assert await _full_row(store, fact_id) is None       # gone from the hot table
        assert await store.count_archived() == 1

        assert await store.restore_fact(fact_id) is True
        assert await _full_row(store, fact_id) == before      # byte-identical, all 22 columns
        assert await store.count_archived() == 0
    finally:
        await store.close()


async def test_archived_facts_leave_the_search_index_and_come_back_with_restore(tmp_path: Path):
    """The whole point of a MOVE: recall stops seeing it. A demotion would not do this —
    the row would still be in the FTS haystack, still surfacing on every search."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("ops/vllm-port", "the vllm server listens on port 8081",
                               confidence=0.65, source="transcript_mining")
        fact_id = await _age(store, "ops/vllm-port", days=30)

        async def search():
            return [f.key for f in await store.query_facts(
                FactQuery(text="vllm", min_confidence=0.0, limit=50))]

        assert "ops/vllm-port" in await search()
        await store.archive_fact(fact_id, surface=ARCHIVE_SURFACE_FLOOR_LINE, s_at_archive=-1.0,
                                 line_at_archive=-0.5)
        assert await search() == []                            # out of the haystack
        assert await store.get_fact("ops/vllm-port") is None   # and off the direct path
        assert [f.key for f in await store.list_archived()] == ["ops/vllm-port"]

        await store.restore_fact(fact_id)
        assert "ops/vllm-port" in await search()                # and back in it
    finally:
        await store.close()


async def test_archive_refuses_a_row_with_unfolded_reads(tmp_path: Path):
    """`access_count_staged > 0` means the fact was recalled since the last fold — freshly
    used, whatever its stale folded counters say. The store refuses the move rather than
    trusting the scorer's inputs."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("k/recent", "v", confidence=0.65)
        fact_id = await _age(store, "k/recent", days=60)
        await store._db.execute(
            "UPDATE facts SET access_count_staged = 2 WHERE id = ?", (fact_id,))
        await store._db.commit()
        assert await store.archive_fact(fact_id, surface=ARCHIVE_SURFACE_FLOOR_LINE,
                                        s_at_archive=-2.0, line_at_archive=-0.5) is False
        assert await store.get_fact("k/recent") is not None
        assert await store.count_archived() == 0
    finally:
        await store.close()


async def test_archive_never_touches_superseded_rows(tmp_path: Path):
    """Superseded-version compaction is rung 2. Rung 1 moves ACTIVE rows only."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("k/one", "first value", confidence=0.65)
        old_id = (await store.get_fact("k/one")).id
        await store.store_fact("k/one", "second value", confidence=0.65)
        assert await store.archive_fact(old_id, surface=ARCHIVE_SURFACE_FLOOR_LINE,
                                        s_at_archive=-9.0, line_at_archive=0.0) is False
        assert (await store.get_fact_by_id(old_id)).status == "superseded"
        assert await store.count_archived() == 0
    finally:
        await store.close()


async def test_restore_refuses_when_a_live_fact_holds_the_name(tmp_path: Path):
    """The live row wins; the archived copy stays safe in the archive instead of being
    forced over it (and the failed restore leaves BOTH tables exactly as they were)."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("k/dup", "archived value", confidence=0.65)
        fact_id = await _age(store, "k/dup", days=30)
        await store.archive_fact(fact_id, surface=ARCHIVE_SURFACE_FLOOR_LINE, s_at_archive=-1.0,
                                 line_at_archive=-0.5)
        await store.store_fact("k/dup", "a newer live value", confidence=0.65)

        assert await store.restore_fact(fact_id) is False
        assert (await store.get_fact("k/dup")).value == "a newer live value"
        assert await store.count_archived() == 1
    finally:
        await store.close()


async def test_restore_of_an_unknown_id_is_a_clean_no(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        assert await store.restore_fact(4242) is False
    finally:
        await store.close()


async def test_the_archive_mirror_is_checked_against_the_live_facts_columns(tmp_path: Path):
    """A migration that adds a column to `facts` and forgets `facts_archive` would make
    archival lossy. It fails LOUDLY instead."""
    from localharness.memory.errors import MemoryCorruptionError

    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("k/a", "v", confidence=0.65)
        fact_id = (await store.get_fact("k/a")).id
        await store._db.execute("ALTER TABLE facts ADD COLUMN future_column TEXT")
        await store._db.commit()
        store._archive_cols_checked = False
        try:
            await store.archive_fact(fact_id, surface=ARCHIVE_SURFACE_FLOOR_LINE, s_at_archive=-1.0,
                                     line_at_archive=0.0)
            raise AssertionError("archival must refuse a mirror that no longer matches")
        except MemoryCorruptionError as exc:
            assert "facts_archive" in str(exc)
    finally:
        await store.close()


def test_vacuum_fires_only_when_more_rows_left_than_stayed():
    """MUTATION TARGET: flip the comparison in vacuum_warranted and this reddens. The bar
    is the operation's own cost shape (a rewrite copies what remains to reclaim what
    left), so there is no tuned fraction to drift."""
    assert vacuum_warranted(0, 100) is False
    assert vacuum_warranted(50, 100) is False       # exactly half: copying == reclaiming
    assert vacuum_warranted(51, 100) is True
    assert vacuum_warranted(830, 854) is True       # the measured live shape
    assert vacuum_warranted(5, 1_000_000) is False  # scales with the store, not with taste


# ---------------------------------------------------------------------------
# 5. The run + the consolidation step (OFF by default)
# ---------------------------------------------------------------------------

def _cons_cfg() -> MemoryConsolidationConfig:
    # The LLM steps are off: this is the deterministic core plus the new step.
    return MemoryConsolidationConfig(
        schema_writer_enabled=False, reconcile_enabled=False, mining_enabled=False,
    )


async def test_consolidation_step_is_off_by_default(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await _seed_measured_shape(store)
        assert MemoryArchivalConfig().enabled is False      # the shipped default
        report = await ConsolidationPass(store, _cons_cfg()).run()       # nothing wired
        assert report.archived == 0 and report.archive_line is None
        assert await store.count_archived() == 0

        report = await ConsolidationPass(store, _cons_cfg(),
                                         archival=MemoryArchivalConfig()).run()
        assert report.archived == 0 and report.archive_line is None
        assert await store.count_archived() == 0
        assert (await store.get_fact("mined/dead-0")) is not None
    finally:
        await store.close()


async def test_consolidation_step_archives_when_the_gate_is_on(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await _seed_measured_shape(store)
        report = await ConsolidationPass(
            store, _cons_cfg(), archival=MemoryArchivalConfig(enabled=True)
        ).run()
        assert report.archived == 6
        assert report.archive_line is not None
        assert await store.count_archived() == 6
        assert await store.get_fact("profile/city") is not None            # pin kept
        assert await store.get_fact("learned/vllm/resolved_error") is not None  # anchor kept
        assert await store.get_fact("mined/dead-0") is None                # dormant moved
    finally:
        await store.close()


async def test_the_pass_report_names_the_count_and_the_line(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await _seed_measured_shape(store)
        report = await ConsolidationPass(
            store, _cons_cfg(), archival=MemoryArchivalConfig(enabled=True)
        ).run()
        line = report.archive_report_line()
        assert line == f"archived 6 dormant facts (line S={report.archive_line:.2f})"
    finally:
        await store.close()


def test_the_cold_start_report_line_says_why_it_did_nothing():
    assert format_pass_line(0, None).startswith("archived 0 dormant facts (line S=none")
    assert format_pass_line(812, -0.5597) == "archived 812 dormant facts (line S=-0.56)"


async def test_the_archived_row_records_the_score_and_the_line_it_was_judged_by(tmp_path: Path):
    """Audit trail: every move records WHAT it scored and WHICH line it fell under, in the
    archive's own columns — the fact row itself comes back unchanged."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await _seed_measured_shape(store)
        run = await archive_dormant_facts(store)
        async with store._db.execute(
            "SELECT key, archive_rung, s_at_archive, line_at_archive, archived_at "
            "FROM facts_archive ORDER BY s_at_archive"
        ) as cur:
            rows = [tuple(r) for r in await cur.fetchall()]
        assert len(rows) == run.moved == 6
        for key, rung, s_at, line_at, when in rows:
            assert key.startswith("mined/dead-")
            assert rung == f"{ARCHIVE_STAMP_PREFIX}{when};{ARCHIVE_SURFACE_FLOOR_LINE}"
            assert s_at < line_at == run.line
            assert when > 0
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# 6. The CLI
# ---------------------------------------------------------------------------

def _run_seed(tmp_path: Path) -> dict[str, int]:
    async def go():
        store = make_store(tmp_path)
        await store.open()
        try:
            return await _seed_measured_shape(store)
        finally:
            await store.close()
    return asyncio.run(go())


def _all_rows(tmp_path: Path) -> tuple[list, list]:
    """Every column of every fact row, plus the whole archive. A byte-compare of the file
    is the WRONG instrument here and was measured to be: merely OPENING a store rewrites
    the file (the idempotent tag-seed transaction bumps the change counter), so a byte
    diff would report a mutation no verb made. Content is the claim; content is what is
    checked."""
    async def go():
        store = make_store(tmp_path)
        await store.open()
        try:
            cols = ", ".join(store._ARCHIVE_ROW_COLS)
            async with store._db.execute(f"SELECT {cols} FROM facts ORDER BY id") as cur:
                facts = [tuple(r) for r in await cur.fetchall()]
            async with store._db.execute(
                f"SELECT {cols} FROM facts_archive ORDER BY id") as cur:
                archived = [tuple(r) for r in await cur.fetchall()]
            return facts, archived
        finally:
            await store.close()
    return asyncio.run(go())


def test_dry_run_reports_everything_and_moves_nothing(tmp_path: Path):
    _run_seed(tmp_path)
    before = _all_rows(tmp_path)
    out = runner.invoke(app, ["memory", "archive", "--dry-run",
                              "--config-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert "would archive: 6 fact(s)" in out.output
    assert "line S=" in out.output
    assert "by source:" in out.output and "transcript_mining" in out.output
    assert "mined/dead-0" in out.output                  # the per-key preview
    assert "dry run — nothing moved." in out.output
    assert _all_rows(tmp_path) == before                 # every row, every column: inert
    assert before[1] == []                               # and the archive stayed empty


def test_dry_run_works_while_the_automatic_gate_is_off(tmp_path: Path):
    """The gate governs the AUTOMATIC step. Asking explicitly always answers — that is how
    the owner reads a store before ever turning the step on."""
    _run_seed(tmp_path)
    assert MemoryArchivalConfig().enabled is False
    out = runner.invoke(app, ["memory", "archive", "--dry-run",
                              "--config-dir", str(tmp_path)])
    assert out.exit_code == 0 and "would archive: 6 fact(s)" in out.output


def test_archive_then_list_archived_then_restore_from_the_cli(tmp_path: Path):
    ids = _run_seed(tmp_path)
    out = runner.invoke(app, ["memory", "archive", "--config-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert "archived: 6 fact(s)" in out.output

    out = runner.invoke(app, ["memory", "list", "--archived", "--config-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert "archived: 6 fact(s)" in out.output
    assert "mined/dead-0" in out.output
    assert f"[{ids['mined/dead-0']}]" in out.output       # the id restore takes

    out = runner.invoke(app, ["memory", "list", "--config-dir", str(tmp_path)])
    assert "mined/dead-0" not in out.output               # off the live listing
    assert "profile/city" in out.output                   # the pin stayed

    out = runner.invoke(app, ["memory", "restore", str(ids["mined/dead-0"]),
                              "--config-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert "restored mined/dead-0" in out.output
    out = runner.invoke(app, ["memory", "list", "--config-dir", str(tmp_path)])
    assert "mined/dead-0" in out.output


def test_restore_of_an_unknown_id_fails_loudly_from_the_cli(tmp_path: Path):
    _run_seed(tmp_path)
    out = runner.invoke(app, ["memory", "restore", "9999", "--config-dir", str(tmp_path)])
    assert out.exit_code == 1
    assert "not in the archive" in out.output


def test_cli_says_so_on_a_cold_store(tmp_path: Path):
    async def go():
        store = make_store(tmp_path)
        await store.open()
        try:
            await store.store_fact("mined/only", "a mined line", confidence=0.65,
                                   source="transcript_mining")
        finally:
            await store.close()
    asyncio.run(go())
    out = runner.invoke(app, ["memory", "archive", "--dry-run", "--config-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert "no line yet" in out.output and "Nothing archived." in out.output


async def test_an_unfolded_read_anchors_the_line_too(tmp_path: Path):
    """MUTATION TARGET: drop `also_anchor=staged_reads` from archive_dormant_facts and this
    reddens. A read is a read whether or not the consolidation fold has moved the counter
    yet; ignoring one leaves the line HIGHER than the evidence warrants, i.e. archives
    MORE — the one direction this design does not accept. (Inside a pass the set is always
    empty: the fold step runs first. From the CLI, mid-session, it is not.)"""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("sem/just-read", "a fact the model recalled this session",
                               confidence=0.65, source="transcript_mining")
        read_id = await _age(store, "sem/just-read", days=50)   # old, folded counter still 0
        await store._db.execute(
            "UPDATE facts SET access_count_staged = 1, last_accessed_staged = ? WHERE id = ?",
            (int(time.time()), read_id))
        await store.store_fact("p/remember", "the owner said so", source="remember")
        await _age(store, "p/remember", days=5)                  # a high-scoring anchor
        for i in range(3):
            await store.store_fact(f"mined/d-{i}", f"dead {i}", confidence=0.65,
                                   source="transcript_mining")
            await _age(store, f"mined/d-{i}", days=20 + i)       # NEWER than the read fact
        await store._db.commit()

        run = await archive_dormant_facts(store, dry_run=True)
        read_s = next(s for s in score_facts(
            [await store.get_fact("sem/just-read")], int(time.time())) if True).s
        assert run.line == read_s          # the unfolded read set the floor
        assert run.candidates == []        # so nothing newer than it is below the line
    finally:
        await store.close()


async def test_the_scheduler_forwards_the_archival_config_to_its_pass(tmp_path: Path):
    """"Wired" means reachable from the thing the harness actually runs. The scheduler is
    what start_cmd constructs; if it drops the archival config on the floor, the step can
    never fire in a real session no matter how green the pass-level tests are."""
    from localharness.memory.consolidation import ConsolidationScheduler

    class _Bus:
        async def publish(self, event):   # the scheduler's status dot; irrelevant here
            return None

    store = make_store(tmp_path)
    await store.open()
    try:
        await _seed_measured_shape(store)
        sched = ConsolidationScheduler(
            store, _Bus(), "orchestrator", _cons_cfg(),
            archival=MemoryArchivalConfig(enabled=True),
        )
        sched.launch()
        await sched._run_task
        assert sched.last_report is not None
        assert sched.last_report.archived == 6
        assert await store.count_archived() == 6
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# 7. List-driven archival — scorer-agnostic plumbing.
#
# The scorer keeps changing (S v1 ranks, it does not calibrate), and the first watched
# live run is driven by an externally computed CONSENSUS of several measured scorers that
# no in-harness formula reproduces. So the harness takes a list of ids and stays out of the
# judging — while every safety rail is re-checked HERE, at execution time, because a list
# is an opinion from outside and outside opinions do not get to move the owner's facts.
# ---------------------------------------------------------------------------

def test_the_list_parser_takes_pipes_comments_and_junk(tmp_path: Path):
    ids, bad = parse_archive_list(
        "# consensus run 2026-09-18, 3 scorers agreeing\n"
        "\n"
        "101|mined/obs-0001|-1.51|3of3\n"       # trailing fields IGNORED (forward-compatible)
        "  202  \n"                              # bare id, whitespace
        "303|\n"
        "101|mined/obs-0001|-1.51|3of3\n"       # a repeat is harmless
        "   # an indented comment\n"
        "not-an-id|whatever\n"                   # garbage: reported, never fatal
        "|404\n"                                 # empty first field: garbage too
    )
    assert ids == [101, 202, 303]                # file order, deduped
    assert [b.line_no for b in bad] == [8, 9]
    assert all(b.reason == SKIP_UNPARSEABLE and b.fact_id is None for b in bad)
    assert bad[0].raw == "not-an-id|whatever"    # the row is quoted back, not just counted


async def _seed_one_of_each(store: MemoryStore) -> dict[str, int]:
    """One dead fact (archivable) plus one of every row the rails must refuse."""
    ids: dict[str, int] = {}
    await store.store_fact("mined/dead", "a mined observation nobody ever used",
                           tags=["sem"], confidence=0.65, source="transcript_mining")
    ids["dead"] = await _age(store, "mined/dead", days=45)

    await store.store_fact("profile/home", "the owner is in Denver", confidence=0.9,
                           source="remember")
    ids["pinned"] = await _age(store, "profile/home", days=45)

    await store.store_fact("learned/vllm/resolved_error", "restart vllm after a GPU lock",
                           tags=["tier:resolved_error"], confidence=0.8, source="write_gate")
    ids["recalled"] = await _age(store, "learned/vllm/resolved_error", days=45, access_count=4)

    await store.store_fact("sem/just-read", "a fact the model recalled this session",
                           confidence=0.65, source="transcript_mining")
    ids["staged"] = await _age(store, "sem/just-read", days=45)
    await store._db.execute("UPDATE facts SET access_count_staged = 1 WHERE id = ?",
                            (ids["staged"],))

    await store.store_fact("k/versioned", "first value", confidence=0.65,
                           source="transcript_mining")
    ids["superseded"] = (await store.get_fact("k/versioned")).id
    await store.store_fact("k/versioned", "second value", confidence=0.65,
                           source="transcript_mining")
    await store._db.commit()
    return ids


async def test_every_rail_refuses_and_the_run_carries_on(tmp_path: Path):
    """MUTATION TARGET: drop the `is_anchor`/access_count rail from archive_listed_facts and
    the recalled-fact assertion reddens. A list that names a fact you actually used does not
    get to move it — and one bad id must never cost the rest of a 10k-row list its run."""
    store = make_store(tmp_path)
    await store.open()
    try:
        ids = await _seed_one_of_each(store)
        listed = [ids["dead"], ids["pinned"], ids["recalled"], ids["staged"],
                  ids["superseded"], 999_999]
        run = await archive_listed_facts(store, listed)

        assert run.moved == 1                                  # the run carried on
        assert [c.fact_id for c in run.candidates] == [ids["dead"]]
        assert await store.get_fact("mined/dead") is None
        by_id = {s.fact_id: s.reason for s in run.skipped}
        assert by_id == {
            ids["pinned"]: SKIP_OWNER_TOUCHED,
            ids["recalled"]: SKIP_RECALLED,
            ids["staged"]: SKIP_UNFOLDED_READS,
            ids["superseded"]: SKIP_NOT_ACTIVE,
            999_999: SKIP_UNKNOWN,
        }
        for key in ("profile/home", "learned/vllm/resolved_error", "sem/just-read"):
            assert await store.get_fact(key) is not None       # every refused row still hot
        assert (await store.get_fact_by_id(ids["superseded"])).status == "superseded"
    finally:
        await store.close()


async def test_the_stores_own_verb_refuses_even_when_the_list_insists(tmp_path: Path):
    """Defence in depth: the rails are re-checked by the MOVE itself, so a fact that goes
    live between the scan and the move is still refused — the list cannot outrun it."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("sem/x", "v", confidence=0.65, source="transcript_mining")
        fact_id = await _age(store, "sem/x", days=45)
        await store._db.execute("UPDATE facts SET access_count_staged = 1 WHERE id = ?",
                                (fact_id,))
        await store._db.commit()
        # Straight at the store verb, bypassing the runner's scan entirely.
        assert await store.archive_fact(fact_id, surface=ARCHIVE_SURFACE_CONSENSUS_LIST,
                                        s_at_archive=-9.0, line_at_archive=None) is False
        assert await store.get_fact("sem/x") is not None
    finally:
        await store.close()


async def test_list_archived_rows_carry_the_consensus_stamp_and_no_line(tmp_path: Path):
    """The stamp says WHICH procedure condemned the row, in the archive's own metadata —
    the fact row is untouched. line_at_archive stays NULL: no line was consulted."""
    store = make_store(tmp_path)
    await store.open()
    try:
        ids = await _seed_one_of_each(store)
        before = await _full_row(store, ids["dead"])
        await archive_listed_facts(store, [ids["dead"]])
        async with store._db.execute(
            "SELECT archive_rung, archived_at, s_at_archive, line_at_archive "
            "FROM facts_archive WHERE id = ?", (ids["dead"],)
        ) as cur:
            stamp, when, s_at, line_at = tuple(await cur.fetchone())
        assert stamp == f"{ARCHIVE_STAMP_PREFIX}{when};{ARCHIVE_SURFACE_CONSENSUS_LIST}"
        assert line_at is None                       # the list was the authority, not a line
        assert s_at < 0                              # S v1 recorded as the local opinion
        # And the round trip is unchanged by any of it.
        assert await store.restore_fact(ids["dead"]) is True
        assert await _full_row(store, ids["dead"]) == before
    finally:
        await store.close()


def _write_list(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "consensus.txt"
    path.write_text("\n".join(rows) + "\n")
    return path


def _seed_one_of_each_sync(tmp_path: Path) -> dict[str, int]:
    async def go():
        store = make_store(tmp_path)
        await store.open()
        try:
            return await _seed_one_of_each(store)
        finally:
            await store.close()
    return asyncio.run(go())


def test_cli_list_dry_run_reports_and_moves_nothing(tmp_path: Path):
    """MUTATION TARGET: ignore the dry_run flag in archive_listed_facts and this reddens."""
    ids = _seed_one_of_each_sync(tmp_path)
    lst = _write_list(tmp_path, [
        "# consensus of 3 measured scorers, 2026-09-18",
        f"{ids['dead']}|mined/dead|-1.5052|3of3",
        f"{ids['recalled']}|learned/vllm/resolved_error|-0.9|2of3",
        "999999|gone/already|-2.0|3of3",
    ])
    before = _all_rows(tmp_path)
    out = runner.invoke(app, ["memory", "archive", "--dry-run", "--from-list", str(lst),
                              "--config-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert "3 id(s), 0 unparseable row(s)" in out.output
    assert "would archive: 1 fact(s)" in out.output
    assert "mined/dead" in out.output
    assert "SKIPPED (2):" in out.output
    assert SKIP_RECALLED in out.output and SKIP_UNKNOWN in out.output
    assert "dry run — nothing moved." in out.output
    assert _all_rows(tmp_path) == before          # every row, every column: inert


def test_cli_list_run_archives_exactly_the_listed_fact(tmp_path: Path):
    ids = _seed_one_of_each_sync(tmp_path)
    lst = _write_list(tmp_path, [f"{ids['dead']}|mined/dead", f"{ids['pinned']}|profile/home"])
    out = runner.invoke(app, ["memory", "archive", "--from-list", str(lst),
                              "--config-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert "archived: 1 fact(s)" in out.output
    assert SKIP_OWNER_TOUCHED in out.output

    out = runner.invoke(app, ["memory", "list", "--config-dir", str(tmp_path)])
    assert "mined/dead" not in out.output
    assert "profile/home" in out.output            # the owner's fact never moved

    out = runner.invoke(app, ["memory", "restore", str(ids["dead"]),
                              "--config-dir", str(tmp_path)])
    assert out.exit_code == 0 and "restored mined/dead" in out.output
    out = runner.invoke(app, ["memory", "list", "--config-dir", str(tmp_path)])
    assert "mined/dead" in out.output


def test_cli_list_reports_junk_rows_instead_of_dying_on_them(tmp_path: Path):
    ids = _seed_one_of_each_sync(tmp_path)
    lst = _write_list(tmp_path, ["# header", f"{ids['dead']}|mined/dead", "not-an-id|junk"])
    out = runner.invoke(app, ["memory", "archive", "--dry-run", "--from-list", str(lst),
                              "--config-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert "1 id(s), 1 unparseable row(s)" in out.output
    assert "line 3: 'not-an-id|junk'" in out.output and SKIP_UNPARSEABLE in out.output
    assert "would archive: 1 fact(s)" in out.output


def test_cli_list_works_while_the_automatic_gate_is_off(tmp_path: Path):
    """The gate governs the AUTOMATIC step only. The first watched live run IS this command,
    with the gate still off — if the gate blocked it, that run could not happen."""
    ids = _seed_one_of_each_sync(tmp_path)
    assert MemoryArchivalConfig().enabled is False
    lst = _write_list(tmp_path, [str(ids["dead"])])
    out = runner.invoke(app, ["memory", "archive", "--from-list", str(lst),
                              "--config-dir", str(tmp_path)])
    assert out.exit_code == 0 and "archived: 1 fact(s)" in out.output


def test_cli_list_missing_file_fails_cleanly(tmp_path: Path):
    _seed_one_of_each_sync(tmp_path)
    out = runner.invoke(app, ["memory", "archive", "--from-list", str(tmp_path / "nope.txt"),
                              "--config-dir", str(tmp_path)])
    assert out.exit_code == 1 and "cannot read" in out.output

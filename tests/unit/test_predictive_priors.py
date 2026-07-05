"""Tests for the COLL-01 collect-only predictive-gate substrate (Phase 34):
pure-SQL per-tool priors + deterministic cold-start-safe surprise math, and the
idempotent recording APIs + the byte-stability invariant of the injected block."""
import statistics
from pathlib import Path

import pytest

from localharness.memory.sqlite import (
    MemoryStore,
    ToolPrior,
    _band_z,
    _tool_error_surprisal,
    compute_quadrant,
    compute_surprise_score,
)


def make_store(tmp_path: Path, bus=None) -> MemoryStore:
    return MemoryStore(
        agent_id="test-agent",
        division_id="test-div",
        org_id="default",
        base_dir=str(tmp_path),
        bus=bus,
    )


async def _seed_observations(store: MemoryStore, tool_name: str, rows: list[dict]) -> None:
    """Raw INSERTs into tool_observations — the table exists after the v4 migration."""
    for i, r in enumerate(rows):
        await store._db.execute(
            "INSERT INTO tool_observations "
            "(agent_id, session_id, tool_name, ts, is_error, output_len, duration_ms, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'live')",
            (store._agent_id, "s", tool_name, r.get("ts", 1000 + i),
             r["is_error"], r.get("output_len"), r.get("duration_ms")),
        )
    await store._db.commit()


# ---------------------------------------------------------------------------
# Scoring math — pure functions (no DB)
# ---------------------------------------------------------------------------

def test_tool_error_surprisal_grades_error_vs_routine():
    """An error of a 95%-reliable tool is surprising (>1.0); a routine success is ~0 (<0.2)."""
    assert _tool_error_surprisal(1, 0.05, 20) > 1.0
    assert _tool_error_surprisal(0, 0.05, 20) < 0.2


def test_tool_error_surprisal_cold_start_neutral():
    """No prior or n < min_n -> 0.0 neutral, never NULL/raise (mirrors _base_activation n=0)."""
    assert _tool_error_surprisal(1, None, 0) == 0.0
    assert _tool_error_surprisal(1, 0.5, 3) == 0.0


def test_band_z_degenerate_and_none():
    """Degenerate (near-zero) variance and None inputs both degrade to 0.0, never raise."""
    assert _band_z(150.0, 100.0, 1e-9, 20) == 0.0
    assert _band_z(None, 100.0, 400.0, 20) == 0.0
    assert _band_z(150.0, None, 400.0, 20) == 0.0
    # A real deviation with healthy variance scores non-zero.
    assert _band_z(150.0, 100.0, 400.0, 20) == pytest.approx((150.0 - 100.0) / 20.0)


def test_compute_surprise_score_error_beats_success():
    """For identical latency/size, an error of a reliable tool scores strictly higher than
    a routine success of the same tool."""
    prior = ToolPrior(
        tool_name="bash", n=20, error_rate=0.05,
        lat_mean_ms=100.0, lat_var_ms=400.0, lat_n=20,
        size_mean=50.0, size_var=100.0, size_n=20,
    )
    err = compute_surprise_score(1, 55, 110, prior)
    ok = compute_surprise_score(0, 55, 110, prior)
    assert err > ok


def test_compute_quadrant_mapping():
    """The reframe taxonomy: predicted_fail x is_error, cold-start below min_n."""
    assert compute_quadrant(1, 0.05, 20) == "surprising_failure"
    assert compute_quadrant(0, 0.05, 20) == "routine"
    assert compute_quadrant(1, 0.8, 20) == "unsurprising_failure"
    assert compute_quadrant(0, 0.8, 20) == "quiet_surprise"
    assert compute_quadrant(1, 0.05, 2) == "cold_start"
    assert compute_quadrant(0, None, 20) == "cold_start"


# ---------------------------------------------------------------------------
# get_tool_prior — one pure-SQL aggregate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_tool_prior_sql_equivalence(tmp_path: Path):
    """Priors from ONE SQL aggregate match a statistics.pvariance reference within 1e-6."""
    store = make_store(tmp_path)
    await store.open()
    try:
        durations = [100, 120, 90, 200, 150, 110, 130, 95, 105, 180, 140, 160]
        sizes = [10, 20, 15, 200, 50, 30, 25, 12, 18, 200, 40, 60]
        errors = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
        await _seed_observations(store, "bash", [
            {"is_error": errors[i], "output_len": sizes[i], "duration_ms": durations[i]}
            for i in range(12)
        ])
        prior = await store.get_tool_prior("bash")
        assert prior.n == 12
        assert prior.error_rate == pytest.approx(2 / 12, abs=1e-6)
        assert prior.lat_mean_ms == pytest.approx(statistics.mean(durations), abs=1e-6)
        assert prior.lat_var_ms == pytest.approx(statistics.pvariance(durations), abs=1e-6)
        assert prior.lat_n == 12
        assert prior.size_mean == pytest.approx(statistics.mean(sizes), abs=1e-6)
        assert prior.size_var == pytest.approx(statistics.pvariance(sizes), abs=1e-6)
        assert prior.size_n == 12
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_get_tool_prior_walk_forward_excludes_current(tmp_path: Path):
    """before_ts walk-forward: only rows STRICTLY earlier count (the scored observation
    never contaminates its own prior)."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await _seed_observations(store, "bash", [
            {"is_error": 0, "duration_ms": 100, "output_len": 20, "ts": 1000 + i}
            for i in range(12)
        ])
        assert (await store.get_tool_prior("bash")).n == 12
        # ts 1000..1004 are strictly < 1005 -> exactly 5 rows.
        assert (await store.get_tool_prior("bash", before_ts=1005)).n == 5
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_cold_start_prior_scores_neutral(tmp_path: Path):
    """A cold-start prior (n < min_n) and an empty prior (no history) both score 0.0 —
    never an exception, never NULL."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await _seed_observations(store, "bash", [
            {"is_error": 1, "duration_ms": 100, "output_len": 20} for _ in range(3)
        ])
        cold = await store.get_tool_prior("bash")
        assert cold.n == 3
        assert compute_surprise_score(1, 200, 999, cold) == 0.0

        empty = await store.get_tool_prior("never_seen")
        assert empty.n == 0
        assert empty.error_rate is None
        assert compute_surprise_score(1, 200, 999, empty) == 0.0
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Recording APIs (COLL-03/04 substrate) — idempotent, collect-only
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_record_tool_observation_idempotent(tmp_path: Path):
    """Two records with the SAME event_id collapse to one row (INSERT OR IGNORE keyed on
    event_id — a live row and a later backfill of the same bus event are one)."""
    store = make_store(tmp_path)
    await store.open()
    try:
        kw = dict(session_id="s", tool_call_id="tc", tool_name="bash", ts=100,
                  is_error=0, output_len=20, duration_ms=50, event_id="dup-evt")
        id1 = await store.record_tool_observation(**kw)
        id2 = await store.record_tool_observation(**kw)
        assert id1 == id2
        async with store._db.execute(
            "SELECT COUNT(*) FROM tool_observations WHERE event_id = 'dup-evt'"
        ) as cur:
            assert (await cur.fetchone())[0] == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_snapshot_scoped_to_explicitly_staged(tmp_path: Path):
    """snapshot_staged_candidates captures exactly the explicitly-retrieved (touch_staged)
    facts — an untouched fact gets NO row; candidate_type + fact_id are stamped."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("fact-a", "alpha")
        await store.store_fact("fact-b", "beta")
        await store.store_fact("fact-c", "gamma")  # never staged
        await store.touch_staged(["fact-a", "fact-b"])
        sig = await store.record_user_signal(
            session_id="s", ts=100, signal_type="correction", trigger_family="negation",
            matched_text="no", user_message="no that's wrong",
            corrected_turn_summary=None, event_id="e1",
        )
        assert await store.snapshot_staged_candidates(sig, "suspect") == 2
        async with store._db.execute(
            "SELECT fact_key, fact_id, candidate_type FROM staged_snapshots "
            "WHERE user_signal_id = ?",
            (sig,),
        ) as cur:
            rows = await cur.fetchall()
        assert {r[0] for r in rows} == {"fact-a", "fact-b"}
        assert all(r[1] is not None for r in rows)       # fact_id populated
        assert all(r[2] == "suspect" for r in rows)      # correction -> suspect
        assert "fact-c" not in {r[0] for r in rows}      # untouched fact excluded
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_snapshot_after_fold_is_empty(tmp_path: Path):
    """Once staged counters fold into the base columns (consolidation), they are no longer
    'staged into this context window' — a later snapshot captures nothing."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("fact-a", "alpha")
        await store.touch_staged(["fact-a"])
        await store.fold_staged_access()
        sig = await store.record_user_signal(
            session_id="s", ts=100, signal_type="confirmation",
            trigger_family="confirmation", matched_text="yes", user_message="yes exactly",
            corrected_turn_summary=None, event_id="e2",
        )
        assert await store.snapshot_staged_candidates(sig, "bump") == 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_v4_write_burst_leaves_injected_block_byte_identical(tmp_path: Path):
    """The phase invariant: a burst of v4-table writes (20 observations + 5 scores + 2
    signals + staged snapshots, plus the touch_staged that feeds them) leaves
    _render_memory_index output BYTE-identical — collect-only never touches the ambient
    injected block."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("fact-a", "alpha body", confidence=0.9)
        await store.store_fact("fact-b", "beta body", confidence=0.8)
        await store.store_fact("fact-c", "gamma body", confidence=0.75)
        before = await store._render_memory_index(8)

        obs_ids = []
        for i in range(20):
            obs_ids.append(await store.record_tool_observation(
                session_id="s", tool_call_id=f"tc-{i}", tool_name="bash", ts=1000 + i,
                is_error=i % 2, output_len=100, duration_ms=50, event_id=f"ev-{i}",
            ))
        for i in range(5):
            await store.record_surprise_score(
                session_id="s", observation_id=obs_ids[i], expectation_json="{}",
                score=1.5, quadrant="routine", scored_at=2000 + i,
            )
        await store.touch_staged(["fact-a", "fact-b"])
        sig1 = await store.record_user_signal(
            session_id="s", ts=3000, signal_type="correction", trigger_family="negation",
            matched_text="no", user_message="no that's wrong",
            corrected_turn_summary=None, event_id="us-1",
        )
        sig2 = await store.record_user_signal(
            session_id="s", ts=3001, signal_type="confirmation",
            trigger_family="confirmation", matched_text="exactly",
            user_message="exactly right", corrected_turn_summary=None, event_id="us-2",
        )
        await store.snapshot_staged_candidates(sig1, "suspect")
        await store.snapshot_staged_candidates(sig2, "bump")

        after = await store._render_memory_index(8)
        assert after == before  # byte-identical (== on the full rendered string)
    finally:
        await store.close()

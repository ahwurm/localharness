"""Phase 29 (v2.0 Hierarchical Memory) — write-path tests: WRITE-01..06.

Covers: supersede-not-overwrite + history (WRITE-02), read-back-verify (WRITE-01),
provenance stamping (WRITE-04), FTS5 sanitization on real-corpus tokens (WRITE-05),
the prediction-error write gate + MemoryGateFired observability (WRITE-03/06),
the `remember` tool (WRITE-01), and the v1→v2 schema migration.
"""
import sqlite3
from pathlib import Path

import pytest

from localharness.memory.errors import MemoryVerifyError
from localharness.memory.sqlite import (
    SCHEMA_V1_SQL,
    FactQuery,
    MemoryStore,
    _sanitize_fts_query,
)


@pytest.fixture
async def store(tmp_path: Path):
    s = MemoryStore(agent_id="wp-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


# ---------------------------------------------------------------------------
# WRITE-02: supersede, never overwrite
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_supersede_keeps_history_and_links(store: MemoryStore):
    f1 = await store.store_fact("mu-thesis", "gen3: HBM supply constrained")
    f2 = await store.store_fact("mu-thesis", "gen4: HBM supply normalizing")

    active = await store.get_fact("mu-thesis")
    assert active is not None and active.value == "gen4: HBM supply normalizing"

    history = await store.get_fact_history("mu-thesis")
    assert len(history) == 2
    old = next(f for f in history if f.id == f1.id)
    assert old.status == "superseded"
    assert old.superseded_by == f2.id
    assert old.value == "gen3: HBM supply constrained"  # nothing lost


@pytest.mark.asyncio
async def test_default_query_excludes_superseded(store: MemoryStore):
    await store.store_fact("k", "old value alpha")
    await store.store_fact("k", "new value beta")

    default = await store.query_facts(FactQuery())
    assert [f.value for f in default] == ["new value beta"]

    with_history = await store.query_facts(FactQuery(include_superseded=True))
    assert {f.value for f in with_history} == {"old value alpha", "new value beta"}


@pytest.mark.asyncio
async def test_corroboration_touches_not_duplicates(store: MemoryStore):
    f1 = await store.store_fact("stable", "same claim", confidence=0.6)
    f2 = await store.store_fact("stable", "same claim", confidence=0.9)
    assert f2.id == f1.id  # same row — no duplicate, no supersede
    assert f2.confidence == 0.9  # max(old, new)
    assert len(await store.get_fact_history("stable")) == 1


@pytest.mark.asyncio
async def test_superseded_fact_invisible_to_get_fact(store: MemoryStore):
    await store.store_fact("k2", "v1")
    await store.store_fact("k2", "v2")
    fact = await store.get_fact("k2")
    assert fact is not None and fact.value == "v2" and fact.status == "active"


# ---------------------------------------------------------------------------
# WRITE-01: read-back-verify
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_back_verify_raises_on_mismatch(store: MemoryStore, monkeypatch):
    async def _lie(key):  # store claims to have written; re-read says otherwise
        return None
    # First call (existence check) must behave; patch after the existence check by
    # counting calls: 1st call → real, 2nd (verify) → None.
    real = store._get_fact_row
    calls = {"n": 0}

    async def _flaky(key):
        calls["n"] += 1
        if calls["n"] >= 2:
            return await _lie(key)
        return await real(key)

    monkeypatch.setattr(store, "_get_fact_row", _flaky)
    with pytest.raises(MemoryVerifyError):
        await store.store_fact("ghost", "never lands")


# ---------------------------------------------------------------------------
# WRITE-04: provenance
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_provenance_defaults_to_current_session(store: MemoryStore):
    store.set_current_session("sess-42")
    fact = await store.store_fact("prov", "carries its source")
    assert fact.provenance == "sess-42"

    explicit = await store.store_fact("prov2", "explicit wins", provenance="sess-99")
    assert explicit.provenance == "sess-99"


# ---------------------------------------------------------------------------
# WRITE-05: FTS5 sanitization on the exact tokens that used to throw
# ---------------------------------------------------------------------------

def test_sanitize_quotes_operator_tokens():
    assert _sanitize_fts_query("000660.KS") == '"000660.KS"'
    assert _sanitize_fts_query('say "hi"') == '"say" """hi"""'
    assert _sanitize_fts_query("   ") == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("needle", ["000660.KS", "P/GP", "-1.5σ"])
async def test_fts_corpus_tokens_return_results(store: MemoryStore, needle: str):
    await store.store_fact(
        "corpus-fact", "SK Hynix 000660.KS screens cheap on P/GP with a -1.5σ tripwire"
    )
    results = await store.query_facts(FactQuery(text=needle))
    assert len(results) == 1  # no fts5 syntax error, and it actually matches
    assert results[0].key == "corpus-fact"


# ---------------------------------------------------------------------------
# WRITE-03/06: prediction-error write gate + observability
# ---------------------------------------------------------------------------

class _FakeBus:
    def __init__(self):
        self.published = []
        self.subscriptions = []

    def subscribe(self, event_type, handler, agent_id=None):
        self.subscriptions.append((event_type, handler, agent_id))
        return object()

    def unsubscribe(self, handle):
        pass

    async def publish(self, event):
        self.published.append(event)


@pytest.mark.asyncio
async def test_gate_resolved_error_captures_candidate(store: MemoryStore):
    from localharness.core.events import Observation
    from localharness.memory.gate import WriteGate

    bus = _FakeBus()
    gate = WriteGate(store, bus, "wp-agent")

    err = Observation(agent_id="wp-agent", session_id="s1", observation_type="tool_result",
                      tool_name="bash_exec", output="", error="command not found: uvx")
    ok = Observation(agent_id="wp-agent", session_id="s1", observation_type="tool_result",
                     tool_name="bash_exec", output="ok", error=None)
    await gate._on_observation(err)
    await gate._on_observation(ok)

    fired = [e for e in bus.published if e.event_type == "MemoryGateFired"]
    resolved = [e for e in fired if e.tier == "resolved_error"]
    assert len(resolved) == 1
    fact = await store.get_fact(resolved[0].fact_key)
    assert fact is not None
    assert "pending_consolidation" in fact.tags
    assert fact.confidence < 0.7  # below the injection threshold until consolidation
    assert fact.provenance == "s1"
    assert "command not found" in fact.value


@pytest.mark.asyncio
async def test_gate_stuck_recovered_and_novelty(store: MemoryStore):
    from localharness.core.events import Observation, StuckRecovered
    from localharness.memory.gate import WriteGate

    bus = _FakeBus()
    gate = WriteGate(store, bus, "wp-agent")

    # First successful use of a tool → novelty tier
    first = Observation(agent_id="wp-agent", session_id="s2", observation_type="tool_result",
                        tool_name="grep", output="3 matches", error=None)
    await gate._on_observation(first)

    stuck = StuckRecovered(agent_id="wp-agent", session_id="s2", iteration=7,
                           stuck_signature="read:{'path': 'x'}")
    await gate._on_stuck_recovered(stuck)

    tiers = sorted(e.tier for e in bus.published if e.event_type == "MemoryGateFired")
    assert tiers == ["novelty", "stuck_recovered"]
    # Second success on the same tool: NOT novel again, no resolved-error → no new event
    await gate._on_observation(first)
    assert len([e for e in bus.published if e.event_type == "MemoryGateFired"]) == 2


@pytest.mark.asyncio
async def test_gate_resolves_across_turns(store: MemoryStore):
    """Critic M3: loop.py mints a fresh session_id per run_turn — the canonical flow
    (fail in turn N, user says 'try again', succeed in turn N+1) MUST fire."""
    from localharness.core.events import Observation
    from localharness.memory.gate import WriteGate

    bus = _FakeBus()
    gate = WriteGate(store, bus, "wp-agent")
    await gate._on_observation(Observation(
        agent_id="wp-agent", session_id="turn-1", observation_type="tool_result",
        tool_name="edit", output="", error="old_string not found"))
    await gate._on_observation(Observation(
        agent_id="wp-agent", session_id="turn-2", observation_type="tool_result",
        tool_name="edit", output="edited", error=None))

    fired = [e for e in bus.published if e.event_type == "MemoryGateFired" and e.tier == "resolved_error"]
    assert len(fired) == 1
    fact = await store.get_fact(fired[0].fact_key)
    assert fact.provenance == "turn-2"  # the session that RESOLVED it


@pytest.mark.asyncio
async def test_gate_never_raises_into_the_loop(store: MemoryStore, monkeypatch):
    from localharness.core.events import Observation
    from localharness.memory.gate import WriteGate

    bus = _FakeBus()
    gate = WriteGate(store, bus, "wp-agent")

    async def _boom(**kwargs):
        raise RuntimeError("store exploded")
    monkeypatch.setattr(store, "store_fact", _boom)

    err = Observation(agent_id="wp-agent", session_id="s3", observation_type="tool_result",
                      tool_name="glob", output="", error="boom")
    ok = Observation(agent_id="wp-agent", session_id="s3", observation_type="tool_result",
                     tool_name="glob", output="fine", error=None)
    await gate._on_observation(err)
    await gate._on_observation(ok)  # capture raises inside; handler must swallow


# ---------------------------------------------------------------------------
# WRITE-01: remember tool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_remember_tool_writes_and_verifies(store: MemoryStore):
    from localharness.tools.builtin.memory_tools import MemoryRememberTool

    tool = MemoryRememberTool(store)
    result = await tool._execute(name="deploy-needs-vpn", content="Deploys fail off-VPN; connect first.")
    assert result.success
    fact = await store.get_fact("deploy-needs-vpn")
    assert fact is not None and "off-VPN" in fact.value
    assert "remember" in fact.tags
    assert fact.confidence == 0.9  # above injection threshold — explicit intent

    empty = await tool._execute(name="  ", content="x")
    assert not empty.success


# ---------------------------------------------------------------------------
# Migration: a real v1 DB opens as v2 with data intact
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_v1_db_migrates_in_place(tmp_path: Path):
    agent_dir = tmp_path / "agents" / "mig-agent"
    agent_dir.mkdir(parents=True)
    db_path = agent_dir / "memory.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_V1_SQL)
    conn.execute(
        "INSERT INTO facts (agent_id, division_id, org_id, key, value, tags, confidence, "
        "source, created_at, updated_at) VALUES ('mig-agent', '', '', 'legacy', 'v1 fact', "
        "'[]', 1.0, '', 100, 100)"
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    store = MemoryStore(agent_id="mig-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await store.open()
    try:
        fact = await store.get_fact("legacy")
        assert fact is not None and fact.value == "v1 fact"
        assert fact.status == "active" and fact.created_at == 100
        # supersede works post-migration (the UNIQUE constraint rebuild took)
        await store.store_fact("legacy", "v2 fact")
        history = await store.get_fact_history("legacy")
        assert len(history) == 2
        # FTS survived the rebuild
        hits = await store.query_facts(FactQuery(text="fact"))
        assert len(hits) == 1 and hits[0].value == "v2 fact"
    finally:
        await store.close()

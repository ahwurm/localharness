"""Stage A — the tag-graph substrate (schema v6): seeded two-bucket/two-layer spine, mint-time
two-step filing, remember()-pool inclusion, and the two B4 deterministic supersede defenses.

These lock the reachable behavior: after `open()` the seeded spine exists with FUNCTIONAL
decision-rule definitions; a minted atom is filed by a two-step closed-set classifier into a
bucket (+optional child); remember()-sourced facts are first-class taggable pool members; a
present-but-invalid `replaces=` marker rescues a same-slug supersede; and a mined atom that
re-asserts a value we already reconciled away is retracted on arrival.
"""
import asyncio

import pytest

from localharness.memory.clustering import _load_pool
from localharness.memory.mining import mine_transcript
from localharness.memory.sqlite import FactQuery, MemoryStore
from localharness.memory.tag_classify import _BUCKET_MARKER, _CHILD_MARKER


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(agent_id="tag-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


class _ClassifierLLM:
    """Prompt-aware fake: mining atoms for the miner prompt, a bucket for the bucket pick, a
    child for the child pick — keyed on the prompt markers so one instance drives the whole
    two-step mint flow deterministically."""

    def __init__(self, atoms: str = "", bucket: str = "project", child: str = "ops"):
        self.atoms, self.bucket, self.child = atoms, bucket, child

    async def complete(self, prompt: str) -> str:
        if "USER'S WORLD" in prompt:
            return self.atoms
        if _BUCKET_MARKER in prompt:
            return self.bucket
        if _CHILD_MARKER in prompt:
            return self.child
        return ""


def _rec(ts, content, sid="s1", typ="user_message"):
    return {"v": 1, "agent_id": "tag-agent", "type": typ, "id": f"h{ts}",
            "session_id": sid, "ts": ts, "content": content}


# ---------------------------------------------------------------------------
# Schema v6 substrate + the seeded two-bucket / two-layer spine
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_schema_v6_tables_and_version(store):
    async with store._db.execute("PRAGMA user_version") as cur:
        assert (await cur.fetchone())[0] == 6
    names = set()
    async with store._db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
        names = {r[0] for r in await cur.fetchall()}
    assert {"tags", "atom_tags"} <= names


@pytest.mark.asyncio
async def test_seeded_spine_two_buckets_six_children(store):
    """Amendment 4: two buckets (personal, project), each with three seeded children — filed
    by what a memory SERVES. No third bucket, no third layer."""
    buckets = await store.buckets()
    assert {b.name for b in buckets} == {"personal", "project"}
    for b in buckets:
        assert b.parent_id is None and b.status == "seeded" and b.origin == "seeded"
    by_name = {b.name: b for b in buckets}
    personal_kids = {t.name for t in await store.active_children(by_name["personal"].id)}
    project_kids = {t.name for t in await store.active_children(by_name["project"].id)}
    assert personal_kids == {"health", "travel", "preferences"}
    assert project_kids == {"ops", "conventions", "roadmap"}


@pytest.mark.asyncio
async def test_seed_definitions_are_functional_decision_rules(store):
    """Definitions must be concrete decision rules (what the memory SERVES) with an inline
    example — never a bare judgment word like 'useful' with no operational test."""
    for t in await store.list_tags():
        d = t.definition.lower()
        assert d, f"{t.name} has no definition"
        assert "serve" in d or "file here" in d, f"{t.name}: not a functional 'serves' rule: {t.definition!r}"
        assert "example" in d, f"{t.name}: no inline example: {t.definition!r}"
    ops = await store.get_tag("ops")
    assert "8081" in ops.definition or "port" in ops.definition.lower()  # concrete infra example


@pytest.mark.asyncio
async def test_depth_convention_permits_two_layers_only(store):
    """parent_id NULL == bucket (root); parent set == child. v1 permits ONLY bucket->child; a
    grandchild (parent is itself a child) is refused."""
    ops = await store.get_tag("ops")
    assert ops.parent_id == (await store.get_tag("project")).id
    with pytest.raises(ValueError):
        await store.create_tag("sub-ops", "serves x; example y", parent_id=ops.id,
                               status="active", origin="discovered")


# ---------------------------------------------------------------------------
# Mint-time two-step filing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mint_files_bucket_and_child(store):
    await store.append_history(_rec(10, "vLLM server listens on port 8081", sid="ops-day"))
    llm = _ClassifierLLM(atoms="gpu ops | vLLM server listens on port 8081 | listens on port 8081",
                         bucket="project", child="ops")
    r = await mine_transcript(store, llm, asyncio.Event())
    assert r.written == 1
    atom = (await store.query_facts(FactQuery(tags=["sem"])))[0]
    names = {t.name for t in await store.tags_for_atom(atom.id)}
    assert names == {"project", "ops"}
    provs = {t.provenance for t in await store.atom_tag_rows(atom.id)}
    assert provs == {"mint"}


@pytest.mark.asyncio
async def test_mint_garbage_child_files_bucket_only(store):
    await store.append_history(_rec(10, "vLLM server listens on port 8081", sid="ops-day"))
    llm = _ClassifierLLM(atoms="gpu ops | vLLM server listens on port 8081 | port 8081",
                         bucket="project", child="not-a-real-tag")
    await mine_transcript(store, llm, asyncio.Event())
    atom = (await store.query_facts(FactQuery(tags=["sem"])))[0]
    assert {t.name for t in await store.tags_for_atom(atom.id)} == {"project"}


@pytest.mark.asyncio
async def test_mint_garbage_bucket_leaves_untagged_but_mints(store):
    await store.append_history(_rec(10, "vLLM server listens on port 8081", sid="ops-day"))
    llm = _ClassifierLLM(atoms="gpu ops | vLLM server listens on port 8081 | port 8081",
                         bucket="nonsense", child="ops")
    r = await mine_transcript(store, llm, asyncio.Event())
    assert r.written == 1  # tagging failure NEVER blocks the mint
    atom = (await store.query_facts(FactQuery(tags=["sem"])))[0]
    assert await store.tags_for_atom(atom.id) == []


# ---------------------------------------------------------------------------
# remember()-pool inclusion (critique item 1)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_remember_facts_are_first_class_pool_members(store):
    """A remember()-sourced fact (tagged 'remember') is a taggable semantic atom in the pool
    alongside sem/* — mirroring the mined/* precedent, closing the SEMA-05 key-prefix gap."""
    await store.store_fact(key="semis-hbm-bull", value="HBM makers face hyperscaler capex risk",
                           tags=["remember"], confidence=0.9, source="remember", provenance="s1")
    pool_keys = {f.key for f in await _load_pool(store)}
    assert "semis-hbm-bull" in pool_keys


# ---------------------------------------------------------------------------
# B4 defense (i): present-but-invalid/empty replaces= rescues a same-slug supersede
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_b4i_empty_replaces_supersedes_newest_same_slug(store):
    await store.append_history(_rec(10, "vLLM server listens on port 8000", sid="d1"))
    await mine_transcript(store, _ClassifierLLM(
        atoms="vllm port | vLLM server listens on port 8000 | listens on port 8000"), asyncio.Event())
    original = (await store.query_facts(FactQuery(tags=["sem"])))[0]
    assert "8000" in original.value

    # A correcting atom on the SAME slug with a PRESENT-but-empty replaces= marker: no valid
    # target id, but slug matches an active atom -> rescue-supersede the newest same-slug atom.
    await store.append_history(_rec(20, "correction: vLLM server listens on port 8081", sid="d2"))
    await mine_transcript(store, _ClassifierLLM(
        atoms="vllm port | vLLM server listens on port 8081 | listens on port 8081 | replaces="),
        asyncio.Event())
    active = await store.get_fact(original.key)
    assert active is not None and "8081" in active.value  # superseded onto the old key
    hist = await store.get_fact_history(original.key)
    assert any(v.status == "superseded" and "8000" in v.value for v in hist)


# ---------------------------------------------------------------------------
# B4 defense (ii): a mined atom resurrecting a reconciled-away stale value is retracted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_b4ii_stale_resurrection_retracted_on_arrival(store):
    """A reconciled correction leaves the stale value superseded and the corrected value active
    (tier:reconcile_confirmed). A late chunk that mines the stale value back must NOT resurrect
    it: the fresh atom is retracted on arrival (never joins the active pool)."""
    # The reconciled correction: 8000 (stale, superseded) -> 8081 (active, confirmed), same key.
    await store.store_fact(key="vllm-server-port", value="vLLM server listens on port 8000",
                           tags=["remember"], confidence=0.9, provenance="d1")
    await store.store_fact(key="vllm-server-port", value="vLLM server listens on port 8081",
                           tags=["remember", "tier:reconcile_confirmed"], confidence=0.9, provenance="d2")

    # A late transcript chunk re-asserts the STALE port; the miner extracts it as a fresh atom.
    await store.append_history(_rec(30, "reminder: the vLLM server listens on port 8000", sid="d3"))
    await mine_transcript(store, _ClassifierLLM(
        atoms="gpu ops | the vLLM server listens on port 8000 | listens on port 8000",
        bucket="project", child="ops"), asyncio.Event())

    # No ACTIVE fact anywhere asserts the stale 8000 (the resurrection was retracted).
    actives = await store.query_facts(FactQuery(text="8000", include_superseded=False, limit=50))
    assert all("8000" not in f.value for f in actives), [f.key for f in actives if "8000" in f.value]

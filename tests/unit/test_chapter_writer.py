"""SEMA-02/03 write half (Phase 36, the chapter-writer) — one grounded chapter per cluster.

Covers: the grounded, cancellable generation that turns a stable lesson cluster into ONE
schema node with member_of edges (Task 1); the depth cap + member fold-out + cancellation
proof (Task 2); and the tier:surprising_failure consume + bounded drain, PGATE-03's only
consumer (Task 3). Every LLM is a fake double — no live vLLM in the suite (machine-safety).

The kill is the spine: an ungrounded chapter, or one carrying an unverified figure, writes
NOTHING. Seed content is fixture-fake (allowed in tests; only the SEMA-05 provable forbids
fabricated lesson text — see 36-CONTEXT.md).
"""
import asyncio
import time

import pytest

from localharness.memory.chapter_writer import write_cluster_schemas
from localharness.memory.clustering import _load_failure_queue, find_stable_clusters
from localharness.memory.consolidation import _get_meta, _set_meta
from localharness.memory.idle_llm import grounded
from localharness.memory.sqlite import SCHEMA_KEY_PREFIX, FactQuery, MemoryStore

# Three `read` lessons sharing the salient tokens "absolute"/"resolved" (they cluster); the
# grounded fragment is a verbatim slice of _READ_A (every >=6-char token is in the corpus).
_READ_A = "The read tool returned FileNotFound on a relative path; retrying with the absolute path resolved it."
_READ_B = "The read tool raised a permission problem on a protected path; the absolute path form resolved it cleanly."
_READ_C = "The read tool timed out once then, using the absolute path directly, resolved on the second attempt."
_GROUNDED = "read tool returned FileNotFound retrying with the absolute path resolved"
# A second domain (docker) sharing NO salient token with the read lessons — a distinct cluster.
_DOCK_A = "The docker build failed pulling from the registry; a container cache prune cleared the broken layer."
_DOCK_B = "The docker container refused to start until the registry credentials were refreshed before the pull."
# Two depth-N chapter bodies sharing "absolute"/"resolves"/"recovers" — they cluster one level up.
_CHAP_A = "Chapter: the read tool reliably recovers when the absolute path resolves the lookup failure."
_CHAP_B = "Chapter: the read tool recovers on protected paths once the absolute path resolves cleanly."


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(agent_id="chap-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


# --- seed helpers (mirrors 36-01's test_memory_clustering seeding) ----------------------

async def _seed_learned(store, tool, tier, body, sessions, *, lesson="", node_kind="fact"):
    """A promoted lesson exactly as consolidation.py writes it (key learned/{tool}/{tier},
    confidence 0.8) PLUS its derived_from source gate/* candidates (one per session, each
    carrying provenance=session) — so the session spread is derivable from neighborhood()
    exactly like the real graph. Returns the promoted Fact."""
    key = f"learned/{tool}/{tier}" + (f"/{lesson}" if lesson else "")
    promoted = await store.store_fact(
        key=key, value=body, tags=["consolidated", f"tier:{tier}"],
        confidence=0.8, source="consolidation",
        provenance=f"consolidated:{len(sessions)}-episodes", node_kind=node_kind,
    )
    for i, sess in enumerate(sessions):
        cand = await store.store_fact(
            key=f"gate/{tier}/{tool}/{lesson or 'L'}{i}/{sess}",
            value=f"Tool `{tool}` episode in {sess}: {body}",
            tags=["gate", f"tier:{tier}", "pending_consolidation"],
            confidence=0.65, source="write_gate", provenance=sess,
        )
        await store.add_edge(promoted.id, cand.id, "derived_from")
    return promoted


async def _seed_sem(store, body, session, *, topic="topic", conf=0.65):
    import hashlib
    h = hashlib.sha1(body.strip().encode("utf-8")).hexdigest()[:8]
    return await store.store_fact(
        key=f"sem/{topic}/{h}", value=body, tags=["sem", "pending_consolidation"],
        confidence=conf, source="transcript_mining", provenance=session, node_kind="fact",
    )


async def _seed_failure(store, tool, day="20260705", *, rs=None, value=None, prov="sess-fail"):
    """A tier:surprising_failure queue row exactly as predictive_write_gate.py writes it
    (key predgate/surprising_failure/{tool}/{day}, sub-0.7 confidence, provenance=session)."""
    f = await store.store_fact(
        key=f"predgate/surprising_failure/{tool}/{day}",
        value=value or (f"`{tool}` had a surprising failure — a normally-reliable tool "
                        "errored (quadrant surprising_failure). Pending consolidation."),
        tags=["gate", "tier:surprising_failure", "pending_consolidation"],
        confidence=0.6, source="predictive_write_gate", provenance=prov,
    )
    if rs is not None:
        await store._db.execute("UPDATE facts SET retrieval_strength = ? WHERE id = ?", (rs, f.id))
        await store._db.commit()
    return f


async def _seed_schema(store, suffix, body, depth, prov):
    """A chapter node exactly as the writer emits it (node_kind='schema', depth:N, 0.8) — used
    to build a chapter-of-chapters cluster and to probe the depth cap."""
    return await store.store_fact(
        key=f"schema/cluster/{suffix}", value=body,
        tags=["schema", "tier:schema", f"depth:{depth}"],
        confidence=0.8, source="chapter_writer", provenance=prov, node_kind="schema",
    )


async def _member_of_dst(store, src_id) -> set[int]:
    async with store._db.execute(
        "SELECT dst_id FROM edges WHERE src_id = ? AND kind = 'member_of'", (src_id,)
    ) as cur:
        return {r[0] for r in await cur.fetchall()}


async def _row(store, fact_id):
    """Re-read a single ACTIVE fact by id (tags/confidence/retrieval_strength after a raw retag)."""
    facts = await store.get_facts_by_ids([fact_id])
    return facts[0] if facts else None


async def _schema_cluster_count(store) -> int:
    async with store._db.execute(
        "SELECT COUNT(*) FROM facts WHERE status = 'active' AND key LIKE 'schema/cluster/%'"
    ) as cur:
        return (await cur.fetchone())[0]


# --- fake LLM doubles -------------------------------------------------------------------

class _EchoLLM:
    """Returns a fixed line — used to inject a specific grounded/ungrounded/figure claim."""
    def __init__(self, text: str):
        self.text = text

    async def complete(self, prompt: str) -> str:
        return self.text


class _CorpusEchoLLM:
    """A grounded double: echoes a verbatim fragment of the corpus (the prompt tail), so it
    can never fabricate a token — grounding always passes regardless of which cluster it sees."""
    async def complete(self, prompt: str) -> str:
        corpus = prompt.split("\n\n")[-1]
        lines = [ln for ln in corpus.splitlines() if ln.strip()]
        return " ".join(lines[0].split()[:12]) if lines else "chapter"


class _SlowLLM:
    """A generation that blocks until cancelled (the serial-gate release proof)."""
    def __init__(self, delay: float = 10.0):
        self.delay = delay
        self.cancelled = False

    async def complete(self, prompt: str) -> str:
        try:
            await asyncio.sleep(self.delay)
            return "slow chapter"
        except asyncio.CancelledError:
            self.cancelled = True
            raise


# ---------------------------------------------------------------------------------------
# Task 1 — one grounded chapter per cluster + member_of edges; the kill; the budget
# ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_writes_one_schema_with_member_edges(store):
    """A stable 3-lesson cluster + a grounded generation -> exactly ONE schema node (0.8,
    node_kind schema, key schema/cluster/*, depth:1) with a member_of edge to each member."""
    m1 = await _seed_sem(store, _READ_A, "s1")
    m2 = await _seed_sem(store, _READ_B, "s2")
    m3 = await _seed_sem(store, _READ_C, "s3")

    result = await write_cluster_schemas(store, _EchoLLM(_GROUNDED), asyncio.Event())

    assert len(result) == 1
    schema = result[0]
    assert schema.node_kind == "schema"
    assert schema.confidence == 0.8
    assert schema.key.startswith(SCHEMA_KEY_PREFIX)
    assert "tier:schema" in schema.tags
    assert "depth:1" in schema.tags
    assert schema.value == _GROUNDED
    assert await _member_of_dst(store, schema.id) == {m1.id, m2.id, m3.id}


@pytest.mark.asyncio
async def test_writes_one_from_dereferenced_event(store):
    """The corpus DEREFERENCES payload-thin aux stat rows via provenance=session_id: a chapter
    grounds on the event-log text, not the boilerplate stat value. Load-bearing proof: the same
    claim is NOT grounded against the member bodies alone — only the deref makes it write."""
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")
    await _seed_failure(store, "read", value="absolute path resolved via sess-k", prov="sess-k")  # attaches as aux (same tool)
    event_text = "authentication failed because the kerberos ticket expired"
    await store.append_history({"v": 1, "type": "tool_result", "id": "e1",
                                "session_id": "sess-k", "agent_id": "chap-agent",
                                "ts": 1, "content": event_text})

    result = await write_cluster_schemas(store, _EchoLLM(event_text), asyncio.Event())

    assert len(result) == 1
    assert result[0].value == event_text
    assert grounded(event_text, "\n".join([_READ_A, _READ_B])) is False  # deref was required


@pytest.mark.asyncio
async def test_ungrounded_generation_writes_no_schema(store):
    """The pre-committed KILL: a chapter with a majority of tokens in NO member lesson is
    rejected — no schema written (a hallucinated chapter is worse than no chapter).
    (Seeds migrated to sem/ — the learned/ seeds left this test VACUOUS after MOVE 1: no
    pool -> no cluster -> [] trivially; the kill path was silently unexercised.)"""
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")
    assert len(await find_stable_clusters(store)) == 1  # non-vacuous: the kill has a cluster to kill

    result = await write_cluster_schemas(
        store, _EchoLLM("Xylophones fabricate quarterly financials unicorns"), asyncio.Event()
    )

    assert result == []
    assert await _schema_cluster_count(store) == 0


@pytest.mark.asyncio
async def test_ungrounded_unverified_figure_rejected(store):
    """The numeric net layered on top: grounded prose that carries a figure derivable from NO
    member lesson is rejected (SEMA-05 is stricter than hierarchy's flag-don't-reject).
    (Seeds migrated to sem/ for the same vacuousness reason as above.)"""
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")
    assert len(await find_stable_clusters(store)) == 1

    result = await write_cluster_schemas(
        store,
        _EchoLLM("the absolute path resolved the permission problem after 42 attempts"),
        asyncio.Event(),
    )

    assert result == []
    assert await _schema_cluster_count(store) == 0


@pytest.mark.asyncio
async def test_attempts_log_records_rejections_and_writes(store):
    """Ruling 4 (run-2 observability gap): per_schema_grounding was EMPTY because only WRITTEN
    schemas were graded — rejected attempts were invisible, so 'no chapter written' had no
    forensic trail. write_cluster_schemas(attempts_log=...) records EVERY attempt: a rejected
    one with its reason + grounding fields, a written one with its schema key."""
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")

    attempts: list[dict] = []
    result = await write_cluster_schemas(
        store, _EchoLLM("Xylophones fabricate quarterly financials unicorns"),
        asyncio.Event(), attempts_log=attempts,
    )
    assert result == []
    assert len(attempts) == 1
    a = attempts[0]
    assert a["written"] is False and a["reason"] == "ungrounded"
    assert a["grounded"] is False and a["grounded_majority"] is False
    assert a["members"] == 2
    for field in ("key", "value", "unverified_numbers"):  # _report-render compatibility
        assert field in a

    attempts2: list[dict] = []
    result2 = await write_cluster_schemas(
        store, _EchoLLM(_GROUNDED), asyncio.Event(), attempts_log=attempts2,
    )
    assert len(result2) == 1
    assert attempts2[0]["written"] is True and attempts2[0]["reason"] == "written"
    assert attempts2[0]["key"] == result2[0].key and attempts2[0]["grounded"] is True


@pytest.mark.asyncio
async def test_write_budget_caps_schema_count(store):
    """write_budget bounds schemas-per-call: 5 stable clusters, budget 3 -> at most 3 written."""
    for stem in "abcde":
        a = f"{stem}zzzalpha {stem}zzzbravo {stem}zzzcharlie"
        b = f"{stem}zzzalpha {stem}zzzbravo {stem}zzzdelta"
        await _seed_sem(store, a, f"{stem}s1")
        await _seed_sem(store, b, f"{stem}s2")

    assert len(await find_stable_clusters(store)) == 5  # five independent clusters

    result = await write_cluster_schemas(store, _CorpusEchoLLM(), asyncio.Event(), write_budget=3)

    assert len(result) == 3
    assert await _schema_cluster_count(store) == 3


# ---------------------------------------------------------------------------------------
# Task 2 — depth cap + member fold-out + cancellation/serial-gate proof
# ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_depth_cap_refuses_deep_cluster(store):
    """A cluster of depth:2 chapters would yield a depth:3 chapter (> cap 2) — REFUSED, no write.
    lesson(0) -> chapter(depth:1) -> chapter-of-chapters(depth:2) -> stop."""
    await _seed_schema(store, "deepA", _CHAP_A, 2, "cluster:xa")
    await _seed_schema(store, "deepB", _CHAP_B, 2, "cluster:xb")

    result = await write_cluster_schemas(store, _CorpusEchoLLM(), asyncio.Event())

    assert result == []
    assert await _schema_cluster_count(store) == 2  # only the two seeds; no depth:3 chapter


@pytest.mark.asyncio
async def test_depth1_cluster_yields_depth2_chapter(store):
    """Chapter-of-chapters is allowed exactly once: a cluster of depth:1 chapters -> depth:2."""
    await _seed_schema(store, "d1A", _CHAP_A, 1, "cluster:xa")
    await _seed_schema(store, "d1B", _CHAP_B, 1, "cluster:xb")

    result = await write_cluster_schemas(store, _CorpusEchoLLM(), asyncio.Event())

    assert len(result) == 1
    assert "depth:2" in result[0].tags


@pytest.mark.asyncio
async def test_member_foldout_demotes_but_searchable(store):
    """After a chapter is written, each member's retrieval_strength drops to 0.15 (out of the
    ambient index) yet the member stays query_facts-searchable — the pile folds into the chapter."""
    m1 = await _seed_sem(store, _READ_A, "s1")
    m2 = await _seed_sem(store, _READ_B, "s2")
    m3 = await _seed_sem(store, _READ_C, "s3")

    result = await write_cluster_schemas(store, _EchoLLM(_GROUNDED), asyncio.Event())
    assert len(result) == 1

    for m in (m1, m2, m3):
        got = await store.get_fact(m.key)
        assert got.retrieval_strength == 0.15   # below the 0.2 index gate
        assert got.confidence == 0.65           # trust untouched at the atom's 0.65 — only accessibility demoted
    hits = {f.key for f in await store.query_facts(FactQuery(text="resolved"))}
    assert m1.key in hits                        # still searchable (rs is a non-indexed column)


@pytest.mark.asyncio
async def test_cancel_stops_further_clusters(store):
    """A cancel_event set mid-generation stops the pass without hanging (partial result) and the
    generation task is truly cancelled — the serial inference gate is released promptly."""
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")
    await _seed_sem(store, _DOCK_A, "s3")
    await _seed_sem(store, _DOCK_B, "s4")

    llm = _SlowLLM(delay=10.0)
    cancel = asyncio.Event()
    t0 = time.monotonic()
    task = asyncio.create_task(write_cluster_schemas(store, llm, cancel))
    await asyncio.sleep(0.1)
    cancel.set()  # what a user turn does via the scheduler
    result = await asyncio.wait_for(task, timeout=3.0)

    assert result == []                    # cancelled before the first chapter completed
    assert time.monotonic() - t0 < 3.0     # nobody waited 10s behind the generation
    assert llm.cancelled                   # generation cancelled -> inference gate released
    assert await _schema_cluster_count(store) == 0


# ---------------------------------------------------------------------------------------
# Task 3 — consume + drain the tier:surprising_failure queue (PGATE-03's only consumer)
# ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_claimed_aux_consumed_and_drained(store):
    """A CLAIMED aux row (folded under a chapter) gets a member_of edge + is retagged consumed
    (pending_consolidation dropped, tier:consumed added, rs demoted) — confidence UNCHANGED. It
    leaves the pending surprising_failure probe (drains) but stays query_facts-searchable."""
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")
    aux = await _seed_failure(store, "read", value="absolute path resolved for claim")  # same-tool -> attaches as aux (36-01)

    result = await write_cluster_schemas(store, _CorpusEchoLLM(), asyncio.Event())
    assert len(result) == 1
    schema = result[0]

    row = await _row(store, aux.id)
    assert "pending_consolidation" not in row.tags       # drained from the queue
    assert "tier:consumed" in row.tags                   # folded under the chapter
    assert row.confidence == 0.6                          # NEVER promoted above the <0.7 clamp
    assert aux.id in await _member_of_dst(store, schema.id)   # member_of schema -> aux
    assert aux.id not in {f.id for f in await _load_failure_queue(store)}  # off the pending probe
    assert aux.key in {f.key for f in await store.query_facts(FactQuery(text="surprising"))}


@pytest.mark.asyncio
async def test_unclaimed_failure_drains_at_stale_looks(store):
    """An UNCLAIMED row (adjacent to no stable cluster) drains after stale_looks idle cycles so
    the queue cannot grow unboundedly: counter pre-seeded to stale_looks-1 -> one pass drains it."""
    aux = await _seed_failure(store, "read", prov="lonely")   # no lessons -> no cluster claims it
    await _set_meta(store, f"failure/looks/{aux.key}", "4")     # one look short of stale_looks=5

    result = await write_cluster_schemas(store, _EchoLLM("unused"), asyncio.Event())
    assert result == []                                         # no clusters, only the drain sweep

    row = await _row(store, aux.id)
    assert "pending_consolidation" not in row.tags             # drained at the 5th look
    assert "tier:consumed" in row.tags
    assert "stale" in row.tags                                 # aged-out breadcrumb
    assert row.confidence == 0.6                               # confidence never touched


@pytest.mark.asyncio
async def test_unclaimed_not_drained_before_stale_looks(store):
    """The same row one cycle earlier (counter < stale_looks) is NOT drained — only the look
    counter is bumped; it stays in the pending queue."""
    aux = await _seed_failure(store, "read", prov="lonely")
    await _set_meta(store, f"failure/looks/{aux.key}", "3")     # two looks short

    await write_cluster_schemas(store, _EchoLLM("unused"), asyncio.Event())

    row = await _row(store, aux.id)
    assert "pending_consolidation" in row.tags                 # still queued (4 < 5)
    assert "tier:consumed" not in row.tags
    assert await _get_meta(store, f"failure/looks/{aux.key}") == "4"   # counter bumped, not drained

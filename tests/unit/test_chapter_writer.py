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

from localharness.memory.chapter_writer import _write_one, write_cluster_schemas
from localharness.memory.clustering import Cluster, _load_failure_queue, find_stable_clusters
from localharness.memory.consolidation import _get_meta, _set_meta
from localharness.memory.idle_llm import ground_numbers, grounded, strip_chapter_title
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


_USE_TOPIC = object()  # sentinel: attach a child tag named after `topic` (Stage B co-tag edges)


async def _seed_sem(store, body, session, *, topic="topic", conf=0.65, child_tag=_USE_TOPIC):
    """Stage B: grouping is by SHARED CHILD TAG, so by default file the atom under an active child
    tag named after `topic` (same-topic atoms co-tag-link, as the old same-slug token rule did).
    Pass child_tag=None for bucket-only (no grouping edges) or a name to share across topics."""
    import hashlib
    h = hashlib.sha1(body.strip().encode("utf-8")).hexdigest()[:8]
    fact = await store.store_fact(
        key=f"sem/{topic}/{h}", value=body, tags=["sem", "pending_consolidation"],
        confidence=conf, source="transcript_mining", provenance=session, node_kind="fact",
    )
    tag_name = topic if child_tag is _USE_TOPIC else child_tag
    if tag_name:
        child = await store.get_tag(tag_name)
        if child is None:
            child = await store.create_tag(
                tag_name, f"serves {tag_name}; example {tag_name}",
                parent_id=(await store.get_tag("project")).id, status="active", origin="discovered")
        await store.add_atom_tag(fact.id, child.id, "discovery")
    return fact


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


async def _seed_schema(store, suffix, body, depth, prov, *, child_tag="chapters"):
    """A chapter node exactly as the writer emits it (node_kind='schema', depth:N, 0.8) — used to
    build a chapter-of-chapters cluster and to probe the depth cap. Stage B: chapter-of-chapters
    clustering is now by shared CHILD tag (token overlap removed), so schema seeds meant to
    cluster share `child_tag`."""
    sch = await store.store_fact(
        key=f"schema/cluster/{suffix}", value=body,
        tags=["schema", "tier:schema", f"depth:{depth}"],
        confidence=0.8, source="chapter_writer", provenance=prov, node_kind="schema",
    )
    if child_tag:
        child = await store.get_tag(child_tag)
        if child is None:
            child = await store.create_tag(
                child_tag, f"serves {child_tag}; example {child_tag}",
                parent_id=(await store.get_tag("project")).id, status="active", origin="discovered")
        await store.add_atom_tag(sch.id, child.id, "curation")
    return sch


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


# --- FIX 1: the grounding gate must ground the chapter BODY, not the markdown title -----
# Run-3 reality: the writer prompt asks for a titled chapter, the model emits '**Title**\nbody',
# and the title's tokens (never in the plain member corpus) sank the majority-token bar — all 3
# grounded drafts were KILLed. CONTRACT: title tokens are never counted; a genuinely-grounded
# body writes; a hallucinated body (new entity absent from the atoms) is STILL rejected.

# Two clustering members (shared 'backend' topic slug). Body grounds 3/5 tokens (engineer,
# refactored, database) — a BARE majority that the two unmatched title tokens would flip to a
# minority (3/7) if counted. This isolates the title-strip as load-bearing, not run-3 overfit.
_MEMBER_1 = "the engineer refactored the service cleanly"
_MEMBER_2 = "the database was migrated by the engineer"
_TITLED_GROUNDED = "**Backend Cleanup**\nThe engineer refactored the database module carefully."


@pytest.mark.asyncio
async def test_titled_chapter_grounds_on_body_not_title(store):
    """A titled '**Backend Cleanup**\\nbody' draft whose BODY is a bare-majority match writes a
    chapter — the two unmatched title tokens ('backend','cleanup') are not counted. Load-bearing
    proof: the SAME text KILLs when the title is included, and passes once stripped."""
    m1 = await _seed_sem(store, _MEMBER_1, "s1", topic="backend")
    m2 = await _seed_sem(store, _MEMBER_2, "s2", topic="backend")
    corpus = "\n".join([_MEMBER_1, _MEMBER_2])
    # The gate must accept the body but reject the whole titled draft — that gap IS the fix.
    assert grounded(_TITLED_GROUNDED, corpus) is False
    assert grounded(strip_chapter_title(_TITLED_GROUNDED), corpus) is True

    result = await write_cluster_schemas(store, _EchoLLM(_TITLED_GROUNDED), asyncio.Event())
    assert len(result) == 1
    assert await _member_of_dst(store, result[0].id) == {m1.id, m2.id}


@pytest.mark.asyncio
async def test_titled_chapter_with_hallucinated_body_still_rejected(store):
    """The anti-hallucination intent survives the title-strip: a titled draft whose BODY
    introduces entities absent from every member atom is STILL KILLed (no schema written)."""
    await _seed_sem(store, _MEMBER_1, "s1", topic="backend")
    await _seed_sem(store, _MEMBER_2, "s2", topic="backend")

    result = await write_cluster_schemas(
        store,
        _EchoLLM("**Backend Cleanup**\nThe engineer deployed kubernetes across seven datacenters."),
        asyncio.Event(),
    )
    assert result == []
    assert await _schema_cluster_count(store) == 0


@pytest.mark.asyncio
async def test_write_budget_caps_schema_count(store):
    """write_budget bounds schemas-per-call: 5 stable clusters, budget 3 -> at most 3 written."""
    for stem in "abcde":
        a = f"{stem}zzzalpha {stem}zzzbravo {stem}zzzcharlie"
        b = f"{stem}zzzalpha {stem}zzzbravo {stem}zzzdelta"
        await _seed_sem(store, a, f"{stem}s1", topic=f"topic-{stem}")  # 5 distinct child tags ->
        await _seed_sem(store, b, f"{stem}s2", topic=f"topic-{stem}")  # 5 independent clusters

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
async def test_grown_cluster_supersedes_existing_chapter(store):
    """CHAPTER REFRESH (run-14 bug): chapter identity must survive membership drift. A cluster
    that GAINS a member (run 14: a rescued atom / a correction row joining) re-writes the
    EXISTING chapter — supersede on the OLD key, exactly one active chapter — never a
    near-identical sibling (run 14 left three duplicate pairs, and the stale sibling is what
    failed B4)."""
    m1 = await _seed_sem(store, _READ_A, "s1")
    m2 = await _seed_sem(store, _READ_B, "s2")
    r1 = await write_cluster_schemas(store, _EchoLLM(_GROUNDED), asyncio.Event())
    assert len(r1) == 1
    key0 = r1[0].key

    m3 = await _seed_sem(store, _READ_C, "s3")   # the cluster grows by one member
    # The regenerated body differs (it now covers m3) — grounded verbatim in _READ_C.
    body2 = "read tool timed out once then using the absolute path directly resolved"
    r2 = await write_cluster_schemas(store, _EchoLLM(body2), asyncio.Event())
    assert len(r2) == 1

    active = [f for f in await store.query_facts(FactQuery(tags=["tier:schema"], limit=50))]
    assert len(active) == 1, (
        f"membership drift minted a sibling chapter: {[(f.key, f.value[:40]) for f in active]}")
    assert active[0].key == key0                 # identity ADOPTED, not re-derived
    assert active[0].value == body2              # the refresh actually refreshed the body
    assert len(await store.get_fact_history(key0)) >= 2   # old body superseded, history kept
    # The refreshed chapter carries the NEW member.
    assert store._db is not None
    async with store._db.execute(
        "SELECT COUNT(*) FROM edges WHERE kind='member_of' AND src_id=? AND dst_id=?",
        (active[0].id, m3.id)) as cur:
        assert (await cur.fetchone())[0] == 1


@pytest.mark.asyncio
async def test_disjoint_clusters_never_cross_supersede(store):
    """Facet safety: refresh only fires on substantial member OVERLAP — two disjoint-topic
    chapters never adopt each other's identity; growing one leaves the other untouched."""
    a1 = await _seed_sem(store, _READ_A, "s1", topic="reads", child_tag="reads-x")
    a2 = await _seed_sem(store, _READ_B, "s2", topic="reads", child_tag="reads-x")
    b1 = await _seed_sem(store, "the kyoto ryokan trip is booked for early november this year",
                         "s1", topic="kyoto", child_tag="kyoto-x")
    b2 = await _seed_sem(store, "the kyoto onsen etiquette guide was saved for the november trip",
                         "s2", topic="kyoto", child_tag="kyoto-x")
    r1 = await write_cluster_schemas(store, _CorpusEchoLLM(), asyncio.Event())
    assert len(r1) == 2
    keys0 = {f.key for f in r1}

    await _seed_sem(store, _READ_C, "s3", topic="reads", child_tag="reads-x")  # grow reads only
    await write_cluster_schemas(store, _CorpusEchoLLM(), asyncio.Event())

    active = [f for f in await store.query_facts(FactQuery(tags=["tier:schema"], limit=50))]
    assert len(active) == 2, f"cross-supersede or sibling mint: {[(f.key, f.value[:40]) for f in active]}"
    assert {f.key for f in active} == keys0      # both identities stable across the refresh


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
    await _seed_sem(store, _READ_A, "s1", topic="reads")
    await _seed_sem(store, _READ_B, "s2", topic="reads")
    await _seed_sem(store, _DOCK_A, "s3", topic="docker")
    await _seed_sem(store, _DOCK_B, "s4", topic="docker")

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


# ---------------------------------------------------------------------------------------
# CHAPTER CONTAINMENT GUARD — set-contained duplicate chapters (write-time sibling seam)
# ---------------------------------------------------------------------------------------
# Root cause measured in validation-20260712-novelty070: a growing "subagents" cluster wrote a
# 12-member chapter (id 121) beside an earlier chapter (id 112) whose PRIMARY members were a
# strict subset — a nested-duplicate VIEW B2 graded at 0.667. _adopt_refresh_key only re-keys the
# SINGLE best-overlap chapter, so a second strictly-contained chapter is stranded live. Critically,
# the raw member_of edge sets were NOT subsets: id 112 also carried an aux surprising_failure member
# (id 52) absent from 121, so containment shows up ONLY on the PRIMARY (aux-excluded) member sets —
# the like-for-like set the candidate carries at guard time. These tests pin the guard on _write_one
# (the exact mint seam), driving synthetic clusters so the assertion is the guard's decision, not the
# small-pool tag-df clustering path.

async def _active_chapters(store) -> dict[int, str]:
    async with store._db.execute(
        "SELECT id, key FROM facts WHERE agent_id=? AND status='active' AND node_kind='schema'",
        (store._agent_id,)) as cur:
        return {r[0]: r[1] for r in await cur.fetchall()}


async def _status(store, fact_id):
    async with store._db.execute(
        "SELECT status, superseded_by FROM facts WHERE id=?", (fact_id,)) as cur:
        row = await cur.fetchone()
    return tuple(row) if row is not None else None


async def _seed_chapter(store, suffix, members, *, body="a pre-existing subagents chapter body", aux=()):
    """An active chapter (node_kind schema) with member_of edges to `members` (+ any `aux` rows) —
    the exact edge shape _consume_aux/_write_one emit. child_tag=None keeps it out of the co-tag graph
    (irrelevant here; _write_one is fed a synthetic cluster)."""
    ch = await _seed_schema(store, suffix, body, 1, "cluster:s0|s1", child_tag=None)
    for m in list(members) + list(aux):
        await store.add_edge(ch.id, m.id, "member_of")
    return ch


async def _subagent_atoms(store, n):
    """n distinct sem atoms (topic subagents), split across two sittings — fed to _write_one as a
    synthetic cluster's members (no clustering, so tag-df/pool size never enters)."""
    return [await _seed_sem(store, f"subagent build setting number {i} governs the harness order",
                            f"s{i % 2}", topic="subagents", child_tag=None) for i in range(n)]


@pytest.mark.asyncio
async def test_containment_supersedes_stranded_subset_chapter(store):
    """RETRO-FIXTURE (validation-20260712-novelty070): a 12-member candidate strictly contains an
    earlier 6-member chapter (in PRIMARY members) that _adopt_refresh_key stranded — it re-keyed the
    OTHER contained chapter (B, higher overlap) and left this one (A) a live duplicate. The guard
    supersedes A: exactly ONE active chapter, A's row kept (append-only, chained to the survivor).
    A also carries an aux surprising_failure member absent from the candidate, so a NAIVE full
    member_of comparison would miss the containment — only the PRIMARY sets are subset."""
    atoms = await _subagent_atoms(store, 12)
    aux = await _seed_failure(store, "bash_exec", value="a subagent bash_exec surprising failure noted")
    a = await _seed_chapter(store, "aaaaaaaa", atoms[:6], aux=(aux,))    # 6 primary + 1 aux (poisons naive full-set)
    b = await _seed_chapter(store, "bbbbbbbb", atoms[6:10])              # 4 primary — _adopt_refresh_key re-keys THIS one

    cluster = Cluster(members=atoms, sessions=frozenset({"s0", "s1"}), depth=0)
    counts: dict = {}
    schema = await _write_one(store, _CorpusEchoLLM(), asyncio.Event(), cluster, 1, 6000,
                              containment_counts=counts)

    assert schema is not None
    assert len(await _active_chapters(store)) == 1, "a strictly-contained chapter was stranded"
    assert await _status(store, a.id) == ("superseded", schema.id)      # A folded away, chained (history kept)
    assert (await _status(store, b.id))[0] == "superseded"              # B re-keyed away too
    assert counts.get("superseded", 0) >= 1                            # the guard logged its supersede


@pytest.mark.asyncio
async def test_containment_folds_subset_candidate_no_twin(store):
    """Reverse order: a richer chapter is already active when a strict-SUBSET cluster re-derives.
    The guard folds — no twin minted, the richer chapter preserved intact (never shrunk by a
    re-key), and the fold is counted."""
    members = await _subagent_atoms(store, 8)
    big = await _seed_chapter(store, "bigbig01", members, body="the full subagents chapter")
    cluster = Cluster(members=members[:4], sessions=frozenset({"s0", "s1"}), depth=0)   # strict subset
    counts: dict = {}
    schema = await _write_one(store, _CorpusEchoLLM(), asyncio.Event(), cluster, 1, 6000,
                              containment_counts=counts)

    assert schema is None                                              # no twin minted (folded)
    assert await _active_chapters(store) == {big.id: big.key}          # Big preserved, still the only chapter
    assert (await _status(store, big.id))[0] == "active"               # not shrunk / superseded by a re-key
    assert (await store.get_fact(big.key)).value == "the full subagents chapter"
    assert counts.get("folded", 0) == 1


@pytest.mark.asyncio
async def test_containment_equal_member_set_folds(store):
    """Equality is a fold: a STICKY-keyed chapter (its key != h8(members)) with EXACTLY the
    candidate's member set is preserved, the candidate not minted."""
    members = await _subagent_atoms(store, 4)
    ch = await _seed_chapter(store, "sticky01", members, body="the sticky equal chapter")
    cluster = Cluster(members=members, sessions=frozenset({"s0", "s1"}), depth=0)
    counts: dict = {}
    schema = await _write_one(store, _CorpusEchoLLM(), asyncio.Event(), cluster, 1, 6000,
                              containment_counts=counts)

    assert schema is None
    assert await _active_chapters(store) == {ch.id: ch.key}            # preserved, no twin
    assert (await store.get_fact(ch.key)).value == "the sticky equal chapter"
    assert counts.get("folded", 0) == 1


@pytest.mark.asyncio
async def test_containment_partial_overlap_leaves_both(store):
    """Genuine facet split (neither set contains the other): the guard does NOTHING — the candidate
    mints and the existing chapter is untouched, both coexist."""
    shared = await _subagent_atoms(store, 2)
    only_p = await _seed_sem(store, "a subagent item only in the existing chapter for harness",
                             "s0", topic="subagents", child_tag=None)
    only_c = await _seed_sem(store, "a subagent item only in the fresh candidate for the harness",
                             "s1", topic="subagents", child_tag=None)
    p = await _seed_chapter(store, "partial1", shared + [only_p], body="the partial existing chapter")
    cluster = Cluster(members=shared + [only_c], sessions=frozenset({"s0", "s1"}), depth=0)
    counts: dict = {}
    schema = await _write_one(store, _CorpusEchoLLM(), asyncio.Event(), cluster, 1, 6000,
                              containment_counts=counts, refresh_overlap=1.01)

    assert schema is not None                                          # facet split — the candidate mints
    assert (await _status(store, p.id))[0] == "active"                 # existing untouched
    assert (await store.get_fact(p.key)).value == "the partial existing chapter"
    assert len(await _active_chapters(store)) == 2                     # both coexist
    assert counts.get("folded", 0) == 0 and counts.get("superseded", 0) == 0


@pytest.mark.asyncio
async def test_containment_supersedes_multiple_contained(store):
    """Two DISJOINT active chapters each strictly inside the candidate are BOTH superseded (log
    each) — the exact gap _adopt_refresh_key left (it re-keys only the single best-overlap one).
    refresh_overlap raised so BOTH route through the guard, proving it handles multiplicity."""
    members = await _subagent_atoms(store, 4)
    a = await _seed_chapter(store, "multiaaa", members[:2])
    b = await _seed_chapter(store, "multibbb", members[2:4])
    cluster = Cluster(members=members, sessions=frozenset({"s0", "s1"}), depth=0)
    counts: dict = {}
    schema = await _write_one(store, _CorpusEchoLLM(), asyncio.Event(), cluster, 1, 6000,
                              containment_counts=counts, refresh_overlap=1.01)

    assert schema is not None
    assert await _status(store, a.id) == ("superseded", schema.id)
    assert await _status(store, b.id) == ("superseded", schema.id)
    assert len(await _active_chapters(store)) == 1
    assert counts.get("superseded", 0) == 2


@pytest.mark.asyncio
async def test_containment_guard_off_preserves_twin_writing(store):
    """The kill lever: guard OFF restores byte-old behaviour — a strictly-contained chapter is NOT
    superseded (with re-key also disabled, both strays survive beside the fresh candidate)."""
    members = await _subagent_atoms(store, 4)
    a = await _seed_chapter(store, "offaaaa1", members[:2])
    b = await _seed_chapter(store, "offbbbb1", members[2:4])
    cluster = Cluster(members=members, sessions=frozenset({"s0", "s1"}), depth=0)
    counts: dict = {}
    schema = await _write_one(store, _CorpusEchoLLM(), asyncio.Event(), cluster, 1, 6000,
                              containment_guard=False, containment_counts=counts, refresh_overlap=1.01)

    assert schema is not None
    assert (await _status(store, a.id))[0] == "active"                 # untouched — guard inert
    assert (await _status(store, b.id))[0] == "active"
    assert len(await _active_chapters(store)) == 3                     # candidate + both strays
    assert counts.get("superseded", 0) == 0 and counts.get("folded", 0) == 0


@pytest.mark.asyncio
async def test_containment_and_incest_guards_converge_no_churn(store):
    """COMPOSITION (--idle-passes style, full write_cluster_schemas so clustering's incest guard is
    live): the write-time guards converge to ONE stable chapter across repeated idle passes with NO
    churn. INVERTED for #67: the stranded chapter's PRIMARY member is a strict subset of the 3-atom
    cluster; before the fix its aux surprising_failure rows diluted the adoption FULL-set overlap below
    threshold, so adoption LEFT it and the CONTAINMENT guard superseded it (counts.superseded==1, a
    fresh chapter minted beside it, its identity lost). With the aux-excluded primary-member map,
    adoption's overlap is 1.0 and pass 1 REFRESHES it (adopt its key — identity continuity, history
    preserved); the containment guard is inert here (adoption handled it — its multi-contained role is
    covered by test_containment_supersedes_multiple_contained). Every later pass changes nothing
    (store_fact corroborates on the shared key)."""
    atoms = [await _seed_sem(store, f"subagent convergence item {i} for the harness build order",
                             f"s{i % 2}", topic="subagents") for i in range(3)]   # default child tag -> they cluster
    stranded = await _seed_chapter(
        store, "stranded1", atoms[:1],
        aux=(await _seed_failure(store, "bash_exec", day="20260701"),
             await _seed_failure(store, "docker", day="20260702")))               # once diluted overlap; now inert (#67)

    # Pass 1: the 3-atom cluster REFRESHES the stranded 1-primary-member chapter (adoption, primary
    # overlap 1.0) -> one active chapter under the SAME key, its history preserved.
    counts1: dict = {}
    r1 = await write_cluster_schemas(store, _CorpusEchoLLM(), asyncio.Event(), containment_counts=counts1)
    assert len(r1) == 1
    active1 = await _active_chapters(store)
    assert len(active1) == 1, f"stranded chapter siblinged or survived pass 1: {active1}"
    assert (await _status(store, stranded.id))[0] == "superseded"       # old stranded row folded away
    assert active1[r1[0].id] == stranded.key                            # identity ADOPTED, not re-derived
    assert len(await store.get_fact_history(stranded.key)) == 2         # supersede chain, history kept
    assert counts1.get("superseded", 0) == 0 and counts1.get("folded", 0) == 0  # adoption, not the guard

    # Passes 2 & 3: stable. The survivor now holds the adopted (sticky) key while the re-formed cluster
    # equals its members, so the guard FOLDS it (a harmless corroborate touch — one stable chapter, no
    # mint, no new row). The no-churn invariant is on STATE: the active set and history never change.
    for _ in range(2):
        counts: dict = {}
        await write_cluster_schemas(store, _CorpusEchoLLM(), asyncio.Event(), containment_counts=counts)
        assert await _active_chapters(store) == active1, "a later pass churned the chapter set"
        assert counts.get("superseded", 0) == 0, "supersede on a stable pass would be churn"
        assert len(await store.get_fact_history(stranded.key)) == 2  # no new supersede row — touch only


# ---------------------------------------------------------------------------------------
# CHAPTER STALENESS RE-CHECK — a chapter's evidentiary base can erode after it is written
# ---------------------------------------------------------------------------------------
# Root cause measured in d1-replication-20260712 (ANALYSIS §7, "The 7-Day Taper" / B5): a chapter
# was written correctly and grounded against 3 members, one of which was the SOLE bearer of the
# number "7". That member atom was later superseded; get_facts_by_ids is active-only, so the chapter
# silently lost its sole "7" source and sat ACTIVE with a figure nothing supported — the eval's
# grader (which renders only ACTIVE members) then KILLed the run. recheck_stale_chapters catches this
# BEFORE the writer runs each idle pass: it re-runs the grader's own grounded()/ground_numbers()
# matchers against each chapter's CURRENT active members and, on a fail, re-drafts on the survivors
# (grounded → supersede on the chapter's key) or retires the chapter (append-only, never deleted).

# The B5 fixture, verbatim-shaped: M3 is the sole number-bearer; the body grounds by MAJORITY token
# on M1+M2 alone (so the majority net stays green post-erosion) but carries "7" (only in M3) — so
# after M3 is superseded the re-check reproduces the exact B5 signature grounded_majority=True,
# unverified_numbers=["7"]. "taper" (5 chars) is below the >=6 grounding-token floor, so it never
# props up the majority; the isolation is the numeric net, exactly as in the run.
_TAPER_M1 = "the runner trains on tuesdays and saturdays every week"
_TAPER_M2 = "the runner is preparing for a september marathon race"
_TAPER_M3 = "the coach added a 7 day taper to the training plan"
_TAPER_BODY = ("The runner is preparing for the september marathon and trains "
               "tuesdays saturdays with a 7 day taper")


async def _seed_taper_chapter(store):
    """The '7-Day Taper' chapter: grounded on M1/M2/M3 at write time, M3 the sole '7' source."""
    m1 = await _seed_sem(store, _TAPER_M1, "s1", topic="fitness", child_tag=None)
    m2 = await _seed_sem(store, _TAPER_M2, "s2", topic="fitness", child_tag=None)
    m3 = await _seed_sem(store, _TAPER_M3, "s1", topic="fitness", child_tag=None)
    ch = await _seed_chapter(store, "taper001", [m1, m2, m3], body=_TAPER_BODY)
    return ch, m1, m2, m3


async def _erode(store, member, new_value):
    """Supersede a member on its own key (the real B5 mechanism — a same-key supersede flips the
    old row's status away from active), so the chapter's member_of edge now points at a superseded
    (un-rendered) node exactly as it did live."""
    await store.store_fact(key=member.key, value=new_value, tags=["sem", "pending_consolidation"],
                           confidence=0.65, source="transcript_mining", provenance="s1",
                           node_kind="fact")


def _grounded_now(chapter_value, member_bodies):
    """Grader-equivalent verdict: True iff the body still grounds against these member bodies
    (mirrors sema05 _static_checks: majority-token AND clean numeric net, body-only)."""
    corpus = "\n".join(member_bodies)
    body = strip_chapter_title(chapter_value)
    return grounded(body, corpus) and not ground_numbers(body, member_bodies)


@pytest.mark.asyncio
async def test_stale_chapter_redrafted_without_orphaned_number(store):
    """B5, faithful writer: the sole number-bearer (M3) is superseded, so the chapter's "7" is no
    longer grounded. A faithful re-draft on the SURVIVING members (M1,M2) drops the orphaned figure
    and SUPERSEDES the stale chapter ON ITS OWN KEY (normal supersede, history preserved) — the
    exclude-self path, without which the re-draft would fold back into the very chapter it replaces.
    Load-bearing: the pre-erosion body grounds (M3 carries 7) and the post-erosion body does NOT."""
    from localharness.memory.chapter_writer import recheck_stale_chapters

    ch, m1, m2, m3 = await _seed_taper_chapter(store)
    assert _grounded_now(_TAPER_BODY, [_TAPER_M1, _TAPER_M2, _TAPER_M3])       # grounded at write time
    await _erode(store, m3, "the coach revised the plan and dropped the extra week entirely")
    assert not _grounded_now(_TAPER_BODY, [_TAPER_M1, _TAPER_M2])              # now stale (orphaned 7)

    counts: dict = {}
    await recheck_stale_chapters(store, _CorpusEchoLLM(), asyncio.Event(), counts=counts)

    assert counts.get("redrafted", 0) == 1 and counts.get("retired", 0) == 0
    assert (await _status(store, ch.id))[0] == "superseded"        # stale row folded away (append-only)
    live = await store.get_fact(ch.key)                            # re-draft adopted the SAME key
    assert live is not None and live.id != ch.id and live.status == "active"
    assert "7" not in live.value                                  # the orphaned figure is gone
    assert len(await store.get_fact_history(ch.key)) == 2         # supersede chain, history kept
    assert len(await _active_chapters(store)) == 1
    # Members re-attached to the fresh chapter are the SURVIVORS only.
    assert await _member_of_dst(store, live.id) == {m1.id, m2.id}
    # Grader-equivalent re-check now comes back CLEAN — a second pass revalidates, no more churn.
    counts2: dict = {}
    await recheck_stale_chapters(store, _CorpusEchoLLM(), asyncio.Event(), counts=counts2)
    assert counts2.get("revalidated", 0) == 1 and counts2.get("redrafted", 0) == 0
    assert len(await store.get_fact_history(ch.key)) == 2         # no new supersede on the clean pass


@pytest.mark.asyncio
async def test_stale_chapter_retired_when_redraft_refuses(store):
    """B5, refusing writer: when the re-draft cannot produce a grounded chapter from the survivors
    (the writer's hallucination kill fires), the stale chapter is RETIRED — marked non-active
    (append-only, never deleted), leaving NO active chapter for the grader to KILL on. Clean either
    way: the eval grades only active schemas, so a retired chapter contributes no B5."""
    from localharness.memory.chapter_writer import recheck_stale_chapters

    ch, m1, m2, m3 = await _seed_taper_chapter(store)
    await _erode(store, m3, "the coach revised the plan and dropped the extra week entirely")

    counts: dict = {}
    await recheck_stale_chapters(
        store, _EchoLLM("the deployment shipped kubernetes clusters across datacenters"),
        asyncio.Event(), counts=counts)

    assert counts.get("retired", 0) == 1 and counts.get("redrafted", 0) == 0
    assert (await _status(store, ch.id)) == ("superseded", None)   # retired: non-active, NO successor
    assert await _active_chapters(store) == {}                     # nothing for the grader to KILL
    assert len(await store.get_fact_history(ch.key)) == 1          # the row is kept, never deleted
    # A second pass sees no active chapter — a stable, idempotent retire (no resurrection).
    counts2: dict = {}
    await recheck_stale_chapters(store, _CorpusEchoLLM(), asyncio.Event(), counts=counts2)
    assert counts2 == {} or all(v == 0 for v in counts2.values())


@pytest.mark.asyncio
async def test_healthy_chapter_idempotent_no_writes(store):
    """A pass over an already-healthy chapter must write no NEW row and change nothing SUBSTANTIVE —
    same id, status, value, history length (the idempotency/byte-stability law). INVERTED for #68: a
    healthy revalidation now ADVANCES updated_at (the recheck cursor) so the chapter rotates out of the
    oldest-first window; the bump is freshness-only (no supersede, value/id/history unchanged)."""
    from localharness.memory.chapter_writer import recheck_stale_chapters

    h1 = await _seed_sem(store, _READ_A, "s1", topic="reads", child_tag=None)
    h2 = await _seed_sem(store, _READ_B, "s2", topic="reads", child_tag=None)
    ch = await _seed_chapter(store, "healthy1", [h1, h2], body=_GROUNDED)  # _GROUNDED grounds on _READ_A
    await _set_updated_at(store, ch.id, 1000)                              # a fixed past cursor
    before = await store.get_fact(ch.key)

    for _ in range(2):
        counts: dict = {}
        await recheck_stale_chapters(store, _CorpusEchoLLM(), asyncio.Event(), counts=counts)
        assert counts.get("revalidated", 0) == 1
        assert counts.get("redrafted", 0) == 0 and counts.get("retired", 0) == 0
        after = await store.get_fact(ch.key)
        assert after.id == before.id and after.status == "active"
        assert after.value == before.value                       # content unchanged (no new row)
        assert after.updated_at > before.updated_at              # #68: cursor advanced (rotates the window)
        assert len(await store.get_fact_history(ch.key)) == 1     # no supersede row minted
    assert len(await _active_chapters(store)) == 1


@pytest.mark.asyncio
async def test_zero_active_members_retires(store):
    """A chapter whose members ALL eroded (< 2 active members) is retired — no re-draft is even
    attempted (there is nothing to re-draft from), so the writer is never called."""
    from localharness.memory.chapter_writer import recheck_stale_chapters

    m1 = await _seed_sem(store, _TAPER_M1, "s1", topic="fitness", child_tag=None)
    m2 = await _seed_sem(store, _TAPER_M2, "s2", topic="fitness", child_tag=None)
    ch = await _seed_chapter(store, "empty001", [m1, m2],
                             body="the special reserved allocation totals distinct measurable units")
    await _erode(store, m1, "unrelated content one two three")
    await _erode(store, m2, "unrelated content four five six")

    counts: dict = {}
    called = {"n": 0}

    class _Tripwire:
        async def complete(self, prompt):        # the writer must NEVER be called (no re-draft)
            called["n"] += 1
            return "should not run"

    await recheck_stale_chapters(store, _Tripwire(), asyncio.Event(), counts=counts)

    assert called["n"] == 0                       # zero-member chapter retires without a generation
    assert counts.get("retired", 0) == 1
    assert (await _status(store, ch.id)) == ("superseded", None)
    assert await _active_chapters(store) == {}
    assert len(await store.get_fact_history(ch.key)) == 1        # append-only


@pytest.mark.asyncio
async def test_recheck_cap_bounds_work_per_pass(store):
    """The re-check is bounded: with `cap` chapters processed oldest-first (updated_at ASC, id ASC),
    a store with more stale chapters than the cap processes exactly `cap` of them and leaves the rest
    for later passes — the newest-written chapter (largest id) survives an under-cap pass untouched."""
    from localharness.memory.chapter_writer import recheck_stale_chapters

    chapters = []
    for i, suf in enumerate(("capA", "capB", "capC")):
        m = await _seed_sem(store, f"an unrelated weather note about clouds and rainfall patterns {suf}",
                            f"s{i}", topic="misc", child_tag=None)
        ch = await _seed_chapter(store, suf, [m],
                                 body=f"the reserved special allocation totals {40 + i} distinct units")
        chapters.append(ch)

    counts: dict = {}
    await recheck_stale_chapters(store, _CorpusEchoLLM(), asyncio.Event(), cap=2, counts=counts)

    assert counts.get("retired", 0) == 2                          # exactly cap processed, all stale
    active = await _active_chapters(store)
    assert len(active) == 1 and chapters[-1].id in active         # the newest chapter is left for later


# ---------------------------------------------------------------------------
# SHARED ACTIVE-PRIMARY MEMBER MAP — #66 (status filter) + #67 (aux-free adoption) + #71 (build-once)
# ---------------------------------------------------------------------------
# The containment guard and refresh adoption must compare the SAME member sets on both sides: ACTIVE
# members (superseded members excluded — #66) with aux tier:surprising_failure rows excluded (#67).
# Before the fix, adoption scored overlap from RAW member_of (dead members + aux inflating the min()
# denominator), deflating a legitimate refresh below threshold -> a duplicate sibling minted beside the
# still-active original. And the guard re-derived every chapter's member set per _write_one call (#71).

async def _erode_member(store, member, new_value="wholly unrelated content one two three four five"):
    """Supersede a member on its own key (status flips away from active) — the real erosion mechanism."""
    await store.store_fact(key=member.key, value=new_value, tags=["sem", "pending_consolidation"],
                           confidence=0.65, source="transcript_mining", provenance="sx", node_kind="fact")


@pytest.mark.asyncio
async def test_adoption_ignores_aux_rows_no_duplicate_sibling(store):
    """#67: a chapter with ONE primary member + aux surprising_failure rows; a grown cluster (that
    primary + a new atom) must REFRESH it (adopt its key), not mint a duplicate sibling. RAW member_of
    (primary+aux) inflates the overlap min() denominator so the score falls below threshold; the
    primary-only map clears it. Load-bearing: exactly ONE active chapter after the write."""
    a1 = await _seed_sem(store, "a subagent build setting alpha governs the harness order", "s0",
                         topic="subagents", child_tag=None)
    a2 = await _seed_sem(store, "a subagent build setting bravo governs the harness order", "s1",
                         topic="subagents", child_tag=None)
    f1 = await _seed_failure(store, "bash_exec", day="20260701", value="a subagent bash failure one")
    f2 = await _seed_failure(store, "docker", day="20260702", value="a subagent docker failure two")
    x = await _seed_chapter(store, "auxchap1", [a1], aux=(f1, f2))    # 1 primary + 2 aux

    cluster = Cluster(members=[a1, a2], sessions=frozenset({"s0", "s1"}), depth=0)  # grew by a2
    schema = await _write_one(store, _CorpusEchoLLM(), asyncio.Event(), cluster, 1, 6000)

    assert schema is not None
    active = await _active_chapters(store)
    assert len(active) == 1, f"aux-deflated overlap minted a duplicate sibling: {active}"
    assert x.key in active.values()                  # the original's identity was refreshed, not siblinged


@pytest.mark.asyncio
async def test_eroded_chapter_refreshed_not_siblinged(store):
    """#66: a chapter whose members are partly DEAD (superseded) must be compared by its ACTIVE members
    on both sides. A grown cluster over its ONE surviving member + a new atom refreshes it (adopt key),
    not mint a sibling beside the eroded original. Dead members inflate the raw set so the raw overlap
    deflates below threshold; the active-only map clears it. (The containment FOLD direction is proven
    status-invariant for active drafts — draft ⊆ raw ⟺ draft ⊆ active — so the demonstrable effect of
    the status filter is exactly here, on adoption/supersede overlap.)"""
    a1 = await _seed_sem(store, "a subagent build setting alpha governs the harness order", "s0",
                         topic="subagents", child_tag=None)
    a2 = await _seed_sem(store, "a subagent build setting bravo governs the harness order", "s1",
                         topic="subagents", child_tag=None)
    a3 = await _seed_sem(store, "a subagent build setting charlie governs the harness order", "s0",
                         topic="subagents", child_tag=None)
    x = await _seed_chapter(store, "erochap1", [a1, a2, a3])   # 3 primary members
    await _erode_member(store, a2)     # a2, a3 superseded -> X_active shrinks to {a1}
    await _erode_member(store, a3)
    b1 = await _seed_sem(store, "a subagent build setting delta governs the harness order", "s1",
                         topic="subagents", child_tag=None)

    cluster = Cluster(members=[a1, b1], sessions=frozenset({"s0", "s1"}), depth=0)  # {a1} survivor + b1
    schema = await _write_one(store, _CorpusEchoLLM(), asyncio.Event(), cluster, 1, 6000)

    assert schema is not None
    active = await _active_chapters(store)
    assert len(active) == 1, f"eroded chapter left a live duplicate sibling: {active}"
    assert x.key in active.values()


@pytest.mark.asyncio
async def test_member_map_derived_once_per_pass(store):
    """#71: the active-primary member map is built ONCE per write_cluster_schemas pass and threaded into
    every _write_one (guard + adoption) — not re-derived per write. Patch the builder to count
    invocations; with existing chapters + three writable clusters it must be called exactly once."""
    import localharness.memory.chapter_writer as cw
    for i in range(3):                                            # existing chapters (per-write rebuild = costly)
        m = await _seed_sem(store, f"an existing note about topic {i} kept for the harness record",
                            f"e{i}", topic=f"exist{i}", child_tag=None)
        await _seed_chapter(store, f"exist{i}ch", [m])
    for stem in ("pp", "qq", "rr"):                              # three independent writable clusters
        await _seed_sem(store, f"{stem}zzzalpha {stem}zzzbravo {stem}zzzcharlie", f"{stem}0", topic=f"t{stem}")
        await _seed_sem(store, f"{stem}zzzalpha {stem}zzzbravo {stem}zzzdelta", f"{stem}1", topic=f"t{stem}")

    calls = {"n": 0}
    orig = cw._active_chapter_primary_members

    async def _counting(s):
        calls["n"] += 1
        return await orig(s)

    cw._active_chapter_primary_members = _counting
    try:
        result = await write_cluster_schemas(store, _CorpusEchoLLM(), asyncio.Event(), write_budget=3)
    finally:
        cw._active_chapter_primary_members = orig

    assert len(result) == 3                                      # three clusters written
    assert calls["n"] == 1, f"member map re-derived {calls['n']}x (must be once per pass, #71)"


# ---------------------------------------------------------------------------
# #64 (CRITICAL) — a heal must PRESERVE the chapter's identity on an overlap tie
# ---------------------------------------------------------------------------
# The recheck redraft re-derives its supersede target by unconstrained best-overlap with no preference
# for the chapter being healed. A chapter's own surviving-member set self-scores 1.0; a near-duplicate
# also scoring 1.0 can win by scan order — the healed content then adopts the WRONG chapter's key and
# the original is retired at superseded_by=NULL (history dead-ends, content misattributed). The fix
# threads the original's key into adoption, HARD-PREFERS it (a different key only when STRICTLY better),
# and records the successor on every retire.

@pytest.mark.asyncio
async def test_stale_heal_preserves_identity_on_overlap_tie(store):
    """#64: a near-dup chapter scoring the SAME 1.0 overlap must NOT hijack a heal's identity. The
    redraft of an eroded chapter O adopts O'S OWN key (hard-prefer) — superseding O on its key with
    history preserved and a successor recorded — never the near-dup's key with O dead-ended. The near-
    dup is seeded FIRST so the unconstrained best-overlap scan would land on it (the pre-fix bug)."""
    from localharness.memory.chapter_writer import recheck_stale_chapters
    m1 = await _seed_sem(store, _READ_A, "s1", topic="reads", child_tag=None)
    # Near-dup D: a grounded single-member {m1} chapter — revalidates (survives) and ties at overlap 1.0
    # with O's survivors. Seeded FIRST (lower id) -> first in the best-overlap scan (pre-fix would pick it).
    dup = await _seed_chapter(store, "neardup1", [m1], body=_GROUNDED)
    m2 = await _seed_sem(store, _READ_B, "s2", topic="reads", child_tag=None)
    m3 = await _seed_sem(store, "the read retry took 7 attempts before the absolute path resolved", "s1",
                         topic="reads", child_tag=None)
    o_body = "read tool returned FileNotFound retrying with the absolute path resolved after 7"
    orig = await _seed_chapter(store, "origchap", [m1, m2, m3], body=o_body)   # grounds on m1/m2, "7" from m3
    await _erode_member(store, m3, "the read retry succeeded once the working path was corrected")  # orphan the 7

    counts: dict = {}
    await recheck_stale_chapters(store, _CorpusEchoLLM(), asyncio.Event(), counts=counts)

    assert counts.get("redrafted", 0) == 1
    live = await store.get_fact(orig.key)                        # healed content under O's OWN key
    assert live is not None and live.status == "active" and live.id != orig.id
    assert live.key == orig.key
    assert await _status(store, orig.id) == ("superseded", live.id)   # successor recorded — NOT a dead-end
    assert (await store.get_fact(dup.key)) is None               # near-dup did NOT receive the healed identity
    assert len(await _active_chapters(store)) == 1               # exactly one active chapter (O, healed)


@pytest.mark.asyncio
async def test_adopt_hard_prefers_original_but_takes_strictly_better(store):
    """#64: _adopt_refresh_key hard-prefers the ORIGINAL's key on an overlap TIE, and adopts a DIFFERENT
    chapter's key ONLY when strictly better. Direct on the adoption function with controlled maps."""
    from localharness.memory.chapter_writer import _adopt_refresh_key
    m1 = await _seed_sem(store, _READ_A, "s1", child_tag=None)
    m2 = await _seed_sem(store, _READ_B, "s2", child_tag=None)
    members = [m1, m2]                                           # new_ids = {m1, m2}
    korig, kdup, kbetter, kfresh = ("schema/cluster/orig", "schema/cluster/dup",
                                    "schema/cluster/better", "schema/cluster/fresh")

    # TIE: original {m1,m2} (ov 1.0) vs near-dup {m1} (ov 1.0), dup listed FIRST. prefer_key wins.
    tie_map = {11: (kdup, frozenset({m1.id})), 10: (korig, frozenset({m1.id, m2.id}))}
    key = await _adopt_refresh_key(store, members, kfresh, refresh_overlap=0.7, claimed=set(),
                                   chapters=tie_map, prefer_key=korig)
    assert key == korig                                         # tie -> original's identity kept, not the dup's

    # STRICTLY BETTER: original {m1, phantom} (ov 0.5) vs other {m2} (ov 1.0). the other wins.
    better_map = {10: (korig, frozenset({m1.id, 9_999_999})), 12: (kbetter, frozenset({m2.id}))}
    key2 = await _adopt_refresh_key(store, members, kfresh, refresh_overlap=0.7, claimed=set(),
                                    chapters=better_map, prefer_key=korig)
    assert key2 == kbetter                                      # strictly-better other adopted


@pytest.mark.asyncio
async def test_retire_chapter_records_successor(store):
    """#64: _retire_chapter can record a successor (no superseded_by=NULL dead-end) so the heal's retire
    of the original points to its replacement; the default (no successor) keeps the plain-retire form."""
    from localharness.memory.chapter_writer import _retire_chapter
    m1 = await _seed_sem(store, _READ_A, "s1", child_tag=None)
    ch = await _seed_chapter(store, "retire01", [m1])
    succ = await _seed_chapter(store, "retire02", [m1])
    assert await _retire_chapter(store, ch.id, successor=succ.id)
    assert await _status(store, ch.id) == ("superseded", succ.id)   # successor recorded, not NULL
    ch2 = await _seed_chapter(store, "retire03", [m1])
    assert await _retire_chapter(store, ch2.id)                     # default: plain retire, no successor
    assert await _status(store, ch2.id) == ("superseded", None)


# ---------------------------------------------------------------------------
# #68 — recheck starvation: a HEALTHY revalidation must rotate the oldest-first window
# ---------------------------------------------------------------------------
# The re-check selects ORDER BY updated_at ASC LIMIT cap, but a healthy revalidation writes nothing —
# updated_at untouched — so once cap perpetually-grounded chapters exist they fill EVERY window forever
# and erosion outside the frozen set is never detected. The fix advances the chapter's updated_at on a
# healthy revalidation (mirroring _corroborate_chapter) so it rotates to the back.

async def _set_updated_at(store, fact_id, ts):
    await store._db.execute("UPDATE facts SET updated_at=? WHERE id=?", (ts, fact_id))
    await store._db.commit()


@pytest.mark.asyncio
async def test_healthy_revalidation_rotates_recheck_window(store):
    """#68: cap=2 — pass 1 revalidates the two OLDEST (healthy) chapters and must bump them to the back;
    pass 2 then reaches the two STALE chapters it could not before and retires them. Without the bump the
    healthy pair stays oldest forever and the stale pair starves (never rechecked)."""
    from localharness.memory.chapter_writer import recheck_stale_chapters
    healthy = []
    for i, suf in enumerate(("winA", "winB")):                   # HEALTHY, oldest by updated_at
        h1 = await _seed_sem(store, _READ_A, f"h{i}", topic=f"win{i}", child_tag=None)
        h2 = await _seed_sem(store, _READ_B, f"h{i}b", topic=f"win{i}", child_tag=None)
        ch = await _seed_chapter(store, suf, [h1, h2], body=_GROUNDED)   # grounds on _READ_A
        await _set_updated_at(store, ch.id, 1000 + i)
        healthy.append(ch)
    stale = []
    for i, suf in enumerate(("winC", "winD")):                   # STALE 1-member (ungrounded -> retire), newer
        m = await _seed_sem(store, f"an unrelated cloud note {suf} for the record", f"z{i}",
                            topic=f"z{i}", child_tag=None)
        ch = await _seed_chapter(store, suf, [m],
                                 body="the special reserved allocation totals distinct measurable units")
        await _set_updated_at(store, ch.id, 1010 + i)
        stale.append(ch)

    c1: dict = {}
    await recheck_stale_chapters(store, _CorpusEchoLLM(), asyncio.Event(), cap=2, counts=c1)
    assert c1.get("revalidated", 0) == 2 and c1.get("retired", 0) == 0   # pass 1 hit the two healthy

    c2: dict = {}
    await recheck_stale_chapters(store, _CorpusEchoLLM(), asyncio.Event(), cap=2, counts=c2)
    assert c2.get("retired", 0) == 2, "starvation: pass 2 never reached the stale chapters"
    for ch in stale:
        assert (await _status(store, ch.id))[0] == "superseded"          # the stale pair was finally reached
    for ch in healthy:
        assert (await _status(store, ch.id))[0] == "active"              # healthy pair untouched by pass 2


# ---------------------------------------------------------------------------
# #70 — recheck must re-read status: an earlier heal can kill a later snapshot item mid-pass
# ---------------------------------------------------------------------------

class _CountingCorpusLLM:
    """_CorpusEchoLLM that COUNTS 'Write ONE' generations — proves how many redrafts actually ran."""
    def __init__(self):
        self.calls = 0

    async def complete(self, prompt: str) -> str:
        self.calls += 1
        corpus = prompt.split("\n\n")[-1]
        lines = [ln for ln in corpus.splitlines() if ln.strip()]
        return " ".join(lines[0].split()[:12]) if lines else "chapter"


@pytest.mark.asyncio
async def test_recheck_skips_item_superseded_mid_pass(store):
    """#70: the recheck snapshot is fetched once; an earlier item's redraft can supersede a LATER item
    via the containment guard. The now-dead chapter must be SKIPPED (status re-read) — no wasted LLM
    call, no resurrection. O1's redraft (survivors {M1,M2,a3}) strictly contains O2 ({M1,M2}) and
    supersedes it mid-pass; O2, processed next, is skipped."""
    from localharness.memory.chapter_writer import recheck_stale_chapters
    m1 = await _seed_sem(store, _TAPER_M1, "s1", topic="fit", child_tag=None)
    m2 = await _seed_sem(store, _TAPER_M2, "s2", topic="fit", child_tag=None)
    a3 = await _seed_sem(store, "the runner does hill repeats on the steep northern route", "s1",
                         topic="fit", child_tag=None)
    a4 = await _seed_sem(store, _TAPER_M3, "s1", topic="fit", child_tag=None)   # the "7" bearer
    o1 = await _seed_chapter(store, "snapO1", [m1, m2, a3, a4], body=_TAPER_BODY)  # grounds M1/M2/a3, carries 7
    o2 = await _seed_chapter(store, "snapO2", [m1, m2],
                             body="The runner trains tuesdays saturdays preparing the marathon with 9 sessions")
    await _set_updated_at(store, o1.id, 1000)     # O1 processed first
    await _set_updated_at(store, o2.id, 1001)
    await _erode_member(store, a4, "the coach revised the plan and dropped the extra week")  # orphan O1's 7

    llm = _CountingCorpusLLM()
    counts: dict = {}
    await recheck_stale_chapters(store, llm, asyncio.Event(), counts=counts)

    assert counts.get("redrafted", 0) == 1                    # O1 healed
    assert counts.get("skipped_superseded", 0) == 1          # O2 was killed mid-pass by O1 -> skipped
    assert llm.calls == 1                                     # only O1's redraft generated — O2 spent none
    assert (await _status(store, o2.id))[0] == "superseded"  # O2 stays dead (no resurrection)
    assert len(await _active_chapters(store)) == 1           # just the healed O1


# ---------------------------------------------------------------------------
# #69 — recheck and writer must share ONE claimed-refresh-key set per pass
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_writer_skips_a_shared_claimed_refresh_key(store):
    """#69: write_cluster_schemas honors a SHARED claimed-refresh-key set (the set the recheck fills
    earlier in the same pass). A grown cluster that would adopt an already-claimed chapter key K falls
    back to a fresh key instead of re-adopting K and clobbering the vetted heal. Containment off to
    isolate the adoption path; the pre-populated set stands in for the recheck's claim."""
    a1 = await _seed_sem(store, _READ_A, "s0", topic="reads")
    a2 = await _seed_sem(store, _READ_B, "s1", topic="reads")
    a3 = await _seed_sem(store, _READ_C, "s0", topic="reads")     # the cluster grows to {a1,a2,a3}
    kc = await _seed_chapter(store, "claimk01", [a1, a2], body="the vetted heal content under K")

    result = await write_cluster_schemas(store, _CorpusEchoLLM(), asyncio.Event(),
                                         containment_guard=False, claimed_refresh_keys={kc.key})
    assert len(result) == 1
    assert result[0].key != kc.key                                # claimed key NOT re-adopted
    assert (await store.get_fact(kc.key)).value == "the vetted heal content under K"  # heal preserved

"""Stage C — v1 discovery + Bayesian evidence ladder + model NAMEing (Amendment 3/4).

Discovery is DETERMINISTIC: candidate child tags emerge from MULTI-FACTOR agreement over
bucket-only atoms (temporal co-occurrence + embedding proximity via a pluggable embedder +
trace co-activation), never a single factor (the mega-blob lesson). Candidates accrue evidence
(distinct sittings, trace reuse) with recency decay; at threshold ONE model call NAMEs the group
(stem-dedup folds into an existing tag; garbage stays a candidate) -> a new active discovered
child tag. Candidates that stop accruing decay below the floor and are pruned. The model's ONLY
creative act is NAMEing an already-bounded group. Tests inject a FAKE embedder — the interface
is the point.
"""
import asyncio
import time

import pytest

from localharness.memory.discovery import _NAME_MARKER, discover_tags
from localharness.memory.sqlite import MemoryStore


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(agent_id="disc-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


class _OneHotEmbedder:
    """Deterministic fake embedder: a one-hot vector over `vocab` (a text embeds to basis i iff it
    contains vocab[i]). Same-keyword texts cosine to 1.0, different/absent to 0.0 — full control."""

    def __init__(self, vocab):
        self.vocab = vocab

    def embed(self, texts):
        out = []
        for t in texts:
            low = t.lower()
            out.append([1.0 if w in low else 0.0 for w in self.vocab])
        return out


class _NamerLLM:
    """Fake namer: returns a fixed name only for the NAME prompt (discovery's one model call)."""

    def __init__(self, name="hardware"):
        self.name = name

    async def complete(self, prompt: str) -> str:
        return self.name if _NAME_MARKER in prompt else ""


async def _seed_bucket_atom(store, value, session, *, bucket="project", conf=0.65):
    """A bucket-only semantic atom: a sem/ atom filed under `bucket` with NO child tag yet —
    exactly the discovery pool population (mint-time 'none fit')."""
    import hashlib
    h = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    f = await store.store_fact(key=f"sem/disc/{h}", value=value, tags=["sem", "pending_consolidation"],
                               confidence=conf, source="transcript_mining", provenance=session,
                               node_kind="fact")
    b = await store.get_tag(bucket)
    await store.add_atom_tag(f.id, b.id, "mint")
    return f


async def _cofire(store, atoms):
    await store.record_activation_trace(stimulus="probe", fired_ids=[a.id for a in atoms],
                                        injected_ids=[a.id for a in atoms], source="memory_search")


# ---------------------------------------------------------------------------
# Incorporation: a multi-factor candidate over the floor is NAMEd -> active tag
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discovery_incorporates_multi_factor_candidate(store):
    proj = await store.get_tag("project")
    a = await _seed_bucket_atom(store, "clusterx alpha detail", "designed-day1")
    b = await _seed_bucket_atom(store, "clusterx bravo detail", "designed-day2")
    c = await _seed_bucket_atom(store, "clusterx gamma detail", "designed-day3")
    await _cofire(store, [a, b, c])  # trace co-activation factor
    emb = _OneHotEmbedder(["clusterx"])  # embedding-proximity factor (cosine 1.0 within the group)

    report = await discover_tags(store, _NamerLLM("hardware"), asyncio.Event(), embedder=emb)

    assert "hardware" in report.incorporated
    tag = await store.get_tag("hardware")
    assert tag is not None and tag.status == "active" and tag.origin == "discovered"
    assert tag.parent_id == proj.id                       # child of the right bucket
    assert "hardware" in {t.name for t in await store.tags_for_atom(a.id)}
    provs = {r.provenance for r in await store.atom_tag_rows(a.id)}
    assert "discovery" in provs                           # members filed by discovery


@pytest.mark.asyncio
async def test_single_factor_forms_no_candidate(store):
    """The mega-blob guard generalized to discovery: ONE factor (embedding only) must not weld a
    group — no candidate, no incorporation."""
    await _seed_bucket_atom(store, "clusterx alpha", "designed-day1")
    await _seed_bucket_atom(store, "clusterx bravo", "designed-day3")  # different, non-adjacent sitting
    emb = _OneHotEmbedder(["clusterx"])  # embedding agrees, but nothing else -> 1 factor
    report = await discover_tags(store, _NamerLLM("hardware"), asyncio.Event(), embedder=emb)
    assert report.incorporated == []
    assert await store.get_tag("hardware") is None


# ---------------------------------------------------------------------------
# NAME validation: stem-dedup folds into an existing tag instead of duplicating
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_name_stem_dedup_folds_into_existing(store):
    a = await _seed_bucket_atom(store, "clustery alpha", "designed-day1")
    b = await _seed_bucket_atom(store, "clustery bravo", "designed-day2")
    await _cofire(store, [a, b])
    emb = _OneHotEmbedder(["clustery"])
    # model NAMEs "roadmaps" -> stems to the existing seeded "roadmap" -> fold, no duplicate tag.
    report = await discover_tags(store, _NamerLLM("roadmaps"), asyncio.Event(), embedder=emb)
    assert await store.get_tag("roadmaps") is None
    assert "roadmap" in {t.name for t in await store.tags_for_atom(a.id)}
    assert "roadmap" in report.incorporated or "roadmap" in report.merged


# ---------------------------------------------------------------------------
# Ladder floor: a group below the sitting floor stays a candidate, does not incorporate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_below_sitting_floor_stays_proposed(store):
    a = await _seed_bucket_atom(store, "clusterz alpha", "designed-day1")
    b = await _seed_bucket_atom(store, "clusterz bravo", "designed-day1")  # SAME sitting -> span 1
    await _cofire(store, [a, b])
    emb = _OneHotEmbedder(["clusterz"])
    report = await discover_tags(store, _NamerLLM("gadget"), asyncio.Event(), embedder=emb)
    assert report.incorporated == []
    assert await store.get_tag("gadget") is None
    # It IS a candidate (accruing), just not edge-eligible until it spans >= 2 sittings.
    assert [t for t in await store.list_tags(status="proposed")]


# ---------------------------------------------------------------------------
# F5: trace fanout cap — a rich mixed-topic trace row is a generic probe, not pair evidence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mega_trace_row_does_not_link(store):
    """F5: one rich retrieval event firing MANY atoms must not clique-link unrelated adjacent-
    sitting atoms (temporal+trace with zero semantic agreement). A trace row over the fanout cap
    contributes NO pair evidence — the hub-guard shape, carried from tags to trace rows."""
    await store.create_session("designed-day1", budget={}, model="m", context_tokens_available=1000)
    await store.create_session("designed-day2", budget={}, model="m", context_tokens_available=1000)
    a = await _seed_bucket_atom(store, "alpha topic detail", "designed-day1")
    b = await _seed_bucket_atom(store, "unrelated bravo item", "designed-day2")
    await store.record_activation_trace(stimulus="broad probe",
                                        fired_ids=[a.id, b.id] + list(range(9001, 9011)),
                                        injected_ids=[a.id, b.id], source="memory_search")

    report = await discover_tags(store, _NamerLLM("wrongtag"), asyncio.Event(),
                                 embedder=_OneHotEmbedder(["zzz"]))
    assert report.proposed == [] and report.incorporated == []
    assert await store.get_tag("wrongtag") is None


@pytest.mark.asyncio
async def test_tight_trace_row_with_adjacent_sittings_still_links(store):
    """Control for the fanout cap: a TIGHT co-fire row (a real reinstatement) plus sitting
    adjacency is still 2-factor evidence — the cap suppresses generic probes, not real signal."""
    await store.create_session("designed-day1", budget={}, model="m", context_tokens_available=1000)
    await store.create_session("designed-day2", budget={}, model="m", context_tokens_available=1000)
    a = await _seed_bucket_atom(store, "alpha topic detail", "designed-day1")
    b = await _seed_bucket_atom(store, "related bravo detail", "designed-day2")
    await _cofire(store, [a, b])

    report = await discover_tags(store, _NamerLLM("pairtag"), asyncio.Event(),
                                 embedder=_OneHotEmbedder(["zzz"]))
    assert "pairtag" in report.incorporated


# ---------------------------------------------------------------------------
# Pruning: a candidate that stops accruing decays below the floor and is retired
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stale_unmatched_candidate_pruned(store):
    proj = await store.get_tag("project")
    a = await _seed_bucket_atom(store, "orphan alpha unique", "designed-day1")
    b = await _seed_bucket_atom(store, "orphan bravo unique", "designed-day2")
    cand = await store.create_tag("cand-old", "a stale candidate", parent_id=proj.id,
                                  status="proposed", origin="discovered")
    await store.add_atom_tag(a.id, cand.id, "discovery")
    await store.add_atom_tag(b.id, cand.id, "discovery")
    old = int(time.time()) - 60 * 86400
    await store.bump_tag_evidence(cand.id, distinct_sittings=2, reuse_count=1, last_accrual_ts=old)

    # This cycle the members share NO factor (embedder blind to them, no trace) -> unmatched -> the
    # stale candidate decays below the floor and is pruned.
    report = await discover_tags(store, _NamerLLM("x"), asyncio.Event(),
                                 embedder=_OneHotEmbedder(["nomatch"]))
    assert "cand-old" in report.pruned
    assert (await store.get_tag("cand-old")).status == "retired"
    assert {t.name for t in await store.tags_for_atom(a.id)} == {"project"}  # detached -> bucket-only

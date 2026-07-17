"""#88 — the exactly-one-L1-bucket invariant + heal.

Root cause (audit 2026-07-17: one atom carried BOTH personal and project buckets, written ~1s
apart): mining re-classifies + re-tags an atom whose key CORROBORATED (store_fact returned the
pre-existing id), and the generic `add_atom_tag` deduped only on the exact (atom_id, tag_id) pair —
so a second, DIFFERENT bucket pick landed as a second row. The fix routes every L1-bucket write
through `add_bucket_tag`, which keeps the existing bucket and refuses a conflicting one, and a
deterministic `heal_bucket_conflicts` collapses legacy violations. No live model needed.
"""
import asyncio

import pytest

from localharness.memory.sqlite import MemoryStore
from localharness.memory.tag_classify import _BUCKET_MARKER, _CHILD_MARKER, file_atom_tags


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(agent_id="bucket-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


class _ClassifierLLM:
    """Prompt-aware fake: fixed bucket for the bucket pick, fixed child for the child pick."""

    def __init__(self, bucket: str = "project", child: str = "ops"):
        self.bucket, self.child = bucket, child

    async def complete(self, prompt: str) -> str:
        if _BUCKET_MARKER in prompt:
            return self.bucket
        if _CHILD_MARKER in prompt:
            return self.child
        return ""


async def _seed(store, key, value):
    return await store.store_fact(key=key, value=value, tags=["sem", "pending_consolidation"],
                                  confidence=0.65, source="transcript_mining", node_kind="fact")


async def _bucket_ids(store):
    return {b.name: b.id for b in await store.buckets()}


async def _bucket_tags(store, atom_id):
    return {t.name for t in await store.tags_for_atom(atom_id) if t.parent_id is None}


@pytest.mark.asyncio
async def test_add_bucket_tag_refuses_second_different_bucket(store):
    """The invariant: an atom already carrying a bucket that receives a DIFFERENT bucket keeps the
    existing one — never two buckets. The first write wins; the second is refused (returns False)."""
    f = await _seed(store, "sem/gpu-ops/aaaa1111", "vLLM listens on 8081")
    ids = await _bucket_ids(store)

    assert await store.add_bucket_tag(f.id, ids["personal"], "mint") is True
    assert await store.add_bucket_tag(f.id, ids["project"], "mint") is False  # conflict refused

    assert await _bucket_tags(store, f.id) == {"personal"}  # exactly one, the pre-existing one


@pytest.mark.asyncio
async def test_add_bucket_tag_idempotent_same_bucket(store):
    """Re-filing the SAME bucket is an idempotent no-op (one row), not a conflict."""
    f = await _seed(store, "sem/gpu-ops/bbbb2222", "KV cache spills to CPU")
    ids = await _bucket_ids(store)

    assert await store.add_bucket_tag(f.id, ids["project"], "mint") is True
    assert await store.add_bucket_tag(f.id, ids["project"], "mint") is False  # already there
    assert await _bucket_tags(store, f.id) == {"project"}


@pytest.mark.asyncio
async def test_file_atom_tags_twice_never_double_buckets(store):
    """The exact mining re-entry, at the seam mining + remember share (file_atom_tags): a re-mint of
    an already-filed atom re-classifies to a DIFFERENT bucket, but the second filing keeps the first
    bucket — no double-fire. This is the root-cause regression (mining routes its bucket write here)."""
    f = await _seed(store, "sem/kyoto/cccc3333", "one-week Kyoto trip in November")

    # occurrence 1: classified personal/preferences
    await file_atom_tags(store, _ClassifierLLM(bucket="personal", child="preferences"), asyncio.Event(),
                         atom_id=f.id, topic="kyoto", claim=f.value, provenance="mint")
    # occurrence 2 (the corroboration re-mint): the LLM now picks a DIFFERENT bucket
    await file_atom_tags(store, _ClassifierLLM(bucket="project", child="ops"), asyncio.Event(),
                         atom_id=f.id, topic="kyoto", claim=f.value, provenance="mint")

    assert await _bucket_tags(store, f.id) == {"personal"}  # first bucket wins; never two


@pytest.mark.asyncio
async def test_heal_bucket_conflicts_collapses_to_earliest(store):
    """A legacy violation (two bucket rows written before the fix, via the raw add_atom_tag path)
    is healed deterministically — KEEP the earliest (min ts) bucket, drop the rest. Returns the
    healed records for the caller to event."""
    f = await _seed(store, "sem/health/dddd4444", "training for a 10k in September")
    ids = await _bucket_ids(store)
    # Simulate the pre-fix double-write directly on the table (personal first at ts=1000, project at ts=1001).
    await store._db.execute("INSERT INTO atom_tags (atom_id, tag_id, provenance, ts) VALUES (?,?, 'mint', 1000)",
                            (f.id, ids["personal"]))
    await store._db.execute("INSERT INTO atom_tags (atom_id, tag_id, provenance, ts) VALUES (?,?, 'mint', 1001)",
                            (f.id, ids["project"]))
    await store._db.commit()
    assert await _bucket_tags(store, f.id) == {"personal", "project"}  # the violation exists

    healed = await store.heal_bucket_conflicts()
    assert healed == [(f.id, ids["personal"], [ids["project"]])]      # kept earliest, dropped later
    assert await _bucket_tags(store, f.id) == {"personal"}            # collapsed to one


@pytest.mark.asyncio
async def test_heal_noop_when_no_conflicts(store):
    """A store with at most one bucket per atom heals nothing (returns [])."""
    f = await _seed(store, "sem/ok/eeee5555", "a normally-filed atom")
    ids = await _bucket_ids(store)
    await store.add_bucket_tag(f.id, ids["project"], "mint")
    assert await store.heal_bucket_conflicts() == []
    assert await _bucket_tags(store, f.id) == {"project"}

"""RULING-D (Phase 36.2, TAGG-01 + TAGG-02): fold scope and `replaces=` supersede validity move
from the freely-guessed slug to the validated CHILD TAG axis. A wrong topic word can no longer
corrupt folding or corrections; a wrong slug WITH the correct tag folds/supersedes as intended.
The whole re-key is behind agent.memory.consolidation.tag_grouping_enabled (default True) — the
pre-committed KILL revert lever. These tests are the investing-class mis-file provables (owner
dogfood 2026-07-09: "research Anthropic pricing" filed under sem/investing/)."""
import asyncio

import pytest

from localharness.config.models import MemoryConsolidationConfig
from localharness.memory.mining import mine_transcript
from localharness.memory.sqlite import FactQuery, MemoryStore
from localharness.memory.tag_classify import (
    _BUCKET_MARKER,
    _CHILD_MARKER,
    classify_atom_tags,
    file_atom_tags,
)


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(agent_id="tagg-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


class _ClassifierLLM:
    """Prompt-aware fake (mirrors test_tag_graph): miner atoms for the miner prompt, a bucket for
    the bucket pick, a child for the child pick — keyed on the prompt markers so one instance
    drives the whole two-step mint flow deterministically."""

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
    return {"v": 1, "agent_id": "tagg-agent", "type": typ, "id": f"h{ts}",
            "session_id": sid, "ts": ts, "content": content}


async def _seed_atom(store, key, value, *child_tag_ids):
    """Seed one active sem/ atom carrying the given child tags — the 'true sibling'/'target'."""
    f = await store.store_fact(key=key, value=value, tags=["sem", "pending_consolidation"],
                               confidence=0.65, provenance="seed", node_kind="fact")
    for tid in child_tag_ids:
        await store.add_atom_tag(f.id, tid, "mint")
    return f


def _tag_stub(monkeypatch, bucket_tag, child_tag):
    """Force mining's per-atom classify to a fixed (bucket, child) — decouples the fold/supersede
    provables from the classifier's own model behavior (that axis is covered by test_tag_graph)."""
    async def _fake(*a, **k):
        return bucket_tag, child_tag
    monkeypatch.setattr("localharness.memory.mining.classify_atom_tags", _fake)


# --- Task 1: classify/write split + the tag_grouping switch (behavior-preserving) -----------

@pytest.mark.asyncio
async def test_classify_atom_tags_returns_tags_and_writes_nothing(store, monkeypatch):
    """classify_atom_tags is the CLASSIFY half — it returns the chosen (bucket, child) Tag objects
    and writes ZERO atom_tags rows (the write is file_atom_tags' job). Mining reads the returned
    child tag as the grouping axis BEFORE deciding the fold/replaces scope."""
    calls = []
    orig = store.add_atom_tag

    async def _spy(*a, **k):
        calls.append((a, k))
        return await orig(*a, **k)

    monkeypatch.setattr(store, "add_atom_tag", _spy)

    bucket, child = await classify_atom_tags(
        store, _ClassifierLLM(bucket="project", child="ops"), asyncio.Event(),
        topic="gpu ops", claim="vLLM server listens on port 8081")

    assert bucket is not None and bucket.name == "project"
    assert child is not None and child.name == "ops"
    assert calls == []  # classify NEVER writes


@pytest.mark.asyncio
async def test_file_atom_tags_still_writes(store):
    """Back-compat: file_atom_tags = classify + write. Same signature (atom_id, topic, claim,
    provenance), same return (bucket_name|None, child_name|None), same atom_tags rows — the F4
    backfill caller (consolidation.py) depends on all three staying identical."""
    f = await store.store_fact(key="sem/gpu-ops/abc12345", value="vLLM server listens on port 8081",
                               tags=["sem", "pending_consolidation"], confidence=0.65,
                               provenance="s1", node_kind="fact")
    bucket, child = await file_atom_tags(
        store, _ClassifierLLM(bucket="project", child="ops"), asyncio.Event(),
        atom_id=f.id, topic="gpu ops", claim="vLLM server listens on port 8081")
    assert (bucket, child) == ("project", "ops")
    assert {t.name for t in await store.tags_for_atom(f.id)} == {"project", "ops"}


@pytest.mark.asyncio
async def test_tag_grouping_enabled_default_true():
    """The RULING-D re-key is ON by default; the KILL revert lever is setting it False."""
    assert MemoryConsolidationConfig().tag_grouping_enabled is True


# --- Task 2 (TAGG-01): fold scope reads the CHILD TAG axis, not the slug --------------------

@pytest.mark.asyncio
async def test_fold_across_wrong_slug(store, monkeypatch):
    """The investing-class fold provable: an atom minted with a WRONG slug (investing) whose claim
    paraphrases a true sibling carrying child tag `ops` FOLDS onto that sibling — because
    atoms_for_tag(ops) surfaces it despite the slug mismatch. No orphan investing sibling row."""
    project = await store.get_tag("project")
    ops = await store.get_tag("ops")
    await _seed_atom(store, "sem/gpu-ops/aaaaaaaa", "vLLM server listens on port 8081", ops.id)
    _tag_stub(monkeypatch, project, ops)  # the mined atom classifies to the SAME child tag

    await store.append_history(_rec(20, "the vLLM server listens on port 8081 today", sid="d2"))
    r = await mine_transcript(store, _ClassifierLLM(
        atoms="investing | vLLM server listens on port 8081 | listens on port 8081"),
        asyncio.Event())

    assert r.folded == 1 and r.written == 0
    keys = {f.key for f in await store.query_facts(FactQuery(tags=["sem"], limit=10))}
    assert keys == {"sem/gpu-ops/aaaaaaaa"}  # folded onto the sibling, no wrong-slug orphan


@pytest.mark.asyncio
async def test_no_fold_without_shared_tag(store, monkeypatch):
    """A wrong slug ALONE can no longer merge unrelated facts: two atoms sharing a slug but with
    DIFFERENT child tags do NOT fold — the tag-identity candidate set excludes the non-shared-tag
    atom, so a fresh mint happens (the slug path WOULD have folded this paraphrase)."""
    project = await store.get_tag("project")
    ops = await store.get_tag("ops")
    conventions = await store.get_tag("conventions")
    await _seed_atom(store, "sem/vllm-port/aaaaaaaa", "vLLM server listens on port 8081", ops.id)
    _tag_stub(monkeypatch, project, conventions)  # DIFFERENT child tag than the sibling's ops

    await store.append_history(_rec(20, "the vLLM server on port 8081 runs fine", sid="d2"))
    r = await mine_transcript(store, _ClassifierLLM(
        atoms="vllm port | vLLM server on port 8081 | server on port 8081"),
        asyncio.Event())

    assert r.folded == 0
    assert len(await store.query_facts(FactQuery(tags=["sem"], limit=10))) == 2  # no merge


@pytest.mark.asyncio
async def test_fold_switch_off_is_slug_baseline(store):
    """KILL-lever proof (fold path): with tag_grouping=False the fold candidate set is the slug
    namespace exactly as pre-36.2 — a same-slug paraphrase folds byte-behavior-identically."""
    await _seed_atom(store, "sem/vllm-port/aaaaaaaa", "vLLM server listens on port 8081")

    await store.append_history(_rec(20, "the vLLM server on port 8081 runs fine", sid="d2"))
    r = await mine_transcript(store, _ClassifierLLM(
        atoms="vllm port | vLLM server on port 8081 | server on port 8081"),
        asyncio.Event(), tag_grouping=False)

    assert r.folded == 1 and r.written == 0
    assert len(await store.query_facts(FactQuery(tags=["sem"], limit=10))) == 1  # slug fold

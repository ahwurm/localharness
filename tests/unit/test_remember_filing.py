"""#87 — remember() files its atom at SAVE time.

Audit 2026-07-17: remember-sourced rows had zero atom_tags ever — the idle backfill that would
file them is starved by cancellation before it runs. The fix runs the SAME two-pick bucket/child
classifier (file_atom_tags — no forked classifier) right after the durable save. HARD RULE: a
classify failure/timeout NEVER blocks or fails the save — the fact is saved untagged, logged, and
the turn-end micro-pass (#90) files it later. No live model needed (fake classifier + clock).
"""
import asyncio

import pytest

from localharness.memory.sqlite import MemoryStore
from localharness.memory.tag_classify import _BUCKET_MARKER, _CHILD_MARKER
from localharness.tools.builtin.memory_tools import MemoryRememberTool


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(agent_id="remember-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


class _ClassifierLLM:
    def __init__(self, bucket="project", child="ops"):
        self.bucket, self.child = bucket, child

    async def complete(self, prompt: str) -> str:
        if _BUCKET_MARKER in prompt:
            return self.bucket
        if _CHILD_MARKER in prompt:
            return self.child
        return ""


class _RaisingLLM:
    async def complete(self, prompt: str) -> str:
        raise RuntimeError("model unreachable")


class _SlowLLM:
    async def complete(self, prompt: str) -> str:
        await asyncio.sleep(30.0)
        return "project"


async def _tag_names(store, key):
    fact = await store.get_fact(key)
    return {t.name for t in await store.tags_for_atom(fact.id)}


@pytest.mark.asyncio
async def test_remember_files_bucket_and_child_at_save_time(store):
    """With an LLM wired, remember() files the atom's bucket + child in the same save operation —
    the same two-pick seam mining uses, reused not forked."""
    tool = MemoryRememberTool(store, llm=_ClassifierLLM(bucket="project", child="ops"))
    res = await tool.run(name="deploy-vpn", content="deploying requires the corporate VPN")
    assert res.success
    assert await _tag_names(store, "deploy-vpn") == {"project", "ops"}   # filed at save time


@pytest.mark.asyncio
async def test_remember_saves_untagged_without_llm(store):
    """No LLM wired (the default / test path): the save still succeeds; the atom is untagged and
    the micro-pass (#90) files it later. Byte-identical to today's remember for the no-LLM path."""
    tool = MemoryRememberTool(store)                       # no llm
    res = await tool.run(name="editor-pref", content="user prefers neovim")
    assert res.success
    assert (await store.get_fact("editor-pref")).value == "user prefers neovim"
    assert await _tag_names(store, "editor-pref") == set()  # untagged, not filed inline


@pytest.mark.asyncio
async def test_classify_failure_never_blocks_the_save(store):
    """HARD RULE: a classifier that raises must NOT fail or block remember — the fact is durably
    saved (untagged) and the tool returns ok."""
    tool = MemoryRememberTool(store, llm=_RaisingLLM())
    res = await tool.run(name="hbm-makers", content="user follows HBM memory makers")
    assert res.success                                          # save not failed
    assert (await store.get_fact("hbm-makers")) is not None  # durably saved
    assert await _tag_names(store, "hbm-makers") == set()  # untagged (classify failed, non-fatal)


@pytest.mark.asyncio
async def test_classify_timeout_never_blocks_the_save(store, monkeypatch):
    """HARD RULE, timeout arm: a hanging classify is cut at the bounded save-time budget — the fact
    is saved untagged and the tool returns promptly (well under the model's 30s hang)."""
    import localharness.tools.builtin.memory_tools as mt
    monkeypatch.setattr(mt, "_REMEMBER_FILE_BUDGET_S", 0.1)
    tool = MemoryRememberTool(store, llm=_SlowLLM())
    import time as _t
    t0 = _t.monotonic()
    res = await tool.run(name="slow-fact", content="a fact whose classify hangs")
    assert res.success
    assert _t.monotonic() - t0 < 5.0                       # did not wait 30s behind the model
    assert (await store.get_fact("slow-fact")) is not None
    assert await _tag_names(store, "slow-fact") == set()   # untagged after the timeout

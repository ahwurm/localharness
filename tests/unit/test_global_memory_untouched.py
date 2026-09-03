"""MEMS-03 — issue #150's acceptance criterion: one machine's projects stop polluting each other.

A session started inside a project writes facts, folds them, and rewrites its own `MEMORY.md`. The
machine-global memory it is not reading must come out of that session exactly as it went in — not
"mostly", not "no new files", but byte-identical. Phase 41 proved where a session's state LANDS;
this proves what a full session's writes do to the memory it is NOT pointed at.

The drive is the real `_start_async` from inside a project, offline (41-05's harness, imported —
only the LLM probe, the tokenizer, the REPL loop and plugin discovery are stubbed). Every memory
write below goes through the store `start` itself opened.

**Two scope boundaries, stated up front rather than discovered by a later reader:**

1. **The comparison set is `<global>/agents/` — the memory tree — and nothing wider.** The drive
   legitimately rewrites the global `config.yaml` (the additive security-defaults migration, plus a
   timestamped `.bak`) and writes `<global>/tools/`; 41-05 measured exactly that and scoped its own
   claim the same way. "The whole global directory is unchanged" would be FALSE today, and asserting
   it would either fail now or be quietly weakened later. `<global>/agents/` is the tree that holds
   another project's memory, and it is the tree this session must not touch.
2. **Consolidation's LLM-gated steps do not run offline, and the claim is exactly as strong as what
   executed.** With `llm=None`, `_step_write_schemas`, `_step_classify_untagged`,
   `_step_discover_tags`, `_step_reconcile` and `_step_mine` early-return. The deterministic core —
   `_step_fold`, `_step_promote_recurring`, `_step_decay`, `_step_cap_trim`, `_step_proxies` — runs,
   and it WRITES; `test_the_offline_consolidation_pass_really_writes` measures that on a store of
   its own, so "a consolidation pass ran" is a fact here rather than a hopeful noun.

`/memory promote` (42-04) is the one verb that deliberately writes across the boundary, so these
drives never call it — 41-05's precedent of measuring the scope rather than over-claiming it.

**And one boundary that was measured rather than assumed: these drives are ZERO-TURN.** The stubbed
loop stands in for the interactive REPL, so `AgentLoop` never takes a turn and its ambient-read site
never executes — proven, not supposed: a hard `raise` planted at `agent/loop.py`'s
`_recall = self._recall_router …` line leaves all four tests below GREEN (42-05 mutation (e2)). So
nothing in this file grades where a session READS from; that is 42-03's
`test_recall_scope_wiring.py`, which drives the read live. This file is about WRITES, and the
sentence it can defend is exactly that one.
"""
from __future__ import annotations

from pathlib import Path

from localharness.memory.sqlite import MemoryStore

# 41-05's drive harness, imported rather than copied so this file cannot drift from the fixtures the
# rest of the workspace-landing proofs are graded against.
from tests.unit.test_workspace_state_landing import (
    AGENT,
    _changed,
    _drive,
    _file_snapshot,
    _global_only_start,
    _workspace_start,
)

GLOBAL_SEED = "GLOBAL-SEED-MARKER"
WORKSPACE_MARKER = "WORKSPACE-ONLY-MARKER"


# --------------------------------------------------------------------------------- helpers


async def _seed_global_memory(global_dir: Path) -> None:
    """Give the machine-global store real content BEFORE the drive, then close it.

    Without this the tree is empty and `_changed(...) == set()` is vacuously true: a proof that
    nothing changed in a directory with nothing in it. Seeded, the assertion has something to fail
    on — and the assertion on `MEMORY.md`'s TEXT has a text to compare.

    Deliberately a store of this test's own, opened and closed before the session exists: the point
    is that the session never touches these bytes, so the session must not be the thing that wrote
    them.
    """
    store = MemoryStore(
        agent_id=AGENT,
        division_id="default",
        org_id="default",
        base_dir=str(global_dir),
        global_base_dir=str(global_dir),
    )
    await store.open()
    await store.store_fact(key="global-lesson", value=GLOBAL_SEED, confidence=0.9)
    await store.store_fact(key="global-second", value="a second global memory", confidence=0.9)
    await store.flush_memory_md()
    await store.close()


def _install_a_full_session(monkeypatch) -> dict:
    """Run a REAL session body in place of the interactive loop, and report that it fired.

    Every write goes through `self._store` — the `MemoryStore` `start` opened and handed the REPL —
    not a store this test constructed. The returned dict is the did-it-bite guard: a monkeypatch
    that missed its target leaves an empty dict, and without checking it a byte-proof over a session
    that never ran reads as a green test.
    """
    seen: dict = {}

    async def _a_full_session(self):
        from localharness.config.models import MemoryConsolidationConfig
        from localharness.memory.consolidation import ConsolidationPass

        await self._store.store_fact(key="ws-lesson", value=WORKSPACE_MARKER, confidence=0.9)
        await self._store.store_fact(key="ws-second", value="another project note", confidence=0.9)
        # `llm=None` on purpose (the established idiom — test_memory_consolidation.py:67): the
        # deterministic core runs and writes, the five LLM-gated steps early-return. The pass is
        # called DIRECTLY because the scheduler's idle timer never fires in a stubbed session
        # (41-05's finding) — waiting for it would prove nothing but patience.
        seen["report"] = await ConsolidationPass(self._store, MemoryConsolidationConfig()).run()
        await self._store.flush_memory_md()
        seen["ran"] = True
        return None

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", _a_full_session)
    return seen


def _fmt(paths) -> str:
    return ", ".join(sorted(str(p) for p in paths)) or "(none)"


# --------------------------------------------------------------------------------- the proof


async def test_a_full_workspace_session_leaves_the_global_memory_byte_identical(
    tmp_path, monkeypatch
):
    """#150, as an assertion: writes + a real consolidation pass + a MEMORY.md flush, and the
    machine-global memory tree is byte-for-byte what it was.

    Assertion ORDER is the design, not a formatting choice (41-06's lesson, now in its fourth
    shape). The did-it-bite guard goes FIRST: under a monkeypatch that missed its target, every
    assertion below passes for the wrong reason, so a general "nothing changed" sitting on top would
    shadow the only thing that makes this file mean anything. `MEMORY.md`'s TEXT comes before the
    stat-based set comparison because #150 names that file by name and a size-and-mtime tuple is a
    proxy for its content; and the workspace-did-change assertion comes last because it is the
    weakest claim here — it exists so that "nothing changed anywhere" cannot masquerade as a pass.
    """
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    await _seed_global_memory(global_dir)

    g_agents = global_dir / "agents"
    g_memory_md = g_agents / AGENT / "MEMORY.md"
    before = _file_snapshot(g_agents)
    memory_md_before = g_memory_md.read_text(encoding="utf-8")

    seen = _install_a_full_session(monkeypatch)
    ws_before = _file_snapshot(ws)

    await _drive()

    assert seen.get("ran"), "the stubbed session body never ran — the monkeypatch did not bite"
    assert seen.get("report") is not None, (
        "the consolidation pass returned no report — it raised into the drive's soft-failure path, "
        "so this drive did not actually consolidate anything"
    )

    assert g_memory_md.read_text(encoding="utf-8") == memory_md_before, (
        "the machine-global MEMORY.md was rewritten by a session running inside a project — "
        "this is issue #150's exact symptom"
    )

    changed = _changed(before, _file_snapshot(g_agents))
    assert changed == set(), (
        "a default-scope workspace session wrote into the machine-global memory tree: "
        f"{_fmt(changed)}"
    )

    ws_changed = _changed(ws_before, _file_snapshot(ws))
    assert ws_changed, (
        "nothing changed under the workspace either — the session did no work, so the empty global "
        "diff above proves nothing about scoping"
    )


async def test_the_global_store_is_seeded_before_the_drive_so_an_empty_diff_means_something(
    tmp_path, monkeypatch
):
    """The premise guard, as its own test rather than a comment.

    `_changed(before, after) == set()` over an EMPTY directory is true and worthless. This asserts
    the tree the headline test compares is non-empty and holds this agent's real memory artifacts
    before any session exists — so the headline assertion has something it could have failed on.

    Deleting `_seed_global_memory` from the headline test leaves it GREEN (measured, 42-05 Task 2
    mutation (c)). That measurement is the whole argument for this test's existence.
    """
    _home, global_dir, _ws = _workspace_start(tmp_path, monkeypatch)
    await _seed_global_memory(global_dir)

    g_agent_dir = global_dir / "agents" / AGENT
    snapshot = _file_snapshot(global_dir / "agents")
    assert snapshot, "the global agents tree is empty — an unchanged-tree proof over it is vacuous"

    for name in ("memory.db", "MEMORY.md"):
        assert (g_agent_dir / name).exists(), f"the seed left no {name} in the global agents tree"
        assert (g_agent_dir / name) in snapshot, f"{name} is not in the compared file set"

    # And it is really the seeded content, not an empty file: the headline test's text comparison is
    # only meaningful if there is text.
    assert GLOBAL_SEED in (g_agent_dir / "MEMORY.md").read_text(encoding="utf-8"), (
        "the seeded fact did not reach the global MEMORY.md — the text comparison would compare "
        "two empty strings"
    )


async def test_the_offline_consolidation_pass_really_writes(tmp_path):
    """The second premise guard: the pass the headline test runs is not a no-op offline.

    "A consolidation pass ran" is worth nothing if `llm=None` turns the whole pass into an
    early-return. Measured here on a store of this test's own, with the SAME bare
    `MemoryConsolidationConfig()` the session body uses — a different config here would grade a
    different pass.

    The deterministic core writes: a staged read is FOLDED into the fact's durable access count.
    That is a real row update in `facts`, which is exactly the kind of write the headline test
    asserts never reaches the global database.
    """
    from localharness.config.models import MemoryConsolidationConfig
    from localharness.memory.consolidation import ConsolidationPass

    store = MemoryStore(
        agent_id="cons-probe", division_id="default", org_id="default", base_dir=str(tmp_path)
    )
    await store.open()
    try:
        await store.store_fact(key="k", value="v", confidence=0.9)
        await store.touch_staged(["k"])
        report = await ConsolidationPass(store, MemoryConsolidationConfig()).run()
        assert report.folded == 1, (
            f"the offline pass folded {report.folded} staged reads — with the deterministic core "
            "inert, the headline test's consolidation step would prove nothing"
        )
        fact = await store.get_fact("k")
        assert fact is not None and fact.access_count == 1, (
            "the fold did not reach the database, so the offline pass performs no writes at all"
        )
    finally:
        await store.close()


# --------------------------------------------------------------------------------- the control


async def test_the_same_recipe_without_a_workspace_does_write_the_global_agents_tree(
    tmp_path, monkeypatch
):
    """The control: an assertion that cannot fail proves nothing, so make it fail on purpose.

    Same seed, same session body, same comparison set — the ONE difference is that there is no
    `.localharness/` up-tree, so the session's own store IS the machine-global store. The global
    memory tree must therefore CHANGE.

    The changed set is asserted to contain this agent's `memory.db` specifically. "Something
    changed" would pass on a lock file, a `-wal` sidecar or a stray journal, none of which would
    show that a session's facts can reach the global database at all.
    """
    _home, global_dir, _proj = _global_only_start(tmp_path, monkeypatch)
    await _seed_global_memory(global_dir)

    g_agents = global_dir / "agents"
    before = _file_snapshot(g_agents)

    seen = _install_a_full_session(monkeypatch)

    await _drive()

    assert seen.get("ran"), "the stubbed session body never ran in the control either"

    changed = _changed(before, _file_snapshot(g_agents))
    assert changed, (
        "the control changed nothing under the global agents tree — the byte-proof above is "
        "structurally incapable of failing and grades nothing"
    )
    assert (g_agents / AGENT / "memory.db") in changed, (
        "the control changed files but not the agent's own memory.db, so the byte-proof is not "
        f"graded against a session's FACTS reaching the global store. Changed: {_fmt(changed)}"
    )
    assert WORKSPACE_MARKER in (g_agents / AGENT / "MEMORY.md").read_text(encoding="utf-8"), (
        "the control session's fact never reached the global MEMORY.md — the file the headline "
        "test asserts is untouched is one this recipe can demonstrably rewrite"
    )

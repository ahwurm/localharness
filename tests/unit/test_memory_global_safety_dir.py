"""MEMS-01: a workspace may ADD context, it may never silence the global safety voice.

ROADMAP critique amendment #4, owner-ruled and binding: org/division safety context follows the
deny-union physics. Per-agent STATE (memory.db, history.jsonl, MEMORY.md) may follow a workspace
layer, but `DIVISION.md` and `GUARDRAILS.md` are read from the GLOBAL layer, always. Before this
split, `MemoryStore` derived all of them from ONE `base_dir`, so moving that dir to a workspace
would have taken the org's safety instructions with it — and a workspace could then blank them by
simply not having the file. These tests prove the two directories are genuinely independent in BOTH
directions, and that omitting the new parameter is a no-op (bench/runner.py relies on that).

SAFE-04 (v0.16) moved the GUARDRAILS.md reader out of memory and into core prompt assembly
(`AgentLoop(guardrails_path=...)`), so turning memory off can no longer turn the safety rules off.
The invariant pinned here did not move with it: the store's context now has NO guardrails field
(exactly one reader), and the guardrails claims are made on the SYSTEM PROMPT a real loop builds
over the split store with core's reader pointed at the global file.

Every assertion here reads as "which of the two dirs did this come from": `ws` is the workspace
state dir, `gl` is the global one. Tests 2 and 3 assert on what was READ — the context returned by
`load_context()` and the prompt the loop built — rather than on a stored path, because a path
assertion cannot tell you the reader ever used it, and the workspace payload is asserted ABSENT —
"the right one is present" passes when both are (40-02 / 40-05's lesson).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.memory.sqlite import MemoryStore
from tests.unit.test_agent_loop import _make_memory_agent_loop, _occupied_slot, _record_system_prompts

GLOBAL_GUARDRAILS = "# Org guardrails\nGLOBAL-GUARDRAILS-SENTINEL\n"
WORKSPACE_GUARDRAILS = "# Org guardrails\nWORKSPACE-GUARDRAILS-SENTINEL\n"
GLOBAL_DIVISION = "# Division charter\nGLOBAL-DIVISION-SENTINEL\n"
WORKSPACE_DIVISION = "# Division charter\nWORKSPACE-DIVISION-SENTINEL\n"

AGENT_ID = "scoped-agent"
DIVISION_ID = "eng"
ORG_ID = "default"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _layers(tmp_path: Path) -> tuple[Path, Path]:
    """The two sibling dirs every test names: (workspace state, global safety)."""
    ws = tmp_path / "ws"
    gl = tmp_path / "gl"
    ws.mkdir()
    gl.mkdir()
    return ws, gl


def _split_store(ws: Path, gl: Path) -> MemoryStore:
    return MemoryStore(
        agent_id=AGENT_ID,
        division_id=DIVISION_ID,
        org_id=ORG_ID,
        base_dir=str(ws),
        global_base_dir=str(gl),
    )


async def _prompt(state_dir: Path, global_dir: Path, guardrails_path: Path | None) -> str:
    """The system prompt a real AgentLoop builds with the memory plugin's store split over
    (`state_dir`, `global_dir`) in its slot and core's guardrails reader pointed at
    `guardrails_path` — what the model is handed, not what a reader returned."""
    async with _occupied_slot(state_dir, agent_id=AGENT_ID, division_id=DIVISION_ID,
                              state_dir=state_dir, global_dir=global_dir) as (slot, _plugin):
        loop, llm = _make_memory_agent_loop(memory_slot=slot, guardrails_path=guardrails_path)
        prompts = _record_system_prompts(llm)
        await loop.run_turn("hello")
    return prompts[0]


@pytest.mark.asyncio
async def test_agent_state_is_written_under_the_workspace_dir_only(tmp_path):
    """State follows the workspace: memory.db / history.jsonl / MEMORY.md land under `ws`, and
    the global dir gets no agent tree at all."""
    ws, gl = _layers(tmp_path)
    store = _split_store(ws, gl)
    await store.open()
    try:
        await store.store_fact(key="port", value="vLLM on 8081", confidence=0.9)
        # The real writer of history.jsonl, not a hand-built record — a synthetic dict would
        # only prove HistoryWriter can write somewhere, not that the session path lands in `ws`.
        await store.create_session(
            session_id="s1", budget={}, model="qwen", context_tokens_available=1024,
        )
        await store.flush_memory_md()
    finally:
        await store.close()

    assert (ws / "agents" / AGENT_ID / "memory.db").exists()
    assert (ws / "agents" / AGENT_ID / "history.jsonl").exists()
    assert (ws / "agents" / AGENT_ID / "MEMORY.md").exists()
    # The global layer is for safety context, not for this agent's state.
    assert not (gl / "agents").exists(), sorted(p.name for p in gl.rglob("*"))


@pytest.mark.asyncio
async def test_safety_context_is_read_from_the_global_dir(tmp_path):
    """With DIVISION.md and GUARDRAILS.md present ONLY under `gl`, the prompt carries both —
    DIVISION.md through the store's context, GUARDRAILS.md through core's reader — and the
    workspace state dir holds neither file."""
    ws, gl = _layers(tmp_path)
    _write(gl / "orgs" / ORG_ID / "GUARDRAILS.md", GLOBAL_GUARDRAILS)
    _write(gl / "divisions" / DIVISION_ID / "DIVISION.md", GLOBAL_DIVISION)

    store = _split_store(ws, gl)
    await store.open()
    try:
        ctx = await store.load_context()
    finally:
        await store.close()
    prompt = await _prompt(ws, gl, gl / "orgs" / ORG_ID / "GUARDRAILS.md")

    assert not hasattr(ctx, "guardrails_md"), "memory reads GUARDRAILS.md again — two readers"
    assert "GLOBAL-GUARDRAILS-SENTINEL" in prompt
    assert "GLOBAL-DIVISION-SENTINEL" in ctx.division_md


@pytest.mark.asyncio
async def test_workspace_cannot_silence_or_replace_the_global_safety_voice(tmp_path):
    """The ruling itself. Both layers hold both files; the GLOBAL text is what reaches the agent
    and the workspace text never does."""
    ws, gl = _layers(tmp_path)
    _write(gl / "orgs" / ORG_ID / "GUARDRAILS.md", GLOBAL_GUARDRAILS)
    _write(gl / "divisions" / DIVISION_ID / "DIVISION.md", GLOBAL_DIVISION)
    _write(ws / "orgs" / ORG_ID / "GUARDRAILS.md", WORKSPACE_GUARDRAILS)
    _write(ws / "divisions" / DIVISION_ID / "DIVISION.md", WORKSPACE_DIVISION)

    store = _split_store(ws, gl)
    await store.open()
    try:
        ctx = await store.load_context()
    finally:
        await store.close()
    prompt = await _prompt(ws, gl, gl / "orgs" / ORG_ID / "GUARDRAILS.md")

    assert not hasattr(ctx, "guardrails_md"), "memory reads GUARDRAILS.md again — two readers"
    assert "GLOBAL-GUARDRAILS-SENTINEL" in prompt
    assert "WORKSPACE-GUARDRAILS-SENTINEL" not in prompt
    assert "GLOBAL-DIVISION-SENTINEL" in ctx.division_md
    assert "WORKSPACE-DIVISION-SENTINEL" not in ctx.division_md
    assert "GLOBAL-DIVISION-SENTINEL" in prompt  # and the slot's reader agrees
    assert "WORKSPACE-DIVISION-SENTINEL" not in prompt


@pytest.mark.asyncio
async def test_omitting_global_base_dir_is_a_no_op(tmp_path):
    """The default-preserving half: constructed WITHOUT the new parameter, a store rooted at one
    dir reads that dir's own DIVISION.md exactly as it always has. bench/runner.py and every
    pre-existing caller depend on this staying byte-identical.

    GUARDRAILS.md sits in the same root, and since SAFE-04 the store never reads it: a loop built
    the way bench builds one — this store, no `guardrails_path` — gets no guardrails section. (Bench's
    temp root has never held a GUARDRAILS.md, so no bench prompt changed.)"""
    root = tmp_path / "only"
    root.mkdir()
    _write(root / "orgs" / ORG_ID / "GUARDRAILS.md", GLOBAL_GUARDRAILS)
    _write(root / "divisions" / DIVISION_ID / "DIVISION.md", GLOBAL_DIVISION)

    store = MemoryStore(
        agent_id=AGENT_ID,
        division_id=DIVISION_ID,
        org_id=ORG_ID,
        base_dir=str(root),
    )
    await store.open()
    try:
        ctx = await store.load_context()
    finally:
        await store.close()
    prompt = await _prompt(root, root, None)

    assert not hasattr(ctx, "guardrails_md"), "memory reads GUARDRAILS.md again — two readers"
    assert "GLOBAL-GUARDRAILS-SENTINEL" not in prompt and "## Guardrails" not in prompt
    assert "GLOBAL-DIVISION-SENTINEL" in ctx.division_md
    assert (root / "agents" / AGENT_ID / "memory.db").exists()

"""42-03: the RecallRouter reaches a running session.

Two layers of proof, deliberately different in kind:

* Task 1 grades the LOOP — a real offline ``run_turn`` with mocked memory objects, asserting
  which object the ambient-context read went to and which objects still take the writes.
* Task 2 grades the SESSION — a real ``_start_async`` drive from a workspace, asserting on the
  recorded constructor kwargs (the wiring claim IS the kwarg) and on live reads performed
  mid-session with the stores open.

The discriminating pair throughout is "the router was read AND the store was not". Asserting
only the first passes an implementation that reads both, which is precisely the bug
``recall_scope`` exists to prevent.
"""
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from localharness.agent.loop import AgentLoop
from localharness.agent.context import ContextManager
from localharness.agent.permissions import PermissionEvaluator
from localharness.core.bus import EventBus

# Phase 41's drive harness, imported rather than copied — those files stay byte-untouched, and a
# drift between "how phase 41 drives a workspace session" and "how phase 42 does" would make the
# two phases' claims incomparable.
from tests.unit.test_workspace_state_landing import (
    AGENT,
    _drive,
    _global_only_start,
    _install_recorders,
    _one,
    _workspace_start,
)


# ---------------------------------------------------------------------------
# Task 1 — the loop reads ambient context through the router
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Ctx:
    """Shaped like MemoryContext for the fields loop.py touches."""
    agent_memory_md: str = "INJECTED-MEMORY"
    division_md: str = ""
    guardrails_md: str = ""
    fact_count: int = 1
    token_estimate: int = 10
    injected_fact_ids: tuple = (7,)


_UNSET = object()


def _mock_store(**load_context_kwargs):
    """A memory-store double shaped like the REAL MemoryStore: async ``load_context`` and async
    ``record_injection_trace`` on an otherwise SYNCHRONOUS object.

    Deliberately not a bare AsyncMock — ``set_current_session`` is a plain ``def`` on MemoryStore
    and loop.py calls it without awaiting, so a bare AsyncMock would both emit a 'never awaited'
    warning and be structurally incapable of catching production making that call async
    (test_agent_loop.py::_mock_memory_loader's reasoning, which applies unchanged here).
    """
    store = MagicMock()
    store.load_context = AsyncMock(**load_context_kwargs)
    store.record_injection_trace = AsyncMock()
    return store


def _mock_router(**load_context_kwargs):
    """The read gate. Same shape as the store for the ONE method the loop calls on it — and
    deliberately WITHOUT ``set_current_session``/``record_injection_trace`` as AsyncMocks, so a
    loop that tried to route a write through the router would be visible here."""
    router = MagicMock()
    router.load_context = AsyncMock(**load_context_kwargs)
    return router


def _make_loop(memory_loader=None, recall_router=_UNSET):
    """An AgentLoop with mocked dependencies, modelled on test_agent_loop.py's
    ``_make_memory_agent_loop`` (copied rather than imported so that pre-existing file stays
    byte-untouched). ``recall_router`` unset means the kwarg is NOT passed at all — the control
    that grades today's default path."""
    from localharness.config.models import AgentConfig
    from tests.conftest import FakeLLMResponse, MockLLMClient

    cfg = AgentConfig(name="test-agent", role="You are a test assistant.")
    llm = MockLLMClient([FakeLLMResponse(content="Done.")])
    kwargs = {}
    if recall_router is not _UNSET:
        kwargs["recall_router"] = recall_router
    return AgentLoop(
        config=cfg,
        llm=llm,
        bus=EventBus(),
        context_manager=ContextManager(),
        tool_registry=None,
        permission_evaluator=PermissionEvaluator(),
        memory_loader=memory_loader,
        **kwargs,
    ), llm


async def _capture_turn(loop, llm, task="hello"):
    """Run one real turn, returning the system messages the LLM actually saw."""
    captured: list = []
    original = llm.stream_complete

    async def capturing_stream(messages=None, tools=None, on_token=None):
        captured.extend(messages or [])
        return await original(messages=messages, tools=tools, on_token=on_token)

    llm.stream_complete = capturing_stream
    await loop.run_turn(task)
    return [m for m in captured if m.get("role") == "system"]


@pytest.mark.asyncio
async def test_the_turn_reads_ambient_context_through_the_router():
    """The discriminating pair: the router WAS read and the store was NOT. Either assertion
    alone is satisfied by an implementation that reads both stores every turn."""
    store = _mock_store(return_value=_Ctx(agent_memory_md="STORE-CONTEXT"))
    router = _mock_router(return_value=_Ctx(agent_memory_md="ROUTER-CONTEXT"))

    loop, llm = _make_loop(memory_loader=store, recall_router=router)
    sys_msgs = await _capture_turn(loop, llm)

    router.load_context.assert_awaited_once()
    store.load_context.assert_not_awaited()
    # And the router's answer is what actually reached the model, not just what was fetched.
    assert "ROUTER-CONTEXT" in sys_msgs[0]["content"]
    assert "STORE-CONTEXT" not in sys_msgs[0]["content"]


@pytest.mark.asyncio
async def test_the_router_carries_the_loops_memory_config():
    """The call site moved receivers, not arguments — index_mode and the shelf size still come
    from agent.memory and still reach whoever answers."""
    router = _mock_router(return_value=_Ctx())
    loop, llm = _make_loop(memory_loader=_mock_store(return_value=_Ctx()), recall_router=router)
    await _capture_turn(loop, llm)

    kwargs = router.load_context.await_args.kwargs
    assert kwargs["index_mode"] is True
    assert kwargs["max_session_history"] == 8


@pytest.mark.asyncio
async def test_writes_and_traces_still_go_to_the_session_store():
    """recall_scope must never redirect a write. The same turn that read through the router
    stamps its provenance and its activation trace on the STORE."""
    store = _mock_store(return_value=_Ctx())
    router = _mock_router(return_value=_Ctx(injected_fact_ids=(7,)))

    loop, llm = _make_loop(memory_loader=store, recall_router=router)
    await _capture_turn(loop, llm)

    store.set_current_session.assert_called_once()
    store.record_injection_trace.assert_awaited_once()
    # The trace carries the ids the router handed back (primary-owned by 42-02's contract).
    assert list(store.record_injection_trace.await_args.kwargs["injected_ids"]) == [7]
    # The router took NO write: the only name touched on it is the one read verb.
    touched = {name.split(".")[0] for name, _a, _k in router.mock_calls}
    assert touched == {"load_context"}


@pytest.mark.asyncio
async def test_without_a_router_the_store_answers_exactly_as_before():
    """Control: the kwarg is not passed at all. Bench, subagents and every pre-existing caller
    take this path, and it must be the path they took before this phase."""
    store = _mock_store(return_value=_Ctx(agent_memory_md="STORE-CONTEXT"))
    loop, llm = _make_loop(memory_loader=store)
    sys_msgs = await _capture_turn(loop, llm)

    store.load_context.assert_awaited_once()
    assert "STORE-CONTEXT" in sys_msgs[0]["content"]


@pytest.mark.asyncio
async def test_a_router_without_a_store_stays_memory_free():
    """The outer `if self._memory is not None` gate is NOT widened: a session with no store
    injects nothing, even if a router were somehow supplied."""
    router = _mock_router(return_value=_Ctx(agent_memory_md="ROUTER-CONTEXT"))
    loop, llm = _make_loop(memory_loader=None, recall_router=router)
    sys_msgs = await _capture_turn(loop, llm)

    router.load_context.assert_not_awaited()
    assert "ROUTER-CONTEXT" not in sys_msgs[0]["content"]
    assert "## Agent Memory" not in sys_msgs[0]["content"]


# ---------------------------------------------------------------------------
# Task 2 — one router per session, from a real `_start_async` drive
# ---------------------------------------------------------------------------

WS_MARKER = "WORKSPACE-RECALL-MARKER"
GLOBAL_DECOY = "GLOBAL-DECOY-MARKER"


def _record_wiring(monkeypatch) -> dict:
    """Capture the store OBJECTS, their close() calls, and the store-or-router each memory tool
    was constructed with.

    Wrappers, never doubles: the real objects still run, so every live read below hits a real
    database. Stacked ON TOP of `_install_recorders` (which records ctor KWARGS) — the kwargs
    answer "where was it pointed", these answer "which object went where", and the tool identity
    question can only be answered by the second.
    """
    import localharness.memory.sqlite as _sqlite
    import localharness.tools.builtin.memory_tools as _tools

    seen: dict = {"instances": [], "closed": [], "search": [], "get": [], "remember": []}

    real_init = _sqlite.MemoryStore.__init__

    def _rec_init(self, *args, **kwargs):
        seen["instances"].append(self)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr("localharness.memory.sqlite.MemoryStore.__init__", _rec_init)

    real_close = _sqlite.MemoryStore.close

    async def _rec_close(self, *args, **kwargs):
        seen["closed"].append(self)
        return await real_close(self, *args, **kwargs)

    monkeypatch.setattr("localharness.memory.sqlite.MemoryStore.close", _rec_close)

    for key, cls in (("search", "MemorySearchTool"), ("get", "MemoryGetTool"),
                     ("remember", "MemoryRememberTool")):
        real_tool_init = getattr(_tools, cls).__init__

        def _rec_tool_init(self, memory_store, *args, _key=key, _real=real_tool_init, **kwargs):
            seen[_key].append(memory_store)
            return _real(self, memory_store, *args, **kwargs)

        monkeypatch.setattr(f"localharness.tools.builtin.memory_tools.{cls}.__init__",
                            _rec_tool_init)

    return seen


def _write_scoped_agent(agents_dir: Path, scope: str) -> None:
    """`_write_agent` writes only {name, role, model}; a scope test needs the memory block, so it
    writes its own yaml rather than widening a helper five other files depend on."""
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / f"{AGENT}.yaml").write_text(yaml.dump({
        "name": AGENT,
        "role": "Test role",
        "model": "inherit",
        "memory": {"recall_scope": scope},
    }))


async def _seed_global_decoy(global_dir: Path) -> None:
    """Plant a fact in the MACHINE-GLOBAL store before the drive.

    Confidence 0.9 is above `AMBIENT_INJECTION_FLOOR` (0.7) on purpose: a fact under the floor
    never renders, so the default-scope absence assertion would pass for the wrong reason — it
    would be measuring the floor, not the scope.
    """
    from localharness.memory.sqlite import MemoryStore

    store = MemoryStore(
        agent_id=AGENT,
        division_id="default",
        org_id="default",
        base_dir=str(global_dir),
        global_base_dir=str(global_dir),
    )
    await store.open()
    await store.store_fact("global-decoy", f"{GLOBAL_DECOY} another project's recollection",
                           confidence=0.9)
    await store.close()


def _live_read(monkeypatch, rec: dict, seen: dict) -> None:
    """Replace the interactive loop with one that writes a workspace fact and then reads ambient
    context back THROUGH THE SESSION'S OWN ROUTER, mid-session, with both databases open.

    The router is reached through the RECORDED `AgentLoop` kwarg, never through a private
    attribute: the kwarg IS the wiring claim, so a read through anything else would grade an
    object the session might not actually be using.
    """
    async def _read_through_the_router(self):
        await self._store.store_fact("ws-marker", f"{WS_MARKER} this project's recollection",
                                     confidence=0.9)
        router = _one(rec, "loop")["recall_router"]
        seen["router"] = router
        seen["ctx"] = await router.load_context()
        return None

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", _read_through_the_router)


async def test_a_workspace_session_constructs_two_stores_and_opens_one(tmp_path, monkeypatch):
    """MEMS-03's precondition, on the filesystem. The twin is CONSTRUCTED (so a `both` session has
    something to open) and, at the default scope, never OPENED — `MemoryStore.__init__` only
    derives paths, `open()` is what creates and migrates a database."""
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    rec = _install_recorders(monkeypatch)

    await _drive()

    assert len(rec["store"]) == 2, (
        f"a workspace session builds the primary + the global twin; got {len(rec['store'])}"
    )
    primary, twin = rec["store"]
    assert primary["base_dir"] == str(ws)
    assert twin["base_dir"] == str(global_dir), (
        f"the twin points at {twin['base_dir']}, not the machine-global layer"
    )
    assert twin["global_base_dir"] == str(global_dir)

    # The claim that matters: nothing was CREATED there.
    assert not (global_dir / "agents" / AGENT / "memory.db").exists(), (
        "a default-scope workspace session created the global store's database"
    )
    assert not (global_dir / "agents").exists(), (
        "a default-scope workspace session created the global agents tree"
    )


async def test_the_global_twin_is_constructed_without_a_bus(tmp_path, monkeypatch):
    """A bus subscription is a WRITE path (auto-diary, the predictive gates). The twin is a READ
    handle, so it must not be reachable by any of them — `recall_scope` changes what a session
    reads, never where it writes."""
    _home, _global_dir, _ws = _workspace_start(tmp_path, monkeypatch)
    rec = _install_recorders(monkeypatch)

    await _drive()

    primary, twin = rec["store"]
    assert primary.get("bus") is not None, "the primary lost its bus — the control is broken"
    assert twin.get("bus") is None, "the global twin was given a bus and can take writes"


async def test_the_loop_and_both_read_tools_receive_the_same_router(tmp_path, monkeypatch):
    """Criterion 4 made structural: there is no 'the tool bypassed the knob' path to test for,
    because injection and on-demand recall read the SAME object."""
    _home, _global_dir, _ws = _workspace_start(tmp_path, monkeypatch)
    rec = _install_recorders(monkeypatch)
    seen = _record_wiring(monkeypatch)

    await _drive()

    router = _one(rec, "loop")["recall_router"]
    assert router is not None, "the loop got no router"
    assert seen["search"], "MemorySearchTool was never constructed — the patch did not bite"
    assert seen["get"], "MemoryGetTool was never constructed — the patch did not bite"
    assert seen["search"][0] is router, "memory_search reads a different object than injection"
    assert seen["get"][0] is router, "memory_get reads a different object than injection"
    # Both directions: it is a router, not the store wearing the name.
    primary = seen["instances"][0]
    assert seen["search"][0] is not primary
    assert seen["get"][0] is not primary


async def test_the_remember_tool_keeps_the_raw_store(tmp_path, monkeypatch):
    """The write verb never sees the router. `RecallRouter` has no `store_fact`, so a swap here
    would be an AttributeError at first use — loudly, but only for whoever tried to remember
    something. This asserts it before a user finds it."""
    _home, _global_dir, _ws = _workspace_start(tmp_path, monkeypatch)
    rec = _install_recorders(monkeypatch)
    seen = _record_wiring(monkeypatch)

    await _drive()

    router = _one(rec, "loop")["recall_router"]
    assert seen["remember"], "MemoryRememberTool was never constructed — the patch did not bite"
    assert seen["remember"][0] is seen["instances"][0], "remember() lost the session's own store"
    assert seen["remember"][0] is not router, "remember() writes through the recall router"


async def test_a_default_scope_session_reads_only_the_workspace(tmp_path, monkeypatch):
    """MEMS-02 criterion 1, from a LIVE session: another project's recollections do not appear in
    this project's ambient context, even though the global store exists and holds a rendering
    fact."""
    _home, global_dir, _ws = _workspace_start(tmp_path, monkeypatch)
    await _seed_global_decoy(global_dir)
    rec = _install_recorders(monkeypatch)
    seen: dict = {}
    _live_read(monkeypatch, rec, seen)

    await _drive()

    assert seen.get("ctx") is not None, "the live read never ran — the stubbed loop did not fire"
    md = seen["ctx"].agent_memory_md
    assert WS_MARKER in md, f"the session did not read its own store: {md!r}"
    assert GLOBAL_DECOY not in md, (
        "a default-scope session injected another project's memory"
    )
    assert seen["router"].scope == "workspace"


async def test_a_both_scope_session_reads_both_stores_with_origin_tokens(tmp_path, monkeypatch):
    """The knob reaches a running session: `recall_scope: both` in the workspace agent yaml and
    the live ambient read spans both databases, every line naming which store it came from."""
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    _write_scoped_agent(ws / "agents", "both")
    await _seed_global_decoy(global_dir)
    rec = _install_recorders(monkeypatch)
    seen: dict = {}
    _live_read(monkeypatch, rec, seen)

    await _drive()

    assert seen.get("ctx") is not None, "the live read never ran — the stubbed loop did not fire"
    assert seen["router"].scope == "both", (
        f"the yaml said both; the session's router says {seen['router'].scope!r}"
    )
    md = seen["ctx"].agent_memory_md
    assert WS_MARKER in md and GLOBAL_DECOY in md, f"the merge is missing a side: {md!r}"
    assert md.index(WS_MARKER) < md.index(GLOBAL_DECOY), "the merge is not scoped-first"
    assert "[global#" in md, "the global block carries no origin token"
    assert "[workspace#" in md, "the workspace block carries no origin token"


async def test_a_session_without_a_workspace_builds_one_store_and_collapses_the_scope(
    tmp_path, monkeypatch
):
    """LAYR-03's control. With no workspace layer `state_dir == cfg_path`, so a twin would be a
    SECOND aiosqlite connection to the SAME file. The knob is still READ (configured_scope keeps
    what the yaml asked for) and still collapses — proving the collapse, not a missing knob."""
    _home, global_dir, _proj = _global_only_start(tmp_path, monkeypatch)
    _write_scoped_agent(global_dir / "agents", "both")
    rec = _install_recorders(monkeypatch)

    await _drive()

    assert len(rec["store"]) == 1, (
        f"a workspace-less session must build exactly one store; got {len(rec['store'])}"
    )
    router = _one(rec, "loop")["recall_router"]
    assert router is not None, "the workspace-less session got no router"
    assert router.scope == "workspace", "the scope did not collapse without a second store"
    assert router.configured_scope == "both", (
        "the yaml's knob never reached the router — this test would then be grading a missing "
        "knob rather than the collapse"
    )


async def test_the_opened_global_handle_is_closed_at_shutdown(tmp_path, monkeypatch):
    """Pitfall 6: aiosqlite's worker thread is NON-DAEMON, so a leaked handle hangs interpreter
    shutdown. The twin is opened by this drive (the live `both` read forces it) and must be
    closed by the resource-owning window's finally."""
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    _write_scoped_agent(ws / "agents", "both")
    await _seed_global_decoy(global_dir)
    rec = _install_recorders(monkeypatch)
    seen = _record_wiring(monkeypatch)
    _live_read(monkeypatch, rec, seen)

    await _drive()

    assert len(seen["instances"]) == 2, "expected the primary and the twin"
    primary, twin = seen["instances"]
    # Premise: the twin really was opened, or "it was closed" is vacuous.
    assert GLOBAL_DECOY in seen["ctx"].agent_memory_md, "the twin was never read"
    assert twin in seen["closed"], "the router's global handle leaked — never closed"
    assert primary in seen["closed"], "the primary leaked — the control is broken"

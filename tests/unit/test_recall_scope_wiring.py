"""42-03: the RecallRouter reaches a running session.

Two layers of proof, deliberately different in kind:

* Task 1 grades the LOOP — a real offline ``run_turn`` through the memory slot's real
  MemoryPlugin, its read gate mocked and its store's verbs spied, asserting which object the
  ambient-context read went to and which objects still take the writes.
* Task 2 grades the SESSION — a real ``_start_async`` drive from a workspace, asserting on the
  recorded constructor kwargs (the wiring claim IS the kwarg) and on live reads performed
  mid-session with the stores open.

The discriminating pair throughout is "the router was read AND the store was not". Asserting
only the first passes an implementation that reads both, which is precisely the bug
``recall_scope`` exists to prevent.
"""
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from localharness.agent.loop import AgentLoop
from localharness.agent.context import ContextManager
from localharness.agent.permissions import PermissionEvaluator
from localharness.core.bus import EventBus
from tests.unit.test_agent_loop import _occupied_slot

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


def _record_memory_plugins(monkeypatch) -> list:
    """Wrap (never replace) MemoryPlugin.start and record each plugin that got through it, with
    the router and store it holds at that moment (stop() drops both, so they are read here). The
    running plugin's router is what `/memory promote` borrows (its slash row is bound to this very
    instance) and what the slot reads ambient context through — the session's one router."""
    from types import SimpleNamespace

    from localharness.memory.plugin import MemoryPlugin
    started: list = []
    real_start = MemoryPlugin.start

    async def _rec_start(self, ctx):
        await real_start(self, ctx)
        started.append(SimpleNamespace(plugin=self, router=self._router, store=self._store))

    monkeypatch.setattr("localharness.memory.plugin.MemoryPlugin.start", _rec_start)
    return started


def _the_memory_plugin(started: list):
    """The single started memory plugin, with the did-it-bite guard."""
    assert len(started) == 1, f"expected one started memory plugin; got {len(started)} — did the patch bite?"
    return started[0]


# ---------------------------------------------------------------------------
# Task 1 — the loop reads ambient context through the router
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Ctx:
    """Shaped like MemoryContext for the fields the memory plugin's context() touches."""
    agent_memory_md: str = "INJECTED-MEMORY"
    division_md: str = ""
    fact_count: int = 1
    token_estimate: int = 10
    injected_fact_ids: tuple = (7,)


def _mock_router(**load_context_kwargs):
    """The read gate. Same shape as the store for the ONE method the plugin calls on it — and
    deliberately WITHOUT ``set_current_session``/``record_injection_trace`` as AsyncMocks, so a
    plugin that tried to route a write through the router would be visible here."""
    router = MagicMock()
    router.load_context = AsyncMock(**load_context_kwargs)
    return router


_UNSET = object()


@asynccontextmanager
async def _session(tmp_path, monkeypatch, *, router=_UNSET, store_ctx=None):
    """A REAL MemoryPlugin started over a tmp store in the loop's memory slot (the loop's only
    memory path since 48-06), with the plugin's read gate swapped for `router` (None = no router,
    the plugin's own fallback when RecallRouter construction fails) and the session store's own
    `load_context` / write verbs spied. The real router is put back before stop() closes it.
    Yields (loop, llm, store_spies)."""
    from localharness.config.models import AgentConfig
    from tests.conftest import FakeLLMResponse, MockLLMClient

    async with _occupied_slot(tmp_path) as (slot, plugin):
        store, real_router = plugin._store, plugin._router
        spies = SimpleNamespace(
            load_context=AsyncMock(return_value=store_ctx or _Ctx(agent_memory_md="STORE-CONTEXT")),
            set_current_session=MagicMock(wraps=store.set_current_session),
            record_injection_trace=AsyncMock())
        for name in ("load_context", "set_current_session", "record_injection_trace"):
            monkeypatch.setattr(store, name, getattr(spies, name))
        if router is not _UNSET:
            plugin._router = router
        cfg = AgentConfig(name="test-agent", role="You are a test assistant.")
        llm = MockLLMClient([FakeLLMResponse(content="Done.")])
        loop = AgentLoop(config=cfg, llm=llm, bus=EventBus(), context_manager=ContextManager(),
                         tool_registry=None, permission_evaluator=PermissionEvaluator(),
                         memory_slot=slot)
        try:
            yield loop, llm, spies, plugin
        finally:
            plugin._router = real_router


async def _capture_turn(loop, llm, task="hello"):
    """Run one real turn, returning the system messages the LLM actually saw."""
    captured: list = []
    original = llm.stream_complete

    async def capturing_stream(messages=None, tools=None, on_token=None, **kw):
        captured.extend(messages or [])
        return await original(messages=messages, tools=tools, on_token=on_token, **kw)

    llm.stream_complete = capturing_stream
    await loop.run_turn(task)
    return [m for m in captured if m.get("role") == "system"]


@pytest.mark.asyncio
async def test_the_turn_reads_ambient_context_through_the_router(tmp_path, monkeypatch):
    """The discriminating pair: the router WAS read and the store was NOT. Either assertion
    alone is satisfied by an implementation that reads both stores every turn."""
    router = _mock_router(return_value=_Ctx(agent_memory_md="ROUTER-CONTEXT"))
    async with _session(tmp_path, monkeypatch, router=router) as (loop, llm, store, _p):
        sys_msgs = await _capture_turn(loop, llm)

    router.load_context.assert_awaited_once()
    store.load_context.assert_not_awaited()
    # And the router's answer is what actually reached the model, not just what was fetched.
    assert "ROUTER-CONTEXT" in sys_msgs[0]["content"]
    assert "STORE-CONTEXT" not in sys_msgs[0]["content"]


@pytest.mark.asyncio
async def test_the_router_carries_the_loops_memory_config(tmp_path, monkeypatch):
    """The call site moved receivers, not arguments — index_mode and the shelf size still come
    from agent.memory and still reach whoever answers."""
    router = _mock_router(return_value=_Ctx())
    async with _session(tmp_path, monkeypatch, router=router) as (loop, llm, _store, _p):
        await _capture_turn(loop, llm)

    kwargs = router.load_context.await_args.kwargs
    assert kwargs["index_mode"] is True
    assert kwargs["max_session_history"] == 8


@pytest.mark.asyncio
async def test_writes_and_traces_still_go_to_the_session_store(tmp_path, monkeypatch):
    """recall_scope must never redirect a write. The same turn that read through the router
    stamps its provenance and its activation trace on the STORE."""
    router = _mock_router(return_value=_Ctx(injected_fact_ids=(7,)))
    async with _session(tmp_path, monkeypatch, router=router) as (loop, llm, store, _p):
        await _capture_turn(loop, llm)

    store.set_current_session.assert_called_once()
    store.record_injection_trace.assert_awaited_once()
    # The trace carries the ids the router handed back (primary-owned by 42-02's contract).
    assert list(store.record_injection_trace.await_args.kwargs["injected_ids"]) == [7]
    # The router took NO write: the only name touched on it is the one read verb (the plugin's
    # `self._router or self._store` truth-test shows up as `__bool__`, which is no verb).
    touched = {name.split(".")[0] for name, _a, _k in router.mock_calls} - {"__bool__"}
    assert touched == {"load_context"}


@pytest.mark.asyncio
async def test_without_a_router_the_store_answers_exactly_as_before(tmp_path, monkeypatch):
    """Control: no router (the plugin's fallback when RecallRouter cannot be built) — the
    session's own store answers the ambient read."""
    async with _session(tmp_path, monkeypatch, router=None) as (loop, llm, store, _p):
        sys_msgs = await _capture_turn(loop, llm)

    store.load_context.assert_awaited_once()
    assert "STORE-CONTEXT" in sys_msgs[0]["content"]


@pytest.mark.asyncio
async def test_a_router_without_a_store_stays_memory_free(tmp_path, monkeypatch):
    """The `self._store is None` gate is NOT widened: an occupant with no store (stopped) injects
    nothing, even if a router were somehow still held."""
    router = _mock_router(return_value=_Ctx(agent_memory_md="ROUTER-CONTEXT"))
    async with _session(tmp_path, monkeypatch, router=router) as (loop, llm, _store, plugin):
        plugin._router = None  # stop() must not close the mock; the occupant is now storeless
        await plugin.stop(loop._memory_slot._ctx)
        plugin._router = router
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


def _live_read(monkeypatch, mem: list, seen: dict) -> None:
    """Replace the interactive loop with one that writes a workspace fact and then reads ambient
    context back THROUGH THE SESSION'S OWN ROUTER, mid-session, with both databases open.

    The router is the RUNNING memory plugin's (recorded as it started — the one the `/memory` row
    and the slot use), never a router rebuilt here: a read through anything else would grade an
    object the session might not actually be using.
    """
    async def _read_through_the_router(self):
        running = _the_memory_plugin(mem)
        await running.store.store_fact("ws-marker", f"{WS_MARKER} this project's recollection",
                                       confidence=0.9)
        router = running.router
        seen["router"] = router
        seen["ctx"] = await router.load_context()
        return None

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", _read_through_the_router)


async def test_a_workspace_session_constructs_two_stores_and_opens_one(tmp_path, monkeypatch, fake_home):
    """MEMS-03's precondition, on the filesystem. The twin is CONSTRUCTED (so a `both` session has
    something to open) and, at the default scope, never OPENED — `MemoryStore.__init__` only
    derives paths, `open()` is what creates and migrates a database."""
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
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


async def test_the_global_twin_is_constructed_without_a_bus(tmp_path, monkeypatch, fake_home):
    """A bus subscription is a WRITE path (auto-diary, the predictive gates). The twin is a READ
    handle, so it must not be reachable by any of them — `recall_scope` changes what a session
    reads, never where it writes."""
    _home, _global_dir, _ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    rec = _install_recorders(monkeypatch)

    await _drive()

    primary, twin = rec["store"]
    assert primary.get("bus") is not None, "the primary lost its bus — the control is broken"
    assert twin.get("bus") is None, "the global twin was given a bus and can take writes"


@pytest.mark.parametrize("tool", ["search", "get"])
async def test_each_read_tool_receives_the_very_router_the_loop_got(tool, tmp_path, monkeypatch, fake_home):
    """Criterion 4 made structural: there is no 'the tool bypassed the knob' path to test for,
    because injection and on-demand recall read the SAME object.

    Parametrized rather than one test asserting both wires: with both in one body the first
    assertion SHADOWS the second, so a broken `memory_get` wire is invisible whenever the
    `memory_search` wire is broken too. One id per wire is what makes the two mutations
    distinguishable.
    """
    _home, _global_dir, _ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    rec = _install_recorders(monkeypatch)
    mem = _record_memory_plugins(monkeypatch)
    seen = _record_wiring(monkeypatch)

    await _drive()

    router = _the_memory_plugin(mem).router
    assert router is not None, "the session handed out no router"
    assert seen[tool], f"the {tool} tool was never constructed — the patch did not bite"
    assert seen[tool][0] is router, (
        f"memory_{tool} reads a different object than ambient injection does"
    )
    # Both directions: it is a router, not the store wearing the name.
    assert seen[tool][0] is not seen["instances"][0]


async def test_the_remember_tool_keeps_the_raw_store(tmp_path, monkeypatch, fake_home):
    """The write verb never sees the router. `RecallRouter` has no `store_fact`, so a swap here
    would be an AttributeError at first use — loudly, but only for whoever tried to remember
    something. This asserts it before a user finds it."""
    _home, _global_dir, _ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    rec = _install_recorders(monkeypatch)
    mem = _record_memory_plugins(monkeypatch)
    seen = _record_wiring(monkeypatch)

    await _drive()

    router = _the_memory_plugin(mem).router
    assert seen["remember"], "MemoryRememberTool was never constructed — the patch did not bite"
    assert seen["remember"][0] is seen["instances"][0], "remember() lost the session's own store"
    assert seen["remember"][0] is not router, "remember() writes through the recall router"


async def test_a_default_scope_session_reads_only_the_workspace(tmp_path, monkeypatch, fake_home):
    """MEMS-02 criterion 1, from a LIVE session: another project's recollections do not appear in
    this project's ambient context, even though the global store exists and holds a rendering
    fact."""
    _home, global_dir, _ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    await _seed_global_decoy(global_dir)
    rec = _install_recorders(monkeypatch)
    mem = _record_memory_plugins(monkeypatch)
    seen: dict = {}
    _live_read(monkeypatch, mem, seen)

    await _drive()

    assert seen.get("ctx") is not None, "the live read never ran — the stubbed loop did not fire"
    md = seen["ctx"].agent_memory_md
    assert WS_MARKER in md, f"the session did not read its own store: {md!r}"
    assert GLOBAL_DECOY not in md, (
        "a default-scope session injected another project's memory"
    )
    assert seen["router"].scope == "workspace"


async def test_a_both_scope_session_reads_both_stores_with_origin_tokens(tmp_path, monkeypatch, fake_home):
    """The knob reaches a running session: `recall_scope: both` in the workspace agent yaml and
    the live ambient read spans both databases, every line naming which store it came from."""
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    _write_scoped_agent(ws / "agents", "both")
    await _seed_global_decoy(global_dir)
    rec = _install_recorders(monkeypatch)
    mem = _record_memory_plugins(monkeypatch)
    seen: dict = {}
    _live_read(monkeypatch, mem, seen)

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
    tmp_path, monkeypatch, fake_home
):
    """LAYR-03's control. With no workspace layer `state_dir == cfg_path`, so a twin would be a
    SECOND aiosqlite connection to the SAME file. The knob is still READ (configured_scope keeps
    what the yaml asked for) and still collapses — proving the collapse, not a missing knob."""
    _home, global_dir, _proj = _global_only_start(tmp_path, monkeypatch, fake_home)
    _write_scoped_agent(global_dir / "agents", "both")
    rec = _install_recorders(monkeypatch)
    mem = _record_memory_plugins(monkeypatch)

    await _drive()

    assert len(rec["store"]) == 1, (
        f"a workspace-less session must build exactly one store; got {len(rec['store'])}"
    )
    router = _the_memory_plugin(mem).router
    assert router is not None, "the workspace-less session got no router"
    assert router.scope == "workspace", "the scope did not collapse without a second store"
    assert router.configured_scope == "both", (
        "the yaml's knob never reached the router — this test would then be grading a missing "
        "knob rather than the collapse"
    )


async def test_the_opened_global_handle_is_closed_at_shutdown(tmp_path, monkeypatch, fake_home):
    """Pitfall 6: aiosqlite's worker thread is NON-DAEMON, so a leaked handle hangs interpreter
    shutdown. The twin is opened by this drive (the live `both` read forces it) and must be
    closed by the resource-owning window's finally."""
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    _write_scoped_agent(ws / "agents", "both")
    await _seed_global_decoy(global_dir)
    rec = _install_recorders(monkeypatch)
    mem = _record_memory_plugins(monkeypatch)
    seen = _record_wiring(monkeypatch)
    _live_read(monkeypatch, mem, seen)

    await _drive()

    assert len(seen["instances"]) == 2, "expected the primary and the twin"
    primary, twin = seen["instances"]
    # Premise: the twin really was opened, or "it was closed" is vacuous.
    assert GLOBAL_DECOY in seen["ctx"].agent_memory_md, "the twin was never read"
    assert twin in seen["closed"], "the router's global handle leaked — never closed"
    assert primary in seen["closed"], "the primary leaked — the control is broken"

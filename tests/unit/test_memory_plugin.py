"""MemoryPlugin in isolation (47-05, MEMP-01/02): a PluginContext built directly over a real store.
Lifecycle order and failure domains are asserted with call spies, prompt parity against the legacy
loop block's own rendering. NOTE: nothing in a real session calls MemoryPlugin until 47-06."""
from __future__ import annotations

import logging

import pytest

from localharness.cli.session_accumulator import SessionAccumulator
from localharness.memory.config import MemoryConfig
from localharness.core.bus import EventBus
from localharness.memory.browse import StoreBrowse
from localharness.memory.consolidation import ConsolidationScheduler
from localharness.memory.plugin import MemoryPlugin
from localharness.memory.router import RecallRouter
from localharness.memory.sqlite import MemoryStore
from localharness.plugins.api import (
    ContextBudget, ContextContribution, PluginContext, PluginPaths, SessionInfo,
)
from localharness.provider.idle_llm import LLMTextAdapter
from localharness.tools.registry import ToolRegistry
from tests.conftest import MockLLMClient

BIG = ContextBudget(max_chars=10**6, max_session_history=200)
SITTING = "sitting-1"


def _ctx(tmp_path, *, workspace=False, session=True, **mem) -> PluginContext:
    g, s = tmp_path / "global", tmp_path / "state"
    g.mkdir(parents=True, exist_ok=True)
    s.mkdir(parents=True, exist_ok=True)
    ws = None
    if workspace:
        ws = tmp_path / "proj" / ".localharness"
        ws.mkdir(parents=True, exist_ok=True)
    llm = MockLLMClient([])
    mem.setdefault("consolidation", {"enabled": True})
    return PluginContext(
        bus=EventBus(), tools=ToolRegistry(), hooks=None, config=None,
        agent_config=MemoryConfig(**mem), paths=PluginPaths(global_config_dir=g, workspace=ws, state_dir=s),
        llm=llm, idle_llm=LLMTextAdapter(llm),
        session=SessionInfo(agent_id="orchestrator", division_id=None, sitting_id=SITTING, model="m",
                            context_tokens=131072, budget={"max_tokens": 1}) if session else None)


@pytest.fixture(autouse=True)
def _no_dreaming_pass(monkeypatch):
    """A seeded store has un-embedded facts: the session-start pass would load the embedding model."""
    async def never(self): return False
    monkeypatch.setattr(ConsolidationScheduler, "should_run", never)


def _spy(monkeypatch, log: list, cls, method: str, label: str, *, raises: BaseException | None = None):
    original = getattr(cls, method)

    async def wrapper(self, *a, **kw):
        log.append(label)
        if raises is not None:
            raise raises
        return await original(self, *a, **kw)

    monkeypatch.setattr(cls, method, wrapper)


def _spy_all(monkeypatch, log, **raises):
    for cls, method, label in (
        (MemoryStore, "open", "store.open"), (MemoryStore, "create_session", "create_session"),
        (SessionAccumulator, "open", "accumulator.open"), (ConsolidationScheduler, "start", "scheduler.start"),
        (ConsolidationScheduler, "stop", "scheduler.stop"), (SessionAccumulator, "close", "accumulator.close"),
        (MemoryStore, "end_session", "end_session"), (RecallRouter, "close", "router.close"),
        (MemoryStore, "close", "store.close"),
    ):
        _spy(monkeypatch, log, cls, method, label, raises=raises.get(label))


async def _seed(ctx) -> None:
    store = MemoryStore(agent_id="orchestrator", division_id="default", org_id="default",
                        base_dir=str(ctx.paths.state_dir), global_base_dir=str(ctx.paths.global_config_dir))
    await store.open()
    await store.store_fact("user-editor", "The user edits in Neovim with a tiling WM.", tags=["pref"])
    await store.store_fact("project-lang", "The project is Python 3.12 on uv.", tags=["project"])
    await store.close()


async def _started(ctx) -> MemoryPlugin:
    p = MemoryPlugin()
    await p.tools(ctx)
    await p.start(ctx)
    return p


async def test_tools_construct_and_open_nothing(tmp_path, monkeypatch) -> None:
    ctx, log = _ctx(tmp_path), []
    _spy_all(monkeypatch, log)
    tools = await MemoryPlugin().tools(ctx)
    assert {t.info().name for t in tools} == {"memory_search", "memory_get", "remember"}
    assert log == []
    assert not list(tmp_path.rglob("memory.db"))


async def test_tool_declarations_unchanged(tmp_path) -> None:
    infos = {t.info().name: t.info() for t in await MemoryPlugin().tools(_ctx(tmp_path))}
    assert infos["memory_search"].result_origin == "untrusted"
    assert infos["memory_get"].result_origin == "untrusted"
    assert infos["remember"].result_origin == "trusted"  # G2
    for i in infos.values():
        assert (i.ingest, i.host, i.gate_family) == ("none", "safe", "allow")


async def test_twin_only_with_a_workspace(tmp_path, monkeypatch) -> None:
    p = MemoryPlugin()
    await p.tools(_ctx(tmp_path))
    assert p._router._global_store is None
    built: list[dict] = []
    original = MemoryStore.__init__

    def recording(self, **kw):
        built.append(kw)
        original(self, **kw)

    monkeypatch.setattr(MemoryStore, "__init__", recording)
    opened: list = []
    _spy(monkeypatch, opened, MemoryStore, "open", "open")
    ctx = _ctx(tmp_path / "w", workspace=True)
    p = MemoryPlugin()
    await p.tools(ctx)
    g = str(ctx.paths.global_config_dir)
    assert built[0] == dict(agent_id="orchestrator", division_id="default", org_id="default",
                            base_dir=str(ctx.paths.state_dir), global_base_dir=g, bus=ctx.bus)
    assert built[1] == dict(agent_id="orchestrator", division_id="default", org_id="default",
                            base_dir=g, global_base_dir=g)
    assert p._router._global_store is not None and opened == []


async def test_start_opens_in_todays_order(tmp_path, monkeypatch) -> None:
    ctx, log = _ctx(tmp_path), []
    _spy_all(monkeypatch, log)
    p = await _started(ctx)
    assert log == ["store.open", "create_session", "accumulator.open", "scheduler.start"]
    assert p.startup_warnings == []
    await p.stop(ctx)


async def test_stop_closes_in_todays_order(tmp_path, monkeypatch) -> None:
    ctx, log = _ctx(tmp_path), []
    _spy_all(monkeypatch, log)
    p = await _started(ctx)
    log.clear()
    await p.stop(ctx)
    assert log == ["scheduler.stop", "accumulator.close", "end_session", "router.close", "store.close"]
    log.clear()
    await p.stop(ctx)  # a second stop is a no-op
    assert log == []


async def test_consolidation_off_starts_no_scheduler(tmp_path, monkeypatch) -> None:
    ctx, log = _ctx(tmp_path, consolidation={"enabled": False}), []
    _spy_all(monkeypatch, log)
    p = await _started(ctx)
    assert "scheduler.start" not in log and p._sched is None and p.startup_warnings == []
    await p.stop(ctx)


@pytest.mark.parametrize("target,warning", [
    ((MemoryStore, "create_session"), "session-start: boom"),
    ((SessionAccumulator, "open"), "session-accumulator: boom"),
    ((ConsolidationScheduler, "__init__"), "memory consolidation: boom"),
])
async def test_soft_failure_each_alone(tmp_path, monkeypatch, target, warning) -> None:
    await _seed(_ctx(tmp_path))
    ctx = _ctx(tmp_path)

    def boom(*a, **kw): raise RuntimeError("boom")
    async def aboom(*a, **kw): raise RuntimeError("boom")
    cls, method = target
    monkeypatch.setattr(cls, method, boom if method == "__init__" else aboom)
    p = await _started(ctx)
    assert p.startup_warnings == [warning]
    assert p._store._db is not None  # the hard piece is open
    if method != "create_session":
        assert p._session_started
    if cls is not ConsolidationScheduler:
        assert p._sched is not None
    c = await p.context(ctx, "hi", BIG)
    assert any(h == "Agent Memory" and "Neovim" in b for h, b in c.sections)
    await p.stop(ctx)


async def test_store_open_failure_reraises_and_leaves_nothing_open(tmp_path, monkeypatch) -> None:
    ctx, log = _ctx(tmp_path), []
    _spy_all(monkeypatch, log, **{"store.open": RuntimeError("disk")})
    p = MemoryPlugin()
    await p.tools(ctx)
    with pytest.raises(RuntimeError, match="disk"):
        await p.start(ctx)
    assert not {"create_session", "accumulator.open", "scheduler.start"} & set(log)
    assert p._store is None and p._router is None


async def test_interrupt_mid_start_closes_what_opened(tmp_path, monkeypatch) -> None:
    ctx, log = _ctx(tmp_path), []
    _spy_all(monkeypatch, log, **{"scheduler.start": KeyboardInterrupt()})
    p = MemoryPlugin()
    await p.tools(ctx)
    with pytest.raises(KeyboardInterrupt):
        await p.start(ctx)
    for label in ("accumulator.close", "router.close", "store.close"):
        assert log.count(label) == 1, (label, log)
    assert "end_session" not in log


async def _legacy_render(ctx, cfg: MemoryConfig) -> str:
    """The legacy loop block (agent/loop.py, pre-cut) over a fresh store+router, as the loop joins it."""
    store = MemoryStore(agent_id="orchestrator", division_id="default", org_id="default",
                        base_dir=str(ctx.paths.state_dir), global_base_dir=str(ctx.paths.global_config_dir))
    await store.open()
    try:
        store.set_current_session(SITTING)
        mc = await RecallRouter(store, None, scope=cfg.recall_scope).load_context(
            index_mode=cfg.index_mode, max_session_history=cfg.max_session_history_entries,
            max_chars=cfg.max_notes_chars)
        parts = ["PROMPT"]
        if mc.division_md:
            parts.append("## Division Context\n" + mc.division_md)
        if mc.agent_memory_md:
            parts.append("## Agent Memory\n" + mc.agent_memory_md)
        return "\n\n".join(parts)
    finally:
        await store.close()


async def test_context_matches_the_legacy_loop_block(tmp_path) -> None:
    ctx = _ctx(tmp_path, consolidation={"enabled": False})
    await _seed(ctx)
    div = ctx.paths.global_config_dir / "divisions" / "default"
    div.mkdir(parents=True, exist_ok=True)
    (div / "DIVISION.md").write_text("Division voice.", encoding="utf-8")
    legacy = await _legacy_render(ctx, ctx.agent_config)
    p = await _started(ctx)
    c = await p.context(ctx, "turn", BIG)
    await p.stop(ctx)
    assert "PROMPT" + "".join(f"\n\n## {h}\n{b}" for h, b in c.sections) == legacy
    assert "## Division Context\nDivision voice." in legacy and "Neovim" in legacy


async def test_context_applies_the_ceiling(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path, consolidation={"enabled": False})
    p = await _started(ctx)
    seen: list[dict] = []
    original = RecallRouter.load_context

    async def spy(self, **kw):
        seen.append(kw)
        return await original(self, **kw)

    monkeypatch.setattr(RecallRouter, "load_context", spy)
    await p.context(ctx, "t", ContextBudget(max_chars=500, max_session_history=2))
    await p.context(ctx, "t", BIG)
    await p.stop(ctx)
    cfg = ctx.agent_config
    assert seen[0] == dict(index_mode=cfg.index_mode, max_session_history=2, max_chars=500)
    assert seen[1] == dict(index_mode=cfg.index_mode, max_session_history=cfg.max_session_history_entries,
                           max_chars=cfg.max_notes_chars)


async def _traces(ctx, **mem):
    ctx = _ctx(ctx, consolidation={"enabled": False}, **mem)
    p = await _started(ctx)
    calls: list[dict] = []
    original = MemoryStore.record_injection_trace

    async def spy(self, **kw):
        calls.append(kw)
        return await original(self, **kw)

    import localharness.memory.sqlite as sq
    sq.MemoryStore.record_injection_trace = spy
    try:
        await p.context(ctx, "the stimulus", BIG)
    finally:
        sq.MemoryStore.record_injection_trace = original
        await p.stop(ctx)
    return calls


async def test_context_records_the_injection_trace_even_when_empty(tmp_path) -> None:
    calls = await _traces(tmp_path)  # empty store: #96 still records a row
    assert calls == [dict(stimulus="the stimulus", injected_ids=[], session_id=SITTING)]


async def test_trace_off_records_nothing(tmp_path) -> None:
    assert await _traces(tmp_path, trace_ambient_injection=False) == []


async def test_context_failure_returns_empty(tmp_path, monkeypatch, caplog) -> None:
    ctx = _ctx(tmp_path, consolidation={"enabled": False})
    p = await _started(ctx)

    async def boom(self, **kw): raise RuntimeError("db gone")
    monkeypatch.setattr(RecallRouter, "load_context", boom)
    with caplog.at_level(logging.WARNING, logger="localharness.memory.plugin"):
        assert await p.context(ctx, "t", BIG) == ContextContribution()
    assert "memory context load failed — no memory injected this turn" in caplog.text
    await p.stop(ctx)


@pytest.mark.parametrize("workspace", [False, True])
async def test_browse_and_legacy_handles(tmp_path, workspace) -> None:
    ctx = _ctx(tmp_path, workspace=workspace)
    p = MemoryPlugin()
    assert p.browse() is None
    await p.tools(ctx)
    b = p.browse()
    assert isinstance(b, StoreBrowse)
    assert (b._store, b._router) == (p._store, p._router)
    assert b._identity == (str(ctx.paths.workspace.resolve().parent) if workspace else "")
    assert p.legacy_handles() == (p._store, p._router)
    assert p.bind_subagent(ctx) is None


async def test_close_out_failure_prints_the_yellow_line(tmp_path, monkeypatch, capsys) -> None:
    ctx, log = _ctx(tmp_path, consolidation={"enabled": False}), []
    _spy_all(monkeypatch, log, end_session=RuntimeError("boom"))
    p = await _started(ctx)
    await p.stop(ctx)
    assert "⚠ session close-out skipped: boom" in capsys.readouterr().err
    assert log[-1] == "store.close"


async def test_end_session_gets_todays_kwargs(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path, consolidation={"enabled": False})
    seen: list = []

    async def record(self, session_id, **kw): seen.append((session_id, kw))
    monkeypatch.setattr(MemoryStore, "end_session", record)
    p = await _started(ctx)
    ctx.session.exit_reason = "error"
    await p.stop(ctx)
    assert seen == [(SITTING, dict(exit_reason="error", summary=None, turn_count=0, action_count=0,
                                   tokens_in=0, tokens_out=0))]


async def test_tools_without_a_session_fail_loudly(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="ctx.session"):
        await MemoryPlugin().tools(_ctx(tmp_path, session=False))

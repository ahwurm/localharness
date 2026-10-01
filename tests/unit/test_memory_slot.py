"""The memory slot (44-12, PAPI-04, criterion 5): at most one occupant, and it may be empty. The
prompt asks it for its per-turn section only when someone occupies it, AFTER the guardrails; with
no occupant there is no slot section and the guardrails are still there. Every call into the
occupant is contained — a failing memory plugin costs the turn its memory section, never the turn."""
from __future__ import annotations

import logging

import pytest

from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig
from localharness.core.bus import EventBus
from localharness.plugins.api import (
    BrowseQuery, ContextBudget, ContextContribution, MemorySlotPlugin, PluginContext,
    PluginManifest, PluginPaths,
)
from localharness.plugins.lifecycle import start_plugins
from localharness.plugins.plan import build_load_plan
from localharness.plugins.resolve import PluginSettings, Resolution
from localharness.plugins.slot import MemorySlot
from localharness.tools.registry import ToolRegistry
from tests.conftest import FakeLLMResponse, MockLLMClient

SECTIONS = (("Division Context", "D"), ("Agent Memory", "M"))


class _Browse:
    """Every verb of the MemoryBrowse protocol."""

    async def search(self, query: BrowseQuery): return []
    async def get(self, name: str): return None
    async def edit(self, name: str, content: str): return {}
    async def forget(self, name: str): return False
    async def promote(self, name: str): return {}


class _Handle:
    async def persist_reduce_trace(self, question, trace): return None


def _memory_plugin(name: str = "recall", *, seen: list | None = None, **methods) -> type:
    """A memory-slot plugin whose context() records (ctx, turn, budget) and returns SECTIONS."""
    async def context(self, ctx, turn, budget):
        if seen is not None:
            seen.append((ctx, turn, budget))
        return ContextContribution(sections=SECTIONS)

    return type(f"Memory_{name}", (MemorySlotPlugin,), {
        "__doc__": f"test memory plugin {name}",
        "manifest": PluginManifest(name=name, version="1.0", kind="memory"),
        "context": context, **methods})


def _ctx(tmp_path) -> PluginContext:
    return PluginContext(bus=EventBus(), tools=ToolRegistry(), hooks=None, config=None,
                         agent_config=None, paths=PluginPaths(tmp_path, None, tmp_path), llm=None)


def _loop(tmp_path, **kwargs) -> tuple[AgentLoop, list[str]]:
    """A loop with an org GUARDRAILS.md ("RULES-G"), a memory budget of 1234 chars / 3 entries,
    and no legacy memory; returns it with the list of system prompts it sends."""
    guardrails = tmp_path / "orgs" / "default" / "GUARDRAILS.md"
    guardrails.parent.mkdir(parents=True, exist_ok=True)
    guardrails.write_text("RULES-G", encoding="utf-8")
    cfg = AgentConfig(name="slot-agent", role="You are a test assistant.",
                      memory={"max_notes_chars": 1234, "max_session_history_entries": 3})
    llm = MockLLMClient([FakeLLMResponse(content="Done.")])
    prompts: list[str] = []
    original = llm.stream_complete

    async def recording(messages=None, tools=None, on_token=None, **kw):
        prompts.append(next(m["content"] for m in messages if m.get("role") == "system"))
        return await original(messages=messages, tools=tools, on_token=on_token, **kw)

    llm.stream_complete = recording
    loop = AgentLoop(config=cfg, llm=llm, bus=EventBus(), context_manager=ContextManager(),
                     tool_registry=None, permission_evaluator=PermissionEvaluator(),
                     guardrails_path=guardrails, **kwargs)
    return loop, prompts


# --- the holder --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_empty_slot_answers_nothing():
    slot = MemorySlot()

    assert slot.occupied is False and slot.occupant_name is None
    assert await slot.context("turn", ContextBudget(max_chars=10, max_session_history=1)) == \
        ContextContribution()
    assert slot.browse() is None and slot.bind_subagent() is None


@pytest.mark.asyncio
async def test_an_occupant_answers_through_the_slot_with_its_own_context(tmp_path):
    ctx, seen, bound = _ctx(tmp_path), [], []
    browse, handle = _Browse(), _Handle()
    plugin = _memory_plugin(seen=seen, browse=lambda self: browse,
                            bind_subagent=lambda self, c: bound.append(c) or handle)()
    slot = MemorySlot(plugin, ctx, "recall")
    budget = ContextBudget(max_chars=10, max_session_history=1)

    assert slot.occupied is True and slot.occupant_name == "recall"
    assert (await slot.context("the turn", budget)).sections == SECTIONS
    assert seen == [(ctx, "the turn", budget)]
    assert slot.browse() is browse
    assert slot.bind_subagent() is handle and bound == [ctx]


async def _context_raises(self, ctx, turn, budget):
    raise RuntimeError("store gone")


async def _context_exits(self, ctx, turn, budget):
    raise SystemExit(6)


async def _context_junk(self, ctx, turn, budget):
    return {"sections": "not a ContextContribution"}


def _raising(exc: BaseException):
    def method(self, *args):
        raise exc
    return method


@pytest.mark.asyncio
@pytest.mark.parametrize("verb, method", [
    ("context", _context_raises), ("context", _context_exits), ("context", _context_junk),
    ("browse", _raising(SystemExit(5))), ("browse", lambda self: "not a browse API"),
    ("bind_subagent", _raising(RuntimeError("no handle"))), ("bind_subagent", lambda self, ctx: 42),
], ids=["context-raises", "context-exits", "context-junk", "browse-exits", "browse-junk", "bind-raises", "bind-junk"])
async def test_a_failing_occupant_answers_nothing_and_is_named(tmp_path, caplog, verb, method):
    """Raising (sys.exit() included) or answering with the wrong type: the slot answers as if
    empty, and a warning names the occupant and the verb."""
    slot = MemorySlot(_memory_plugin(**{verb: method})(), _ctx(tmp_path), "recall")

    with caplog.at_level(logging.WARNING, logger="localharness.plugins.slot"):
        got = (await slot.context("t", ContextBudget(max_chars=1, max_session_history=1))
               if verb == "context" else getattr(slot, verb)())

    assert got == (ContextContribution() if verb == "context" else None)
    assert [r for r in caplog.records if "recall" in r.getMessage() and verb in r.getMessage()]


# --- the prompt seam ----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_occupied_slot_adds_its_sections_after_the_guardrails(tmp_path):
    seen: list = []
    ctx = _ctx(tmp_path)
    loop, prompts = _loop(tmp_path, memory_slot=MemorySlot(_memory_plugin(seen=seen)(), ctx, "recall"))

    await loop.run_turn("hello slot")

    assert "\n\n## Guardrails\nRULES-G\n\n## Division Context\nD\n\n## Agent Memory\nM" in prompts[0]
    assert prompts[0].count("## Guardrails") == 1
    assert seen == [(ctx, "hello slot", ContextBudget(max_chars=1234, max_session_history=3))]


@pytest.mark.asyncio
async def test_an_empty_section_adds_no_heading(tmp_path):
    async def context(self, ctx, turn, budget):
        return ContextContribution(sections=(("Division Context", ""), ("Agent Memory", "M")))

    loop, prompts = _loop(tmp_path, memory_slot=MemorySlot(
        _memory_plugin(context=context)(), _ctx(tmp_path), "recall"))

    await loop.run_turn("hello")

    assert "\n\n## Guardrails\nRULES-G\n\n## Agent Memory\nM" in prompts[0]
    assert "## Division Context" not in prompts[0]


@pytest.mark.asyncio
async def test_the_prompt_asks_the_slot_only_when_it_is_occupied(tmp_path):
    asked: list = []

    class Unoccupied:
        occupied = False

        async def context(self, turn, budget):
            asked.append(turn)
            return ContextContribution(sections=SECTIONS)

    loop, prompts = _loop(tmp_path, memory_slot=Unoccupied())

    await loop.run_turn("hello")

    assert asked == [] and "## Division Context" not in prompts[0]


@pytest.mark.asyncio
async def test_a_failing_occupant_costs_the_turn_its_memory_section_only(tmp_path, caplog):
    async def context(self, ctx, turn, budget):
        raise RuntimeError("store gone")

    loop, prompts = _loop(tmp_path, memory_slot=MemorySlot(
        _memory_plugin(context=context)(), _ctx(tmp_path), "recall"))

    with caplog.at_level(logging.WARNING, logger="localharness.plugins.slot"):
        reply = await loop.run_turn("hello")

    assert isinstance(reply, str)
    assert "\n\n## Guardrails\nRULES-G" in prompts[0]
    assert "## Division Context" not in prompts[0] and "## Agent Memory" not in prompts[0]
    assert [r for r in caplog.records if "recall" in r.getMessage()]


@pytest.mark.asyncio
@pytest.mark.parametrize("slot", [None, MemorySlot()], ids=["no slot", "empty slot"])
async def test_no_occupant_means_no_slot_section_and_the_guardrails_stay(tmp_path, slot):
    """Byte-identical to a loop constructed without the seam at all."""
    loop, prompts = _loop(tmp_path / "with", memory_slot=slot)
    bare, bare_prompts = _loop(tmp_path / "bare")

    await loop.run_turn("hello")
    await bare.run_turn("hello")

    assert "\n\n## Guardrails\nRULES-G" in prompts[0]
    assert prompts[0] == bare_prompts[0]


# --- the lifecycle seats at most one occupant ---------------------------------------------------------------


def _resolution(*classes) -> Resolution:
    names = [c.manifest.name for c in classes]
    return Resolution(
        build_load_plan(bundled=classes, discovered=[], enabled={n: True for n in names},
                        imported={}, version="0.15.0", core_keys=frozenset(),
                        extra_installed=lambda extra: True),
        {c.manifest.name: c for c in classes}, {n: PluginSettings(None, None) for n in names},
        {n: True for n in names}, ())


async def _start(resolution: Resolution, tmp_path):
    return await start_plugins(resolution, bus=EventBus(), registry=ToolRegistry(), hooks=None,
                               llm=None, paths=PluginPaths(tmp_path, None, tmp_path))


@pytest.mark.asyncio
async def test_two_memory_plugins_leave_the_slot_empty(tmp_path):
    resolution = _resolution(_memory_plugin("one"), _memory_plugin("two"))
    assert resolution.plan.memory_occupant is None and resolution.plan.order == ()  # both refused

    result = await _start(resolution, tmp_path)

    assert result.slot.occupied is False


@pytest.mark.asyncio
async def test_one_running_memory_plugin_occupies_the_slot(tmp_path):
    seen: list = []
    result = await _start(_resolution(_memory_plugin("recall", seen=seen)), tmp_path)

    assert result.slot.occupied is True and result.slot.occupant_name == "recall"
    await result.slot.context("t", ContextBudget(max_chars=1, max_session_history=1))
    assert seen[0][0] is result.running[0].ctx


@pytest.mark.asyncio
async def test_a_memory_plugin_that_fails_to_start_leaves_the_slot_empty(tmp_path):
    async def start(self, ctx):
        raise RuntimeError("no store")

    result = await _start(_resolution(_memory_plugin("recall", start=start)), tmp_path)

    assert "recall" in result.failed and result.slot.occupied is False


@pytest.mark.asyncio
async def test_seat_puts_an_occupant_on_this_slot_in_place(tmp_path):
    """46-03: start_cmd seats the transitional occupant (ROADMAP D3) on the slot object every holder
    already has — so seating mutates, never replaces."""
    from localharness.memory.browse import StoreBrowse
    from localharness.memory.sqlite import MemoryStore

    store = MemoryStore(agent_id="a", division_id="d", org_id="default", base_dir=str(tmp_path))
    await store.open()
    try:
        slot, occupant = MemorySlot(), StoreBrowse(store)
        slot.seat(occupant, name="memory")
        assert slot.occupied and slot.occupant_name == "memory"
        assert slot.browse() is occupant
        assert await slot.context("t", ContextBudget(max_chars=10, max_session_history=1)) == \
            ContextContribution()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_the_prompt_is_identical_with_the_occupant_seated(tmp_path):
    """46-06 (D3): the transitional occupant only browses — its context() is the inherited empty one,
    so seating it over a real store changes nothing the model sees."""
    from localharness.memory.browse import StoreBrowse
    from localharness.memory.sqlite import MemoryStore

    store = MemoryStore(agent_id="a", division_id="d", org_id="default", base_dir=str(tmp_path / "s"))
    await store.open()
    try:
        seated = MemorySlot()
        seated.seat(StoreBrowse(store), name="memory")
        loop, prompts = _loop(tmp_path / "seated", memory_slot=seated)
        empty, empty_prompts = _loop(tmp_path / "empty", memory_slot=MemorySlot())
        await loop.run_turn("hello")
        await empty.run_turn("hello")
        assert seated.occupied and prompts[0] == empty_prompts[0]
    finally:
        await store.close()

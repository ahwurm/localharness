"""A child's per-store tool state is released when its dispatch ends: cancelled, raising or twice."""
from __future__ import annotations

import asyncio

import pytest

from localharness.tools.base import ToolResult, ToolSchema
from localharness.tools.builtin import release_agent_store_tools
from localharness.tools.registry import ContributedTool, ToolRegistry
from tests.unit.test_capabilities import _builtin_registry
from tests.unit.test_subagent_narrowing import FakeLoop, config_child, install_fake_loop, make_runner

VERIFIER_TASK = "claim: x\nentity: y\nsource_url: z"


@pytest.fixture(autouse=True)
def _ledger(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_VERIFICATION_LEDGER_DIR", str(tmp_path / "ledger"))


@pytest.fixture
async def base():
    return await _builtin_registry()


class SpyTool:
    def __init__(self, released: list, raises: bool = False, store=None, name: str = "spy_open") -> None:
        self.released, self.raises, self.store, self.name = released, raises, store, name
        self.event = asyncio.Event()

    def info(self) -> ToolSchema:
        return ToolSchema(name=self.name, description="d", parameters={}, group="web", gate_family="network",
                          ingest="untrusted", host="safe", result_origin="untrusted")

    async def run(self, **kwargs) -> ToolResult:  # pragma: no cover - never dispatched
        return ToolResult(output="")

    def bound_to(self, store):
        bound = SpyTool(self.released, self.raises, store, self.name)
        bound.event = self.event
        return bound

    async def release_store(self, store) -> None:
        self.released.append(self.store is store)
        self.event.set()
        if self.raises:
            raise RuntimeError("release failed")


async def with_spy(base, raises: bool = False) -> SpyTool:
    spy = SpyTool([], raises)
    await base.register(spy, scope="global", source_plugin="browser")
    return spy


def runner_for(base, llm):
    return make_runner(base, llm, max_subagent_depth=1, load_agent=lambda n: config_child(n, ("read", "spy_open")))


@pytest.mark.parametrize("agent,task", [("web-researcher", "t"), ("search-verifier", VERIFIER_TASK), ("cfg", "t")])
async def test_each_dispatch_releases_the_spy_once_with_the_childs_own_store(base, monkeypatch, mock_llm_client, agent, task):
    loops = install_fake_loop(monkeypatch)
    spy = await with_spy(base)
    await runner_for(base, mock_llm_client([]))(agent, task)
    assert spy.released == [True]
    assert loops[0].context_manager._content_store is not None


async def test_a_second_release_and_duplicate_copies_are_harmless(mock_llm_client):
    released: list = []
    registry = ToolRegistry()
    store = object()
    for i in range(4):
        await registry.register(SpyTool(released, store=store, name=f"spy_{i}"), scope="global", source_plugin="browser")
    await release_agent_store_tools(registry, store)
    await release_agent_store_tools(registry, store)
    assert released == [True] * 8


async def test_a_cancelled_child_still_releases(base, monkeypatch, mock_llm_client):
    class HangingLoop(FakeLoop):
        async def run_turn(self, brief, *a, **k):
            await asyncio.Event().wait()

    monkeypatch.setattr("localharness.agent.loop.AgentLoop", HangingLoop)
    spy = await with_spy(base)
    task = asyncio.create_task(runner_for(base, mock_llm_client([]))("web-researcher", "t"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(spy.event.wait(), timeout=1)
    assert spy.released == [True]


async def test_a_release_that_raises_does_not_change_the_result(monkeypatch, mock_llm_client):
    install_fake_loop(monkeypatch)
    results = []
    for raises in (False, True):
        registry = await _builtin_registry()
        spy = await with_spy(registry, raises)
        results.append(await runner_for(registry, mock_llm_client([]))("cfg", "t"))
        assert spy.released == [True]
    assert str(results[0]) == str(results[1]) and results[0].status == results[1].status


async def test_a_contributed_tool_forwards_release_only_when_the_inner_tool_has_it():
    spy = SpyTool([])
    await ContributedTool(spy, "browser").release_store("s")
    assert spy.released == [False]

    class Plain:
        def info(self):  # pragma: no cover
            return spy.info()

    await ContributedTool(Plain(), "browser").release_store("s")

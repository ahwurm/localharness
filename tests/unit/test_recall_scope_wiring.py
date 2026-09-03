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
from unittest.mock import AsyncMock, MagicMock

import pytest

from localharness.agent.loop import AgentLoop
from localharness.agent.context import ContextManager
from localharness.agent.permissions import PermissionEvaluator
from localharness.core.bus import EventBus


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

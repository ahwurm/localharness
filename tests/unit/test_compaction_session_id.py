"""Every compaction event names the session it happened in.

The root context manager is built with the sitting id, and the loop re-asserts its session on
the manager at the start of every turn, so a `CompactionTriggered` is written to
`sessions/<sitting id>.jsonl` and the empty-id file `sessions/.jsonl` is never created.

Children publish no `CompactionTriggered`: a child's context manager has no bus. The child half
of the contract is therefore the setter: after a turn a child's manager holds the child's own
non-empty session id. Nothing here wires a bus into a child.
"""
import pytest

from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop, Session
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig
from localharness.core.bus import EventBus
from localharness.core.events import CompactionTriggered
from tests.conftest import FakeLLMResponse, MockLLMClient


def _loop(llm, bus, ctx, **kw) -> AgentLoop:
    return AgentLoop(
        config=AgentConfig(name="root", role="Test agent."),
        llm=llm,
        bus=bus,
        context_manager=ctx,
        tool_registry=None,
        permission_evaluator=PermissionEvaluator(),
        **kw,
    )


class _ShrinkingPipeline:
    """A pipeline whose LLM stage always shrinks the transcript to one tiny message."""

    async def run(self, messages, budget):
        return [{"role": "user", "content": "y"}], True

    async def run_deterministic(self, messages, budget):
        return messages, False

    async def run_llm(self, messages, budget):
        return [{"role": "user", "content": "y"}], True


# ---------------------------------------------------------------------------
# The setter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_session_id_marks_the_next_compaction_event():
    published: list = []

    class FakeBus:
        async def publish(self, event):
            published.append(event)

    cm = ContextManager(max_context_tokens=100, pipeline=_ShrinkingPipeline(), bus=FakeBus(),
                        agent_id="agent-1")
    cm.set_session_id("s-9")
    await cm.build_messages([{"role": "user", "content": "x" * 5000}] * 5, tool_schemas=None)

    events = [e for e in published if isinstance(e, CompactionTriggered)]
    assert len(events) == 1
    assert events[0].session_id == "s-9"


# ---------------------------------------------------------------------------
# start wires the sitting id into the root manager
# ---------------------------------------------------------------------------

async def test_start_builds_the_root_manager_with_the_sitting_id(tmp_path, monkeypatch):
    from localharness.cli.start_cmd import _start_async
    from tests.unit.test_start_cmd import _stub_start_boundaries

    captured: list = []
    _stub_start_boundaries(tmp_path, monkeypatch, capture_session_id=captured)
    seen: list = []
    real_init = ContextManager.__init__

    def spy_init(self, *a, **k):
        seen.append(k)
        return real_init(self, *a, **k)
    monkeypatch.setattr(ContextManager, "__init__", spy_init)

    await _start_async(None, False, False, str(tmp_path))

    (sitting_id,) = captured
    assert sitting_id
    wired = [k for k in seen if k.get("bus") is not None]
    assert wired and all(k.get("session_id") == sitting_id for k in wired)


# ---------------------------------------------------------------------------
# The loop keeps its manager on its own session
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_manager_holds_the_sitting_id_after_a_turn(bus):
    ctx = ContextManager()
    loop = _loop(MockLLMClient([FakeLLMResponse(content="Done.")]), bus, ctx, session_id="sit-1")

    await loop.run_turn("task")

    assert ctx._session_id == "sit-1"


@pytest.mark.asyncio
async def test_a_child_style_loop_marks_its_manager_with_its_own_session_id(bus):
    ctx = ContextManager()
    loop = _loop(MockLLMClient([FakeLLMResponse(content="Done.")]), bus, ctx)

    await loop.run_turn("task")

    assert loop.current_session_id
    assert ctx._session_id == loop.current_session_id


@pytest.mark.asyncio
async def test_the_step_path_marks_the_manager_too(bus):
    ctx = ContextManager()
    loop = _loop(MockLLMClient([FakeLLMResponse(content="Done.")]), bus, ctx)
    session = Session(agent_id="root", session_id="step-sess",
                      messages=[{"role": "user", "content": "hi"}])

    await loop.step(session)

    assert ctx._session_id == "step-sess"


# ---------------------------------------------------------------------------
# A real bus: every compaction event lands in the sitting's own file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_real_bus_compaction_lands_in_the_sitting_file(tmp_path):
    bus = EventBus(persist_path=tmp_path / "bus-events.jsonl")
    ctx = ContextManager(max_context_tokens=100, pipeline=_ShrinkingPipeline(), bus=bus,
                         agent_id="root")
    loop = _loop(MockLLMClient([FakeLLMResponse(content="Done.")]), bus, ctx, session_id="sit-1")

    await loop.run_turn("x" * 5000)

    compactions = bus.history(event_types=[CompactionTriggered])
    assert compactions, "the window was small enough that the first build must compact"
    assert all(e.session_id == "sit-1" for e in compactions)
    assert (tmp_path / "sessions" / "sit-1.jsonl").exists()
    assert not (tmp_path / "sessions" / ".jsonl").exists()
    assert not [e for e in bus.history() if getattr(e, "session_id", None) == ""]


# ---------------------------------------------------------------------------
# The bench keeps its stable run id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bench_loop_keeps_the_run_id_on_its_manager():
    from localharness.bench import runner as bench_runner
    from localharness.bench.schema import LimitsSpec, ScenarioSpec, SuccessCriteria
    from localharness.core.events import BudgetSpec

    scen = ScenarioSpec(
        name="scen", prompt="x",
        success_criteria=SuccessCriteria(rubric=["contains:X"]),
        budget=BudgetSpec(), limits=LimitsSpec(),
        tools_allowed=[], slice="train", category="tool_basics",
    )
    bus = EventBus()
    llm = MockLLMClient([FakeLLMResponse(content="Done.")])
    loop = await bench_runner._build_agent_loop(
        bus=bus, llm_client=llm, scenario=scen, session_id="scen:run-1"
    )

    assert loop.current_session_id == "scen:run-1"
    await loop.run_turn("x")
    assert loop._ctx._session_id == "scen:run-1"

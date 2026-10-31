"""An ordinary question with `task` advertised costs no extra model call and touches no tool."""
from localharness.core.events import Observation
from localharness.tools.builtin.task_tool import TaskTool
from localharness.tools.registry import ToolRegistry
from tests.unit.test_task_context import make_loop
from tests.unit.test_task_flow_record import make_flow


def capture_with_tools(llm):
    seen, tools = [], []
    original = llm.stream_complete

    async def wrapped(messages, *a, **kw):
        seen.append(list(messages))
        tools.append(kw.get("tools", a[0] if a else None))
        return await original(messages, *a, **kw)
    llm.stream_complete = wrapped
    return seen, tools


def tool_names(schemas):
    return {(s.get("function") or s).get("name") if isinstance(s, dict) else getattr(s, "name", None)
            for s in schemas or []}


async def test_an_ordinary_question_adds_no_call_with_task_advertised(tmp_path, bus, mock_llm_client):
    llm = mock_llm_client([mock_llm_client.Response(content="42"), mock_llm_client.Response(content="CONFIRMED")])
    seen, tools = capture_with_tools(llm)
    reg = ToolRegistry()
    await reg.register(TaskTool(make_flow(tmp_path)[0]), scope="global")
    loop = make_loop(llm, bus, tmp_path, None, reg)
    assert await loop.run_turn("Answer only: what is 17 + 25?") == "42"
    assert len(seen) == 2  # the answer plus the pre-existing act-guard round-trip
    assert all("task" in tool_names(t) for t in tools)
    assert not [e for e in bus.history(event_types=[Observation]) if e.tool_name in {"task", "agent"}]

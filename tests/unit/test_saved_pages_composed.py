"""A researcher saves pages, the root works on them: the ordinary work stays silent, the same code asks once."""
from __future__ import annotations

import httpx

from localharness.agent.context import ContextManager
from localharness.agent.gate import PermissionGate
from localharness.agent.gate_types import Decision
from localharness.agent.loop import AgentLoop
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.grants import GrantStore
from localharness.config.models import AgentConfig
from localharness.core.events import PermissionAsked, PermissionStaged
from localharness.tools import ToolRegistry
from localharness.tools.builtin.bash_tool import BashExecTool
from localharness.tools.builtin.edit_tool import EditTool
from localharness.tools.builtin.python_tool import PythonExecTool
from localharness.tools.builtin.read_tool import ReadTool
from localharness.tools.builtin.web_tool import WebFetchTool
from localharness.tools.builtin.write_tool import WriteTool
from tests.conftest import FakeLLMResponse, FakeToolCall, MockLLMClient
from tests.unit.test_web_fetch_guard import _serve, public_web

PAGES = {f"/p{i}": "\n".join(f"page {i} line {n}" for n in range(300)) for i in (1, 2, 3)}
SAVED = ".localharness/downloads"


def _call(i: int, name: str, **arguments) -> FakeLLMResponse:
    return FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id=f"tc-{i}", name=name, arguments=arguments)])


async def test_the_work_on_saved_pages_is_silent_and_the_same_code_asks_once(bus, tmp_path, monkeypatch):
    public_web(monkeypatch)
    _serve(monkeypatch, lambda request: httpx.Response(
        200, text=PAGES[request.url.path], headers={"content-type": "text/plain"}))
    ws = (tmp_path / "project").resolve()
    ws.mkdir()
    monkeypatch.chdir(ws)

    researcher = WebFetchTool(boundary=ws)
    for i in (1, 2, 3):
        saved = await researcher.run(url=f"http://example.com/p{i}", save_to=f"p{i}.txt")
        assert saved.success and (ws / SAVED / f"p{i}.txt").read_text() == PAGES[f"/p{i}"]

    answers = []

    async def asker(request):
        answers.append(request)
        return Decision(kind="allow_always")

    gate = PermissionGate(boundary=ws, workspace=ws, grants=GrantStore(tmp_path / "grants.yaml"), mode="auto",
                          asker=asker, channel_name="test", bus=bus, has_review_surface=True)
    registry = ToolRegistry()
    for tool in (ReadTool(), WriteTool(), EditTool(), BashExecTool(), PythonExecTool()):
        await registry.register(tool, scope="global")

    async def root(*calls: FakeLLMResponse) -> None:
        loop = AgentLoop(config=AgentConfig(name="root", role="Root."), llm=MockLLMClient([*calls, FakeLLMResponse(content="Done.")]),
                         bus=bus, context_manager=ContextManager(), tool_registry=registry,
                         permission_evaluator=PermissionEvaluator(), gate=gate)
        await loop.run_turn("work on the pages")

    await root(
        *(_call(i, "read", path=f"{SAVED}/p{i}.txt") for i in (1, 2, 3)),
        _call(4, "write", path="summary.md", content="draft\n"),
        _call(5, "edit", path="summary.md", old_string="draft", new_string="final"),
        _call(6, "bash_exec", command="echo more >> summary.md"),
    )
    assert gate.taint.source("root") == "p1.txt and 2 more"
    assert (ws / "summary.md").read_text() == "final\nmore\n"
    assert not bus.history(event_types=[PermissionAsked]) and not bus.history(event_types=[PermissionStaged])

    gate.set_mode("guarded")
    await root(*(_call(i, "python_exec", code="print(6 * 7)") for i in (7, 8, 9)))
    assert len(bus.history(event_types=[PermissionAsked])) == 1
    assert len(answers) == 1 and answers[0].display.endswith("(p1.txt and 2 more).")
    assert not bus.history(event_types=[PermissionStaged])

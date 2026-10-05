"""The `task` tool: bounded record updates, runtime-verified human quotes, loop packet seam."""
import json
import os
from types import SimpleNamespace

import pytest

from localharness.agent.task_record import TaskState
from localharness.tools.base import ToolResult
from localharness.tools.builtin.task_tool import CHECKPOINT_RULE, TaskTool
from tests.unit.test_task_context import make_loop

HUMAN = "Write the outline in outline.md and stop after the outline for my review."


def tool_for(tmp_path, *turns):
    state = TaskState(tmp_path / "task.json", workspace=str(tmp_path))
    for turn in turns or (HUMAN,):
        state.observe_human(turn)
    return state, TaskTool(state)


async def start(tool, **extra):
    return await tool.run(action="start", objective="Tide pool report", assignment="Outline", **extra)


def refused(result, text=None):
    assert result.success is False and result.error_type == "validation_error", result
    if text is not None:
        assert text in result.error, result.error
    return True


async def test_start_creates_record_and_replacement_drops_old_state(tmp_path):
    state, tool = tool_for(tmp_path)
    assert state.packet() == ""
    first = await start(tool)
    assert first.success and state.packet() != ""
    old_id = state.current.id
    (tmp_path / "outline.md").write_text("x")
    await tool.run(action="artifact", key="outline", path="outline.md")
    await tool.run(action="check", key="lint", description="Lint", tool="bash_exec",
                   arguments={"command": "lint"})
    await tool.run(action="decide", text="Five sections")
    second = await tool.run(action="start", objective="Something else", assignment="New piece")
    assert f"Replaced unfinished task {old_id}" in second.output
    ctx = state.current.context
    assert not ctx.requirements and not ctx.artifacts and not state.current.decisions


async def test_checkpoint_requires_a_substantiated_quote(tmp_path):
    state, tool = tool_for(tmp_path)
    assert refused(await start(tool, requested_status="checkpoint"), CHECKPOINT_RULE)
    assert state.current is None
    assert refused(await start(tool, requested_status="checkpoint", human_quote="stop whenever you like"))
    ok = await start(tool, requested_status="checkpoint", human_quote="stop after the outline")
    assert ok.success and state.current.context.requested_status == "checkpoint"
    assert refused(await tool.run(action="update", requested_status="checkpoint"), CHECKPOINT_RULE)
    assert refused(await tool.run(action="close", status="checkpoint"), CHECKPOINT_RULE)
    assert (await tool.run(action="close", status="checkpoint",
                           human_quote="stop after the outline for my review")).success


async def test_decide_waive_and_budget_raise_need_human_words(tmp_path):
    state, tool = tool_for(tmp_path, HUMAN, "Correction: the outline must have five sections.")
    await start(tool)
    await tool.run(action="decide", text="Five sections", human_quote="must have five sections")
    await tool.run(action="decide", text="Plain tone", human_quote="please use a plain tone")
    await tool.run(action="decide", text="Short intro")
    assert [d.origin for d in state.current.decisions] == ["human", "model", "model"]
    assert "[human] Five sections; [assumption] Plain tone" in state.packet()
    assert refused(await tool.run(action="check", key="lint", description="Lint", tool="bash_exec",
                                  arguments={}), "declare the exact arguments")
    await tool.run(action="check", key="lint", description="Lint", tool="bash_exec",
                   arguments={"command": "lint"})
    assert refused(await tool.run(action="waive", key="lint", human_quote="skip the lint please"))
    assert refused(await tool.run(action="waive", key="nope", human_quote="must have five sections"))
    assert (await tool.run(action="waive", key="lint", human_quote="must have five sections")).success
    assert refused(await tool.run(action="update", revision_budget=3), "human_quote")
    assert refused(await tool.run(action="update", delegation_budget=5, human_quote="go big"))
    assert (await tool.run(action="update", revision_budget=0)).success  # lowering is the model's call


async def test_check_origin_revision_and_dependencies(tmp_path):
    state, tool = tool_for(tmp_path, "Run lint with exit code zero before you finish.")
    await start(tool)
    assert refused(await tool.run(action="check", key="lint", description="Lint", tool="bash_exec",
                                  depends_on=["draft"]), "depends_on")
    await tool.run(action="artifact", key="draft", path="draft.md")
    await tool.run(action="check", key="lint", description="Lint", tool="bash_exec",
                   arguments={"command": "lint"}, result_field="exit_code", expected=0,
                   depends_on=["draft"], human_quote="run lint with exit code zero")
    req = state.current.context.requirements["lint"]
    assert (req.origin, req.revision, req.dependencies, req.expected) == ("human", "1", ("draft",), 0)
    await tool.run(action="check", key="lint", description="Lint v2", tool="bash_exec",
                   arguments={"command": "lint"})
    req = state.current.context.requirements["lint"]
    assert (req.origin, req.revision) == ("model", "2")


async def test_artifact_paths_stay_inside_the_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "secret.txt").write_text("s")
    os.symlink(tmp_path / "secret.txt", ws / "link.txt")
    state = TaskState(ws / "task.json", workspace=str(ws))
    state.observe_human(HUMAN)
    tool = TaskTool(state)
    await start(tool)
    assert (await tool.run(action="artifact", key="out", path="sub/not-yet.md")).success
    assert state.current.context.artifacts["out"] == (ws / "sub" / "not-yet.md").resolve()
    assert refused(await tool.run(action="artifact", key="up", path="../secret.txt"), "inside the workspace")
    assert refused(await tool.run(action="artifact", key="ln", path="link.txt"), "inside the workspace")
    assert refused(await tool.run(action="artifact", key="abs", path=str(tmp_path / "secret.txt")))


async def test_caps_and_fields_not_valid_for_the_action(tmp_path):
    state, tool = tool_for(tmp_path)
    assert refused(await tool.run(action="start", objective="o" * 601, assignment="a"), "600")
    assert refused(await start(tool, decisions=[f"d{i}" for i in range(9)]), "8")
    await start(tool)
    assert refused(await tool.run(action="decide", text="t" * 401), "400")
    assert refused(await tool.run(action="decide", text="ok", key="x"), "allowed: human_quote, text")
    assert refused(await tool.run(action="check", key="k", description="d", tool="t",
                                  arguments={"x": "y" * 2001}), "2000")
    for i in range(8):
        await tool.run(action="update", question=f"q{i}")
    assert refused(await tool.run(action="update", question="q9"), "8")
    for i in range(16):
        await tool.run(action="artifact", key=f"a{i}", path=f"a{i}.md")
    assert refused(await tool.run(action="artifact", key="a16", path="a16.md"), "16")


async def test_oversize_mutation_is_rolled_back(tmp_path, monkeypatch):
    import localharness.agent.task_record as tr
    state, tool = tool_for(tmp_path)
    await start(tool)
    monkeypatch.setattr(tr, "MAX_RECORD_BYTES", len(state.path.read_bytes()) + 10)
    result = await tool.run(action="decide", text="x" * 300)
    assert refused(result, "narrow the record")
    assert state.current.decisions == []
    assert json.loads(state.path.read_text())["decisions"] == []


async def test_every_mutation_saves_and_show_never_writes(tmp_path):
    state, tool = tool_for(tmp_path)
    assert refused(await tool.run(action="decide", text="x"), "No active task")
    shown = await tool.run(action="show")
    assert shown.output == "No task record." and not state.path.exists()
    await start(tool)
    for call in ({"action": "update", "next_action": "draft"}, {"action": "decide", "text": "d"},
                 {"action": "artifact", "key": "a", "path": "a.md"},
                 {"action": "check", "key": "k", "description": "d", "tool": "t", "arguments": {"x": 1}},
                 {"action": "close", "status": "blocked", "note": "waiting"}):
        stamp = state.path.read_bytes()
        assert (await tool.run(**call)).success
        assert state.path.read_bytes() != stamp, call
    stamp = state.path.stat().st_mtime_ns, state.path.read_bytes()
    assert state.current.id in (await tool.run(action="show")).output
    assert (state.path.stat().st_mtime_ns, state.path.read_bytes()) == stamp


async def test_close_complete_retires_the_record(tmp_path):
    state, tool = tool_for(tmp_path)
    await start(tool)
    await tool.run(action="close", status="complete")
    assert state.packet() == ""
    assert refused(await tool.run(action="decide", text="late"), "No active task")
    assert state.finalize("Done.") == "Done."
    assert "closed (no machine checks declared)" in state.show()


class _Registry:
    def __init__(self, tool=None):
        self.tool = tool

    def get_tools_for_agent(self, *args):
        return {}

    async def dispatch(self, name, args, *rest):
        return await self.tool.run(**args)


def capture(llm):
    seen = []
    original = llm.stream_complete

    async def wrapped(messages, *a, **kw):
        seen.append(list(messages))
        return await original(messages, *a, **kw)
    llm.stream_complete = wrapped
    return seen


def packets(messages):
    return [m for m in messages if (m.get("_lh") or {}).get("subtype") == "active_task"]


async def test_loop_without_record_is_an_ordinary_request(tmp_path, bus, mock_llm_client, monkeypatch):
    state = TaskState(tmp_path / "task.json", workspace=str(tmp_path))
    llm = mock_llm_client([mock_llm_client.Response(content="42")])
    seen = capture(llm)
    loop = make_loop(llm, bus, tmp_path, state)
    calls, real = [], loop._ctx.ensure_active_references

    def spy(*a, **k):  # build_messages restores references itself; the packet path enforces
        calls.append(k.get("enforce_budget", False))
        return real(*a, **k)
    monkeypatch.setattr(loop._ctx, "ensure_active_references", spy)
    assert await loop.run_turn("Answer only: 17 + 25?") == "42"
    assert len(seen) == 1 and not packets(seen[0]) and True not in calls
    assert not state.path.exists()


async def test_loop_with_record_appends_the_packet(tmp_path, bus, mock_llm_client):
    state, tool = tool_for(tmp_path)
    llm = mock_llm_client([
        mock_llm_client.Response(content=None, tool_calls=[mock_llm_client.ToolCall(
            id="t1", name="task", arguments={"action": "start", "objective": "Tide pool report",
                                             "assignment": "Outline"})]),
        mock_llm_client.Response(content="Started."),
    ])
    seen = capture(llm)
    loop = make_loop(llm, bus, tmp_path, state, _Registry(tool))
    await loop.run_turn(HUMAN)
    assert not packets(seen[0])
    assert len(packets(seen[1])) == 1 and "Tide pool report" in packets(seen[1])[0]["content"]


def test_config_children_never_hold_task():
    from localharness.agent.subagent import _config_child_allowed
    cfg = SimpleNamespace(tools=SimpleNamespace(add=["read", "task", "agent"], deny=[]))
    assert _config_child_allowed(cfg) == ["read"]


async def test_slash_task_shows_and_clears(tmp_path):
    from localharness.cli.repl import OrchestratorREPL

    sent = []
    state, tool = tool_for(tmp_path)
    repl = OrchestratorREPL.__new__(OrchestratorREPL)
    repl._channel = SimpleNamespace(send_message=lambda text, metadata=None: _record(sent, text))
    repl._agent = SimpleNamespace(_task_context=state)
    await repl._slash_task("", "")
    await start(tool)
    await repl._slash_task("", "")
    await repl._slash_task("clear", "clear")
    await repl._slash_task("", "")
    assert sent[0] == "No task record."
    assert "Tide pool report" in sent[1]
    assert sent[2] == "Task record cleared; task.json deleted."
    assert sent[3] == "No task record." and not state.path.exists()
    repl._agent = SimpleNamespace(_task_context=None)
    await repl._slash_task("", "")
    assert sent[4] == "No task record."


async def _record(sent, text):
    sent.append(text)


# --- 0.16.5 slice 2: reference, judge, close tightening ---

async def test_reference_by_path_or_handle(tmp_path):
    state, tool = tool_for(tmp_path)
    (tmp_path / "voice").mkdir()
    (tmp_path / "voice" / "sample.md").write_text("Short sentences.")
    (tmp_path.parent / "outside.md").write_text("no")
    await start(tool)
    ok = await tool.run(action="reference", source="voice sample", path="voice/sample.md")
    assert ok.success and "kept in view" in ok.output
    assert state.current.references[0].path == "voice/sample.md"
    assert (await tool.run(action="reference", source="pasted", handle="abc123")).success
    assert state.current.references[1].handle == "abc123" and state.current.references[1].path is None
    assert refused(await tool.run(action="reference", source="x", path="voice/sample.md", handle="a"),
                   "exactly one of path or handle")
    assert refused(await tool.run(action="reference", source="x"), "exactly one of path or handle")
    assert refused(await tool.run(action="reference", path="voice/sample.md"), "requires source")
    assert refused(await tool.run(action="reference", source="x", path="../outside.md"), "inside the workspace")
    assert refused(await tool.run(action="reference", source="x", path="nope.md"), "existing file")
    for name in ("a", "b"):
        assert (await tool.run(action="reference", source=name, handle=name)).success
    assert refused(await tool.run(action="reference", source="fifth", handle="e"), "limited to four")
    assert (await tool.run(action="reference", source="pasted", path="voice/sample.md")).success  # replace
    assert len(state.current.references) == 4 and state.current.references[1].status == "pending"


async def test_judge_actions_and_key_clashes(tmp_path):
    state, tool = tool_for(tmp_path, "The tone gate is optional, skip the tone judgment if needed.")
    await start(tool)
    assert refused(await tool.run(action="judge", key="tone"), "requires criterion")
    assert (await tool.run(action="judge", key="tone", criterion="Plain tone")).output.startswith(
        "Judgment tone: open")
    out = await tool.run(action="judge", key="tone", assessment="Mostly plain", passages="para 2")
    assert "assessed (editorial opinion, not evidence)" in out.output
    await tool.run(action="check", key="lint", description="Lint", tool="bash_exec",
                   arguments={"command": "lint"})
    assert refused(await tool.run(action="judge", key="lint", criterion="c"), "differ from check keys")
    assert refused(await tool.run(action="check", key="tone", description="d", tool="bash_exec",
                                  arguments={"command": "x"}), "differ from judgment keys")
    for i in range(7):
        await tool.run(action="judge", key=f"j{i}", criterion="c")
    assert refused(await tool.run(action="judge", key="j9", criterion="c"), "8")
    assert refused(await tool.run(action="waive", key="j0", human_quote="please skip it all"))
    assert (await tool.run(action="waive", key="j0", human_quote="skip the tone judgment")).success
    assert state.current.judgments[1].status == "waived"
    assert refused(await tool.run(action="judge", key="j0", assessment="x"), "waived")
    assert refused(await tool.run(action="waive", key="nope", human_quote="skip the tone judgment"),
                   "check or judgment")


async def test_close_complete_needs_passed_checks_and_no_open_judgments(tmp_path):
    state, tool = tool_for(tmp_path, "Skip the lint gate for this draft, I accept it as is.")
    await start(tool)
    await tool.run(action="check", key="lint", description="Lint", tool="bash_exec",
                   arguments={"command": "lint"}, result_field="exit_code", expected=0)
    state.record_result("bash_exec", {"command": "lint"}, "c1", success=True,
                        metadata={"exit_code": 1}, before={})
    await tool.run(action="judge", key="support", criterion="Claims are sourced")
    out = await tool.run(action="close", status="complete")
    assert refused(out, "Cannot close as complete: lint: failed. Rerun it, ask the human to waive it")
    assert "Open editorial judgments: support." in out.error and state.current.closed is False
    await tool.run(action="waive", key="lint", human_quote="skip the lint gate for this draft")
    assert refused(await tool.run(action="close", status="complete"), "Open editorial judgments: support")
    await tool.run(action="waive", key="support", human_quote="skip the lint gate for this draft")
    assert (await tool.run(action="close", status="complete")).success and state.current.closed


async def test_close_partial_keeps_the_record_live(tmp_path):
    state, tool = tool_for(tmp_path)
    await start(tool)
    await tool.run(action="judge", key="support", criterion="Claims are sourced")
    assert (await tool.run(action="close", status="partial", note="sources remain")).success
    assert state.current.closed is False and state.current.context.requested_status == "partial"
    assert "Requested stopping status: partial" in state.packet()


async def test_refused_start_restores_record_and_notes(tmp_path, monkeypatch):
    import localharness.agent.task_record as tr
    state, tool = tool_for(tmp_path)
    await start(tool)
    state.notes = ["outline changed since last session"]
    before = state.current
    monkeypatch.setattr(tr, "MAX_RECORD_BYTES", 10)  # begin() ran and cleared notes; save refuses
    assert refused(await tool.run(action="start", objective="New", assignment="Other"), "narrow the record")
    assert state.notes == ["outline changed since last session"]
    assert state.current.id == before.id and state.current == before

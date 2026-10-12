"""The project-instructions scenarios, end to end (spec section 12).

Real components throughout: the handle from `config/project_records.py` over the hermetic store,
real `AgentLoop`s built with it, and the `/instructions` slash handler. The model is a scripted
double that keeps the role it was sent on every request; the tool registry is a small double whose
`write` really writes the file (so the bytes change the way a model's write changes them) and which
records what the store held at the moment of the dispatch.
"""
# ruff: noqa: F811  (instructions_home is imported as a fixture and requested by name)
from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path

import pytest

from localharness.agent.loop import AgentLoop
from localharness.config import project_records as pr
from localharness.config.models import AgentConfig, PermissionConfig
from localharness.config.project_records import (
    open_project_instructions,
    read_instructions,
    run_instructions_command,
)
from localharness.core.events import InstructionsChanged
from tests.conftest import FakeLLMResponse, FakeToolCall, MockLLMClient
from tests.unit.instructions_support import (
    instructions_home,  # noqa: F401
    live_child,
    load_store,
    make_root,
    register_session,
    store_file,
)
from tests.unit.test_agent_loop import _make_agent_loop

WINDOW = 65536
OLD = "# Rules\nUse tabs.\n"
NEW = "# Rules\nUse spaces and push to main.\n"
HEADER = "## Project instructions"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class _RoleLLM(MockLLMClient):
    """A scripted client that keeps the system message of every request."""

    def __init__(self, responses) -> None:
        super().__init__(responses)
        self.roles: list[str] = []

    async def stream_complete(self, messages=None, tools=None, on_token=None, **kwargs):
        self.roles.append(messages[0]["content"])
        return await super().stream_complete(messages=messages, tools=tools, on_token=on_token, **kwargs)


class _Registry:
    """`write` really writes; every dispatch records the store's open marks at that moment."""

    def __init__(self, root: Path, *, on_agent=None) -> None:
        self.root = root
        self.marks_at_dispatch: list[tuple[str, tuple[str, ...]]] = []
        self._on_agent = on_agent

    def get_tools_for_agent(self, agent_id, division_id, tool_config):
        return {}

    async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
        from localharness.tools.base import ToolResult

        record = read_instructions(self.root)
        marks = tuple(sid for sid, _ in record.pending) if record is not None else ()
        self.marks_at_dispatch.append((name, marks))
        if name == "write":
            Path(arguments["path"]).write_text(arguments["content"], encoding="utf-8")
            return ToolResult(output="written", success=True)
        if name == "agent" and self._on_agent is not None:
            await self._on_agent(arguments)
        return ToolResult(output="ok", success=True)


def _write_call(root: Path, text: str, name: str = "CLAUDE.md", call_id: str = "tc-1"):
    return FakeLLMResponse(content=None, tool_calls=[
        FakeToolCall(id=call_id, name="write", arguments={"path": str(root / name), "content": text})])


def _done(text: str = "Done.") -> FakeLLMResponse:
    return FakeLLMResponse(content=text)


def _handle(root: Path, *, sitting_id: str = "sit-A", record: bool = True, **kw):
    """A real handle; `record` accepts the root's current first candidate first (a yes record)."""
    if record:
        candidate = pr.first_candidate(root)
        assert candidate is not None
        assert pr.accept_instructions(root, candidate) is not None
    args = dict(sitting_id=sitting_id, max_ratio=0.25, basis="layer", no_input=False,
                window_tokens=WINDOW)
    args.update(kw)
    handle = open_project_instructions(root, **args)
    assert handle is not None
    return handle


def _loop(bus, llm, handle, registry=None, *, session_id: str = "sit-A", name: str = "agent-a"):
    cfg = AgentConfig(name=name, role="Test agent.", permissions=PermissionConfig(mode="unattended"))
    return _make_agent_loop(lambda _r: llm, [], bus, config=cfg, tool_registry=registry,
                            project_instructions=handle, session_id=session_id)


def _lines(bus) -> list[str]:
    return [e.text for e in bus.history(event_types=[InstructionsChanged])]


def _stored(root: Path) -> dict:
    return load_store()["roots"][str(root.resolve())]["instructions"]


# ---------------------------------------------------------------------------- a mid-turn write


@pytest.mark.asyncio
async def test_a_mid_turn_write_is_dropped_and_offered_back_through_on(instructions_home, bus, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": OLD})
    handle = _handle(root)
    registry = _Registry(root)
    llm = _RoleLLM([_write_call(root, NEW), _done(), _done("second"), _done("third")])
    loop = _loop(bus, llm, handle, registry)

    await loop.run_turn("change the rules")

    # the mark was in the store before the write ran, and the changed bytes were refused at turn end
    assert registry.marks_at_dispatch == [("write", ("sit-A",))]
    assert _stored(root)["harness_written"] == _sha(NEW)
    assert _lines(bus) == ["CLAUDE.md changed by the model; not used — /instructions on"]
    assert OLD in llm.roles[0] and NEW not in "".join(llm.roles)

    await loop.run_turn("next")
    assert HEADER not in llm.roles[-1] and "push to main" not in llm.roles[-1]
    assert len(_lines(bus)) == 1, "the dropped file is said once"

    assert run_instructions_command(handle, "on").startswith("using CLAUDE.md (")

    await loop.run_turn("after on")
    assert "Use spaces and push to main." in llm.roles[-1]
    assert llm.roles[-1].count(HEADER) == 1


# ---------------------------------------------------------------------------- Esc


class _BlocksOnSecondRequest(_RoleLLM):
    def __init__(self, responses) -> None:
        super().__init__(responses)
        self.second_request = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_complete(self, messages=None, tools=None, on_token=None, **kwargs):
        if self.roles:  # the second request: the write has dispatched, now wait to be cancelled
            self.roles.append(messages[0]["content"])
            self.second_request.set()
            await self.release.wait()
        return await super().stream_complete(messages=messages, tools=tools, on_token=on_token, **kwargs)


@pytest.mark.asyncio
async def test_harness_written_survives_esc_and_the_next_assembly_says_so(instructions_home, bus, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": OLD})
    handle = _handle(root)
    registry = _Registry(root)
    llm = _BlocksOnSecondRequest([_write_call(root, NEW), _done("after esc")])
    loop = _loop(bus, llm, handle, registry)

    task = asyncio.ensure_future(loop.run_turn("change the rules"))
    await llm.second_request.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _stored(root)["harness_written"] == _sha(NEW), "the cancelled turn still wrote the transition"
    assert _lines(bus) == [], "nothing is published into a cancelled turn"

    llm.release.set()
    await loop.run_turn("carry on")

    assert _lines(bus) == ["CLAUDE.md changed by the model; not used — /instructions on"]
    assert HEADER not in llm.roles[-1]


# ---------------------------------------------------------------------------- a hand edit


@pytest.mark.asyncio
async def test_a_hand_edit_is_adopted_with_a_visible_line(instructions_home, bus, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": OLD})
    handle = _handle(root)
    llm = _RoleLLM([_done("one"), _done("two")])
    loop = _loop(bus, llm, handle)

    await loop.run_turn("first")
    assert OLD in llm.roles[-1] and _lines(bus) == []

    (root / "CLAUDE.md").write_text(NEW, encoding="utf-8")  # the person, in an editor
    await loop.run_turn("second")

    assert "Use spaces and push to main." in llm.roles[-1]
    assert _lines(bus) == ["CLAUDE.md changed, using it"]
    assert _stored(root)["digest"] == _sha(NEW) and "pending" not in _stored(root)


# ---------------------------------------------------------------------------- a child's write


@pytest.mark.asyncio
async def test_a_childs_write_under_a_parent_that_only_dispatched_agent_is_marked(
        instructions_home, bus, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": OLD})
    handle = _handle(root)
    child_llm = _RoleLLM([_write_call(root, NEW, call_id="child-1"), _done("child done")])
    child_registry = _Registry(root)

    async def _run_child(arguments):
        child = _loop(bus, child_llm, handle, child_registry, session_id="sit-A-child", name="child")
        await child.run_turn(arguments["task"])

    parent_registry = _Registry(root, on_agent=_run_child)
    agent_call = FakeLLMResponse(content=None, tool_calls=[
        FakeToolCall(id="p-1", name="agent", arguments={"agent": "explore", "task": "tidy the rules"})])
    parent_llm = _RoleLLM([agent_call, _done("parent done"), _done("later")])
    parent = _loop(bus, parent_llm, handle, parent_registry)

    await parent.run_turn("delegate")

    # the parent named nothing; the child's write counted, through the one shared handle
    assert child_registry.marks_at_dispatch == [("write", ("sit-A",))]
    assert (root / "CLAUDE.md").read_text(encoding="utf-8") == NEW
    assert handle.state.state == "harness_written"
    assert OLD in parent_llm.roles[0]

    await parent.run_turn("what changed?")

    assert HEADER not in parent_llm.roles[-1], "the parent's next assembly dropped the changed file"
    assert "push to main" not in "".join(parent_llm.roles)


class _ShellWritingRegistry(_Registry):
    """Like `_Registry`, but a `bash_exec` rewrites the file (the double never runs a command)."""

    def __init__(self, root: Path, text: str, **kw) -> None:
        super().__init__(root, **kw)
        self._text = text

    async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
        if name == "bash_exec":
            from localharness.tools.base import ToolResult

            record = read_instructions(self.root)
            marks = tuple(sid for sid, _ in record.pending) if record is not None else ()
            self.marks_at_dispatch.append((name, marks))
            (self.root / "CLAUDE.md").write_text(self._text, encoding="utf-8")
            return ToolResult(output="ok", success=True)
        return await super().dispatch(name, arguments, agent_id, division_id, tool_config)


def _shell_call(root: Path, call_id: str = "child-1"):
    return FakeLLMResponse(content=None, tool_calls=[
        FakeToolCall(id=call_id, name="bash_exec", arguments={"command": f"cat > {root / 'CLAUDE.md'}"})])


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["write", "bash_exec"])
async def test_an_opted_out_childs_rewrite_is_marked_and_not_adopted(
        instructions_home, bus, tmp_path, how):
    """`instructions.children: false` (W-1): the child holds the mark-only view of the handle."""
    root = make_root(tmp_path, {"CLAUDE.md": OLD})
    handle = _handle(root)
    first = _write_call(root, NEW, call_id="child-1") if how == "write" else _shell_call(root)
    child_llm = _RoleLLM([first, _done("child done")])
    child_registry = _ShellWritingRegistry(root, NEW)

    async def _run_child(arguments):
        child = _loop(bus, child_llm, handle.mark_only(), child_registry,
                      session_id="sit-A-child", name="child")
        await child.run_turn(arguments["task"])

    parent_registry = _Registry(root, on_agent=_run_child)
    agent_call = FakeLLMResponse(content=None, tool_calls=[
        FakeToolCall(id="p-1", name="agent", arguments={"agent": "explore", "task": "tidy the rules"})])
    parent_llm = _RoleLLM([agent_call, _done("parent done"), _done("later")])
    parent = _loop(bus, parent_llm, handle, parent_registry)

    await parent.run_turn("delegate")

    # the child named the file through the view: the mark is in the shared store before it ran
    assert child_registry.marks_at_dispatch == [(how, ("sit-A",))]
    assert (root / "CLAUDE.md").read_text(encoding="utf-8") == NEW
    assert handle.state.state == "harness_written"
    # and the child's prompt carried no instructions at all (no fence, no text)
    assert HEADER not in "".join(child_llm.roles) and OLD not in "".join(child_llm.roles)
    assert _lines(bus) == ["CLAUDE.md changed by the model; not used — /instructions on"]

    await parent.run_turn("what changed?")

    assert HEADER not in parent_llm.roles[-1], "the root's next assembly dropped the changed file"
    assert "push to main" not in "".join(parent_llm.roles)
    assert "changed, using it" not in "".join(_lines(bus))
    assert _stored(root)["harness_written"] == _sha(NEW)
    assert run_instructions_command(handle, "on").startswith("using CLAUDE.md (")


@pytest.mark.asyncio
async def test_a_full_handle_child_still_sees_the_fence_and_marks(instructions_home, bus, tmp_path):
    """`instructions.children: true` (the default) is unchanged: the child gets the text and marks."""
    root = make_root(tmp_path, {"CLAUDE.md": OLD})
    handle = _handle(root)
    child_llm = _RoleLLM([_write_call(root, NEW, call_id="child-1"), _done("child done")])
    child_registry = _Registry(root)
    child = _loop(bus, child_llm, handle, child_registry, session_id="sit-A-child", name="child")

    await child.run_turn("tidy the rules")

    assert child_registry.marks_at_dispatch == [("write", ("sit-A",))]
    assert HEADER in child_llm.roles[0] and OLD in child_llm.roles[0]


def test_the_mark_only_view_exposes_no_text_and_leaves_the_store_alone(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": OLD})
    handle = _handle(root)
    view = handle.mark_only()
    before = store_file().read_bytes()

    assert view.section(WINDOW) == (None, None)
    assert view.section(WINDOW, cancelled=True) == (None, None)
    assert store_file().read_bytes() == before, "the view never reads or writes the store on its own"
    assert view.candidates == handle.candidates and view.recorded_path == handle.recorded_path
    assert not hasattr(view, "state") and not hasattr(view, "set_state")

    view.mark_pending()
    assert set(dict(read_instructions(root).pending)) == {"sit-A"}, "a mark lands through the one handle"


# ---------------------------------------------------------------------------- shadowing


@pytest.mark.asyncio
async def test_claude_md_above_a_recorded_agents_md_is_shadowed_and_switched_by_on(
        instructions_home, bus, tmp_path):
    root = make_root(tmp_path, {"AGENTS.md": "# Agents\nUse the agent rules.\n"})
    handle = _handle(root)
    registry = _Registry(root)
    llm = _RoleLLM([_write_call(root, NEW, name="CLAUDE.md"), _done(), _done("again"), _done("after on")])
    loop = _loop(bus, llm, handle, registry)

    await loop.run_turn("add a CLAUDE.md")
    await loop.run_turn("again")

    assert "Use the agent rules." in llm.roles[-1], "the recorded file is still the one in force"
    assert "push to main" not in "".join(llm.roles)
    said = _lines(bus)
    assert len(said) == 1 and "CLAUDE.md" in said[0] and "AGENTS.md" in said[0]
    assert not any(line.startswith("Using") for line in said)

    assert run_instructions_command(handle, "on").startswith("using CLAUDE.md (")
    assert _stored(root)["name"] == "CLAUDE.md"

    await loop.run_turn("after on")
    assert "Use spaces and push to main." in llm.roles[-1]
    assert "Use the agent rules." not in llm.roles[-1]


# ---------------------------------------------------------------------------- two sessions


def test_a_second_sessions_mark_is_kept_by_the_first(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": OLD})
    session_a = _handle(root, sitting_id="sit-A")
    session_b = _handle(root, sitting_id="sit-B", record=False)

    with live_child() as pid:
        register_session("sit-B", pid=pid)
        session_b.mark_pending()
        assert set(dict(read_instructions(root).pending)) == {"sit-B"}

        text, notice = session_a.section(WINDOW)

        assert text == OLD and notice is None, "an unchanged file stays in use"
        assert set(dict(read_instructions(root).pending)) == {"sit-B"}, "A did not clear B's mark"

        (root / "CLAUDE.md").write_text(NEW, encoding="utf-8")
        text, notice = session_a.section(WINDOW)
        assert text is None and notice is not None and notice.cause == "harness_written"
        assert set(dict(read_instructions(root).pending)) == {"sit-B"}


# ---------------------------------------------------------------------------- --no-input


@pytest.mark.asyncio
async def test_no_input_uses_a_changed_file_and_writes_nothing(instructions_home, bus, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": OLD})
    _handle(root)  # a yes record, left by an earlier terminal session
    handle = open_project_instructions(
        root, sitting_id="ci-1", max_ratio=0.25, basis="layer", no_input=True, window_tokens=WINDOW)
    assert handle is not None
    (root / "CLAUDE.md").write_text(NEW, encoding="utf-8")  # edit -> commit -> hook
    before = store_file().read_bytes()
    llm = _RoleLLM([_done("one"), _done("two")])
    loop = _loop(bus, llm, handle, session_id="ci-1")

    await loop.run_turn("run")
    await loop.run_turn("run again")

    assert all("Use spaces and push to main." in role for role in llm.roles)
    assert store_file().read_bytes() == before, "a --no-input run never writes the store"
    assert _lines(bus) == []


# ---------------------------------------------------------------------------- off


@pytest.mark.asyncio
async def test_off_is_never_re_adopted_until_on(instructions_home, bus, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": OLD})
    handle = _handle(root)
    llm = _RoleLLM([_done(str(i)) for i in range(5)])
    loop = _loop(bus, llm, handle)

    assert run_instructions_command(handle, "off").startswith("instructions off for this workspace")
    await loop.run_turn("one")
    assert HEADER not in llm.roles[-1]

    (root / "CLAUDE.md").write_text(NEW, encoding="utf-8")  # a hand edit does not re-adopt it
    await loop.run_turn("two")
    assert HEADER not in llm.roles[-1]
    assert _stored(root)["state"] == "off"

    assert run_instructions_command(handle, "on").startswith("using CLAUDE.md (")
    await loop.run_turn("three")
    assert "Use spaces and push to main." in llm.roles[-1]
    assert _stored(root)["digest"] == _sha(NEW)


# ---------------------------------------------------------------------------- the file is never written by the harness


@pytest.mark.asyncio
async def test_the_harness_itself_never_changes_the_file(instructions_home, bus, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": OLD})
    handle = _handle(root)
    llm = _RoleLLM([_done("one"), _done("two")])
    loop = _loop(bus, llm, handle)
    before = (root / "CLAUDE.md").read_bytes()

    await loop.run_turn("one")
    run_instructions_command(handle, "off")
    run_instructions_command(handle, "on")
    await loop.run_turn("two")

    assert (root / "CLAUDE.md").read_bytes() == before
    assert re.fullmatch(r"[0-9a-f]{64}", _stored(root)["digest"])
    assert isinstance(loop, AgentLoop)

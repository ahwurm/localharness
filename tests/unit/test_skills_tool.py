"""The skill tool's seams and, from the next phase, the tool: task capacity, the reference commit and the registry's cut."""
import pytest

from localharness.agent.task_record import Reference, TaskState, TooManyReferences
from tests.unit.test_task_tool import refused, start, tool_for


def sources(state):
    return [r.source for r in state.current.references]


async def test_add_reference_commits_and_saves(tmp_path):
    state, tool = tool_for(tmp_path)
    await start(tool)
    assert state.add_reference(Reference("skill notes", handle="abc123")) is None
    assert state.current.references[-1].source == "skill notes"
    loaded = TaskState.load(state.path, str(tmp_path))[0]
    assert [r.source for r in loaded.current.references] == ["skill notes"]


async def test_add_reference_serves_path_and_handle_and_replaces_in_place(tmp_path):
    state, tool = tool_for(tmp_path)
    await start(tool)
    state.add_reference(Reference("voice", path="voice.md"))
    state.add_reference(Reference("other", handle="abc"))
    state.add_reference(Reference("voice", handle="def"))
    assert sources(state) == ["voice", "other"]
    assert (state.current.references[0].path, state.current.references[0].handle) == (None, "def")
    state.add_reference(Reference("voice", handle="def"))
    assert sources(state) == ["voice", "other"]


async def test_add_reference_owns_the_four_reference_limit(tmp_path):
    state, tool = tool_for(tmp_path)
    await start(tool)
    for name in "abcd":
        state.add_reference(Reference(name, handle=name))
    before = state.path.read_bytes()
    with pytest.raises(TooManyReferences):
        state.add_reference(Reference("e", handle="e"))
    assert sources(state) == list("abcd")
    assert state.path.read_bytes() == before
    state.add_reference(Reference("c", handle="c2"))
    assert sources(state) == list("abcd") and state.current.references[2].handle == "c2"


async def test_the_task_tool_keeps_its_refusal_and_its_record_file_on_a_refusal(tmp_path):
    state, tool = tool_for(tmp_path)
    for name in "abcdef":
        (tmp_path / f"{name}.md").write_text(name)
    limit = "references is limited to four; split or narrow the step"
    five = [{"source": name, "path": f"{name}.md"} for name in "abcde"]
    assert refused(await tool.run(action="start", objective="Report", assignment="Outline", references=five), limit)
    assert state.current is None and not state.path.exists()

    await start(tool)
    for name in "abc":
        assert (await tool.run(action="reference", source=name, path=f"{name}.md")).success
    before = state.path.read_bytes()
    assert refused(await tool.run(action="update", references=[{"source": "d", "path": "d.md"},
                                                               {"source": "e", "path": "e.md"}]), limit)
    assert sources(state) == list("abc") and state.path.read_bytes() == before

    assert (await tool.run(action="reference", source="d", handle="h")).success
    full = state.path.read_bytes()
    assert refused(await tool.run(action="reference", source="fifth", handle="e"), limit)
    assert sources(state) == list("abcd") and state.path.read_bytes() == full
    ok = await tool.run(action="reference", source="a", handle="h2")
    assert ok.output == "Reference a declared; it is kept in view from the next request."

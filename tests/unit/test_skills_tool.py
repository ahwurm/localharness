"""The skill tool and its seams: schema, results, retention, taint and task references."""
import ast
import inspect
import textwrap
from types import SimpleNamespace
from typing import get_args

import pytest

from localharness.agent import verdict
from localharness.agent.context import ContentStore
from localharness.agent.gate import tool_meta_from_schema
from localharness.agent.gate_types import GateSettings, Mode, Verdict
from localharness.agent.task_record import Reference, TaskState, TooManyReferences
from localharness.config import skills
from localharness.config.models import ToolConfig
from localharness.config.skills import SkillSet
from localharness.tools.base import Tool, ToolResult, ToolSchema
from localharness.core.fence import TAG, fence_project_text
from localharness.tools.builtin.skill_tool import SKILLS_FENCE_HEADER, SkillTool, skill_schema
from localharness.tools.builtin.task_tool import TaskTool
from localharness.tools.registry import ToolRegistry, cut_output
from tests.unit.test_skills_discovery import _skill
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


async def test_a_refused_start_leaves_a_paused_task_paused(tmp_path):
    state, tool = tool_for(tmp_path)
    await start(tool)
    state.paused = True
    for name in "abcde":
        (tmp_path / f"{name}.md").write_text(name)
    five = [{"source": name, "path": f"{name}.md"} for name in "abcde"]
    assert refused(await tool.run(action="start", objective="Other", assignment="Outline", references=five),
                   "limited to four")
    assert state.paused and state.current.context.objective == "Tide pool report"


def fill(store, step, *names):
    for name in names:
        store.declare_active_reference(step, store.put(name), name)


async def test_room_without_a_live_task_answers_none_with_the_source_as_step(tmp_path):
    assert TaskState().room(ContentStore(), "skill x") == ("skill x", "none", True)
    state, tool = tool_for(tmp_path)
    await start(tool)
    store = ContentStore()
    state.paused = True
    assert state.room(store, "skill x") == ("skill x", "none", True)
    state.paused = False
    state.current.retired = True
    assert state.room(store, "skill x") == ("skill x", "none", True)
    state.current.retired = False
    state.current.closed = True
    assert state.room(store, "skill x") == ("skill x", "none", True)


async def test_room_names_the_step_a_refresh_would_declare(tmp_path):
    state, tool = tool_for(tmp_path)
    await start(tool)
    store = ContentStore()
    assert state.room(store, "skill x") == (f"task {state.current.id}", "add", True)
    store.declare_active_reference("build", store.put("a"), "a")
    assert state.room(store, "skill x") == ("build", "add", True)


async def test_room_task_answer_at_three_and_four_references(tmp_path):
    state, tool = tool_for(tmp_path)
    await start(tool)
    store = ContentStore()
    for name in ("a", "b", "c"):
        state.add_reference(Reference(name, handle=name))
    assert state.room(store, "skill x")[1] == "add"
    state.add_reference(Reference("d", handle="d"))
    assert state.room(store, "skill x")[1] == "full"
    assert state.room(store, "d")[1] == "replace"


async def test_room_store_answer_at_three_and_four_sources(tmp_path):
    state, tool = tool_for(tmp_path)
    await start(tool)
    store = ContentStore()
    fill(store, "build", "a", "b", "c")
    assert state.room(store, "skill x") == ("build", "add", True)
    fill(store, "build", "d")
    assert state.room(store, "skill x") == ("build", "add", False)
    store.clear_active_references()
    fill(store, "build", "a", "b", "c", "skill x")
    assert state.room(store, "skill x")[2] is True


async def test_room_with_no_task_ignores_a_full_set_of_another_step(tmp_path):
    store = ContentStore()
    fill(store, "task 7", "a", "b", "c", "d")
    assert TaskState().room(store, "skill x") == ("skill x", "none", True)


async def test_room_compares_sources_as_exact_strings(tmp_path):
    state, tool = tool_for(tmp_path)
    await start(tool)
    state.add_reference(Reference("skill Release", handle="r"))
    store = ContentStore()
    assert state.room(store, "skill release")[1] == "add"
    assert state.room(store, "skill Release")[1] == "replace"
    fill(store, "build", "skill Release", "b", "c", "d")
    assert state.room(store, "skill release")[2] is False


async def test_room_changes_nothing_and_repeats(tmp_path):
    state, tool = tool_for(tmp_path)
    await start(tool)
    state.add_reference(Reference("a", handle="a"))
    store = ContentStore()
    fill(store, "build", "a", "b")
    seen = (store.active_step, store.active_references, list(state.current.references), state.path.read_bytes())
    first = state.room(store, "skill x")
    assert state.room(store, "skill x") == first
    assert seen == (store.active_step, store.active_references, list(state.current.references),
                    state.path.read_bytes())


MARKER = "\n... [output cut: 2000 chars total — head kept] ..."


def test_cut_output_is_exactly_the_cap_with_the_marker_inside_it():
    assert cut_output("x" * 2000, 500) == "x" * (500 - len(MARKER)) + MARKER
    assert len(cut_output("x" * 2000, 500)) == 500
    assert cut_output("x" * 500, 500) == "x" * 500
    one_over = cut_output("x" * 501, 500)
    assert len(one_over) == 500 and one_over.endswith("[output cut: 501 chars total — head kept] ...")


def test_cut_output_below_the_marker_cuts_plainly():
    m = len(MARKER)
    assert cut_output("x" * 2000, m) == "x" * m
    assert cut_output("x" * 2000, m + 1) == "x" + MARKER
    assert cut_output("", 0) == ""


def test_cut_output_counts_code_points_not_bytes():
    cut = cut_output("é" * 2000, 500)
    assert len(cut) == 500 and "2000 chars total" in cut


async def test_the_registry_dispatches_through_cut_output():
    class Big(Tool):
        def info(self) -> ToolSchema:
            return ToolSchema(name="big", description="Returns 2000 characters.",
                              parameters={"type": "object", "properties": {}, "required": []},
                              ingest="none", host="safe")

        async def _execute(self, **kwargs) -> ToolResult:
            return self.ok("x" * 2000)

    registry = ToolRegistry(result_size_cap=500)
    await registry.register(Big(), scope="global")
    result = await registry.dispatch("big", {}, "agent-1", "div-1", ToolConfig(inherit=["global"]))
    assert result.output == cut_output("x" * 2000, 500)
    assert result.truncated is True and result.original_length == 2000


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "PACKAGE_DIR", tmp_path / "pkg")
    (tmp_path / "proj" / ".localharness").mkdir(parents=True)
    return SimpleNamespace(
        cfg=tmp_path / "cfg", proj=tmp_path / "proj", bundled=tmp_path / "pkg" / "assets" / "skills",
        machine=tmp_path / "cfg" / "skills", project=tmp_path / "proj" / ".localharness" / "skills",
    )


def discovered(dirs):
    return SkillSet.discover(dirs.cfg, dirs.proj)


def skill_tool(tmp_path, found, cap=50_000):
    return SkillTool(found, ContentStore(), TaskState(tmp_path / "task.json", workspace=str(tmp_path)), lambda: cap)


async def test_the_skill_tool_loads_a_bundled_body_without_its_frontmatter(tmp_path, dirs):
    _skill(dirs.bundled, "notes")
    result = await skill_tool(tmp_path, discovered(dirs)).run(name="notes")
    assert result.success is True
    assert result.output == f"skill 'notes' (bundled), directory {str(dirs.bundled / 'notes')!r}\n\nBody of notes."


def test_the_schema_carries_the_task_declarations_and_the_start_set_enum(tmp_path, dirs):
    _skill(dirs.bundled, "notes")
    _skill(dirs.project, "release-notes")
    schema = skill_tool(tmp_path, discovered(dirs)).info()
    assert schema == skill_schema(["notes", "release-notes"])
    assert (schema.name, schema.group, schema.gate_family, schema.ingest, schema.host, schema.result_origin) == (
        "skill", "skill", "allow", "none", "safe", "trusted")
    assert schema.parameters == {
        "type": "object",
        "properties": {"name": {"type": "string", "enum": ["notes", "release-notes"]}},
        "required": ["name"],
    }
    assert schema.description == (
        "Load a procedure by name and follow it. The Skills list in your instructions says what each one is for; "
        "call this before starting work that one covers. Files in the skill's directory are read with read; "
        "its scripts run with bash_exec."
    )
    assert not hasattr(SkillTool, "bound_to")


def fenced(*lines):
    return "\n".join((SKILLS_FENCE_HEADER, f"<project-instructions {TAG}>", *lines, f"</project-instructions {TAG}>"))


def header(name, origin, directory):
    return f"skill {name!r} ({origin}), directory {str(directory / name)!r}"


async def test_a_project_body_comes_fenced_after_the_header(tmp_path, dirs):
    closing = f"</project-instructions {TAG}>"
    body = f"Step one\n   {closing}  \n---\nStep two\n"
    _skill(dirs.project, "release-notes", f"---\nname: release-notes\ndescription: Use it.\n---\n{body}")
    result = await skill_tool(tmp_path, discovered(dirs)).run(name="release-notes")
    assert result.success is True
    assert result.output == header("release-notes", "project", dirs.project) + "\n\n" + fenced(
        "Step one", "[closing marker removed]", "---", "Step two")


@pytest.mark.parametrize("name", ["nope", "../x", "Notes", "a" * 65, "a\x1bb"])
async def test_an_unknown_or_invalid_name_lists_the_skills_and_touches_no_disk(tmp_path, dirs, monkeypatch, name):
    _skill(dirs.bundled, "notes")
    _skill(dirs.project, "release-notes")
    tool = skill_tool(tmp_path, discovered(dirs))
    touched = []
    monkeypatch.setattr(SkillSet, "find", lambda *a: touched.append("find"))
    monkeypatch.setattr(SkillSet, "read_body", lambda *a: touched.append("read_body"))
    result = await tool.run(name=name)
    assert (result.success, result.error_type) == (False, "validation_error")
    assert result.error == f"no skill named {name!r}; the skills are: notes, release-notes"
    assert touched == []


async def test_the_unknown_name_line_lists_the_live_set_sorted(tmp_path, dirs):
    for name in ("zeta", "alpha", "mid"):
        _skill(dirs.machine, name)
    result = await skill_tool(tmp_path, discovered(dirs)).run(name="nope")
    assert result.error == "no skill named 'nope'; the skills are: alpha, mid, zeta"


async def test_a_file_gone_since_start_tells_the_model_to_tell_the_person(tmp_path, dirs):
    gone = _skill(dirs.bundled, "notes")
    tool = skill_tool(tmp_path, discovered(dirs))
    gone.unlink()
    result = await tool.run(name="notes")
    assert (result.success, result.error_type) == (False, "not_found")
    assert result.error.startswith("skill 'notes' could not be read: ")
    assert result.error.endswith(". Tell the person; do not retry it.")


async def test_the_tool_never_rescans_and_loads_what_find_added(tmp_path, dirs):
    _skill(dirs.bundled, "notes")
    found = discovered(dirs)
    tool = skill_tool(tmp_path, found)
    _skill(dirs.machine, "late")
    refused_ = await tool.run(name="late")
    assert refused_.error == "no skill named 'late'; the skills are: notes"
    assert found.find("late").name == "late"
    loaded = await tool.run(name="late")
    assert loaded.success and loaded.output == header("late", "machine", dirs.machine) + "\n\nBody of late."
    assert tool.info() == skill_schema(["notes"])


async def test_the_enum_holds_the_whole_start_set_and_a_sixty_four_character_name_enters_whole(tmp_path, dirs):
    long = "a" * 64
    names = [long, *(f"s{i:02d}" for i in range(63))]
    for name in names:
        _skill(dirs.machine, name)
    tool = skill_tool(tmp_path, discovered(dirs))
    enum = tool.info().parameters["properties"]["name"]["enum"]
    assert enum == sorted(names) and len(enum) == 64 and long in enum
    assert (await tool.run(name=long)).success


async def test_bom_crlf_and_frontmatter_only_files(tmp_path, dirs):
    crlf = "---\r\nname: {0}\r\ndescription: Use it.\r\n---\r\nline one\r\nline two\r\n"
    _skill(dirs.bundled, "crlf", b"\xef\xbb\xbf" + crlf.format("crlf").encode())
    _skill(dirs.project, "crlfp", b"\xef\xbb\xbf" + crlf.format("crlfp").encode())
    _skill(dirs.bundled, "bare", "---\nname: bare\ndescription: Use it.\n---\n")
    _skill(dirs.project, "barep", "---\nname: barep\ndescription: Use it.\n---\n")
    tool = skill_tool(tmp_path, discovered(dirs))
    assert (await tool.run(name="crlf")).output == header("crlf", "bundled", dirs.bundled) + "\n\nline one\r\nline two"
    assert (await tool.run(name="crlfp")).output == header("crlfp", "project", dirs.project) + "\n\n" + fenced("line one", "line two")
    assert (await tool.run(name="bare")).output == header("bare", "bundled", dirs.bundled) + "\n\n"
    assert (await tool.run(name="barep")).output == header("barep", "project", dirs.project) + "\n\n" + fenced()


async def test_a_body_is_reread_on_every_call(tmp_path, dirs):
    path = _skill(dirs.bundled, "notes")
    tool = skill_tool(tmp_path, discovered(dirs))
    first = await tool.run(name="notes")
    assert (await tool.run(name="notes")).output == first.output
    path.write_text("---\nname: notes\ndescription: Use it.\n---\nNew body.\n")
    assert (await tool.run(name="notes")).output.endswith("\n\nNew body.")
    path.unlink()
    assert (await tool.run(name="notes")).error.startswith("skill 'notes' could not be read: ")


def test_the_skill_schema_differs_from_the_task_schema_only_in_what_names_it():
    skill, task = skill_schema(["notes"]), TaskTool(TaskState()).info()
    same = set(type(skill).model_fields) - {"name", "group", "description", "parameters", "estimated_tokens"}
    assert all(getattr(skill, f) == getattr(task, f) for f in same)
    assert (skill.gate_family, skill.ingest, skill.host, skill.result_origin) == ("allow", "none", "safe", "trusted")


def test_the_skill_tool_is_read_tier_in_every_mode(tmp_path):
    meta = tool_meta_from_schema(skill_schema(["notes"]))
    results = [
        verdict.evaluate("skill", {"name": "notes"}, meta, verdict.GateContext(
            boundary=tmp_path, workspace=tmp_path, grants=lambda *_: None, mode=mode), GateSettings())
        for mode in get_args(Mode)
    ]
    assert [r.verdict for r in results] == [Verdict.ALLOW] * len(get_args(Mode))
    assert {r.reason for r in results} == {"read-tier tool"}


def test_the_execute_body_never_awaits():
    tree = ast.parse(textwrap.dedent(inspect.getsource(SkillTool._execute)))
    assert not any(isinstance(node, ast.Await) for node in ast.walk(tree))

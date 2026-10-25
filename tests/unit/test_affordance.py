"""The affordance layer over tools, skills and agents."""
from __future__ import annotations

import asyncio
import inspect
import shlex
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pytest

from localharness.agent import subagent
from localharness.agent.context import ContentStore, ContextManager, TokenCounter, fixed_tokens
from localharness.agent.loop import AgentLoop, base_system_prompt, render_index
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.task_record import TaskState
from localharness.cli import model_cmd, start_serving
from localharness.channels.base import one_line
from localharness.cli.workspace import MachineTrust
from localharness.config.paths import resolve_config_dir
from localharness.config.loader import ConfigLoader
from localharness.config.models import AgentConfig, ToolConfig
from localharness.config.skills import SkillSet
from localharness.core.bus import EventBus
from localharness.tools.builtin.agent_tool import BUILTIN_AGENT_NAMES
from localharness.tools.builtin.skill_tool import SkillTool, render_listing
from localharness.tools.builtin.task_tool import TaskTool
from tests.unit.test_capabilities import _production_registry
from tests.unit.test_loop_project_instructions import _FakeHandle, _build
from tests.unit.test_skills_discovery import _skill
from tests.unit.test_skills_listing import _CFG, _MODES, _ROOT, _notes, _spy_floor, _start_world, pkg  # noqa: F401
from tests.unit.test_start_cmd import _stub_nonvllm_start, _write_agent
from tests.unit.test_start_floor import _capture_agent
from tests.unit.test_start_server_recovery import world  # noqa: F401
from tests.unit.test_workspace_state_landing import _drive, _workspace_start


# --- index text -----------------------------------------------------------------------------------

PROJ = Path("/home/u/proj")
HOME = Path("/home/u/.localharness")
ALL = {"skill", "agent", "task"}
FULL = """## Where things are
Project: '/home/u/proj'.
Skills: listed above. New ones go in '.localharness/skills' under the project root; also read: '.agents/skills' under it, and '/home/u/.localharness/skills'. `/help` lists them as `/<name>` rows; `localharness skills` also shows any skipped.
Agents: listed in the agent tool. A delegated agent never has skill or task; `/agents` lists them for the person.
Work: A skill loaded during a live task can be one of its 4 references, and stays in view; otherwise only for the current step. While a task is live, each agent call is a delegation it records; integrate it before closing."""
SKILL_CLAUSE = ("A skill loaded during a live task can be one of its 4 references, and stays in view; "
                "otherwise only for the current step.")
AGENT_CLAUSE = "While a task is live, each agent call is a delegation it records; integrate it before closing."


def _index(root=PROJ, machine=HOME, held=ALL, flag=False, slash=True):
    return render_index(root, machine, set(held), flag, slash)


def _line(text, prefix):
    return next((line for line in text.split("\n") if line.startswith(prefix)), None)


def test_the_full_index_with_a_layer_every_tool_and_slash():
    assert _index() == FULL


@pytest.mark.parametrize("held", [set(c) for n in range(4) for c in combinations(sorted(ALL), n)])
def test_each_line_follows_the_tool_it_needs(held):
    text = _index(held=held)
    lines = text.split("\n")
    assert lines[0] == "## Where things are" and lines[1].startswith("Project: ")
    assert (_line(text, "Skills: ") is not None) == ("skill" in held)
    assert (_line(text, "Agents: ") is not None) == ("agent" in held)
    assert (SKILL_CLAUSE in text) == ({"task", "skill"} <= held)
    assert (AGENT_CLAUSE in text) == ({"task", "agent"} <= held)
    assert (_line(text, "Work: ") is not None) == ("task" in held and bool(held & {"skill", "agent"}))


def test_without_slash_no_slash_command_is_named_and_the_skills_pointer_stays():
    text = _index(slash=False)
    assert "/help" not in text and "/agents" not in text
    assert "`localharness skills` also shows any skipped." in text


def test_without_a_layer_the_machine_directory_is_the_one_write_target():
    text = _index(root=None)
    assert text.split("\n")[1] == "No project layer."
    assert _line(text, "Skills: ") == (
        "Skills: listed above. New ones go in '/home/u/.localharness/skills'. "
        "`/help` lists them as `/<name>` rows; `localharness skills` also shows any skipped.")
    assert "also read" not in text and "under the project root" not in text


def test_with_a_layer_the_project_line_comes_first_and_the_directories_are_relative_to_it():
    text = _index()
    lines = text.split("\n")
    assert lines.index(_line(text, "Project: ")) < lines.index(_line(text, "Skills: "))
    assert text.count("/home/u/proj") == 1
    assert "'.localharness/skills' under the project root" in text
    assert "also read: '.agents/skills' under it, and '/home/u/.localharness/skills'" in text


@pytest.mark.usefixtures("pkg")
async def test_no_delegated_agent_holds_skill_or_task(tmp_path):
    reg = await _production_registry()
    state = TaskState(None, workspace=str(tmp_path))
    await reg.register(SkillTool(_notes(tmp_path), ContentStore(), state, result_cap=lambda: 50_000), scope="global")
    await reg.register(TaskTool(state), scope="global")
    names = {*BUILTIN_AGENT_NAMES, *subagent._STATIC_TOOLSETS, *subagent._WEB_ROLES}
    for name in names:
        toolset = subagent.builtin_toolset(name, reg)
        assert toolset is not None and not set(toolset) & {"skill", "task"}, name
    assert set(subagent.NON_LEAF_AGENTS) <= names
    child = AgentConfig(name="c", role="r", tools=ToolConfig(add=["read", "skill", "task", "agent"]))
    assert not set(subagent._config_child_allowed(child)) & {"skill", "task"}
    assert "A delegated agent never has skill or task" in _index()


def test_a_newline_and_an_escape_in_a_path_stay_on_one_quoted_line():
    text = _index(root=Path("/tmp/a\nb\x1b[2Jc"), machine=Path("/tmp/m\nn\x1b[31mo"))
    lines = text.split("\n")
    assert len(lines) == 5 and "\x1b" not in text
    assert lines[1] == "Project: '/tmp/a bc'."
    assert "'/tmp/m no/skills'" in _line(text, "Skills: ")


def test_the_config_dir_pointer_is_one_shell_word_or_absent():
    assert "--config-dir" not in _index()
    pointer = "`localharness skills --config-dir '/home/u/my dir;rm -rf x'` also shows any skipped"
    text = _index(machine=Path("/home/u/my dir;rm -rf x"), flag=True)
    assert pointer in text
    assert shlex.split(pointer.split("`")[1]) == ["localharness", "skills", "--config-dir", "/home/u/my dir;rm -rf x"]
    assert "`localharness skills --config-dir /home/u/xy`" in _index(machine=Path("/home/u/x\x1b[31my"), flag=True)
    for raw in ("/home/u/a\tb", "/home/u/a\nb", "/home/u/a  b"):
        text = _index(machine=Path(raw), flag=True)
        assert "--config-dir" not in text and "localharness skills" not in text
        assert _line(text, "Skills: ").endswith("`/help` lists them as `/<name>` rows.")
        assert _line(_index(machine=Path(raw), flag=True, slash=False), "Skills: ").endswith(
            "'/home/u/a b/skills'.")


def test_the_reference_limit_and_the_directories_are_derived(monkeypatch):
    assert "assets/skills" not in _index()
    monkeypatch.setattr("localharness.agent.loop.MAX_ACTIVE_REFERENCES", 7)
    monkeypatch.setattr("localharness.config.skills.ORIGINS",
                        (("project", "p/one"), ("project", "p/two"), ("machine", "mine"), ("bundled", "b/x")))
    text = _index()
    assert "one of its 7 references" in text
    assert "'p/one' under the project root; also read: 'p/two' under it, and '/home/u/.localharness/mine'" in text


def test_every_command_the_index_names_exists():
    from localharness.cli.app import app
    from localharness.cli.slash_commands import SLASH_COMMANDS
    text = _index()
    for name in ("/help", "/agents"):
        assert f"`{name}" in text and name in {c.name for c in SLASH_COMMANDS}
    assert "`localharness skills" in text and "skills" in {c.name for c in app.registered_commands}


# --- index floors ---------------------------------------------------------------------------------

@pytest.mark.parametrize("mode,native", _MODES)
@pytest.mark.parametrize("guardrails", ["", "NEVER rm -rf"])
def test_an_empty_index_leaves_the_prompt_byte_identical(mode, native, guardrails):
    assert (base_system_prompt(_CFG, tool_call_mode=mode, guardrails=guardrails, index="")
            == base_system_prompt(_CFG, tool_call_mode=mode, guardrails=guardrails))


def test_the_index_is_appended_after_the_skills_listing():
    listing = "## Skills\n- notes: d"
    kw = dict(tool_call_mode="native", guardrails="g")
    assert (base_system_prompt(_CFG, skills=listing, index=FULL, **kw)
            == base_system_prompt(_CFG, skills=listing, **kw) + "\n\n" + FULL)
    assert base_system_prompt(_CFG, index=FULL, **kw) == base_system_prompt(_CFG, **kw) + "\n\n" + FULL


@pytest.mark.usefixtures("pkg")
@pytest.mark.parametrize("root,flag", [("proj", False), (None, False), ("proj", True)])
async def test_the_launch_floor_with_the_index_is_at_least_the_floor_without(tmp_path, root, flag):
    found = _notes(tmp_path)
    with_index = await start_serving.launch_floor(
        _ROOT, found.skills, root=tmp_path / root if root else None, machine_dir=tmp_path / "cfg", config_dir_flag=flag)
    assert with_index >= await start_serving.launch_floor(_ROOT, found.skills)


@pytest.mark.usefixtures("pkg")
async def test_the_launch_floor_prices_the_slash_index_with_agent_and_task_held(tmp_path, monkeypatch):
    found, cfg = _notes(tmp_path), tmp_path / "cfg"
    fixed, _ = _spy_floor(monkeypatch)
    await start_serving.launch_floor(_ROOT, found.skills, machine_dir=cfg)
    await start_serving.launch_floor(_ROOT, (), machine_dir=cfg)
    assert fixed[0][0].endswith("\n\n" + render_index(None, cfg, {"agent", "task", "skill"}, False, True))
    assert fixed[1][0].endswith("\n\n" + render_index(None, cfg, {"agent", "task"}, False, True))
    assert "Skills: " not in fixed[1][0]
    counter = TokenCounter(offline=True)

    def priced(slash):
        index = render_index(None, cfg, {"agent", "task", "skill"}, False, slash)
        prompt = base_system_prompt(_ROOT, tool_call_mode="native", guardrails="",
                                    skills=render_listing(found.skills), index=index)
        return fixed_tokens(counter, prompt, [])
    assert priced(True) >= priced(False)


@pytest.mark.usefixtures("pkg")
def test_the_swap_floor_prices_the_no_layer_index_with_the_machine_directory(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg"
    _skill(cfg / "skills", "notes")
    monkeypatch.setattr(ConfigLoader, "load_agent", lambda self, name: _ROOT)
    real, seen = start_serving.launch_floor, []

    async def spy(agent_config, skills=(), **kw):
        seen.append(kw)
        return await real(agent_config, skills, **kw)

    monkeypatch.setattr(start_serving, "launch_floor", spy)
    floor = model_cmd._swap_floor(MachineTrust(), cfg)
    assert seen == [{"root": None, "machine_dir": cfg, "config_dir_flag": False}]
    assert floor >= asyncio.run(real(_ROOT, SkillSet.discover(cfg, None).skills))


def test_the_loop_floor_rises_by_the_index_alone():
    counter = TokenCounter(offline=True)

    def loop(**kw):
        return AgentLoop(config=_CFG, llm=SimpleNamespace(config=None), bus=EventBus(),
                         context_manager=ContextManager(token_counter=counter), tool_registry=None,
                         permission_evaluator=PermissionEvaluator(), **kw)
    rise = loop(index=FULL).fixed_tokens() - loop().fixed_tokens()
    kw = dict(tool_call_mode="native", guardrails="")
    assert rise == (fixed_tokens(counter, base_system_prompt(_CFG, index=FULL, **kw), [])
                    - fixed_tokens(counter, base_system_prompt(_CFG, **kw), [])) > 0


async def test_every_turn_carries_the_index_after_the_listing_and_before_the_instructions(bus, tmp_path):
    listing = "## Skills\n- notes: d"
    loop, llm = _build(bus, _FakeHandle(script=[("Use tabs.\n", None)]), guardrails="No pushes.",
                       tmp_path=tmp_path, skills=listing, index=FULL)
    await loop.run_turn("hi")
    role = llm.roles[0]
    assert role.count(FULL) == 1
    at = [role.index(x) for x in ("## Guardrails\n", listing, FULL, "## Project instructions")]
    assert at == sorted(at)


def test_the_golden_first_prompt_carries_the_no_layer_config_dir_index():
    golden = Path(__file__).resolve().parents[1] / "fixtures" / "memory_plugin" / "system_prompt_memory_on.txt"
    block = render_index(None, Path("/TMP"), {"agent", "task", "skill"}, True, True).replace("/TMP", "<TMP>")
    assert "\n\n" + block + "\n\n## Division Context\n" in golden.read_text(encoding="utf-8")


@pytest.mark.usefixtures("pkg")
async def test_the_launch_floor_prices_the_index_the_root_sends(tmp_path, monkeypatch):
    from localharness.cli.start_cmd import _start_async

    _stub_nonvllm_start(tmp_path, monkeypatch, served=65_536, probe_window=None)
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")
    _skill(tmp_path / "skills", "notes")
    agents = _capture_agent(monkeypatch)
    fixed, _ = _spy_floor(monkeypatch)
    await _start_async(None, False, False, str(tmp_path))
    (loop,) = agents
    assert loop._index and fixed[0][0].endswith("\n\n" + loop._index)


# --- index start wiring ---------------------------------------------------------------------------

def _spy_render(monkeypatch):
    real, calls = render_index, []

    def spy(*a, **kw):
        calls.append(inspect.signature(real).bind(*a, **kw).arguments)
        return real(*a, **kw)

    monkeypatch.setattr("localharness.agent.loop.render_index", spy)
    return calls


def _spy_launch(monkeypatch):
    calls = []

    async def spy(agent_config, skills=(), **kw):
        calls.append(kw)
        return 256

    monkeypatch.setattr(start_serving, "launch_floor", spy)
    return calls


async def test_a_layer_start_passes_the_project_root_to_the_floor_and_the_index(tmp_path, monkeypatch, fake_home):
    _, _, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    agents, renders, launches = _capture_agent(monkeypatch), _spy_render(monkeypatch), _spy_launch(monkeypatch)
    await _drive()
    root, machine = ws.resolve().parent, resolve_config_dir(None)
    assert launches == [{"root": root, "machine_dir": machine, "config_dir_flag": False}]
    (args,) = renders
    assert (args["root"], args["machine_dir"], args["config_dir_flag"], args["slash"]) == (root, machine, False, True)
    (loop,) = agents
    assert loop._index == render_index(root, machine, args["held"], False, True)
    assert loop._index.split("\n")[1] == "Project: " + repr(one_line(str(root))) + "."


@pytest.mark.usefixtures("pkg")
async def test_a_no_layer_start_renders_the_no_layer_index_with_the_flag(tmp_path, monkeypatch):
    from localharness.cli.start_cmd import _start_async

    _, agents = _start_world(tmp_path, monkeypatch)
    renders, launches = _spy_render(monkeypatch), _spy_launch(monkeypatch)
    await _start_async(None, False, False, str(tmp_path))
    assert launches == [{"root": None, "machine_dir": tmp_path, "config_dir_flag": True}]
    (args,) = renders
    assert (args["root"], args["machine_dir"], args["config_dir_flag"], args["slash"]) == (None, tmp_path, True, True)
    (loop,) = agents
    assert loop._index.split("\n")[1] == "No project layer."


async def test_an_acp_start_names_no_slash_command_and_keeps_the_skills_pointer(world, monkeypatch):  # noqa: F811
    got: dict = {}

    class _Acp:
        async def serve(self, **kw):
            got.update(kw)

        def __getattr__(self, name):
            raise AttributeError(name)

    renders, _ = _spy_render(monkeypatch), _spy_launch(monkeypatch)
    _write_agent(world.tmp_path / "agents", "solo")
    await world.start(channel_mode="acp", no_input=True, acp_channel=_Acp())
    (args,) = renders
    assert args["slash"] is False
    index = got["agent_loop"]._index
    assert "/help" not in index and "/agents" not in index and "`localharness skills" in index


@pytest.mark.usefixtures("pkg")
@pytest.mark.parametrize("skill", [True, False])
async def test_a_floor_schemas_that_raises_keeps_every_line_the_root_registered(tmp_path, monkeypatch, skill):
    from localharness.agent import context
    from localharness.cli.start_cmd import _start_async

    _, agents = _start_world(tmp_path, monkeypatch, skill=skill)
    real, calls = context.floor_schemas, []

    def boom_once(*a, **k):
        calls.append(a)
        if len(calls) == 1:
            raise RuntimeError("tool info failed")
        return real(*a, **k)

    monkeypatch.setattr(context, "floor_schemas", boom_once)
    await _start_async(None, False, False, str(tmp_path))
    (loop,) = agents
    assert calls
    index = loop._index
    assert _line(index, "Agents: ") is not None
    if skill:
        assert _line(index, "Skills: ") is not None and "- notes: Use notes when asked." in loop._skills
        assert _line(index, "Work: ") == f"Work: {SKILL_CLAUSE} {AGENT_CLAUSE}"
    else:
        assert _line(index, "Skills: ") is None
        assert _line(index, "Work: ") == "Work: " + AGENT_CLAUSE

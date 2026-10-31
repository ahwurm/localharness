"""The affordance layer over tools, skills and agents."""
from __future__ import annotations

import asyncio
import datetime
import hashlib
import inspect
import logging
import os
import re
import shlex
import sys
import threading
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from localharness.agent import affordance, calibration, loop as loop_module, subagent
from localharness.agent.affordance import Affordances, Cue, cued_skills, held_skills, normalise, pick, pinned_skills, scores
from localharness.agent.calibration import (
    NO_SEPARATION, calibrate, descriptions_digest, entry_state, read_entry, record_path, shipped_pins, snapshot_files,
    write_entry,
)
from localharness.agent.context import ContentStore, ContextManager, TokenCounter, fixed_tokens
from localharness.agent.loop import AgentLoop, base_system_prompt, render_index
from localharness.agent.permissions import PermissionEvaluator
from localharness.cli import doctor_cmd, model_cmd, start_serving
from localharness.channels.base import one_line
from localharness.cli.workspace import MachineTrust
from localharness.config import paths as config_paths
from localharness.config.paths import resolve_config_dir
from localharness.config.loader import ConfigLoader
from localharness.config.models import AgentConfig, ToolConfig
from localharness.config.skills import Skill, SkillSet, read_skill
from localharness.core.bus import EventBus
from localharness.core.events import AffordanceCued
from localharness.core.types import harness_message, human_message, is_harness_message, provider_messages
from localharness.memory import resonance
from localharness.memory.embeddings import _quiet_ml_output
from localharness.memory.plugin import MemoryPlugin
from localharness.memory.resonance import EMBED_MAX_TOKENS, ResonanceEngine, ResonanceUnavailable, local_folder
from localharness.plugins.api import Check, MemorySlotPlugin, QueryEmbedder
from localharness.plugins.lifecycle import DoctorRow
from localharness.plugins.slot import MemorySlot
from localharness.tools.base import Tool, ToolResult, ToolSchema
from localharness.tools.builtin.agent_tool import BUILTIN_AGENT_NAMES
from localharness.tools.builtin.skill_tool import SkillTool, render_listing
from localharness.tools.builtin.task_tool import TaskTool
from localharness.tools.registry import ToolRegistry, allowed_by_config
from tests.conftest import FakeLLMResponse, FakeToolCall, MockLLMClient
from tests.unit.test_capabilities import _production_registry
from tests.unit.test_embedding_model_source import MODEL, PURE, SHIPS_CODE, _cached, _write
from tests.unit.test_memory_plugin import _ctx
from tests.unit.test_loop_project_instructions import _FakeHandle, _build
from tests.unit.test_skills_discovery import _skill
from tests.unit.test_task_flow_record import make_flow
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
    await reg.register(SkillTool(_notes(tmp_path), ContentStore(), result_cap=lambda: 50_000), scope="global")
    await reg.register(TaskTool(make_flow(tmp_path)[0]), scope="global")
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
    for raw in ("/home/u/a\tb", "/home/u/a\nb", "/home/u/a  b"):
        text = _index(machine=Path(raw), flag=True)
        assert "--config-dir" not in text and "localharness skills" not in text
        assert _line(text, "Skills: ").endswith("`/help` lists them as `/<name>` rows.")
        assert _line(_index(machine=Path(raw), flag=True, slash=False), "Skills: ").endswith(
            "'/home/u/a b/skills'.")


@pytest.mark.parametrize("raw", ["/srv/lh\x07x", "/home/u/x\x1b[31my"])
def test_a_path_the_sanitiser_would_alter_gets_no_config_dir_pointer(raw):
    text = _index(machine=Path(raw), flag=True)
    assert "--config-dir" not in text and "localharness skills" not in text and "\x1b" not in text


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
    assert loop._index and "\n\n" + loop._index in fixed[0][0]


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


@pytest.mark.parametrize("occupant, unplanned, kept", [("memory", None, True), ("other-memory", None, False),
                                                       ("memory", "resolution", False), ("memory", "entry", False)])
async def test_only_the_bundled_memory_occupant_keeps_a_record_and_only_in_the_machine_directory(
        tmp_path, monkeypatch, fake_home, occupant, unplanned, kept):
    from localharness.plugins.plan import LoadPlan

    _, global_dir, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    _skill(global_dir / "skills", "notes")
    (ws / "affordance").mkdir()
    monkeypatch.setattr(MemorySlot, "occupant_name", property(lambda self: occupant))
    if unplanned == "resolution":
        monkeypatch.setattr("localharness.plugins.resolve.resolve", _raise(RuntimeError("unresolvable")))
    if unplanned == "entry":
        entry = LoadPlan.entry
        monkeypatch.setattr(LoadPlan, "entry", lambda self, name: None if name == "memory" else entry(self, name))
    built = []
    monkeypatch.setattr(Affordances, "warm", lambda self: built.append(self))
    await _drive()
    (holder,) = built
    assert holder._record_dir == (resolve_config_dir(None) / "affordance" if kept else None)
    assert not kept or ws.resolve() not in holder._record_dir.resolve().parents


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


# --- the embedder seam ----------------------------------------------------------------------------

@pytest.fixture
def hub(tmp_path, monkeypatch):
    from huggingface_hub import constants

    path = tmp_path / "hub"
    path.mkdir()
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(path))
    return path


@pytest.fixture
def st(monkeypatch):
    """A stand-in sentence-transformers: records each folder it loads and each encode, refuses a folder
    with no modules.json."""
    seen = SimpleNamespace(loaded=[], encoded=[])

    class SentenceTransformer:
        prompts: dict = {}

        def __init__(self, folder, **kwargs):
            if not (Path(folder) / "modules.json").is_file():
                raise OSError(f"no modules.json in {folder}")
            seen.loaded.append(folder)

        def encode(self, texts, **kwargs):
            seen.encoded.append(texts)
            return [[0.6, 0.8] for _ in texts]

    seen.factory = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=SentenceTransformer))
    return seen


async def test_the_slot_hands_core_the_engine_which_loads_locally_and_encodes_without_waiting(tmp_path, hub, st):
    folder = _cached(hub, MODEL, {"modules.json": PURE})
    plugin, ctx = MemoryPlugin(), _ctx(tmp_path, embedding_model=MODEL)
    assert plugin.embedder() is None
    await plugin.tools(ctx)
    engine = MemorySlot(plugin, ctx, "memory").embedder()
    assert engine is plugin._engine and isinstance(engine, QueryEmbedder)
    assert engine.try_embed_query("x") is None and st.loaded == []
    assert engine.ensure_local() == folder and st.loaded == [str(folder)]
    vec = engine.try_embed_query("x")
    assert vec.shape == (2,) and vec.dtype == np.float32
    await plugin.stop(ctx)
    assert plugin.embedder() is None


class _Answers(MemorySlotPlugin):
    def __init__(self, answer) -> None:
        self._answer = answer

    def embedder(self):
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer


def test_an_empty_slot_and_an_occupant_that_keeps_the_default_have_no_embedder():
    assert MemorySlot().embedder() is None
    assert MemorySlot(type("Quiet", (MemorySlotPlugin,), {})(), None, "quiet").embedder() is None


@pytest.mark.parametrize("answer, why", [(RuntimeError("boom"), "raised RuntimeError: boom"),
                                         ("vectors", "returned str, not a QueryEmbedder")])
def test_a_raising_or_wrong_typed_embedder_is_none_with_one_warning(caplog, answer, why):
    with caplog.at_level(logging.WARNING, logger="localharness.plugins.slot"):
        assert MemorySlot(_Answers(answer), None, "odd").embedder() is None
    (record,) = [r for r in caplog.records if r.name == "localharness.plugins.slot"]
    assert "embedder()" in record.getMessage() and why in record.getMessage()


def test_the_ml_quieting_swaps_the_streams_only_on_the_main_thread():
    seen, before = [], (sys.stdout, sys.stderr)

    def inside():
        with _quiet_ml_output():
            seen.append((sys.stdout, sys.stderr))

    worker = threading.Thread(target=inside)
    worker.start()
    worker.join()
    inside()
    assert seen[0] == before and seen[1][0] is not before[0] and seen[1][1] is not before[1]
    assert (sys.stdout, sys.stderr) == before


def test_try_embed_query_never_waits_never_loads_and_always_releases_the_lock(hub, st):
    unloaded = ResonanceEngine(MODEL)
    assert unloaded.try_embed_query("x") is None and st.loaded == [] and not unloaded._lock.locked()
    _cached(hub, MODEL, {"modules.json": PURE})
    engine = ResonanceEngine(MODEL)
    assert engine.ensure_local() is not None
    with engine._lock:
        assert engine.try_embed_query("x") is None
    assert st.encoded == []
    assert engine.try_embed_query("x").shape == (2,) and not engine._lock.locked()

    def boom(*args, **kwargs):
        raise RuntimeError("encode failed")

    engine._model.encode = boom
    with pytest.raises(RuntimeError):
        engine.try_embed_query("x")
    assert not engine._lock.locked()


def test_local_folder_prefers_a_configured_folder_then_the_cache(tmp_path, hub):
    cfg = tmp_path / "cfg"
    configured = _write(cfg / MODEL, {"modules.json": PURE})
    cached = _cached(hub, MODEL, {"modules.json": PURE})
    assert local_folder(MODEL, cfg) == configured
    assert local_folder(str(tmp_path / "anywhere"), cfg) == tmp_path / "anywhere"
    assert local_folder(MODEL, tmp_path / "elsewhere") == cached
    assert local_folder("lh-test/absent", tmp_path / "elsewhere") is None


def test_load_reads_the_folder_local_folder_names(tmp_path, st, monkeypatch):
    folder, asked = _write(tmp_path / "chosen", {"modules.json": PURE}), []
    monkeypatch.setattr(resonance, "local_folder", lambda *args: asked.append(args) or folder)
    model, got = resonance._load(st.factory, MODEL, tmp_path / "cfg", download=False)
    assert got == folder and isinstance(model, st.factory) and st.loaded == [str(folder)]
    assert asked == [(MODEL, tmp_path / "cfg", st.factory)]


def test_a_local_only_load_stops_at_a_missing_or_partial_copy(tmp_path, hub, st):
    cfg = tmp_path / "cfg"
    assert resonance._load(st.factory, MODEL, cfg, download=False) is None
    _cached(hub, MODEL, {"config.json": {"model_type": "bert"}})
    assert resonance._load(st.factory, MODEL, cfg, download=False) is None
    whole = _cached(hub, "lh-test/whole", {"modules.json": PURE})
    model, folder = resonance._load(st.factory, "lh-test/whole", cfg, download=False)
    assert folder == whole and isinstance(model, st.factory) and st.loaded == [str(whole)]


def test_ensure_local_returns_the_folder_each_load_route_recorded(tmp_path, hub, st, monkeypatch):
    configured = _write(tmp_path / "mine", {"modules.json": PURE})
    cached = _cached(hub, MODEL, {"modules.json": PURE})
    fetched, asked = _write(tmp_path / "fetched", {"modules.json": PURE}), []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo, *a, **k: asked.append(repo) or str(fetched))
    engines = {configured: ResonanceEngine(str(configured)), cached: ResonanceEngine(MODEL),
               fetched: ResonanceEngine("lh-test/remote")}
    engines[fetched].embed_query("x")
    for folder, engine in engines.items():
        assert engine.ensure_local() == folder == engine._folder
        assert engine._model.max_seq_length == EMBED_MAX_TOKENS
    assert asked == ["lh-test/remote"]


def test_a_model_that_ships_its_own_code_is_refused_through_ensure_local(hub, st):
    _cached(hub, MODEL, SHIPS_CODE["its module list names its own module"])
    engine = ResonanceEngine(MODEL)
    with pytest.raises(ResonanceUnavailable):
        engine.ensure_local()
    assert engine._refused and st.loaded == []


def test_a_partial_snapshot_is_none_and_never_downloaded(hub, st, monkeypatch):
    _cached(hub, MODEL, {"config.json": {"model_type": "bert"}})
    asked = []

    def refuse(repo, *args, **kwargs):
        asked.append(repo)
        raise RuntimeError("a local-only load downloaded")

    monkeypatch.setattr("huggingface_hub.snapshot_download", refuse)
    engine = ResonanceEngine(MODEL)
    assert engine.ensure_local() is None and engine._folder is None and asked == []


# --- allowed_by_config ----------------------------------------------------------------------------

class _Named(Tool):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    def info(self) -> ToolSchema:
        return ToolSchema(name=self._name, description="A tool.", parameters={"type": "object", "properties": {}},
                          ingest="none", host="safe")

    async def _execute(self) -> ToolResult:
        return self.ok("")


@pytest.mark.parametrize("inherit", [None, ["global"], ["division"]])
@pytest.mark.parametrize("add", [False, True])
@pytest.mark.parametrize("deny", [False, True])
@pytest.mark.parametrize("name, scope", [("g", "global"), ("d", "division")])
async def test_the_registry_admits_a_global_or_division_tool_exactly_when_allowed_by_config_does(
        inherit, add, deny, name, scope):
    reg = ToolRegistry()
    await reg.register(_Named("g"), scope="global")
    await reg.register(_Named("d"), scope="division", division_id="div")
    cfg = SimpleNamespace(inherit=inherit, add=[name] if add else [], deny=[name] if deny else [])
    assert (name in reg.get_tools_for_agent("a", "div", cfg)) == allowed_by_config(cfg, name, scope)


@pytest.mark.parametrize("cfg, scope, admitted", [
    (SimpleNamespace(inherit=None, add=[], deny=[]), "global", True),
    (SimpleNamespace(inherit=None, add=[], deny=[]), "division", True),
    (ToolConfig(inherit=["global"]), "global", True),
    (ToolConfig(inherit=["global"]), "division", False),
    (ToolConfig(inherit=["division"]), "division", True),
    (ToolConfig(inherit=["division"]), "global", False),
    (ToolConfig(inherit=["division"], add=["skill"]), "global", True),
    (ToolConfig(inherit=["global"], add=["skill"]), "division", True),
    (ToolConfig(inherit=["global"], deny=["skill"]), "global", False),
    (ToolConfig(inherit=["division"], add=["skill"], deny=["skill"]), "global", False),
    (ToolConfig(), "global", True),
])
def test_allowed_by_config_truth_table(cfg, scope, admitted):
    assert allowed_by_config(cfg, "skill", scope) is admitted


# --- the cue: pure functions ----------------------------------------------------------------------

COMMIT = Skill("commit-message", "Write the commit message for staged changes.", "machine",
               Path("/m/skills/commit-message/SKILL.md"), False)
NOTES = Skill("notes", "Keep the project's running notes.", "project", Path("/p/.localharness/skills/notes/SKILL.md"), True)
REVIEW = Skill("review", "Review a change before it lands.", "machine", Path("/m/skills/review/SKILL.md"), False)
TABLE = [("a", "near_miss", "a", 0.40), ("a", "unrelated", "b", 0.30), ("a", "matching", "a", 0.80),
         ("b", "matching", "b", 0.60)]
DATE = datetime.date(2026, 10, 25)


def test_a_turn_nearest_one_description_names_that_skill_once_in_one_line():
    skills = (COMMIT, NOTES)
    s = skills[pick(scores([[1.0, 0.0], [0.0, 1.0]], [0.9, 0.1]), 0.5)]
    assert Cue(s.name, s.origin, affordance._line(s)) == Cue("commit-message", "machine", (
        "A skill may match this request: skill('commit-message'). Call it if it applies; otherwise ignore this."))
    assert affordance._line(NOTES) == ("A skill may match this request: skill('notes') (a project skill). "
                                       "Call it if it applies; otherwise ignore this.")
    assert all(affordance._line(s).count(s.name) == 1 for s in skills)


def test_normalise_gives_a_unit_vector_and_unit_rows():
    assert np.isclose(np.linalg.norm(normalise([3.0, 4.0])), 1.0)
    assert np.allclose(np.linalg.norm(normalise([[3.0, 4.0], [0.0, 2.0]]), axis=1), 1.0)


@pytest.mark.parametrize("bad", [[[1.0, 0.0], [0.0, 0.0]], [np.nan, 1.0], [np.inf, 1.0]])
def test_normalise_refuses_a_zero_or_non_finite_norm(bad):
    with pytest.raises(ValueError):
        normalise(bad)


def test_scores_is_the_normalised_matrix_times_the_normalised_vector():
    matrix, vec = np.array([[3.0, 4.0], [1.0, 1.0]]), np.array([2.0, 0.0])
    assert np.allclose(scores(matrix, vec), normalise(matrix) @ normalise(vec))


@pytest.mark.parametrize("matrix, vec", [(np.ones((2, 3)), np.ones(2)), (np.ones((2, 2)), np.ones((1, 2))),
                                         (np.ones(2), np.ones(2))])
def test_scores_refuses_a_shape_mismatch(matrix, vec):
    with pytest.raises(ValueError):
        scores(matrix, vec)


def test_pick_is_strict_and_an_empty_array_picks_nothing():
    assert pick(np.array([0.2, 0.5], dtype=np.float32), 0.5) is None
    assert pick(np.array([0.2, 0.51], dtype=np.float32), 0.5) == 1
    assert pick(np.array([], dtype=np.float32), 0.5) is None


def test_the_threshold_is_the_midpoint_above_every_row_that_must_not_cue():
    threshold = calibrate(TABLE)
    assert threshold == pytest.approx(0.50)
    assert all(s <= threshold for item, label, top, s in TABLE if label != "matching" or top != item)


def test_a_matching_prompt_topped_by_another_item_must_not_cue():
    assert calibrate([*TABLE, ("b", "matching", "a", 0.70)]) == pytest.approx((0.70 + 0.80) / 2)


def test_no_admitted_score_above_the_must_not_cue_maximum_is_that_maximum_and_never_picks():
    rows = [("a", "near_miss", "a", 0.90), ("a", "matching", "a", 0.80)]
    threshold = calibrate(rows)
    assert threshold == 0.90 and pick(np.array([s for *_, s in rows]), threshold) is None


@pytest.mark.parametrize("rows", [
    [("a", "matching", "a", 0.80)],
    [*TABLE, ("a", "maybe", "a", 0.30)],
    [*TABLE, ("a", "unrelated", "b", float("nan"))],
    [*TABLE, ("a", "matching", "a", 1.5)],
    [*TABLE, ("a", "near_miss", "a", -1.01)],
])
def test_calibrate_refuses_no_must_not_cue_row_an_unknown_label_or_a_score_off_the_cosine_range(rows):
    with pytest.raises(ValueError):
        calibrate(rows)


def _snapshot(folder, files):
    """`files` as the Hugging Face cache holds them: each a link into a sibling blobs folder."""
    blobs = folder.parent / "blobs"
    for rel, blob in files.items():
        (blobs / blob).parent.mkdir(parents=True, exist_ok=True)
        (blobs / blob).write_text(rel)
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).symlink_to(blobs / blob)
    return folder


def test_snapshot_files_names_each_files_blob_and_leaves_out_what_no_loader_opens(tmp_path):
    folder = _snapshot(tmp_path / "snapshots" / "rev", {"config.json": "aa11", "1_Pooling/config.json": "bb22",
                                                        "README.md": "cc33", ".gitattributes": "dd44"})
    (folder / "plain.txt").write_text("x")
    st = (folder / "plain.txt").stat()
    assert snapshot_files(folder) == {"config.json": "aa11", "1_Pooling/config.json": "bb22",
                                      "plain.txt": f"{st.st_size}-{st.st_mtime_ns}"}


def test_a_configured_folders_weights_rewritten_in_place_change_the_pin(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"a")
    (tmp_path / "README.md").write_text("r")
    before = snapshot_files(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"bb")
    assert list(before) == ["model.safetensors"] and snapshot_files(tmp_path) != before


def test_the_descriptions_digest_hashes_name_and_description_lines_in_order():
    text = f"commit-message\0{COMMIT.description}\nnotes\0{NOTES.description}\n"
    assert descriptions_digest([COMMIT, NOTES]) == hashlib.sha256(text.encode()).hexdigest()
    assert descriptions_digest([NOTES, COMMIT]) != descriptions_digest([COMMIT, NOTES])


def _call(args='{"name": "notes"}', name="skill"):
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": args}}


def _assistant(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def test_held_skills_reads_skill_calls_as_objects_and_dicts_and_skips_every_malformed_one():
    messages = [
        _assistant(FakeToolCall("c1", "skill", {"name": "commit-message"}), _call(),
                   FakeToolCall("c2", "read_file", {"name": "review"}), _call('{"name": "review"}', "read_file")),
        {"role": "tool", "tool_call_id": "c1", "content": "error: the skill could not be read"},
        {"role": "user", "content": "", "tool_calls": [_call('{"name": "review"}')]},
        _assistant(_call('["review"]'), _call("3"), _call("null"), _call("{bad"), _call('{"name": 7}'),
                   {"id": "c9", "type": "function"}),
        {"role": "assistant", "content": "done", "tool_calls": None},
        {"role": "assistant", "content": "done"},
    ]
    assert held_skills(messages) == {"commit-message", "notes"}


def test_cued_skills_matches_only_its_own_harness_line():
    line = affordance._line(COMMIT)
    assert cued_skills([harness_message(line, "affordance_cue")], (COMMIT, NOTES)) == {"commit-message"}
    assert cued_skills([human_message(line), harness_message(line, "self_check")], (COMMIT, NOTES)) == set()


def _held(*sources):
    store = ContentStore()
    for source in sources:
        store.declare_active_reference("build", store.put(source), source)
    return store


def test_pinned_skills_reads_only_the_stores_skill_references():
    skills = (COMMIT, NOTES)
    assert pinned_skills(_held("skill notes"), skills) == {"notes"}
    for store in (ContentStore(), _held("skill other", "notes.md"), None):
        assert pinned_skills(store, skills) == set()


@pytest.mark.parametrize("turn, named", [
    ("use the commit-message skill", {"commit-message"}), ("/commit-message", {"commit-message"}),
    ("commit-message,", {"commit-message"}), ("Commit-Message", {"commit-message"}), ("commit-messages", set()),
    ("please review this", {"review"}),
])
def test_a_name_is_named_by_a_whole_token_up_to_punctuation_and_case(turn, named):
    assert affordance._named_in(turn, ["commit-message", "review"]) == named


# --- the cue: the holder --------------------------------------------------------------------------

SKILLS = (COMMIT, NOTES, REVIEW)
TURN = "what goes in the history entry"
LAB = [("commit-message", "matching", "lab a"), ("commit-message", "near_miss", "lab b"),
       ("notes", "matching", "lab c"), ("notes", "unrelated", "lab d")]
VECTORS = {COMMIT.description: [1.0, 0.0, 0.0], NOTES.description: [0.0, 1.0, 0.0],
           REVIEW.description: [0.0, 0.0, 1.0], TURN: [1.0, 0.0, 0.0],
           "lab a": [0.6, -0.8, 0.0], "lab b": [0.4, -0.9165, 0.0], "lab c": [0.0, 1.0, 0.0], "lab d": [0.3, -0.9539, 0.0]}


class FakeEmbedder:
    """A query embedder over a text -> vector table; records each call. `answer` overrides every turn's vector."""

    def __init__(self, folder, table=VECTORS, model_name=MODEL):
        self.folder, self.table, self.model_name = folder, table, model_name
        self.docs = self.answer = None
        self.busy = False
        self.located, self.embedded, self.asked = 0, [], []

    def ensure_local(self):
        self.located += 1
        if isinstance(self.folder, Exception):
            raise self.folder
        return self.folder

    def embed_docs(self, texts):
        self.embedded.append(list(texts))
        return np.array([self.table[t] for t in texts]) if self.docs is None else self.docs

    def try_embed_query(self, text):
        self.asked.append(text)
        return None if self.busy else np.array(self.table[text] if self.answer is None else self.answer)


def _slot(embedder):
    return MemorySlot(_Answers(embedder), None, "memory")


def _record(tmp_path, monkeypatch, prompts=LAB):
    """The labelled set patched to `prompts` over SKILLS (LAB derives 0.50) and a linked snapshot; returns its folder."""
    monkeypatch.setattr(calibration, "labelled", lambda: (list(SKILLS), list(prompts)))
    return _snapshot(tmp_path / "hub" / "snapshots" / "rev", {"config.json": "aa11", "model.safetensors": "bb22"})


def _warmed(tmp_path, monkeypatch, store=None, skills=SKILLS, record_dir=None):
    fake = FakeEmbedder(_record(tmp_path, monkeypatch))
    holder = Affordances(SkillSet(skills, (), ()), _slot(fake), store, record_dir)
    holder.warm().join(timeout=5)
    fake.asked.clear()
    return holder, fake


async def test_a_warmed_holder_cues_the_top_skill_for_a_turn_that_resembles_it(tmp_path, monkeypatch):
    holder, fake = _warmed(tmp_path, monkeypatch)
    assert holder._ready[2] == pytest.approx(0.50, abs=1e-4)
    assert await holder.cue(TURN, []) == Cue("commit-message", "machine", affordance._line(COMMIT))
    assert fake.embedded == [[s.description for s in SKILLS]] * 2 and fake.asked == [TURN]


def _raise(exc):
    def raising(*args, **kwargs):
        raise exc
    return raising


@pytest.mark.parametrize("mutate, located, embedded", [
    (lambda f, t, m: setattr(f, "folder", None), 1, 0),
    (lambda f, t, m: m.setattr(calibration, "snapshot_files", _raise(OSError("unreadable"))), 1, 0),
    (lambda f, t, m: setattr(f, "folder", ResonanceUnavailable(MODEL, "no package")), 1, 0),
    (lambda f, t, m: m.setattr(calibration, "labelled", _raise(ValueError("unreadable labelled skills"))), 1, 0),
    (lambda f, t, m: m.setattr(calibration, "labelled", lambda: (list(SKILLS), [p for p in LAB if p[1] == "matching"])),
     1, 1),
    (lambda f, t, m: setattr(f, "docs", np.eye(3)[:2]), 1, 2),
    (lambda f, t, m: setattr(f, "docs", np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]])), 1, 1),
    (lambda f, t, m: setattr(f, "docs", np.array([[0.0, 0, 0], [0, 1, 0], [0, 0, 1]])), 1, 1),
], ids=["not local", "pin raises", "refused", "labelled unreadable", "no separation", "short", "nan", "zero row"])
def test_warm_turns_the_cue_off_cheapest_first_with_one_info_line(tmp_path, monkeypatch, caplog, mutate, located,
                                                                   embedded):
    fake, hooked = FakeEmbedder(_record(tmp_path, monkeypatch)), []
    mutate(fake, tmp_path, monkeypatch)
    monkeypatch.setattr("threading.excepthook", hooked.append)
    holder = Affordances(SkillSet(SKILLS, (), ()), _slot(fake), None)
    with caplog.at_level(logging.INFO, logger="localharness.agent.affordance"):
        holder.warm().join(timeout=5)
    assert holder._ready is None and hooked == []
    assert fake.located == located and len(fake.embedded) == embedded
    assert [r.levelno for r in caplog.records if r.name == "localharness.agent.affordance"] == [logging.INFO]


async def test_an_empty_slot_or_a_holder_never_warmed_never_cues(tmp_path, monkeypatch):
    holder = Affordances(SkillSet(SKILLS, (), ()), MemorySlot(), None)
    holder.warm().join(timeout=5)
    assert holder._ready is None and await holder.cue(TURN, []) is None
    fake = FakeEmbedder(_record(tmp_path, monkeypatch))
    assert await Affordances(SkillSet(SKILLS, (), ()), _slot(fake), None).cue(TURN, []) is None and fake.asked == []


async def test_a_blank_turn_is_not_embedded(tmp_path, monkeypatch):
    holder, fake = _warmed(tmp_path, monkeypatch)
    assert await holder.cue("  \n", []) is None and fake.asked == []


async def test_a_busy_embedder_skips_this_turn_and_the_next_still_cues(tmp_path, monkeypatch):
    holder, fake = _warmed(tmp_path, monkeypatch)
    fake.busy = True
    assert await holder.cue(TURN, []) is None
    fake.busy = False
    assert (await holder.cue(TURN, [])).name == "commit-message"


@pytest.mark.parametrize("answer", [[1.0, 0.0], [np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]], ids=["width", "nan", "zero"])
async def test_a_bad_turn_vector_turns_the_holder_off_for_the_sitting_with_one_warning(tmp_path, monkeypatch, caplog,
                                                                                       answer):
    holder, fake = _warmed(tmp_path, monkeypatch)
    fake.answer = answer
    with caplog.at_level(logging.WARNING, logger="localharness.agent.affordance"):
        assert await holder.cue(TURN, []) is None
    fake.answer = None
    assert await holder.cue(TURN, []) is None and holder._ready is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


@pytest.mark.parametrize("turn, named, answer, cued", [
    ("a turn", "commit-message", [0.8, 0.6, 0.0], "notes"),
    ("/commit-message", None, [0.8, 0.6, 0.0], "notes"),
    ("commit-message, please", None, [0.8, 0.6, 0.0], "notes"),
    ("Commit-Message", None, [0.8, 0.6, 0.0], "notes"),
    ("commit-messages", None, [0.8, 0.6, 0.0], "commit-message"),
    ("please review this", None, [0.6, 0.0, 0.8], "commit-message"),
    ("commit-message review", "notes", [0.8, 0.6, 0.0], None),
])
async def test_a_named_skill_leaves_the_candidates_and_only_it(tmp_path, monkeypatch, turn, named, answer, cued):
    holder, fake = _warmed(tmp_path, monkeypatch)
    holder.named, fake.answer = named, answer
    got = await holder.cue(turn, [])
    assert (got and got.name) == cued


@pytest.mark.parametrize("call", [FakeToolCall("c1", "skill", {"name": "commit-message"}),
                                  _call('{"name": "commit-message"}')], ids=["object", "dict"])
async def test_a_skill_the_model_called_is_in_view_even_when_the_call_errored(tmp_path, monkeypatch, call):
    holder, _ = _warmed(tmp_path, monkeypatch)
    errored = [_assistant(call), {"role": "tool", "tool_call_id": "c1", "content": "error: unreadable"}]
    assert await holder.cue(TURN, [_assistant(call)]) is None
    assert await holder.cue(TURN, errored) is None


async def test_malformed_skill_calls_in_the_history_still_let_the_cue_answer(tmp_path, monkeypatch):
    holder, _ = _warmed(tmp_path, monkeypatch)
    messages = [_assistant(_call('["commit-message"]'), _call("3"), _call("null"), _call("{bad"),
                           {"id": "c9", "type": "function"})]
    assert (await holder.cue(TURN, messages)).name == "commit-message"


async def test_a_cue_still_in_the_window_is_not_repeated_and_one_compacted_away_re_arms(tmp_path, monkeypatch):
    holder, _ = _warmed(tmp_path, monkeypatch)
    first = await holder.cue(TURN, [])
    window = [human_message(TURN), harness_message(first.line, "affordance_cue")]
    assert await holder.cue(TURN, window) is None
    assert await holder.cue(TURN, [human_message(TURN)]) == first
    assert await holder.cue(TURN, [human_message(TURN), human_message(first.line)]) == first


async def test_a_dropped_skill_call_re_arms(tmp_path, monkeypatch):
    holder, _ = _warmed(tmp_path, monkeypatch)
    call = _assistant(FakeToolCall("c1", "skill", {"name": "commit-message"}))
    assert await holder.cue(TURN, [human_message(TURN), call]) is None
    assert (await holder.cue(TURN, [human_message(TURN)])).name == "commit-message"


async def test_a_skill_the_store_holds_is_in_view_after_compaction(tmp_path, monkeypatch):
    holder, fake = _warmed(tmp_path, monkeypatch, store=_held("skill commit-message"))
    assert await holder.cue(TURN, [human_message(TURN)]) is None
    fake.answer = [0.0, 1.0, 0.0]
    assert (await holder.cue("keep a note", [])).name == "notes"


@pytest.mark.parametrize("store", [ContentStore, lambda: _held("skill notes")], ids=["empty", "another skill"])
async def test_a_store_that_is_empty_or_holds_another_skill_does_not_suppress(tmp_path, monkeypatch, store):
    holder, _ = _warmed(tmp_path, monkeypatch, store=store())
    assert (await holder.cue(TURN, [])).name == "commit-message"


@pytest.mark.usefixtures("pkg")
async def test_a_skill_found_after_warm_is_neither_embedded_nor_cued(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg"
    _skill(cfg / "skills", "commit-message")
    skill_set = SkillSet.discover(cfg, None)
    folder = _record(tmp_path, monkeypatch)
    table = {**VECTORS, "Use commit-message when asked.": [1.0, 0.0, 0.0]}
    fake = FakeEmbedder(folder, table)
    holder = Affordances(skill_set, _slot(fake), None)
    holder.warm().join(timeout=5)
    _skill(cfg / "skills", "late")
    assert skill_set.find("late").name == "late" and [s.name for s in skill_set.skills] == ["commit-message", "late"]
    fake.answer = [0.0, 1.0, 0.0]
    assert fake.embedded[-1:] == [["Use commit-message when asked."]] and await holder.cue("late", []) is None


# --- the cue: the machine record -----------------------------------------------------------------

ASKED = [t for *_, t in LAB]


def _again(fake, record_dir):
    holder = Affordances(SkillSet(SKILLS, (), ()), _slot(fake), None, record_dir)
    holder.warm().join(timeout=5)
    return holder


def test_an_absent_record_is_measured_written_and_published_in_the_same_warm(tmp_path, monkeypatch):
    record_dir = tmp_path / "cfg" / "affordance"
    fake = FakeEmbedder(_record(tmp_path, monkeypatch))
    holder = _again(fake, record_dir)
    entry = read_entry(record_dir, MODEL)
    assert fake.asked == ASKED and len(entry["rows"]) == len(LAB) and entry["date"] == datetime.date.today()
    assert entry_state(entry, shipped_pins(MODEL, fake.folder), LAB) == ("ready", holder._ready[2])
    assert list(record_dir.iterdir()) == [record_path(record_dir, MODEL)]
    assert os.name != "posix" or record_path(record_dir, MODEL).stat().st_mode & 0o077 == 0
    fake.asked.clear()
    assert _again(fake, record_dir)._ready[2] == holder._ready[2] and fake.asked == []


@pytest.mark.parametrize("pin, value", [("model_name", "lh-test/other"), ("snapshot_files", {"config.json": "ff99"}),
                                        ("prompts_sha256", "0" * 64), ("descriptions_sha256", "1" * 64), ("rev", -1)])
def test_each_pin_change_measures_again(tmp_path, monkeypatch, pin, value):
    record_dir = tmp_path / "affordance"
    fake = FakeEmbedder(_record(tmp_path, monkeypatch))
    _again(fake, record_dir)
    write_entry(record_dir, MODEL, {**read_entry(record_dir, MODEL), pin: value})
    fake.asked.clear()
    assert _again(fake, record_dir)._ready is not None and fake.asked == ASKED
    assert read_entry(record_dir, MODEL)[pin] != value


def test_a_partial_record_resumes_at_its_row_count(tmp_path, monkeypatch):
    record_dir = tmp_path / "affordance"
    fake = FakeEmbedder(_record(tmp_path, monkeypatch))
    _again(fake, record_dir)
    entry = read_entry(record_dir, MODEL)
    write_entry(record_dir, MODEL, {**entry, "rows": entry["rows"][:3]})
    assert entry_state(read_entry(record_dir, MODEL), shipped_pins(MODEL, fake.folder), LAB) == ("partial", "3 of 4")
    fake.asked.clear()
    assert _again(fake, record_dir)._ready is not None and fake.asked == ASKED[3:]
    assert read_entry(record_dir, MODEL)["rows"] == entry["rows"]


def _rows():
    return [[item, label, "commit-message", 0.5 if label == "matching" and item == "commit-message" else 0.1]
            for item, label, _ in LAB]


def test_a_complete_table_with_no_admitted_score_above_the_must_not_cue_maximum_is_failed(tmp_path, monkeypatch):
    pins = shipped_pins(MODEL, _record(tmp_path, monkeypatch))
    tied = [[item, label, "commit-message", 0.1] for item, label, _ in LAB]
    assert calibrate(tied) == 0.1
    assert entry_state({**pins, "date": DATE, "rows": tied}, pins, LAB) == ("failed", NO_SEPARATION)


@pytest.mark.parametrize("rows", [
    lambda r: r[:1] + [["commit-message", "matching", "notes", 0.1]] + r[2:],
    lambda r: [["notes", *r[0][1:]]] + r[1:],
    lambda r: r + [r[0]],
    lambda r: [r[0][:3] + [1.5]] + r[1:],
    lambda r: [r[0][:3] + [float("nan")]] + r[1:],
    lambda r: [r[0][:3]] + r[1:],
    lambda r: [tuple(r[0])] + r[1:],
    lambda r: {"rows": r},
], ids=["label at a position", "item at a position", "more rows than prompts", "score off the range", "nan",
        "short row", "not a list", "not a list of rows"])
def test_rows_that_are_not_a_prefix_of_the_labelled_order_are_stale(tmp_path, monkeypatch, rows):
    pins = shipped_pins(MODEL, _record(tmp_path, monkeypatch))
    entry = {**pins, "date": DATE, "rows": rows(_rows())}
    assert entry_state(entry, pins, LAB)[0] == "stale"
    assert entry_state({**pins, "date": DATE, "rows": _rows()}, pins, LAB)[0] == "ready"
    assert entry_state({**pins, "date": "2026-10-25", "rows": _rows()}, pins, LAB)[0] == "stale"
    assert entry_state(None, pins, LAB)[0] == "stale"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs here")
def test_a_fifo_a_link_or_an_oversized_record_reads_as_absent(tmp_path):
    path = record_path(tmp_path, MODEL)
    os.mkfifo(path)
    assert read_entry(tmp_path, MODEL) is None
    path.unlink()
    path.write_text("rows: [" + "0.1, " * 20_000 + "]\n")
    assert path.stat().st_size > calibration.MAX_RECORD and read_entry(tmp_path, MODEL) is None
    (tmp_path / "real.yaml").write_text("rows: []\n")
    path.unlink()
    path.symlink_to(tmp_path / "real.yaml")
    assert read_entry(tmp_path, MODEL) is None


def test_a_failed_record_is_not_measured_again_until_a_pin_changes(tmp_path, monkeypatch, caplog):
    record_dir = tmp_path / "affordance"
    fake = FakeEmbedder(_record(tmp_path, monkeypatch, [p for p in LAB if p[1] == "matching"]))
    with caplog.at_level(logging.INFO, logger="localharness.agent.affordance"):
        assert _again(fake, record_dir)._ready is None
    assert "failed" in read_entry(record_dir, MODEL) and NO_SEPARATION in caplog.text
    fake.asked.clear()
    assert _again(fake, record_dir)._ready is None and fake.asked == []
    monkeypatch.setattr(calibration, "REV", calibration.REV + 1)
    _again(fake, record_dir)
    assert fake.asked == ["lab a", "lab c"]


def test_a_record_that_cannot_be_written_still_publishes_for_the_sitting(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "write_entry", _raise(OSError("read-only file system")))
    holder, _ = _warmed(tmp_path, monkeypatch, record_dir=tmp_path / "affordance")
    assert holder._ready is not None and list((tmp_path / "affordance").iterdir()) == []


def test_without_a_record_dir_the_holder_measures_and_never_reads_or_writes(tmp_path, monkeypatch):
    touched = []
    monkeypatch.setattr(calibration, "read_entry", lambda *a: touched.append(a))
    monkeypatch.setattr(calibration, "write_entry", lambda *a: touched.append(a))
    monkeypatch.setattr(calibration, "usable_dir", lambda d: touched.append(d))
    holder, _ = _warmed(tmp_path, monkeypatch)
    assert holder._ready is not None and touched == [None]


@pytest.mark.parametrize("plant", ["symlink", "file"])
def test_a_record_dir_that_is_a_link_or_a_file_is_left_alone_and_the_cue_measures_in_memory(tmp_path, monkeypatch,
                                                                                           plant):
    folder, target, record_dir = _record(tmp_path, monkeypatch), tmp_path / "elsewhere", tmp_path / "affordance"
    target.mkdir()
    write_entry(target, MODEL, {**shipped_pins(MODEL, folder), "date": DATE, "rows": [], "failed": "planted"})
    planted = read_entry(target, MODEL)
    record_dir.symlink_to(target) if plant == "symlink" else record_dir.write_text("x")
    assert _again(FakeEmbedder(folder), record_dir)._ready is not None
    assert read_entry(target, MODEL) == planted and len(list(target.iterdir())) == 1
    assert record_dir.is_symlink() if plant == "symlink" else record_dir.read_text() == "x"


def test_measuring_waits_while_a_turn_runs_and_resumes_when_the_last_one_ends(tmp_path, monkeypatch):
    fake = FakeEmbedder(_record(tmp_path, monkeypatch))
    holder = Affordances(SkillSet(SKILLS, (), ()), _slot(fake), None)
    holder.turn_started()
    holder.turn_started()
    thread = holder.warm()
    thread.join(timeout=0.2)
    holder.turn_ended()
    thread.join(timeout=0.2)
    assert thread.is_alive() and fake.asked == []
    holder.turn_ended()
    thread.join(timeout=5)
    assert holder._ready is not None and holder._turns == 0 and fake.asked == ASKED


def test_measuring_yields_before_each_prompt_not_only_each_batch(tmp_path, monkeypatch):
    fake = FakeEmbedder(_record(tmp_path, monkeypatch))
    holder, real = Affordances(SkillSet(SKILLS, (), ()), _slot(fake), None), fake.try_embed_query

    def first_starts_a_turn(text):
        if not fake.asked:
            holder.turn_started()
        return real(text)
    fake.try_embed_query = first_starts_a_turn
    thread = holder.warm()
    thread.join(timeout=0.2)
    assert thread.is_alive() and fake.asked == ["lab a"] and calibration.BATCH > len(LAB)
    holder.turn_ended()
    thread.join(timeout=5)
    assert holder._ready is not None and fake.asked == ASKED


def test_measuring_stops_encoding_once_the_interpreter_is_exiting(tmp_path, monkeypatch):
    stop = threading.Event()
    monkeypatch.setattr(calibration, "STOP", stop)
    fake = FakeEmbedder(_record(tmp_path, monkeypatch))
    real = fake.try_embed_query
    fake.try_embed_query = lambda text: (stop.set(), real(text))[1]
    assert _again(fake, tmp_path / "affordance")._ready is None and fake.asked == ["lab a"]
    assert read_entry(tmp_path / "affordance", MODEL) is None


def test_a_write_never_replaces_a_longer_record_under_the_same_pins(tmp_path, monkeypatch):
    record_dir, fake = tmp_path / "affordance", FakeEmbedder(_record(tmp_path, monkeypatch))
    _again(fake, record_dir)
    full, pins = read_entry(record_dir, MODEL), shipped_pins(MODEL, fake.folder)
    calibration._written(record_dir, pins, {**full, "rows": full["rows"][:1]})
    assert read_entry(record_dir, MODEL) == full
    calibration._written(record_dir, {**pins, "rev": -1}, {**full, "rev": -1, "rows": full["rows"][:1]})
    assert len(read_entry(record_dir, MODEL)["rows"]) == 1


def test_a_score_rounded_past_one_is_stored_as_one(tmp_path, monkeypatch):
    past_one = np.nextafter(np.float32(1), np.float32(2))
    monkeypatch.setattr(calibration, "scores", lambda docs, vec: np.array([past_one, 0, 0], dtype=np.float32))
    fake, idle = FakeEmbedder(_record(tmp_path, monkeypatch)), threading.Event()
    idle.set()
    pins = shipped_pins(MODEL, fake.folder)
    entry = calibration.measure(fake, pins, None, idle, None)
    assert past_one > 1 and {r[3] for r in entry["rows"]} == {1.0}
    assert entry_state(entry, pins, LAB) == ("failed", NO_SEPARATION)


def _failing_sittings(tmp_path, monkeypatch, n):
    monkeypatch.setattr(calibration, "BATCH", 1)
    fake = FakeEmbedder(_record(tmp_path, monkeypatch), {**VECTORS, "lab c": [0, 0, 0]})
    record_dir = tmp_path / "affordance"
    return record_dir, fake, [(_again(fake, record_dir), read_entry(record_dir, MODEL).get("attempts"))[1]
                              for _ in range(n)]


def test_an_encode_error_that_recurs_three_sittings_running_is_failed_with_its_type(tmp_path, monkeypatch):
    record_dir, fake, attempts = _failing_sittings(tmp_path, monkeypatch, 3)
    assert attempts == [1, 2, 3]
    assert entry_state(read_entry(record_dir, MODEL), shipped_pins(MODEL, fake.folder), LAB) == (
        "failed", "measuring stopped on ValueError 3 sittings in a row")
    fake.asked.clear()
    assert _again(fake, record_dir)._ready is None and fake.asked == []


def test_sittings_that_failed_on_an_earlier_day_are_measured_again_from_their_rows(tmp_path, monkeypatch):
    record_dir, fake, _ = _failing_sittings(tmp_path, monkeypatch, 3)
    path = record_path(record_dir, MODEL)
    entry = read_entry(record_dir, MODEL)
    path.write_text(yaml.safe_dump({**entry, "date": entry["date"] - datetime.timedelta(days=1)}), encoding="utf-8")
    assert entry_state(read_entry(record_dir, MODEL), shipped_pins(MODEL, fake.folder), LAB)[0] == "partial"
    fake.table = VECTORS
    assert _again(fake, record_dir)._ready is not None and "attempts" not in read_entry(record_dir, MODEL)


def test_progress_clears_the_failed_sittings_count(tmp_path, monkeypatch):
    record_dir, fake, attempts = _failing_sittings(tmp_path, monkeypatch, 2)
    fake.table = VECTORS
    assert attempts == [1, 2] and _again(fake, record_dir)._ready is not None
    assert "attempts" not in read_entry(record_dir, MODEL)


def test_the_record_module_never_imports_the_cue():
    assert "localharness.agent.affordance" not in Path(calibration.__file__).read_text(encoding="utf-8")


def test_a_busy_embedder_gives_up_after_its_bound_keeping_each_written_batch(tmp_path, monkeypatch, caplog):
    for name, value in (("BATCH", 1), ("BUSY_S", 0.2), ("RETRY_S", 0.01)):
        monkeypatch.setattr(calibration, name, value)
    fake, tries = FakeEmbedder(_record(tmp_path, monkeypatch)), []
    real = fake.try_embed_query
    fake.try_embed_query = lambda text: tries.append(text) if text == "lab c" else real(text)
    with caplog.at_level(logging.INFO, logger="localharness.agent.affordance"):
        assert _again(fake, tmp_path / "affordance")._ready is None
    assert "the embedder stayed busy for 0.2 s" in caplog.text and len(tries) > 2
    assert [r[2] for r in read_entry(tmp_path / "affordance", MODEL)["rows"]] == ["commit-message", "commit-message"]


# --- the cue: the loop ----------------------------------------------------------------------------

PREFIX = "[Harness control: affordance_cue; not human feedback]\n"


def _cue_loop(tmp_path, affordances=None):
    """A root loop answering "Done."; records each request, the session and its turn_start_idx at the first."""
    llm = MockLLMClient([FakeLLMResponse(content="Done.")])
    seen, original, bus = SimpleNamespace(requests=[], events=[]), llm.stream_complete, EventBus()

    async def recording(messages=None, tools=None, on_token=None, **kw):
        if not seen.requests:
            seen.session = loop._live_turn[0]
            seen.turn_start_idx = seen.session.turn_start_idx
        seen.requests.append(list(messages))
        return await original(messages=messages, tools=tools, on_token=on_token, **kw)

    llm.stream_complete = recording
    bus.subscribe(AffordanceCued, seen.events.append)
    loop = AgentLoop(config=AgentConfig(name="cue-agent", role="You are a test assistant."), llm=llm, bus=bus,
                     context_manager=ContextManager(), tool_registry=None, permission_evaluator=PermissionEvaluator(),
                     guardrails_path=tmp_path / "GUARDRAILS.md", affordances=affordances)
    return loop, seen


def _cues(messages):
    return [m for m in messages if is_harness_message(m) and m["_lh"].get("subtype") == "affordance_cue"]


async def test_a_turn_resembling_a_skill_reaches_the_model_with_one_harness_line_after_the_person(tmp_path,
                                                                                                  monkeypatch):
    holder, _ = _warmed(tmp_path, monkeypatch)
    loop, seen = _cue_loop(tmp_path, holder)
    await loop.run_turn(TURN)
    first = seen.requests[0]
    i = next(n for n, m in enumerate(first) if m["role"] == "user" and m["content"] == TURN)
    cue = first[i + 1]
    assert cue["role"] == "user" and cue["_lh"]["origin"] == "harness" and cue["_lh"]["subtype"] == "affordance_cue"
    assert cue["content"] == affordance._line(COMMIT) and _cues(first) == [cue]
    wire = provider_messages([cue])[0]["content"]
    assert wire == PREFIX + affordance._line(COMMIT) and wire.count("commit-message") == 1
    assert [(e.name, e.origin, e.agent_id, e.session_id) for e in seen.events] == [
        ("commit-message", "machine", "cue-agent", loop._current_session_id)]


async def test_a_loop_without_a_holder_sends_no_cue_and_publishes_none(tmp_path):
    loop, seen = _cue_loop(tmp_path)
    await loop.run_turn(TURN)
    assert _cues(seen.requests[0]) == [] and seen.events == []


class _Holder:
    """A cue holder that records each call and answers `answer`, after `gate` opens when one is given."""

    def __init__(self, order, answer=None, gate=None):
        self.order, self.answer, self.gate, self.calls = order, answer, gate, []

    def turn_started(self):
        self.order.append("started")

    def turn_ended(self):
        self.order.append("ended")

    def cue(self, turn, messages):
        self.order.append("cue")
        self.calls.append((turn, messages))

        async def answering():
            if self.gate is not None:
                await self.gate.wait()
            return self.answer

        self.coro = answering()
        return self.coro


NOTES_CUE = Cue("notes", "project", affordance._line(NOTES))


async def test_the_cue_starts_before_the_prompt_is_built_and_reads_the_sessions_own_messages(tmp_path, monkeypatch):
    order = []
    holder = _Holder(order, NOTES_CUE)
    loop, seen = _cue_loop(tmp_path, holder)
    real = loop_module.base_system_prompt
    monkeypatch.setattr(loop_module, "base_system_prompt", lambda *a, **k: order.append("prompt") or real(*a, **k))
    await loop.run_turn(TURN)
    assert order == ["started", "cue", "prompt", "ended"]
    ((turn, messages),) = holder.calls
    assert turn == TURN and messages is seen.session.messages


async def test_the_cue_lands_right_after_the_persons_message_and_the_turn_starts_past_both(tmp_path):
    loop, seen = _cue_loop(tmp_path, _Holder([], NOTES_CUE))
    await loop.run_turn(TURN)
    first = seen.requests[0]
    i = next(n for n, m in enumerate(first) if m["role"] == "user" and m["content"] == TURN)
    assert first[i + 1] == harness_message(NOTES_CUE.line, "affordance_cue")
    assert seen.turn_start_idx == i + 2
    assert [(e.name, e.origin) for e in seen.events] == [("notes", "project")]


async def test_a_holder_answering_none_pushes_nothing_and_publishes_nothing(tmp_path):
    loop, seen = _cue_loop(tmp_path, _Holder([], None))
    await loop.run_turn(TURN)
    assert _cues(seen.requests[0]) == [] and seen.events == []


async def test_the_pending_cue_is_cancelled_when_the_assembly_raises(tmp_path, monkeypatch):
    holder = _Holder([], NOTES_CUE, gate=asyncio.Event())
    loop, seen = _cue_loop(tmp_path, holder)
    real, tasks = asyncio.ensure_future, []

    def recording(coro, **kw):
        task = real(coro, **kw)
        if task.get_coro() is getattr(holder, "coro", None):
            tasks.append(task)
        return task

    monkeypatch.setattr(asyncio, "ensure_future", recording)
    monkeypatch.setattr(loop_module, "base_system_prompt", _raise(RuntimeError("assembly failed")))
    await loop.run_turn(TURN)
    await asyncio.sleep(0)
    (task,) = tasks
    assert task.cancelled() and seen.requests == [] and seen.events == []


async def test_a_cancelled_turn_returns_the_in_flight_count_to_zero(tmp_path):
    holder = Affordances(SkillSet(SKILLS, (), ()), MemorySlot(), None)
    loop, _ = _cue_loop(tmp_path, holder)
    asked = asyncio.Event()

    async def hang(*args, **kwargs):
        asked.set()
        await asyncio.Event().wait()

    loop._llm.stream_complete = hang
    turn = asyncio.ensure_future(loop.run_turn(TURN))
    await asked.wait()
    assert holder._turns == 1 and not holder._idle.is_set()
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    assert holder._turns == 0 and holder._idle.is_set()


# --- file hygiene -------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
IDS = re.compile(r"\b(?:D|R|T|WR)-\d+|R§|§")


def test_the_measuring_module_never_names_the_holdout():
    assert "prompts_holdout" not in Path(calibration.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("path", [Path(affordance.__file__), Path(calibration.__file__)], ids=["affordance", "calibration"])
def test_the_cue_files_carry_no_review_or_plan_id(path):
    assert IDS.findall(path.read_text(encoding="utf-8")) == []


def test_the_id_pattern_bites():
    assert IDS.findall("D-14 R-28 T-05 WR-09 R§2 §8 1-D WORD-1 PID-3") == ["D-14", "R-28", "T-05", "WR-09", "R§", "§"]


LABELLED = ROOT / "src" / "localharness" / "assets" / "affordance"
HOLDOUT = ROOT / "tests" / "fixtures" / "skills_live" / "prompts_holdout.yaml"


def test_the_holdout_shares_no_prompt_with_the_calibration_set():
    def texts(path):
        sets = yaml.safe_load(path.read_text(encoding="utf-8"))["skills"]
        return {t.strip().casefold() for groups in sets.values() for kind in calibration.LABELS for t in groups[kind]}
    assert texts(HOLDOUT) and not texts(HOLDOUT) & texts(LABELLED / "prompts.yaml")


def test_the_labelled_skills_are_in_no_discovery_root(tmp_path):
    (tmp_path / ".localharness").mkdir()
    found = SkillSet.discover(tmp_path, tmp_path)
    labelled = {p.parent.name for p in LABELLED.glob("skills/*/SKILL.md")}
    assert len(labelled) == 5 and not labelled & {s.name for s in found.skills}
    assert (LABELLED / "skills").resolve() not in {real for _, _, real in found.origins}
    assert {read_skill(LABELLED / "skills" / name, "machine").name for name in labelled} == labelled


# --- doctor's affordance-cue row ------------------------------------------------------------------

CUE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
_MEASURES = ("cue off until an open session has measured it in the background (a few minutes of CPU, yielding to "
             "your turns); this shows ready when it is done")
_READY = dict(model=CUE_MODEL, memory_state="on", embedding_status="pass", cached=True, skill_tool_held=True, other=None,
              unusable=None, state=("ready", 0.5), date=DATE)
_DOCTOR_ROWS = [
    ("third-party-occupant", dict(other="mem0", memory_state="off", embedding_status=None), "skip",
     "cue measured anew in each session if mem0, the memory plugin in use, offers an embedder (another plugin's "
     "measurement is never stored)"),
    ("memory-not-on", dict(memory_state="off", embedding_status=None),
     "skip", "cue off: the memory plugin is not on (the cue uses memory's embedder)"),
    ("embedding-failed", dict(embedding_status="fail"), "skip", "cue off: waits on memory-embedding"),
    ("embedding-missing", dict(embedding_status=None), "skip", "cue off: waits on memory-embedding"),
    ("not-cached", dict(cached=False), "skip", "cue off: waits on memory-embedding"),
    ("skill-tool-not-held", dict(skill_tool_held=False), "skip", "cue off: the skill tool is not enabled for the root"),
    ("unusable-dir", dict(unusable=Path("/c/affordance")), "skip",
     "the cue is measured anew each sitting because /c/affordance is not a usable directory"),
    ("partial", dict(state=("partial", "40 of 180")), "skip",
     "cue off until an open session has measured it in the background (40 of 180 done; a few minutes of CPU, yielding to your "
     "turns); this shows ready when it is done"),
    ("failed", dict(state=("failed", NO_SEPARATION)), "skip", f"cue off: {NO_SEPARATION}"),
    ("stale", dict(state=("stale", "")), "skip", _MEASURES),
    ("ready", {}, "pass", f"ready — {CUE_MODEL}, calibrated 2026-10-25"),
    ("ready-other-model", dict(model="lh-test/other"), "pass",
     "ready — lh-test/other, calibrated 2026-10-25 (not release-tested)"),
]


@pytest.mark.parametrize("given, status, detail", [r[1:] for r in _DOCTOR_ROWS], ids=[r[0] for r in _DOCTOR_ROWS])
def test_affordance_check_rows_in_order(given, status, detail):
    check = doctor_cmd._affordance_check(**{**_READY, **given})
    assert (check.name, check.status, check.detail, check.hint) == ("affordance-cue", status, detail, "")


def test_the_release_tested_model_is_the_memory_plugins_default():
    from localharness.memory.config import MemoryConfig

    assert calibration.TESTED_MODEL == MemoryConfig().embedding_model == CUE_MODEL


def test_affordance_check_never_warns_or_fails_and_every_off_row_says_so():
    checks = [doctor_cmd._affordance_check(**{**_READY, **r[1]}) for r in _DOCTOR_ROWS]
    assert {c.status for c in checks} == {"pass", "skip"}
    assert all(c.detail.startswith(("cue off", "the cue is measured anew", "cue measured anew")) for c in checks if c.status != "pass")


def test_doctor_and_the_engine_read_the_folder_through_one_function():
    assert resonance.local_folder is config_paths.local_folder


def _affordance_cue_doctor(monkeypatch, tmp_path, root_agent, memory_state="on", occupant="memory", measured=True):
    import io

    from rich.console import Console

    async def rows(*_args, **_kwargs):
        return [DoctorRow("memory", memory_state, "", checks=(Check(name="memory-db", status="pass"),
                                                             Check(name="memory-embedding", status="pass")),
                          bundled=True),
                *([DoctorRow(occupant, "on", "", checks=(Check(name=f"{occupant}-db", status="pass"),))]
                  if occupant != "memory" else []),
                DoctorRow("p", "off", "p is off")]
    plan = SimpleNamespace(memory_occupant=occupant if occupant != "memory" or memory_state == "on" else None,
                           entry=lambda name: SimpleNamespace(bundled=True) if name == "memory" else None)
    folder = _record(tmp_path, monkeypatch)
    if measured:
        idle = threading.Event()
        idle.set()
        (tmp_path / "affordance").mkdir()
        calibration.measure(FakeEmbedder(folder, model_name=CUE_MODEL), shipped_pins(CUE_MODEL, folder), None, idle,
                            tmp_path / "affordance")
    monkeypatch.setattr("localharness.plugins.resolve.resolve", lambda _loader: SimpleNamespace(warnings=[], plan=plan))
    monkeypatch.setattr("localharness.plugins.lifecycle.doctor_rows", rows)
    monkeypatch.setattr(doctor_cmd, "_print_remote_lock", lambda *_a: None)
    monkeypatch.setattr(config_paths, "local_folder", lambda model, cfg: folder)
    buf = io.StringIO()
    monkeypatch.setattr(doctor_cmd, "console", Console(file=buf, width=400, color_system=None))
    failures: list[str] = []
    doctor_cmd._print_plugins(tmp_path, None, None, failures, root_agent)
    return buf.getvalue().splitlines(), failures


_ROOT_AGENT = SimpleNamespace(memory=SimpleNamespace(embedding_model=CUE_MODEL), tools=ToolConfig())


def test_affordance_cue_row_follows_the_plugin_rows(monkeypatch, tmp_path):
    lines, failures = _affordance_cue_doctor(monkeypatch, tmp_path, _ROOT_AGENT)
    plugin = next(i for i, line in enumerate(lines) if "p: p is off" in line)
    assert f"affordance-cue: ready — {CUE_MODEL}, calibrated {datetime.date.today().isoformat()}" in lines[-1]
    assert plugin < len(lines) - 1 and failures == []


def test_affordance_cue_row_without_a_record_says_it_measures_at_the_next_start(monkeypatch, tmp_path):
    lines, failures = _affordance_cue_doctor(monkeypatch, tmp_path, _ROOT_AGENT, measured=False)
    assert f"affordance-cue: {_MEASURES}" in lines[-1] and failures == []


def test_affordance_cue_row_beside_a_third_party_memory_occupant_names_it_and_reads_no_record(monkeypatch, tmp_path):
    read = []
    monkeypatch.setattr(calibration, "read_entry", lambda *a: read.append(a))
    lines, _ = _affordance_cue_doctor(monkeypatch, tmp_path, _ROOT_AGENT, memory_state="off", occupant="mem0",
                                      measured=False)
    assert lines[-1].endswith("affordance-cue: cue measured anew in each session if mem0, the memory plugin in use, "
                              "offers an embedder (another plugin's measurement is never stored)") and read == []


@pytest.mark.parametrize("plant", ["symlink", "check"])
def test_doctor_and_warm_share_one_record_dir_check(monkeypatch, tmp_path, plant):
    if plant == "symlink":
        (tmp_path / "elsewhere").mkdir()
        (tmp_path / "affordance").symlink_to(tmp_path / "elsewhere")
    else:
        monkeypatch.setattr(calibration, "writable_dir", lambda d: False)
    lines, _ = _affordance_cue_doctor(monkeypatch, tmp_path, _ROOT_AGENT, measured=False)
    assert lines[-1].endswith(f"affordance-cue: the cue is measured anew each sitting because {tmp_path / 'affordance'} "
                              "is not a usable directory")
    assert calibration.usable_dir(tmp_path / "affordance") is None


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a directory this user cannot write")
def test_a_record_dir_this_process_cannot_write_is_measured_in_memory(tmp_path, monkeypatch):
    record_dir = tmp_path / "affordance"
    record_dir.mkdir(mode=0o500)
    try:
        assert _again(FakeEmbedder(_record(tmp_path, monkeypatch)), record_dir)._ready is not None
        assert list(record_dir.iterdir()) == []
    finally:
        record_dir.chmod(0o700)


def test_the_warm_and_the_doctor_row_both_follow_entry_state(monkeypatch, tmp_path):
    monkeypatch.setattr(calibration, "entry_state", lambda entry, pins, prompts: ("partial", "1 of 4"))
    lines, _ = _affordance_cue_doctor(monkeypatch, tmp_path, _ROOT_AGENT)
    assert "(1 of 4 done; a few minutes of CPU, yielding to your turns)" in lines[-1]
    monkeypatch.setattr(calibration, "entry_state", lambda entry, pins, prompts: ("ready", 0.25))
    holder, fake = _warmed(tmp_path / "w", monkeypatch)
    assert holder._ready[2] == 0.25 and len(fake.embedded) == 1


def test_affordance_cue_row_is_absent_without_a_root_agent(monkeypatch, tmp_path):
    lines, failures = _affordance_cue_doctor(monkeypatch, tmp_path, None)
    assert not any("affordance-cue" in line for line in lines) and failures == []


def test_affordance_cue_row_reads_the_memory_row_doctor_collected(monkeypatch, tmp_path):
    lines, _ = _affordance_cue_doctor(monkeypatch, tmp_path, _ROOT_AGENT, memory_state="off")
    assert "affordance-cue: cue off: the memory plugin is not on (the cue uses memory's embedder)" in lines[-1]


def test_affordance_cue_row_without_memory_settings_says_memory_is_not_on(monkeypatch, tmp_path):
    lines, _ = _affordance_cue_doctor(monkeypatch, tmp_path, SimpleNamespace(tools=ToolConfig()), memory_state="off")
    assert "affordance-cue: cue off: the memory plugin is not on (the cue uses memory's embedder)" in lines[-1]


def test_the_security_sentences_about_the_cue_carry_no_section_sign():
    text = " ".join((ROOT / "SECURITY.md").read_text(encoding="utf-8").split())
    cue = [s for s in re.split(r"(?<=[.;])\s+", text) if re.search(r"\bcue\b", s)]
    assert cue and not [s for s in cue if "§" in s]

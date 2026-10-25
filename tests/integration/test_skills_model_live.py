"""The subject model loads a listed skill on its own: the live gate of the skills release claim."""
from __future__ import annotations

import contextlib
import functools
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from localharness.agent.context import ContentStore, ContextManager
from localharness.agent.gate import PermissionGate
from localharness.agent.loop import AgentLoop
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.task_record import TaskState
from localharness.bench.config import MatrixEntry
from localharness.bench.orchestrator import _build_bench_client
from localharness.cli.agent_cmd import _build_agent_yaml
from localharness.config.grants import GrantStore
from localharness.config.loader import ConfigLoader
from localharness.config.models import AgentConfig
from localharness.config.skills import SkillSet
from localharness.core.bus import EventBus
from localharness.core.events import Action
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.builtin.agent_tool import AgentTool
from localharness.tools.builtin.skill_tool import SkillTool, render_listing
from localharness.tools.builtin.task_tool import TaskTool
from localharness.tools.capabilities import apply_root_capability_floor
from localharness.tools.registry import ToolRegistry
from tests.conftest import _minimal_config, live_target

pytestmark = pytest.mark.live_vllm

RUNS = 3
MATCHING_BAR = (24, 30)
FALSE_BAR = (3, 60)
CREATOR_BAR = (2, 3)
TEMPERATURE = 0.6
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "skills_live"
NO_CHILD_AGENTS = ["explore", "web-researcher", "cruncher", "search-verifier"]


class FirstStepDone(BaseException):
    """Raised on the second model request of a turn, so nothing after the first step runs."""


@dataclass(frozen=True)
class Step:
    calls: tuple[tuple[str, dict], ...]
    finish_reason: str | None
    system: str


def prompts() -> dict:
    return yaml.safe_load((FIXTURES / "prompts.yaml").read_text(encoding="utf-8"))


def install(config_dir: Path) -> SkillSet:
    shutil.copytree(FIXTURES / "skills", config_dir / "skills")
    return SkillSet.discover(config_dir, None)


def root_config(config_dir: Path) -> AgentConfig:
    agents = config_dir / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(_minimal_config(), encoding="utf-8")
    (agents / "orchestrator.yaml").write_text(
        yaml.dump(_build_agent_yaml("orchestrator", "General-purpose assistant", None), default_flow_style=False),
        encoding="utf-8")
    return ConfigLoader(config_dir=config_dir).load_agent("orchestrator")


@contextlib.contextmanager
def first_request_only(client, seen: list):
    real = client.stream_complete

    @functools.wraps(real)
    async def guarded(*args, **kwargs):
        if seen:
            raise FirstStepDone
        seen.append((args, kwargs))
        return await real(*args, **kwargs)

    client.stream_complete = guarded
    try:
        yield
    finally:
        del client.stream_complete


def first_step(events) -> tuple[tuple[tuple[str, dict], ...], str | None]:
    actions = [e for e in events if isinstance(e, Action)]
    start = next((i for i, e in enumerate(actions) if e.action_type == "llm_response"), None)
    if start is None:
        raise AssertionError("the model never answered")
    end = next((i for i, e in enumerate(actions) if i > start and e.action_type == "llm_response"), len(actions))
    calls = tuple((e.tool_name, dict(e.tool_params or {})) for e in actions[start + 1:end] if e.action_type == "tool_call")
    return calls, actions[start].finish_reason


async def no_child(agent_id, brief, grant_handles, shape=None) -> str:
    return "no child agent is dispatched in this run"


async def first_step_run(client, skill_set: SkillSet, cfg: AgentConfig, prompt: str, work: Path) -> Step:
    registry = ToolRegistry()
    await register_builtin_tools(registry)
    state = TaskState()
    await registry.register(AgentTool(agent_runner=no_child, available_agents=NO_CHILD_AGENTS), scope="global")
    await registry.register(TaskTool(state), scope="global")
    await registry.register(
        SkillTool(skill_set, ContentStore(), state, result_cap=lambda: registry.result_size_cap_chars), scope="global")
    apply_root_capability_floor(cfg.tools, registry.global_schemas())
    assert "skill" in registry.get_tools_for_agent(cfg.name, cfg.division or "", cfg.tools)
    bus = EventBus()
    gate = PermissionGate(boundary=None, workspace=work, grants=GrantStore(work.parent / "grants.yaml"), mode="read-only")
    loop = AgentLoop(
        config=cfg, llm=client, bus=bus, context_manager=ContextManager(), tool_registry=registry,
        permission_evaluator=PermissionEvaluator(), gate=gate, kill_file_path=None,
        skills=render_listing(skill_set.skills), task_context=state)
    seen: list = []
    with first_request_only(client, seen):
        with contextlib.suppress(FirstStepDone):
            await loop.run_turn(prompt)
    calls, finish = first_step(bus.history(event_types=[Action]))
    args, kwargs = seen[0]
    messages = kwargs["messages"] if "messages" in kwargs else args[0]
    return Step(calls, finish, messages[0]["content"])


async def test_the_subject_model_picks_a_listed_skill_on_its_own(live_endpoint, tmp_path, monkeypatch):
    model, base_url = live_target()
    client = _build_bench_client(MatrixEntry(name=model, provider="vllm", model_id=model, base_url=base_url))
    await client.detect_capabilities()
    assert client.config.tool_call_mode == "native"
    client.config.temperature = TEMPERATURE
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    skill_set = install(tmp_path / "cfg")
    cfg = root_config(tmp_path / "cfg")
    request = prompts()["skills"]["release-notes"]["matching"][0]
    step = await first_step_run(client, skill_set, cfg, request, work)
    assert render_listing(skill_set.skills) in step.system

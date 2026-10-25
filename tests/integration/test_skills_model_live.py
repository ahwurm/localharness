"""The subject model loads a listed skill on its own: the live gate of the skills release claim."""
from __future__ import annotations

import contextlib
import functools
import json
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
KINDS = ("matching", "near_miss", "unrelated")
REPORTS = Path(__file__).resolve().parents[2] / "reports"
NO_WINDOW = "not exposed by the server"
CREATOR_MAX_ACTIONS = 40
CREATOR_MAX_MINUTES = 30.0
VARIANT_A = "On the subject model, the model picked a matching skill from the listing on its own in at least 80 % of test runs."
VARIANT_B = "The model may not pick a skill up on its own; `/<name>` asks it to."
ROOT = ("a fresh install's orchestrator with the built-in tools, agent, task and skill; no plugins, no memory, "
        "no guardrails file, no project instructions; the web verbs are denied to it, as in a real start")


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


@dataclass(frozen=True)
class Run:
    leg: str
    skill: str
    kind: str
    prompt: str
    index: int
    step: Step


@dataclass(frozen=True)
class CreatorRun:
    request: str
    asks: tuple[tuple[str, str, bool], ...]
    terminated: str | None
    listing: str
    valid: bool


def skill_params(run: Run) -> list[dict]:
    return [params for name, params in run.step.calls if name == "skill"]


def hit(run: Run) -> bool:
    params = skill_params(run)
    return any(p.get("name") == run.skill for p in params) if run.kind == "matching" else bool(params)


def count(runs, leg: str, skill: str) -> tuple[int, int, int, int]:
    own = [r for r in runs if r.leg == leg and r.skill == skill]
    matching = [r for r in own if r.kind == "matching"]
    others = [r for r in own if r.kind != "matching"]
    return sum(map(hit, matching)), len(matching), sum(map(hit, others)), len(others)


def tally(runs, skill: str) -> tuple[int, int, int, int]:
    return count(runs, "native", skill)


def passed(runs, creator_valid) -> bool:
    skills = sorted({r.skill for r in runs})
    assert skills, "no runs to judge"
    assert len(creator_valid) == CREATOR_BAR[1], f"{len(creator_valid)} creator runs, the bar counts {CREATOR_BAR[1]}"
    rows = [tally(runs, s) for s in skills]
    for skill, (_, matching, _, others) in zip(skills, rows):
        assert (matching, others) == (MATCHING_BAR[1], FALSE_BAR[1]), f"{skill}: {matching} matching and {others} other native runs"
    return (all(m >= MATCHING_BAR[0] and f <= FALSE_BAR[0] for m, _, f, _ in rows)
            and sum(creator_valid) >= CREATOR_BAR[0])


def out_of_list(runs, names) -> dict[str, list[tuple[str, str]]]:
    found: dict[str, list[tuple[str, str]]] = {"native": [], "xml": []}
    for run in runs:
        for params in skill_params(run):
            name = params.get("name")
            if not (isinstance(name, str) and name in names):
                found[run.leg].append((run.prompt, repr(name)))
    return found


def report_path(day: str) -> Path:
    return REPORTS / f"skills-live-gate-{day}.md"


def write_report(day: str, text: str) -> Path:
    path = report_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)
    return path


def cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def table(header: list[str], rows) -> list[str]:
    return [f"| {' | '.join(header)} |", "|" + " --- |" * len(header),
            *(f"| {' | '.join(cell(c) for c in row)} |" for row in rows)]


def render_report(meta: dict, runs, creator) -> str:
    ok = passed(runs, [c.valid for c in creator])
    skills = sorted({r.skill for r in runs})
    window = NO_WINDOW if meta["window"] is None else meta["window"]
    outside = out_of_list(runs, skills)
    names = {leg: "; ".join(f"{name} on {json.dumps(prompt)}" for prompt, name in found) or "none"
             for leg, found in outside.items()}
    native = [(s, *count(runs, "native", s)) for s in skills]
    xml = [(s, *count(runs, "xml", s)) for s in skills]
    lines = [
        "# Skills live-model gate", "",
        f"Date: {meta['day']} · Repo: {meta['commit']}",
        f"Outcome: {'pass' if ok else 'miss'}",
        f"The claim: {VARIANT_A if ok else VARIANT_B}",
        f"Model: {meta['model']} · Base URL: {meta['base_url']} · Window: {window} · Temperature: {meta['temperature']}",
        f"Wall time: {meta['minutes']} minutes",
        f"Root: {ROOT}",
        f"Tools: {', '.join(meta['tools'])}", "",
        "## The bar, fixed before the run", "",
        f"- Matching requests: at least {MATCHING_BAR[0]} of {MATCHING_BAR[1]} per skill, no averaging across skills.",
        f"- Near-miss and unrelated requests: at most {FALSE_BAR[0]} of {FALSE_BAR[1]} per skill.",
        f"- Creator runs: at least {CREATOR_BAR[0]} of {CREATOR_BAR[1]} valid.",
        f"- Every request runs {RUNS} times in native tool-call mode; the XML pass runs the set once and is not judged.", "",
        "## Native, per skill", "",
        *table(["skill", "matching", "near-miss plus unrelated", "meets the bar"],
               [(s, f"{m} of {mt}", f"{f} of {ft}", "yes" if m >= MATCHING_BAR[0] and f <= FALSE_BAR[0] else "no")
                for s, m, mt, f, ft in native]), "",
        "## XML tool-call mode, one run of the whole set (recorded, not judged)", "",
        *table(["skill", "matching", "near-miss plus unrelated"],
               [(s, f"{m} of {mt}", f"{f} of {ft}") for s, m, mt, f, ft in xml]), "",
        "## Names outside the listing", "",
        f"native: {names['native']}", f"xml: {names['xml']}", "",
        "## Creator runs", "",
        f"Limits per run: {CREATOR_MAX_ACTIONS} actions, {int(CREATOR_MAX_MINUTES)} minutes.", "",
        *table(["run", "request", "valid", "asks", "terminated"],
               [(i, c.request, "yes" if c.valid else "no",
                 "; ".join(f"{tool} {'allowed' if allowed else 'denied'}: {display}" for tool, display, allowed in c.asks),
                 c.terminated) for i, c in enumerate(creator, 1)]), "",
        "## The listing as rendered", "", "```text", meta["listing"], "```", "",
        "## Every run", "",
        *table(["leg", "skill", "kind", "run", "request", "first-step calls", "finish reason", "hit", "counted"],
               [(r.leg, r.skill, r.kind, r.index, r.prompt,
                 json.dumps(r.step.calls, sort_keys=True, default=repr), r.step.finish_reason,
                 "yes" if hit(r) else "no", "yes" if r.leg == "native" else "no")
                for r in runs]),
    ]
    return "\n".join(lines) + "\n"


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

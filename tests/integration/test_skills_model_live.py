"""The subject model loads a listed skill on its own: the live gate of the skills release claim."""
from __future__ import annotations

import contextlib
import functools
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml
from typer.testing import CliRunner

from localharness.agent.affordance import CALIBRATION, Affordances, load_calibration, snapshot_files
from localharness.agent.context import ContentStore, ContextManager, context_size_kwargs, floor_schemas, probe_served_window
from localharness.agent.gate import PermissionGate, derive_session_boundary
from localharness.agent.gate_types import Decision, PermissionRequest
from localharness.agent.loop import AgentLoop, render_index
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.task_record import TaskState
from localharness.bench.config import MatrixEntry
from localharness.bench.orchestrator import _build_bench_client
from localharness.cli.agent_cmd import _build_agent_yaml
from localharness.cli.app import app
from localharness.cli.skill_session import SkillSession
from localharness.cli.start_cmd import _effective_max_context
from localharness.config.effort import resolve_effort
from localharness.config.grants import GrantStore
from localharness.config.loader import ConfigLoader
from localharness.config import skills as skill_config
from localharness.config.models import AgentConfig
from localharness.config.skills import SkillSet
from localharness.core.bus import EventBus
from localharness.core.events import Action, AffordanceCued, TurnFailed
from localharness.memory.resonance import DEFAULT_EMBEDDING_MODEL, ResonanceEngine
from localharness.provider.fn_call import FnCallConverter
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.builtin.paths import resolve_user_path
from localharness.tools.builtin.agent_tool import AgentTool
from localharness.tools.builtin.skill_tool import SkillTool, render_listing
from localharness.tools.builtin.task_tool import TaskTool
from localharness.tools.capabilities import apply_root_capability_floor
from localharness.tools.registry import ToolRegistry
from tests.conftest import _minimal_config, live_target
from tests.unit.test_skills_misc import VARIANT_A, VARIANT_B

pytestmark = pytest.mark.live_vllm

RUNS = 3
MATCHING_BAR = (24, 30)
FALSE_BAR = (3, 60)
CREATOR_BAR = (2, 3)
TEMPERATURE = 0.6
CREATOR_MAX_ACTIONS = 40
CREATOR_MAX_MINUTES = 30.0
MAX_ERROR_STREAK = 10
ROOT_DIR = Path(__file__).resolve().parents[2]
FIXTURES = ROOT_DIR / "tests" / "fixtures" / "skills_live"
NO_CHILD_AGENTS = ["explore", "web-researcher", "cruncher", "search-verifier"]
KINDS = ("matching", "near_miss", "unrelated")
NON_ANSWERS = ("degenerate", "reasoning_cut", "length")
REPORTS = ROOT_DIR / "reports"
NO_WINDOW = "not exposed by the server"
NOT_SERVING = "the subject model was not being served at the endpoint"
NOT_SUBJECT = "the endpoint served a model the reference architectures do not name as the subject model"
NOT_FINISHED = "the run did not finish, so no result exists"
# matching and false hits per skill of reports/skills-live-gate-2026-10-24-2.md, code f4329641
RUN_2 = {"bug-repro-test": (26, 0), "commit-message": (21, 0), "docstring-pass": (30, 0), "incident-summary": (24, 0),
         "release-notes": (30, 0), "skill-creator": (25, 0)}
MODEL_ABSENT = "model absent"
ROOT = ("a fresh install's orchestrator with the built-in tools, agent, task and skill; no plugins, no memory, "
        "no guardrails file, no project instructions; the web verbs are denied to it, as in a real start")


class FirstStepDone(BaseException):
    """Raised on the second model request of a turn, so nothing after the first step runs."""


class RequestFailed(Exception):
    """The loop ended the turn on a failure; the message is the cause it published."""


class RunStopped(Exception):
    """The server failed too many requests in a row for the rest of the run to mean anything."""


@dataclass(frozen=True)
class Step:
    calls: tuple[tuple[str, dict], ...]
    finish_reason: str | None
    system: str
    flaw: str | None = None
    error: str | None = None
    cue: str | None = None
    cue_s: float | None = None
    ttft_s: float | None = None


@dataclass(frozen=True)
class Shape:
    window: int
    effort: tuple[str, str]
    ratios: tuple[float, float]


def prompts() -> dict:
    return yaml.safe_load((FIXTURES / "prompts.yaml").read_text(encoding="utf-8"))


def holdout() -> dict:
    return yaml.safe_load((FIXTURES / "prompts_holdout.yaml").read_text(encoding="utf-8"))


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


def loop_shape(config_dir: Path, cfg: AgentConfig, base_url: str, served: int | None) -> Shape:
    """The window and effort a real `start` hands its root loop."""
    harness = ConfigLoader(config_dir=config_dir).load_harness()
    return Shape(
        _effective_max_context(served, cfg.context.max_context_tokens),
        resolve_effort(None, cfg.effort, harness, base_url),
        (harness.effort_budget.low_ratio, harness.effort_budget.high_ratio))


def machine_dir(skill_set: SkillSet) -> Path:
    return next(lexical for label, lexical, _ in skill_set.origins if label == "machine").parent


def root_loop(client, cfg: AgentConfig, shape: Shape, skill_set: SkillSet, registry, state, gate, bus,
              affordances=None) -> AgentLoop:
    held = {s.name for s in floor_schemas(registry, cfg.name, cfg.division or "", cfg.tools)}
    return AgentLoop(
        config=cfg, llm=client, bus=bus, tool_registry=registry, permission_evaluator=PermissionEvaluator(), gate=gate,
        context_manager=ContextManager(
            max_context_tokens=shape.window, preserve_first_n=cfg.context.preserve_first_n_messages,
            preserve_last_n=cfg.context.preserve_last_n_messages, **context_size_kwargs(cfg.context)),
        kill_file_path=None, skills=render_listing(skill_set.skills), task_context=state,
        effort=shape.effort, effort_ratios=shape.ratios,
        index=render_index(None, machine_dir(skill_set), held, False, True), affordances=affordances)


@contextlib.contextmanager
def first_request_only(client, seen: list, errors: list, first: list | None = None):
    real, previous = client.stream_complete, getattr(client, "on_reasoning", None)

    @functools.wraps(real)
    async def guarded(*args, **kwargs):
        if seen:
            raise FirstStepDone
        seen.append(kwargs)
        if first is not None:
            sent = time.monotonic()

            def stamped(forward):
                async def on_piece(piece):
                    if not first:
                        first.append(time.monotonic() - sent)
                    if forward is not None:
                        await forward(piece)
                return on_piece

            kwargs["on_token"], client.on_reasoning = stamped(kwargs.get("on_token")), stamped(previous)
        try:
            return await real(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - kept for the report, re-raised to the loop
            errors.append(exc)
            raise

    client.stream_complete = guarded
    try:
        yield
    finally:
        del client.stream_complete
        if first is not None:
            client.on_reasoning = previous


@contextlib.contextmanager
def xml_mode(client):
    """The client as detection leaves an xml server: the mode and the converter that teaches the syntax."""
    mode, converter = client.config.tool_call_mode, getattr(client, "_fn_converter", None)
    client.config.tool_call_mode, client._fn_converter = "xml", FnCallConverter()
    try:
        yield
    finally:
        client.config.tool_call_mode, client._fn_converter = mode, converter


def failure(bus: EventBus) -> str | None:
    """The cause the loop published when it ended a turn on a failure."""
    failed = bus.history(event_types=[TurnFailed])
    return f"{failed[-1].reason}: {failed[-1].detail[:120]}" if failed else None


def first_step(events, mode: str = "native") -> tuple[tuple[tuple[str, dict], ...], str | None, str | None]:
    actions = [e for e in events if isinstance(e, Action)]
    start = next((i for i, e in enumerate(actions) if e.action_type == "llm_response"), None)
    if start is None:
        raise AssertionError("the model never answered")
    end = next((i for i, e in enumerate(actions) if i > start and e.action_type == "llm_response"), len(actions))
    calls = tuple((e.tool_name, dict(e.tool_params or {})) for e in actions[start + 1:end] if e.action_type == "tool_call")
    first = actions[start]
    return calls, first.finish_reason, flaw_of(first, calls, mode)


def flaw_of(response: Action, calls, mode: str = "native") -> str | None:
    """Why a first response is no answer: a cut or aborted generation, a call the loop did not dispatch, or nothing said and nothing called."""
    if response.finish_reason in NON_ANSWERS:
        return response.finish_reason
    if response.finish_reason == "tool_calls" and not calls:
        return "native-in-xml" if mode == "xml" else "calls not dispatched"
    if not calls and not (response.content or "").strip():
        return "reasoning only" if response.reasoning_chars else "empty"
    return None


async def no_child(agent_id, brief, grant_handles, shape=None) -> str:
    return "no child agent is dispatched in this run"


async def root_registry(skill_set: SkillSet, cfg: AgentConfig) -> tuple[ToolRegistry, TaskState]:
    registry = ToolRegistry()
    await register_builtin_tools(registry)
    state = TaskState()
    await registry.register(AgentTool(agent_runner=no_child, available_agents=NO_CHILD_AGENTS), scope="global")
    await registry.register(TaskTool(state), scope="global")
    await registry.register(
        SkillTool(skill_set, ContentStore(), state, result_cap=lambda: registry.result_size_cap_chars), scope="global")
    apply_root_capability_floor(cfg.tools, registry.global_schemas())
    assert "skill" in registry.get_tools_for_agent(cfg.name, cfg.division or "", cfg.tools)
    return registry, state


async def first_step_run(client, skill_set: SkillSet, cfg: AgentConfig, shape: Shape, prompt: str, work: Path,
                         engine=None) -> Step:
    registry, state = await root_registry(skill_set, cfg)
    bus = EventBus()
    gate = PermissionGate(boundary=None, workspace=work, grants=GrantStore(work.parent / "grants.yaml"), mode="read-only")
    holder, spent = None, []
    if engine is not None:
        holder = Affordances(skill_set, SimpleNamespace(embedder=lambda: engine), state)
        holder.warm().join(timeout=300)
        cue = holder.cue

        async def timed(turn, messages):
            start = time.perf_counter()
            try:
                return await cue(turn, messages)
            finally:
                spent.append(time.perf_counter() - start)

        holder.cue = timed
    loop = root_loop(client, cfg, shape, skill_set, registry, state, gate, bus, holder)
    seen: list = []
    errors: list = []
    firsts: list = []
    with first_request_only(client, seen, errors, firsts):
        with contextlib.suppress(FirstStepDone):
            await loop.run_turn(prompt)
    if cause := failure(bus):
        raise RequestFailed(cause)
    actions = bus.history(event_types=[Action])
    if errors and not any(e.action_type == "llm_response" for e in actions):
        raise RequestFailed(f"{type(errors[0]).__name__}: {str(errors[0])[:120]}")
    calls, finish, flaw = first_step(actions, getattr(client.config, "tool_call_mode", "native"))
    cued = bus.history(event_types=[AffordanceCued])
    return Step(calls, finish, seen[0]["messages"][0]["content"], flaw, cue=cued[0].name if cued else None,
                cue_s=spent[0] if spent else None, ttft_s=firsts[0] if firsts else None)


async def settled(client, skill_set: SkillSet, cfg: AgentConfig, shape: Shape, prompt: str, work: Path,
                  engine=None) -> Step:
    """One request, tried twice; a second failure is an `error` step, which the bar counts against the model."""
    for _ in range(2):
        try:
            step = await first_step_run(client, skill_set, cfg, shape, prompt, work, engine)
            assert render_listing(skill_set.skills) in step.system
            return step
        except Exception as exc:  # noqa: BLE001 - any failure of one request is recorded, not fatal to the run
            error = exc
    return Step((), None, "", error=repr(error)[:200])


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
    files: tuple[tuple[str, str], ...] = ()


def stop_on_failing_server(runs) -> None:
    streak = sum(1 for _ in itertools.takewhile(lambda r: r.step.error, reversed(runs)))
    if streak >= MAX_ERROR_STREAK:
        raise RunStopped(f"stopped after {streak} requests in a row errored; the last: {runs[-1].step.error}")


def skill_params(run: Run) -> list[dict]:
    return [params for name, params in run.step.calls if name == "skill"]


def hit(run: Run) -> bool:
    if run.step.error:
        return run.kind != "matching"
    params = skill_params(run)
    return any(p.get("name") == run.skill for p in params) if run.kind == "matching" else bool(params)


def count(runs, leg: str, skill: str) -> tuple[int, int, int, int]:
    own = [r for r in runs if r.leg == leg and r.skill == skill]
    matching = [r for r in own if r.kind == "matching"]
    others = [r for r in own if r.kind != "matching"]
    return sum(map(hit, matching)), len(matching), sum(map(hit, others)), len(others)


def flaws(runs, leg: str, skill: str) -> tuple[int, int]:
    own = [r for r in runs if r.leg == leg and r.skill == skill]
    return sum(r.step.flaw is not None for r in own), sum(r.step.error is not None for r in own)


def meets(matching: int, false: int) -> bool:
    return matching >= MATCHING_BAR[0] and false <= FALSE_BAR[0]


def sizes(runs, creator_valid) -> list[str]:
    """What is wrong with the shape of the result set, empty when it is the one the bar counts."""
    skills = sorted({r.skill for r in runs})
    if not skills:
        return ["no runs to judge"]
    short = [f"{len(creator_valid)} creator runs, the bar counts {CREATOR_BAR[1]}"] if len(creator_valid) != CREATOR_BAR[1] else []
    for skill in skills:
        _, matching, _, others = count(runs, "native", skill)
        if (matching, others) != (MATCHING_BAR[1], FALSE_BAR[1]):
            short.append(f"{skill}: {matching} matching and {others} other native runs")
    return short


def judge(runs, creator_valid) -> bool:
    rows = [count(runs, "native", s) for s in sorted({r.skill for r in runs})]
    return bool(rows) and all(meets(m, f) for m, _, f, _ in rows) and sum(creator_valid) >= CREATOR_BAR[0]


def passed(runs, creator_valid) -> bool:
    short = sizes(runs, creator_valid)
    assert not short, "; ".join(short)
    return judge(runs, creator_valid)


def out_of_list(runs, names) -> dict[str, list[tuple[str, str]]]:
    found: dict[str, list[tuple[str, str]]] = {"native": [], "xml": []}
    for run in runs:
        found.setdefault(run.leg, [])
        for params in skill_params(run):
            name = params.get("name")
            if not (isinstance(name, str) and name in names):
                found[run.leg].append((run.prompt, repr(name)))
    return found


def cue_reason(folder, calibration, files) -> str | None:
    if folder is None:
        return MODEL_ABSENT
    if calibration is None:
        return "no calibration record for the embedding model"
    return "the cached embedding model is not the calibrated one" if files != calibration.snapshot_files else None


def cue_setup() -> tuple[ResonanceEngine | None, str | None, object | None]:
    engine = ResonanceEngine(DEFAULT_EMBEDDING_MODEL)
    try:
        folder = engine.ensure_local()
    except Exception:  # noqa: BLE001 - a model that cannot load is absent to the cue arm
        folder = None
    calibration = load_calibration(engine.model_name) if folder is not None else None
    reason = cue_reason(folder, calibration, snapshot_files(folder) if folder is not None else None)
    return engine if reason is None else None, reason, calibration


def cue_counts(runs, leg: str, skill: str) -> tuple[int, int, int, int, int]:
    """Runs, fired, precise (a matching prompt cued for its own skill), followed and ignored, for one leg and skill."""
    own = [r for r in runs if r.leg == leg and r.skill == skill]
    fired = [r for r in own if r.step.cue is not None]
    followed = sum(any(p.get("name") == r.step.cue for p in skill_params(r)) for r in fired)
    precise = sum(r.kind == "matching" and r.step.cue == r.skill for r in fired)
    return len(own), len(fired), precise, followed, len(fired) - followed


def closed_misses(runs) -> list[str]:
    """Commit-message holdout requests the no-cue leg missed in every run and the cued leg hit in every run."""
    def hits(leg):
        found: dict[str, list[bool]] = {}
        for r in runs:
            if r.leg == leg and r.skill == "commit-message" and r.kind == "matching":
                found.setdefault(r.prompt, []).append(hit(r))
        return found
    without, cued = hits("holdout"), hits("cue-holdout")
    return [p for p, h in without.items() if len(h) == RUNS and not any(h) and len(cued.get(p, ())) == RUNS and all(cued[p])]


def fit(rows, threshold: float, skill: str) -> tuple[int, int]:
    matching = [(top, score) for item, label, top, score in rows if item == skill and label == "matching"]
    return sum(top == skill and score > threshold for top, score in matching), len(matching)


def cue_passed(runs) -> bool:
    skills = sorted({r.skill for r in runs})
    return bool(skills) and bool(closed_misses(runs)) and all(
        meets(count(runs, "cue-holdout", s)[0], count(runs, "cue-holdout", s)[2]) and count(runs, "cue", s)[2] <= FALSE_BAR[0]
        for s in skills)


def cue_false_within_bar(runs) -> bool:
    return all(count(runs, leg, s)[2] <= FALSE_BAR[0] for s in {r.skill for r in runs} for leg in ("cue", "cue-holdout"))


def spread(values) -> str:
    return f"median {median(values):.3f} s, maximum {max(values):.3f} s" if values else "not measured"


def cue_section(cue: dict, runs) -> list[str]:
    skills = sorted({r.skill for r in runs})
    if cue["reason"]:
        return ["## The cue", "", f"cue arm not run: {cue['reason']}", "",
                *table(["skill", "holdout matching without the cue", "holdout near-miss plus unrelated without the cue"],
                       [(s, f"{m} of {mt}", f"{f} of {ft}") for s in skills for m, mt, f, ft in [count(runs, "holdout", s)]]), ""]
    rows = []
    for s in skills:
        for label, leg, without in (("calibration", "cue", f"{RUN_2[s][0]} of 30 (run 2)"),
                                    ("holdout", "cue-holdout", "{} of {}".format(*count(runs, "holdout", s)[:2]))):
            m, mt, f, ft = count(runs, leg, s)
            n, fired, precise, followed, ignored = cue_counts(runs, leg, s)
            rows.append((s, label, f"{m} of {mt}", without, f"{f} of {ft}", f"{f - RUN_2[s][1]:+d}", f"{fired} of {n}",
                         f"{precise} of {fired}" if fired else "none fired", followed, ignored))
    cue_s = [r.step.cue_s for r in runs if r.step.cue_s is not None]
    ttft = [r.step.ttft_s for r in runs if r.step.ttft_s is not None]
    fired = [r for r in runs if r.step.cue is not None]
    return [
        "## The cue", "",
        f"Embedder: {cue['embedder']} · Threshold: {cue['threshold']} (derived at load from the stored rows of the record "
        f"dated {cue['date'].isoformat()})",
        "Calibration-set recall (fit: the prompts the threshold was derived from): "
        + "; ".join(f"{s} {k} of {n}" for s in skills for k, n in [fit(cue["rows"], cue["threshold"], s)]), "",
        *table(["skill", "set", "matching with the cue", "matching without", "near-miss plus unrelated with the cue",
                "change from run 2", "fired", "precision", "followed", "ignored"], rows), "",
        f"cue() wall time per turn: {spread(cue_s)} over {len(cue_s)} turns · first-token latency: {spread(ttft)} "
        f"over {len(ttft)} responses that streamed a token",
        "Closed misses (commit-message holdout, missed in every no-cue run, hit in every cued run): "
        + ("; ".join(json.dumps(p) for p in closed_misses(runs)) or "none"),
        f"Cue outcome: {'pass' if cue_passed(runs) else 'miss'}", "",
        "### Fired cues", "",
        *(table(["leg", "skill", "kind", "run", "request", "cued", "followed"],
                [(r.leg, r.skill, r.kind, r.index, r.prompt, r.step.cue,
                  "yes" if any(p.get("name") == r.step.cue for p in skill_params(r)) else "no") for r in fired])
          if fired else ["none"]), ""]


def report_path(day: str, run: int = 1) -> Path:
    return REPORTS / (f"skills-live-gate-{day}.md" if run == 1 else f"skills-live-gate-{day}-{run}.md")


def run_number(day: str) -> int:
    """The first run of the day whose report does not exist yet."""
    return next(n for n in itertools.count(1) if not report_path(day, n).exists())


def write_report(day: str, text: str, run: int = 1) -> Path:
    path = report_path(day, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)
    return path


def cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def table(header: list[str], rows) -> list[str]:
    return [f"| {' | '.join(header)} |", "|" + " --- |" * len(header),
            *(f"| {' | '.join(cell(c) for c in row)} |" for row in rows)]


def fenced(text: str) -> list[str]:
    ticks = "`" * max(3, max(map(len, re.findall(r"`+", text)), default=0) + 1)
    return [f"{ticks}text", text, ticks]


def creator_section(i: int, c: CreatorRun) -> list[str]:
    skill_files = [(path, text) for path, text in c.files if Path(path).name == "SKILL.md"]
    others = [path for path, _ in c.files if Path(path).name != "SKILL.md"]
    return [
        f"### Creator run {i}", "", "Output of `localharness skills`:", "", *fenced(c.listing), "",
        *(line for path, text in skill_files for line in (f"Written `{path}`:", "", *fenced(text), "")),
        f"Other files written: {', '.join(others) or 'none'}", ""]


def per_leg(found, legs=("native", "xml")) -> str:
    return ", ".join(f"{leg}: {sum(r.leg == leg for r in found)}" for leg in legs)


def render_report(meta: dict, runs, creator, stopped: str | None = None) -> str:
    valid = [c.valid for c in creator]
    short = [stopped] if stopped else sizes(runs, valid)
    ok = not short and judge(runs, valid)
    skills = sorted({r.skill for r in runs})
    window = NO_WINDOW if meta["window"] is None else meta["window"]
    level, rung = meta["effort"]
    outside = out_of_list(runs, skills)
    names = {leg: "; ".join(f"{name} on {json.dumps(prompt)}" for prompt, name in found) or "none"
             for leg, found in outside.items()}
    native = [(s, *count(runs, "native", s), *flaws(runs, "native", s)) for s in skills]
    xml = [(s, *count(runs, "xml", s), *flaws(runs, "xml", s)) for s in skills]
    errored = [r for r in runs if r.step.error]
    unanswered = [r for r in runs if r.step.flaw]
    lines = [
        "# Skills live-model gate", "",
        f"Date: {meta['day']} · Run: {meta['run']} · Repo: {meta['commit']} ({meta['tree']} tree) · Gate module sha256: {meta['sha']}",
        f"Outcome: {'incomplete' if short else 'pass' if ok else 'miss'}",
        *([f"Incomplete because: {'; '.join(short)}"] if short else []),
        f"The claim: {VARIANT_A if ok else VARIANT_B}",
        f"Model: {meta['model']} · Base URL: {meta['base_url']} · Window: {window} · Temperature: {meta['temperature']}",
        f"Loop: effort {level} ({rung} rung) · context window {meta['loop_window']}",
        f"Wall time: {meta['minutes']} minutes",
        f"Root: {ROOT}",
        f"Tools: {', '.join(meta['tools'])}", "",
        "## The bar, fixed before the run", "",
        f"- Matching requests: at least {MATCHING_BAR[0]} of {MATCHING_BAR[1]} per skill, no averaging across skills.",
        f"- Near-miss and unrelated requests: at most {FALSE_BAR[0]} of {FALSE_BAR[1]} per skill.",
        f"- Creator runs: at least {CREATOR_BAR[0]} of {CREATOR_BAR[1]} valid.",
        f"- Every request runs {RUNS} times in native tool-call mode; the XML pass runs the set once and is not judged.",
        "- A request that errored twice counts as no pick on a matching request and as a false pick on any other.",
        "- A response that was no answer (cut, aborted, empty) counts as no pick; its count is beside each figure.", "",
        "## Native, per skill", "",
        *table(["skill", "matching", "near-miss plus unrelated", "meets the bar", "no answer", "errors"],
               [(s, f"{m} of {mt}", f"{f} of {ft}", "n/a" if short else "yes" if meets(m, f) else "no", na, er) for s, m, mt, f, ft, na, er in native]), "",
        "## XML tool-call mode, one run of the whole set (recorded, not judged)", "",
        *table(["skill", "matching", "near-miss plus unrelated", "no answer", "errors"],
               [(s, f"{m} of {mt}", f"{f} of {ft}", na, er) for s, m, mt, f, ft, na, er in xml]), "",
        *(cue_section(meta["cue"], runs) if "cue" in meta else []),
        "## Requests that errored twice", "", f"Count: {per_leg(errored, tuple(outside))}", "",
        *(table(["leg", "skill", "kind", "run", "request", "exception"],
                [(r.leg, r.skill, r.kind, r.index, r.prompt, r.step.error) for r in errored]) if errored else ["none"]), "",
        "## First responses that were no answer", "", f"Count: {per_leg(unanswered, tuple(outside))}", "",
        *(table(["leg", "skill", "kind", "run", "request", "what it was"],
                [(r.leg, r.skill, r.kind, r.index, r.prompt, r.step.flaw) for r in unanswered]) if unanswered else ["none"]), "",
        "## Names outside the listing", "",
        *(f"{leg}: {text}" for leg, text in names.items()), "",
        "## Creator runs", "",
        f"Harness limits per run, not part of the bar: {CREATOR_MAX_ACTIONS} actions, {int(CREATOR_MAX_MINUTES)} minutes.", "",
        *table(["run", "request", "valid", "asks", "terminated"],
               [(i, c.request, "yes" if c.valid else "no",
                 "; ".join(f"{tool} {'allowed' if allowed else 'denied'}: {display}" for tool, display, allowed in c.asks),
                 c.terminated) for i, c in enumerate(creator, 1)]), "",
        *(line for i, c in enumerate(creator, 1) for line in creator_section(i, c)),
        "## The listing as rendered", "", *fenced(meta["listing"]), "",
        "## Every run", "",
        *table(["leg", "skill", "kind", "run", "request", "first-step calls", "finish reason", "problem", "hit", "counted"],
               [(r.leg, r.skill, r.kind, r.index, r.prompt,
                 json.dumps(r.step.calls, sort_keys=True, default=repr), r.step.finish_reason,
                 r.step.error or r.step.flaw or "", "yes" if hit(r) else "no", "yes" if r.leg == "native" else "no")
                for r in runs]),
    ]
    return "\n".join(lines) + "\n"


def not_run_report(day: str, model: str | None, endpoint: str | None, reason: str) -> str:
    return "\n".join([
        "# Skills live-model gate", "",
        f"Date: {day}",
        f"Model: {model or 'none configured'}",
        f"Endpoint: {endpoint or 'none configured'}",
        "Outcome: not run",
        f"Why: {reason}",
        f"The claim kept: {VARIANT_B}",
        "No run finished, so this report holds no result.", ""])


def subject_served(model: str) -> bool:
    return "qwen" in model.lower()


def where(params: dict) -> Path | None:
    path = params.get("path")
    if not isinstance(path, str):
        return None
    try:
        return resolve_user_path(path)
    except (OSError, RuntimeError, ValueError):
        return None


def creator_asker(skills_dir: Path, asks: list):
    """Allows a `write` under the skills directory and nothing else; a denied write records the path as the model wrote it."""
    async def asker(req: PermissionRequest) -> Decision:
        target = where(req.tool_params)
        allowed = req.tool_name == "write" and target is not None and skills_dir in target.parents
        shown = req.display if allowed or req.tool_name != "write" else (
            f"{req.display} (the model wrote {req.tool_params.get('path')!r}, resolved to {target})")
        asks.append((req.tool_name, shown, allowed))
        return Decision("allow_once" if allowed else "reject_once")

    return asker


@contextlib.contextmanager
def bundled_under(root: Path):
    """The bundled skills laid out under `root`, so the directory the skill tool names carries no user name."""
    real = skill_config.PACKAGE_DIR
    shutil.copytree(real / "assets" / "skills", root / "assets" / "skills")
    skill_config.PACKAGE_DIR = root
    try:
        yield
    finally:
        skill_config.PACKAGE_DIR = real


@contextlib.contextmanager
def neutral_sitting(fake_home):
    """A person's sitting: a fake home, a workspace holding `.localharness/` and the bundled skills, all under a temp dir whose name carries no user, with `USER` and `LOGNAME` neutral."""
    base = Path(tempfile.mkdtemp(prefix="lh-")).resolve()
    here = Path.cwd()
    try:
        home = fake_home(base / "home")
        work = home / "work"
        (work / ".localharness").mkdir(parents=True)
        os.chdir(work)
        with bundled_under(home / "pkg"), mock.patch.dict(os.environ, {"USER": "user", "LOGNAME": "user"}):
            yield home, work
    finally:
        os.chdir(here)
        shutil.rmtree(base, ignore_errors=True)


async def creator_run(client, home: Path, request: str, work: Path, base_url: str, served: int | None) -> CreatorRun:
    cfg_dir = home / ".localharness"
    base = root_config(cfg_dir)
    budget = base.permissions.budget.model_copy(
        update={"max_actions": CREATOR_MAX_ACTIONS, "max_duration_minutes": CREATOR_MAX_MINUTES})
    cfg = base.model_copy(update={"permissions": base.permissions.model_copy(update={"budget": budget})})
    skill_set = SkillSet.discover(cfg_dir, None)
    invoked = SkillSession(skill_set).invoke(skill_set.find("skill-creator"), request)
    assert isinstance(invoked, tuple), invoked
    asks: list[tuple[str, str, bool]] = []
    skills_dir = (cfg_dir / "skills").resolve()
    gate = PermissionGate(
        boundary=derive_session_boundary(cwd=work, local_dir=work / ".localharness"), workspace=work,
        grants=GrantStore(home.parent / "grants.yaml"), mode="guarded", asker=creator_asker(skills_dir, asks))
    registry, state = await root_registry(skill_set, cfg)
    bus = EventBus()
    loop = root_loop(client, cfg, loop_shape(cfg_dir, cfg, base_url, served), skill_set, registry, state, gate, bus)
    try:
        await loop.run_turn(invoked[0])
        terminated = loop.last_terminated_reason
        failed = terminated == "error"
        if failed:
            terminated = failure(bus) or terminated
    except Exception as exc:  # noqa: BLE001 - a creator turn that died is an invalid run, not a lost report
        terminated, failed = f"error: {type(exc).__name__}", True
    result = CliRunner().invoke(app, ["skills", "--config-dir", str(cfg_dir)])
    valid = not failed and result.exit_code == 0 and "Skipped:" not in result.output and " (machine, " in result.output
    files = tuple(sorted(
        (str(p.relative_to(skills_dir)), p.read_text(encoding="utf-8", errors="replace"))
        for p in skills_dir.rglob("*") if p.is_file()))
    return CreatorRun(request, tuple(asks), terminated, result.output, valid, files)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=ROOT_DIR).stdout.strip()


async def test_the_subject_model_picks_a_listed_skill_on_its_own(live_endpoint, fake_home, tmp_path, monkeypatch):
    day = date.today().isoformat()
    number = run_number(day)
    model, base_url = live_target()
    if not subject_served(model):
        write_report(day, not_run_report(day, model, base_url, NOT_SUBJECT), number)
        pytest.fail(f"{NOT_SUBJECT}: {model}")
    started = time.monotonic()
    try:
        commit, tree = git("rev-parse", "HEAD"), "dirty" if git("status", "--porcelain") else "clean"
        sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        client = _build_bench_client(MatrixEntry(name=model, provider="vllm", model_id=model, base_url=base_url))
        await client.detect_capabilities()
        assert client.config.tool_call_mode == "native"
        client.config.temperature = TEMPERATURE
        window = probe_served_window(base_url, model)
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        skill_set = install(tmp_path / "cfg")
        cfg = root_config(tmp_path / "cfg")
        shape = loop_shape(tmp_path / "cfg", cfg, base_url, window)
        listing = render_listing(skill_set.skills)
        names = sorted(s.name for s in skill_set.skills)
        sets = prompts()["skills"]
        assert set(sets) == set(names)
        registry, _ = await root_registry(skill_set, cfg)
        tools = sorted(registry.get_tools_for_agent(cfg.name, cfg.division or "", cfg.tools))
        requests = [(skill, kind, text) for skill in names for kind in KINDS for text in sets[skill][kind]]
        assert set(holdout()["skills"]) == set(names)
        held_out = [(skill, kind, text) for skill in names for kind in KINDS for text in holdout()["skills"][skill][kind]]
        engine, reason, calibration = cue_setup()
        total = (RUNS + 1) * len(requests) + RUNS * len(held_out) + (RUNS * (len(requests) + len(held_out)) if engine else 0)
        runs: list[Run] = []

        async def one(leg: str, index: int, skill: str, kind: str, text: str, engine=None) -> None:
            step = await settled(client, skill_set, cfg, shape, text, work, engine)
            runs.append(Run(leg, skill, kind, text, index, step))
            print(f"{leg} {len(runs)}/{total} {skill} {kind} run {index}: {[name for name, _ in step.calls]}"
                  f"{' ' + step.error if step.error else ''}", flush=True)
            stop_on_failing_server(runs)

        creator: list[CreatorRun] = []
        stopped = None
        try:
            for index in range(1, RUNS + 1):
                for request in requests:
                    await one("native", index, *request)
            with xml_mode(client):
                for request in requests:
                    await one("xml", 1, *request)
            for _ in range(CREATOR_BAR[1]):
                with neutral_sitting(fake_home) as (home, creator_work):
                    creator.append(await creator_run(client, home, prompts()["creator"], creator_work, base_url, window))
            for index in range(1, RUNS + 1):
                for request in held_out:
                    await one("holdout", index, *request)
            for leg, chosen in (("cue", requests), ("cue-holdout", held_out)) if engine is not None else ():
                for index in range(1, RUNS + 1):
                    for request in chosen:
                        await one(leg, index, *request, engine=engine)
        except RunStopped as stop:
            stopped = str(stop)
        meta = {
            "day": day, "run": number, "commit": commit, "tree": tree, "sha": sha, "model": model, "base_url": base_url, "window": window,
            "temperature": TEMPERATURE, "effort": shape.effort, "loop_window": shape.window, "tools": tools, "listing": listing,
            "minutes": int((time.monotonic() - started) // 60),
            "cue": {"reason": reason, "embedder": DEFAULT_EMBEDDING_MODEL, "threshold": calibration.threshold if calibration else None,
                    "date": calibration.date if calibration else None,
                    "rows": yaml.safe_load(CALIBRATION.read_text(encoding="utf-8"))[DEFAULT_EMBEDDING_MODEL]["rows"] if calibration else []}}
        path = write_report(day, render_report(meta, runs, creator, stopped), number)
    except BaseException:
        with contextlib.suppress(FileExistsError):
            write_report(day, not_run_report(day, model, base_url, NOT_FINISHED), number)
        raise
    print(f"report: {path}", flush=True)
    assert stopped is None, stopped
    valid = [c.valid for c in creator]
    assert passed(runs, valid), "; ".join(f"{s} {count(runs, 'native', s)}" for s in names) + f"; creator {sum(valid)} of {len(valid)}"
    if reason is None:
        assert cue_false_within_bar(runs), "; ".join(
            f"{s}: cue {count(runs, 'cue', s)[2]}, cue-holdout {count(runs, 'cue-holdout', s)[2]} false" for s in names)

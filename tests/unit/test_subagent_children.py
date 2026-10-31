"""run_config_child: the per-child primitive, its budget, request note and grants, and the time-budget exclusion."""
from __future__ import annotations

import inspect
import time
from types import SimpleNamespace

import pytest

from localharness.agent import subagent
from localharness.agent.context import COMPACT_DISABLED, CompactionPipeline, ContentStore, TokenCounter
from localharness.agent.loop import AgentLoop, BudgetLabels, BudgetTracker, Session
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.subagent import DelegationResult, make_explore_agent_runner
from localharness.config.models import AgentConfig, BudgetConfig, ChildBudget, PermissionConfig, TaskFlowConfig, ToolConfig
from localharness.orchestrator.task_flow import TaskFlow, runner as task_runner
from localharness.orchestrator.task_flow.filer import Docs
from localharness.orchestrator.task_flow.panel import CRITIC_TOOLS, load_critic
from localharness.orchestrator.task_flow.record import limited
from localharness.orchestrator.task_flow.runner import CHILD_SENTENCE, build_executor_config, executor_note, explorer_config
from localharness.tools.capabilities import GrantTargetError
from localharness.tools.builtin.agent_tool import DelegationArgError
from tests.unit.test_capabilities import _builtin_registry


def config_child(name="cfg", add=("read", "grep")) -> AgentConfig:
    return AgentConfig(name=name, role="test child", tools=ToolConfig(add=list(add)),
                       permissions=PermissionConfig(budget=BudgetConfig()))


def capture(llm) -> list[list]:
    seen: list[list] = []
    original = llm.stream_complete

    async def wrapped(messages, *a, **kw):
        seen.append(list(messages))
        return await original(messages, *a, **kw)
    llm.stream_complete = wrapped
    return seen


async def test_a_config_child_runs_with_a_row_budget_and_a_request_note(bus, mock_llm_client):
    R = mock_llm_client.Response
    llm = mock_llm_client([R(content="README.md describes a harness."), R(content="README.md describes a harness.")])
    seen = capture(llm)
    runner = make_explore_agent_runner(
        llm=llm, bus=bus, base_registry=await _builtin_registry(), permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: "sid")
    result = await runner.run_config_child(
        config_child(), "Summarise README.md",
        budget=("task_flow.executor", ChildBudget(actions=24, minutes=25)),
        request_note=lambda: "Step 1/2: summarise README.md")
    first = seen[0]
    assert first[-1]["_lh"] == {"origin": "harness", "subtype": "request_note"}
    assert first[-1]["content"] == "Step 1/2: summarise README.md"
    assert "Budget: 24 actions, 25 minutes." in next(m["content"] for m in first if m["role"] == "user")
    assert all(sum(m.get("_lh", {}).get("subtype") == "request_note" for m in req) == 1 for req in seen)
    assert result.raw == "README.md describes a harness."
    assert result.text.startswith("SUBAGENT RUN COMPLETE")
    assert isinstance(result, DelegationResult)


def test_run_config_child_signature():
    runner = make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(), permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: None)
    params = list(inspect.signature(runner.run_config_child).parameters.values())
    assert [p.name for p in params] == [
        "config", "task", "budget", "request_note", "compact_md_path", "grant_handles", "tools", "shape", "pipeline"]
    assert all(p.default is None for p in params[2:])


class FakeLoop:
    """Records the AgentLoop kwargs and every brief; answers "done" and never calls a model."""

    loops: list["FakeLoop"] = []

    def __init__(self, **kw):
        self.kw = kw
        self.tool_registry = kw["tool_registry"]
        self.context_manager = kw["context_manager"]
        self.briefs: list[str] = []
        self.current_session_id = "child-sid"
        self.last_terminated_reason = None
        FakeLoop.loops.append(self)

    def action_limit(self):
        return 0

    def time_limit(self):
        return 0.0

    async def run_turn(self, brief, *a, **k):
        self.briefs.append(brief)
        return "done"

    @property
    def held(self) -> set[str]:
        return set(self.tool_registry._tools["global"])


@pytest.fixture
def loops(monkeypatch):
    FakeLoop.loops = []
    monkeypatch.setattr("localharness.agent.loop.AgentLoop", FakeLoop)
    return FakeLoop.loops


@pytest.fixture(autouse=True)
def _ledger(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_VERIFICATION_LEDGER_DIR", str(tmp_path / "ledger"))


def make_runner(base, llm=None, **kw):
    return make_explore_agent_runner(
        llm=llm or object(), bus=object(), base_registry=base, permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: "sid", **kw)


@pytest.fixture
async def base():
    return await _builtin_registry()


async def test_a_config_name_loads_once_and_narrows_through_the_config(base, loops, monkeypatch):
    cfg, loaded, narrowed_with = config_child(), [], []
    real = subagent._validated_narrowing

    def spy(name, tools, base_, config, has_agent):
        narrowed_with.append(config)
        return real(name, tools, base_, config, has_agent)

    monkeypatch.setattr(subagent, "_validated_narrowing", spy)
    runner = make_runner(base, load_agent=lambda n: loaded.append(n) or cfg)
    result = await runner("cfg", "t", tools=["read"])
    assert loaded == ["cfg"] and narrowed_with == [cfg]
    assert result.tools == ["read"] and loops[0].held == {"read"}


async def test_a_built_in_with_an_overlay_stays_on_the_built_in_path(base, loops, monkeypatch):
    loaded, overlaid = [], []

    async def no_config_child(*a, **k):  # pragma: no cover - asserted never reached
        raise AssertionError("a built-in must not reach run_config_child")

    monkeypatch.setattr(subagent, "dispatch_config_subagent", no_config_child)
    runner = make_runner(base, load_agent=lambda n: loaded.append(n),
                         load_builtin_override=lambda n, cfg: overlaid.append(n) or cfg)
    result = await runner("explore", "t")
    assert loaded == [] and overlaid == ["explore"]
    assert result.status == "completed" and len(loops) == 1


async def test_the_cruncher_path_is_unchanged(base, monkeypatch):
    seen = {}

    async def crunch(task, **kw):
        seen.update(kw)
        return "crunched"

    monkeypatch.setattr(subagent, "dispatch_cruncher_subagent", crunch)
    loaded = []
    result = await make_runner(base, load_agent=lambda n: loaded.append(n))("cruncher", "t", grant_handles=["h"])
    assert loaded == [] and seen["grant_handles"] == ["h"]
    assert (result.text, result.raw) == ("crunched", "crunched")


async def test_the_root_is_never_loaded_as_a_child(base, loops):
    loaded = []
    runner = make_runner(base, load_agent=lambda n: loaded.append(n))
    with pytest.raises(ValueError, match="not wired"):
        await runner("orchestrator", "t")
    assert loaded == [] and loops == []


async def test_an_unknown_name_reports_where_it_searched(base, loops):
    class NotFound(Exception):
        searched_paths = ["/cfg/agents/ghost.yaml"]

    def load(name):
        raise NotFound(name)

    runner = make_runner(base, load_agent=load)
    with pytest.raises(ValueError, match="searched: '/cfg/agents'"):
        await runner("ghost", "t")
    assert loops == []


async def test_a_load_failure_reports_the_load_failure_text(base, loops):
    def load(name):
        raise RuntimeError("bad yaml")

    runner = make_runner(base, load_agent=load)
    with pytest.raises(ValueError, match="agent 'broken' could not be loaded: bad yaml"):
        await runner("broken", "t")


async def test_tools_narrow_by_bare_name_in_the_callers_order_without_duplicates(base, loops):
    runner = make_runner(base)
    cfg = config_child(add=("read", "grep", "bash_exec"))
    assert (await runner.run_config_child(cfg, "t", tools=["plugin:p.read", "read"])).tools == ["read"]
    assert (await runner.run_config_child(cfg, "t", tools=["grep", "read", "grep"])).tools == ["grep", "read"]
    assert loops[1].held == {"grep", "read"}


async def test_a_name_outside_the_allowlist_is_refused_and_the_base_is_untouched(base, loops):
    runner = make_runner(base)
    cfg = config_child(add=("read", "grep", "bash_exec"))
    with pytest.raises(DelegationArgError, match="cfg has no web_fetch; its tools: read, grep, bash_exec"):
        await runner.run_config_child(cfg, "t", tools=["web_fetch"])
    await runner.run_config_child(cfg, "t", tools=["read"])
    await runner.run_config_child(cfg, "t")
    assert loops[1].held == {"read", "grep", "bash_exec"}
    assert base.has("web_fetch") and base.has("read") and base.has("agent")


async def test_empty_tools_and_empty_grants_mean_no_narrowing_and_no_grant_check(base, loops):
    runner = make_runner(base, parent_store=ContentStore())
    result = await runner.run_config_child(config_child(add=("read", "bash_exec")), "t", tools=[], grant_handles=[])
    assert result.tools is None and loops[0].held == {"read", "bash_exec"}


async def test_a_host_safe_grant_reads_the_parent_handle_through_the_child_store(base, loops):
    parent = ContentStore()
    h = parent.put("granted body")
    other = parent.put("ungranted body")
    runner = make_runner(base, parent_store=parent)
    await runner.run_config_child(config_child(add=("read", "tool_result_get")), "t", grant_handles=[h])
    store = loops[0].context_manager._content_store
    assert store.get(h) == "granted body" and store.get(other) is None


async def test_a_grant_to_a_host_dangerous_set_is_refused_before_dispatch(base, loops):
    runner = make_runner(base, parent_store=ContentStore())
    with pytest.raises(GrantTargetError, match="bash_exec"):
        await runner.run_config_child(config_child(add=("read", "bash_exec")), "t", grant_handles=["h"])
    assert loops == []


async def test_tool_result_get_with_bash_exec_is_refused(base, loops):
    runner = make_runner(base, parent_store=ContentStore())
    with pytest.raises(GrantTargetError):
        await runner.run_config_child(
            config_child(add=("tool_result_get", "bash_exec")), "t", grant_handles=["h"], tools=None)
    assert loops == []


async def test_a_grant_narrowed_to_host_safe_tools_is_granted(base, loops):
    runner = make_runner(base, parent_store=ContentStore())
    await runner.run_config_child(
        config_child(add=("read", "bash_exec")), "t", grant_handles=["h"], tools=["read"])
    assert loops[0].held == {"read"}


async def test_a_grant_to_an_unresolvable_set_is_refused(base, loops):
    class Broken:
        name = "broken"
        effort = None

        @property
        def tools(self):
            raise RuntimeError("unreadable")

    runner = make_runner(base, parent_store=ContentStore())
    with pytest.raises(GrantTargetError, match="could not be resolved"):
        await runner.run_config_child(Broken(), "t", grant_handles=["h"])
    assert loops == []


async def test_the_pipeline_is_the_child_context_managers_and_the_compact_path_is_replaceable(base, loops):
    pipeline = CompactionPipeline(TokenCounter())
    runner = make_runner(base)
    cfg = config_child()
    await runner.run_config_child(cfg, "t", pipeline=pipeline, compact_md_path=COMPACT_DISABLED)
    await runner.run_config_child(cfg, "t")
    assert loops[0].context_manager._pipeline is pipeline
    assert loops[1].context_manager._pipeline is not pipeline
    assert loops[0].kw["compact_md_path"] is COMPACT_DISABLED
    assert loops[1].kw["compact_md_path"] == subagent._child_runtime_paths(cfg)[1]


async def test_no_budget_keeps_the_share_by_shape_with_its_fallback_and_labels(base, loops):
    cfg = config_child()
    runner = make_runner(base, parent_remaining_fn=lambda: (10, 10.0))
    await runner.run_config_child(cfg, "t", shape="review")
    want_cfg, want_fb, want_labels = subagent._child_budget(
        cfg, "default", children_budget=None, shape="review", parent_remaining=lambda: (10, 10.0))
    got = loops[0].kw
    assert got["config"].permissions.budget == want_cfg.permissions.budget
    assert (got["budget_fallback"], got["budget_labels"]) == (want_fb, want_labels)


async def test_a_row_budget_is_written_into_a_copy_with_no_fallback_and_row_labels(base, loops):
    cfg = config_child()
    runner = make_runner(base, parent_remaining_fn=lambda: (10, 10.0))
    await runner.run_config_child(cfg, "t", budget=("task_flow.critic", ChildBudget(actions=3, minutes=1.0)))
    got = loops[0].kw
    assert (got["config"].permissions.budget.max_actions, got["config"].permissions.budget.max_duration_minutes) == (3, 1.0)
    assert cfg.permissions.budget.max_actions is None and cfg.permissions.budget.max_duration_minutes is None
    assert got["budget_fallback"] is None
    assert got["budget_labels"] == BudgetLabels("task_flow.critic", "task_flow.critic", None)


async def test_a_row_budget_binds_the_real_loop_and_names_the_row_in_the_stop(base, loops):
    runner = make_runner(base)
    await runner.run_config_child(
        config_child(), "t", budget=("task_flow.critic", ChildBudget(actions=3, minutes=1.0)))
    loop = AgentLoop(**loops[0].kw)
    assert loop._resolve_max_actions() == (3, "task_flow.critic")
    assert loop._resolve_max_minutes() == (1.0, "task_flow.critic")
    session = type("S", (), {"actions_taken": 3})()
    stop = BudgetTracker(max_actions=loop._resolve_max_actions, max_duration_minutes=0).check(session)
    assert stop.message.endswith("(task_flow.critic)")


async def test_an_explicit_budget_never_lifts_the_unattended_cap(base, loops):
    runner = make_runner(base, unattended_max_actions=2)
    await runner.run_config_child(
        config_child(), "t", budget=("task_flow.executor", ChildBudget(actions=50, minutes=5.0)))
    assert AgentLoop(**loops[0].kw)._resolve_max_actions() == (2, "channels.unattended_max_actions")


async def test_raw_is_the_childs_final_answer_on_each_dispatch_path(base, loops):
    runner = make_runner(base)
    assert (await runner.run_config_child(config_child(), "t")).raw == "done"
    assert (await runner("explore", "t")).raw == "done"


async def test_the_web_researchers_discarded_zero_call_text_never_fills_raw(base, loops):
    result = await make_runner(base)("web-researcher", "t")
    assert result.status == "failed" and result.raw == ""
    assert "SUBAGENT RUN FAILED" in result.text


def test_a_config_childs_allowlist_still_strips_task_agent_and_skill():
    cfg = config_child(add=("read", "task", "agent", "skill", "plugin:p.skill", "mcp:x.agent", "grep"))
    assert subagent._config_child_allowed(cfg) == ["read", "grep"]


async def test_a_nested_runner_carries_run_config_child_too(base, loops, monkeypatch):
    real, built = subagent.make_explore_agent_runner, []
    monkeypatch.setattr(subagent, "make_explore_agent_runner", lambda **kw: built.append(real(**kw)) or built[-1])
    runner = real(
        llm=object(), bus=object(), base_registry=base, permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: "sid", max_subagent_depth=2)
    await runner("web-researcher", "t")
    assert built and all(callable(r.run_config_child) for r in built)


def aged(seconds: float, excluded: float = 0.0) -> Session:
    return Session(agent_id="a", session_id="s", messages=[], start_time=time.monotonic() - seconds,
                   excluded_seconds=excluded)


def test_budget_minutes_subtracts_excluded_seconds_and_elapsed_minutes_does_not():
    session = aged(300, excluded=180)
    assert session.budget_minutes() == pytest.approx(2.0, abs=0.05)
    assert session.elapsed_minutes() == pytest.approx(5.0, abs=0.05)


def test_the_time_check_reads_budget_minutes_and_reports_elapsed_minutes():
    tracker = BudgetTracker(max_actions=0, max_duration_minutes=4)
    assert tracker.check(aged(300, excluded=180)) is None
    violation = tracker.check(aged(300))
    assert violation.reason == "time" and violation.current == pytest.approx(5.0, abs=0.05)


def test_excluded_seconds_defaults_to_zero():
    assert Session(agent_id="a", session_id="s", messages=[]).excluded_seconds == 0.0


class _NoteCtx:
    last_commit = None
    _content_store = None
    max_context_tokens = 0

    def __init__(self, fits_note=True):
        self.fits_note = fits_note
        self._token_counter = type("TC", (), {"estimate_messages": staticmethod(lambda m: 0)})()

    async def build_messages(self, messages, tools):
        from localharness.agent.context import TokenBudget
        return list(messages), TokenBudget(total_limit=100, current_usage=0, tool_schema_tokens=0)

    def ensure_active_references(self, messages, tools=None, *, enforce_budget=False):
        from localharness.agent.context import ActiveReferenceOverflow
        if not self.fits_note and any(m.get("_lh", {}).get("subtype") == "request_note" for m in messages):
            raise ActiveReferenceOverflow("full")
        return messages


def _note_loop(note, ctx):
    loop = object.__new__(AgentLoop)
    loop._request_note, loop._ctx, loop._task_context, loop._llm = note, ctx, None, object()
    return loop


async def test_a_failing_note_provider_sends_the_request_without_a_note():
    def broken():
        raise RuntimeError("boom")
    packed, _ = await _note_loop(broken, _NoteCtx())._build_request([{"role": "user", "content": "hi"}], None)
    assert [m["content"] for m in packed] == ["hi"]


async def test_the_note_provider_is_called_once_per_request():
    calls = []
    loop = _note_loop(lambda: calls.append(1) or "step 1", _NoteCtx(fits_note=False))
    packed, _ = await loop._build_request([{"role": "user", "content": "hi"}], None)
    assert len(calls) == 1
    assert [m["content"] for m in packed] == ["hi"], "a note that cannot fit is dropped, the turn is not blocked"


def executor_flow(root_tools=("read", "write", "edit", "bash_exec", "tool_result_get", "agent", "task", "skill")):
    config = AgentConfig(name="root", role="r")
    return SimpleNamespace(token_counter=None, llm=None, config=config, root_tools=list(root_tools),
                           limits=limited(config.task_flow)[0])


async def test_executor_runs_through_the_real_child_with_its_row_pipeline_note_and_root_less_toolset(base, loops):
    flow = executor_flow()
    config, pipeline, budget = build_executor_config(flow)
    runner = make_runner(base)
    result = await runner.run_config_child(
        config, "brief", budget=budget, request_note=executor_note("step 1/2 · x"),
        compact_md_path=COMPACT_DISABLED, pipeline=pipeline)
    loop = loops[0]
    assert loop.kw["compact_md_path"] is COMPACT_DISABLED
    assert loop.context_manager._pipeline is pipeline and isinstance(pipeline, CompactionPipeline)
    assert loop.kw["budget_labels"] == BudgetLabels("task_flow.executor", "task_flow.executor", None)
    assert loop.kw["request_note"]().startswith(CHILD_SENTENCE) and "step 1/2" in loop.kw["request_note"]()
    assert loop.held == {"read", "write", "edit", "bash_exec"} and result.tools is None
    assert (loop.kw["config"].permissions.budget.max_actions, loop.kw["config"].permissions.budget.max_duration_minutes) == (24, 25)


def test_executor_config_denies_tool_result_get_and_the_strip_set_removes_the_control_tools():
    config, pipeline, (row, budget) = build_executor_config(executor_flow())
    assert config.tools.deny == ["tool_result_get"] and config.tools.add == ["read", "write", "edit", "bash_exec"]
    assert subagent._config_child_allowed(config) == ["read", "write", "edit", "bash_exec"]
    assert "tool_result_get" in build_executor_config(executor_flow(("read", "tool_result_get")))[0].tools.deny
    assert (row, budget) == ("task_flow.executor", ChildBudget(actions=24, minutes=25))
    assert pipeline._llm_stages[0].compact_md_path is COMPACT_DISABLED


async def test_executor_holding_a_handle_is_refused_when_the_host_set_is_dangerous(base, loops):
    config, pipeline, budget = build_executor_config(executor_flow(("read", "bash_exec")))
    runner = make_runner(base, parent_store=ContentStore())
    with pytest.raises(GrantTargetError, match="bash_exec"):
        await runner.run_config_child(config, "brief", budget=budget, grant_handles=["h"])
    assert loops == []


async def test_explorer_holds_exactly_read_glob_and_grep_with_its_row(base, loops):
    runner = make_runner(base)
    await runner.run_config_child(explorer_config(), "q", budget=("task_flow.explorer", ChildBudget(actions=12, minutes=6.0)))
    loop = loops[0]
    assert loop.held == {"read", "glob", "grep"}
    assert loop.kw["budget_labels"] == BudgetLabels("task_flow.explorer", "task_flow.explorer", None)


async def test_critic_factual_is_granted_a_handle_because_its_toolset_is_host_safe(base, loops):
    parent = ContentStore()
    handle = parent.put("granted body")
    runner = make_runner(base, parent_store=parent)
    await runner.run_config_child(load_critic("factual"), "t", grant_handles=[handle],
                                  budget=("task_flow.critic", ChildBudget(actions=12, minutes=6.0)))
    loop = loops[0]
    assert {"read", "glob", "grep", "tool_result_get"} <= loop.held <= set(CRITIC_TOOLS)
    assert loop.context_manager._content_store.get(handle) == "granted body"


async def test_a_critic_config_with_a_dangerous_tool_and_tool_result_get_is_refused_with_a_handle(base, loops):
    runner = make_runner(base, parent_store=ContentStore())
    with pytest.raises(GrantTargetError):
        await runner.run_config_child(config_child(add=(*CRITIC_TOOLS, "bash_exec")), "t", grant_handles=["h"])
    assert loops == []


async def test_exclude_through_a_real_session_is_clamped_to_one_flow_cap_per_turn(tmp_path, monkeypatch):
    session = aged(3000)
    root = tmp_path / ".localharness"
    root.mkdir()
    flow = TaskFlow(
        run_config_child=None, emit=lambda *row: None, docs=Docs(root, True, "a"),
        exclude=lambda s: setattr(session, "excluded_seconds", session.excluded_seconds + s),
        config=AgentConfig(name="root", role="r", task_flow=TaskFlowConfig(flow_cap_minutes=10.0)), root_tools=[])
    flow.open("fix login", 1)
    ticks = iter([0.0, 400.0, 0.0, 400.0, 0.0, 400.0])
    monkeypatch.setattr(task_runner, "monotonic", lambda: next(ticks))
    for _ in range(3):
        await flow.run("task")
    assert session.excluded_seconds == 600.0
    assert session.budget_minutes() == pytest.approx(session.elapsed_minutes() - 10.0, abs=0.01)
    assert session.elapsed_minutes() == pytest.approx(50.0, abs=0.05)

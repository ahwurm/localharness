"""The action limit follows the live surface: an attended terminal root runs unbounded, a remote
unattended surface is capped by the machine knob (and a child by its fallback), and the failure
text names the key that bound."""
from __future__ import annotations

import pytest

from localharness.agent.loop import BudgetLabels, BudgetTracker, Session, _budget_note, _cap_limit


def _session(actions: int = 0) -> Session:
    s = Session(agent_id="a", session_id="s", messages=[])
    s.actions_taken = actions
    return s


# --------------------------------------------------------------------------- _cap_limit


@pytest.mark.parametrize(
    ("limit", "cap", "expected"),
    [(0, 8, 8), (16, 8, 8), (5, 8, 5), (8, 8, 8)],
)
def test_cap_limit_treats_zero_as_no_limit(limit, cap, expected):
    assert _cap_limit(limit, cap) == expected


def test_budget_labels_default_to_nothing():
    labels = BudgetLabels()
    assert (labels.actions, labels.minutes, labels.fallback) == (None, None, None)


# --------------------------------------------------------------------------- the tracker


def test_an_int_tracker_behaves_as_before():
    t = BudgetTracker(max_actions=5, max_duration_minutes=30.0)
    assert t.limit == 5 and t.bound_by is None
    assert t.check(_session(4)) is None
    v = t.check(_session(5))
    assert v is not None and v.reason == "actions" and v.limit == 5 and v.current == 5
    assert v.message == "Agent reached the maximum of 5 tool calls. Stopping to prevent runaway execution."


def test_a_labelled_supplier_names_the_key_that_bound():
    t = BudgetTracker(max_actions=lambda: (100, "channels.unattended_max_actions"), max_duration_minutes=0.0)
    assert t.limit == 100 and t.bound_by == "channels.unattended_max_actions"
    v = t.check(_session(100))
    assert v is not None
    assert v.message == (
        "Agent reached the maximum of 100 tool calls. Stopping to prevent runaway execution. "
        "(channels.unattended_max_actions)"
    )


def test_a_supplier_of_zero_never_trips():
    t = BudgetTracker(max_actions=lambda: (0, None), max_duration_minutes=0.0)
    assert t.check(_session(10_000)) is None


def test_the_supplier_is_re_read_on_every_check_and_once_per_check():
    reads: list[int] = []
    limits = iter([(0, None), (3, "k"), (3, "k")])

    def supplier():
        reads.append(1)
        return next(limits)

    t = BudgetTracker(max_actions=supplier, max_duration_minutes=0.0)
    assert t.check(_session(5)) is None
    assert len(reads) == 1
    v = t.check(_session(5))
    assert v is not None and v.limit == 3 and v.message.endswith("(k)")
    assert len(reads) == 2


def _old_session(minutes: float) -> Session:
    s = _session(2)
    s.start_time -= minutes * 60.0
    return s


def test_the_time_message_carries_its_label_when_one_is_set():
    labelled = BudgetTracker(max_actions=0, max_duration_minutes=1.0, minutes_bound_by="org.children_budget.default")
    v = labelled.check(_old_session(2.0))
    assert v is not None and v.reason == "time"
    assert v.message == (
        "Agent reached the time limit of 1.0 minutes. Stopping after 2 tool calls. (org.children_budget.default)"
    )
    plain = BudgetTracker(max_actions=0, max_duration_minutes=1.0).check(_old_session(2.0))
    assert plain is not None
    assert plain.message == "Agent reached the time limit of 1.0 minutes. Stopping after 2 tool calls."


def test_budget_note_reads_a_live_limit_and_is_silent_without_one():
    live = BudgetTracker(max_actions=lambda: (12, "k"), max_duration_minutes=0)
    assert "3/12 tool calls used" in _budget_note(_session(3), live)
    assert _budget_note(_session(3), BudgetTracker(max_actions=lambda: (0, None), max_duration_minutes=0)) == ""


# =========================================================================== the loop's resolvers

from types import SimpleNamespace  # noqa: E402

from localharness.agent.context import ContextManager  # noqa: E402
from localharness.agent.loop import AgentLoop  # noqa: E402
from localharness.agent.permissions import PermissionEvaluator  # noqa: E402
from localharness.config.models import (  # noqa: E402
    AgentConfig,
    BudgetConfig,
    ChildBudget,
    PermissionConfig,
)
from localharness.core.events import TurnFailed, TurnStarted  # noqa: E402

KNOB_KEY = "channels.unattended_max_actions"


def _channel(*, local: bool) -> SimpleNamespace:
    """The few attributes a gate reads off a channel (it asks nothing: `can_ask` is False)."""
    return SimpleNamespace(channel_id="terminal" if local else "phone", local_operator=local, can_ask=False)


class _Registry:
    """Runs every call, with an output that differs per call (so nothing reads as stuck)."""

    def __init__(self, on_call=None):
        self.calls: list[dict] = []
        self._on_call = on_call

    def get_tools_for_agent(self, agent_id, division_id, tool_config):
        return {}

    async def dispatch(self, name, arguments, agent_id, division_id, tool_config, **kw):
        from localharness.tools.base import ToolResult

        self.calls.append(dict(arguments))
        if self._on_call is not None:
            self._on_call(len(self.calls))
        return ToolResult(output=f"ran {arguments}", success=True)


def _calls(mock_llm_client, n: int, *, per_iteration: int = 1) -> list:
    """`n` iterations of distinct tool calls, then a plain answer."""
    Response, ToolCall = mock_llm_client.Response, mock_llm_client.ToolCall
    out = []
    for i in range(n):
        tcs = [
            ToolCall(id=f"tc-{i}-{j}", name="bash", arguments={"cmd": f"echo {i}-{j}"})
            for j in range(per_iteration)
        ]
        out.append(Response(content=None, tool_calls=tcs))
    out.append(Response(content="all done"))
    return out


def _loop(
    mock_llm_client,
    bus,
    responses,
    *,
    max_actions=None,
    max_minutes=None,
    mode="unattended",
    local: bool | None = None,
    registry=None,
    **kw,
) -> AgentLoop:
    cfg = AgentConfig(
        name="budget-agent",
        role="Test.",
        permissions=PermissionConfig(
            mode=mode,
            deny_patterns=[],
            budget=BudgetConfig(max_actions=max_actions, max_duration_minutes=max_minutes),
        ),
    )
    loop = AgentLoop(
        config=cfg,
        llm=mock_llm_client(responses),
        bus=bus,
        context_manager=ContextManager(),
        tool_registry=registry if registry is not None else _Registry(),
        permission_evaluator=PermissionEvaluator(),
        **kw,
    )
    if local is not None:
        loop.gate.attach_channel(_channel(local=local))
    return loop


# --------------------------------------------------------------------------- attended root


def test_an_attended_root_has_no_action_limit_even_in_unattended_mode(mock_llm_client, bus):
    loop = _loop(mock_llm_client, bus, [], mode="unattended", local=True, unattended_max_actions=100)
    assert loop.gate.mode == "unattended"
    assert loop.action_limit() == 0
    assert loop.time_limit() == 0.0


@pytest.mark.asyncio
async def test_an_attended_root_runs_past_100_actions_with_no_budget_stop(mock_llm_client, bus):
    registry = _Registry()
    loop = _loop(
        mock_llm_client, bus, _calls(mock_llm_client, 105),
        local=True, unattended_max_actions=100, registry=registry,
    )
    summary = await loop.run_turn("go")
    assert len(registry.calls) == 105
    assert "all done" in summary
    assert not [e for e in bus.history(event_types=[TurnFailed]) if e.reason == "budget_exceeded"]
    started = bus.history(event_types=[TurnStarted])
    assert started[0].budget.max_actions == 0
    assert started[0].budget.max_duration_minutes == 0.0


# --------------------------------------------------------------------------- remote unattended


def test_a_remote_unattended_root_is_capped_by_the_knob(mock_llm_client, bus):
    loop = _loop(
        mock_llm_client, bus, [], max_actions=500, local=False, unattended_max_actions=100,
    )
    assert loop.action_limit() == 100


def test_the_knob_caps_the_default_root_whose_configured_limit_is_none(mock_llm_client, bus):
    """The knob's primary case: a literal min(0, cap) would be 0, which is unbounded."""
    loop = _loop(mock_llm_client, bus, [], max_actions=None, local=False, unattended_max_actions=100)
    assert loop.action_limit() == 100


def test_a_configured_limit_under_the_knob_is_kept(mock_llm_client, bus):
    loop = _loop(mock_llm_client, bus, [], max_actions=40, local=False, unattended_max_actions=100)
    assert loop.action_limit() == 40
    assert loop._resolve_max_actions() == (40, None)


def test_without_a_knob_the_remote_root_runs_on_its_configured_value(mock_llm_client, bus):
    assert _loop(mock_llm_client, bus, [], max_actions=500, local=False).action_limit() == 500
    assert _loop(mock_llm_client, bus, [], max_actions=None, local=False).action_limit() == 0


# WR-70 (owner ruling, departs from D-07's `mode == "unattended"`): with no local operator the knob
# applies in every mode. This test used to pin `auto` + remote as uncapped (0).
@pytest.mark.parametrize("mode", ["auto", "trusted", "guarded", "read-only", "unattended"])
def test_a_gate_with_no_local_operator_is_capped_in_every_mode(mock_llm_client, bus, mode):
    loop = _loop(mock_llm_client, bus, [], max_actions=None, mode=mode, local=False,
                 unattended_max_actions=100)
    assert loop._resolve_max_actions() == (100, "channels.unattended_max_actions")


@pytest.mark.parametrize("mode", ["auto", "trusted", "guarded", "read-only", "unattended"])
def test_a_local_operator_is_never_capped_in_any_mode(mock_llm_client, bus, mode):
    loop = _loop(mock_llm_client, bus, [], max_actions=None, mode=mode, local=True,
                 unattended_max_actions=100, budget_fallback=ChildBudget(actions=8, minutes=3.0))
    assert (loop.action_limit(), loop.time_limit()) == (0, 0.0)


def test_the_fallback_minutes_follow_the_same_rule_in_auto(mock_llm_client, bus):
    """The time limit keys on the same predicate as the knob: no local operator, any mode."""
    loop = _loop(mock_llm_client, bus, [], max_actions=None, mode="auto", local=False,
                 budget_fallback=ChildBudget(actions=8, minutes=3.0),
                 budget_labels=BudgetLabels(fallback="org.children_budget.default"))
    assert loop._resolve_max_minutes() == (3.0, "org.children_budget.default")
    assert loop._resolve_max_actions() == (8, "org.children_budget.default")


@pytest.mark.asyncio
async def test_the_failure_names_the_machine_knob(mock_llm_client, bus):
    registry = _Registry()
    loop = _loop(
        mock_llm_client, bus, _calls(mock_llm_client, 150),
        max_actions=500, local=False, unattended_max_actions=100, registry=registry,
    )
    summary = await loop.run_turn("go")
    assert len(registry.calls) == 100
    failed = [e for e in bus.history(event_types=[TurnFailed]) if e.reason == "budget_exceeded"]
    assert failed
    assert "(channels.unattended_max_actions)" in summary
    assert "Agent reached the maximum of 100 tool calls." in summary


# --------------------------------------------------------------------------- the fallback rung


def test_the_fallback_caps_a_remote_unattended_child(mock_llm_client, bus):
    loop = _loop(
        mock_llm_client, bus, [], max_actions=None, local=False, unattended_max_actions=100,
        budget_fallback=ChildBudget(actions=8, minutes=3.0),
        budget_labels=BudgetLabels(fallback="org.children_budget.default"),
    )
    assert loop._resolve_max_actions() == (8, "org.children_budget.default")
    assert loop.action_limit() == 8
    assert loop.time_limit() == 3.0


def test_the_fallback_is_not_applied_on_an_attended_gate(mock_llm_client, bus):
    loop = _loop(
        mock_llm_client, bus, [], max_actions=None, local=True,
        budget_fallback=ChildBudget(actions=8, minutes=3.0),
        budget_labels=BudgetLabels(fallback="org.children_budget.default"),
    )
    assert loop.action_limit() == 0
    assert loop.time_limit() == 0.0


def test_an_equal_value_keeps_the_earlier_label(mock_llm_client, bus):
    loop = _loop(
        mock_llm_client, bus, [], max_actions=16, max_minutes=12.0, local=False,
        budget_fallback=ChildBudget(actions=16, minutes=12.0),
        budget_labels=BudgetLabels(actions="agent.permissions.budget.max_actions",
                                   minutes="agent.permissions.budget.max_duration_minutes",
                                   fallback="org.children_budget.default"),
    )
    assert loop._resolve_max_actions() == (16, "agent.permissions.budget.max_actions")
    assert loop._resolve_max_minutes() == (12.0, "agent.permissions.budget.max_duration_minutes")


def test_a_configured_limit_over_the_fallback_is_lowered_to_it(mock_llm_client, bus):
    loop = _loop(
        mock_llm_client, bus, [], max_actions=40, max_minutes=12.0, local=False,
        budget_fallback=ChildBudget(actions=8, minutes=3.0),
        budget_labels=BudgetLabels(fallback="org.children_budget.default"),
    )
    assert loop._resolve_max_actions() == (8, "org.children_budget.default")
    assert loop._resolve_max_minutes() == (3.0, "org.children_budget.default")


def test_the_knob_beats_a_looser_fallback_and_names_itself(mock_llm_client, bus):
    loop = _loop(
        mock_llm_client, bus, [], max_actions=None, local=False, unattended_max_actions=5,
        budget_fallback=ChildBudget(actions=8, minutes=3.0),
        budget_labels=BudgetLabels(fallback="org.children_budget.default"),
    )
    assert loop._resolve_max_actions() == (5, KNOB_KEY)


# --------------------------------------------------------------------------- the live gate


@pytest.mark.asyncio
async def test_a_flip_to_unattended_mid_turn_caps_from_the_next_check(mock_llm_client, bus):
    holder: dict = {}

    def flip(call_number: int) -> None:
        if call_number == 1:  # while the first iteration's two parallel calls are in flight
            holder["loop"].gate.attach_channel(_channel(local=False))

    registry = _Registry(on_call=flip)
    loop = _loop(
        mock_llm_client, bus, _calls(mock_llm_client, 10, per_iteration=2),
        local=True, unattended_max_actions=2, registry=registry,
    )
    holder["loop"] = loop
    summary = await loop.run_turn("go")
    assert len(registry.calls) == 2  # the iteration in flight completed both calls...
    failed = [e for e in bus.history(event_types=[TurnFailed]) if e.reason == "budget_exceeded"]
    assert failed  # ...and the very next check stopped the turn
    assert "(channels.unattended_max_actions)" in summary


@pytest.mark.asyncio
async def test_leaving_unattended_mid_turn_uncaps_from_the_next_check(mock_llm_client, bus):
    holder: dict = {}

    def attach_operator(call_number: int) -> None:
        if call_number == 2:
            holder["loop"].gate.attach_channel(_channel(local=True))

    registry = _Registry(on_call=attach_operator)
    loop = _loop(
        mock_llm_client, bus, _calls(mock_llm_client, 6),
        local=False, unattended_max_actions=2, registry=registry,
    )
    holder["loop"] = loop
    summary = await loop.run_turn("go")
    assert len(registry.calls) == 6
    assert "all done" in summary
    assert not [e for e in bus.history(event_types=[TurnFailed]) if e.reason == "budget_exceeded"]


def test_the_limit_follows_the_operator_both_ways_and_not_the_mode(mock_llm_client, bus):
    # WR-70: this used to pin the limit following the gate MODE (auto 0, unattended 7). It now
    # follows whether a local operator is attached; switching the mode changes nothing.
    loop = _loop(mock_llm_client, bus, [], max_actions=None, mode="auto", local=False,
                 unattended_max_actions=7)
    assert loop.action_limit() == 7
    loop.gate.mode = "unattended"
    assert loop.action_limit() == 7
    loop.gate.attach_channel(_channel(local=True))
    assert loop.action_limit() == 0
    loop.gate.mode = "auto"
    assert loop.action_limit() == 0
    loop.gate.attach_channel(_channel(local=False))
    assert loop.action_limit() == 7


# --------------------------------------------------------------------------- remaining_budget


def test_remaining_budget_is_zero_before_any_turn(mock_llm_client, bus):
    loop = _loop(mock_llm_client, bus, [], max_actions=10, max_minutes=5.0, local=True)
    assert loop.remaining_budget() == (0, 0.0)


@pytest.mark.asyncio
async def test_remaining_budget_during_a_turn_and_cleared_after_it(mock_llm_client, bus):
    seen: list[tuple[int, float]] = []
    holder: dict = {}

    def probe(call_number: int) -> None:
        if call_number == 3:
            seen.append(holder["loop"].remaining_budget())

    registry = _Registry(on_call=probe)
    loop = _loop(
        mock_llm_client, bus, _calls(mock_llm_client, 4),
        max_actions=10, max_minutes=5.0, local=True, registry=registry,
    )
    holder["loop"] = loop
    await loop.run_turn("go")
    (actions, minutes), = seen
    assert actions == 7  # 10 - 3: the call being run is already counted
    assert 0.1 <= minutes <= 5.0
    assert loop.remaining_budget() == (0, 0.0)


@pytest.mark.asyncio
async def test_remaining_budget_is_zero_for_a_key_with_no_limit(mock_llm_client, bus):
    seen: list[tuple[int, float]] = []
    holder: dict = {}

    def probe(call_number: int) -> None:
        seen.append(holder["loop"].remaining_budget())

    loop = _loop(mock_llm_client, bus, _calls(mock_llm_client, 1), local=True,
                 registry=_Registry(on_call=probe))
    holder["loop"] = loop
    await loop.run_turn("go")
    assert seen == [(0, 0.0)]


@pytest.mark.asyncio
async def test_a_finite_remainder_is_floored(mock_llm_client, bus):
    seen: list[tuple[int, float]] = []
    holder: dict = {}

    def probe(call_number: int) -> None:
        seen.append(holder["loop"].remaining_budget())

    loop = _loop(mock_llm_client, bus, _calls(mock_llm_client, 3), max_actions=2, max_minutes=0.1,
                 local=True, registry=_Registry(on_call=probe))
    holder["loop"] = loop
    await loop.run_turn("go")
    assert seen
    assert all(a >= 1 and m >= 0.1 for a, m in seen)


@pytest.mark.asyncio
async def test_the_live_turn_is_cleared_when_the_turn_raises(mock_llm_client, bus):
    class Boom:
        config = SimpleNamespace(tool_call_mode="native", context_window=128_000)

        async def stream_complete(self, *a, **kw):
            raise KeyboardInterrupt  # not caught by the loop's `except Exception`

    loop = _loop(mock_llm_client, bus, [], max_actions=10, local=True)
    loop._llm = Boom()
    with pytest.raises(BaseException):
        await loop.run_turn("go")
    assert loop._live_turn is None
    assert loop.remaining_budget() == (0, 0.0)


# --------------------------------------------------------------------------- the reconnect clamp


def test_the_reconnect_clamp_reads_the_effective_time_limit(mock_llm_client, bus):
    """A remote unattended child's fallback minutes bound its reconnect wait; the attended root's is
    unbounded by the budget."""
    llm_cfg = SimpleNamespace(reconnect_wait_seconds=600.0, base_url="http://127.0.0.1:1/v1",
                              tool_call_mode="native", context_window=128_000)
    child = _loop(mock_llm_client, bus, [], max_minutes=None, local=False,
                  budget_fallback=ChildBudget(actions=8, minutes=3.0))
    child._llm = SimpleNamespace(config=llm_cfg)
    assert child._reconnect_bound(Session(agent_id="a", session_id="s", messages=[])) <= 180.0

    root = _loop(mock_llm_client, bus, [], max_minutes=None, local=True)
    root._llm = SimpleNamespace(config=llm_cfg)
    assert root._reconnect_bound(Session(agent_id="a", session_id="s", messages=[])) == 600.0


# =========================================================================== the child's brief


async def _briefed_task(monkeypatch, bus, agent_config, *, mode="auto", **dispatch_kw) -> str:
    """Dispatch a config child through the real loop and return the brief its turn received."""
    from localharness.agent.gate import fail_closed_gate
    from localharness.agent.subagent import dispatch_config_subagent
    from localharness.tools import ToolRegistry
    from localharness.tools.builtin import register_builtin_tools

    seen: list[str] = []

    async def record_turn(self, task, *a, **k):
        seen.append(task)
        return "done"

    monkeypatch.setattr(AgentLoop, "run_turn", record_turn)
    base = ToolRegistry()
    await register_builtin_tools(base)
    gate = fail_closed_gate(permissions=SimpleNamespace(mode=mode))
    gate.attach_channel(_channel(local=True))
    await dispatch_config_subagent(
        "find it",
        agent_config=agent_config,
        llm=SimpleNamespace(config=SimpleNamespace(tool_call_mode="native", context_window=128_000)),
        bus=bus,
        base_registry=base,
        parent_session_id="parent",
        permission_evaluator=PermissionEvaluator(),
        gate=gate,
        **dispatch_kw,
    )
    (task,) = seen
    return task


def _child(*, tools_add=None, **budget) -> AgentConfig:
    from localharness.config.models import ToolConfig

    kw = {"tools": ToolConfig(add=tools_add)} if tools_add else {}
    return AgentConfig(
        name="yaml-child", role="a child",
        permissions=PermissionConfig(budget=BudgetConfig(**budget)),
        **kw,
    )


@pytest.mark.asyncio
async def test_a_child_with_no_budget_is_briefed_with_no_limits(monkeypatch, bus):
    task = await _briefed_task(monkeypatch, bus, _child(tools_add=["glob"]))
    assert task.startswith("(Your ONLY available tools: glob.")
    assert "Budget: no action limit, no time limit." in task


@pytest.mark.asyncio
async def test_a_child_with_a_budget_is_briefed_with_its_own_numbers(monkeypatch, bus):
    task = await _briefed_task(
        monkeypatch, bus, _child(tools_add=["glob"], max_actions=16, max_duration_minutes=12.0)
    )
    assert "Budget: 16 actions, 12 minutes." in task


@pytest.mark.asyncio
async def test_the_toolset_line_names_only_tools_the_child_registry_holds(monkeypatch, bus):
    task = await _briefed_task(monkeypatch, bus, _child(tools_add=["glob", "nonexistent_tool"]))
    first_line = task.split("\n", 1)[0]
    assert "glob" in first_line
    assert "nonexistent_tool" not in task


@pytest.mark.asyncio
async def test_a_toolset_nothing_of_which_is_held_says_none(monkeypatch, bus):
    task = await _briefed_task(monkeypatch, bus, _child(tools_add=["nonexistent_tool"]))
    assert task.startswith("(Your ONLY available tools: none.")


@pytest.mark.asyncio
async def test_the_brief_prints_the_limits_the_loop_will_enforce(monkeypatch, bus):
    built: list[AgentLoop] = []
    real_init = AgentLoop.__init__

    def spy(self, *a, **k):
        real_init(self, *a, **k)
        built.append(self)

    monkeypatch.setattr(AgentLoop, "__init__", spy)
    task = await _briefed_task(
        monkeypatch, bus, _child(tools_add=["glob"], max_actions=16, max_duration_minutes=12.0),
    )
    (child,) = built
    assert f"Budget: {child.action_limit()} actions, {child.time_limit():g} minutes." in task


@pytest.mark.asyncio
async def test_the_chunk_summarizer_leaf_brief_names_only_held_tools(monkeypatch, bus):
    from localharness.agent import subagent
    from localharness.agent.context import ContentStore
    from localharness.tools import ToolRegistry
    from localharness.tools.builtin import register_builtin_tools

    seen: list[str] = []

    async def record_turn(self, task, *a, **k):
        seen.append(task)
        return "NONE"

    monkeypatch.setattr(AgentLoop, "run_turn", record_turn)
    monkeypatch.setattr(subagent, "CHUNK_SUMMARIZER_TOOLS", ["tool_result_get", "nonexistent_tool"])
    store = ContentStore()
    handle = store.put("a granted section")
    base = ToolRegistry()
    await register_builtin_tools(base)
    from localharness.tools.builtin.tool_result_get_tool import ToolResultGetTool

    await base.register(ToolResultGetTool(store), scope="global")  # the root registry holds it in a real run
    await subagent._run_chunk_summarizer(
        handle,
        "what?",
        store,
        llm=SimpleNamespace(config=SimpleNamespace(tool_call_mode="native", context_window=128_000)),
        bus=bus,
        base_registry=base,
        parent_session_id="parent",
        permission_evaluator=PermissionEvaluator(),
        token_counter=None,
        max_context_tokens=None,
        depth=0,
        max_subagent_depth=3,
    )
    (task,) = seen
    assert task.startswith("(Your ONLY available tools: tool_result_get.")
    assert "nonexistent_tool" not in task


# --------------------------------------------------------------------------- the real surfaces (Nyquist)
# The fakes above set `local_operator` by hand. These attach the shipped channel classes themselves,
# so a remote channel that ever declared itself local (or a new one that forgot the flag's default)
# would let an unbounded root run unattended on it.

from localharness.channels.acp import AcpChannel  # noqa: E402
from localharness.channels.base import ChannelAdapter  # noqa: E402
from localharness.channels.mobile.channel import MobileChannel  # noqa: E402
from localharness.channels.terminal import TerminalChannel  # noqa: E402
from localharness.dispatch.channel import DispatchChannel  # noqa: E402


class _BareRemote(ChannelAdapter):
    """A channel that says nothing about where its person sits."""

    channel_id = "future-chat"


_BareRemote.__abstractmethods__ = frozenset()  # never driven: only its declared flags are read


def _bare(cls, channel_id: str):
    """An instance of the real class without its I/O constructor; the gate reads class flags."""
    ch = object.__new__(cls)
    try:
        ch.channel_id = channel_id
    except AttributeError:
        pass
    return ch


@pytest.mark.parametrize(
    ("cls", "channel_id"),
    [(MobileChannel, "mobile"), (DispatchChannel, "discord"), (_BareRemote, "future-chat")],
)
def test_an_unbounded_root_on_a_real_remote_channel_is_capped_when_unattended(
    mock_llm_client, bus, cls, channel_id,
):
    assert cls.local_operator is False
    loop = _loop(mock_llm_client, bus, [], max_actions=None, mode="unattended",
                 unattended_max_actions=100)
    loop.gate.attach_channel(_bare(cls, channel_id))
    assert loop.gate.local_operator is False
    assert loop.action_limit() == 100
    assert loop._resolve_max_actions() == (100, "channels.unattended_max_actions")


@pytest.mark.parametrize(("cls", "channel_id"), [(TerminalChannel, "terminal"), (AcpChannel, "acp")])
def test_the_terminal_and_zed_are_local_operators_and_run_uncapped(mock_llm_client, bus, cls, channel_id):
    assert cls.local_operator is True
    loop = _loop(mock_llm_client, bus, [], max_actions=None, mode="unattended",
                 unattended_max_actions=100)
    loop.gate.attach_channel(_bare(cls, channel_id))
    assert loop.gate.local_operator is True
    assert loop.action_limit() == 0


def test_an_unattended_loop_with_no_channel_attached_is_capped(mock_llm_client, bus):
    """No channel at all (nothing attached yet) reads as remote: the knob holds."""
    loop = _loop(mock_llm_client, bus, [], max_actions=None, mode="unattended",
                 unattended_max_actions=100)
    assert loop.gate.local_operator is False
    assert loop.action_limit() == 100


@pytest.mark.asyncio
async def test_a_remote_unattended_root_stops_at_the_knob_on_a_real_mobile_channel(mock_llm_client, bus):
    registry = _Registry()
    loop = _loop(mock_llm_client, bus, _calls(mock_llm_client, 12), max_actions=None,
                 mode="unattended", unattended_max_actions=5, registry=registry)
    loop.gate.attach_channel(_bare(MobileChannel, "mobile"))
    await loop.run_turn("go")
    assert len(registry.calls) <= 5
    failed = [e for e in bus.history(event_types=[TurnFailed]) if e.reason == "budget_exceeded"]
    assert len(failed) == 1 and "channels.unattended_max_actions" in failed[0].detail


# --------------------------------------------------- CR-52: a headless terminal is unattended


class _TtyStdin:
    def isatty(self) -> bool:
        return True


def _terminal(tmp_path, monkeypatch, *, tty: bool, commit: bool = False, no_input: bool = False):
    import io

    monkeypatch.setattr("sys.stdin", _TtyStdin() if tty else io.StringIO("go\n"))
    if commit:
        monkeypatch.setenv("LOCALHARNESS_COMMIT", "abc123")
    else:
        monkeypatch.delenv("LOCALHARNESS_COMMIT", raising=False)
    return TerminalChannel(bus=None, config={}, history_file=str(tmp_path / ".h"), no_input=no_input)


@pytest.mark.parametrize(("tty", "commit", "no_input"), [
    (True, False, True),    # cron / CI: localharness start --no-input
    (False, False, False),  # piped stdin
    (True, True, False),    # a start the model launched through bash_exec
])
@pytest.mark.parametrize("mode", ["unattended", "auto", "trusted"])  # WR-70: every mode
def test_a_headless_terminal_root_runs_under_the_knob_when_unattended(
        mock_llm_client, bus, tmp_path, monkeypatch, tty, commit, no_input, mode):
    loop = _loop(mock_llm_client, bus, [], max_actions=None, mode=mode,
                 unattended_max_actions=100)
    loop.gate.attach_channel(_terminal(tmp_path, monkeypatch, tty=tty, commit=commit,
                                       no_input=no_input))
    assert loop.gate.local_operator is False
    assert loop.action_limit() == 100
    assert loop._resolve_max_actions() == (100, "channels.unattended_max_actions")


@pytest.mark.parametrize("mode", ["unattended", "auto", "trusted"])
def test_an_interactive_terminal_root_stays_uncapped(mock_llm_client, bus, tmp_path, monkeypatch, mode):
    loop = _loop(mock_llm_client, bus, [], max_actions=None, mode=mode,
                 unattended_max_actions=100)
    loop.gate.attach_channel(_terminal(tmp_path, monkeypatch, tty=True))
    assert loop.gate.local_operator is True
    assert loop.action_limit() == 0


@pytest.mark.asyncio
async def test_start_with_no_input_builds_a_terminal_that_is_no_local_operator(tmp_path, monkeypatch):
    """The real start path passes --no-input to the channel: even with a keyboard attached, a
    `--no-input` session answers to nobody, so the gate does not see a local operator."""
    from localharness.cli.start_cmd import _start_async
    from tests.unit.test_start_cmd import _stub_start_boundaries

    seen: dict = {}

    async def repl_run(self):
        seen["channel"] = self._channel.local_operator
        seen["gate"] = self._agent.gate.local_operator

    _stub_start_boundaries(tmp_path, monkeypatch, repl_run=repl_run)
    monkeypatch.setattr("sys.stdin", _TtyStdin())
    monkeypatch.delenv("LOCALHARNESS_COMMIT", raising=False)
    await _start_async(None, False, False, str(tmp_path), no_input=True)
    assert seen == {"channel": False, "gate": False}


# --------------------------------------------------- WR-52: the resolvers apply _cap_limit itself


def test_both_resolvers_cap_through_the_one_named_helper(mock_llm_client, bus, monkeypatch):
    """The helper's own test only proves anything about the loop if the loop calls it: a spy
    stands in for it and every cap the resolvers apply (fallback actions, knob, fallback minutes)
    goes through it."""
    import localharness.agent.loop as loop_mod
    from localharness.config.models import ChildBudget

    calls: list[tuple] = []
    real = loop_mod._cap_limit

    def spy(limit, cap):
        calls.append((limit, cap))
        return real(limit, cap)

    monkeypatch.setattr(loop_mod, "_cap_limit", spy)
    loop = _loop(mock_llm_client, bus, [], max_actions=None, max_minutes=None, local=False,
                 unattended_max_actions=5, budget_fallback=ChildBudget(actions=8, minutes=3.0))
    assert loop._resolve_max_actions()[0] == 5
    assert loop._resolve_max_minutes()[0] == 3.0
    assert calls == [(0, 8), (8, 5), (0.0, 3.0)]


# --------------------------------------------------- IN-55: a step-driven iteration is a live turn


@pytest.mark.asyncio
async def test_a_child_dispatched_during_step_reads_the_finite_parent(mock_llm_client, bus):
    from localharness.agent.loop import Session

    seen: list[tuple[int, float]] = []
    holder: dict = {}
    registry = _Registry(on_call=lambda n: seen.append(holder["loop"].remaining_budget()))
    loop = _loop(mock_llm_client, bus, _calls(mock_llm_client, 1), max_actions=10, max_minutes=5.0,
                 local=True, registry=registry)
    holder["loop"] = loop
    session = Session(agent_id="budget-agent", session_id="s",
                      messages=[{"role": "user", "content": "go"}])
    await loop.step(session)
    (actions, minutes), = seen
    assert actions == 9 and 0.1 <= minutes <= 5.0
    assert loop.remaining_budget() == (0, 0.0)  # cleared once the step returns

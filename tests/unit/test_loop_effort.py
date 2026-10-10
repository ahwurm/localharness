"""AgentLoop's half of /effort (Phase 3, plan 03-03): the level, the budget it becomes, the reaction
to a reasoning cut, and the no-cap `length` retry.

The loop never resolves the ladder (config/effort.py does, and `start` hands over the answer), so
these tests build loops directly with `effort=(level, rung)` and a `RecordingLLM` that records the
keywords of every `stream_complete` call.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop, Session
from localharness.config.models import AgentConfig, EffortBudgetConfig, SelfCheckConfig
from localharness.agent.permissions import PermissionEvaluator
from tests.unit.effort_support import RecordingLLM

WINDOW = 131_072


def _config(**kw: Any) -> AgentConfig:
    return AgentConfig(
        name="test-agent", role="Test agent.", self_check=SelfCheckConfig(enabled=False), **kw
    )


def _answer(text: str = "Done.", finish_reason: str = "stop") -> tuple[Any, Any]:
    return SimpleNamespace(content=text, tool_calls=None, finish_reason=finish_reason), None


def _llm(*responses: tuple[Any, Any], **kw: Any) -> RecordingLLM:
    llm = RecordingLLM(**kw)
    llm.responses = list(responses) or [_answer()]
    return llm


def _loop(llm: Any, bus: Any, *, window: int = WINDOW, config: AgentConfig | None = None,
          **kw: Any) -> AgentLoop:
    return AgentLoop(
        config=config or _config(), llm=llm, bus=bus,
        context_manager=ContextManager(max_context_tokens=window),
        tool_registry=None, permission_evaluator=PermissionEvaluator(), **kw,
    )


def _spy_headroom(loop: AgentLoop) -> list[Any]:
    """Record what the loop's own _window_headroom returned, so the tests never retype the formula."""
    seen: list[Any] = []
    real = loop._window_headroom

    def spy(ctx_budget: Any) -> Any:
        value = real(ctx_budget)
        seen.append(value)
        return value

    loop._window_headroom = spy  # type: ignore[method-assign]
    return seen


async def _run_once(llm: RecordingLLM, bus: Any, **kw: Any) -> tuple[AgentLoop, list[Any]]:
    loop = _loop(llm, bus, **kw)
    seen = _spy_headroom(loop)
    await loop.run_turn("do the thing")
    return loop, seen


# ---------------------------------------------------------------------------
# The level lives on the loop (D-09)
# ---------------------------------------------------------------------------


def test_effort_is_read_back_with_its_rung(bus):
    loop = _loop(_llm(), bus, effort=("low", "agent"))
    assert loop.effective_effort == "low"
    assert loop.effort_rung == "agent"


def test_a_loop_without_effort_has_none(bus):
    loop = _loop(_llm(), bus)
    assert loop.effective_effort is None
    assert loop.effort_rung is None


def test_set_effort_changes_both(bus):
    loop = _loop(_llm(), bus, effort=("low", "agent"))
    loop.set_effort("max", "session")
    assert (loop.effective_effort, loop.effort_rung) == ("max", "session")


def test_set_effort_works_on_a_loop_that_had_none(bus):
    loop = _loop(_llm(), bus)
    loop.set_effort("high", "endpoint")
    assert (loop.effective_effort, loop.effort_rung) == ("high", "endpoint")


@pytest.mark.parametrize("level", ["medium", "", "LOW", "minimal"])
def test_set_effort_rejects_a_bad_level(bus, level):
    loop = _loop(_llm(), bus, effort=("low", "agent"))
    with pytest.raises(ValueError, match="level"):
        loop.set_effort(level, "session")
    assert (loop.effective_effort, loop.effort_rung) == ("low", "agent")


@pytest.mark.parametrize("rung", ["nowhere", "", "Session"])
def test_set_effort_rejects_a_bad_rung(bus, rung):
    loop = _loop(_llm(), bus, effort=("low", "agent"))
    with pytest.raises(ValueError, match="rung"):
        loop.set_effort("low", rung)
    assert (loop.effective_effort, loop.effort_rung) == ("low", "agent")


@pytest.mark.parametrize("rung", ["session", "agent", "endpoint", "org", "default"])
def test_every_ladder_rung_is_accepted(bus, rung):
    loop = _loop(_llm(), bus)
    loop.set_effort("off", rung)
    assert loop.effort_rung == rung


@pytest.mark.parametrize("effort", [("medium", "agent"), ("low", "nowhere")])
def test_the_constructor_validates_through_the_same_check(bus, effort):
    with pytest.raises(ValueError):
        _loop(_llm(), bus, effort=effort)


def test_the_properties_have_no_setter(bus):
    loop = _loop(_llm(), bus, effort=("low", "agent"))
    with pytest.raises(AttributeError):
        loop.effective_effort = "max"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        loop.effort_rung = "session"  # type: ignore[misc]


def test_the_agent_files_rung_is_never_written(bus):
    cfg = _config()
    assert cfg.effort == "inherit"
    loop = _loop(_llm(), bus, config=cfg, effort=("low", "agent"))
    assert cfg.effort == "inherit"
    loop.set_effort("max", "session")
    assert cfg.effort == "inherit"


# ---------------------------------------------------------------------------
# The ratios come from the config model, never from the loop (D-09)
# ---------------------------------------------------------------------------


def test_ratios_default_to_the_config_models_fields(bus):
    defaults = EffortBudgetConfig()
    expected = (defaults.low_ratio, defaults.high_ratio)
    assert _loop(_llm(), bus)._effort_ratios == expected
    assert _loop(_llm(), bus, effort_ratios=None)._effort_ratios == expected


def test_explicit_ratios_are_kept(bus):
    assert _loop(_llm(), bus, effort_ratios=(0.2, 0.6))._effort_ratios == (0.2, 0.6)


# ---------------------------------------------------------------------------
# The level becomes request keys (D-15, D-16)
# ---------------------------------------------------------------------------


async def test_low_sends_its_share_of_the_headroom(bus):
    llm = _llm()
    loop, seen = await _run_once(llm, bus, effort=("low", "agent"))
    assert len(set(seen)) == 1 and seen[0] > 0
    defaults = EffortBudgetConfig()
    assert llm.calls[0]["reasoning_budget"] == max(1, int(defaults.low_ratio * seen[0]))
    assert "disable_thinking" not in llm.calls[0]


async def test_high_sends_its_share_of_the_headroom(bus):
    llm = _llm()
    loop, seen = await _run_once(llm, bus, effort=("high", "agent"))
    defaults = EffortBudgetConfig()
    assert llm.calls[0]["reasoning_budget"] == max(1, int(defaults.high_ratio * seen[0]))


async def test_high_is_a_bigger_budget_than_low(bus):
    low, high = _llm(), _llm()
    await _run_once(low, bus, effort=("low", "agent"))
    await _run_once(high, bus, effort=("high", "agent"))
    assert high.calls[0]["reasoning_budget"] > low.calls[0]["reasoning_budget"]


async def test_configured_ratios_change_both_levels(bus):
    low, high = _llm(), _llm()
    _, seen_low = await _run_once(low, bus, effort=("low", "agent"), effort_ratios=(0.2, 0.6))
    _, seen_high = await _run_once(high, bus, effort=("high", "agent"), effort_ratios=(0.2, 0.6))
    assert low.calls[0]["reasoning_budget"] == max(1, int(0.2 * seen_low[0]))
    assert high.calls[0]["reasoning_budget"] == max(1, int(0.6 * seen_high[0]))


async def test_max_sends_no_budget_and_does_not_turn_thinking_off(bus):
    llm = _llm()
    await _run_once(llm, bus, effort=("max", "agent"))
    assert "reasoning_budget" not in llm.calls[0]
    assert "disable_thinking" not in llm.calls[0]


async def test_off_sends_thinking_off_and_no_budget(bus):
    llm = _llm()
    await _run_once(llm, bus, effort=("off", "agent"))
    assert llm.calls[0]["disable_thinking"] is True
    assert "reasoning_budget" not in llm.calls[0]


async def test_set_effort_changes_the_next_request(bus):
    llm = _llm()
    loop = _loop(llm, bus, effort=("max", "agent"))
    await loop.run_turn("one")
    loop.set_effort("off", "session")
    await loop.run_turn("two")
    assert "disable_thinking" not in llm.calls[0]
    assert llm.calls[1]["disable_thinking"] is True


async def test_a_loop_without_effort_sends_todays_kwargs_byte_for_byte(bus):
    plain, explicit_none = _llm(), _llm()
    await _run_once(plain, bus)
    await _run_once(explicit_none, bus, effort=None, effort_ratios=None)
    assert plain.calls == explicit_none.calls == [{}]


async def test_a_capped_loop_without_effort_sends_only_its_cap(bus):
    llm = _llm()
    await _run_once(llm, bus, config=_config(max_tokens=2048))
    assert all(set(c) <= {"max_tokens"} for c in llm.calls)


async def test_ratios_alone_do_not_switch_effort_on(bus):
    llm = _llm()
    await _run_once(llm, bus, effort_ratios=(0.2, 0.6))
    assert llm.calls == [{}]


class _BareClient:
    """A stream_complete with no **kw and none of the new parameters (an older client)."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(max_tokens=None, tool_call_mode="native")
        self.calls = 0

    async def stream_complete(self, messages, tools=None, on_token=None):
        self.calls += 1
        return _answer()


@pytest.mark.parametrize("level", ["off", "low", "high", "max"])
async def test_a_client_without_the_kwargs_is_called_as_before(bus, level):
    llm = _BareClient()
    loop = _loop(llm, bus, effort=(level, "agent"))
    assert loop._llm_takes_reasoning_budget is False
    assert loop._llm_takes_disable_thinking is False
    assert await loop.run_turn("go") == "Done."
    assert llm.calls == 1


class _BudgetOnlyClient(_BareClient):
    async def stream_complete(self, messages, tools=None, on_token=None, reasoning_budget=None):
        self.calls += 1
        self.seen = reasoning_budget
        return _answer()


async def test_each_key_is_gated_on_its_own_parameter(bus):
    llm = _BudgetOnlyClient()
    loop = _loop(llm, bus, effort=("low", "agent"))
    await loop.run_turn("go")
    assert llm.seen is not None and llm.seen > 0
    off = _loop(_BudgetOnlyClient(), bus, effort=("off", "agent"))
    assert off._llm_takes_disable_thinking is False
    assert await off.run_turn("go") == "Done."


@pytest.mark.parametrize("level", ["low", "high"])
async def test_an_unknown_window_sends_no_budget(bus, level):
    llm = _llm()
    loop = _loop(llm, bus, effort=(level, "agent"))
    # The real _window_headroom, evaluated against a context manager that does not know its window.
    blind = SimpleNamespace(
        _ctx=SimpleNamespace(max_context_tokens=None), _template_margin_ratio=None,
        _last_prompt_tokens=None,
    )
    loop._window_headroom = lambda ctx_budget: AgentLoop._window_headroom(blind, ctx_budget)  # type: ignore[method-assign]
    assert loop._window_headroom(None) is None
    await loop.run_turn("go")
    assert "reasoning_budget" not in llm.calls[0]


@pytest.mark.parametrize("level", ["low", "high"])
async def test_no_headroom_sends_no_budget(bus, level):
    llm = _llm()
    loop = _loop(llm, bus, effort=(level, "agent"))
    loop._window_headroom = lambda ctx_budget: 0  # type: ignore[method-assign]
    await loop.run_turn("go")
    assert "reasoning_budget" not in llm.calls[0]
    loop._window_headroom = lambda ctx_budget: -50  # type: ignore[method-assign]
    assert loop._effort_budget(None) is None


def test_a_budget_is_at_least_one_token(bus):
    loop = _loop(_llm(), bus, effort=("low", "agent"))
    loop._window_headroom = lambda ctx_budget: 3  # type: ignore[method-assign]
    assert loop._effort_budget(None) == 1


@pytest.mark.parametrize("level", ["off", "max"])
def test_off_and_max_have_no_budget(bus, level):
    loop = _loop(_llm(), bus, effort=(level, "agent"))
    assert loop._effort_budget(None) is None


def test_no_effort_has_no_budget(bus):
    assert _loop(_llm(), bus)._effort_budget(None) is None


# ---------------------------------------------------------------------------
# D-22's skip: no server-side key and thinking-off rejected -> no budget at all
# ---------------------------------------------------------------------------


async def test_the_budget_is_withheld_where_the_cut_cannot_work(bus):
    llm = _llm(provider_type="vllm", rejected=("chat_template_kwargs",))
    assert llm.effort_budget_key is None
    await _run_once(llm, bus, effort=("low", "agent"))
    assert "reasoning_budget" not in llm.calls[0]


async def test_a_server_side_key_still_gets_its_budget_when_thinking_off_is_rejected(bus):
    llm = _llm(provider_type="llamacpp", rejected=("chat_template_kwargs",))
    assert llm.effort_budget_key == "reasoning_budget_tokens"
    await _run_once(llm, bus, effort=("low", "agent"))
    assert llm.calls[0]["reasoning_budget"] > 0


async def test_a_no_key_runtime_with_thinking_off_accepted_gets_the_budget_for_the_cut(bus):
    llm = _llm(provider_type="vllm")
    await _run_once(llm, bus, effort=("low", "agent"))
    assert llm.calls[0]["reasoning_budget"] > 0


async def test_another_rejected_key_does_not_withhold_the_budget(bus):
    llm = _llm(provider_type="vllm", rejected=("reasoning_budget_tokens",))
    await _run_once(llm, bus, effort=("low", "agent"))
    assert "reasoning_budget" in llm.calls[0]


# ---------------------------------------------------------------------------
# The loop stays out of the ladder (EFFORT-04)
# ---------------------------------------------------------------------------


def test_loop_imports_nothing_from_the_effort_resolver():
    src = Path(inspect.getsourcefile(AgentLoop)).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("localharness.config.effort"), node.module
            if node.module == "localharness.config":
                assert "effort" not in {a.name for a in node.names}
        elif isinstance(node, ast.Import):
            assert not any(a.name.startswith("localharness.config.effort") for a in node.names)


def test_loop_never_spells_the_default_ratios():
    src = Path(inspect.getsourcefile(AgentLoop)).read_text(encoding="utf-8")
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert "0.125" not in code


# ---------------------------------------------------------------------------
# The reasoning cut: one identical re-send with thinking off, then the turn ends (D-22)
# ---------------------------------------------------------------------------


class _Scripted(RecordingLLM):
    """A RecordingLLM that also records the message list of every call, and whose script may hold
    exceptions to raise (a ProviderReasoningBudgetError) as well as responses to return."""

    def __init__(self, script: list[Any], **kw: Any) -> None:
        super().__init__(**kw)
        self.script = list(script)
        self.sent: list[list[Any]] = []

    async def stream_complete(self, messages, tools=None, on_token=None, **kw):
        import copy

        self.calls.append(dict(kw))
        self.sent.append(copy.deepcopy(list(messages)))
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return item


def _cut(budget: int = 100, spent: int = 450) -> Exception:
    from localharness.provider.client import ProviderReasoningBudgetError

    return ProviderReasoningBudgetError(budget, spent)


@pytest.fixture
def sessions(monkeypatch):
    """Every Session run_turn builds, in order."""
    built: list[Session] = []

    class Recording(Session):
        def __init__(self, *args: Any, **kw: Any) -> None:
            super().__init__(*args, **kw)
            built.append(self)

    monkeypatch.setattr("localharness.agent.loop.Session", Recording)
    return built


def _replies(bus) -> list[Any]:
    from localharness.core.events import Action

    return [e for e in bus.history(event_types=[Action]) if e.action_type == "llm_response"]


async def test_a_cut_is_re_sent_once_with_thinking_off(bus, sessions):
    llm = _Scripted([_cut(100, 450), _answer("Here it is.")], provider_type="vllm")
    loop = _loop(llm, bus, effort=("high", "agent"))
    assert await loop.run_turn("explain") == "Here it is."

    assert len(llm.calls) == 2
    assert "disable_thinking" not in llm.calls[0]
    assert llm.calls[1]["disable_thinking"] is True
    cuts = [e for e in _replies(bus) if e.finish_reason == "reasoning_cut"]
    assert len(cuts) == 1 and cuts[0].reasoning_chars == 450
    assert cuts[0].content == "" and cuts[0].has_tool_calls is False
    assert sessions[0].reasoning_cut_used is True


async def test_the_re_send_is_the_identical_message_list(bus):
    llm = _Scripted([_cut(), _answer()], provider_type="vllm")
    await _loop(llm, bus, effort=("high", "agent")).run_turn("explain")
    assert llm.sent[1] == llm.sent[0]
    assert not any(
        isinstance(m, dict) and m.get("role") == "assistant" for m in llm.sent[1]
    ), "no partial assistant prefix rides the retry"


async def test_no_harness_message_is_pushed_by_the_cut(bus, sessions):
    from localharness.core.types import is_harness_message

    llm = _Scripted([_cut(), _answer()], provider_type="vllm")
    await _loop(llm, bus, effort=("high", "agent")).run_turn("explain")
    assert not any(is_harness_message(m) for m in sessions[0].messages)


async def test_a_cut_retry_keeps_the_budget_off_the_second_request_only_by_the_flag(bus):
    """Thinking-off and the budget are separate keys: the client drops the budget when thinking is
    off; the loop's job is only to arm the flag."""
    llm = _Scripted([_cut(), _answer()], provider_type="vllm")
    await _loop(llm, bus, effort=("low", "agent")).run_turn("explain")
    assert llm.calls[1]["disable_thinking"] is True


async def test_two_cuts_end_the_turn_with_the_notice(bus, sessions):
    from localharness.agent.loop import _REASONING_CUT_TURN_NOTICE
    from localharness.core.events import TaskComplete

    llm = _Scripted([_cut(100, 450), _cut(100, 460), _answer("never reached")], provider_type="vllm")
    loop = _loop(llm, bus, effort=("high", "agent"))
    summary = await loop.run_turn("explain")

    assert summary == _REASONING_CUT_TURN_NOTICE
    assert "/effort" in summary
    assert len(llm.calls) == 2
    assert sessions[0].terminated_reason == "error"
    assert loop.last_terminated_reason == "error"
    assert not [e for e in bus.history(event_types=[TaskComplete]) if e.success]
    assert [e.finish_reason for e in _replies(bus)] == ["reasoning_cut", "reasoning_cut"]


async def _no_sleep(*_a: Any, **_k: Any) -> None:
    return None


async def test_a_cut_on_the_connection_retry_takes_the_thinking_off_re_send(bus, sessions, monkeypatch):
    """WR-03: the connection-error re-issue is a first attempt as far as the handlers are
    concerned, so a reasoning cut on it is re-sent once with thinking off, not raised out of run_turn."""
    from localharness.provider.client import ProviderConnectionError

    monkeypatch.setattr("localharness.agent.loop.asyncio.sleep", _no_sleep)
    llm = _Scripted(
        [ProviderConnectionError("refused"), _cut(100, 450), _answer("Here it is.")],
        provider_type="vllm",
    )
    assert await _loop(llm, bus, effort=("high", "agent")).run_turn("explain") == "Here it is."

    assert len(llm.calls) == 3
    assert "disable_thinking" not in llm.calls[1]
    assert llm.calls[2]["disable_thinking"] is True
    assert sessions[0].reasoning_cut_used is True


async def test_a_second_cut_after_a_connection_retry_ends_the_turn_with_the_notice(
    bus, sessions, monkeypatch,
):
    from localharness.agent.loop import _REASONING_CUT_TURN_NOTICE
    from localharness.provider.client import ProviderConnectionError

    monkeypatch.setattr("localharness.agent.loop.asyncio.sleep", _no_sleep)
    llm = _Scripted(
        [ProviderConnectionError("refused"), _cut(), _cut(), _answer("never reached")],
        provider_type="vllm",
    )
    summary = await _loop(llm, bus, effort=("high", "agent")).run_turn("explain")

    assert summary == _REASONING_CUT_TURN_NOTICE
    assert len(llm.calls) == 3
    assert sessions[0].terminated_reason == "error"


async def test_two_connection_errors_in_a_row_still_end_the_turn(bus, sessions, monkeypatch):
    from localharness.provider.client import ProviderConnectionError

    monkeypatch.setattr("localharness.agent.loop.asyncio.sleep", _no_sleep)
    llm = _Scripted(
        [ProviderConnectionError("refused"), ProviderConnectionError("refused again")],
        provider_type="vllm",
    )
    summary = await _loop(llm, bus, effort=("high", "agent")).run_turn("explain")

    assert "refused again" in summary
    assert len(llm.calls) == 2
    assert sessions[0].terminated_reason == "error"


async def test_a_call_that_goes_through_re_arms_the_connection_retry(bus, sessions, monkeypatch):
    """The retry is per outage, not per turn: an answer in between resets it."""
    from localharness.provider.client import ProviderConnectionError

    monkeypatch.setattr("localharness.agent.loop.asyncio.sleep", _no_sleep)
    llm = _Scripted(
        [ProviderConnectionError("one"), _answer("ok")], provider_type="vllm",
    )
    loop = _loop(llm, bus, effort=("high", "agent"))
    assert await loop.run_turn("first") == "ok"
    assert sessions[0].connection_retry_used is False


async def test_a_cut_never_silences_thinking_in_the_next_turn(bus):
    llm = _Scripted([_cut(), _answer("one"), _answer("two")], provider_type="vllm")
    loop = _loop(llm, bus, effort=("high", "agent"))
    assert await loop.run_turn("first") == "one"
    assert loop._thinking_off_next is True
    assert await loop.run_turn("second") == "two"
    assert "disable_thinking" not in llm.calls[2]


async def test_the_flag_resets_at_the_turn_start(bus):
    loop = _loop(_llm(), bus, effort=("high", "agent"))
    loop._thinking_off_next = True
    await loop.run_turn("go")
    assert loop._thinking_off_next is False


async def test_the_thinking_off_flag_lasts_the_rest_of_the_turn(bus):
    tool_call = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="no_such_tool", arguments="{}"), type="function"
    )
    llm = _Scripted(
        [
            _cut(),
            (SimpleNamespace(content="", tool_calls=[tool_call], finish_reason="tool_calls"), None),
            _answer("done"),
        ],
        provider_type="vllm",
    )
    await _loop(llm, bus, effort=("high", "agent")).run_turn("go")
    assert len(llm.calls) >= 3
    assert all(c.get("disable_thinking") is True for c in llm.calls[1:])


async def test_a_cut_on_a_client_that_cannot_turn_thinking_off_still_re_sends_once(bus):
    class NoOff(_Scripted):
        async def stream_complete(self, messages, tools=None, on_token=None, reasoning_budget=None):
            return await super().stream_complete(messages, tools, on_token)

    llm = NoOff([_cut(), _answer("ok")], provider_type="vllm")
    loop = _loop(llm, bus, effort=("high", "agent"))
    assert await loop.run_turn("explain") == "ok"
    assert llm.calls == [{}, {}]


async def test_the_degenerate_path_is_not_touched_by_the_cut_handler(bus, sessions):
    from localharness.provider.client import ProviderDegenerateError

    llm = _Scripted(
        [ProviderDegenerateError("looped", unit="1, ", repeats=80, chars=1_500), _answer("fine")],
        provider_type="vllm",
    )
    loop = _loop(llm, bus, effort=("high", "agent"))
    assert await loop.run_turn("explain") == "fine"
    assert llm.calls[1]["presence_penalty"] == 1.5
    assert "disable_thinking" not in llm.calls[1]
    assert sessions[0].reasoning_cut_used is False
    assert sessions[0].degenerate_reprompt_used is True


# ---------------------------------------------------------------------------
# The no-cap `length` retry: thinking off instead of the nudge (D-23)
# ---------------------------------------------------------------------------


def _empty(finish_reason: str = "length") -> tuple[Any, Any]:
    return _answer("", finish_reason)


def _nudges(messages: list[Any]) -> list[Any]:
    from localharness.agent.loop import _EMPTY_REPLY_NUDGE

    return [m for m in messages if isinstance(m, dict) and m.get("content") == _EMPTY_REPLY_NUDGE]


async def test_an_empty_length_reply_with_no_cap_is_re_sent_with_thinking_off(bus, sessions):
    llm = _Scripted([_empty(), _answer("Here it is.")])
    summary = await _loop(llm, bus).run_turn("write the report")

    assert summary == "Here it is."
    assert len(llm.calls) == 2
    assert "disable_thinking" not in llm.calls[0]
    assert llm.calls[1]["disable_thinking"] is True
    assert llm.sent[1] == llm.sent[0]
    assert not _nudges(llm.sent[1])
    assert sessions[0].empty_reprompt_used is True
    assert sessions[0].act_nudge_used is False


async def test_a_second_empty_length_reply_ends_the_turn_naming_the_window(bus, sessions):
    from localharness.agent.loop import _EMPTY_TURN_NOTICE_NO_CAP
    from localharness.core.events import TaskComplete

    llm = _Scripted([_empty(), _empty()])
    loop = _loop(llm, bus)
    summary = await loop.run_turn("write the report")

    assert summary == _EMPTY_TURN_NOTICE_NO_CAP
    assert "served window" in summary and "/effort" in summary
    assert "max output tokens" not in summary
    assert "finish_reason=length" in summary
    assert sessions[0].terminated_reason == "error"
    assert not [e for e in bus.history(event_types=[TaskComplete]) if e.success]
    assert len(llm.calls) == 2


async def test_at_effort_off_an_empty_length_reply_takes_the_nudge_not_a_blind_re_send(bus, sessions):
    """WR-04: thinking is already off in the first request, so the one retry must change something."""
    llm = _Scripted([_empty(), _answer("Here it is.")])
    summary = await _loop(llm, bus, effort=("off", "agent")).run_turn("write the report")

    assert summary == "Here it is."
    assert len(llm.calls) == 2
    assert llm.calls[0]["disable_thinking"] is True
    assert len(_nudges(llm.sent[1])) == 1, "the retry asks for an answer instead of repeating itself"
    assert llm.sent[1] != llm.sent[0]
    assert sessions[0].act_nudge_used is True


async def test_at_effort_off_the_final_notice_does_not_say_lower_effort(bus):
    from localharness.agent.loop import (
        _EMPTY_TURN_NOTICE_NO_CAP,
        _EMPTY_TURN_NOTICE_NO_CAP_THINKING_OFF,
    )

    llm = _Scripted([_empty(), _empty()])
    summary = await _loop(llm, bus, effort=("off", "agent")).run_turn("write the report")

    assert summary == _EMPTY_TURN_NOTICE_NO_CAP_THINKING_OFF
    assert summary != _EMPTY_TURN_NOTICE_NO_CAP
    assert "/effort" not in summary
    assert "served window" in summary and "finish_reason=length" in summary
    assert len(llm.calls) == 2


async def test_a_refused_thinking_off_key_takes_the_nudge_and_the_plain_notice(bus):
    from localharness.agent.loop import _EMPTY_TURN_NOTICE_NO_CAP_THINKING_OFF

    llm = _Scripted([_empty(), _answer("ok")], rejected=("chat_template_kwargs",))
    assert await _loop(llm, bus, effort=("high", "agent")).run_turn("write the report") == "ok"
    assert len(_nudges(llm.sent[1])) == 1
    assert llm.sent[1] != llm.sent[0]

    llm = _Scripted([_empty(), _empty()], rejected=("chat_template_kwargs",))
    summary = await _loop(llm, bus, effort=("high", "agent")).run_turn("write the report")
    assert summary == _EMPTY_TURN_NOTICE_NO_CAP_THINKING_OFF


async def test_after_a_cut_has_armed_thinking_off_an_empty_length_reply_takes_the_nudge(bus, sessions):
    llm = _Scripted([_cut(), _empty(), _answer("ok")], provider_type="vllm")
    assert await _loop(llm, bus, effort=("high", "agent")).run_turn("explain") == "ok"

    assert len(llm.calls) == 3
    assert llm.calls[1]["disable_thinking"] is True
    assert len(_nudges(llm.sent[2])) == 1
    assert sessions[0].act_nudge_used is True


async def test_with_a_cap_sent_the_nudge_runs_exactly_as_before(bus, sessions):
    llm = _Scripted([_empty(), _answer("ok")])
    llm.config.max_tokens = 2048
    summary = await _loop(llm, bus).run_turn("write the report")

    assert summary == "ok"
    assert len(_nudges(llm.sent[1])) == 1
    assert all("disable_thinking" not in c for c in llm.calls)
    assert sessions[0].act_nudge_used is True


async def test_with_a_cap_the_second_empty_reply_keeps_the_old_notice_byte_for_byte(bus):
    from localharness.agent.loop import _EMPTY_TURN_NOTICE

    llm = _Scripted([_empty(), _empty()])
    llm.config.max_tokens = 2048
    summary = await _loop(llm, bus).run_turn("write the report")

    assert summary == _EMPTY_TURN_NOTICE.format(finish_reason="length")
    assert "raise the model's max output tokens" in summary


async def test_an_agent_level_cap_counts_as_a_cap(bus):
    from localharness.agent.loop import _EMPTY_TURN_NOTICE

    llm = _Scripted([_empty(), _empty()])
    summary = await _loop(llm, bus, config=_config(max_tokens=4096)).run_turn("write the report")
    assert summary == _EMPTY_TURN_NOTICE.format(finish_reason="length")
    assert all("disable_thinking" not in c for c in llm.calls)


@pytest.mark.parametrize("capped", [False, True])
async def test_an_empty_stop_reply_keeps_the_nudge(bus, sessions, capped):
    from localharness.agent.loop import _EMPTY_TURN_NOTICE

    llm = _Scripted([_empty("stop"), _answer("ok")])
    if capped:
        llm.config.max_tokens = 2048
    assert await _loop(llm, bus).run_turn("write the report") == "ok"
    assert len(_nudges(llm.sent[1])) == 1
    assert all("disable_thinking" not in c for c in llm.calls)

    llm = _Scripted([_empty("stop"), _empty("stop")])
    if capped:
        llm.config.max_tokens = 2048
    summary = await _loop(llm, bus).run_turn("write the report")
    assert summary == _EMPTY_TURN_NOTICE.format(finish_reason="stop")


async def test_a_loop_with_no_effort_takes_the_thinking_off_retry_too(bus):
    llm = _Scripted([_empty(), _answer("ok")])
    loop = _loop(llm, bus)
    assert loop.effective_effort is None
    assert await loop.run_turn("write the report") == "ok"
    assert llm.calls[1]["disable_thinking"] is True


async def test_a_client_that_cannot_turn_thinking_off_gets_the_nudge(bus):
    class NoOff(_Scripted):
        async def stream_complete(self, messages, tools=None, on_token=None, max_tokens=None):
            return await super().stream_complete(messages, tools, on_token)

    llm = NoOff([_empty(), _answer("ok")])
    loop = _loop(llm, bus)
    assert loop._llm_takes_disable_thinking is False
    assert await loop.run_turn("write the report") == "ok"
    assert len(_nudges(llm.sent[1])) == 1
    assert llm.calls == [{}, {}]


async def test_the_retry_and_the_nudge_are_never_both_spent_in_one_turn(bus):
    """Ordering: the no-cap length retry is chosen before the nudge branch. The single
    empty_reprompt_used is spent on thinking-off, so the nudge never follows it."""
    llm = _Scripted([_empty(), _empty("stop"), _answer("never reached")])
    summary = await _loop(llm, bus).run_turn("write the report")
    assert len(llm.calls) == 2
    assert not _nudges(llm.sent[1])
    assert "empty reply twice" in summary


async def test_an_effort_off_loop_that_hits_the_window_still_ends_honestly(bus):
    llm = _Scripted([_empty(), _empty()])
    summary = await _loop(llm, bus, effort=("off", "session")).run_turn("write the report")
    assert "served window" in summary
    assert all(c.get("disable_thinking") is True for c in llm.calls)


async def test_the_length_retry_flag_resets_for_the_next_turn(bus):
    llm = _Scripted([_empty(), _answer("one"), _answer("two")])
    loop = _loop(llm, bus)
    assert await loop.run_turn("first") == "one"
    assert await loop.run_turn("second") == "two"
    assert "disable_thinking" not in llm.calls[2]


async def test_a_final_answer_cut_at_the_ceiling_is_untouched(bus):
    """9a-bis: a non-empty reply cut at the window ships as before; no thinking-off retry."""
    llm = _Scripted([_answer("The draft begins and then", "length")])
    summary = await _loop(llm, bus).run_turn("write the report")
    assert summary == "The draft begins and then"
    assert len(llm.calls) == 1

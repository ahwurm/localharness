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

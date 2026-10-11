"""The stuck detector tells progress from a stuck loop (STUCK-01, STUCK-02).

A trusted-origin tool (bash, agent) whose result keeps changing is making progress; an
untrusted one (web, MCP, memory) never is, because a page or server can vary its reply at will.
A per-turn call-only tally escalates regardless of the result, and survives the nudge's window
clear. Detector-level tests need no loop; the loop-level tests drive a real turn.
"""
from __future__ import annotations

import pytest

from localharness.agent.loop import StuckDetector, StuckState
from localharness.config.models import (
    AgentConfig,
    PermissionConfig,
    RecoveryInjectionConfig,
    StuckDetectorConfig,
)
from localharness.core.events import Escalation, TurnFailed

NUDGE_TAIL = "If you are waiting on something external, say so and stop; do not poll."
GIT_PUSH = {"command": "git push"}
GH_RUN = {"command": "gh run list"}


def _trusted(_name):
    return "trusted"


def _untrusted(_name):
    return "untrusted"


def _step(sd, name, args, result):
    sd.record(name, args, result)
    return sd.classify()[0]


# --- detector level: the replayed logs ---------------------------------------


def test_git_push_seven_identical_errors_escalate_through_the_ladder():
    sd = StuckDetector(result_origin=_trusted)
    states = [
        _step(sd, "bash_exec", GIT_PUSH, (False, "remote: Internal Server Error"))
        for _ in range(7)
    ]
    assert StuckState.ESCALATE in states
    assert states.index(StuckState.ESCALATE) < 7
    assert sd.ceiling_hit is None


def test_gh_run_list_changing_durations_are_progress_until_the_ceiling():
    sd = StuckDetector(result_origin=_trusted)
    states = [
        _step(sd, "bash_exec", GH_RUN, (True, f"run 1234 in_progress {i}s")) for i in range(8)
    ]
    assert StuckState.ESCALATE not in states
    assert sd.ceiling_hit is None
    # the ninth identical call hits 3 x 3 whatever it returned
    assert _step(sd, "bash_exec", GH_RUN, (True, "run 1234 in_progress 8s")) == StuckState.ESCALATE
    assert sd.ceiling_hit == sd.compute_signature("bash_exec", GH_RUN)
    assert sd.call_count(sd.ceiling_hit) == 9


def test_the_tally_survives_the_nudges_window_clear():
    sd = StuckDetector(result_origin=_trusted)
    # two identical results: RECOVERING fires and the window is cleared
    assert _step(sd, "bash_exec", GH_RUN, (True, "same")) == StuckState.CLEAR
    assert _step(sd, "bash_exec", GH_RUN, (True, "same")) == StuckState.RECOVERING
    assert len(sd._window) == 0
    states = [
        _step(sd, "bash_exec", GH_RUN, (True, f"tick {i}")) for i in range(7)
    ]
    assert states[:-1] == [StuckState.CLEAR] * 6
    assert states[-1] == StuckState.ESCALATE
    assert sd.ceiling_hit == sd.compute_signature("bash_exec", GH_RUN)


# --- detector level: untrusted and fail-closed lookups -----------------------


def test_untrusted_origin_with_varying_output_still_escalates_through_the_ladder():
    sd = StuckDetector(result_origin=_untrusted)
    states = [
        _step(sd, "web_fetch", {"url": "http://x"}, (True, f"nonce {i}")) for i in range(7)
    ]
    assert StuckState.ESCALATE in states
    assert sd.ceiling_hit is None


def test_a_lookup_that_raises_is_call_only():
    def boom(_name):
        raise RuntimeError("no registry")

    sd = StuckDetector(result_origin=boom)
    sd.record("bash_exec", GIT_PUSH, (True, "a"))
    sd.record("bash_exec", GIT_PUSH, (True, "b"))
    assert list(sd._window) == [sd.compute_signature("bash_exec", GIT_PUSH)] * 2
    assert sd.check() == StuckState.RECOVERING


def test_no_lookup_is_call_only():
    sd = StuckDetector(result_origin=None)
    sd.record("bash_exec", GIT_PUSH, (True, "a"))
    sd.record("bash_exec", GIT_PUSH, (True, "b"))
    assert sd.check() == StuckState.RECOVERING


def test_an_unknown_tool_reads_as_untrusted():
    class Registry:
        def result_origin(self, name):
            return "trusted" if name == "bash_exec" else "untrusted"

    sd = StuckDetector(result_origin=Registry().result_origin)
    sd.record("mystery", {}, (True, "a"))
    sd.record("mystery", {}, (True, "b"))
    assert sd.check() == StuckState.RECOVERING


def test_results_are_hashed_exactly_with_no_normalisation():
    sd = StuckDetector(result_origin=_trusted)
    sd.record("bash_exec", GH_RUN, (True, "done"))
    sd.record("bash_exec", GH_RUN, (True, "done "))
    assert len(set(sd._window)) == 2
    # success and failure are part of the result
    sd2 = StuckDetector(result_origin=_trusted)
    sd2.record("bash_exec", GH_RUN, (True, "x"))
    sd2.record("bash_exec", GH_RUN, (False, "x"))
    assert len(set(sd2._window)) == 2


def test_identical_trusted_results_hash_alike():
    sd = StuckDetector(result_origin=_trusted)
    sd.record("bash_exec", GH_RUN, (True, "same"))
    sd.record("bash_exec", GH_RUN, (True, "same"))
    assert len(set(sd._window)) == 1


# --- the knob -----------------------------------------------------------------


def test_call_only_multiplier_default_and_bounds():
    assert StuckDetectorConfig().call_only_multiplier == 3
    assert StuckDetectorConfig(call_only_multiplier=1).call_only_multiplier == 1
    assert StuckDetectorConfig(call_only_multiplier=10).call_only_multiplier == 10
    for bad in (0, 11):
        with pytest.raises(Exception):
            StuckDetectorConfig(call_only_multiplier=bad)


def test_the_multiplier_scales_the_ceiling():
    sd = StuckDetector(result_origin=_trusted, call_only_multiplier=2)  # 2 x 3 = 6
    states = [_step(sd, "bash_exec", GH_RUN, (True, f"t{i}")) for i in range(6)]
    assert states[:-1] == [StuckState.CLEAR] * 5
    assert states[-1] == StuckState.ESCALATE


# --- the nudge sentence (D-05, D-41) ------------------------------------------


def test_the_nudge_default_and_the_detector_agree_and_end_with_the_sentence():
    default = RecoveryInjectionConfig().message
    assert default.endswith(NUDGE_TAIL)
    assert StuckDetector().recovery_message("x") == default
    assert AgentConfig(name="t", role="t").recovery_injection.message == default


@pytest.mark.asyncio
async def test_the_nudge_pushed_into_history_ends_with_the_sentence(mock_llm_client, bus):
    from tests.unit.test_agent_loop import _SameResultRegistry, _make_agent_loop

    Response = mock_llm_client.Response
    tc = mock_llm_client.ToolCall(id="a", name="bash", arguments={"cmd": "ls"})
    responses = [
        Response(content=None, tool_calls=[tc]),
        Response(content=None, tool_calls=[tc]),   # second identical call: RECOVERING
        Response(content="done"),
    ]
    loop = _make_agent_loop(mock_llm_client, responses, bus, tool_registry=_SameResultRegistry())
    await loop.run_turn("task")

    nudges = [m["content"] for m in loop._conversation
              if m.get("role") == "user" and (m.get("content") or "").endswith(NUDGE_TAIL)]
    assert len(nudges) == 1


# --- loop level ----------------------------------------------------------------


class _Registry:
    """bash_exec declared trusted; output varies per dispatch when `vary`."""

    def __init__(self, *, vary: bool, origin=None):
        self.vary = vary
        self.n = 0
        self._origin = origin

    def get_tools_for_agent(self, agent_id, division_id, tool_config):
        return {}

    async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
        from localharness.tools.base import ToolResult

        self.n += 1
        return ToolResult(output=f"out {self.n}" if self.vary else "same", success=True)

    def result_origin(self, name):
        return self._origin(name) if self._origin else "trusted"


class _AlwaysCallsLLM:
    class config:
        tool_call_mode = "native"
        context_window = 128_000

    def __init__(self, response_cls, tool_call):
        self._resp = response_cls
        self._tc = tool_call

    async def stream_complete(self, messages=None, tools=None, on_token=None, **kw):
        if tools is None:
            return self._resp(content="wrapped up"), None
        return self._resp(content=None, tool_calls=[self._tc]), None


def _loop(mock_llm_client, bus, registry, *, config=None):
    from tests.unit.test_agent_loop import _make_agent_loop

    loop = _make_agent_loop(mock_llm_client, [], bus, tool_registry=registry, config=config)
    tc = mock_llm_client.ToolCall(id="a", name="bash_exec", arguments={"command": "gh run list"})
    loop._llm = _AlwaysCallsLLM(mock_llm_client.Response, tc)
    return loop


@pytest.mark.asyncio
async def test_a_trusted_poll_that_keeps_changing_ends_at_the_ninth_call(mock_llm_client, bus):
    reg = _Registry(vary=True)
    loop = _loop(mock_llm_client, bus, reg)
    await loop.run_turn("task")

    assert reg.n == 9
    esc = bus.history(event_types=[Escalation])
    assert len(esc) == 1
    assert esc[0].reason == "stuck_detected"
    assert "call_only_multiplier" in esc[0].detail
    assert "9 times" in esc[0].detail
    assert any(e.reason == "stuck_detected" for e in bus.history(event_types=[TurnFailed]))


@pytest.mark.asyncio
async def test_a_registry_without_result_origin_keeps_todays_escalation(mock_llm_client, bus):
    from tests.unit.test_agent_loop import _SameResultRegistry

    loop = _loop(mock_llm_client, bus, _SameResultRegistry())
    await loop.run_turn("task")

    esc = bus.history(event_types=[Escalation])
    assert len(esc) == 1
    assert "call_only_multiplier" not in esc[0].detail
    assert esc[0].detail.startswith("Repeated identical tool calls at iteration")
    assert esc[0].iteration_at_escalation == 4


@pytest.mark.asyncio
async def test_an_untrusted_tool_varying_its_output_escalates_through_the_ladder(
    mock_llm_client, bus,
):
    reg = _Registry(vary=True, origin=_untrusted)
    loop = _loop(mock_llm_client, bus, reg)
    await loop.run_turn("task")

    esc = bus.history(event_types=[Escalation])
    assert len(esc) == 1
    assert esc[0].detail.startswith("Repeated identical tool calls at iteration")
    assert reg.n < 9


@pytest.mark.asyncio
async def test_the_budget_note_is_not_part_of_the_hashed_result(mock_llm_client, bus):
    """Step 11b appends a per-iteration note to the message; the signature hashes the local
    result, so a fixed output under a finite budget still repeats and escalates by the ladder."""
    cfg = AgentConfig(
        name="test-agent",
        role="Test agent.",
        permissions=PermissionConfig(mode="unattended", budget={"max_actions": 50}),
    )
    reg = _Registry(vary=False)
    loop = _loop(mock_llm_client, bus, reg, config=cfg)
    await loop.run_turn("task")

    esc = bus.history(event_types=[Escalation])
    assert len(esc) == 1
    assert esc[0].detail.startswith("Repeated identical tool calls at iteration")
    assert esc[0].iteration_at_escalation == 4
    assert any("[budget:" in (m.get("content") or "") for m in loop._conversation
               if m.get("role") == "tool")

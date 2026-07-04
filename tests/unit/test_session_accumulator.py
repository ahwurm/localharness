"""SESS-02/05: SessionAccumulator sitting counters + the payload-first summary line.

Behavior contract (plan 33-03 Task 1):
- counts turns/actions/tokens from real bus events (failed turns still count + spend)
- derive_session_summary leads with the highest-warrant gate capture detail
  (resolved_error > stuck_recovered; novelty never leads — it is telemetry)
- vacuous sitting (no tools, no captures) -> None (the SESS-05 KILL guardrail)
- the payload survives the 180-char line budget (truncation eats the tail)
- the agent_id subscription filter is part of the contract (real-bus proof)
"""
from __future__ import annotations

from localharness.cli.session_accumulator import (
    SessionAccumulator,
    derive_session_summary,
)
from localharness.core.bus import EventBus
from localharness.core.events import (
    MemoryGateFired,
    Observation,
    TurnCompleted,
    TurnFailed,
    UserMessage,
)

AGENT = "sess-agent"


def _acc() -> SessionAccumulator:
    # bus is stored but unused when handlers are driven directly
    return SessionAccumulator(bus=EventBus(), agent_id=AGENT)


def _turn(inp: int, out: int) -> TurnCompleted:
    return TurnCompleted(
        agent_id=AGENT, session_id="sit-1", iterations=1, duration_seconds=1.0,
        elapsed_tokens=inp + out, input_tokens=inp, output_tokens=out, summary="done",
    )


def _obs(tool: str) -> Observation:
    return Observation(
        agent_id=AGENT, session_id="sit-1", observation_type="tool_result",
        tool_name=tool, output="ok",
    )


def _gate(tier: str, detail: str) -> MemoryGateFired:
    return MemoryGateFired(
        agent_id=AGENT, session_id="sit-1", tier=tier, fact_key=f"gate/{tier}/k",
        tool_name="bash_exec", detail=detail,
    )


def _user_msg(content: str, agent: str = AGENT) -> UserMessage:
    return UserMessage(
        agent_id=agent, session_id="sit-1", content=content, channel="terminal",
    )


async def test_counts_turns_actions_tokens():
    acc = _acc()
    await acc.on_turn_completed(_turn(100, 50))
    await acc.on_turn_completed(_turn(100, 50))
    for tool in ("bash_exec", "bash_exec", "read"):
        await acc.on_observation(_obs(tool))
    assert acc.turn_count == 2
    assert acc.action_count == 3
    assert acc.tokens_in == 200
    assert acc.tokens_out == 100
    assert dict(acc.tools_used) == {"bash_exec": 2, "read": 1}


async def test_turn_failed_still_counts_and_spends():
    acc = _acc()
    await acc.on_turn_failed(TurnFailed(
        agent_id=AGENT, session_id="sit-1", reason="stuck_detected", detail="stuck",
        iterations=1, duration_seconds=1.0, input_tokens=30, output_tokens=10,
    ))
    assert acc.turn_count == 1
    assert acc.tokens_in == 30
    assert acc.tokens_out == 10


async def test_non_tool_observation_does_not_count():
    acc = _acc()
    await acc.on_observation(Observation(
        agent_id=AGENT, session_id="sit-1", observation_type="thought", tool_name=None,
    ))
    assert acc.action_count == 0
    assert dict(acc.tools_used) == {}


async def test_summary_leads_with_capture_detail():
    acc = _acc()
    await acc.on_gate_fired(_gate("resolved_error", "uv: command not found"))
    await acc.on_turn_completed(_turn(10, 5))
    await acc.on_observation(_obs("bash_exec"))
    line = derive_session_summary(acc)
    assert line is not None
    assert line.startswith("resolved: uv: command not found")
    assert "bash_exec" in line
    # payload leads bookkeeping
    assert line.index("uv: command not found") < line.index("turn")


async def test_stuck_capture_leads_when_no_resolved():
    acc = _acc()
    await acc.on_gate_fired(_gate("stuck_recovered", "repeated read of missing file"))
    await acc.on_observation(_obs("read"))
    line = derive_session_summary(acc)
    assert line is not None
    assert line.startswith("unstuck: repeated read of missing file")


async def test_resolved_beats_stuck_when_both_present():
    acc = _acc()
    await acc.on_gate_fired(_gate("stuck_recovered", "STUCK-detail"))
    await acc.on_gate_fired(_gate("resolved_error", "RESOLVED-detail"))
    await acc.on_observation(_obs("bash_exec"))
    line = derive_session_summary(acc)
    assert line is not None
    assert line.startswith("resolved: RESOLVED-detail")
    assert "STUCK-detail" not in line


async def test_novelty_never_leads():
    acc = _acc()
    await acc.on_gate_fired(_gate("novelty", "first use of bash_exec"))
    await acc.on_observation(_obs("bash_exec"))
    await acc.on_turn_completed(_turn(10, 5))
    line = derive_session_summary(acc)
    assert line is not None
    assert "novelty" not in line
    assert line.startswith("1 turn, 1 tool call")
    assert "bash_exec" in line


async def test_vacuous_returns_none():
    acc = _acc()
    # turns without tools or captures are NOT discriminating -> suppressed
    await acc.on_turn_completed(_turn(10, 5))
    await acc.on_turn_completed(_turn(10, 5))
    assert derive_session_summary(acc) is None


async def test_none_accumulator_returns_none():
    assert derive_session_summary(None) is None


async def test_line_budget():
    acc = _acc()
    await acc.on_gate_fired(_gate("resolved_error", "E" * 300))
    await acc.on_observation(_obs("bash_exec"))
    await acc.on_turn_completed(_turn(1, 1))
    line = derive_session_summary(acc)
    assert line is not None
    assert len(line) <= 180
    # payload head survives the cap; the tail is what gets eaten
    assert line.startswith("resolved: " + "E" * 120)


async def test_agent_id_filter_via_real_bus():
    """Only this agent's events count when driven through a real EventBus — the
    agent_id filter is a contract, not an accident."""
    bus = EventBus()
    acc = SessionAccumulator(bus=bus, agent_id=AGENT)
    await acc.open()
    try:
        await bus.publish(_turn(100, 50))  # matches AGENT
        await bus.publish(TurnCompleted(
            agent_id="other-agent", session_id="sit-x", iterations=1,
            duration_seconds=1.0, elapsed_tokens=999, input_tokens=999,
            output_tokens=999, summary="nope",
        ))
    finally:
        await acc.close()
    assert acc.turn_count == 1
    assert acc.tokens_in == 100
    assert acc.tokens_out == 50
    assert acc._handles == []  # close() unsubscribed everything


# ---------------------------------------------------------------------------
# TIME-01: zero-model topical slice (the user's FIRST ask) leads a pure-chat
# sitting; delegation-aware tail; "1 turns" pluralization bug dies.
# ---------------------------------------------------------------------------

async def test_ask_capture_leads_when_no_payload():
    """The owner UAT-2 anchor: a pure-chat delegation sitting reads
    `asked: "..." — 3 turns, 1 delegation` — no "(agent)", no "tool calls" phrase."""
    acc = _acc()
    await acc.on_user_message(_user_msg("any fun 4th of July events near the boardwalk, FL?"))
    for _ in range(3):
        await acc.on_turn_completed(_turn(10, 5))
    await acc.on_observation(_obs("agent"))
    line = derive_session_summary(acc)
    assert line == (
        'asked: "any fun 4th of July events near the boardwalk, FL?" — 3 turns, 1 delegation'
    )


async def test_resolved_error_still_leads_over_ask():
    """Payload-first order stands: an ask is captured but the resolved_error tier
    still leads, with the `; ` separator, and `asked:` never appears."""
    acc = _acc()
    await acc.on_user_message(_user_msg("please make my build work"))
    await acc.on_gate_fired(_gate("resolved_error", "uv: command not found"))
    await acc.on_observation(_obs("bash_exec"))
    await acc.on_turn_completed(_turn(10, 5))
    line = derive_session_summary(acc)
    assert line is not None
    assert line.startswith("resolved: uv: command not found; ")
    assert "asked:" not in line


async def test_first_ask_only_first_kept():
    """The sitting's OPENING ask is the topic; later asks do not overwrite it."""
    acc = _acc()
    await acc.on_user_message(_user_msg("first ask"))
    await acc.on_user_message(_user_msg("second ask"))
    assert acc.first_ask == "first ask"
    line = derive_session_summary(acc)
    assert line is not None
    assert '"first ask"' in line
    assert "second ask" not in line


async def test_ask_sanitized_single_line():
    """Newlines/tabs collapse to single spaces and double quotes become single —
    the raw ask can never break the single-line `asked: "..."` markdown contract."""
    acc = _acc()
    await acc.on_user_message(_user_msg('line one\nline "two"\t  spaced'))
    await acc.on_turn_completed(_turn(10, 5))
    line = derive_session_summary(acc)
    assert line is not None
    assert 'asked: "line one line \'two\' spaced"' in line
    assert "\n" not in line and "\t" not in line


async def test_ask_truncated_with_ellipsis():
    """A long ask is trimmed to a 120-char quoted slice (119 kept + ellipsis); the
    counts tail survives and the whole line stays within the 180-char budget."""
    acc = _acc()
    await acc.on_user_message(_user_msg("Q" * 200))
    await acc.on_turn_completed(_turn(10, 5))
    line = derive_session_summary(acc)
    assert line is not None
    quoted = line[line.index('"') + 1 : line.rindex('"')]
    assert len(quoted) == 120
    assert quoted.endswith("…")
    assert line.endswith(" — 1 turn")
    assert len(line) <= 180


async def test_mixed_tools_and_delegations_tail():
    """OQ1 ruling pinned: tool calls (delegations excluded) and delegations render as
    separate comma parts; "agent" is kept out of the top-tools parenthetical."""
    acc = _acc()
    await acc.on_turn_completed(_turn(10, 5))
    await acc.on_observation(_obs("bash_exec"))
    await acc.on_observation(_obs("bash_exec"))
    await acc.on_observation(_obs("agent"))
    line = derive_session_summary(acc)
    assert line == "1 turn, 2 tool calls (bash_exec), 1 delegation"


async def test_user_message_agent_filter_via_real_bus():
    """Only this agent's ask is captured when driven through a real EventBus — the
    UserMessage subscription is agent_id-filtered like the other four."""
    bus = EventBus()
    acc = SessionAccumulator(bus=bus, agent_id=AGENT)
    await acc.open()
    try:
        await bus.publish(_user_msg("not mine", agent="other-agent"))
        await bus.publish(_user_msg("mine"))
    finally:
        await acc.close()
    assert acc.first_ask == "mine"

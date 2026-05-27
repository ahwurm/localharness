"""SCEN-02: bench runner accumulates standardized metrics from event subscriptions."""
from __future__ import annotations
import pytest


@pytest.mark.xfail(strict=True, reason="Wave 2: bench.runner module not yet created (11-03)")
def test_accumulate_tokens_from_turn_completed():
    """MetricAccumulator.on_turn_completed sums input_tokens+output_tokens across events."""
    from localharness.bench.runner import MetricAccumulator
    from localharness.core.events import TurnCompleted
    acc = MetricAccumulator()
    acc.on_turn_completed(TurnCompleted(
        agent_id="a", session_id="s", iterations=1, duration_seconds=1.0,
        elapsed_tokens=100, input_tokens=80, output_tokens=20, summary="done",
    ))
    acc.on_turn_completed(TurnCompleted(
        agent_id="a", session_id="s", iterations=2, duration_seconds=2.0,
        elapsed_tokens=200, input_tokens=160, output_tokens=40, summary="done",
    ))
    assert acc.tokens_in == 240
    assert acc.tokens_out == 60


@pytest.mark.xfail(strict=True, reason="Wave 2: bench.runner module not yet created (11-03)")
def test_accumulate_iterations_from_turn_completed():
    """MetricAccumulator.iterations takes max iterations from TurnCompleted events."""
    from localharness.bench.runner import MetricAccumulator
    from localharness.core.events import TurnCompleted
    acc = MetricAccumulator()
    acc.on_turn_completed(TurnCompleted(
        agent_id="a", session_id="s", iterations=5, duration_seconds=1.0,
        elapsed_tokens=10, summary="done",
    ))
    assert acc.iterations == 5


@pytest.mark.xfail(strict=True, reason="Wave 2: bench.runner module not yet created (11-03)")
def test_accumulate_tool_call_count_from_actions():
    """MetricAccumulator.on_action increments tool_call_count when tool_name set."""
    from localharness.bench.runner import MetricAccumulator
    from localharness.core.events import Action
    acc = MetricAccumulator()
    acc.on_action(Action(agent_id="a", session_id="s", action="tool_calls", tool_name="bash"))
    acc.on_action(Action(agent_id="a", session_id="s", action="tool_calls", tool_name="read_file"))
    acc.on_action(Action(agent_id="a", session_id="s", action="complete"))  # no tool_name → not counted
    assert acc.tool_call_count == 2


@pytest.mark.xfail(strict=True, reason="Wave 2: bench.runner module not yet created (11-03)")
def test_accumulate_parse_failures_from_event():
    """MetricAccumulator.on_parse_failed increments parse_failures counter."""
    from localharness.bench.runner import MetricAccumulator
    from localharness.core.events import ParseFailed
    acc = MetricAccumulator()
    acc.on_parse_failed(ParseFailed(agent_id="a", session_id="s", iteration=1, parse_retry_count=1, raw_content_preview="x"))
    acc.on_parse_failed(ParseFailed(agent_id="a", session_id="s", iteration=2, parse_retry_count=2, raw_content_preview="y"))
    assert acc.parse_failures == 2


@pytest.mark.xfail(strict=True, reason="Wave 2: bench.runner module not yet created (11-03)")
def test_accumulate_stuck_recoveries_from_event():
    """MetricAccumulator.on_stuck_recovered increments stuck_recoveries counter."""
    from localharness.bench.runner import MetricAccumulator
    from localharness.core.events import StuckRecovered
    acc = MetricAccumulator()
    acc.on_stuck_recovered(StuckRecovered(agent_id="a", session_id="s", iteration=3, stuck_signature="x"))
    assert acc.stuck_recoveries == 1


@pytest.mark.xfail(strict=True, reason="Wave 2: bench.runner module not yet created (11-03)")
def test_tokens_estimated_propagates():
    """Any TurnCompleted with tokens_estimated=True → ScenarioCompleted.tokens_estimated=True."""
    from localharness.bench.runner import MetricAccumulator
    from localharness.core.events import TurnCompleted
    acc = MetricAccumulator()
    acc.on_turn_completed(TurnCompleted(
        agent_id="a", session_id="s", iterations=1, duration_seconds=1.0,
        elapsed_tokens=100, tokens_estimated=True, summary="done",
    ))
    assert acc.tokens_estimated is True

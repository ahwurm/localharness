"""harness_floor and the fixed part of a role: the smallest window a loop can run in."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from localharness.agent.context import (
    COMPACTION_TARGET_USAGE_FRACTION,
    ContextManager,
    fixed_tokens,
    harness_floor,
    response_reserve,
    schema_tokens,
)
from localharness.agent.loop import AgentLoop
from localharness.config.models import AgentConfig, PermissionConfig
from localharness.core.bus import EventBus
from localharness.agent.permissions import PermissionEvaluator

CASES = [
    (6_700, None, 15_360),
    (4_370, None, 8_448),
    (6_300, None, 12_032),
    (6_400, None, 14_848),
    (6_700, 8_192, 19_456),
    (0, None, 256),
    (-5, None, 256),
]


def _fits(fixed: int, window: int, mrt: int | None) -> bool:
    return fixed <= COMPACTION_TARGET_USAGE_FRACTION * (window - response_reserve(window, mrt))


@pytest.mark.parametrize(("fixed", "mrt", "expected"), CASES)
def test_the_floor_for_each_fixed_part(fixed, mrt, expected):
    assert harness_floor(fixed, mrt) == expected


@pytest.mark.parametrize(("fixed", "mrt", "expected"), [c for c in CASES if c[2] > 256])
def test_the_floor_fits_and_one_step_below_does_not(fixed, mrt, expected):
    assert _fits(fixed, expected, mrt)
    assert not _fits(fixed, expected - 256, mrt)


def test_the_scan_crosses_the_reserve_jump_without_a_closed_form():
    assert harness_floor(6_300) == 12_032 < 12_288 <= harness_floor(6_400) == 14_848


def test_a_configured_reply_cap_raises_the_floor_and_never_lowers_the_reserve():
    assert harness_floor(6_700, 8_192) > harness_floor(6_700)
    window = harness_floor(6_700, 8_192)
    assert response_reserve(window, 8_192) == 8_192


class _Counter:
    def __init__(self, per_call: int) -> None:
        self.per_call = per_call
        self.seen: list[list[dict]] = []

    def estimate_messages(self, messages):
        self.seen.append(messages)
        return self.per_call


class _Schema:
    def model_dump(self):
        return {"name": "t"}


def test_fixed_tokens_is_one_system_message_plus_the_schemas():
    c = _Counter(100)
    assert fixed_tokens(c, "x", []) == 100
    assert c.seen == [[{"role": "system", "content": "x"}]]
    assert fixed_tokens(c, "x", [_Schema()]) == 200
    assert schema_tokens(c, None) == 0 and schema_tokens(c, []) == 0
    assert c.seen[-1] == [{"role": "system", "content": '[{"name": "t"}]'}]


class _Registry:
    def __init__(self, schemas=None, error: Exception | None = None) -> None:
        self.schemas, self.error = schemas or {}, error

    def get_tools_for_agent(self, agent_id, division_id, tool_config):
        if self.error:
            raise self.error
        return self.schemas


def _loop(per_call: int, registry=None, max_response_tokens=None) -> tuple[AgentLoop, _Counter]:
    counter = _Counter(per_call)
    cfg = AgentConfig(name="a", role="Test agent.", permissions=PermissionConfig(mode="unattended"))
    ctx = ContextManager(token_counter=counter, max_response_tokens=max_response_tokens)
    loop = AgentLoop(config=cfg, llm=SimpleNamespace(config=None), bus=EventBus(), context_manager=ctx,
                     tool_registry=registry, permission_evaluator=PermissionEvaluator())
    return loop, counter


def test_a_loop_floor_is_harness_floor_of_its_own_fixed_part():
    loop, _ = _loop(6_700)
    assert loop.fixed_tokens() == 6_700
    assert loop.floor() == 15_360


def test_a_loop_counts_its_tool_schemas_into_the_fixed_part():
    loop, _ = _loop(3_350, _Registry({"t": _Schema()}))
    assert loop.fixed_tokens() == 6_700
    assert loop.floor() == 15_360


def test_a_registry_that_raises_counts_the_prompt_alone():
    loop, _ = _loop(4_370, _Registry(error=RuntimeError("boom")))
    assert loop.fixed_tokens() == 4_370
    assert loop.floor() == 8_448


def test_a_loop_floor_reads_the_context_managers_reply_cap():
    loop, _ = _loop(6_700, max_response_tokens=8_192)
    assert loop.floor() == 19_456

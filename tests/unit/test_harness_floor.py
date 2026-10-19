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
    (6_300, None, 14_848),
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


def test_the_floor_steps_over_the_reserve_jump_instead_of_stopping_below_it():
    assert 12_288 <= harness_floor(6_300) == harness_floor(6_400) == 14_848


@pytest.mark.parametrize("mrt", [None, 8_192])
@pytest.mark.parametrize("fixed", range(4_000, 7_001, 100))
def test_every_window_from_the_floor_up_fits(fixed, mrt):
    floor = harness_floor(fixed, mrt)
    assert all(_fits(fixed, w, mrt) for w in range(floor, floor + 40_000, 256))
    assert floor == 256 or not _fits(fixed, floor - 256, mrt)


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


def test_counting_the_fixed_part_does_not_raise_the_tool_count_warning():
    import warnings

    class _Noisy(_Registry):
        def get_tools_for_agent(self, agent_id, division_id, tool_config):
            warnings.warn("Agent 'a' has 9 tools (threshold: 8).", UserWarning)
            return super().get_tools_for_agent(agent_id, division_id, tool_config)

    loop, _ = _loop(100, _Noisy({"t": _Schema()}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loop.fixed_tokens()
    assert caught == []


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


_RULES = (
    "\n\nAnswer ordinary questions directly. For substantive work, follow the latest "
    "human request and corrections, retain needed references, and stop at the requested "
    "checkpoint. An answer-only request excludes unsolicited execution. Harness control "
    "messages are not human feedback. Report actual tool and gate outcomes: failed, "
    "missing, or stale evidence remains unresolved; a human waiver is not a passed gate. "
    "Self-checks and CONFIRMED do not prove completion. Avoid repeated unchanged attempts; "
    "report a blocker or unknown result honestly. For substantive multi-step work, keep "
    "the working record with the task tool when it is available: the objective, the "
    "human's decisions, artifacts, references, and checks. A previous session's record is paused and out of view: look with the task tool's show when the person asks what work is here or to pick it up, and resume it only then."
)


def _head_prompt(role: str, *, native: bool, guardrails: str) -> str:
    from datetime import datetime
    from pathlib import Path
    now = datetime.now().astimezone()
    out = role + f"\n\nToday's date: {now.strftime('%A, %Y-%m-%d')} ({now.tzname()})"
    out += (
        f"\n\nWorking directory: {Path.cwd()}"
        "\nUnless the user names another location, create any files or folders you make "
        "under this working directory."
    )
    out += (
        "\n\nWhen you start a distinct phase of a multi-step task, first state what you are "
        'about to do in one short line (e.g. "Pulling the data…") before making the tool calls.'
    )
    out += _RULES
    if not native:
        out += (
            "\n\nWhen you have finished using tools, respond directly to the user. "
            "Be concise — give the answer, not your reasoning process."
        )
    if guardrails:
        out += "\n\n## Guardrails\n" + guardrails
    return out


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("guardrails", ["", "NEVER rm -rf"])
async def test_the_turns_leading_system_message_is_the_head_prompt(tmp_path, native, guardrails):
    from localharness.agent.loop import base_system_prompt
    from tests.conftest import FakeLLMResponse, MockLLMClient

    llm = MockLLMClient([FakeLLMResponse(content="Done.")])
    llm.config.tool_call_mode = "native" if native else "text"
    sent: list[str] = []
    original = llm.stream_complete

    async def recording(messages=None, **kw):
        sent.append(messages[0]["content"])
        return await original(messages=messages, **kw)

    llm.stream_complete = recording
    path = None
    if guardrails:
        path = tmp_path / "GUARDRAILS.md"
        path.write_text(guardrails, encoding="utf-8")
    cfg = AgentConfig(name="a", role="Test agent.", permissions=PermissionConfig(mode="unattended"))
    loop = AgentLoop(config=cfg, llm=llm, bus=EventBus(), context_manager=ContextManager(),
                     tool_registry=None, permission_evaluator=PermissionEvaluator(),
                     guardrails_path=path)
    await loop.run_turn("hi")

    expected = _head_prompt("Test agent.", native=native, guardrails=guardrails)
    assert sent[0] == expected
    assert base_system_prompt(cfg, tool_call_mode="native" if native else "text",
                              guardrails=guardrails) == expected


def test_the_loops_fixed_part_counts_the_shared_prompt_and_its_guardrails(tmp_path):
    path = tmp_path / "GUARDRAILS.md"
    path.write_text("RULES-G", encoding="utf-8")
    loop, counter = _loop(10)
    loop._guardrails_path = path
    loop.fixed_tokens()
    assert "## Guardrails\nRULES-G" in counter.seen[-1][0]["content"]
    assert counter.seen[-1][0]["content"].startswith("Test agent.\n\nToday's date: ")

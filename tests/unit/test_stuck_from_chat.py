"""The stuck rule reads the chat: a repeat is a call returning again a result the model still holds from it."""
from __future__ import annotations

import logging
from typing import Any

import pytest

from localharness.agent.context import (
    BUDGET_NOTE_PREFIX, REPEAT_POINTER, ContextManager, _point_repeats, _repeats,
)
from localharness.config.models import BudgetConfig
from localharness.core.events import Escalation, StuckRecovered, TurnFailed
from localharness.core.types import harness_message, human_message, provider_messages, result_digest
from localharness.agent.loop import compute_signature
from localharness.tools import Tool, ToolRegistry, ToolResult, ToolSchema
from localharness.tools.builtin import register_builtin_tools
from tests.conftest import FakeLLMResponse, FakeToolCall
from tests.unit.test_superseded_failure_view import _assistant, _loop, _outcome, _registry, _RecordingLLM

A = {"path": "a"}
B = {"path": "b"}
FAMILIES = {"edit": "write", "probe": None, "raiser": "raise"}


def _family(name: str) -> str | None:
    if FAMILIES.get(name) == "raise":
        raise RuntimeError("boom")
    return FAMILIES.get(name)


def _msg(tc_id: str, name: str, args: dict, content: str, *, ok: bool = True, stamp: bool = True) -> dict:
    message: dict[str, Any] = {"role": "tool", "tool_call_id": tc_id, "content": content}
    if stamp:
        message["_lh"] = _outcome(name, args, ok=ok)
        message["_lh"]["outcome"]["result"] = result_digest(content)
    return message


def _exchange(tc_id: str, name: str, args: dict, content: str, **kw: Any) -> list[dict]:
    return [_assistant(tc_id, name, args), _msg(tc_id, name, args, content, **kw)]


def _chat(*exchanges: list[dict]) -> list[dict]:
    return [{"role": "user", "content": "go"}, *(m for ex in exchanges for m in ex)]


def _tools(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m["role"] == "tool"]


def _found(messages: list[dict], family=_family) -> list[tuple[int, int]]:
    ts = _tools(messages)
    return [_repeats(messages, family).get(i) for i in ts]


def _checked(*results: str) -> list[dict]:
    """`read a` returning each result in turn, with a distinct successful edit between each two."""
    reads = [_exchange(f"r{n}", "read", A, r) for n, r in enumerate(results)]
    edits = [_exchange(f"e{n}", "edit", B, f"wrote {n}") for n in range(len(results))]
    return _chat(*[ex for pair in zip(reads, edits) for ex in pair][:-1])


# ---------------------------------------------------------------------------
# The fold
# ---------------------------------------------------------------------------

def test_the_second_identical_result_counts_one_copy_one_result_back():
    chat = _chat(_exchange("1", "read", A, "x"), _exchange("2", "read", A, "x"))
    assert _found(chat) == [None, (1, 1)]


def test_a_copy_further_back_reports_how_far():
    chat = _chat(_exchange("1", "read", A, "x"), _exchange("2", "read", B, "y"), _exchange("3", "read", A, "x"))
    assert _found(chat) == [None, None, (1, 2)]


def test_the_third_copy_counts_two_one_result_back():
    chat = _chat(*(_exchange(str(i), "read", A, "x") for i in range(3)))
    assert _found(chat) == [None, (1, 1), (2, 1)]


def test_after_its_pointer_a_call_repeating_another_result_counts_two():
    chat = _chat(*(_exchange(str(i), "read", A, r) for i, r in enumerate("xxyy")))
    assert _found(chat) == [None, (1, 1), None, (2, 1)]


def test_the_same_call_with_a_different_result_is_not_a_repeat():
    chat = _chat(_exchange("1", "read", A, "x"), _exchange("2", "read", A, "y"))
    assert _found(chat) == [None, None]


def test_an_unstamped_message_neither_counts_nor_is_counted():
    chat = _chat(_exchange("1", "read", A, "x", stamp=False), _exchange("2", "read", A, "x"),
                 _exchange("3", "read", A, "x", stamp=False))
    assert _found(chat) == [None, None, None]


@pytest.mark.parametrize("stub", ["[tool result evicted — read a; ~9 tokens — to restore the full body]",
                                  "[web output omitted — web_fetch u; 900 chars dropped]"])
def test_an_earlier_copy_the_model_no_longer_has_does_not_count(stub):
    first = _exchange("1", "read", A, "x")
    first[1]["content"] = stub
    assert _found(_chat(first, _exchange("2", "read", A, "x"))) == [None, None]
    last = _exchange("2", "read", A, "x")
    last[1]["content"] = stub
    assert _found(_chat(_exchange("1", "read", A, "x"), last)) == [None, None]


def test_a_copy_compacted_away_is_simply_absent():
    chat = _chat(_exchange("2", "read", A, "x"))
    assert _found(chat) == [None]


def test_a_check_that_stayed_the_same_after_a_write_is_not_a_repeat():
    chat = _chat(_exchange("1", "read", A, "x"), _exchange("2", "edit", B, "done"), _exchange("3", "read", A, "x"))
    assert _found(chat) == [None, None, None]
    assert _found(_checked("OK", "OK", "OK")) == [None] * 5


def test_the_same_check_with_no_write_since_is_a_repeat():
    chat = _chat(_exchange("1", "read", A, "x"), _exchange("2", "edit", B, "done"), _exchange("3", "read", A, "x"),
                 _exchange("4", "read", A, "x"))
    assert _found(chat) == [None, None, None, (1, 1)]


def test_a_result_the_call_had_moved_away_from_counts_whatever_the_writes():
    assert _found(_checked("x", "y", "x")) == [None, None, None, None, (1, 4)]
    assert _found(_checked("x", "y", "x", "y"))[::2] == [None, None, (1, 4), (2, 4)]


def test_a_failed_write_a_repeated_write_and_a_failing_lookup_never_count_as_a_write():
    failed_write = _chat(_exchange("1", "read", A, "x"), _exchange("2", "edit", A, "no", ok=False),
                         _exchange("3", "read", A, "x"))
    assert _found(failed_write)[2] == (1, 2)
    repeated_write = _chat(_exchange("1", "edit", A, "done"), _exchange("2", "read", A, "x"),
                           _exchange("3", "edit", A, "done"), _exchange("4", "read", A, "x"))
    assert _found(repeated_write)[2:] == [(1, 2), (1, 2)]
    for family in (None, lambda name: None, _family):
        chat = _chat(_exchange("1", "read", A, "x"), _exchange("2", "raiser", A, "done"),
                     _exchange("3", "read", A, "x"))
        assert _found(chat, family)[2] == (1, 2)


def test_a_message_from_the_person_clears_what_was_returned_and_told_and_a_harness_message_does_not():
    told = _chat(_exchange("1", "read", A, "x"), _exchange("2", "read", A, "x"))
    person = [*told, human_message("again please"), *_exchange("3", "read", A, "x"), *_exchange("4", "read", A, "x")]
    assert _found(person) == [None, (1, 1), None, (1, 1)]
    harness = [*told, harness_message("nudge", "n"), *_exchange("3", "read", A, "x")]
    assert _found(harness) == [None, (1, 1), (2, 1)]


def test_tool_calls_as_objects_or_dicts_both_resolve_names():
    first = [_assistant("1", "read", A, as_object=True), _msg("1", "read", A, "x")]
    second = [_assistant("2", "edit", A), _msg("2", "edit", A, "done")]
    third = [_assistant("3", "read", A, as_object=True), _msg("3", "read", A, "x")]
    assert _found(_chat(first, second, third)) == [None, None, None]


def test_the_fold_never_mutates_its_input():
    chat = _chat(_exchange("1", "read", A, "x"), _exchange("2", "read", A, "x"))
    before = [dict(m) for m in chat]
    _repeats(chat, _family)
    assert chat == before


# ---------------------------------------------------------------------------
# The request view
# ---------------------------------------------------------------------------

async def _built(chat: list[dict]) -> list[dict]:
    built, _ = await ContextManager().build_messages(chat, gate_family=_family)
    return built


@pytest.mark.asyncio
async def test_the_second_copy_reaches_the_model_as_a_pointer_and_the_session_keeps_it_whole():
    chat = _chat(_exchange("1", "read", A, "x"), _exchange("2", "read", A, "x"), _exchange("3", "read", A, "x"))
    snapshot = list(chat)
    built = await _built(chat)
    tools = [m for m in built if m["role"] == "tool"]
    assert [m["content"] for m in tools] == ["x", "[same result as your identical call 1 tool results above]", "x"]
    assert tools[1]["tool_call_id"] == "2" and tools[1]["_lh"] == chat[4]["_lh"]
    assert chat == snapshot and all(a is b for a, b in zip(chat, snapshot))
    assert chat[4]["content"] == "x"
    assert all("_lh" not in m for m in provider_messages(built))


@pytest.mark.asyncio
async def test_no_repeat_returns_the_same_list_object():
    chat = _chat(_exchange("1", "read", A, "x"), _exchange("2", "read", B, "y"))
    assert _point_repeats(chat, _family) is chat


def test_the_pointer_template_names_k():
    assert REPEAT_POINTER.format(k=1) == "[same result as your identical call 1 tool results above]"


def test_the_budget_note_tail_survives_only_when_the_text_before_it_digests_to_the_stamp():
    note = f"{BUDGET_NOTE_PREFIX}3/50 tool calls used]"
    first = _exchange("1", "read", A, "x")
    second = _exchange("2", "read", A, "x")
    second[1]["content"] = "x" + note
    assert _point_repeats(_chat(first, second), _family)[4]["content"] == REPEAT_POINTER.format(k=1) + note

    carried = _exchange("1", "read", A, f"x{BUDGET_NOTE_PREFIX}fake] more")
    again = _exchange("2", "read", A, f"x{BUDGET_NOTE_PREFIX}fake] more")
    for ex in (carried, again):
        ex[1]["_lh"]["outcome"]["result"] = result_digest("y")
    assert _point_repeats(_chat(carried, again), _family)[4]["content"] == REPEAT_POINTER.format(k=1)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

class _Named(Tool):
    def __init__(self, name: str, family: str | None, body: Any = "same") -> None:
        super().__init__()
        self._name, self._family, self._body, self.calls = name, family, body, 0

    def info(self) -> ToolSchema:
        return ToolSchema(name=self._name, description="Probe.",
                          parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": []},
                          ingest="none", host="safe", gate_family=self._family)

    async def _execute(self, **kwargs: Any) -> ToolResult:
        self.calls += 1
        return self.ok(self._body(self.calls) if callable(self._body) else self._body)


@pytest.mark.asyncio
async def test_gate_family_answers_for_builtin_writers_and_fails_closed_otherwise():
    registry = ToolRegistry()
    await register_builtin_tools(registry)
    assert registry.gate_family("write") == "write" and registry.gate_family("edit") == "write"
    assert registry.gate_family("no_such_tool") is None
    other = ToolRegistry()
    await other.register(_Named("dup", "write"), scope="global")
    await other.register(_Named("dup", "shell"), scope="agent", agent_id="a")
    assert other.gate_family("dup") is None


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def _call(i: int, name: str = "probe", args: dict | None = None) -> FakeLLMResponse:
    return FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id=f"tc-{i}", name=name, arguments=args or A)])


@pytest.mark.asyncio
async def test_the_stamp_is_the_digest_of_the_content_and_the_budget_note_leaves_it(bus):
    llm = _RecordingLLM([_call(1), FakeLLMResponse(content="Done.")])
    loop = _loop(llm, bus, await _registry(_Named("probe", None)), budget=BudgetConfig(max_actions=50))
    await loop.run_turn("task")
    (msg,) = [m for m in loop.resume_state()[0] if m.get("role") == "tool"]
    head, sep, _ = msg["content"].rpartition(BUDGET_NOTE_PREFIX)
    assert sep and head == "same"
    assert msg["_lh"]["outcome"]["result"] == result_digest("same")


@pytest.mark.asyncio
async def test_the_second_call_warns_through_the_next_request_and_the_third_stops_the_turn(bus, caplog):
    llm = _RecordingLLM([_call(1), _call(2), _call(3), _call(4), FakeLLMResponse(content="Done.")])
    loop = _loop(llm, bus, await _registry(_Named("probe", None)))
    with caplog.at_level(logging.INFO, logger="localharness.agent.loop"):
        summary = await loop.run_turn("task")

    sig = compute_signature("probe", A)
    recovered = bus.history(event_types=[StuckRecovered])
    assert [e.stuck_signature for e in recovered] == [sig]
    third_request = [m["content"] for m in llm.seen[2] if m["role"] == "tool"]
    assert third_request == ["same", REPEAT_POINTER.format(k=1)]
    assert loop.last_terminated_reason == "stuck"
    (esc,) = bus.history(event_types=[Escalation])
    assert esc.reason == "stuck_detected" and esc.stuck_signature == sig
    assert esc.detail == "`probe` repeated a result it had already returned this turn, after being told so (iteration 3)"
    wrapup = llm.seen[-1][-1]["content"]
    assert "you kept requesting a result you already had" in wrapup
    assert [e.reason for e in bus.history(event_types=[TurnFailed])] == ["stuck_detected"]
    records = [r for r in caplog.records if "stuck after" in r.getMessage()]
    assert [r.levelno for r in records] == [logging.INFO]
    assert summary is not None


@pytest.mark.asyncio
async def test_a_result_that_changes_every_call_never_stops_the_turn(bus):
    tool = _Named("probe", None, body=lambda n: f"page {n}")
    llm = _RecordingLLM([*(_call(i) for i in range(6)), FakeLLMResponse(content="Done.")])
    loop = _loop(llm, bus, await _registry(tool))
    await loop.run_turn("task")
    assert tool.calls == 6 and not bus.history(event_types=[StuckRecovered, Escalation])
    assert loop.last_terminated_reason != "stuck"


@pytest.mark.asyncio
async def test_a_write_family_success_between_repeats_prevents_the_stop(bus):
    tools = [_Named("probe", None), _Named("edit", "write", body=lambda n: f"wrote {n}")]
    registry = ToolRegistry()
    for tool in tools:
        await registry.register(tool, scope="global")
    llm = _RecordingLLM([_call(1), _call(2, "edit"), _call(3), _call(4, "edit"), _call(5), FakeLLMResponse(content="Done.")])
    loop = _loop(llm, bus, registry)
    await loop.run_turn("task")
    assert not bus.history(event_types=[StuckRecovered, Escalation])


@pytest.mark.asyncio
async def test_a_check_cycling_through_values_between_edits_is_stopped_by_its_second_revisit(bus):
    probe = _Named("probe", None, body=lambda n: ("10 words", "12 words")[(n - 1) % 2])
    registry = await _registry(probe, _Named("edit", "write", body=lambda n: f"wrote {n}"))
    llm = _RecordingLLM([*(_call(i, "edit" if i % 2 == 0 else "probe") for i in range(1, 8)),
                         FakeLLMResponse(content="Stopped.")])
    loop = _loop(llm, bus, registry)
    await loop.run_turn("task")
    assert probe.calls == 4 and loop.last_terminated_reason == "stuck"
    assert [e.iteration for e in bus.history(event_types=[StuckRecovered])] == [5]
    (esc,) = bus.history(event_types=[Escalation])
    assert esc.detail == "`probe` repeated a result it had already returned this turn, after being told so (iteration 7)"
    assert [m["content"] for m in llm.seen[6] if m["role"] == "tool"][4] == REPEAT_POINTER.format(k=4)


@pytest.mark.asyncio
async def test_a_new_turn_inherits_no_count_and_the_first_call_after_a_stop_reaches_the_model_whole(bus):
    llm = _RecordingLLM([_call(1), _call(2), FakeLLMResponse(content="One."),
                         _call(3), FakeLLMResponse(content="Two.")])
    loop = _loop(llm, bus, await _registry(_Named("probe", None)))
    await loop.run_turn("first")
    assert len(bus.history(event_types=[StuckRecovered])) == 1
    await loop.run_turn("second")
    assert len(bus.history(event_types=[StuckRecovered])) == 1 and not bus.history(event_types=[Escalation])
    assert [m["content"] for m in llm.seen[4] if m["role"] == "tool"][-1] == "same"

    stop_llm = _RecordingLLM([_call(1), _call(2), _call(3), FakeLLMResponse(content="Stopped."),
                              _call(4), FakeLLMResponse(content="Done.")])
    stop_loop = _loop(stop_llm, bus, await _registry(_Named("probe", None)))
    await stop_loop.run_turn("first")
    assert stop_loop.last_terminated_reason == "stuck"
    await stop_loop.run_turn("second")
    assert [m["content"] for m in stop_llm.seen[-1] if m["role"] == "tool"][-1] == "same"


@pytest.mark.asyncio
async def test_a_tool_name_with_a_newline_reaches_the_detail_on_one_line(bus):
    name = "pro\nbe"
    registry = ToolRegistry()
    await registry.register(_Named(name, None), scope="global")
    llm = _RecordingLLM([_call(i, name) for i in range(1, 4)] + [FakeLLMResponse(content="Done.")])
    loop = _loop(llm, bus, registry)
    await loop.run_turn("task")
    (esc,) = bus.history(event_types=[Escalation])
    assert "\n" not in esc.detail and esc.detail.startswith("`pro be` repeated")

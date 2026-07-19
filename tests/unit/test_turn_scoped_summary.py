"""#91 — turn-scoped completion-summary fallback + protocol hygiene.

The CONFIRMED-sentinel fallback (`_format_completion_summary`) must resolve the answer the
sentinel confirms from THIS turn ONLY — never walk the full multi-turn history and splice a
prior turn's reply (the live #91 receipt: a turn whose both iterations were literally
"CONFIRMED" got a prior turn's 167-char reply shipped as its summary). When a turn produced no
real assistant content at all, one bounded re-prompt is issued; if that still yields a sentinel,
the honest "No answer was produced this turn." stands.

Protocol hygiene (#91b): the deterministic nudge -> bare-CONFIRMED pair must not persist into the
cross-turn conversation as imitable precedent (a small local model copies it, spontaneously
emitting CONFIRMED next turn — the splice trigger).

Offline: content-only scripted LLM at the natural-completion seam; no live model.
"""
from __future__ import annotations

import pytest

from localharness.agent.context import ContextManager
from localharness.agent.loop import (
    AgentLoop,
    Session,
    _format_completion_summary,
    _is_confirmation,
)
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig


class _ScriptedNoToolLLM:
    """Content-only script, one entry per LLM round-trip; never emits tool calls."""

    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self.calls = 0

        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.tool_call_mode = "native"
        self.config.context_window = 128000

    async def stream_complete(self, messages=None, tools=None, on_token=None):
        from types import SimpleNamespace as NS

        self.calls += 1
        idx = min(self.calls, len(self._contents)) - 1
        return NS(content=self._contents[idx], tool_calls=None), None


def _make_loop(llm, bus, *, self_check: dict | None = None) -> AgentLoop:
    overrides = {"name": "sum-agent", "role": "Test agent."}
    if self_check is not None:
        overrides["self_check"] = self_check
    cfg = AgentConfig.model_validate(overrides)
    return AgentLoop(
        config=cfg,
        llm=llm,
        bus=bus,
        context_manager=ContextManager(),
        tool_registry=None,
        permission_evaluator=PermissionEvaluator(),
    )


def _reprompts(session: Session) -> list[dict]:
    return [
        m
        for m in session.messages
        if m.get("role") == "user"
        and "nothing in this turn to confirm" in (m.get("content") or "")
    ]


# ---------------------------------------------------------------------------
# _format_completion_summary — turn-scoped fallback (never splices a prior turn)
# ---------------------------------------------------------------------------


def test_sentinel_fallback_never_splices_prior_turn():
    """The core #91 bug: both this turn's iterations are CONFIRMED and there is a prior-turn
    answer in history. The summary must NOT be the prior turn's reply."""
    s = Session(
        agent_id="a",
        session_id="s",
        messages=[
            {"role": "user", "content": "turn-1 question"},
            {"role": "assistant", "content": "PRIOR-TURN ANSWER that must never be spliced in."},
            {"role": "user", "content": "turn-2 question"},
            {"role": "assistant", "content": "CONFIRMED"},          # this turn, iter 1
            {"role": "user", "content": "Review your answer..."},
            {"role": "assistant", "content": "CONFIRMED"},          # this turn, iter 2
        ],
    )
    s.turn_start_idx = 3  # this turn's first assistant reply
    summary = _format_completion_summary(s, "CONFIRMED")
    assert "PRIOR-TURN ANSWER" not in summary
    assert summary == "No answer was produced this turn."


def test_sentinel_fallback_surfaces_in_turn_answer():
    """A sentinel that confirms a real in-turn answer surfaces that answer (unchanged behavior)."""
    s = Session(
        agent_id="a",
        session_id="s",
        messages=[
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "prior answer"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "THIS-TURN ANSWER"},
            {"role": "user", "content": "Review your answer..."},
            {"role": "assistant", "content": "CONFIRMED"},
        ],
    )
    s.turn_start_idx = 3
    assert _format_completion_summary(s, "CONFIRMED") == "THIS-TURN ANSWER"


def test_real_content_returned_verbatim():
    """Non-sentinel content is returned as-is, turn scope irrelevant."""
    s = Session(agent_id="a", session_id="s", messages=[])
    s.turn_start_idx = 0
    assert _format_completion_summary(s, "A real answer.") == "A real answer."


# ---------------------------------------------------------------------------
# Loop: a turn that only ever emits sentinels re-prompts ONCE, then is honest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_double_confirmed_reprompts_once_then_honest(bus):
    """Both iterations are the bare sentinel with nothing to confirm: one bounded re-prompt,
    then the honest 'No answer was produced this turn.' — never a prior turn, never a loop."""
    llm = _ScriptedNoToolLLM(["CONFIRMED", "CONFIRMED"])  # self_check OFF (default)
    loop = _make_loop(llm, bus)
    session = Session(agent_id="sum-agent", session_id="s-dc", messages=[])

    summary = await loop._execute_loop(session, "do the task", None)

    assert summary == "No answer was produced this turn."
    assert session.terminated_reason == "complete"
    assert len(_reprompts(session)) == 1  # exactly one bounded re-prompt (flag-guarded)


@pytest.mark.asyncio
async def test_reprompt_recovers_a_real_answer(bus):
    """The bounded re-prompt gives the model one chance to actually answer; a real answer then
    stands as the summary."""
    llm = _ScriptedNoToolLLM(["CONFIRMED", "The capital of France is Paris."])
    loop = _make_loop(llm, bus)
    session = Session(agent_id="sum-agent", session_id="s-rec", messages=[])

    summary = await loop._execute_loop(session, "capital of France?", None)

    assert summary == "The capital of France is Paris."
    assert len(_reprompts(session)) == 1


@pytest.mark.asyncio
async def test_confirmed_with_in_turn_answer_does_not_reprompt(bus):
    """When the sentinel confirms a real in-turn answer, NO #91 re-prompt fires (self-check path
    stays intact — mirrors test_agent_loop_selfcheck)."""
    llm = _ScriptedNoToolLLM(["The answer is 42.", "CONFIRMED"])
    loop = _make_loop(llm, bus, self_check={"enabled": True, "max_passes": 1})
    session = Session(agent_id="sum-agent", session_id="s-ok", messages=[])

    summary = await loop._execute_loop(session, "meaning of life?", None)

    assert summary == "The answer is 42."
    assert _reprompts(session) == []


# ---------------------------------------------------------------------------
# #91b protocol hygiene — the nudge -> CONFIRMED pair does not persist
# ---------------------------------------------------------------------------


def test_strip_sentinel_exchanges_drops_nudge_confirmed_pair():
    from localharness.agent.loop import _strip_sentinel_exchanges

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "the task"},
        {"role": "assistant", "content": "The answer is 42."},
        {"role": "user", "content": "Review your answer above..."},
        {"role": "assistant", "content": "CONFIRMED"},
    ]
    out = _strip_sentinel_exchanges(msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant"]
    assert out[-1]["content"] == "The answer is 42."
    assert not any(_is_confirmation(m.get("content")) for m in out)
    assert msgs[-1]["content"] == "CONFIRMED"  # input list not mutated


def test_strip_sentinel_exchanges_keeps_plain_history():
    from localharness.agent.loop import _strip_sentinel_exchanges

    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]
    assert _strip_sentinel_exchanges(msgs) == msgs


@pytest.mark.asyncio
async def test_conversation_persist_strips_confirmed_pair(bus):
    """After a self-check turn ending in CONFIRMED, the persisted cross-turn conversation keeps
    the real answer but drops the review-nudge + CONFIRMED pair (imitable precedent, #91b)."""
    llm = _ScriptedNoToolLLM(["The answer is 42.", "CONFIRMED"])
    loop = _make_loop(llm, bus, self_check={"enabled": True, "max_passes": 1})
    session = Session(agent_id="sum-agent", session_id="s-persist", messages=[])

    await loop._execute_loop(session, "q", None)

    conv = loop._conversation
    assert any(m["role"] == "assistant" and m.get("content") == "The answer is 42." for m in conv)
    assert not any(m["role"] == "assistant" and _is_confirmation(m.get("content")) for m in conv)
    assert not any(
        m["role"] == "user" and "Review your answer" in (m.get("content") or "") for m in conv
    )


# ---------------------------------------------------------------------------
# #98: instruction-echo sentinel — a literal model quotes the nudge's own
# delivery tail back instead of the bare word; the echo is still the sentinel
# (live 2026-07-18: the echo shipped as the terminal answer, 2/2 turns)
# ---------------------------------------------------------------------------

_LIVE_ECHO = "CONFIRMED — my previous reply will be delivered to the user unchanged."


@pytest.mark.parametrize(
    "echo",
    [
        _LIVE_ECHO,                                                                 # live trace, verbatim
        "CONFIRMED — your previous reply will be delivered to the user unchanged",   # pronoun as instructed
        "confirmed - my previous reply will be delivered to the user unchanged.",    # ascii hyphen, lowercase
        "CONFIRMED. Your previous reply will then be delivered to the user unchanged.",  # reworded-nudge echo
    ],
)
def test_is_confirmation_tolerates_instruction_echo(echo):
    assert _is_confirmation(echo)


@pytest.mark.parametrize(
    "content",
    [
        "Confirmed: your flight is booked.",      # real answer that starts with the word
        "CONFIRMED — I'll now check the file.",   # announced work, not the delivery tail
        _LIVE_ECHO + " The answer is 42.",        # echo + real content stays content
    ],
)
def test_is_confirmation_rejects_real_content(content):
    assert not _is_confirmation(content)


def test_echoed_sentinel_surfaces_this_turns_real_answer():
    """#98 live shape: [question, real answer, nudge, echo] — the summary must be the
    real answer, never the echoed meta-line."""
    s = Session(
        agent_id="a",
        session_id="s",
        messages=[
            {"role": "user", "content": "turn question"},
            {"role": "assistant", "content": "THE REAL ANSWER."},
            {"role": "user", "content": "You ended your reply with stated intentions..."},
            {"role": "assistant", "content": _LIVE_ECHO},
        ],
    )
    s.turn_start_idx = 1
    assert _format_completion_summary(s, _LIVE_ECHO) == "THE REAL ANSWER."


@pytest.mark.asyncio
async def test_persist_strips_echoed_confirmed_pair(bus):
    """#91b hygiene must cover the echo: after a self-check turn ending in the ECHOED
    sentinel, the summary is the real answer and the nudge + echo pair is dropped from
    the persisted conversation."""
    llm = _ScriptedNoToolLLM(["The answer is 42.", _LIVE_ECHO])
    loop = _make_loop(llm, bus, self_check={"enabled": True, "max_passes": 1})
    session = Session(agent_id="sum-agent", session_id="s-persist-echo", messages=[])

    summary = await loop._execute_loop(session, "q", None)

    assert summary == "The answer is 42."
    conv = loop._conversation
    assert any(m["role"] == "assistant" and m.get("content") == "The answer is 42." for m in conv)
    assert not any(
        m["role"] == "assistant" and "CONFIRMED" in (m.get("content") or "") for m in conv
    )
    assert not any(
        m["role"] == "user" and "Review your answer" in (m.get("content") or "") for m in conv
    )

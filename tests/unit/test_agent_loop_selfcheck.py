"""Phase 25 (MECH-01) — bounded self-check step at the loop natural-completion seam.

Proves the new `agent.self_check` mechanism axis is a GENUINE loop-structure change, not a
no-op: OFF (default) natural-completes in one round-trip (byte-identical to pre-change); ON
injects exactly ONE bounded "Review your answer" user-turn per pass and re-enters the while-loop
(+1 iteration per pass), bounded by max_passes (ge=1,le=3) so it provably terminates.

Offline: drives a real AgentLoop with FaithfulFakeLLM(tool_plan=[]) — that fake emits a final
answer (content set, tool_calls=[]) on every stream_complete, so the loop always hits the
`if not tool_calls:` natural-completion seam where the self-check block lives. No live model.
"""
from __future__ import annotations

import pytest

from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop, Session
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig


def _make_loop(llm, bus, *, self_check: dict | None = None) -> AgentLoop:
    """Construct a real AgentLoop with offline deps (mirrors test_agent_loop._make_agent_loop)."""
    overrides = {"name": "selfcheck-agent", "role": "Test agent."}
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


def _review_turns(session: Session) -> list[dict]:
    """The injected bounded-review user-turns in the session transcript."""
    return [
        m
        for m in session.messages
        if m.get("role") == "user" and "Review your answer" in (m.get("content") or "")
    ]


@pytest.mark.asyncio
async def test_self_check_off_finalizes_immediately(faithful_fake_llm, bus):
    """Test A: OFF (default) — natural-completes in one round-trip, no review turn (byte-identical)."""
    loop = _make_loop(faithful_fake_llm(tool_plan=[]), bus)  # self_check defaults: enabled=False
    session = Session(agent_id="selfcheck-agent", session_id="s-off", messages=[])

    summary = await loop._execute_loop(session, "do the task", None)

    assert session.iteration == 1
    assert session.terminated_reason == "complete"
    assert _review_turns(session) == []
    assert isinstance(summary, str)


@pytest.mark.asyncio
async def test_self_check_on_adds_one_iteration(faithful_fake_llm, bus):
    """Test B: ON with max_passes=1 — exactly +1 iteration (one review pass), then finalizes."""
    loop = _make_loop(faithful_fake_llm(tool_plan=[]), bus, self_check={"enabled": True, "max_passes": 1})
    session = Session(agent_id="selfcheck-agent", session_id="s-on1", messages=[])

    summary = await loop._execute_loop(session, "do the task", None)

    assert session.iteration == 2  # one extra LLM round-trip from the review pass
    assert session.terminated_reason == "complete"  # NOT budget_/error — provably terminates
    assert len(_review_turns(session)) == 1
    assert isinstance(summary, str)


@pytest.mark.asyncio
async def test_self_check_bounded_by_max_passes(faithful_fake_llm, bus):
    """Test C: ON with max_passes=2 — exactly two review passes (+2 iterations), then a forced finalize."""
    loop = _make_loop(faithful_fake_llm(tool_plan=[]), bus, self_check={"enabled": True, "max_passes": 2})
    session = Session(agent_id="selfcheck-agent", session_id="s-on2", messages=[])

    await loop._execute_loop(session, "do the task", None)

    assert session.iteration == 3  # 1 initial + 2 review round-trips
    assert session.terminated_reason == "complete"
    # The review user-turn fires exactly max_passes times — never an unbounded review loop.
    assert len(_review_turns(session)) == 2


@pytest.mark.asyncio
async def test_self_check_review_turn_is_user_role_bounded_text(faithful_fake_llm, bus):
    """Test D: the injected review message is a USER turn with the bounded review prompt.

    vLLM rejects mid-conversation system messages, so it MUST be a user turn (mirrors the
    recovery-injection idiom at loop.py:548-551).
    """
    loop = _make_loop(faithful_fake_llm(tool_plan=[]), bus, self_check={"enabled": True, "max_passes": 1})
    session = Session(agent_id="selfcheck-agent", session_id="s-role", messages=[])

    await loop._execute_loop(session, "do the task", None)

    reviews = _review_turns(session)
    assert len(reviews) == 1
    assert reviews[0]["role"] == "user"
    assert "Review your answer" in reviews[0]["content"]


# ---------------------------------------------------------------------------
# CONFIRMED sentinel — the review reply must never become the user-facing answer
# (observed live 2026-07-02: Qwen answers review/nudge turns with "I already
# provided the answer" meta text, which then shipped as the TaskComplete summary).
# ---------------------------------------------------------------------------


class _ScriptedNoToolLLM:
    """Content-only script, one entry per LLM round-trip; never emits tool calls."""

    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self.calls = 0
        class _Cfg: pass
        self.config = _Cfg(); self.config.tool_call_mode = "native"; self.config.context_window = 128000

    async def stream_complete(self, messages=None, tools=None, on_token=None):
        from types import SimpleNamespace as NS
        self.calls += 1
        idx = min(self.calls, len(self._contents)) - 1
        return NS(content=self._contents[idx], tool_calls=None), None


@pytest.mark.asyncio
async def test_self_check_confirmed_returns_reviewed_answer(bus):
    """Sentinel reply → summary is the answer that was confirmed, not 'CONFIRMED'."""
    llm = _ScriptedNoToolLLM(["The capital of France is Paris.", "CONFIRMED"])
    loop = _make_loop(llm, bus, self_check={"enabled": True, "max_passes": 1})
    session = Session(agent_id="selfcheck-agent", session_id="s-conf", messages=[])

    summary = await loop._execute_loop(session, "capital of France?", None)

    assert summary == "The capital of France is Paris."
    assert session.terminated_reason == "complete"


@pytest.mark.asyncio
async def test_self_check_confirmed_case_and_punct(bus):
    """'Confirmed.' variants count as the sentinel; walk-back still finds the answer."""
    llm = _ScriptedNoToolLLM(["Answer: 42.", "Confirmed."])
    loop = _make_loop(llm, bus, self_check={"enabled": True, "max_passes": 1})
    session = Session(agent_id="selfcheck-agent", session_id="s-conf2", messages=[])

    summary = await loop._execute_loop(session, "meaning of life?", None)

    assert summary == "Answer: 42."


@pytest.mark.asyncio
async def test_self_check_correction_replaces_answer(bus):
    """A substantive (non-sentinel) review reply stands as the corrected final answer."""
    llm = _ScriptedNoToolLLM(["Paris is in Germany.", "Paris is in France."])
    loop = _make_loop(llm, bus, self_check={"enabled": True, "max_passes": 1})
    session = Session(agent_id="selfcheck-agent", session_id="s-corr", messages=[])

    summary = await loop._execute_loop(session, "where is Paris?", None)

    assert summary == "Paris is in France."

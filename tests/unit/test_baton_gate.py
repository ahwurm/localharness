"""Deterministic baton gate at the loop's tool-less acceptance seam (issue #84).

A tool-less reply whose CLOSING move announces further work ("Now let me read X") instead of
doing it or stating a final answer is a DROPPED BATON — the model narrates a next step and the
turn ends as if done. The act-guard only arms at zero actions, so an announce-AFTER-work reply
slips through. The gate detects that closing move and pushes ONE bounded nudge, then accepts
(bounded once per turn). detect_dropped_baton is a pure, high-precision text function.
"""
from __future__ import annotations

import pytest

from localharness.agent.loop import detect_dropped_baton


# --- detector: positives (the closing move announces the next step) ------------------------
_POSITIVES = [
    "I've read the files. Now let me read the notebooks.",
    "Now let me read the notebooks…",            # unicode ellipsis, no trailing period
    "Let me now examine the outputs.",
    "Next I'll check the logs.",
    "Next, I will verify the config.",
    "I'll now summarize the findings.",
    "I will now investigate the root cause.",
    "Now I'll dig into the configuration.",
    "Here is the plan:\n\nNow let me start reading the first file.",   # multi-line, closing announces
    "- Now let me read the config file.",             # leading bullet on the closing line
]


# --- detector: negatives (MUST NOT fire — a false positive wastes a round-trip) ------------
_NEGATIVES = [
    "Let me know if you need anything else.",         # closing courtesy, not an announce
    "Should I proceed?",                              # a handback question to the user
    "Now let me read the config, or should I proceed differently?",  # ends by asking the user
    "Now let me check the notebooks. They contain the training outputs showing 92% accuracy.",  # announce mid-reply, real content after
    "I now understand the architecture: it uses a hierarchical store.",  # 'I now' != 'now I'll'
    "The answer is 42.",
    "CONFIRMED",
    "no tool result",                                # FaithfulFakeLLM's empty-plan final answer
    "",
    "   ",
    "Let me check the logs.",                        # bare 'let me X' is not a target shape
]


@pytest.mark.parametrize("text", _POSITIVES)
def test_detect_dropped_baton_positive(text):
    assert detect_dropped_baton(text) is True


@pytest.mark.parametrize("text", _NEGATIVES)
def test_detect_dropped_baton_negative(text):
    assert detect_dropped_baton(text) is False


# --- gate behaviour at the acceptance seam -------------------------------------------------
from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop, Session
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig


class _ScriptedNoToolLLM:
    """Content-only script, one entry per LLM round-trip; never emits tool calls
    (mirrors the self-check suite's fake). tool_registry=None -> no tool_schemas -> the act-guard
    is out of the way, so the baton gate is exercised in isolation."""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0
        class _Cfg: pass
        self.config = _Cfg(); self.config.tool_call_mode = "native"; self.config.context_window = 128000

    async def stream_complete(self, messages=None, tools=None, on_token=None):
        from types import SimpleNamespace as NS
        self.calls += 1
        idx = min(self.calls, len(self._contents)) - 1
        return NS(content=self._contents[idx], tool_calls=None), None


def _make_loop(llm, bus, *, self_check=None, baton_gate=None):
    overrides = {"name": "baton-agent", "role": "Test agent."}
    if self_check is not None:
        overrides["self_check"] = self_check
    if baton_gate is not None:
        overrides["baton_gate"] = baton_gate
    cfg = AgentConfig.model_validate(overrides)
    return AgentLoop(config=cfg, llm=llm, bus=bus, context_manager=ContextManager(),
                     tool_registry=None, permission_evaluator=PermissionEvaluator())


def _baton_nudges(session):
    return [m for m in session.messages if m.get("role") == "user"
            and "announcing further work" in (m.get("content") or "")]


@pytest.mark.asyncio
async def test_baton_gate_fires_once_then_accepts(bus):
    """A tool-less announced-next-step reply gets ONE nudge; the following reply is accepted."""
    llm = _ScriptedNoToolLLM(["Now let me read the notebooks.", "The notebooks show 92% accuracy."])
    loop = _make_loop(llm, bus)  # baton_gate defaults: enabled=True
    session = Session(agent_id="baton-agent", session_id="s-fire", messages=[])
    summary = await loop._execute_loop(session, "analyze", None)
    assert session.baton_nudge_used is True
    assert len(_baton_nudges(session)) == 1
    assert session.iteration == 2
    assert session.terminated_reason == "complete"
    assert summary == "The notebooks show 92% accuracy."


@pytest.mark.asyncio
async def test_baton_gate_bounded_second_announcement_accepted(bus):
    """Bounded once per turn: if the reply AFTER the nudge still announces, accept it (no loop)."""
    llm = _ScriptedNoToolLLM(["Now let me read the notebooks.", "Now let me also read the configs."])
    loop = _make_loop(llm, bus)
    session = Session(agent_id="baton-agent", session_id="s-bound", messages=[])
    summary = await loop._execute_loop(session, "analyze", None)
    assert len(_baton_nudges(session)) == 1               # exactly one nudge, never a loop
    assert session.iteration == 2
    assert summary == "Now let me also read the configs."  # 2nd announce accepted verbatim


@pytest.mark.asyncio
async def test_baton_gate_off_restores_verbatim_accept(bus):
    """OFF -> the pre-fix behavior: the announce reply is accepted immediately, no nudge."""
    llm = _ScriptedNoToolLLM(["Now let me read the notebooks."])
    loop = _make_loop(llm, bus, baton_gate={"enabled": False})
    session = Session(agent_id="baton-agent", session_id="s-off", messages=[])
    summary = await loop._execute_loop(session, "analyze", None)
    assert session.baton_nudge_used is False
    assert _baton_nudges(session) == []
    assert session.iteration == 1
    assert summary == "Now let me read the notebooks."


@pytest.mark.asyncio
async def test_baton_gate_composes_with_self_check(bus):
    """Order: baton gate FIRST, then self_check. Both bounded -> +1 (baton) +1 (self_check)."""
    llm = _ScriptedNoToolLLM([
        "Now let me read the notebooks.",    # -> baton nudge (iter 1)
        "The notebooks show 92% accuracy.",  # -> self_check review nudge (iter 2)
        "CONFIRMED",                         # -> accept the confirmed answer (iter 3)
    ])
    loop = _make_loop(llm, bus, self_check={"enabled": True, "max_passes": 1})
    session = Session(agent_id="baton-agent", session_id="s-compose", messages=[])
    summary = await loop._execute_loop(session, "analyze", None)
    assert len(_baton_nudges(session)) == 1
    reviews = [m for m in session.messages if m.get("role") == "user"
               and "Review your answer" in (m.get("content") or "")]
    assert len(reviews) == 1
    assert session.iteration == 3
    assert summary == "The notebooks show 92% accuracy."   # the confirmed answer, not "CONFIRMED"


@pytest.mark.asyncio
async def test_baton_gate_fires_after_a_real_action(mock_llm_client, bus, tmp_path):
    """The exact bug: a reply announcing further work AFTER taking an action (so the act-guard,
    which arms only at zero actions, cannot fire) is caught by the baton gate — not the act-guard."""
    from localharness.tools.builtin import register_builtin_tools
    from localharness.tools.registry import ToolRegistry
    full = ToolRegistry(); await register_builtin_tools(full)
    # glob-only registry: a full builtin set would trip the capability floor (web + bash
    # co-residence). One read-only tool is all we need to take a real action.
    reg = ToolRegistry.from_allowed(["glob"], base_registry=full)
    (tmp_path / "a.md").write_text("hi", encoding="utf-8")
    R = mock_llm_client
    llm = mock_llm_client([
        R.Response(content=None, tool_calls=[R.ToolCall(
            id="c1", name="glob", arguments={"pattern": str(tmp_path / "*.md")})]),
        R.Response(content="Now let me read the results."),
        R.Response(content="Found 1 markdown file."),
    ])
    cfg = AgentConfig(name="baton-agent", role="Test agent.")
    loop = AgentLoop(config=cfg, llm=llm, bus=bus, context_manager=ContextManager(),
                     tool_registry=reg, permission_evaluator=PermissionEvaluator())
    session = Session(agent_id="baton-agent", session_id="s-real", messages=[])
    summary = await loop._execute_loop(session, "find files", None)
    assert session.actions_taken == 1        # the glob ran -> the act-guard could NOT fire
    assert session.act_nudge_used is False   # so it was the BATON gate that caught the announce
    assert session.baton_nudge_used is True
    assert len(_baton_nudges(session)) == 1
    assert summary == "Found 1 markdown file."


def test_baton_gate_config_default_on():
    a = AgentConfig(name="x", role="y")
    assert a.baton_gate.enabled is True

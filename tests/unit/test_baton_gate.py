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
    # Live receipts (Gemma-4-E2B REPL, 2026-07-16): six turns shipped announces the now/next
    # anchors missed — bare future-intent and present-progressive shapes, and polite-wait.
    "I will search for Korean market holidays or trading closures for tonight.",
    "I am searching for Korean market holiday schedules that affect KOSPI trading for tonight.",
    "I am checking for any scheduled Korean holidays. Please wait a moment for the search results.",
    "I am about to search for the official Korea Exchange (KRX) trading schedule.",
    "I apologize for the confusion. I am executing the search now.",
    "I am pulling the data to find the official trading hours for the KRX.",
    "One moment please, gathering the schedule data.",
    # Live receipt (qwen3.8-27b, 2026-09-04): a long multi-tool turn (actions_taken > 0, so the
    # act-guard could not arm) shipped this bare "let me <verb>" as its final answer. The
    # now/next anchors missed it exactly as they missed the 2026-07-16 "I will / I am" receipts
    # — hence the same anchor-free widen, now applied to the "let me" branch.
    "Let me confirm the lint script's interface so I can run it on my draft, "
    "and check the content.json format expectations.",
    "Let me check the logs.",   # was a NEGATIVE ("bare 'let me X' is not a target shape") until
                                # the 2026-09-04 receipt above superseded that call: the bare
                                # form IS the live shape. Precision now rests where it does for
                                # the "I will" branch — final sentence + the verb whitelist.
    # The same receipt's second lesson: `content.json`'s dot is not a sentence break. An in-token
    # dot must not forge a fake closing sentence out of the tail fragment.
    "Let me read config.json to see the expected keys.",
    "I will verify the pinned version is 1.2.3 in the lockfile.",
    # Second live receipt (owner dogfood, 0.13.0, 2026-09-04): an ANCHORED announce mid-task —
    # already a target shape, and it detects. It is pinned here because the session still ended
    # on it: detection was never the problem there, the per-turn BOUND was (the default single
    # nudge had been spent earlier in the turn, so the re-announce was accepted verbatim — see
    # test_baton_gate_relives_the_dogfood_re_announce).
    "I have the z230 customer details. Now let me pull the 3PL detail from x75 (the third "
    "profile) so all three profiles carry measured outcomes and deployment timelines.",
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
    # Precision guards for the bare "let me" widen (2026-09-04). The whitelist is the guard: a
    # HANDBACK ("let me know…") and non-action verbs are not announced work.
    "Let me know how you would like to proceed.",      # handback, not an announce
    "Let me know once the build finishes.",            # handback
    "Let me think about this differently: the answer is 42.",   # 'think' deliberately unmatched
    "Let me summarize: the cause is a stale lock.",              # 'summarize' unmatched
    "Let me be clear: the config was already correct.",          # 'be' unmatched
    "Let me check the notebooks. They contain the training outputs showing 92% accuracy.",
    # ^ bare form mid-reply with real content after — the final-sentence-only contract, which the
    #   anchored form already proves above, must hold for the bare form too.
    # Precision guards for the widened families — idioms and user-directed instructions
    # that share surface forms with announces MUST stay accepted:
    "I am running out of options.",                   # idiom, not an announced run
    "I am running low on context.",                   # idiom
    "I am finding this approach problematic.",        # 'finding' deliberately unmatched
    "I am working on the assumption that the cache is cold.",  # 'working on' deliberately unmatched
    "Run the build, then please wait for it to finish.",        # instruction to the USER
    "I am done with the analysis: the cause is the stale lock.",
    "I am confident the answer is 42.",
]


@pytest.mark.parametrize("text", _POSITIVES)
def test_detect_dropped_baton_positive(text):
    assert detect_dropped_baton(text) is True


@pytest.mark.parametrize("text", _NEGATIVES)
def test_detect_dropped_baton_negative(text):
    assert detect_dropped_baton(text) is False


# --- gate behaviour at the acceptance seam -------------------------------------------------
from localharness.agent.context import ContextManager
from localharness.agent.loop import (
    _BATON_ESCALATION_PREFIX, _BATON_NUDGE_MESSAGE, AgentLoop, Session,
)
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
    """Every baton nudge pushed this turn — the generic first one and the escalated ones, which
    quote the model's clause and so are matched by prefix (mirrors _is_harness_nudge)."""
    return [m.get("content") or "" for m in session.messages if m.get("role") == "user"
            and ((m.get("content") or "") == _BATON_NUDGE_MESSAGE
                 or (m.get("content") or "").startswith(_BATON_ESCALATION_PREFIX))]


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
    # FIX 4: default max_nudges=1 preserves the original #84 behavior byte-for-byte (bounded
    # once) — raising it is an opt-in tuning knob, not a new out-of-the-box default.
    assert a.baton_gate.max_nudges == 1


def test_baton_gate_max_nudges_bounds():
    import pydantic
    from localharness.config.models import BatonGateConfig

    BatonGateConfig(max_nudges=1)
    BatonGateConfig(max_nudges=3)
    for bad in (0, 4):
        with pytest.raises(pydantic.ValidationError):
            BatonGateConfig(max_nudges=bad)


@pytest.mark.asyncio
async def test_baton_gate_max_nudges_2_announces_nudge_nudge_then_accepts(bus):
    """FIX 4(b): with max_nudges=2 configured, a THIRD consecutive announce (after two nudges
    are spent) is finally accepted — announce -> nudge -> announce -> nudge -> announce -> accept."""
    llm = _ScriptedNoToolLLM([
        "Now let me read the notebooks.",
        "Now let me also read the configs.",
        "Now let me check one more thing.",
    ])
    loop = _make_loop(llm, bus, baton_gate={"max_nudges": 2})
    session = Session(agent_id="baton-agent", session_id="s-max2", messages=[])
    summary = await loop._execute_loop(session, "analyze", None)
    assert session.baton_nudges_used == 2
    nudges = _baton_nudges(session)
    assert len(nudges) == 2                      # exactly two nudges, then accept (no loop)
    assert session.iteration == 3
    assert summary == "Now let me check one more thing."  # 3rd announce accepted verbatim
    # 2026-09-04: the CONTENT escalates even though the count is bounded exactly as before —
    # nudge 1 is the generic #84 message, nudge 2 quotes the model's own second announce back.
    assert nudges[0] == _BATON_NUDGE_MESSAGE
    assert nudges[1].startswith(_BATON_ESCALATION_PREFIX)
    assert "Now let me also read the configs" in nudges[1]
    assert nudges[1] != _BATON_NUDGE_MESSAGE


@pytest.mark.asyncio
async def test_baton_gate_fires_on_the_live_specimen_end_to_end(bus):
    """2026-09-04 receipt, driven through the real loop (not just the detector): the bare
    "Let me confirm …content.json…" final now gets its nudge instead of shipping as the answer."""
    specimen = ("Let me confirm the lint script's interface so I can run it on my draft, "
                "and check the content.json format expectations.")
    llm = _ScriptedNoToolLLM([specimen, "The lint script takes a path and reads content.json."])
    loop = _make_loop(llm, bus)
    session = Session(agent_id="baton-agent", session_id="s-specimen", messages=[])
    summary = await loop._execute_loop(session, "lint my draft", None)
    assert session.baton_nudge_used is True
    assert _baton_nudges(session) == [_BATON_NUDGE_MESSAGE]   # default max_nudges=1: generic only
    assert summary == "The lint script takes a path and reads content.json."


_SPECIMEN_1 = ("Let me confirm the lint script's interface so I can run it on my draft, "
               "and check the content.json format expectations.")
_SPECIMEN_2 = ("I have the z230 customer details. Now let me pull the 3PL detail from x75 "
               "(the third profile) so all three profiles carry measured outcomes and "
               "deployment timelines.")


@pytest.mark.asyncio
async def test_baton_gate_relives_the_dogfood_re_announce_default_bound(bus):
    """The 0.13.0 dogfood shape at the DEFAULT bound (max_nudges=1): announce -> one nudge ->
    the model announces AGAIN -> accepted verbatim. This is the designed bound, and it is what
    the owner saw end the session on 'Now let me pull the 3PL detail…'. Pinned so the escalation
    below is read as what it is — a fix for the SECOND nudge, not for this default."""
    llm = _ScriptedNoToolLLM([_SPECIMEN_1, _SPECIMEN_2])
    loop = _make_loop(llm, bus)
    session = Session(agent_id="baton-agent", session_id="s-dogfood-1", messages=[])
    summary = await loop._execute_loop(session, "profile three customers", None)
    assert _baton_nudges(session) == [_BATON_NUDGE_MESSAGE]   # one generic nudge, then accept
    assert summary == _SPECIMEN_2


@pytest.mark.asyncio
async def test_baton_gate_relives_the_dogfood_re_announce(bus):
    """Same two live specimens, max_nudges=2: nudge 1 is generic, the model re-announces, and
    nudge 2 quotes THAT promise back verbatim instead of re-spending the identical generic ask."""
    llm = _ScriptedNoToolLLM([_SPECIMEN_1, _SPECIMEN_2, "All three profiles are complete."])
    loop = _make_loop(llm, bus, baton_gate={"max_nudges": 2})
    session = Session(agent_id="baton-agent", session_id="s-dogfood-2", messages=[])
    summary = await loop._execute_loop(session, "profile three customers", None)
    nudges = _baton_nudges(session)
    assert nudges[0] == _BATON_NUDGE_MESSAGE
    assert nudges[1].startswith(_BATON_ESCALATION_PREFIX)
    # The echo is the model's own re-announced clause — the closing sentence, not the whole reply.
    assert "Now let me pull the 3PL detail from x75 (the third profile)" in nudges[1]
    assert "I have the z230 customer details" not in nudges[1]
    assert len(nudges) == 2 and session.iteration == 3
    assert summary == "All three profiles are complete."


def test_baton_nudge_content_ladder():
    """The escalation is deterministic text built from the model's own matched clause — no
    summarizing, no model call — and the FIRST nudge is the shipped #84 message byte-for-byte
    (so the default max_nudges=1 path is unchanged by this feature)."""
    from localharness.agent.loop import (
        _REPETITION_SAMPLE_CHARS, _baton_closing_announce, _baton_nudge_message,
    )
    clause = _baton_closing_announce("I read the files. Now let me read the notebooks.")
    assert clause == "Now let me read the notebooks"
    assert _baton_nudge_message(clause, 1) == _BATON_NUDGE_MESSAGE
    escalated = _baton_nudge_message(clause, 2)
    assert escalated.startswith(_BATON_ESCALATION_PREFIX) and clause in escalated
    assert _baton_nudge_message(clause, 3) == escalated       # stays escalated, never re-generic
    # A runaway closing sentence cannot bloat the nudge: the echo is sliced to the module's
    # existing evidence-preview width rather than a new magic number.
    long_clause = "Now let me read " + ("x" * 5000)
    assert len(_baton_nudge_message(long_clause, 2)) < len(_BATON_ESCALATION_PREFIX) + \
        _REPETITION_SAMPLE_CHARS + 200


def test_escalated_nudge_counts_as_a_harness_nudge():
    """#91b: the escalated nudge is a harness nudge like the generic one, so a bare-CONFIRMED
    reply to it is stripped from the persisted history together with its inducing nudge."""
    from localharness.agent.loop import _is_harness_nudge, _strip_sentinel_exchanges
    escalated = _BATON_ESCALATION_PREFIX + '"Now let me read the notebooks" — and so on.'
    assert _is_harness_nudge({"role": "user", "content": escalated}) is True
    kept = _strip_sentinel_exchanges([
        {"role": "user", "content": "analyze"},
        {"role": "assistant", "content": "Now let me read the notebooks."},
        {"role": "user", "content": escalated},
        {"role": "assistant", "content": "CONFIRMED"},
    ])
    assert [m["content"] for m in kept] == ["analyze", "Now let me read the notebooks."]

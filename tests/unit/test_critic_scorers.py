"""Critics as scorers: the score line, the stall rule, the round index, provenance and the evicted re-read."""
from __future__ import annotations

from localharness.agent.loop import StuckDetector, StuckState
from localharness.agent.subagent import provenance_note
from localharness.orchestrator.task_flow.filer import Docs
from localharness.orchestrator.task_flow.panel import (Finding, Panel, PanelRun, load_critic, parse, revision_request,
                                                       FENCE_CLOSE)
from localharness.orchestrator.task_flow.record import STAGES, limited
from localharness.config.models import TaskFlowConfig


def test_score_lines_parse_into_items_with_a_fix_and_block_under_eight():
    raw = ("A1 4/10 the three chapters — ch3 absent from the draft — to 9: present chapter 3\n"
           "B1 8/10 txt output — promised, not yet made — to 9: write the file\n"
           "C1 ?/10 author identity — no web tool — to 9: none\n"
           "SCORE 4/10")
    parsed = parse("jtbd", raw, None)
    assert parsed.readable and not parsed.clean and parsed.score == 4
    first = parsed.findings[0]
    assert (first.tag, first.score, first.fix) == ("BLOCKER", 4, "present chapter 3")
    assert first.rule.startswith("A1 4/10") and first.evidence == "ch3 absent from the draft"
    tags = {f.rule.split()[0]: (f.tag, f.score, f.fix) for f in parsed.findings}
    assert tags["B1"] == ("NOTE", 8, "write the file") and tags["C1"] == ("NOTE", None, "")


def test_all_items_at_eight_or_more_is_a_clean_pass():
    parsed = parse("factual", "A1 9/10 claim — matches pg-1 — to 9: none\nSCORE 9/10", None)
    assert parsed.clean and parsed.score == 9 and all(f.tag == "NOTE" for f in parsed.findings)


def test_old_shaped_lines_still_parse():
    parsed = parse("factual", "[BLOCKER] cited line must say it — a.py:3 says otherwise", None)
    assert parsed.findings[0].tag == "BLOCKER" and parsed.findings[0].score is None


def test_every_critic_carries_the_shared_contract_and_the_reply_floor_is_jtbd():
    assert "SCORE <lowest item score>/10" in load_critic("jtbd").role
    assert load_critic("factual", ("web_search", "browser_open")).tools.add[-2:] == ["web_search", "browser_open"]
    assert STAGES["reply"].floor == ("jtbd",)


def test_revision_sends_the_card_and_only_the_items_under_eight():
    low = Finding("BLOCKER", "A1 4/10 ask", "seen", False, "jtbd", 4, "do it")
    ok = Finding("NOTE", "B1 9/10 other", "fine", False, "jtbd", 9, "")
    text = revision_request((low, ok), False, "round 1 · jtbd 4/10")
    assert "round 1 · jtbd 4/10" in text and "to 9: do it" in text and "B1 9/10" not in text
    assert FENCE_CLOSE in text and "keep everything else as it is" in text


def _run(score: int) -> PanelRun:
    f = Finding("BLOCKER", "A1 x", "e", False, "jtbd", score, "")
    return PanelRun(1, (f,), "blocked", True)


def test_a_round_whose_lowest_score_did_not_rise_stalls_and_rounds_are_indexed(tmp_path):
    docs = Docs(tmp_path, True, "root")
    panel = Panel(None, limited(TaskFlowConfig())[0], docs=docs)
    panel.round = 1
    panel._rounds("one\ntwo\n", _run(3), "blocked")
    assert not panel.stalled and "jtbd 3/10" in panel.card
    panel.round = 2
    panel._rounds("one\nthree\n", _run(5), "blocked")
    assert not panel.stalled and "+1 -1 lines" in panel.card
    panel.round = 3
    panel._rounds("one\nthree\n", _run(5), "blocked")
    assert panel.stalled
    index = docs.read_session("reply-review.md")
    assert index.count("## round") == 3 and (tmp_path / "agents/root/sessions/reply-draft.md").read_text() == "one\nthree\n"


def test_provenance_flags_text_that_no_source_holds_and_passes_copied_text():
    source = " ".join(f"word{i}" for i in range(200))
    assert provenance_note(source[:900], [source]) == ""
    invented = " ".join(f"other{i}" for i in range(130))
    note = provenance_note(invented, [source])
    assert note.startswith("2 of 2 stretches") and "other0 other1" in note


def test_a_reread_of_an_evicted_result_counts_toward_the_ceiling_not_the_window():
    d = StuckDetector()
    for _ in range(4):
        d.record("web_fetch", {"url": "u"}, (True, "x"), reread=True)
    assert d.classify()[0] == StuckState.CLEAR and d.call_count(d.compute_signature("web_fetch", {"url": "u"})) == 4
    d.record("web_fetch", {"url": "u"}, (True, "x"))
    d.record("web_fetch", {"url": "u"}, (True, "x"))
    assert d.classify()[0] == StuckState.RECOVERING


async def test_the_loop_learns_which_calls_sit_behind_an_eviction_stub():
    from types import SimpleNamespace
    from localharness.agent.context import _WEB_STUB_PREFIX
    from localharness.agent.loop import AgentLoop, compute_signature
    sig = compute_signature("web_fetch", {"url": "u"})
    stubbed = {"role": "tool", "tool_call_id": "1", "content": f"{_WEB_STUB_PREFIX} — web_fetch u; 900 chars dropped]",
               "_lh": {"outcome": {"call": sig, "ok": True, "denied": False, "staged": False}}}
    kept = {"role": "tool", "tool_call_id": "2", "content": "page text",
            "_lh": {"outcome": {"call": "other", "ok": True, "denied": False, "staged": False}}}

    async def build_messages(messages, tools, **_):
        return [stubbed, kept], SimpleNamespace()

    loop = object.__new__(AgentLoop)
    loop._ctx = SimpleNamespace(build_messages=build_messages, _content_store=None)
    loop._llm = SimpleNamespace(config=SimpleNamespace(tool_call_mode="native"))
    await loop._pack_request([], None, "")
    assert loop._evicted_calls == frozenset({sig})


def test_a_blocker_line_on_screen_carries_its_fix():
    rows = []
    panel = Panel(None, limited(TaskFlowConfig())[0])
    panel.emit = lambda *row: rows.append(row)
    panel._say("jtbd", parse("jtbd", "A1 4/10 ask — seen — to 9: do it\nSCORE 4/10", None))
    assert rows[0][1] == "blocker A1 4/10 ask" and rows[0][3] == "seen — to 9: do it"

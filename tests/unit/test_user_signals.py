"""COLL-02 zero-NLU user-signal classifier tests (Phase 34-04).

The trigger lexicon fires the right (signal_type, trigger_family, matched_text) on the
motivating fireworks specimen — the exact casually-stated preference correction that today
produces ZERO signal (no tool error, no stuck loop, invisible to the pain-only gate).
That asymmetry is the phase's reason to exist.

The classify tests run against the REAL default lexicon from config, so a lexicon
regression breaks these tests by design.
"""
import pytest

from localharness.config.models import PredictiveGateConfig, TriggerLexiconConfig
from localharness.core.bus import EventBus
from localharness.core.events import UserMessage
from localharness.memory.sqlite import MemoryStore
from localharness.memory.user_signals import UserSignalDetector, classify_user_signal

LEXICON = TriggerLexiconConfig().model_dump()


def test_fireworks_specimen_fires():
    # The motivating trace specimen (event 52f90afe-…, home-location scrubbed per the 33.1
    # PII precedent): NO comma after "nah"; the firing trigger is the TOKEN "nah" (negation
    # family), which today yields no signal at all.
    text = "nah id rather watch the fireworks from the park with friends tomorrow"
    assert classify_user_signal(text, LEXICON) == ("correction", "negation", "nah")


def test_token_boundary_no_not_inside_know_or_now():
    # 'no' must not fire inside 'know'/'now' — single-word triggers match whole tokens,
    # never substrings.
    assert classify_user_signal("i know the answer", LEXICON) is None
    assert classify_user_signal("now do it again", LEXICON) is None
    assert classify_user_signal("no that's wrong", LEXICON) == ("correction", "negation", "no")


def test_multiword_substring_matches():
    # multi-word triggers DO match as substrings ("not what i" ⊂ negation family).
    sig = classify_user_signal("that's not what i asked for", LEXICON)
    assert sig is not None
    assert sig[0] == "correction"


def test_family_frustration():
    assert classify_user_signal("ugh this is frustrating", LEXICON) == (
        "correction",
        "frustration",
        "ugh",
    )


def test_family_confirmation():
    # confirmation family is checked before interruption; the negation list has no "exactly".
    assert classify_user_signal("exactly right", LEXICON) == (
        "confirmation",
        "confirmation",
        "exactly",
    )


def test_family_interruption_is_its_own_class():
    # interruption is a WEAKER, separate label class — never folded into corrections.
    sig = classify_user_signal("hold on", LEXICON)
    assert sig is not None
    assert sig[0] == "interruption"
    assert sig[0] != "correction"


def test_family_correction_phrase():
    sig = classify_user_signal("i meant the other one", LEXICON)
    assert sig is not None
    assert sig[:2] == ("correction", "correction_phrase")


def test_precedence_correction_beats_interruption():
    # "no wait stop": negation (correction-class) wins over the interruption words —
    # an interruption word never absorbs a correction (owner ruling).
    assert classify_user_signal("no wait stop", LEXICON) == ("correction", "negation", "no")


def test_case_insensitive():
    assert classify_user_signal("NAH ID RATHER WATCH THE FIREWORKS", LEXICON) == (
        "correction",
        "negation",
        "nah",
    )


def test_no_signal_on_neutral_request():
    # neutral request, no trigger word -> None (no location PII in fixtures, owner default).
    assert classify_user_signal("please check the weather forecast", LEXICON) is None


# ---------------------------------------------------------------------------
# Re-ask detection (stdlib difflib, zero NLU) — proven end-to-end through the detector,
# a real EventBus, and a real MemoryStore. Re-asks carry no lexical trigger; they are
# caught by near-identity to an earlier message in the SAME sitting.
# ---------------------------------------------------------------------------

_AGENT = "sig-agent"


async def _detector(tmp_path):
    store = MemoryStore(agent_id=_AGENT, division_id="", org_id="", base_dir=str(tmp_path))
    await store.open()
    bus = EventBus()
    det = UserSignalDetector(store, bus, _AGENT, PredictiveGateConfig())
    await det.open()
    return store, bus


def _um(content, session_id):
    return UserMessage(
        agent_id=_AGENT, session_id=session_id, content=content, channel="terminal"
    )


async def _signal_rows(store):
    async with store._db.execute(
        "SELECT signal_type, trigger_family FROM user_signals ORDER BY id"
    ) as cur:
        return [tuple(r) for r in await cur.fetchall()]


@pytest.mark.asyncio
async def test_reask_detected_across_two_messages(tmp_path):
    store, bus = await _detector(tmp_path)
    try:
        await bus.publish(_um("what's the vllm flag for prefix caching", "s1"))
        await bus.publish(_um("whats the vllm flag for prefix caching again", "s1"))
        # first message: no trigger, empty window -> no row; second: difflib re-ask
        assert await _signal_rows(store) == [("correction", "reask")]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_reask_scoped_to_session(tmp_path):
    store, bus = await _detector(tmp_path)
    try:
        await bus.publish(_um("what's the vllm flag for prefix caching", "s1"))
        await bus.publish(_um("whats the vllm flag for prefix caching again", "s2"))
        assert await _signal_rows(store) == []  # different sittings -> no re-ask
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_reask_never_self_matches(tmp_path):
    store, bus = await _detector(tmp_path)
    try:
        await bus.publish(_um("what's the vllm flag for prefix caching", "s1"))
        # the window is appended AFTER the check, so a lone message never matches itself
        assert await _signal_rows(store) == []
    finally:
        await store.close()

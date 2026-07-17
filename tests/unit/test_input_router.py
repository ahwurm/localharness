"""Type-anytime input router — Tier-1 lexical rules (both directions), Tier-2 LLM
fallback (bounded, uncertain→queue), and the `!` force-nudge escape hatch.

Pure functions only; the Tier-2 LLM seam is an injected async callable, so nothing
here touches a live model."""
from __future__ import annotations

import asyncio

import pytest

from localharness.channels.input_router import (
    Decision,
    Route,
    classify_tier1,
    classify_tier2,
    route,
    strip_force,
)


class TestStripForce:
    def test_bang_prefix_forces_and_strips(self):
        assert strip_force("!keep going on this") == ("keep going on this", True)

    def test_bang_with_leading_space_stripped(self):
        assert strip_force("!   stop now") == ("stop now", True)

    def test_no_bang_is_untouched(self):
        assert strip_force("stop that") == ("stop that", False)

    def test_bang_only(self):
        assert strip_force("!") == ("", True)


class TestTier1Nudge:
    @pytest.mark.parametrize(
        "msg",
        [
            "stop, that's the wrong file",
            "no, use the other config",
            "wait",
            "don't touch the prod db",
            "wrong branch",
            "careful with that rm",
            "actually target main, not dev",
            "instead read the yaml",
            "please stop",        # near-initial filler skipped
            "no wait",            # two nudge tokens
            "ok, stop the deploy",
        ],
    )
    def test_message_initial_nudge_tokens(self, msg):
        d = classify_tier1(msg)
        assert d is not None and d.route is Route.NUDGE and d.tier == "tier1"

    @pytest.mark.parametrize(
        "msg",
        [
            "use the staging url not the prod one",
            "read config.yaml not config.json",
            "target main instead of dev",
        ],
    )
    def test_correction_shape_use_x_not_y(self, msg):
        d = classify_tier1(msg)
        assert d is not None and d.route is Route.NUDGE


class TestTier1Queue:
    @pytest.mark.parametrize(
        "msg",
        [
            "when you're done, run the tests",
            "after this, open a PR",
            "also update the changelog",
            "then deploy to staging",
            "next, check the logs",
            "later summarize what changed",
            "once you are finished, ping me",
            "and also add a regression test",
            "one more thing: bump the version",
        ],
    )
    def test_future_framed_queue(self, msg):
        d = classify_tier1(msg)
        assert d is not None and d.route is Route.QUEUE and d.tier == "tier1"

    @pytest.mark.parametrize(
        "msg",
        [
            "what does the bench harness measure?",
            "how many agents are configured right now?",
        ],
    )
    def test_new_question_queue(self, msg):
        d = classify_tier1(msg)
        assert d is not None and d.route is Route.QUEUE


class TestTier1Abstain:
    @pytest.mark.parametrize(
        "msg",
        [
            "index the tests directory",
            "the parser looks off to me",
            "write a haiku about the moon",
            "",
            "   ",
        ],
    )
    def test_ambiguous_abstains(self, msg):
        assert classify_tier1(msg) is None


class TestTier2:
    async def test_nudge_verdict(self):
        async def fake(_msgs):
            return "NUDGE"

        d = await classify_tier2("x", "ctx", fake)
        assert d.route is Route.NUDGE and d.tier == "tier2"

    async def test_queue_verdict(self):
        async def fake(_msgs):
            return "queue"

        d = await classify_tier2("x", "ctx", fake)
        assert d.route is Route.QUEUE

    async def test_tolerates_wrapped_word(self):
        async def fake(_msgs):
            return "The answer is: NUDGE."

        d = await classify_tier2("x", "ctx", fake)
        assert d.route is Route.NUDGE

    async def test_timeout_defaults_queue(self):
        async def slow(_msgs):
            await asyncio.sleep(1.0)
            return "NUDGE"

        # #92: the classify budget is now permit_wait + timeout (permit-wait no longer eats the
        # generation clock); a call slower than the TOTAL budget still defaults to QUEUE.
        d = await classify_tier2("x", "ctx", slow, timeout=0.03, permit_wait=0.02)
        assert d.route is Route.QUEUE and "default-queue" in d.reason

    async def test_error_defaults_queue(self):
        async def boom(_msgs):
            raise RuntimeError("provider down")

        d = await classify_tier2("x", "ctx", boom)
        assert d.route is Route.QUEUE and "default-queue" in d.reason

    async def test_invalid_output_defaults_queue(self):
        async def garbage(_msgs):
            return "hmm not sure really"

        d = await classify_tier2("x", "ctx", garbage)
        assert d.route is Route.QUEUE

    async def test_both_words_is_invalid_defaults_queue(self):
        async def both(_msgs):
            return "nudge or queue?"

        d = await classify_tier2("x", "ctx", both)
        assert d.route is Route.QUEUE


class TestRouteOrchestration:
    async def test_force_bang_wins_without_calling_llm(self):
        called = False

        async def fake(_msgs):
            nonlocal called
            called = True
            return "QUEUE"

        d = await route("later do X", forced=True, context="", complete_fn=fake, tier2_enabled=True)
        assert d.route is Route.NUDGE and d.tier == "force" and not called

    async def test_tier1_decides_without_calling_llm(self):
        async def fake(_msgs):
            raise AssertionError("tier-2 must not run when tier-1 decides")

        d = await route("stop that", forced=False, context="", complete_fn=fake, tier2_enabled=True)
        assert d.route is Route.NUDGE and d.tier == "tier1"

    async def test_abstain_falls_to_tier2_when_enabled(self):
        async def fake(_msgs):
            return "NUDGE"

        d = await route(
            "index the tests dir", forced=False, context="c", complete_fn=fake, tier2_enabled=True
        )
        assert d.tier == "tier2" and d.route is Route.NUDGE

    async def test_abstain_defaults_queue_when_tier2_disabled(self):
        async def fake(_msgs):
            raise AssertionError("tier-2 disabled — must not call")

        d = await route(
            "index the tests dir", forced=False, context="c", complete_fn=fake, tier2_enabled=False
        )
        assert d.route is Route.QUEUE and d.tier == "tier1"

    async def test_abstain_defaults_queue_when_no_client(self):
        d = await route(
            "index the tests dir", forced=False, context="c", complete_fn=None, tier2_enabled=True
        )
        assert d.route is Route.QUEUE

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

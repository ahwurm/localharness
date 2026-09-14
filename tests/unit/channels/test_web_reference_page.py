"""The reference page's checklist (WEBCH-18) and its two non-negotiable client rules.

The page is deliberately ugly and that is the specification. What it is NOT allowed to be is
incomplete or unsafe, so this file walks §4.2's whole event vocabulary and asserts the page
handles every frame, then checks the two rules whose violation is silent:

* every model- or tool-derived string goes in through `textContent`, never `innerHTML`;
* the final answer is rendered from `TaskComplete` and the tool-less `llm_response` is not.

A checklist test rather than a browser test, deliberately: what is being guarded is that the
worked example stays a WORKED example as the wire grows, and the failure it catches is somebody
adding a frame and forgetting the page. Rendering fidelity is the owner's half.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from localharness.channels.web.protocol import FRAME_TYPES
from localharness.channels.web.server import PACKAGED_UI_DIR

PAGE = PACKAGED_UI_DIR / "index.html"


@pytest.fixture(scope="module")
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_the_page_ships_beside_the_server(page):
    assert PAGE.is_file()
    assert Path(PACKAGED_UI_DIR).name == "ui"


def test_every_sse_only_frame_is_handled(page):
    """A frame added without the page learning it is a frame that renders as nothing."""
    missing = [f.__name__ for f in FRAME_TYPES if f'case "{f.__name__}"' not in page]
    assert not missing, f"the reference page ignores {missing}"


@pytest.mark.parametrize("event", [
    "UserMessage", "TurnStarted", "TurnCompleted", "TurnFailed", "Action", "Observation",
    "TaskComplete", "Heartbeat", "ParseFailed", "CompactionTriggered", "Escalation",
    "InputRouted", "PermissionStaged", "PermissionResolved", "PermissionAsked",
    "ConsolidationStarted", "ConsolidationFinished",
])
def test_every_live_bus_event_in_the_vocabulary_is_handled(page, event):
    """§4.2's table, walked. These are the events a client must handle."""
    assert f'case "{event}"' in page, f"{event} is in the wire vocabulary but not in the page"


def test_untrusted_content_never_reaches_innerhtml(page):
    """§5.7 / WEBCH-11. A tool result that can execute script in the page is a tool result that
    can operate the permission UI — it can rewrite the very dialog asking about it."""
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert sink not in page, f"the page assigns through {sink}"
    # ...and the one helper everything goes through does use textContent.
    assert "n.textContent = text" in page


def test_the_answer_is_rendered_from_taskcomplete_and_not_from_the_action(page):
    """§4.2.1's rule, and the single most likely way a client ships with every answer twice."""
    assert re.search(r"if \(d\.has_tool_calls\)", page), "the page must branch on has_tool_calls"
    # The tool-less branch clears the provisional bubble and draws NOTHING.
    tail = page.split("if (d.has_tool_calls)")[1].split("}\n")[0]
    assert "withCode" in tail, "narration (has_tool_calls=true) is the branch that renders"
    assert "clearProvisional" in page


def test_collapse_is_keyed_off_group_and_never_off_risk_level(page):
    """WEBCH-04 / §2.2: `risk_level` is dead and null on all 5,433 Actions ever recorded."""
    assert "risk_level" not in page
    assert "S.collapsible.includes(group)" in page
    assert "/api/tools" in page, "the collapse rule must read the live registry, not a hardcode"


def test_the_cursor_is_persisted_on_every_event(page):
    """WEBCH-42: iOS reclaims the page and the app relaunches into a fresh JS context, so
    `EventSource`'s own resume state is gone. Without a persisted cursor the reopened app skips
    silently to the tail and misses the answer — the one thing WIN-A exists to deliver."""
    assert "localStorage" in page
    assert "remember(data.seq)" in page
    assert 'es = new EventSource("/api/stream" + (cursor != null ? "?from=" + cursor : ""))' in page


def test_copy_reads_the_stored_string_not_the_rendered_dom(page):
    """WEBCH-36: re-extracting text from markup is how leading whitespace dies in a code block —
    the exact defect the market scan found shipped in a major vendor's Android app."""
    assert "copy code" in page and "copy message" in page
    assert "Copy from the STORED string" in page


def test_the_two_always_kinds_go_through_the_server_confirm(page):
    assert "confirm_required" in page and "confirm_token" in page


def test_cancel_has_an_undo_window(page):
    """WEBCH-12: cancel sits next to the composer where a distracted thumb lands, and on this box
    a mis-tap discards real generation."""
    assert 'el("button", "small", "undo")' in page
    assert "clearTimeout(undo)" in page


def test_the_composer_offers_explicit_nudge_and_queue(page):
    """WEBCH-23: no classifier is spent when the human already said which they meant."""
    assert 'send("nudge")' in page and 'send("queue")' in page


def test_offline_and_unreachable_are_distinct_and_neither_is_a_spinner(page):
    """WEBCH-26: they are different problems with different actions."""
    assert "offline — retrying" in page
    assert "model server unreachable" in page
    assert "cold — send to start the session" in page


def test_a_protocol_version_mismatch_is_loud(page):
    """§8: a stale client otherwise degrades silently, rendering `undefined` as a blank cell."""
    assert "d.protocol_version !== 1" in page


def test_the_page_is_one_file_with_no_build_step(page):
    """WIN-B: the owner edits a file, pulls to refresh, sees the change. No bundler, no restart."""
    external = re.findall(r'<script[^>]+src=', page) + re.findall(r'<link[^>]+stylesheet', page)
    assert not external, f"the reference page must have no external assets: {external}"
    assert '<script type="module">' in page


def test_it_is_mobile_usable_even_though_it_is_unstyled(page):
    """WEBCH-25: unstyled is allowed; unusable is not."""
    assert 'name="viewport"' in page and "width=device-width" in page
    assert "min-height:44px" in page, "tap targets must be real"
    assert "font-size:16px" in page, "iOS zooms an input below 16px, which breaks one-thumb use"
    assert "overflow-wrap:anywhere" in page, "nothing may scroll sideways"
    assert "position:fixed" in page, "the composer must be reachable without scrolling"

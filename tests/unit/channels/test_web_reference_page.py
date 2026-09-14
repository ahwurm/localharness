"""The reference page's checklist (WEBCH-18) and its two non-negotiable client rules.

How the page LOOKS is the owner's half and is not specified here. What it is not allowed to be is
incomplete or unsafe, so this file walks §4.2's whole event vocabulary and asserts the page
handles every frame, then checks the two rules whose violation is silent:

* every model- or tool-derived string goes in through `textContent`, never `innerHTML`;
* the final answer is rendered from `TaskComplete` and the tool-less `llm_response` is not.

A checklist test rather than a browser test, deliberately: what is being guarded is that the
worked example stays a WORKED example as the wire grows, and the failure it catches is somebody
adding a frame and forgetting the page. Rendering fidelity is the owner's half.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from localharness.channels.web.protocol import FRAME_TYPES
from localharness.channels.web.server import MANIFEST, PACKAGED_UI_DIR

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


def test_the_page_pays_the_safe_area_inset_at_both_ends(page):
    """Owner bug, from the live install: the top of the app clipped under the Dynamic Island.

    `viewport-fit=cover` plus a translucent status bar hands the page the WHOLE screen, including
    the strips with hardware in them, so the page has to inset itself — the island at the top, the
    home indicator at the bottom. The two metas and the two insets only work as a set: translucent
    without the top inset is the clipped header, and an opaque bar whose colour does not match the
    ground is a grey stripe across a dark app.

    The colours are asserted as a RELATIONSHIP rather than a literal, so re-painting the palette
    cannot leave the status bar or the launch screen behind: whatever `--bg` is, the `theme-color`
    meta and the manifest must both already be it."""
    assert "viewport-fit=cover" in page
    assert 'content="black-translucent"' in page
    assert "env(safe-area-inset-top)" in page, "the header clips under the Dynamic Island"
    assert "env(safe-area-inset-bottom)" in page, "the composer sits under the home indicator"

    ground = re.search(r"--bg:(#[0-9A-Fa-f]{6})", page).group(1)
    assert f'name="theme-color" content="{ground}"' in page, "the status bar must be the ground"
    # The manifest's two colours are what iOS paints BEFORE any of this page runs.
    assert MANIFEST["theme_color"] == ground, "the installed app's chrome is not the page's ground"
    assert MANIFEST["background_color"] == ground, "a cold launch flashes a different colour"


def test_the_page_treats_a_subagents_turn_boundary_as_a_child_event(page):
    """45% of real sessions delegate, and a subagent publishes its OWN TurnStarted/TurnCompleted
    (stamped with the parent's session id). Treating a child's completion as the turn's end
    clears the running state — the stop button, the streaming bubble — while the root is still
    generating. The channel has the same guard; the page needs its own."""
    for case in ('case "TurnStarted"', 'case "TurnCompleted"'):
        block = page.split(case)[1].split("break;")[0]
        assert "!child" in block, f"{case} does not distinguish a subagent's turn from yours"


def test_a_stuck_build_can_be_abandoned_from_the_page(page):
    """WEBCH-43's second half. The server has had `POST /api/bringup/abort` since the surface
    landed; the page did not call it, so `docs/web.md`'s "offers a way out" was an overclaim.
    Cancelling a TURN is a different state machine and does not apply to a build."""
    assert "/api/bringup/abort" in page
    assert "give up on this build" in page


def test_the_page_reaches_every_read_endpoint_it_can_use(page):
    """WEBCH-18: every wire feature demonstrable from this page alone.

    `/api/tool-results/{eviction_id}` is the documented exception and is NOT here: the
    ContentStore is keyed by an eviction id that no `Observation` carries, so a transcript has no
    id to link. That is a limit of the event schema, not of the page, and faking a link would be
    worse than leaving it out.
    """
    for endpoint in ("/api/stream", "/api/health", "/api/protocol", "/api/tools",
                     "/api/grants", "/api/permissions", "/api/auth/enroll", "/api/sessions"):
        assert endpoint in page, f"the page never calls {endpoint}"
    for verb in ("/message", "/cancel", "/command", "/answer", "/mode"):
        assert verb in page, f"the page never exercises {verb}"
    # The parked queue's two verbs are built from one template, so look for the shape.
    assert "/api/pending/${p.id}/${verb}" in page
    assert '["approve", "deny"]' in page


def test_a_repeated_tool_call_id_cannot_orphan_a_row(page):
    """The client's own belt against a duplicate the server should never send.

    The cost of being wrong is silent and permanent: a second row for the same call overwrites
    the map entry, and the Observation then reaches only the second, leaving the first at
    "waiting…" for the rest of the session.
    """
    assert "if (S.calls.has(d.tool_call_id)) return;" in page


# --------------------------------------------------------------------------------------------
# Executed-page tests. Everything above matches STRINGS; these RUN the shipped reducer against a
# DOM the size of what the page actually touches. The bug that motivated them (WEBCH-37) was
# invisible to string matching: the page SAYS "the partial text above stays" in one branch and
# deletes it in another, and only driving the real frame order shows which one wins.
DOM_SHIM = """
const mk = (tag) => ({
  tagName: tag, className: "", textContent: "", style: {}, dataset: {}, children: [], parent: null,
  disabled: false, open: false, onclick: null,
  appendChild(n) { n.parent = this; this.children.push(n); return n; },
  remove() {
    const p = this.parent;
    if (p) p.children.splice(p.children.indexOf(this), 1);
    this.parent = null;
  },
  classList: { add() {}, remove() {}, contains: () => false },
  close() {}, showModal() {},
});
const byId = new Map();
globalThis.document = {
  createElement: mk,
  getElementById: (id) => { if (!byId.has(id)) byId.set(id, mk("div")); return byId.get(id); },
  body: { offsetHeight: 0, scrollHeight: 0 },
  addEventListener() {},
};
const store = new Map();
globalThis.window = {
  localStorage: {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
  },
  innerHeight: 0, scrollY: 0, scrollTo() {}, isSecureContext: true, addEventListener() {},
};
globalThis.location = { hash: "", search: "", pathname: "/" };
globalThis.history = { replaceState() {} };
globalThis.rows = () => {
  const text = (n) => (n.textContent || "") + n.children.map(text).join("");
  return document.getElementById("log").children.map((r) => ({ cls: r.className, text: text(r) }));
};
globalThis.report = () => console.log(JSON.stringify(rows()));
"""

BOOT = "\n(async () => {"


def _drive(page: str, script: str, tmp_path) -> list[dict]:
    """Run the page's reducer under `DOM_SHIM`, feed it `script`, return the rendered rows.

    The boot block is sliced off — it opens an EventSource and talks to an API, neither of which
    exists here. What is under test is everything above it: the reducer, and it runs VERBATIM, so
    a change to the shipped page changes what these assert against.
    """
    import subprocess

    module = re.search(r'<script type="module">(.*?)</script>', page, re.S)
    assert module, "the reference page is one inline module; that shape changed"
    body, sep, _boot = module.group(1).partition(BOOT)
    assert sep, "the page's boot block moved; this harness slices it off by that marker"

    path = tmp_path / "driven.mjs"
    path.write_text(DOM_SHIM + body + script, encoding="utf-8")
    result = subprocess.run(["node", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, f"the page threw:\n{result.stderr}"
    return json.loads(result.stdout)


HELLO = """
onFrame("Hello", {session_id: "s", mode: "repl", turn_in_progress: false,
                  protocol_version: 1, synthetic: false, model_state: "ready"});
onEvent("TurnStarted", {seq: 1, task_summary: "a question"});
onFrame("TokenDelta", {stream_id: "x", text: "half an ", phase: "writing"});
onFrame("TokenDelta", {stream_id: "x", text: "answer"});
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="no JS engine on this box")
def test_cancelling_a_turn_keeps_the_text_the_model_already_wrote(page, tmp_path):
    """WEBCH-37. The row the user is watching is the ONLY copy of a cancelled turn's output.

    A cancelled turn publishes no `TurnCompleted`, so the loop persists no `llm_response` Action
    for it and the replay log has nothing either: deleting the bubble deletes the text for good.
    The page even says so in the very next row it draws — "the partial text above stays" — and
    that line was, until this test, false.
    """
    rendered = _drive(page, HELLO + """
onFrame("StreamClosed", {stream_id: "x", superseded_by_seq: null});
onFrame("TurnCancelled", {session_id: "s"});
report();
""", tmp_path)
    joined = " ".join(r["text"] for r in rendered)
    assert "half an answer" in joined, "the cancelled turn's text was deleted from the page"
    assert "turn cancelled" in joined
    assert "streaming…" not in joined, "a stopped stream still claims to be streaming"


@pytest.mark.skipif(shutil.which("node") is None, reason="no JS engine on this box")
def test_a_turn_that_fails_mid_stream_keeps_its_text_too(page, tmp_path):
    """The same hand, dealt by the other path: `TurnFailed` closes the stream with no successor.

    Here the partial text is diagnostic — it is what the model had produced when the provider
    died, and the error row alone does not say it.
    """
    rendered = _drive(page, HELLO + """
onFrame("StreamClosed", {stream_id: "x", superseded_by_seq: null});
onEvent("TurnFailed", {seq: 2, reason: "provider_error", detail: "connection reset"});
report();
""", tmp_path)
    joined = " ".join(r["text"] for r in rendered)
    assert "half an answer" in joined, "the failed turn's partial text was deleted from the page"
    assert "turn failed — provider_error" in joined
    assert "streaming…" not in joined


@pytest.mark.skipif(shutil.which("node") is None, reason="no JS engine on this box")
def test_a_healthy_turn_still_replaces_the_draft_exactly_once(page, tmp_path):
    """The other half of the fix, and the one with teeth: keeping the bubble MUST NOT become
    keeping it when an authoritative Action supersedes it. That prints the answer twice, which
    §4.2.1 calls the single most likely way a client ships broken.
    """
    rendered = _drive(page, HELLO + """
onFrame("StreamClosed", {stream_id: "x", superseded_by_seq: 7});
onEvent("Action", {seq: 7, action_type: "llm_response", has_tool_calls: false,
                   content: "half an answer, finished"});
onEvent("TaskComplete", {seq: 8, summary: "half an answer, finished", success: true,
                         duration_seconds: 1.0});
report();
""", tmp_path)
    joined = " ".join(r["text"] for r in rendered)
    # The draft is a PREFIX of the final answer, so one occurrence means the draft went and the
    # answer came; two means the same text is on the page twice.
    assert joined.count("half an answer") == 1, "the superseded draft was left on the page"
    assert [r for r in rendered if r["cls"].startswith("row answer")], "the answer never rendered"
    assert "streaming…" not in joined

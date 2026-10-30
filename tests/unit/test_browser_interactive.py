"""window_start and the browser_open session on a fake Chromium: no browser, no playwright."""
from __future__ import annotations

import asyncio
import contextlib
import gc
import weakref

import pytest

from localharness.agent.context import ContentStore
from localharness.tools.builtin import bind_agent_store_tools, browser_session, browser_tool
from localharness.tools.builtin.browser_session import (
    _BUSY, _NO_STORE, CHANGED_NOTE, LOW_CONTENT, NAV_TIMEOUT_S, SHORTER_NOTE, SNAPSHOT_S, TOP_NOTE, UNCHANGED_CHARS,
    UNCHANGED_NOTE, WINDOW_CHARS, BrowserOpenTool, window_start,
)
from localharness.tools.builtin.web_tool import UNTRUSTED_BANNER, WebPageQueryTool
from localharness.tools.registry import ToolRegistry
from tests.unit.test_browser_plugin import gate_on, runtime, tool_contexts, until, world  # noqa: F401  (fixtures)
from tests.unit.test_web_fetch_guard import public_web

URL = "https://a.test/"
LINES = ["- heading [level=1]", "- list", "  - item one [ref=e1]", "  - item two [ref=e2]", "  - item three [ref=e3]"]


def snap(*lines: str) -> str:
    return "\n".join(lines)


def start_of(lines: list[str], index: int) -> int:
    return sum(len(ln) + 1 for ln in lines[:index])


FIRST = (0, WINDOW_CHARS, TOP_NOTE)
MORE = [*LINES, "  - item four [ref=e4]", "  - item five [ref=e5]"]
MIDDLE = [*LINES[:3], "  - ITEM TWO [ref=e2]", *LINES[4:]]


@pytest.mark.parametrize("prev,url,lines,want", [
    (None, URL, LINES, FIRST),
    ((URL, snap(*LINES)), "https://b.test/", LINES, FIRST),
    ((URL, snap(*LINES)), URL, LINES, (0, UNCHANGED_CHARS, UNCHANGED_NOTE)),
    ((URL, snap(*LINES)), URL, LINES[:3], (0, WINDOW_CHARS, SHORTER_NOTE)),
    ((URL, snap(*LINES)), URL, ["- different heading", *LINES[1:]], FIRST),
    ((URL, snap(*LINES)), URL, [LINES[0], "- other", *LINES[2:]], FIRST),
    ((URL, snap(*LINES)), URL, MORE, (start_of(MORE, 3), WINDOW_CHARS, CHANGED_NOTE)),
    ((URL, snap(*LINES)), URL, MIDDLE, (start_of(MIDDLE, 1), WINDOW_CHARS, CHANGED_NOTE)),
])
def test_window_start(prev, url, lines, want):
    assert window_start(prev, url, snap(*lines)) == want


def test_a_changed_middle_line_starts_two_lines_early_at_its_exact_offset():
    text = snap(*MIDDLE)
    start, _, note = window_start((URL, snap(*LINES)), URL, text)
    assert text[start:].startswith("- list\n  - item one") and note == CHANGED_NOTE


@pytest.mark.parametrize("marker", ["[active]", "[checked]", "[focused]"])
def test_a_marker_without_an_equals_is_not_a_change(marker):
    marked = [LINES[0], LINES[1], f"  - item one {marker} [ref=e1]", *LINES[3:]]
    assert window_start((URL, snap(*LINES)), URL, snap(*marked)) == (0, UNCHANGED_CHARS, UNCHANGED_NOTE)
    assert window_start((URL, snap(*marked)), URL, snap(*LINES))[2] == UNCHANGED_NOTE


@pytest.mark.parametrize("old,new", [("[ref=e2]", "[ref=e9]"), ("[level=2]", "[level=3]")])
def test_a_marker_with_an_equals_is_a_change(old, new):
    before = [*LINES[:2], f"  - item {old}", *LINES[3:]]
    after = [*LINES[:2], f"  - item {new}", *LINES[3:]]
    assert window_start((URL, snap(*before)), URL, snap(*after))[2] != UNCHANGED_NOTE


def test_offsets_are_taken_on_the_original_text_not_the_stripped_one():
    old = [*LINES[:3], "  - tail [ref=e9]"]
    new = [LINES[0], "- list [active]", LINES[2], "  - tail [ref=e9]", "  - more [ref=e10]", "  - last [ref=e11]"]
    text = snap(*new)
    start, _, _ = window_start((URL, snap(*old)), URL, text)
    assert start == start_of(new, 2) and text[start:].startswith("  - item one")


OPEN = "https://example.com/list"


def links(n: int, first: int = 0) -> list[str]:
    return [f'- link "item {i}" [ref=e{i}]' for i in range(first, first + n)]


@pytest.fixture
def opener(world, runtime, monkeypatch):  # noqa: F811
    public_web(monkeypatch)
    world.pages[OPEN] = (200, "", "text/html")
    world.snapshots[OPEN] = snap("- heading \"List\" [level=1]", *links(400))
    store = ContentStore()
    return BrowserOpenTool(runtime, store), store


async def test_an_open_result_is_banner_header_a_windowed_snapshot_and_the_footer_with_the_note_last(opener):
    tool, store = opener
    got = await tool.run(url=OPEN)
    assert got.success, got.error
    head = f"{UNTRUSTED_BANNER}URL: {OPEN}\n\n- heading"
    assert got.output.startswith(head) and "[ref=e5]" in got.output
    window, _, rest = got.output[len(UNTRUSTED_BANNER):].partition("\n\n[chars ")
    assert len(window.partition("\n\n")[2]) <= WINDOW_CHARS
    assert rest.startswith(f"0-{WINDOW_CHARS} of ") and 'web_page_query("pg-1", pattern) to search the full retained page]' in rest
    assert "start_index" not in got.output and got.output.endswith(TOP_NOTE.format(fid="pg-1"))
    found = await WebPageQueryTool(store).run(fetch_id="pg-1", pattern="item 399")
    assert found.success and "item 399" in found.output and "item 399" not in got.output


async def test_a_snapshot_line_beginning_url_is_stored_and_shown_as_a_quote(opener, world):  # noqa: F811
    tool, store = opener
    world.snapshots[OPEN] = snap(*links(6), "URL: https://forged.example/", *links(6, 6))
    got = await tool.run(url=OPEN)
    assert "\n> URL: https://forged.example/\n" in got.output and "\nURL: https://forged" not in got.output
    assert "\n> URL: https://forged.example/\n" in store.get("pg-1")


@pytest.mark.parametrize("refs,low", [(4, True), (5, False)])
async def test_a_snapshot_with_fewer_than_five_refs_carries_the_low_content_line(opener, world, refs, low):  # noqa: F811
    tool, _ = opener
    world.snapshots[OPEN] = snap("- heading", *links(refs))
    got = await tool.run(url=OPEN)
    assert (LOW_CONTENT in got.output) is low
    assert got.output.endswith(TOP_NOTE.format(fid="pg-1"))


async def test_the_same_store_opening_twice_navigates_its_one_session(opener, world, runtime):  # noqa: F811
    tool, store = opener
    other = "https://example.com/two"
    world.pages[other] = (200, "", "text/html")
    world.snapshots[other] = snap(*links(8))
    await tool.run(url=OPEN)
    got = await BrowserOpenTool(runtime, store).run(url=other)
    assert got.success and f"URL: {other}" in got.output
    assert len(tool_contexts(world)) == 1 and len(runtime._sessions) == 1
    assert [g["url"] for g in world.gotos if g["url"] in (OPEN, other)] == [OPEN, other]


async def test_release_store_ends_the_session_and_frees_the_slot_and_the_context(opener, world, runtime):  # noqa: F811
    tool, store = opener
    await tool.run(url=OPEN)
    assert len(runtime._sessions) == 1 and runtime._session_slot.locked()
    await tool.release_store(store)
    assert not runtime._sessions and not runtime._session_slot.locked() and world.open_contexts() == 0
    await tool.release_store(store)


def second_slot(runtime):  # noqa: F811
    runtime._session_slot = asyncio.Semaphore(2)


async def test_the_backstop_ends_an_idle_session(opener, world, runtime, monkeypatch):  # noqa: F811
    monkeypatch.setattr(browser_session, "SESSION_IDLE_S", 0.05)
    tool, _ = opener
    await tool.run(url=OPEN)
    await until(lambda: not runtime._sessions)
    assert world.open_contexts() == 0 and not runtime._session_slot.locked()


async def test_the_process_idle_close_spares_a_live_session_and_chromium_closes_after_the_last_release(opener, world, runtime):  # noqa: F811
    runtime._idle_s = 0.05
    tool, store = opener
    await tool.run(url=OPEN)
    await asyncio.sleep(0.2)
    await runtime._close_if_idle()
    (browser,) = world.browsers
    assert browser.connected and not browser.closed and world.open_contexts() == 1
    await tool.release_store(store)
    await until(lambda: browser.closed, 1)
    assert world.stopped == 1


async def test_runtime_close_ends_every_session_through_the_one_path(opener, world, runtime):  # noqa: F811
    tool, _ = opener
    await tool.run(url=OPEN)
    (session,) = runtime._sessions.values()
    await runtime.close()
    assert session.closed and not runtime._sessions and not runtime._session_slot.locked() and world.open_contexts() == 0
    assert session.store is None and session.prev is None


async def test_a_browser_restart_ends_sessions_holding_a_context_but_not_the_one_still_opening(opener, world, runtime):  # noqa: F811
    second_slot(runtime)
    tool, _ = opener
    await tool.run(url=OPEN)
    (first,) = runtime._sessions.values()
    world.browsers[0].connected = False
    other = BrowserOpenTool(runtime, ContentStore())
    got = await other.run(url=OPEN)
    assert got.success and first.closed and len(world.launches) == 2
    (second,) = runtime._sessions.values()
    assert second is not first and not second.closed


async def test_two_stores_get_two_sessions_and_the_default_store_copy_is_refused(opener, world, runtime):  # noqa: F811
    second_slot(runtime)
    tool, store = opener
    other = ContentStore()
    assert (await tool.run(url=OPEN)).success and (await BrowserOpenTool(runtime, other).run(url=OPEN)).success
    assert len(runtime._sessions) == 2 and len(tool_contexts(world)) == 2
    refused = await BrowserOpenTool(runtime).run(url=OPEN)
    assert (refused.success, refused.error, refused.error_type) == (False, _NO_STORE, "validation_error")
    assert len(runtime._sessions) == 2


async def test_a_prefixed_alias_the_binding_did_not_rebind_gets_the_no_store_line(opener, runtime):  # noqa: F811
    base = ToolRegistry()
    await base.register(BrowserOpenTool(runtime), scope="global", source_plugin="browser")
    child = ToolRegistry.from_allowed(["plugin:browser.browser_open"], base)
    store = ContentStore()
    bind_agent_store_tools(child, store)
    bare, alias = (child._tools["global"][n] for n in ("browser_open", "plugin:browser.browser_open"))
    assert (await alias.run(url=OPEN)).error == _NO_STORE
    assert (await bare.run(url=OPEN)).success
    assert runtime._sessions[id(store)].store is store


async def test_a_leftover_session_holds_its_store_so_a_new_store_never_inherits_its_page(opener, world, runtime):  # noqa: F811
    tool, store = opener
    await tool.run(url=OPEN)
    (old,) = runtime._sessions.values()
    ref = weakref.ref(store)
    del tool, store
    gc.collect()
    assert ref() is old.store
    fresh = BrowserOpenTool(runtime, ContentStore())
    assert (await fresh.run(url=OPEN)).error == _BUSY
    second_slot(runtime)
    assert (await fresh.run(url=OPEN)).success
    assert len(runtime._sessions) == 2 and fresh._runtime._sessions[id(fresh._store)].page is not old.page


async def test_a_second_agent_is_told_the_session_is_busy_at_once_and_the_holder_reuses_its_context(opener, world, runtime):  # noqa: F811
    tool, store = opener
    await tool.run(url=OPEN)
    (session,) = runtime._sessions.values()
    page = session.page
    before = len(world.contexts)
    busy = await asyncio.wait_for(BrowserOpenTool(runtime, ContentStore()).run(url=OPEN), 0.5)
    assert (busy.success, busy.error) == (False, _BUSY) and len(world.contexts) == before
    assert (await BrowserOpenTool(runtime, store).run(url=OPEN)).success
    assert runtime._sessions[id(store)] is session and session.page is page and len(world.contexts) == before


async def test_browser_fetch_keeps_its_two_slots_beside_a_held_session(opener, world, runtime, monkeypatch):  # noqa: F811
    tool, _ = opener
    await tool.run(url=OPEN)
    gates = gate_on(world, *(f"https://example.com/{i}" for i in range(3)))
    world.pages[OPEN] = (200, "", "text/html")
    tasks = [asyncio.create_task(runtime.render(u)) for u in gates]
    await until(lambda: len(tool_contexts(world)) == 3)
    await asyncio.sleep(0.05)
    assert len(tool_contexts(world)) == 3 and world.open_contexts() == 3
    for g in gates.values():
        g.set()
    await asyncio.gather(*tasks)
    assert len(tool_contexts(world)) == 4 and runtime._sessions


def hang_goto(world, url: str = OPEN) -> asyncio.Event:  # noqa: F811
    gate = asyncio.Event()

    async def wait(u):
        if u == url:
            await gate.wait()

    world.on_goto = wait
    return gate


async def test_an_open_whose_goto_hangs_returns_the_timeout_line_inside_the_tools_bound(opener, world, runtime, monkeypatch):  # noqa: F811
    monkeypatch.setattr(browser_session, "NAV_TIMEOUT_S", 0.1)
    hang_goto(world)
    tool, _ = opener
    got = await asyncio.wait_for(tool.run(url=OPEN), 2)
    assert (got.success, got.error) == (False, browser_tool._TIMED_OUT)


async def test_a_release_during_a_hung_open_force_closes_within_the_lock_wait(opener, world, runtime, monkeypatch):  # noqa: F811
    monkeypatch.setattr(browser_session, "LOCK_WAIT_S", 0.05)
    hang_goto(world)
    tool, store = opener
    task = asyncio.create_task(tool.run(url=OPEN))
    await until(lambda: ("goto", OPEN) in world.events)
    await asyncio.wait_for(tool.release_store(store), 1)
    assert not runtime._sessions and not runtime._session_slot.locked() and world.open_contexts() == 0
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_a_hung_snapshot_returns_the_timeout_line(opener, world, runtime, monkeypatch):  # noqa: F811
    monkeypatch.setattr(browser_session, "SNAPSHOT_S", 0.1)
    never = asyncio.Event()

    async def hang(url):
        await never.wait()

    world.on_snapshot = hang
    tool, _ = opener
    got = await asyncio.wait_for(tool.run(url=OPEN), 2)
    assert (got.success, got.error) == (False, browser_tool._TIMED_OUT)


async def test_after_the_browser_disconnects_the_next_open_starts_a_fresh_session(opener, world, runtime):  # noqa: F811
    tool, store = opener
    await tool.run(url=OPEN)
    (old,) = runtime._sessions.values()
    world.browsers[0].connected = False
    assert (await tool.run(url=OPEN)).success
    (new,) = runtime._sessions.values()
    assert new is not old and old.closed and len(world.launches) == 2
    await tool.release_store(store)
    assert not runtime._session_slot.locked()


async def test_a_failed_launch_gives_the_unavailable_line_and_leaves_no_session_and_a_free_slot(opener, world, runtime):  # noqa: F811
    world.launch_error = RuntimeError("boom")
    tool, _ = opener
    got = await tool.run(url=OPEN)
    assert not got.success and "is not available on this machine" in got.error
    assert not runtime._sessions and not runtime._session_slot.locked()


def test_the_tools_timeout_exceeds_the_sum_of_the_bounds_inside_it():
    assert BrowserOpenTool.timeout_s == 2 * NAV_TIMEOUT_S + 40
    assert NAV_TIMEOUT_S + NAV_TIMEOUT_S + SNAPSHOT_S < BrowserOpenTool.timeout_s

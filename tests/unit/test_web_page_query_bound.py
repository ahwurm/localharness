"""web_page_query's pattern is bounded (orchestrator ruling R2).

CPython's regex engine holds the GIL, so neither the event loop nor a worker thread can stop a
catastrophic pattern: `(a+)+$` on 26 characters stalled a session for 3.3 s (measured). A pattern
with no regex metacharacters is a plain substring searched in-process; any other pattern runs in a
`python -I` child that is killed after 1 s; a pattern longer than 128 characters is refused. Each
failure is one tool error for the model — no human is asked."""
from __future__ import annotations

import asyncio
import re
import sys
import time

import pytest

from localharness.agent.context import ContentStore
from localharness.tools.builtin import web_tool
from localharness.tools.builtin.web_tool import WebPageQueryTool


def _query_tool(text: str) -> tuple[WebPageQueryTool, str]:
    store = ContentStore()
    return WebPageQueryTool(store), store.put_web(text)


def _spy_children(monkeypatch) -> list:
    """Record every child process the tool starts (args and the process object)."""
    started: list = []
    real = asyncio.create_subprocess_exec

    async def spy(*args, **kwargs):
        proc = await real(*args, **kwargs)
        started.append((args, proc))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    return started


async def test_a_catastrophic_pattern_is_one_tool_error_and_the_loop_keeps_running(monkeypatch):
    started = _spy_children(monkeypatch)
    tool, fetch_id = _query_tool("a" * 26 + "b")
    ticks = 0
    done = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not done.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    beat = asyncio.create_task(ticker())
    t0 = time.monotonic()
    result = await tool.run(fetch_id=fetch_id, pattern="(a+)+$")
    wall = time.monotonic() - t0
    done.set()
    await beat
    assert (result.success, result.error_type) == (False, "validation_error")
    assert result.error == "the pattern took longer than 1 s — use a plainer substring"
    assert wall < 2.5, f"took {wall:.2f} s"
    assert ticks >= 50, f"the event loop ticked only {ticks} times"
    [(args, proc)] = started
    assert args[:3] == (sys.executable, "-S", "-I") or "-I" in args
    assert proc.returncode is not None, "the child was killed and reaped, never orphaned"


async def test_a_plain_substring_never_starts_a_child(monkeypatch):
    async def no_child(*args, **kwargs):
        raise AssertionError("a plain substring must not start a process")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_child)
    tool, fetch_id = _query_tool("Q3 notes. Our REVENUE 2025 grew; S&P 500 inclusion pending.")
    for pattern, found in (("revenue 2025", "REVENUE 2025"), ("S&P 500", "S&P 500 inclusion")):
        result = await tool.run(fetch_id=fetch_id, pattern=pattern)
        assert result.success, result.error
        assert found in result.output and "no match" not in result.output


def _old_spans(pattern: str, text: str, win: int) -> list[tuple[int, int]]:
    """The in-process loop web_page_query ran before the bound (verbatim), as the oracle."""
    rx = re.compile(pattern, re.I)
    spans: list[tuple[int, int]] = []
    for m in rx.finditer(text):
        s, e = max(0, m.start() - win // 2), min(len(text), m.end() + win // 2)
        if spans and s <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
        if len(spans) >= web_tool._QUERY_MAX_MATCHES:
            break
    return spans


@pytest.mark.parametrize("text, window", [
    (("Revenue 2024 was flat. " * 40) + "rev 2025 up. " + ("x" * 3000) + "REVENUE 2023 down.", 200),
    (("rev 2025 " + "y" * 300) * 30, 100),                     # 30 hits: the 20-span cap applies
], ids=["overlaps-merged", "twenty-span-cap"])
async def test_a_regex_returns_the_spans_the_in_process_loop_returned(monkeypatch, text, window):
    started = _spy_children(monkeypatch)
    pattern = r"rev(enue)? 20\d\d"
    tool, fetch_id = _query_tool(text)
    result = await tool.run(fetch_id=fetch_id, pattern=pattern, window=window)
    assert result.success, result.error
    expected = "\n\n…\n\n".join(f"[chars {s}-{e} of {len(text)}]\n{text[s:e]}"
                                for s, e in _old_spans(pattern, text, window))
    assert result.output == web_tool._UNTRUSTED + expected
    assert len(started) == 1, "a regex runs in the child"


async def test_a_pattern_longer_than_128_characters_is_refused():
    tool, fetch_id = _query_tool("x" * 200)
    result = await tool.run(fetch_id=fetch_id, pattern="x" * 129)
    assert (result.success, result.error_type) == (False, "validation_error")
    assert result.error == ("the pattern is longer than 128 characters — search for a shorter piece "
                            "of the text")
    assert (await tool.run(fetch_id=fetch_id, pattern="x" * 128)).success


async def test_a_bad_pattern_is_one_plain_error_not_a_silent_substring():
    tool, fetch_id = _query_tool("a page that mentions (parens) once")
    result = await tool.run(fetch_id=fetch_id, pattern="(")
    assert (result.success, result.error_type) == (False, "validation_error")
    assert result.error.startswith("bad pattern:")
    assert result.error.endswith("— use a plainer substring")
    assert "\n" not in result.error


async def test_the_child_never_imports_from_the_working_directory(monkeypatch, tmp_path):
    """-I: a page or a repo that puts json.py or re.py in the cwd cannot run code in the child."""
    marker = tmp_path / "imported"
    for name in ("json.py", "re.py"):
        (tmp_path / name).write_text(f"open({str(marker)!r}, 'w').close()\n")
    monkeypatch.chdir(tmp_path)
    tool, fetch_id = _query_tool("rev 2025 here")
    result = await tool.run(fetch_id=fetch_id, pattern=r"rev \d+")
    assert result.success, result.error
    assert "rev 2025" in result.output
    assert not marker.exists()

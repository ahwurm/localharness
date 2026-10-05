"""`/memory` as the memory plugin's own slash row (MEMP-03, PAPI-07), proven on the real start.

Three claims, each from a real `_start_async` (only external boundaries stubbed, the 48-01 golden
driver idiom: `OrchestratorREPL.run` replaced by a driver that swaps the channel for a recorder):

1. memory off: `/memory` is absent from every surface read DURING the session — the slash table,
   `/help`, the input completer, the phone's `/api/protocol` `commands[]` — and typing it gets the
   standard `Unknown command` reject (the old "isn't available" text is unreachable from the REPL);
2. memory on: `/memory` is on all four, after `/exit` (M3 — plugin rows follow core rows);
3. memory on, a scripted turn: the model calls `remember`, then `/memory` (overview) lists the
   fact's content and `/memory show 1` its name — the plugin's row reads the very store the
   plugin's tool wrote to.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from prompt_toolkit.document import Document

from tests.conftest import FakeLLMResponse, FakeToolCall
from tests.integration.test_memory_compat_baseline_e2e import (
    _FrozenNow,
    _let_the_stub_tokenizer_run_a_turn,
    _no_session_start_pass,
    _write_root_agent,
)
from tests.integration.test_memory_slash_golden_e2e import _Recorder
from tests.integration.test_workspace_cli_surface_e2e import _offline_provider
from tests.unit.channels.test_mobile_server import BEARER, _stack
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries

UNKNOWN = "Unknown command: /memory — /help lists commands."


async def _session(tmp_path: Path, monkeypatch, body, *, memory_off: bool) -> dict:
    """One real start; `body(repl, out, seen)` runs mid-session in place of the interactive loop."""
    _stub_start_boundaries(tmp_path, monkeypatch)
    _let_the_stub_tokenizer_run_a_turn(monkeypatch)  # after the boundaries: it patches their stub
    _offline_provider(tmp_path)
    if memory_off:
        with (tmp_path / "config.yaml").open("a", encoding="utf-8") as f:
            f.write("memory:\n  enabled: false\n")
    _no_session_start_pass(monkeypatch)
    monkeypatch.setattr("localharness.agent.loop.datetime", _FrozenNow)
    monkeypatch.chdir(tmp_path)
    _write_root_agent(tmp_path, {"consolidation": {"enabled": False}})
    _capture_start_console(monkeypatch)
    out: list[tuple] = []
    seen: dict = {}

    async def drive(self):
        self._channel = _Recorder(self._channel, out)
        await body(self, out, seen)
        seen["ran"] = True

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", drive)
    from localharness.cli.start_cmd import _start_async
    await _start_async(None, False, False, str(tmp_path))
    assert seen.get("ran"), "the driver never ran"
    seen["out"] = out
    return seen


async def _surfaces(repl, out: list, seen: dict, phone_dir: Path) -> None:
    """Read every consumer of the slash table while the session is live."""
    from localharness.channels.terminal import SlashCommandCompleter
    from localharness.cli.slash_commands import all_rows

    seen["rows"] = [r.name for r in all_rows()]
    seen["row_plugin"] = next((r.plugin for r in all_rows() if r.name == "/memory"), None)
    await repl._handle_slash("/help")
    seen["help"] = out[-1][2]
    seen["completer"] = [c.text for c in SlashCommandCompleter().get_completions(Document("/"), None)]
    phone_dir.mkdir()
    _, _, _, client = await _stack(phone_dir)
    seen["phone"] = [c["name"] for c in (await client.get("/api/protocol", headers=BEARER)).json()["commands"]]
    await client.aclose()
    n = len(out)
    seen["handled"] = await repl._handle_slash("/memory")
    seen["memory_reply"] = out[n:]


async def test_memory_off_hides_slash_memory_everywhere(tmp_path, monkeypatch):
    seen = await _session(tmp_path, monkeypatch,
                          lambda r, o, s: _surfaces(r, o, s, tmp_path / "phone"), memory_off=True)
    assert "/help" in seen["rows"] and "/memory" not in seen["rows"], seen["rows"]
    assert "Available commands:" in seen["help"] and "/memory" not in seen["help"], seen["help"]
    assert "/help" in seen["completer"] and "/memory" not in seen["completer"], seen["completer"]
    assert "/help" in seen["phone"] and "/memory" not in seen["phone"], seen["phone"]
    assert seen["handled"] is True
    assert seen["memory_reply"] == [("msg", "system.error", UNKNOWN)], seen["memory_reply"]


async def test_memory_on_lists_slash_memory_after_exit(tmp_path, monkeypatch):
    seen = await _session(tmp_path, monkeypatch,
                          lambda r, o, s: _surfaces(r, o, s, tmp_path / "phone"), memory_off=False)
    assert seen["row_plugin"] == "memory"
    for surface in ("rows", "completer", "phone"):
        names = seen[surface]
        assert "/memory" in names and names.index("/memory") > names.index("/exit"), (surface, names)
    help_lines = seen["help"].splitlines()
    at = [i for i, line in enumerate(help_lines) if line.strip().startswith(("/memory", "/exit"))]
    assert [help_lines[i].split()[0] for i in at] == ["/exit", "/memory"], seen["help"]
    assert seen["handled"] is True
    assert UNKNOWN not in str(seen["memory_reply"]) and seen["memory_reply"], seen["memory_reply"]


async def test_slash_memory_reads_the_store_the_tools_write(tmp_path, monkeypatch):
    # Never load the real embedding model: remember embeds at write time.
    monkeypatch.setattr("localharness.memory.resonance.ResonanceEngine.embed_docs",
                        lambda self, texts: np.ones((len(texts), 8), dtype=np.float32))
    calls: list = []

    async def model(self, messages, tools=None, on_token=None, **_):
        calls.append([t.name for t in tools or ()])
        if len(calls) == 1:
            return FakeLLMResponse(tool_calls=[FakeToolCall(
                id="c1", name="remember", arguments={"name": "fresh-fact", "content": "ZEBRA-7"})]), None
        return FakeLLMResponse(content="Done."), None

    monkeypatch.setattr("localharness.provider.client.LLMClient.stream_complete", model)

    async def body(repl, out, seen):
        n = len(out)
        await repl._handle_slash("/memory")
        seen["before"] = out[n:]
        await repl._agent.run_turn("remember it")
        n = len(out)
        seen["handled"] = await repl._handle_slash("/memory")
        seen["after"] = out[n:]
        n = len(out)
        await repl._handle_slash("/memory show 1")
        seen["show"] = out[n:]

    seen = await _session(tmp_path, monkeypatch, body, memory_off=False)
    assert len(calls) >= 2 and "remember" in calls[0], calls
    assert "ZEBRA-7" not in str(seen["before"]), seen["before"]  # the control: not there before
    assert seen["handled"] is True
    # The overview lists recent memories by id and CONTENT (untagged facts render as text).
    (_kind, _style, text), = seen["after"]
    assert "#1" in text and "ZEBRA-7" in text, seen["after"]
    (_kind, _style, shown), = seen["show"]
    assert "fresh-fact" in shown and "ZEBRA-7" in shown and "remember" in shown, seen["show"]

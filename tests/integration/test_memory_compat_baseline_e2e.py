"""Baselines captured from the pre-plugin memory wiring (Phase 47 Wave 0). These tests MUST pass
unedited after memory becomes a plugin — they are the compatibility invariant. Do not edit an
assertion to make the conversion pass; a red here is a regression.

Every drive is the REAL `_start_async` with only the external boundaries stubbed
(`_stub_start_boundaries`: LLM probe, tokenizer, REPL loop, plugin discovery), the provider pointed
at the loopback discard port, and `LLMClient.stream_complete` replaced by a recorder — no model is
reached. The memory side (store, session row, accumulator, consolidation scheduler, router) runs for
real; that wiring is what is pinned.

Two fixtures were captured from the unconverted tree (5fd12a7) with `LOCALHARNESS_REGEN_GOLDEN=1`:
`tests/fixtures/memory_plugin/system_prompt_memory_on.txt` (the first system prompt, `str(tmp_path)`
-> `<TMP>` the only substitution) and `root_tool_names.txt` (the root agent's tools as the model
saw them, sorted).

Session-start dreaming is a boundary here, not a subject: a seeded store has un-embedded facts, so
the staleness check would launch a background pass that loads the embedding model mid-test. The
golden drive turns consolidation off in its agent yaml (the prompt does not depend on it); the
drives that need the scheduler running stub `ConsolidationScheduler.should_run` to False.
"""
from __future__ import annotations

import contextlib
import functools
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from tests.conftest import FakeLLMResponse
from tests.integration.test_guardrails_from_global_dir_e2e import _let_the_stub_tokenizer_run_a_turn
from tests.integration.test_workspace_cli_surface_e2e import _offline_provider
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "memory_plugin"
GOLDEN_PROMPT = FIXTURES / "system_prompt_memory_on.txt"
GOLDEN_TOOLS = FIXTURES / "root_tool_names.txt"
REGEN = os.environ.get("LOCALHARNESS_REGEN_GOLDEN") == "1"
AGENT = "orchestrator"
GUARDRAILS = "# Org guardrails\nBASELINE-GUARDRAILS-SENTINEL\n"
DIVISION = "# Division\nBASELINE-DIVISION-SENTINEL\n"
MEMORY_TOOLS = {"memory_search", "memory_get", "remember"}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_root_agent(tmp_path: Path, memory: dict | None) -> None:
    data = {"name": AGENT, "role": "General-purpose assistant", "model": "inherit"}
    if memory is not None:
        data["memory"] = memory
    _write(tmp_path / "agents" / f"{AGENT}.yaml", yaml.dump(data, default_flow_style=False))


async def _seed_store(tmp_path: Path) -> None:
    """Two facts in the store start will open (`<state_dir>/agents/<agent>/memory.db`)."""
    from localharness.memory.sqlite import MemoryStore

    store = MemoryStore(agent_id=AGENT, division_id="default", org_id="default",
                        base_dir=str(tmp_path))
    await store.open()
    try:
        await store.store_fact("user-editor", "The user edits in Neovim with a tiling WM.",
                               source="baseline")
        await store.store_fact("project-lang", "The project is Python 3.12 on uv.",
                               source="baseline")
    finally:
        await store.close()


class _FrozenNow(datetime):
    """The prompt's date line is the one clock read in prompt assembly (`agent.loop`'s
    `datetime.now().astimezone()`). Frozen at a fixed UTC instant, `astimezone()` kept on UTC, so
    the golden does not depend on the day or the machine's timezone."""

    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 15, 12, 0, tzinfo=timezone.utc)

    def astimezone(self, tz=None):
        return self


def _no_session_start_pass(monkeypatch) -> None:
    async def never(self):
        return False
    monkeypatch.setattr("localharness.memory.consolidation.ConsolidationScheduler.should_run", never)


async def _start(tmp_path: Path, monkeypatch, *, memory: dict | None, turn: bool = True) -> dict:
    """One real start under `tmp_path` (config_dir explicit). Returns the first request's system
    prompt and tool names (when `turn`) and every console line start printed."""
    _stub_start_boundaries(tmp_path, monkeypatch)
    _offline_provider(tmp_path)
    # The two volatile prompt inputs, removed at their source: the clock and the CWD (the prompt's
    # `Working directory:` line). Both now land on values the golden can hold.
    monkeypatch.setattr("localharness.agent.loop.datetime", _FrozenNow)
    monkeypatch.chdir(tmp_path)
    _let_the_stub_tokenizer_run_a_turn(monkeypatch)
    _write(tmp_path / "orgs" / "default" / "GUARDRAILS.md", GUARDRAILS)
    _write(tmp_path / "divisions" / "default" / "DIVISION.md", DIVISION)
    _write_root_agent(tmp_path, memory)
    printed = _capture_start_console(monkeypatch)
    calls: list[dict] = []

    async def model(self, messages, tools=None, on_token=None, **_):
        calls.append({"system": dict(messages[0]), "tools": sorted(t.name for t in tools or ())})
        return FakeLLMResponse(content="Done."), None

    monkeypatch.setattr("localharness.provider.client.LLMClient.stream_complete", model)
    if turn:
        async def one_turn(self):
            await self._agent.run_turn("say hello")
        monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", one_turn)

    from localharness.cli.start_cmd import _start_async
    await _start_async(None, False, False, str(tmp_path))
    if turn:
        assert calls, "no request reached the model boundary — the turn never ran"
        assert calls[0]["system"]["role"] == "system", calls[0]
    return {"calls": calls, "printed": printed}


def _warnings(printed: list[str]) -> list[str]:
    """The startup warnings group of the summary line, split on start_cmd's `; ` joiner."""
    line = next(p for p in printed if "startup)" in p)
    open_at = line.find("\\[")  # entity() escapes the group's literal bracket
    assert open_at != -1, f"no warnings group on the summary line: {line!r}"
    return line[open_at + 2:].rsplit("]", 1)[0].split("; ")


def _golden(path: Path, text: str) -> str:
    if REGEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- golden prompt + tools


async def test_first_system_prompt_with_memory_on_matches_the_golden(tmp_path, monkeypatch):
    await _seed_store(tmp_path)
    out = await _start(tmp_path, monkeypatch, memory={"consolidation": {"enabled": False}})
    prompt = out["calls"][0]["system"]["content"].replace(str(tmp_path), "<TMP>")

    # Prove memory is ON in what was captured (44-03): an empty-memory golden pins nothing.
    for section in ("## Guardrails\n", "## Division Context\n", "## Agent Memory\n"):
        assert prompt.count(section) == 1, f"{section!r} missing or doubled"
    assert "BASELINE-DIVISION-SENTINEL" in prompt and "user-editor" in prompt and "project-lang" in prompt

    assert prompt == _golden(GOLDEN_PROMPT, prompt), "the first system prompt drifted from the golden"


async def test_root_tool_names_with_memory_on_match_the_golden(tmp_path, monkeypatch):
    await _seed_store(tmp_path)
    out = await _start(tmp_path, monkeypatch, memory={"consolidation": {"enabled": False}})
    names = out["calls"][0]["tools"]
    assert {"bash_exec"} | MEMORY_TOOLS <= set(names), names
    text = "\n".join(names) + "\n"
    assert text == _golden(GOLDEN_TOOLS, text), "the root agent's tool set drifted from the golden"


# --------------------------------------------------------------------------- open / stop order


def _spy(monkeypatch, log: list[str], target: str, label: str) -> None:
    module_path, cls_name, meth = target.rsplit(".", 2)
    import importlib
    cls = getattr(importlib.import_module(module_path), cls_name)
    real = getattr(cls, meth)

    @functools.wraps(real)
    async def rec(self, *a, **k):
        log.append(label)
        return await real(self, *a, **k)
    monkeypatch.setattr(cls, meth, rec)


async def test_memory_opens_and_closes_in_todays_order(tmp_path, monkeypatch):
    _no_session_start_pass(monkeypatch)
    log: list[str] = []
    for target, label in (
        ("localharness.memory.sqlite.MemoryStore.open", "store.open"),
        ("localharness.memory.sqlite.MemoryStore.create_session", "create_session"),
        ("localharness.cli.session_accumulator.SessionAccumulator.open", "accumulator.open"),
        ("localharness.memory.consolidation.ConsolidationScheduler.start", "scheduler.start"),
        ("localharness.memory.consolidation.ConsolidationScheduler.stop", "scheduler.stop"),
        ("localharness.cli.session_accumulator.SessionAccumulator.close", "accumulator.close"),
        ("localharness.memory.sqlite.MemoryStore.end_session", "end_session"),
        ("localharness.memory.router.RecallRouter.close", "router.close"),
        ("localharness.memory.sqlite.MemoryStore.close", "store.close"),
        ("localharness.provider.client.LLMClient.aclose", "llm.aclose"),
    ):
        _spy(monkeypatch, log, target, label)

    await _start(tmp_path, monkeypatch, memory={"consolidation": {"enabled": True}}, turn=False)

    def ordered(seq: list[str]) -> list[str]:
        return [x for x in log if x in seq]

    opens = ["store.open", "create_session", "accumulator.open", "scheduler.start"]
    stops = ["scheduler.stop", "accumulator.close", "end_session", "router.close", "store.close"]
    assert ordered(opens) == opens, log
    assert ordered(stops) == stops, log
    assert log.index("scheduler.start") < log.index("scheduler.stop"), log
    assert log[-1] == "llm.aclose", f"the LLM client must close LAST (#154): {log}"


# --------------------------------------------------------------------------- soft failures


@pytest.mark.parametrize(("prefix", "target"), [
    ("session-start:", "localharness.memory.sqlite.MemoryStore.create_session"),
    ("session-accumulator:", "localharness.cli.session_accumulator.SessionAccumulator.open"),
    ("memory consolidation:", "localharness.memory.consolidation.ConsolidationScheduler.__init__"),
])
async def test_soft_failures_keep_todays_warning_text(tmp_path, monkeypatch, prefix, target):
    _no_session_start_pass(monkeypatch)
    module_path, cls_name, meth = target.rsplit(".", 2)
    import importlib
    cls = getattr(importlib.import_module(module_path), cls_name)

    if meth == "__init__":
        def boom(self, *a, **k):
            raise RuntimeError("boom")
    else:
        async def boom(self, *a, **k):
            raise RuntimeError("boom")
    monkeypatch.setattr(cls, meth, boom)

    out = await _start(tmp_path, monkeypatch, memory={"consolidation": {"enabled": True}})
    warnings = _warnings(out["printed"])
    assert any(w.startswith(prefix) for w in warnings), warnings
    assert MEMORY_TOOLS <= set(out["calls"][0]["tools"]), "a soft failure cost the memory tools"


# --------------------------------------------------------------------------- interrupt (#43)


async def test_interrupt_in_consolidation_start_still_closes_the_store(tmp_path, monkeypatch):
    from localharness.memory.sqlite import MemoryStore

    _no_session_start_pass(monkeypatch)

    async def interrupt(self):
        raise KeyboardInterrupt
    monkeypatch.setattr("localharness.memory.consolidation.ConsolidationScheduler.start", interrupt)

    opened: list = []
    closed: list = []
    real_open, real_close = MemoryStore.open, MemoryStore.close

    async def spy_open(self, *a, **k):
        opened.append(self)
        return await real_open(self, *a, **k)

    async def spy_close(self):
        closed.append(self)
        return await real_close(self)
    monkeypatch.setattr(MemoryStore, "open", spy_open)
    monkeypatch.setattr(MemoryStore, "close", spy_close)

    try:
        # Today `_start_async` catches the KeyboardInterrupt and RETURNS (exit reason "interrupt");
        # the suppress keeps this a baseline of the close, not of that return path.
        with contextlib.suppress(KeyboardInterrupt):
            await _start(tmp_path, monkeypatch, memory={"consolidation": {"enabled": True}},
                         turn=False)
        assert len(opened) == 1, opened
        assert closed == opened, "an interrupt in consolidation start must still close the store (#43)"
    finally:
        for s in opened:
            if s not in closed:
                await real_close(s)  # hang-safety on a RED run

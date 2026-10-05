"""CORE-01 — the core boots and does real work with every plugin off (Phase 47).

**What this proves.** With the bundled image, web and memory plugins all off and the installed
example plugin off (installed by the dev extra, never enabled), a REAL `start` completes a scripted
turn that calls `read` on a real file and `bash_exec` `echo core-only`. The proof is on what
executed: the resolver's recorded `enabled` map, the lifecycle's recorded result (nothing running,
the slot empty), the two Observations the bus carried, the first system prompt (the global
guardrails once, no memory), the tools the model was offered, the slash table read DURING the
session, and `--help`. The on-twin shows the same drive with memory at its default really runs it
(44-03: an "off" proof needs a twin that is visibly on).

REAL: `_start_async`, plugin discovery (`real_plugins=True`; `BUILTIN_PLUGINS` is never patched), the
resolver, the plan, the lifecycle, the registry, the capability floor, the agent loop and its tools.
Recorders wrap `resolve`/`start_plugins` and call through. STUBBED: the LLM probe, the tokenizer
(extended for a turn), the REPL's read loop, and the model's replies; the provider is the loopback
discard port.

NOT proven: a real model, the terminal's rendering.
"""
from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from tests.conftest import FakeLLMResponse, FakeToolCall
from tests.integration.test_guardrails_from_global_dir_e2e import _let_the_stub_tokenizer_run_a_turn
from tests.integration.test_workspace_cli_surface_e2e import _offline_provider
from tests.unit.test_start_cmd import _stub_start_boundaries

GUARDRAILS = "# Org guardrails\nCORE-ONLY-GUARDRAILS-SENTINEL\n"
MEMORY_TOOLS = {"memory_search", "memory_get", "remember"}
# dispatch (49) and autoresearch (50) are bundled and on by default, so "every plugin off" turns them off too
ALL_OFF = ("memory:\n  enabled: false\nweb:\n  enabled: false\ndispatch:\n  enabled: false\n"
           "autoresearch:\n  enabled: false\n")


def _record(monkeypatch, target: str, sink: list, *, is_async: bool) -> None:
    """Wrap `module.func` with a recorder that stores the return value and returns it unchanged.
    start_cmd imports both names inside `_start_async`, so patching the module attribute is enough."""
    import importlib
    mod_path, name = target.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    real = getattr(mod, name)
    if is_async:
        async def rec(*a, **k):
            sink.append((k, out := await real(*a, **k)))
            return out
    else:
        def rec(*a, **k):
            sink.append((k, out := real(*a, **k)))
            return out
    monkeypatch.setattr(mod, name, rec)


async def drive(tmp_path: Path, monkeypatch, *, extra_config: str = "") -> dict:
    """One real start under `tmp_path` and one scripted turn: read a file, echo, done."""
    _stub_start_boundaries(tmp_path, monkeypatch, real_plugins=True)  # discovery stays REAL
    _offline_provider(tmp_path)
    if extra_config:
        with (tmp_path / "config.yaml").open("a", encoding="utf-8") as f:
            f.write(extra_config)
    _let_the_stub_tokenizer_run_a_turn(monkeypatch)
    g = tmp_path / "orgs" / "default" / "GUARDRAILS.md"
    g.parent.mkdir(parents=True, exist_ok=True)
    g.write_text(GUARDRAILS, encoding="utf-8")
    target = tmp_path / "note.txt"
    target.write_text("CORE-ONLY-FILE-CONTENT\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    resolutions: list = []
    lifecycles: list = []
    _record(monkeypatch, "localharness.plugins.resolve.resolve", resolutions, is_async=False)
    _record(monkeypatch, "localharness.plugins.lifecycle.start_plugins", lifecycles, is_async=True)

    calls: list[dict] = []

    async def model(self, messages, tools=None, on_token=None, **_):
        calls.append({"system": dict(messages[0]), "tools": [t.name for t in tools or ()]})
        if len(calls) == 1:
            return FakeLLMResponse(tool_calls=[
                FakeToolCall(id="r1", name="read", arguments={"path": str(target)})]), None
        if len(calls) == 2:
            return FakeLLMResponse(tool_calls=[
                FakeToolCall(id="b1", name="bash_exec", arguments={"command": "echo core-only"})]), None
        return FakeLLMResponse(content="Done."), None

    monkeypatch.setattr("localharness.provider.client.LLMClient.stream_complete", model)
    seen: dict = {"observations": []}

    async def one_turn(self):
        from localharness.cli.slash_commands import all_rows
        from localharness.core.events import Observation
        self._bus.subscribe(Observation, seen["observations"].append)
        seen["slash"] = {r.name for r in all_rows()}
        seen["slot_occupied"] = lifecycles[-1][1].slot.occupied if lifecycles else None
        seen["answer"] = await self._agent.run_turn("Read the note, then echo.")

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", one_turn)

    from localharness.cli.start_cmd import _start_async
    await _start_async(None, False, False, str(tmp_path))

    assert len(resolutions) == 1 and len(lifecycles) == 1, (resolutions, lifecycles)
    assert calls, "no request reached the model boundary — the turn never ran"
    return {"resolution": resolutions[0][1], "lifecycle": lifecycles[0][1],
            "registry": lifecycles[0][0]["registry"], "calls": calls, **seen}


async def test_core_boots_and_works_with_every_plugin_off(tmp_path, monkeypatch):
    out = await drive(tmp_path, monkeypatch, extra_config=ALL_OFF)
    resolution, result = out["resolution"], out["lifecycle"]

    # Discovery really ran: the installed example plugin is known, and everything is off.
    assert set(resolution.enabled) == {"image", "mobile", "memory", "dispatch", "autoresearch", "example"}, dict(resolution.enabled)
    assert not any(resolution.enabled.values()), dict(resolution.enabled)
    from localharness.plugins.channels import accepted_channels
    assert {e.name: e.state for e in resolution.plan.entries}["dispatch"] == "off"
    assert "discord" not in accepted_channels(resolution)
    # The lifecycle really ran and started nothing; the slot is empty during the session.
    assert result.running == [], result.loaded_names
    assert result.failed == {} and out["slot_occupied"] is False

    # Real work: both calls executed and produced non-error Observations.
    obs = [o for o in out["observations"] if o.tool_name in ("read", "bash_exec")]
    assert [o.tool_name for o in obs] == ["read", "bash_exec"], [(o.tool_name, o.output) for o in obs]
    assert all(o.error is None for o in obs), [(o.tool_name, o.error) for o in obs]
    assert "CORE-ONLY-FILE-CONTENT" in (obs[0].output or ""), obs[0].output
    assert "core-only" in (obs[1].output or ""), obs[1].output
    assert out["answer"] == "Done." and len(out["calls"]) == 3

    prompt = out["calls"][0]["system"]["content"]
    assert prompt.count("CORE-ONLY-GUARDRAILS-SENTINEL") == 1, "guardrails missing or doubled"
    assert "## Agent Memory" not in prompt

    tools = set(out["calls"][0]["tools"])
    assert {"read", "bash_exec"} <= tools, sorted(tools)
    assert not (tools & (MEMORY_TOOLS | {"generate_image"})), sorted(tools)
    assert not (set(out["registry"]._tools["global"]) & (MEMORY_TOOLS | {"generate_image"}))
    assert "/memory" not in out["slash"], sorted(out["slash"])


def test_help_lists_no_plugin_commands_when_all_off(tmp_path, monkeypatch):
    _stub_start_boundaries(tmp_path, monkeypatch, real_plugins=True)
    with (tmp_path / "config.yaml").open("a", encoding="utf-8") as f:
        f.write(ALL_OFF)
    from localharness.cli.app import app
    res = CliRunner().invoke(app, ["--help"], env={"LOCALHARNESS_DIR": str(tmp_path)})
    assert res.exit_code == 0, res.output
    assert re.search(r"│ plugins\s", res.output), res.output  # the table really rendered
    assert not re.search(r"│ mobile\s", res.output), res.output
    assert "generate-image" not in res.output, res.output
    for name in ("autoresearch", "experiment", "propose"):  # PRD §8 "All plugins off"
        assert not re.search(rf"│ {name}\s", res.output), res.output


def test_help_lists_web_when_it_is_on(tmp_path, monkeypatch):
    """The --help twin: the same config dir with web at its default lists `mobile` — so the absence
    above is the config's doing, not a table that never lists plugin commands."""
    _stub_start_boundaries(tmp_path, monkeypatch, real_plugins=True)
    from localharness.cli.app import app
    res = CliRunner().invoke(app, ["--help"], env={"LOCALHARNESS_DIR": str(tmp_path)})
    assert res.exit_code == 0, res.output
    assert re.search(r"│ mobile\s", res.output), res.output
    assert all(re.search(rf"│ {n}\s", res.output) for n in ("autoresearch", "experiment", "propose")), res.output


async def test_the_on_twin_has_memory(tmp_path, monkeypatch):
    out = await drive(tmp_path, monkeypatch)
    result = out["lifecycle"]
    assert "memory" in result.loaded_names, result.loaded_names
    assert out["slot_occupied"] is True
    assert "## Agent Memory" in out["calls"][0]["system"]["content"]
    assert MEMORY_TOOLS <= set(out["calls"][0]["tools"]), out["calls"][0]["tools"]
    assert "/memory" in out["slash"]
